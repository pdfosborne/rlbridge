"""Build RLIP Flesh and Blood card/hero DB from a comprehensive upstream feed.

Usage:
    python import_from_talishar.py --out-cards cards.json --out-heroes heroes.json
    python import_from_talishar.py --source /path/to/card.json --out-cards cards.json --out-heroes heroes.json
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = (
    "https://raw.githubusercontent.com/the-fab-cube/flesh-and-blood-cards/master/json/english/card.json"
)

_KNOWN_CLASSES = {
    "ASSASSIN",
    "BRUTE",
    "GUARDIAN",
    "ILLUSIONIST",
    "MECHANOLOGIST",
    "NINJA",
    "RANGER",
    "RUNEBLADE",
    "WARRIOR",
    "WIZARD",
    "GENERIC",
}
_KNOWN_TALENTS = {
    "LIGHT",
    "SHADOW",
    "ELEMENTAL",
    "EARTH",
    "ICE",
    "LIGHTNING",
    "CHAOS",
    "DRACONIC",
    "MYSTIC",
}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _pick(obj: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in obj and obj[k] is not None:
            return obj[k]
    return default


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.strip().lower())
    return re.sub(r"_+", "_", s).strip("_")


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", ""}:
        return False
    return default


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [p.strip() for p in text.split(",") if p.strip()]


def _find_class_and_talent(types: list[str], classes: list[str], talents: list[str]) -> tuple[str, Any]:
    cls_tokens = [t.upper() for t in classes if t]
    talent_tokens = [t.upper() for t in talents if t]
    type_tokens = [t.upper() for t in types if t]

    if not cls_tokens:
        cls_tokens = [t for t in type_tokens if t in _KNOWN_CLASSES]
    if not talent_tokens:
        talent_tokens = [t for t in type_tokens if t in _KNOWN_TALENTS]

    card_class = cls_tokens[0].title() if cls_tokens else "Generic"
    talent = talent_tokens[0].title() if talent_tokens else None
    return card_class, talent


def _normalize_card_types(types: list[str]) -> list[str]:
    tset = {t.strip().lower() for t in types if t.strip()}
    out: list[str] = []
    if "action" in tset and "attack" in tset:
        out.append("attack_action")
    if "defense reaction" in tset:
        out.append("defense_reaction")
    if "attack reaction" in tset:
        out.append("attack_reaction")
    if "action" in tset and "attack" not in tset:
        out.append("utility_action")
    if "equipment" in tset or "weapon" in tset:
        out.append("utility_item")
    if "hero" in tset:
        out.append("hero")
    return out


def normalize_record(raw: dict[str, Any]) -> dict[str, Any]:
    card_id = str(_pick(raw, "id", "cardIdentifier", "card_id", "unique_id", default="")).strip()
    if not card_id:
        name_part = str(_pick(raw, "name", default="unnamed")).strip().lower().replace(" ", "_")
        card_id = f"generated_{name_part}"

    name = str(_pick(raw, "name", default=card_id)).strip()
    color = str(_pick(raw, "color", default="")).strip().lower()
    if card_id == str(_pick(raw, "unique_id", default=card_id)).strip():
        slug = _slug(name)
        cid = f"{slug}_{color}" if color else slug
        card_id = cid or card_id

    card_types = _pick(raw, "card_types", default=None)
    if card_types is None:
        maybe_types = _as_list(_pick(raw, "types", "type", default=[]))
        card_types = _normalize_card_types(maybe_types)

    keywords = _pick(raw, "keywords", default=[])
    if isinstance(keywords, str):
        keywords = [k.strip().lower().replace(" ", "_") for k in keywords.split(",") if k.strip()]

    legality = _pick(raw, "legality", default={})
    if not isinstance(legality, dict):
        legality = {}
    if not legality:
        cc_legal = _as_bool(_pick(raw, "cc_legal", default=True), default=True) and not _as_bool(
            _pick(raw, "cc_banned", default=False), default=False
        )
        silver_legal = _as_bool(_pick(raw, "silver_age_legal", default=cc_legal), default=cc_legal) and not _as_bool(
            _pick(raw, "silver_age_banned", default=False), default=False
        )
        legality = {
            "classic_constructed": "legal" if cc_legal else "banned",
            "silver_age": "legal" if silver_legal else "banned",
        }

    types = _as_list(_pick(raw, "types", "type", default=[]))
    classes = _as_list(_pick(raw, "classes", "class", default=[]))
    talents = _as_list(_pick(raw, "talents", "talent", default=[]))
    card_class, talent = _find_class_and_talent(types, classes, talents)

    return {
        "id": card_id,
        "name": name,
        "pitch": _as_int(_pick(raw, "pitch", "resource", "resourceValue", default=0), default=0),
        "cost": _as_int(_pick(raw, "cost", default=0), default=0),
        "power": _as_int(_pick(raw, "power", "attack", default=0), default=0),
        "defense": _as_int(_pick(raw, "defense", "defence", "block", default=0), default=0),
        "type_line": str(_pick(raw, "type_line", "typeText", "type", default="")),
        "card_types": card_types,
        "class": card_class,
        "talent": talent,
        "rarity": str(_pick(raw, "rarity", default="Common")),
        "set": str(_pick(raw, "set", "set_code", "setCode", default="SIM")),
        "keywords": keywords,
        "text": str(_pick(raw, "text", "rulesText", default="")),
        "legality": legality,
    }


def load_cards(path: Path | None = None, source_url: str | None = None) -> list[dict[str, Any]]:
    if source_url:
        raw = json.loads(urllib.request.urlopen(source_url, timeout=60).read().decode("utf-8"))
    elif path is not None:
        raw = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError("Either path or source_url must be provided")

    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("cards", "data", "records"):
            if key in raw and isinstance(raw[key], list):
                return raw[key]
    raise ValueError("Unsupported source JSON format. Expected list or object containing cards/data/records.")


def derive_heroes(cards_raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for rec in cards_raw:
        if not isinstance(rec, dict):
            continue
        types = _as_list(rec.get("types"))
        tset = {t.strip().lower() for t in types}
        if "hero" not in tset:
            continue

        name = str(rec.get("name") or "").strip()
        if not name:
            continue
        life = _as_int(_pick(rec, "health", "life", default=40), default=40)
        intellect = _as_int(_pick(rec, "intelligence", "intellect", default=4), default=4)

        classes = _as_list(rec.get("classes"))
        talents = _as_list(rec.get("talents"))
        hero_class, talent = _find_class_and_talent(types, classes, talents)

        hero_id = f"hero_{_slug(name)}"
        row = {
            "id": hero_id,
            "name": name,
            "class": hero_class,
            "talent": talent,
            "life": life,
            "intellect": intellect,
            "weapon": {
                "id": f"weapon_{_slug(name)}_basic",
                "name": "Basic Weapon",
                "attack": 3,
                "cost": 1,
                "type_line": "Weapon",
            },
        }

        existing = by_name.get(name)
        if existing is None or int(existing.get("life", 0)) < life:
            by_name[name] = row

    heroes = list(by_name.values())
    heroes.sort(key=lambda h: str(h.get("name", "")).lower())
    return heroes


def main() -> int:
    parser = argparse.ArgumentParser(description="Import card data into RLIP Flesh and Blood card_db format")
    parser.add_argument("--source", default="", help="Path to source cards JSON file")
    parser.add_argument("--source-url", default=DEFAULT_SOURCE, help="URL to source cards JSON feed")
    parser.add_argument("--out-cards", required=True, help="Path to output cards.json file")
    parser.add_argument("--out-heroes", required=True, help="Path to output heroes.json file")
    args = parser.parse_args()

    source = Path(args.source) if args.source else None
    out_cards = Path(args.out_cards)
    out_heroes = Path(args.out_heroes)

    cards_raw = load_cards(path=source, source_url=(None if source else args.source_url))
    normalized = [normalize_record(rec) for rec in cards_raw if isinstance(rec, dict)]
    heroes = derive_heroes(cards_raw)

    out_cards.write_text(json.dumps(normalized, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    out_heroes.write_text(json.dumps(heroes, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(normalized)} cards to {out_cards}")
    print(f"Wrote {len(heroes)} heroes to {out_heroes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
