"""Convert an upstream Flesh and Blood card export into RLIP card_db format.

Usage:
    python import_from_talishar.py --source /path/to/cards.json --out cards.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def normalize_record(raw: dict[str, Any]) -> dict[str, Any]:
    card_id = str(_pick(raw, "id", "cardIdentifier", "card_id", default="")).strip()
    if not card_id:
        name_part = str(_pick(raw, "name", default="unnamed")).strip().lower().replace(" ", "_")
        card_id = f"generated_{name_part}"

    card_types = _pick(raw, "card_types", default=None)
    if card_types is None:
        maybe_types = _pick(raw, "types", "type", default=[])
        if isinstance(maybe_types, str):
            card_types = [t.strip().lower().replace(" ", "_") for t in maybe_types.split(",") if t.strip()]
        elif isinstance(maybe_types, list):
            card_types = [str(t).strip().lower().replace(" ", "_") for t in maybe_types if str(t).strip()]
        else:
            card_types = []

    keywords = _pick(raw, "keywords", default=[])
    if isinstance(keywords, str):
        keywords = [k.strip().lower().replace(" ", "_") for k in keywords.split(",") if k.strip()]

    legality = _pick(raw, "legality", default={})
    if not isinstance(legality, dict):
        legality = {}

    return {
        "id": card_id,
        "name": str(_pick(raw, "name", default=card_id)),
        "pitch": _as_int(_pick(raw, "pitch", "resource", "resourceValue", default=0), default=0),
        "cost": _as_int(_pick(raw, "cost", default=0), default=0),
        "power": _as_int(_pick(raw, "power", "attack", default=0), default=0),
        "defense": _as_int(_pick(raw, "defense", "defence", "block", default=0), default=0),
        "type_line": str(_pick(raw, "type_line", "typeText", "type", default="")),
        "card_types": card_types,
        "class": str(_pick(raw, "class", "classes", default="Generic")),
        "talent": _pick(raw, "talent", default=None),
        "rarity": str(_pick(raw, "rarity", default="Common")),
        "set": str(_pick(raw, "set", "set_code", "setCode", default="SIM")),
        "keywords": keywords,
        "text": str(_pick(raw, "text", "rulesText", default="")),
        "legality": legality,
        "metadata": raw,
    }


def load_cards(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("cards", "data", "records"):
            if key in raw and isinstance(raw[key], list):
                return raw[key]
    raise ValueError("Unsupported source JSON format. Expected list or object containing cards/data/records.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Import card data into RLIP Flesh and Blood card_db format")
    parser.add_argument("--source", required=True, help="Path to source cards JSON file")
    parser.add_argument("--out", required=True, help="Path to output cards.json file")
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)

    cards_raw = load_cards(source)
    normalized = [normalize_record(rec) for rec in cards_raw if isinstance(rec, dict)]

    out.write_text(json.dumps(normalized, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(normalized)} cards to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
