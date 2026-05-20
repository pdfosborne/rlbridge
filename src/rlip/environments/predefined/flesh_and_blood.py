"""
RLIP Flesh and Blood Environment (Talishar-Inspired)
====================================================
A turn-based TCG simulator inspired by Talishar-style gameplay loops.

This environment intentionally focuses on deterministic RL-friendly mechanics:

- Two heroes (agent vs scripted opponent), both start at 40 life.
- Hand/deck/discard zones with draw-up-to-intellect each turn.
- Action phase: play attack cards or pass.
- Defense phase: defending player may block with hand cards or pass.
- Combat resolution applies net damage after blocks.

Reward signal summary:

- Terminal win: `+1.0`; terminal loss: `-1.0`.
- Dense shaping: `+0.01 * damage_dealt` and `-0.005 * damage_received` per combat resolution.
- Step penalty of `-0.005` per step discourages passive play.
- Illegal actions receive a penalty (`-0.1`, except pass is auto-coerced when it is the only legal action).
- Invalid block/play attempts apply `-0.05`.
- Blocking carries no shaping bonus; win/loss is the primary signal.

The simulator is designed for training and protocol integration, not as a full
competitive-rules implementation.
"""

from __future__ import annotations

import base64
import importlib
import json
import math
import random
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from ...protocol.messages import (
    EnvironmentInfo,
    RenderResult,
    ResetResult,
    StepResult,
    SuggestedHyperparameters,
    TextSpace,
)
from ..base import RLIPEnvironment, RLIPEnvironmentFactory


_FAB_DB_DIR = Path(__file__).with_name("card_db") / "flesh_and_blood"
_CARDS_PATH = _FAB_DB_DIR / "cards.json"
_HEROES_PATH = _FAB_DB_DIR / "heroes.json"
_FABRARY_DECKS_PATH = _FAB_DB_DIR / "fabrary_decks.json"

_FORMAT_ALIASES = {
    "cc": "classic_constructed",
    "classic_constructed": "classic_constructed",
    "classic constructed": "classic_constructed",
    "silver_age": "silver_age",
    "silver age": "silver_age",
    "sage": "silver_age",
}

_FAB_CUSTOM_TOOLS_REGISTERED = False


@dataclass(frozen=True)
class Card:
    id: str
    name: str
    pitch: int
    cost: int
    power: int
    defense: int
    type_line: str
    card_types: tuple[str, ...]
    card_class: str
    talent: Optional[str]
    rarity: str
    set_code: str
    keywords: tuple[str, ...]
    text: str
    legality: dict[str, str]


@dataclass(frozen=True)
class Hero:
    id: str
    name: str
    hero_class: str
    talent: Optional[str]
    life: int
    intellect: int
    weapon_name: str
    weapon_attack: int
    weapon_cost: int


@dataclass
class PlayerState:
    hero: Hero
    life: int
    resources: int
    action_points: int
    deck: list[str]
    hand: list[str]
    discard: list[str]


@dataclass
class CombatState:
    attacker: int
    defender: int
    attack_card_id: str
    attack_power: int
    blocks: list[tuple[int, str, int]]


def _load_cards() -> dict[str, Card]:
    if not _CARDS_PATH.exists():
        raise FileNotFoundError(f"Missing Flesh and Blood card DB at {_CARDS_PATH}")

    raw = json.loads(_CARDS_PATH.read_text(encoding="utf-8"))
    cards: dict[str, Card] = {}
    for rec in raw:
        card = Card(
            id=rec["id"],
            name=rec["name"],
            pitch=int(rec.get("pitch", 0)),
            cost=int(rec.get("cost", 0)),
            power=int(rec.get("power", 0)),
            defense=int(rec.get("defense", 0)),
            type_line=rec.get("type_line", ""),
            card_types=tuple(rec.get("card_types", [])),
            card_class=rec.get("class", "Generic"),
            talent=rec.get("talent"),
            rarity=rec.get("rarity", "Common"),
            set_code=rec.get("set", "SIM"),
            keywords=tuple(rec.get("keywords", [])),
            text=rec.get("text", ""),
            legality=rec.get("legality", {}),
        )
        cards[card.id] = card
    return cards


def _load_heroes() -> dict[str, Hero]:
    if not _HEROES_PATH.exists():
        raise FileNotFoundError(f"Missing Flesh and Blood hero DB at {_HEROES_PATH}")

    raw = json.loads(_HEROES_PATH.read_text(encoding="utf-8"))
    heroes: dict[str, Hero] = {}
    for rec in raw:
        weapon = rec.get("weapon", {})
        hero = Hero(
            id=rec["id"],
            name=rec["name"],
            hero_class=rec.get("class", "Generic"),
            talent=rec.get("talent"),
            life=int(rec.get("life", 40)),
            intellect=int(rec.get("intellect", 4)),
            weapon_name=weapon.get("name", "Basic Weapon"),
            weapon_attack=int(weapon.get("attack", 3)),
            weapon_cost=int(weapon.get("cost", 1)),
        )
        heroes[hero.id] = hero
    return heroes


class FleshAndBloodEnvironment(RLIPEnvironment):
    """Talishar-inspired RL simulator for Flesh and Blood gameplay loops."""

    def __init__(
        self,
        *,
        seed: Optional[int] = None,
        agent_hero_id: str = "hero_dorinthea_ironsong",
        opponent_hero_id: str = "hero_rhinar_reckless_rampage",
        max_turns: int = 60,
        deck_size: int = 36,
        agent_deck_style: str = "balanced",
        opponent_deck_style: str = "balanced",
        format: str = "classic_constructed",
        two_phase_deckbuild: bool = False,
        self_play: bool = False,
        render_mode: Optional[str] = None,
    ) -> None:
        self._cards = _load_cards()
        self._heroes = _load_heroes()
        self._rng = random.Random(seed)

        if agent_hero_id not in self._heroes:
            raise ValueError(f"Unknown agent_hero_id: {agent_hero_id}")
        if opponent_hero_id not in self._heroes:
            raise ValueError(f"Unknown opponent_hero_id: {opponent_hero_id}")

        self._agent_hero_id = agent_hero_id
        self._opponent_hero_id = opponent_hero_id
        self._max_turns = max_turns
        self._deck_size = deck_size
        self._agent_deck_style = str(agent_deck_style)
        self._opponent_deck_style = str(opponent_deck_style)
        self._format = self._normalize_format(format)
        self._two_phase_deckbuild = bool(two_phase_deckbuild)

        self._players: list[PlayerState] = []
        self._turn = 0
        self._active_player = 0
        self._phase = "action"
        self._pending_combat: Optional[CombatState] = None
        self._initialized = False
        self._last_event = ""
        self._render_mode = render_mode
        self._self_play = self_play
        self._selection_stage = False
        self._selected_deck_option: Optional[dict[str, Any]] = None
        self._selected_opponent_deck_option: Optional[dict[str, Any]] = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> ResetResult:
        if seed is not None:
            self._rng.seed(seed)

        opts = options or {}
        self._agent_hero_id = str(opts.get("agent_hero_id", self._agent_hero_id))
        self._opponent_hero_id = str(opts.get("opponent_hero_id", self._opponent_hero_id))
        self._agent_deck_style = str(opts.get("agent_deck_style", self._agent_deck_style))
        self._opponent_deck_style = str(opts.get("opponent_deck_style", self._opponent_deck_style))
        self._format = self._normalize_format(str(opts.get("format", self._format)))
        self._two_phase_deckbuild = bool(opts.get("two_phase_deckbuild", self._two_phase_deckbuild))
        self._max_turns = int(opts.get("max_turns", self._max_turns))
        self._deck_size = int(opts.get("deck_size", self._deck_size))

        min_cards, max_cards = self._format_deck_bounds()
        self._deck_size = max(min_cards, min(self._deck_size, max_cards))

        self._pending_combat = None
        self._selected_deck_option = None
        self._selected_opponent_deck_option = None

        if self._two_phase_deckbuild:
            self._players = []
            self._turn = 0
            self._active_player = 0
            self._phase = "deck_selection"
            self._selection_stage = True
            self._last_event = "Select a deck to begin the match"
        else:
            self._selection_stage = False
            self._start_match(
                agent_hero_id=self._agent_hero_id,
                opponent_hero_id=self._opponent_hero_id,
                agent_deck_style=self._agent_deck_style,
                opponent_deck_style=self._opponent_deck_style,
            )

        self._initialized = True
        return ResetResult(
            observation=self._observation(),
            info={
                "legal_actions": self._legal_actions(),
                "agent_hero": (self._players[0].hero.name if self._players else None),
                "opponent_hero": (self._players[1].hero.name if self._players else None),
                "format": self._format,
                "stage": ("deck_selection" if self._selection_stage else "play"),
            },
        )

    def step(self, action: Any) -> StepResult:
        if not self._initialized:
            raise RuntimeError("Call reset() before step().")

        if self._selection_stage:
            legal = self._legal_actions()
            parsed = self._normalize_action(action)
            if parsed not in legal:
                if isinstance(action, int) and legal:
                    parsed = legal[action % len(legal)]
                else:
                    return StepResult(
                        observation=self._observation(),
                        reward=-0.1,
                        terminated=False,
                        truncated=False,
                        info={
                            "error": f"Illegal action {parsed!r}",
                            "legal_actions": legal,
                        },
                    )

            choice_key = parsed.split(" ", 1)[1]
            choice = next((o for o in self._deck_options_for_format() if o["key"] == choice_key), None)
            if choice is None:
                return StepResult(
                    observation=self._observation(),
                    reward=-0.1,
                    terminated=False,
                    truncated=False,
                    info={"error": f"Unknown deck option: {choice_key}", "legal_actions": legal},
                )

            self._selected_deck_option = dict(choice)
            self._agent_hero_id = str(choice["hero_id"])
            self._deck_size = int(choice["deck_size"])

            opp_choice = self._sample_opponent_matchup(agent_hero_id=self._agent_hero_id)
            self._selected_opponent_deck_option = dict(opp_choice)
            self._opponent_hero_id = str(opp_choice["hero_id"])

            self._selection_stage = False
            # Pass pre-resolved card IDs when a fabrary deck option was chosen.
            agent_card_ids: Optional[list[str]] = choice.get("_card_ids")  # type: ignore[assignment]
            opp_card_ids: Optional[list[str]] = opp_choice.get("_card_ids")  # type: ignore[assignment]
            self._start_match(
                agent_hero_id=self._agent_hero_id,
                opponent_hero_id=self._opponent_hero_id,
                agent_deck_style=str(choice["style"]),
                opponent_deck_style=str(opp_choice["style"]),
                agent_deck_ids=agent_card_ids,
                opponent_deck_ids=opp_card_ids,
            )
            self._last_event = f"Deck selected: {choice['label']} | Matchup: {opp_choice['label']}"

            info = {
                "legal_actions": self._legal_actions(),
                "phase": self._phase,
                "last_event": self._last_event,
                "turn": self._turn,
                "stage": "play",
            }
            return StepResult(
                observation=self._observation(),
                reward=0.0,
                terminated=False,
                truncated=False,
                info=info,
            )

        legal = self._legal_actions()
        parsed = self._normalize_action(action)
        if parsed not in legal:
            if legal == ["pass"]:
                # Only pass is legal — coerce so policies never deadlock.
                parsed = "pass"
            elif isinstance(action, int) and legal:
                # RL agents emit integer indices; treat them as indices into the
                # legal action list so out-of-range ints never get stuck.
                parsed = legal[action % len(legal)]
            else:
                return StepResult(
                    observation=self._observation(),
                    reward=-0.1,
                    terminated=False,
                    truncated=False,
                    info={
                        "error": f"Illegal action {parsed!r}",
                        "legal_actions": legal,
                    },
                )

        reward = 0.0

        if self._self_play:
            reward += self._step_self_play(parsed)
        else:
            if self._phase == "action":
                if parsed == "pass":
                    self._last_event = "Agent passed turn"
                    self._end_turn_and_run_opponent()
                elif parsed.startswith("play "):
                    idx = int(parsed.split(" ")[1])
                    reward += self._agent_play_attack(idx)
            elif self._phase == "defense":
                if parsed == "pass":
                    reward += self._resolve_combat()
                    self._end_turn(1)  # opponent draws up after their attack resolves
                    self._phase = "action"
                    self._start_turn(0)
                elif parsed.startswith("block "):
                    idx = int(parsed.split(" ")[1])
                    reward += self._agent_block(idx)
                    # One block per attack: resolve immediately after blocking
                    reward += self._resolve_combat()
                    self._end_turn(1)
                    self._phase = "action"
                    self._start_turn(0)

        terminated = self._is_terminal()
        truncated = self._turn >= self._max_turns and not terminated
        if truncated:
            self._last_event = "Reached max turns"

        # Terminal win/loss bonus (primary signal)
        if terminated:
            if self._self_play:
                # In self-play the acting player just caused termination;
                # reward from the perspective of the player whose life just hit 0.
                loser = next(i for i, p in enumerate(self._players) if p.life <= 0)
                winner = 1 - loser
                # Reward for the policy: +1 if the last acting player won, -1 if lost.
                if self._active_player == winner:
                    reward += 1.0
                else:
                    reward -= 1.0
            elif self._players[0].life <= 0:
                reward -= 1.0  # agent lost
            else:
                reward += 1.0  # agent won
        else:
            reward -= 0.005  # step penalty to discourage passive looping

        info = {
            "legal_actions": [] if (terminated or truncated) else self._legal_actions(),
            "phase": self._phase,
            "last_event": self._last_event,
            "turn": self._turn,
        }
        return StepResult(
            observation=self._observation(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def close(self) -> None:
        self._players = []
        self._pending_combat = None
        self._initialized = False

    @property
    def observation_space(self) -> TextSpace:
        return TextSpace(min_length=0, max_length=16000)

    @property
    def action_space(self) -> TextSpace:
        return TextSpace(min_length=1, max_length=32)

    def sample_action(self) -> str:
        if not self._initialized:
            return "pass"
        if self._selection_stage:
            legal = self._legal_actions()
            return self._rng.choice(legal) if legal else "pass"
        if len(self._players) < 2:
            return "pass"
        legal = self._legal_actions()
        return self._rng.choice(legal) if legal else "pass"

    def render(self) -> RenderResult:
        if self._render_mode == "rgb_array":
            return self._render_rgb()
        return RenderResult(mode="ansi", text=self._render_text())

    def _render_rgb(self) -> RenderResult:
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            return RenderResult(mode="rgb_array", text="pillow not installed")

        obs = self._observation()
        p_agent, p_opp = self._estimate_win_probabilities(obs)
        width, height = 1200, 720
        bg = (18, 24, 33)
        panel = (30, 40, 54)
        panel_alt = (41, 53, 69)
        accent = (245, 181, 82)
        fg = (240, 243, 248)
        sub = (176, 186, 201)
        enemy = (240, 103, 103)
        ally = (96, 214, 138)

        img = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(img)

        try:
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
            font_subtitle = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            font_text = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except Exception:
            font_title = ImageFont.load_default()
            font_subtitle = ImageFont.load_default()
            font_text = ImageFont.load_default()
            font_small = ImageFont.load_default()

        def _panel(x0: int, y0: int, x1: int, y1: int, fill: tuple[int, int, int]) -> None:
            draw.rounded_rectangle((x0, y0, x1, y1), radius=14, fill=fill)

        def _text(x: int, y: int, msg: str, font: Any, fill: tuple[int, int, int]) -> None:
            draw.text((x, y), msg, fill=fill, font=font)

        if obs.get("stage") == "deck_selection":
            _text(24, 20, "Flesh and Blood (Talishar-inspired)", font_title, fg)
            _text(24, 58, f"Format: {obs['format']} | Stage: deck_selection", font_subtitle, sub)
            _panel(20, 100, 1180, 680, panel)
            _text(40, 128, "Choose a deck to start the episode", font_subtitle, accent)
            for i, item in enumerate(obs.get("deck_options", [])[:12]):
                _text(40, 170 + i * 34, f"[{i}] {item['label']} ({item['deck_size']} cards)", font_text, fg)
            legal = ", ".join(obs.get("legal_actions", [])[:12])
            _text(40, 640, f"Legal actions: {legal}", font_small, sub)
            buf = BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return RenderResult(mode="rgb_array", data=b64, width=width, height=height)

        _text(24, 20, "Flesh and Blood (Talishar-inspired)", font_title, fg)
        _text(
            24,
            58,
            f"Turn {obs['turn']} | Format: {obs['format']} | Phase: {obs['phase']} | Active: P{obs['active_player']}",
            font_subtitle,
            sub,
        )

        _panel(20, 100, 580, 240, panel)
        _panel(620, 100, 1180, 240, panel)
        _panel(20, 260, 1180, 360, panel_alt)
        _panel(20, 380, 1180, 700, panel)

        agent = obs["agent"]
        opponent = obs["opponent"]

        _text(40, 120, "Agent", font_subtitle, ally)
        _text(40, 150, f"Hero: {agent['hero']}", font_text, fg)
        _text(40, 174, f"Life: {agent['life']}", font_text, fg)
        _text(220, 174, f"Resources: {agent['resources']}", font_text, fg)
        _text(440, 174, f"AP: {agent['action_points']}", font_text, fg)
        _text(40, 200, f"Deck: {agent['deck']}  Discard: {agent['discard']}", font_text, sub)
        _text(40, 224, f"Win %: {p_agent:.1%}", font_text, accent)

        _text(640, 120, "Opponent", font_subtitle, enemy)
        _text(640, 150, f"Hero: {opponent['hero']}", font_text, fg)
        _text(640, 174, f"Life: {opponent['life']}", font_text, fg)
        _text(820, 174, f"Resources: {opponent['resources']}", font_text, fg)
        _text(1040, 174, f"AP: {opponent['action_points']}", font_text, fg)
        _text(
            640,
            200,
            f"Deck: {opponent['deck']}  Discard: {opponent['discard']}  Hand: {opponent['hand_size']}",
            font_text,
            sub,
        )
        _text(640, 224, f"Win %: {p_opp:.1%}", font_text, accent)

        pending = obs.get("pending_combat")
        if pending:
            combat_line = (
                f"Combat | Attacker: P{pending['attacker']}  Defender: P{pending['defender']}  "
                f"{pending['attack_card']} ATK {pending['attack_power']}  BLOCK {pending['total_block']}"
            )
        else:
            combat_line = "Combat | none"
        _text(40, 292, combat_line, font_text, fg)
        _text(40, 320, f"Last event: {obs['last_event']}", font_text, sub)

        _text(40, 400, "Agent Hand", font_subtitle, accent)

        hand = agent["hand"]
        card_w, card_h = 215, 128
        cols = 5
        x_start = 40
        y_start = 432
        x_gap = 14
        y_gap = 12
        for idx, card in enumerate(hand[:15]):
            row = idx // cols
            col = idx % cols
            x0 = x_start + col * (card_w + x_gap)
            y0 = y_start + row * (card_h + y_gap)
            x1 = x0 + card_w
            y1 = y0 + card_h
            shade = panel_alt if idx % 2 == 0 else (36, 47, 62)
            draw.rounded_rectangle((x0, y0, x1, y1), radius=10, fill=shade)
            _text(x0 + 10, y0 + 8, f"[{card['index']}] {card['name'][:24]}", font_small, fg)
            _text(x0 + 10, y0 + 34, f"Cost {card['cost']}  ATK {card['power']}  DEF {card['defense']}", font_small, sub)
            ctype = ",".join(card.get("card_types", []))[:28]
            _text(x0 + 10, y0 + 58, f"Type: {ctype}", font_small, sub)
            keys = ",".join(card.get("keywords", [])) or "none"
            _text(x0 + 10, y0 + 82, f"Keywords: {keys[:24]}", font_small, sub)

        legal = ", ".join(obs.get("legal_actions", [])[:12])
        _text(40, 676, f"Legal actions: {legal}", font_small, sub)

        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return RenderResult(mode="rgb_array", data=b64, width=width, height=height)

    def _new_player(self, hero: Hero, hero_slot: int, deck_style: str = "balanced", deck_ids: Optional[list[str]] = None) -> PlayerState:
        if deck_ids is not None:
            deck = list(deck_ids)
        else:
            deck = self._build_deck(hero, deck_style=deck_style)
        self._rng.shuffle(deck)
        return PlayerState(
            hero=hero,
            life=self._starting_life(hero),
            resources=0,
            action_points=0,
            deck=deck,
            hand=[],
            discard=[],
        )

    def _build_deck(self, hero: Hero, deck_style: str = "balanced") -> list[str]:
        candidates: list[Card] = []
        for card in self._cards.values():
            if not self._is_card_legal_for_format(card.id):
                continue
            if not self._is_card_compatible_with_hero(card, hero):
                continue
            if not self._is_deck_candidate(card):
                continue
            candidates.append(card)

        if not candidates:
            raise ValueError(f"No legal cards available for hero={hero.name!r} format={self._format}")

        if deck_style == "aggro":
            ranked = sorted(
                candidates,
                key=lambda c: (
                    1 if "attack_action" in c.card_types else 0,
                    c.power,
                    -c.cost,
                    c.defense,
                ),
                reverse=True,
            )
        elif deck_style == "control":
            ranked = sorted(
                candidates,
                key=lambda c: (
                    c.defense,
                    c.pitch,
                    1 if "defense_reaction" in c.card_types else 0,
                    1 if "attack_action" in c.card_types else 0,
                ),
                reverse=True,
            )
        else:
            ranked = sorted(
                candidates,
                key=lambda c: (
                    (c.power + c.defense),
                    c.pitch,
                    1 if "attack_action" in c.card_types else 0,
                ),
                reverse=True,
            )

        copy_limit = 2 if self._format == "silver_age" else 3
        deck: list[str] = []
        counts: dict[str, int] = {}

        while len(deck) < self._deck_size:
            added_this_pass = False
            for card in ranked:
                cid = card.id
                if counts.get(cid, 0) >= copy_limit:
                    continue
                deck.append(cid)
                counts[cid] = counts.get(cid, 0) + 1
                added_this_pass = True
                if len(deck) >= self._deck_size:
                    break
            if not added_this_pass:
                deck.append(self._rng.choice(ranked).id)

        return deck[: self._deck_size]

    def _normalize_format(self, fmt: str) -> str:
        normalized = _FORMAT_ALIASES.get(str(fmt).strip().lower())
        if normalized is None:
            allowed = ", ".join(sorted(set(_FORMAT_ALIASES.values())))
            raise ValueError(f"Unknown format {fmt!r}. Expected one of: {allowed}")
        return normalized

    def _format_deck_bounds(self) -> tuple[int, int]:
        if self._format == "silver_age":
            return 40, 55
        return 60, 80

    def _deck_options_for_format(self) -> list[dict[str, Any]]:
        min_cards, max_cards = self._format_deck_bounds()
        secondary_size = min(max_cards, min_cards + 5)
        options: list[dict[str, Any]] = []
        for hero in self._hero_pool_for_format():
            slug = hero.id.replace("hero_", "")
            options.append(
                {
                    "key": f"{slug}_aggro",
                    "label": f"{hero.name} Aggro",
                    "hero_id": hero.id,
                    "style": "aggro",
                    "deck_size": min_cards,
                }
            )
            options.append(
                {
                    "key": f"{slug}_control",
                    "label": f"{hero.name} Control",
                    "hero_id": hero.id,
                    "style": "control",
                    "deck_size": secondary_size,
                }
            )
        # Append fabrary deck options (distinct keys prefixed with "fab_")
        options.extend(self._fabrary_deck_options_for_format())
        return options

    def _sample_opponent_matchup(self, agent_hero_id: str) -> dict[str, Any]:
        options = self._deck_options_for_format()
        candidates = [o for o in options if str(o["hero_id"]) != agent_hero_id]
        if not candidates:
            candidates = options
        return dict(self._rng.choice(candidates))

    # ------------------------------------------------------------------
    # Fabrary deck database
    # ------------------------------------------------------------------

    def _load_fabrary_db(self) -> list[dict[str, Any]]:
        """Load and return raw deck entries from fabrary_decks.json."""
        if not _FABRARY_DECKS_PATH.exists():
            return []
        try:
            data = json.loads(_FABRARY_DECKS_PATH.read_text(encoding="utf-8"))
            return list(data.get("decks", []))
        except Exception:
            return []

    def _resolve_fabrary_deck(self, deck_entry: dict[str, Any]) -> list[str]:
        """Map a fabrary deck entry to a list of card IDs legal for the current format.

        Strategy:
        - For each {name, count} entry look up all pitch versions of that card
          that are legal and compatible with the hero.
        - Include up to ``count`` copies of *each* pitch version found.
        - Apply the per-card-ID copy limit (2 for silver_age, 3 for CC).
        - If the list exceeds ``max_deck_size`` it is randomly trimmed.
        - If it falls below ``min_deck_size`` it is padded with generic filler.
        """
        hero_id = str(deck_entry.get("hero_id", ""))
        if hero_id not in self._heroes:
            return []
        hero = self._heroes[hero_id]

        # Build name → [Card, ...] lookup restricted to this hero + format.
        by_name: dict[str, list[Card]] = {}
        for card in self._cards.values():
            if not self._is_card_legal_for_format(card.id):
                continue
            if not self._is_deck_candidate(card):
                continue
            if not self._is_card_compatible_with_hero(card, hero):
                continue
            by_name.setdefault(card.name.lower(), []).append(card)

        copy_limit = 2 if self._format == "silver_age" else 3
        deck: list[str] = []

        for entry in deck_entry.get("cards", []):
            raw_name = str(entry.get("name", "")).strip()
            count = max(1, int(entry.get("count", 1)))
            matches = by_name.get(raw_name.lower(), [])
            if not matches:
                continue  # card not in DB or not legal/compatible for this hero+format
            per_copy = min(count, copy_limit)
            # Sort pitch=1 (red) first so trimming preserves the strongest copies.
            for card in sorted(matches, key=lambda c: c.pitch):
                deck.extend([card.id] * per_copy)

        if not deck:
            return []

        min_cards, max_cards = self._format_deck_bounds()

        # Trim excess randomly if deck exceeds format maximum.
        if len(deck) > max_cards:
            self._rng.shuffle(deck)
            deck = deck[:max_cards]

        # Pad below minimum with generic legal filler cards.
        if len(deck) < min_cards:
            existing_ids = set(deck)
            filler_pool = [
                c for c in self._cards.values()
                if self._is_card_legal_for_format(c.id)
                and self._is_deck_candidate(c)
                and self._is_card_compatible_with_hero(c, hero)
                and c.id not in existing_ids
            ]
            self._rng.shuffle(filler_pool)
            for filler in filler_pool:
                if len(deck) >= min_cards:
                    break
                deck.append(filler.id)

        return deck

    def _fabrary_deck_options_for_format(self) -> list[dict[str, Any]]:
        """Return deck option dicts for all fabrary decks matching the current format.

        Each option has the same shape as procedural options plus extra fields:
        ``source``, ``source_url``, ``description``, and ``_card_ids``
        (the pre-resolved card ID list used by ``_new_player``).
        Keys are prefixed with ``"fab_"`` to avoid collisions with procedural keys.
        """
        raw_decks = self._load_fabrary_db()
        options: list[dict[str, Any]] = []
        for deck_entry in raw_decks:
            if str(deck_entry.get("format", "")) != self._format:
                continue
            hero_id = str(deck_entry.get("hero_id", ""))
            if hero_id not in self._heroes:
                continue
            card_ids = self._resolve_fabrary_deck(deck_entry)
            if not card_ids:
                continue
            options.append(
                {
                    "key": deck_entry["id"],
                    "label": deck_entry.get("name", deck_entry["id"]),
                    "hero_id": hero_id,
                    "style": str(deck_entry.get("style", "balanced")),
                    "deck_size": len(card_ids),
                    "source": "fabrary",
                    "source_url": deck_entry.get("source_url", ""),
                    "description": deck_entry.get("description", ""),
                    "_card_ids": card_ids,
                }
            )
        return options

    def _start_match(
        self,
        *,
        agent_hero_id: str,
        opponent_hero_id: str,
        agent_deck_style: str,
        opponent_deck_style: str,
        agent_deck_ids: Optional[list[str]] = None,
        opponent_deck_ids: Optional[list[str]] = None,
    ) -> None:
        self._players = [
            self._new_player(self._heroes[agent_hero_id], hero_slot=0, deck_style=agent_deck_style, deck_ids=agent_deck_ids),
            self._new_player(self._heroes[opponent_hero_id], hero_slot=1, deck_style=opponent_deck_style, deck_ids=opponent_deck_ids),
        ]
        self._turn = 1
        self._active_player = 0
        self._phase = "action"
        self._pending_combat = None
        self._last_event = "Game start"
        self._draw_up(0)
        self._draw_up(1)
        self._start_turn(0)

    def _starting_life(self, hero: Hero) -> int:
        # Talishar profile: Silver Age uses young-hero life profile.
        if self._format == "silver_age":
            return min(hero.life, 20)
        return hero.life

    def _is_deck_candidate(self, card: Card) -> bool:
        if "hero" in card.card_types:
            return False
        if "attack_action" in card.card_types:
            return True
        if card.defense > 0:
            return True
        return False

    def _is_card_compatible_with_hero(self, card: Card, hero: Hero) -> bool:
        card_class = (card.card_class or "Generic").strip().upper()
        hero_class = (hero.hero_class or "Generic").strip().upper()
        if card_class not in {"", "GENERIC", "NONE"} and card_class != hero_class:
            return False

        card_talent = (card.talent or "").strip().upper()
        hero_talent = (hero.talent or "").strip().upper()
        if card_talent not in {"", "NONE"} and card_talent != hero_talent:
            return False

        return True

    def _hero_pool_for_format(self) -> list[Hero]:
        heroes = list(self._heroes.values())
        if self._format == "silver_age":
            filtered = [h for h in heroes if h.life < 30]
        else:
            filtered = [h for h in heroes if h.life >= 30]
        if not filtered:
            filtered = heroes
        filtered.sort(key=lambda h: h.name.lower())
        return filtered

    def _is_card_legal_for_format(self, card_id: str) -> bool:
        card = self._cards[card_id]
        if self._format == "classic_constructed":
            return str(card.legality.get("classic_constructed", "legal")).lower() == "legal"

        if self._format == "silver_age":
            return str(card.legality.get("silver_age", "banned")).lower() == "legal"

        return True

    def _draw_card(self, player_idx: int) -> bool:
        p = self._players[player_idx]
        if not p.deck and p.discard:
            p.deck = p.discard
            p.discard = []
            self._rng.shuffle(p.deck)
        if not p.deck:
            return False
        p.hand.append(p.deck.pop())
        return True

    def _draw_up(self, player_idx: int) -> None:
        p = self._players[player_idx]
        while len(p.hand) < p.hero.intellect:
            if not self._draw_card(player_idx):
                break

    def _start_turn(self, player_idx: int) -> None:
        p = self._players[player_idx]
        p.resources = 3
        p.action_points = 1
        self._active_player = player_idx
        self._phase = "action"

    def _end_turn(self, player_idx: int) -> None:
        p = self._players[player_idx]
        # End-of-turn cleanup: cycle remaining hand so players do not deadlock
        # with unplayable hands across turns.
        if p.hand:
            p.discard.extend(p.hand)
            p.hand = []
        self._draw_up(player_idx)
        p.resources = 0
        p.action_points = 0
        self._turn += 1

    def _agent_play_attack(self, hand_idx: int) -> float:
        attacker = self._players[0]
        defender = self._players[1]

        card_id = attacker.hand[hand_idx]
        card = self._cards[card_id]

        if "attack_action" not in card.card_types:
            self._last_event = f"{card.name} is not an attack action"
            return -0.05
        if attacker.resources < card.cost or attacker.action_points <= 0:
            self._last_event = "Insufficient resources or action points"
            return -0.05

        attacker.resources -= card.cost
        attacker.action_points -= 1
        attacker.hand.pop(hand_idx)

        self._pending_combat = CombatState(
            attacker=0,
            defender=1,
            attack_card_id=card.id,
            attack_power=card.power,
            blocks=[],
        )
        self._phase = "defense"

        # Opponent blocks greedily, then combat resolves immediately on agent attacks.
        self._opponent_auto_block()
        reward = self._resolve_combat()
        # Combat is fully resolved synchronously in this simplified simulator.
        self._phase = "action"

        # If card had go again, restore one action point.
        if "go_again" in card.keywords:
            attacker.action_points += 1

        if attacker.action_points <= 0:
            self._last_event += " | Action points depleted"

        return reward

    def _agent_block(self, hand_idx: int) -> float:
        if self._pending_combat is None:
            self._last_event = "No attack to defend"
            return -0.05

        blocker = self._players[0]
        card_id = blocker.hand[hand_idx]
        card = self._cards[card_id]

        if card.defense <= 0:
            self._last_event = f"{card.name} cannot block"
            return -0.05

        blocker.hand.pop(hand_idx)
        blocker.discard.append(card.id)
        self._pending_combat.blocks.append((0, card.id, card.defense))
        self._last_event = f"Agent blocks with {card.name} for {card.defense}"
        return 0.0

    def _step_self_play(self, parsed: str) -> float:
        """Single-policy self-play transition logic (controls both players)."""
        acting_player = self._active_player

        if self._phase == "action":
            if parsed == "pass":
                self._last_event = f"P{acting_player} passed turn"
                self._end_turn(acting_player)
                self._start_turn(1 - acting_player)
                return 0.0
            if parsed.startswith("play "):
                idx = int(parsed.split(" ")[1])
                return self._active_play_attack(acting_player, idx)
            return 0.0

        if self._phase == "defense":
            if parsed == "pass":
                if self._pending_combat is None:
                    self._phase = "action"
                    return 0.0
                attacker_idx = self._pending_combat.attacker
                reward = self._resolve_combat(perspective_idx=acting_player)
                # If the attacker has no AP left, auto-advance to the next
                # player's turn so no wasted "pass" step is needed.
                attacker = self._players[attacker_idx]
                if attacker.action_points <= 0:
                    self._end_turn(attacker_idx)
                    self._start_turn(1 - attacker_idx)
                else:
                    self._phase = "action"
                    self._active_player = attacker_idx
                return reward
            if parsed.startswith("block "):
                idx = int(parsed.split(" ")[1])
                reward_block = self._active_block(acting_player, idx)
                # One block per attack: resolve immediately after blocking
                attacker_idx = self._pending_combat.attacker
                reward_block += self._resolve_combat(perspective_idx=acting_player)
                attacker = self._players[attacker_idx]
                if attacker.action_points <= 0:
                    self._end_turn(attacker_idx)
                    self._start_turn(1 - attacker_idx)
                else:
                    self._phase = "action"
                    self._active_player = attacker_idx
                return reward_block

        return 0.0

    def _active_play_attack(self, attacker_idx: int, hand_idx: int) -> float:
        """Play an attack card for whichever player is currently active."""
        attacker = self._players[attacker_idx]
        defender_idx = 1 - attacker_idx

        card_id = attacker.hand[hand_idx]
        card = self._cards[card_id]

        if "attack_action" not in card.card_types:
            self._last_event = f"{card.name} is not an attack action"
            return -0.05
        if attacker.resources < card.cost or attacker.action_points <= 0:
            self._last_event = "Insufficient resources or action points"
            return -0.05

        attacker.resources -= card.cost
        attacker.action_points -= 1
        attacker.hand.pop(hand_idx)

        self._pending_combat = CombatState(
            attacker=attacker_idx,
            defender=defender_idx,
            attack_card_id=card.id,
            attack_power=card.power,
            blocks=[],
        )
        self._phase = "defense"
        self._active_player = defender_idx
        self._last_event = f"P{attacker_idx} attacks with {card.name} ({card.power})"
        return 0.0

    def _active_block(self, blocker_idx: int, hand_idx: int) -> float:
        """Block with the currently acting defender in self-play mode."""
        if self._pending_combat is None:
            self._last_event = "No attack to defend"
            return -0.05

        blocker = self._players[blocker_idx]
        card_id = blocker.hand[hand_idx]
        card = self._cards[card_id]

        if card.defense <= 0:
            self._last_event = f"{card.name} cannot block"
            return -0.05

        blocker.hand.pop(hand_idx)
        blocker.discard.append(card.id)
        self._pending_combat.blocks.append((blocker_idx, card.id, card.defense))
        self._last_event = f"P{blocker_idx} blocks with {card.name} for {card.defense}"
        return 0.0

    def _opponent_turn(self) -> None:
        self._start_turn(1)
        opp = self._players[1]

        playable: list[tuple[int, Card]] = []
        for i, cid in enumerate(opp.hand):
            c = self._cards[cid]
            if "attack_action" in c.card_types and c.cost <= opp.resources:
                playable.append((i, c))

        if not playable:
            self._last_event = "Opponent passed"
            self._end_turn(1)
            self._start_turn(0)
            return

        idx, best = max(playable, key=lambda x: (x[1].power, -x[1].cost))
        opp.resources -= best.cost
        opp.action_points -= 1
        opp.hand.pop(idx)

        self._pending_combat = CombatState(
            attacker=1,
            defender=0,
            attack_card_id=best.id,
            attack_power=best.power,
            blocks=[],
        )
        self._phase = "defense"
        self._last_event = f"Opponent attacks with {best.name} ({best.power})"

    def _opponent_auto_block(self) -> None:
        """Opponent blocks with at most one card (the highest-defense card in hand).
        Limiting to a single blocker keeps attacks meaningful and damage non-zero.
        """
        if self._pending_combat is None:
            return
        if self._pending_combat.defender != 1:
            return

        defender = self._players[1]

        # Pick the single best blocking card.
        best_idx: int | None = None
        best_def = 0
        for i, cid in enumerate(defender.hand):
            c = self._cards[cid]
            if c.defense > 0 and c.defense > best_def:
                best_def = c.defense
                best_idx = i

        if best_idx is not None:
            cid = defender.hand.pop(best_idx)
            card = self._cards[cid]
            defender.discard.append(cid)
            self._pending_combat.blocks.append((1, cid, card.defense))

    def _resolve_combat(self, perspective_idx: Optional[int] = None) -> float:
        if self._pending_combat is None:
            return 0.0

        combat = self._pending_combat
        attacker = self._players[combat.attacker]
        defender = self._players[combat.defender]

        total_block = sum(b[2] for b in combat.blocks)
        damage = max(0, combat.attack_power - total_block)
        defender.life -= damage

        attacker.discard.append(combat.attack_card_id)
        self._pending_combat = None

        if perspective_idx is not None:
            if perspective_idx == combat.attacker:
                self._last_event = f"P{combat.attacker} dealt {damage} damage"
                return float(damage) * 0.01
            self._last_event = f"P{combat.attacker} dealt {damage} damage to P{combat.defender}"
            return -float(damage) * 0.01

        if combat.attacker == 0:
            self._last_event = f"Agent dealt {damage} damage"
            return float(damage) * 0.005

        self._last_event = f"Opponent dealt {damage} damage"
        return -float(damage) * 0.005

    def _end_turn_and_run_opponent(self) -> None:
        self._end_turn(0)
        self._opponent_turn()

    def _legal_actions(self) -> list[str]:
        if self._selection_stage:
            return [f"choose_deck {o['key']}" for o in self._deck_options_for_format()]
        if not self._initialized or len(self._players) < 2:
            return ["pass"]
        if self._is_terminal():
            return []

        if self._phase == "action":
            actor_idx = self._active_player
            actor = self._players[actor_idx]
            if not self._self_play and actor_idx != 0:
                return ["pass"]

            actions = ["pass"]
            if actor.action_points > 0:
                for i, cid in enumerate(actor.hand):
                    card = self._cards[cid]
                    if "attack_action" in card.card_types and card.cost <= actor.resources:
                        actions.append(f"play {i}")
            return actions

        if self._phase == "defense":
            if self._pending_combat is None:
                return ["pass"]
            if not self._self_play and self._pending_combat.defender != 0:
                return ["pass"]
            if self._self_play and self._pending_combat.defender != self._active_player:
                return ["pass"]

            blocker_idx = self._active_player if self._self_play else 0

            actions = ["pass"]
            for i, cid in enumerate(self._players[blocker_idx].hand):
                if self._cards[cid].defense > 0:
                    actions.append(f"block {i}")
            return actions

        return ["pass"]

    def _is_terminal(self) -> bool:
        if not self._players:
            return False
        return self._players[0].life <= 0 or self._players[1].life <= 0

    def _normalize_action(self, action: Any) -> str:
        if self._selection_stage and isinstance(action, str):
            return action.strip().lower()
        if isinstance(action, int):
            if self._phase == "action":
                return f"play {action}"
            if self._phase == "defense":
                return f"block {action}"
        text = str(action).strip().lower()
        if text in {"pass", "end", "end turn"}:
            return "pass"
        if text.startswith("play"):
            parts = text.split()
            if len(parts) == 2 and parts[1].isdigit():
                return f"play {parts[1]}"
        if text.startswith("block"):
            parts = text.split()
            if len(parts) == 2 and parts[1].isdigit():
                return f"block {parts[1]}"
        return text

    def _observation(self) -> dict[str, Any]:
        if self._selection_stage:
            return {
                "turn": self._turn,
                "format": self._format,
                "stage": "deck_selection",
                "phase": self._phase,
                "active_player": self._active_player,
                "deck_options": self._deck_options_for_format(),
                "selected_deck": self._selected_deck_option,
                "matchup": self._selected_opponent_deck_option,
                "legal_actions": self._legal_actions(),
                "last_event": self._last_event,
            }

        agent = self._players[0]
        opp = self._players[1]

        hand_view = []
        for i, cid in enumerate(agent.hand):
            c = self._cards[cid]
            hand_view.append(
                {
                    "index": i,
                    "id": c.id,
                    "name": c.name,
                    "cost": c.cost,
                    "power": c.power,
                    "defense": c.defense,
                    "card_types": list(c.card_types),
                    "keywords": list(c.keywords),
                }
            )

        pending = None
        if self._pending_combat is not None:
            attack_card = self._cards[self._pending_combat.attack_card_id]
            pending = {
                "attacker": self._pending_combat.attacker,
                "defender": self._pending_combat.defender,
                "attack_card": attack_card.name,
                "attack_power": self._pending_combat.attack_power,
                "total_block": sum(b[2] for b in self._pending_combat.blocks),
            }

        return {
            "turn": self._turn,
            "format": self._format,
            "stage": "play",
            "phase": self._phase,
            "active_player": self._active_player,
            "selected_deck": self._selected_deck_option,
            "matchup": self._selected_opponent_deck_option,
            "agent": {
                "hero": agent.hero.name,
                "life": agent.life,
                "resources": agent.resources,
                "action_points": agent.action_points,
                "deck": len(agent.deck),
                "discard": len(agent.discard),
                "hand": hand_view,
            },
            "opponent": {
                "hero": opp.hero.name,
                "life": opp.life,
                "resources": opp.resources,
                "action_points": opp.action_points,
                "deck": len(opp.deck),
                "discard": len(opp.discard),
                "hand_size": len(opp.hand),
            },
            "pending_combat": pending,
            "legal_actions": self._legal_actions(),
            "last_event": self._last_event,
        }

    def _estimate_win_probabilities(self, obs: Optional[dict[str, Any]] = None) -> tuple[float, float]:
        """Heuristic win probability estimate from current FaB state.

        Returns ``(agent_win_prob, opponent_win_prob)``.
        """
        if obs is None:
            obs = self._observation()

        if obs.get("stage") == "deck_selection":
            return 0.5, 0.5

        agent = obs.get("agent") if isinstance(obs.get("agent"), dict) else {}
        opp = obs.get("opponent") if isinstance(obs.get("opponent"), dict) else {}

        agent_life = float(agent.get("life", 0.0))
        opp_life = float(opp.get("life", 0.0))

        if opp_life <= 0 < agent_life:
            return 1.0, 0.0
        if agent_life <= 0 < opp_life:
            return 0.0, 1.0

        agent_hand_size = len(agent.get("hand", [])) if isinstance(agent.get("hand"), list) else 0
        opp_hand_size = int(opp.get("hand_size", 0) or 0)
        agent_resources = float(agent.get("resources", 0.0))
        opp_resources = float(opp.get("resources", 0.0))
        agent_ap = float(agent.get("action_points", 0.0))
        opp_ap = float(opp.get("action_points", 0.0))
        agent_deck = float(agent.get("deck", 0.0))
        opp_deck = float(opp.get("deck", 0.0))

        agent_score = (
            1.8 * agent_life
            + 1.0 * agent_hand_size
            + 0.6 * agent_resources
            + 0.8 * agent_ap
            + 0.05 * agent_deck
        )
        opp_score = (
            1.8 * opp_life
            + 1.0 * opp_hand_size
            + 0.6 * opp_resources
            + 0.8 * opp_ap
            + 0.05 * opp_deck
        )

        pending = obs.get("pending_combat")
        if isinstance(pending, dict):
            atk = float(pending.get("attack_power", 0.0) or 0.0)
            blk = float(pending.get("total_block", 0.0) or 0.0)
            net = max(0.0, atk - blk)
            attacker = int(pending.get("attacker", 0) or 0)
            if attacker == 0:
                agent_score += 1.5 * net
            else:
                opp_score += 1.5 * net

        active_player = int(obs.get("active_player", 0) or 0)
        if active_player == 0:
            agent_score += 0.4
        else:
            opp_score += 0.4

        diff = (agent_score - opp_score) / 8.0
        agent_p = 1.0 / (1.0 + math.exp(-diff))
        agent_p = max(0.0, min(1.0, agent_p))
        return agent_p, 1.0 - agent_p

    def _render_text(self) -> str:
        obs = self._observation()
        if obs.get("stage") == "deck_selection":
            lines = [
                "Flesh and Blood (Talishar-inspired)",
                f"Format: {obs['format']} | Stage: deck_selection",
                "Choose a deck:",
            ]
            for item in obs.get("deck_options", []):
                lines.append(f"  - {item['label']} ({item['deck_size']} cards) -> choose_deck {item['key']}")
            lines.append("Legal actions: " + ", ".join(obs.get("legal_actions", [])))
            return "\n".join(lines)

        p_agent, p_opp = self._estimate_win_probabilities(obs)
        lines = [
            "Flesh and Blood (Talishar-inspired)",
            f"Turn: {obs['turn']} | Format: {obs['format']} | Phase: {obs['phase']} | Active: P{obs['active_player']}",
            f"Agent ({obs['agent']['hero']}): life={obs['agent']['life']} hand={len(obs['agent']['hand'])} resources={obs['agent']['resources']} AP={obs['agent']['action_points']}",
            f"Opponent ({obs['opponent']['hero']}): life={obs['opponent']['life']} hand={obs['opponent']['hand_size']} resources={obs['opponent']['resources']} AP={obs['opponent']['action_points']}",
            f"Win % | Agent: {p_agent:.1%}  Opponent: {p_opp:.1%}",
            f"Last event: {obs['last_event']}",
            "Legal actions: " + ", ".join(obs["legal_actions"]),
        ]
        if obs["pending_combat"]:
            lines.append(
                "Combat: "
                f"{obs['pending_combat']['attack_card']} "
                f"atk={obs['pending_combat']['attack_power']} "
                f"block={obs['pending_combat']['total_block']}"
            )
        return "\n".join(lines)


class FleshAndBloodFactory(RLIPEnvironmentFactory):
    def __init__(
        self,
        env_id: str,
        *,
        agent_hero_id: str = "hero_dorinthea_ironsong",
        opponent_hero_id: str = "hero_rhinar_reckless_rampage",
        max_turns: int = 60,
        deck_size: int = 36,
        format: str = "classic_constructed",
        two_phase_deckbuild: bool = False,
        self_play: bool = False,
    ) -> None:
        self._env_id = env_id
        self._agent_hero_id = agent_hero_id
        self._opponent_hero_id = opponent_hero_id
        self._max_turns = max_turns
        self._deck_size = deck_size
        self._format = format
        self._two_phase_deckbuild = two_phase_deckbuild
        self._self_play = self_play

    @property
    def env_info(self) -> EnvironmentInfo:
        return EnvironmentInfo(
            env_id=self._env_id,
            description=(
                "Talishar-inspired Flesh and Blood simulation with structured card "
                "database, hand/deck/combat phases, and scripted opponent policy."
                if not self._self_play
                else "Talishar-inspired Flesh and Blood self-play simulation where "
                "one policy controls both heroes across alternating turns."
            ),
            tags=[
                "tcg",
                "flesh-and-blood",
                "card-game",
                "turn-based",
                "simulator",
                *( ["deck-selection"] if self._two_phase_deckbuild else [] ),
                *(["self-play"] if self._self_play else []),
            ],
            namespace="flesh_and_blood",
            render_modes=["ansi", "rgb_array"],
            max_episode_steps=self._max_turns,
            suggested_hyperparameters=SuggestedHyperparameters(
                agent_type="tabular_q",
                n_episodes_baseline=300,
                n_episodes_instruction=300,
                max_steps=self._max_turns,
                alpha=0.1,
                gamma=0.99,
                epsilon=1.0,
                epsilon_min=0.05,
                epsilon_decay=0.997,
                sub_goal_threshold=0.5,
                top_k=3,
                min_episode_visits=2,
            ),
        )

    def create(
        self,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> FleshAndBloodEnvironment:
        _ = render_mode
        return FleshAndBloodEnvironment(
            seed=kwargs.get("seed"),
            agent_hero_id=kwargs.get("agent_hero_id", self._agent_hero_id),
            opponent_hero_id=kwargs.get("opponent_hero_id", self._opponent_hero_id),
            max_turns=int(kwargs.get("max_turns", self._max_turns)),
            deck_size=int(kwargs.get("deck_size", self._deck_size)),
            format=str(kwargs.get("format", self._format)),
            two_phase_deckbuild=bool(kwargs.get("two_phase_deckbuild", self._two_phase_deckbuild)),
            self_play=bool(kwargs.get("self_play", self._self_play)),
            render_mode=render_mode,
        )


FLESH_AND_BLOOD_TALISHAR_V0 = FleshAndBloodFactory("FleshAndBlood-Talishar-v0")
FLESH_AND_BLOOD_SELFPLAY_V0 = FleshAndBloodFactory(
    "FleshAndBlood-SelfPlay-v0",
    self_play=True,
)
FLESH_AND_BLOOD_DECKBUILD_V0 = FleshAndBloodFactory(
    "FleshAndBlood-DeckBuild-v0",
    two_phase_deckbuild=True,
)
ALL_FAB_FACTORIES: list[FleshAndBloodFactory] = [
    FLESH_AND_BLOOD_TALISHAR_V0,
    FLESH_AND_BLOOD_SELFPLAY_V0,
    FLESH_AND_BLOOD_DECKBUILD_V0,
]


def register_mcp_tools(
    *, mcp: Any, registry: Any, log: Any, trained_agents: Optional[dict] = None
) -> int:
    """Register environment-specific MCP tools for Flesh and Blood.

    This function is discovered and called by the MCP plugin at startup.
    Returning an integer allows the plugin to report how many tools were added.

    Parameters
    ----------
    trained_agents:
        When provided (passed by the MCP plugin), any agent trained by
        ``fab_evaluate_deck_matchup`` will be stored here so that
        ``rl_render_policy`` can replay it later.
    """
    global _FAB_CUSTOM_TOOLS_REGISTERED
    if _FAB_CUSTOM_TOOLS_REGISTERED:
        return 0
    if registry is None:
        return 0

    def _build_agent(agent_type: str, hyperparams: dict[str, Any]) -> Any:
        if agent_type == "tabular_q":
            mod = importlib.import_module("rlip.rl_agents.tabular_q")
            return mod.TabularQAgent(
                alpha=float(hyperparams.get("alpha", 0.1)),
                gamma=float(hyperparams.get("gamma", 0.99)),
                epsilon=float(hyperparams.get("epsilon", 1.0)),
                epsilon_min=float(hyperparams.get("epsilon_min", 0.01)),
                epsilon_decay=float(hyperparams.get("epsilon_decay", 0.995)),
                seed=hyperparams.get("seed"),
            )
        if agent_type == "dqn":
            mod = importlib.import_module("rlip.rl_agents.dqn")
            return mod.DQNAgent(
                hidden_size=int(hyperparams.get("hidden_size", 64)),
                lr=float(hyperparams.get("lr", 1e-3)),
                gamma=float(hyperparams.get("gamma", 0.99)),
                epsilon=float(hyperparams.get("epsilon", 1.0)),
                epsilon_min=float(hyperparams.get("epsilon_min", 0.01)),
                epsilon_decay=float(hyperparams.get("epsilon_decay", 0.995)),
                buffer_size=int(hyperparams.get("buffer_size", 10000)),
                batch_size=int(hyperparams.get("batch_size", 64)),
                target_update_freq=int(hyperparams.get("target_update_freq", 100)),
                seed=hyperparams.get("seed"),
            )
        if agent_type == "ppo":
            mod = importlib.import_module("rlip.rl_agents.ppo")
            return mod.PPOAgent(
                hidden_size=int(hyperparams.get("hidden_size", 64)),
                lr_actor=float(hyperparams.get("lr_actor", 1e-3)),
                lr_critic=float(hyperparams.get("lr_critic", 1e-3)),
                gamma=float(hyperparams.get("gamma", 0.99)),
                lam=float(hyperparams.get("lam", 0.95)),
                clip_eps=float(hyperparams.get("clip_eps", 0.2)),
                n_steps=int(hyperparams.get("n_steps", 256)),
                ppo_epochs=int(hyperparams.get("ppo_epochs", 4)),
                mini_batch_size=int(hyperparams.get("mini_batch_size", 64)),
                seed=hyperparams.get("seed"),
            )
        raise ValueError(f"Unsupported agent type: {agent_type!r}")

    def _run_eval_episode(env: Any, agent: Any, max_steps: int, seed: Optional[int]) -> dict[str, Any]:
        reset_out = env.reset(seed=seed)
        obs = reset_out.observation if hasattr(reset_out, "observation") else reset_out.get("observation", reset_out)
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        for step in range(1, max_steps + 1):
            if hasattr(agent, "act_greedy"):
                action = agent.act_greedy(obs)
            else:
                action = agent.act(obs)
            out = env.step(action)
            obs = out.observation if hasattr(out, "observation") else out.get("observation", obs)
            reward = float(out.reward if hasattr(out, "reward") else out.get("reward", 0.0))
            terminated = bool(out.terminated if hasattr(out, "terminated") else out.get("terminated", False))
            truncated = bool(out.truncated if hasattr(out, "truncated") else out.get("truncated", False))
            total_reward += reward
            steps = step
            if terminated or truncated:
                break

        return {
            "steps": steps,
            "total_reward": total_reward,
            "terminated": terminated,
            "truncated": truncated,
            "final_observation": obs,
        }

    def _fab_win_probabilities(obs: Any) -> tuple[float, float]:
        if not isinstance(obs, dict):
            return 0.5, 0.5

        agent = obs.get("agent") if isinstance(obs.get("agent"), dict) else {}
        opp = obs.get("opponent") if isinstance(obs.get("opponent"), dict) else {}

        agent_life = float(agent.get("life", 0.0))
        opp_life = float(opp.get("life", 0.0))

        if opp_life <= 0 < agent_life:
            return 1.0, 0.0
        if agent_life <= 0 < opp_life:
            return 0.0, 1.0

        agent_hand_size = len(agent.get("hand", [])) if isinstance(agent.get("hand"), list) else 0
        opp_hand_size = int(opp.get("hand_size", 0) or 0)

        agent_resources = float(agent.get("resources", 0.0))
        opp_resources = float(opp.get("resources", 0.0))
        agent_ap = float(agent.get("action_points", 0.0))
        opp_ap = float(opp.get("action_points", 0.0))
        agent_deck = float(agent.get("deck", 0.0))
        opp_deck = float(opp.get("deck", 0.0))

        agent_score = (
            1.8 * agent_life
            + 1.0 * agent_hand_size
            + 0.6 * agent_resources
            + 0.8 * agent_ap
            + 0.05 * agent_deck
        )
        opp_score = (
            1.8 * opp_life
            + 1.0 * opp_hand_size
            + 0.6 * opp_resources
            + 0.8 * opp_ap
            + 0.05 * opp_deck
        )

        pending = obs.get("pending_combat")
        if isinstance(pending, dict):
            atk = float(pending.get("attack_power", 0.0) or 0.0)
            blk = float(pending.get("total_block", 0.0) or 0.0)
            net = max(0.0, atk - blk)
            attacker = int(pending.get("attacker", 0) or 0)
            if attacker == 0:
                agent_score += 1.5 * net
            else:
                opp_score += 1.5 * net

        active_player = int(obs.get("active_player", 0) or 0)
        if active_player == 0:
            agent_score += 0.4
        else:
            opp_score += 0.4

        diff = (agent_score - opp_score) / 8.0
        agent_p = 1.0 / (1.0 + math.exp(-diff))
        agent_p = max(0.0, min(1.0, agent_p))
        return agent_p, 1.0 - agent_p

    def _fab_outcome_score(obs: Any, *, terminated: bool) -> float:
        if isinstance(obs, dict):
            agent = obs.get("agent") if isinstance(obs.get("agent"), dict) else {}
            opp = obs.get("opponent") if isinstance(obs.get("opponent"), dict) else {}
            agent_life = float(agent.get("life", 0.0) or 0.0)
            opp_life = float(opp.get("life", 0.0) or 0.0)
            if terminated:
                if opp_life <= 0 < agent_life:
                    return 1.0
                if agent_life <= 0 < opp_life:
                    return 0.0
                if agent_life == opp_life:
                    return 0.5
        p_agent, _ = _fab_win_probabilities(obs)
        return float(p_agent)

    def _get_deck_options(format_name: str, seed: Optional[int]) -> list[dict[str, Any]]:
        env = registry.create("FleshAndBlood-DeckBuild-v0", render_mode=None, format=format_name)
        try:
            reset_out = env.reset(seed=seed, options={"format": format_name, "two_phase_deckbuild": True})
            obs = reset_out.observation if hasattr(reset_out, "observation") else reset_out.get("observation", {})
            options = obs.get("deck_options") if isinstance(obs, dict) else None
            return list(options) if isinstance(options, list) else []
        finally:
            env.close()

    def _evaluate_deck_vs_matchup(
        *,
        deck_option: dict[str, Any],
        matchup_option: dict[str, Any],
        format_name: str,
        inner_agent_type: str,
        inner_train_episodes: int,
        inner_eval_episodes: int,
        inner_max_steps: int,
        seed: Optional[int],
    ) -> dict[str, Any]:
        env_kwargs: dict[str, Any] = {
            "render_mode": None,
            "format": format_name,
            "agent_hero_id": str(deck_option.get("hero_id")),
            "opponent_hero_id": str(matchup_option.get("hero_id")),
            "deck_size": int(deck_option.get("deck_size", 40) or 40),
            "agent_deck_style": str(deck_option.get("style", "balanced")),
            "opponent_deck_style": str(matchup_option.get("style", "balanced")),
        }

        agent = _build_agent(inner_agent_type, {})
        train_env = registry.create("FleshAndBlood-Talishar-v0", **env_kwargs)
        try:
            train_result = agent.train(
                train_env,
                n_episodes=inner_train_episodes,
                max_steps=inner_max_steps,
                seed=seed,
            )
        finally:
            train_env.close()

        eval_env = registry.create("FleshAndBlood-Talishar-v0", **env_kwargs)
        eval_scores: list[float] = []
        try:
            base_seed = 0 if seed is None else int(seed)
            for ep in range(inner_eval_episodes):
                ep_seed = base_seed + 10_000 + ep
                out = _run_eval_episode(eval_env, agent, max_steps=inner_max_steps, seed=ep_seed)
                eval_scores.append(
                    _fab_outcome_score(
                        out.get("final_observation"),
                        terminated=bool(out.get("terminated", False)),
                    )
                )
        finally:
            eval_env.close()

        win_rate = (sum(eval_scores) / len(eval_scores)) if eval_scores else 0.5
        return {
            "win_rate": float(win_rate),
            "train_mean_reward": float(train_result.mean_reward),
            "train_best_reward": float(train_result.best_reward),
            "_agent": agent,
            "_train_result": train_result,
        }

    @mcp.tool()
    def fab_list_deck_options(
        format_name: str = "silver_age",
        seed: Optional[int] = None,
    ) -> str:
        """List all available hero/deck options for a Flesh and Blood format."""
        try:
            options = _get_deck_options(format_name, seed)
        except Exception as exc:
            log.exception("fab_list_deck_options error")
            return f"Error listing deck options: {exc}"

        result = {
            "format": format_name,
            "deck_options_count": len(options),
            "deck_options": options,
        }
        return json.dumps(result, indent=2)

    @mcp.tool()
    def fab_estimate_win_probabilities(observation_json: str) -> str:
        """Estimate win probabilities for both players from a FaB observation."""
        try:
            obs = json.loads(observation_json)
        except json.JSONDecodeError as exc:
            return f"Error: invalid JSON - {exc}"

        try:
            agent_p, opp_p = _fab_win_probabilities(obs)
        except Exception as exc:
            log.exception("fab_estimate_win_probabilities error")
            return f"Error computing win probabilities: {exc}"

        agent = obs.get("agent") if isinstance(obs.get("agent"), dict) else {}
        opp = obs.get("opponent") if isinstance(obs.get("opponent"), dict) else {}

        result = {
            "agent_win_probability": round(agent_p, 4),
            "opponent_win_probability": round(opp_p, 4),
            "inputs": {
                "agent_life": agent.get("life"),
                "opponent_life": opp.get("life"),
                "agent_hand_size": len(agent.get("hand", [])) if isinstance(agent.get("hand"), list) else agent.get("hand_size"),
                "opponent_hand_size": opp.get("hand_size"),
                "active_player": obs.get("active_player"),
            },
            "reasoning": (
                "Logistic model over life totals (x1.8), hand size (x1.0), "
                "resources (x0.6), action points (x0.8), deck size (x0.05), "
                "pending combat net damage (x1.5), and initiative bonus (+/-0.4)."
            ),
        }
        return json.dumps(result, indent=2)

    @mcp.tool()
    def fab_evaluate_deck_matchup(
        deck_key: str,
        matchup_key: str,
        format_name: str = "silver_age",
        inner_agent_type: str = "tabular_q",
        inner_train_episodes: int = 50,
        inner_eval_episodes: int = 10,
        inner_max_steps: int = 200,
        seed: Optional[int] = None,
    ) -> str:
        """Train and evaluate an inner gameplay agent for one FaB deck/matchup pair."""
        try:
            all_options = _get_deck_options(format_name, seed)
        except Exception as exc:
            return f"Error fetching deck options: {exc}"

        deck_option = next((o for o in all_options if str(o.get("key")) == deck_key), None)
        matchup_option = next((o for o in all_options if str(o.get("key")) == matchup_key), None)

        if deck_option is None:
            known = [str(o.get("key")) for o in all_options]
            return (
                f"Error: deck_key {deck_key!r} not found for format {format_name!r}.\n"
                f"Known keys: {known}"
            )
        if matchup_option is None:
            known = [str(o.get("key")) for o in all_options]
            return (
                f"Error: matchup_key {matchup_key!r} not found for format {format_name!r}.\n"
                f"Known keys: {known}"
            )

        try:
            stats = _evaluate_deck_vs_matchup(
                deck_option=deck_option,
                matchup_option=matchup_option,
                format_name=format_name,
                inner_agent_type=inner_agent_type,
                inner_train_episodes=inner_train_episodes,
                inner_eval_episodes=inner_eval_episodes,
                inner_max_steps=inner_max_steps,
                seed=seed,
            )
        except Exception as exc:
            log.exception("fab_evaluate_deck_matchup error")
            return f"Error evaluating deck matchup: {exc}"

        # Extract non-serialisable internal keys before building the result dict.
        _trained_agent = stats.pop("_agent", None)
        _train_result = stats.pop("_train_result", None)
        stats.pop("_env_kwargs", None)

        result: dict[str, Any] = {
            "deck_key": deck_key,
            "deck_label": deck_option.get("label", deck_key),
            "matchup_key": matchup_key,
            "matchup_label": matchup_option.get("label", matchup_key),
            "format": format_name,
            "inner_agent_type": inner_agent_type,
            "inner_train_episodes": inner_train_episodes,
            "inner_eval_episodes": inner_eval_episodes,
            **stats,
        }

        # If the plugin passed a trained_agents store, register the agent so
        # rl_render_policy can replay the policy directly.
        if trained_agents is not None and _trained_agent is not None:
            import uuid as _uuid  # noqa: PLC0415

            _agent_id = _uuid.uuid4().hex[:12]
            _registered_env_id = f"FleshAndBlood-matchup-{_agent_id}"

            # Register a factory baked with this matchup's hero / format so
            # rl_render_policy can recreate the exact environment.
            matchup_factory = FleshAndBloodFactory(
                _registered_env_id,
                agent_hero_id=str(deck_option.get("hero_id", "hero_dorinthea_ironsong")),
                opponent_hero_id=str(matchup_option.get("hero_id", "hero_rhinar_reckless_rampage")),
                deck_size=int(deck_option.get("deck_size", 40) or 40),
                format=format_name,
            )
            registry.register(matchup_factory)

            trained_agents[_agent_id] = {
                "agent":                _trained_agent,
                "env_id":               _registered_env_id,
                "agent_type":           inner_agent_type,
                "best_episode_history": getattr(_train_result, "best_episode_history", []),
                "use_language_state":   False,
                "train_result":         _train_result,
                "training_config": {
                    "n_episodes":  inner_train_episodes,
                    "max_steps":   inner_max_steps,
                    "seed":        seed,
                    "deck_key":    deck_key,
                    "matchup_key": matchup_key,
                },
            }

            result["agent_id"] = _agent_id
            result["registered_env_id"] = _registered_env_id

        return json.dumps(result, indent=2)

    @mcp.tool()
    def fab_meta_reward_for_deck(
        deck_key: str,
        format_name: str = "silver_age",
        inner_agent_type: str = "tabular_q",
        inner_train_episodes: int = 50,
        inner_eval_episodes: int = 10,
        matchups_per_deck: int = 3,
        inner_max_steps: int = 200,
        seed: Optional[int] = None,
    ) -> str:
        """Compute the meta-reward for a FaB deck by sampling matchups."""
        try:
            all_options = _get_deck_options(format_name, seed)
        except Exception as exc:
            return f"Error fetching deck options: {exc}"

        deck_option = next((o for o in all_options if str(o.get("key")) == deck_key), None)
        if deck_option is None:
            known = [str(o.get("key")) for o in all_options]
            return (
                f"Error: deck_key {deck_key!r} not found for format {format_name!r}.\n"
                f"Known keys: {known}"
            )

        opponent_pool = [o for o in all_options if str(o.get("hero_id")) != str(deck_option.get("hero_id"))]
        if not opponent_pool:
            opponent_pool = [o for o in all_options if str(o.get("key")) != deck_key]
        if not opponent_pool:
            opponent_pool = list(all_options)

        rng = random.Random(seed)
        if matchups_per_deck >= len(opponent_pool):
            sampled = list(opponent_pool)
        else:
            sampled = rng.sample(opponent_pool, matchups_per_deck)

        matchup_results: list[dict[str, Any]] = []
        base_seed = 0 if seed is None else int(seed)

        for i, matchup in enumerate(sampled):
            ep_seed = base_seed + i * 1000
            try:
                stats = _evaluate_deck_vs_matchup(
                    deck_option=deck_option,
                    matchup_option=matchup,
                    format_name=format_name,
                    inner_agent_type=inner_agent_type,
                    inner_train_episodes=inner_train_episodes,
                    inner_eval_episodes=inner_eval_episodes,
                    inner_max_steps=inner_max_steps,
                    seed=ep_seed,
                )
                stats.pop("_agent", None)
                stats.pop("_train_result", None)
                stats.pop("_env_kwargs", None)
                matchup_results.append(
                    {
                        "matchup_key": str(matchup.get("key", "")),
                        "matchup_label": str(matchup.get("label", matchup.get("key", "unknown"))),
                        "hero_id": matchup.get("hero_id"),
                        "win_rate": float(stats["win_rate"]),
                        "train_mean_reward": float(stats["train_mean_reward"]),
                        "error": None,
                    }
                )
            except Exception as exc:
                log.exception("fab_meta_reward_for_deck matchup error")
                matchup_results.append(
                    {
                        "matchup_key": str(matchup.get("key", "")),
                        "matchup_label": str(matchup.get("label", "")),
                        "hero_id": matchup.get("hero_id"),
                        "win_rate": 0.5,
                        "error": str(exc),
                    }
                )

        valid = [r for r in matchup_results if r.get("error") is None]
        meta_reward = (sum(r["win_rate"] for r in valid) / len(valid)) if valid else 0.5

        result = {
            "deck_key": deck_key,
            "deck_label": deck_option.get("label", deck_key),
            "format": format_name,
            "inner_agent_type": inner_agent_type,
            "inner_train_episodes": inner_train_episodes,
            "inner_eval_episodes": inner_eval_episodes,
            "matchups_per_deck": matchups_per_deck,
            "meta_reward": round(meta_reward, 4),
            "matchups_evaluated": len(matchup_results),
            "matchup_results": matchup_results,
        }
        return json.dumps(result, indent=2)

    @mcp.tool()
    def fab_resolve_deck_from_url(
        fabrary_url: str,
        side: str = "agent",
        format_name: str = "silver_age",
    ) -> str:
        """Resolve a Flesh and Blood deck from a fabrary.net public link.

        Parses the fabrary.net deck URL to extract the deck ID, looks up the deck
        in the static database, and resolves it to a card ID list legal for the
        specified format. The resolved deck can be used directly with environment
        setup or the fab_evaluate_deck_matchup tool.

        Args:
            fabrary_url: Full fabrary.net deck URL, e.g.
                "https://fabrary.net/decks/01KR40W4Z2ZS9EQPT6VT6CDSPE".
            side: Descriptive context – "agent" or "opponent". Does not affect
                resolution but is included in response for clarity.
            format_name: FaB format (silver_age, classic_constructed, cc, sa, blitz).
                Defaults to silver_age. The deck is verified against this format's
                legality.

        Returns:
            JSON string with deck_id, deck_name, hero_id, format, style,
            _card_ids (pre-resolved legal card list), source_url, and error
            info if resolution failed.
        """
        import re

        # Normalize format name
        try:
            # Create temp env to access format normalization
            temp_env = registry.create("FleshAndBlood-Talishar-v0", render_mode=None, format=format_name)
            normalized_format = temp_env._format
            temp_env.close()
        except Exception as exc:
            return json.dumps({
                "error": f"Invalid format {format_name!r}: {exc}",
                "side": side,
                "url": fabrary_url,
            }, indent=2)

        # Extract deck ID from URL: https://fabrary.net/decks/01KR40W4Z2ZS9EQPT6VT6CDSPE
        match = re.search(r"/decks/([a-zA-Z0-9]+)\b", fabrary_url)
        if not match:
            return json.dumps({
                "error": f"Could not parse deck ID from URL: {fabrary_url!r}",
                "expected_format": "https://fabrary.net/decks/{{DECK_ID}}",
                "side": side,
                "format": normalized_format,
            }, indent=2)

        deck_id = match.group(1)
        deck_key = f"fab_{deck_id.lower()}"

        # Load fabrary database and find the deck
        try:
            if not _FABRARY_DECKS_PATH.exists():
                return json.dumps({
                    "error": f"Fabrary deck database not found at {_FABRARY_DECKS_PATH}",
                    "side": side,
                }, indent=2)

            data = json.loads(_FABRARY_DECKS_PATH.read_text(encoding="utf-8"))
            raw_decks = list(data.get("decks", []))
            deck_entry = next((d for d in raw_decks if str(d.get("id", "")).lower() == deck_key), None)

            if not deck_entry:
                known_ids = [d.get("id", "") for d in raw_decks]
                return json.dumps({
                    "error": f"Deck {deck_key!r} not found in fabrary database",
                    "deck_id_from_url": deck_id,
                    "available_decks": known_ids[:10],
                    "side": side,
                }, indent=2)

            # Check format match
            deck_format = str(deck_entry.get("format", ""))
            if deck_format != normalized_format:
                return json.dumps({
                    "error": f"Deck is {deck_format!r} but requested format is {normalized_format!r}",
                    "deck_id": deck_key,
                    "deck_name": deck_entry.get("name", deck_key),
                    "side": side,
                }, indent=2)

            # Resolve the deck to card IDs using the environment's logic
            temp_env = registry.create("FleshAndBlood-Talishar-v0", render_mode=None, format=normalized_format)
            card_ids = temp_env._resolve_fabrary_deck(deck_entry)
            temp_env.close()

            if not card_ids:
                return json.dumps({
                    "error": f"Deck {deck_key!r} resolved to 0 legal cards for format {normalized_format!r}",
                    "deck_id": deck_key,
                    "deck_name": deck_entry.get("name", deck_key),
                    "side": side,
                }, indent=2)

            # Return as a deck option dict
            return json.dumps({
                "deck_id": deck_key,
                "deck_name": deck_entry.get("name", deck_key),
                "hero_id": str(deck_entry.get("hero_id", "")),
                "format": deck_format,
                "style": str(deck_entry.get("style", "balanced")),
                "deck_size": len(card_ids),
                "_card_ids": card_ids,
                "source": "fabrary",
                "source_url": fabrary_url,
                "description": deck_entry.get("description", ""),
                "side": side,
            }, indent=2)

        except Exception as exc:
            log.exception("fab_resolve_deck_from_url error")
            return json.dumps({
                "error": f"Failed to resolve deck: {exc}",
                "side": side,
                "deck_id_from_url": deck_id if match else None,
            }, indent=2)

    _FAB_CUSTOM_TOOLS_REGISTERED = True
    return 5
