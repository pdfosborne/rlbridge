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
    TextSpace,
)
from ..base import RLIPEnvironment, RLIPEnvironmentFactory


_FAB_DB_DIR = Path(__file__).with_name("card_db") / "flesh_and_blood"
_CARDS_PATH = _FAB_DB_DIR / "cards.json"
_HEROES_PATH = _FAB_DB_DIR / "heroes.json"


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

        self._players: list[PlayerState] = []
        self._turn = 0
        self._active_player = 0
        self._phase = "action"
        self._pending_combat: Optional[CombatState] = None
        self._initialized = False
        self._last_event = ""
        self._render_mode = render_mode
        self._self_play = self_play

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
        self._max_turns = int(opts.get("max_turns", self._max_turns))
        self._deck_size = int(opts.get("deck_size", self._deck_size))

        self._players = [
            self._new_player(self._heroes[self._agent_hero_id], hero_slot=0),
            self._new_player(self._heroes[self._opponent_hero_id], hero_slot=1),
        ]

        self._turn = 1
        self._active_player = 0
        self._phase = "action"
        self._pending_combat = None
        self._last_event = "Game start"

        self._draw_up(0)
        self._draw_up(1)
        self._start_turn(0)

        self._initialized = True
        return ResetResult(
            observation=self._observation(),
            info={
                "legal_actions": self._legal_actions(),
                "agent_hero": self._players[0].hero.name,
                "opponent_hero": self._players[1].hero.name,
            },
        )

    def step(self, action: Any) -> StepResult:
        if not self._initialized:
            raise RuntimeError("Call reset() before step().")

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
        if not self._initialized or len(self._players) < 2:
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

        _text(24, 20, "Flesh and Blood (Talishar-inspired)", font_title, fg)
        _text(
            24,
            58,
            f"Turn {obs['turn']} | Phase: {obs['phase']} | Active: P{obs['active_player']}",
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

    def _new_player(self, hero: Hero, hero_slot: int) -> PlayerState:
        deck = self._build_deck(hero_slot)
        self._rng.shuffle(deck)
        return PlayerState(
            hero=hero,
            life=hero.life,
            resources=0,
            action_points=0,
            deck=deck,
            hand=[],
            discard=[],
        )

    def _build_deck(self, hero_slot: int) -> list[str]:
        warrior_pool = [
            "warrior_savage_feast_red",
            "warrior_snatch_red",
            "warrior_hit_and_run_red",
            "generic_raging_onslaught_red",
            "generic_wounding_blow_red",
            "generic_head_jab_red",
            "generic_scar_for_a_scar_red",
            "generic_red_pitch_card",
            "generic_yellow_pitch_card",
            "generic_blue_pitch_card",
            "warrior_sink_below_red",
            "warrior_fate_foreseen_red",
            "warrior_unmovable_red",
        ]
        brute_pool = [
            "brute_wild_ride_red",
            "brute_pack_hunt_red",
            "brute_wrecker_romp_red",
            "generic_raging_onslaught_red",
            "generic_wounding_blow_red",
            "generic_head_jab_red",
            "generic_scar_for_a_scar_red",
            "generic_red_pitch_card",
            "generic_yellow_pitch_card",
            "generic_blue_pitch_card",
            "warrior_sink_below_red",
            "warrior_fate_foreseen_red",
            "warrior_unmovable_red",
        ]

        pool = warrior_pool if hero_slot == 0 else brute_pool
        deck: list[str] = []
        while len(deck) < self._deck_size:
            deck.extend(pool)
        return deck[: self._deck_size]

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
            "phase": self._phase,
            "active_player": self._active_player,
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
        p_agent, p_opp = self._estimate_win_probabilities(obs)
        lines = [
            "Flesh and Blood (Talishar-inspired)",
            f"Turn: {obs['turn']} | Phase: {obs['phase']} | Active: P{obs['active_player']}",
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
        self_play: bool = False,
    ) -> None:
        self._env_id = env_id
        self._agent_hero_id = agent_hero_id
        self._opponent_hero_id = opponent_hero_id
        self._max_turns = max_turns
        self._deck_size = deck_size
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
                *(["self-play"] if self._self_play else []),
            ],
            namespace="flesh_and_blood",
            render_modes=["ansi", "rgb_array"],
            max_episode_steps=self._max_turns,
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
            self_play=bool(kwargs.get("self_play", self._self_play)),
            render_mode=render_mode,
        )


FLESH_AND_BLOOD_TALISHAR_V0 = FleshAndBloodFactory("FleshAndBlood-Talishar-v0")
FLESH_AND_BLOOD_SELFPLAY_V0 = FleshAndBloodFactory(
    "FleshAndBlood-SelfPlay-v0",
    self_play=True,
)
ALL_FAB_FACTORIES: list[FleshAndBloodFactory] = [
    FLESH_AND_BLOOD_TALISHAR_V0,
    FLESH_AND_BLOOD_SELFPLAY_V0,
]
