"""
rlbridge Pokemon Red Environment - Gary's First Battle
====================================================

Wraps *Pokemon Red* (Game Boy, 1996) via the `PyBoy
<https://github.com/Baekalfen/PyBoy>`_ emulator.

Requirements
------------
1. **ROM** - Place your *Pokemon Red (International)* ROM at::

       ~/.rlbridge/pokemon_red/rom.gb

   or point the ``POKEMON_RED_ROM`` environment variable at the file.

2. **Save state** - A PyBoy ``.state`` file capturing the emulator
   state at the exact frame Gary's first trainer battle begins
   (D057 == 2).  The expected path is::

       ~/.rlbridge/pokemon_red/gary_battle.state

   To create this file interactively, call::

       from rlbridge.environments.predefined.pokemon_red import setup_save_state
       setup_save_state()

   This opens Pokemon Red with a window; play until Gary's battle
   starts and the state is saved automatically.

Environment variants
--------------------
``PokemonRed-GaryBattle-v0``
    Single episode = one battle against Gary (Blue) in Oak's Lab.
    The player's starter (Lv 5) fights Gary's counter-type starter (Lv 5).

    * **Actions** - button string: ``"a"``, ``"b"``, ``"up"``,
      ``"down"``, ``"left"``, ``"right"``, ``"start"``, ``"select"``
    * **Observations** - rich text description of the battle state
      (HP, moves, turn indicator, battle status)
    * **Rewards** - ``+1.0`` on victory, ``-1.0`` on defeat, ``0``
      per step
    * **Terminal** - when the battle ends (win **or** lose)
    * **Max steps** - 300 button presses (generous for text-heavy dialogue)
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Optional

from ..base import RLIPEnvironment, RLIPEnvironmentFactory
from ...protocol.messages import (
    DiscreteSpace,
    EnvironmentInfo,
    RenderResult,
    ResetResult,
    StepResult,
    TextSpace,
)

# ── File paths ────────────────────────────────────────────────────────────────

_STATE_ROOT = Path.home() / ".rlbridge" / "pokemon_red"

_DEFAULT_ROM_PATH   = _STATE_ROOT / "rom.gb"
_DEFAULT_STATE_PATH = _STATE_ROOT / "gary_battle.state"

# ── Pokemon Red (International) WRAM addresses ────────────────────────────────

# Battle control
_ADDR_BATTLE_TYPE     = 0xD057  # 0=none, 1=wild, 2=trainer
_ADDR_OPTIONS         = 0xD355  # bit7=anim off, low nibble=text speed
_ADDR_TEXT_SPEED_2    = 0xD358  # text delay override flags

# Player's party slot 1 (in-party data - reliable after battle starts)
_ADDR_P_SPECIES       = 0xD16B  # species ID (duplicate field)
_ADDR_P_HP_HI         = 0xD16C  # current HP high byte
_ADDR_P_HP_LO         = 0xD16D  # current HP low byte
_ADDR_P_LEVEL         = 0xD18C  # actual level
_ADDR_P_MAX_HP_HI     = 0xD18D  # max HP high byte
_ADDR_P_MAX_HP_LO     = 0xD18E  # max HP low byte
_ADDR_P_MOVE1         = 0xD173  # move slot 1 ID
_ADDR_P_MOVE2         = 0xD174
_ADDR_P_MOVE3         = 0xD175
_ADDR_P_MOVE4         = 0xD176
_ADDR_P_PP1           = 0xD188  # PP for move slot 1
_ADDR_P_PP2           = 0xD189
_ADDR_P_PP3           = 0xD18A
_ADDR_P_PP4           = 0xD18B

# Enemy's in-battle data (battle-mon scratch RAM - valid only mid-battle)
_ADDR_E_SPECIES       = 0xCFE5  # enemy species ID (in battle)
_ADDR_E_HP_HI         = 0xCFE6  # enemy current HP high byte
_ADDR_E_HP_LO         = 0xCFE7  # enemy current HP low byte
_ADDR_E_LEVEL         = 0xCFE8  # enemy level
_ADDR_E_MAX_HP_HI     = 0xCFF4  # enemy max HP high byte
_ADDR_E_MAX_HP_LO     = 0xCFF5  # enemy max HP low byte

# Turn indicator (valid during battle)
_ADDR_BATTLE_TURN     = 0xFFF3  # 0=player, 1=opponent

# Overworld state (Gen 1 WRAM; best-effort for text observations)
_ADDR_MAP_ID          = 0xD35E
_ADDR_PLAYER_Y        = 0xD361
_ADDR_PLAYER_X        = 0xD362

# ── Lookup tables ─────────────────────────────────────────────────────────────

# Pokemon Red internal species index → display name (International).
# The internal index ≠ National Dex number.
_SPECIES: dict[int, str] = {
    0x01: "Rhydon",       0x02: "Kangaskhan",   0x03: "Nidoran♂",
    0x04: "Clefairy",     0x05: "Spearow",       0x06: "Voltorb",
    0x07: "Nidoking",     0x08: "Slowbro",       0x09: "Ivysaur",
    0x0A: "Exeggutor",    0x0B: "Lickitung",     0x0C: "Exeggcute",
    0x0D: "Grimer",       0x0E: "Gengar",        0x0F: "Nidoran♀",
    0x10: "Nidoqueen",    0x11: "Cubone",        0x12: "Rhyhorn",
    0x13: "Lapras",       0x14: "Arcanine",      0x15: "Mew",
    0x16: "Gyarados",     0x17: "Shellder",      0x18: "Tentacool",
    0x19: "Gastly",       0x1A: "Scyther",       0x1B: "Staryu",
    0x1C: "Blastoise",    0x1D: "Pinsir",        0x1E: "Tangela",
    0x21: "Growlithe",    0x22: "Onix",          0x23: "Fearow",
    0x24: "Pidgey",       0x25: "Slowpoke",      0x26: "Kadabra",
    0x27: "Graveler",     0x28: "Chansey",       0x29: "Machoke",
    0x2B: "Drowzee",      0x2D: "Golem",         0x31: "Magmar",
    0x33: "Electabuzz",   0x34: "Magneton",      0x35: "Koffing",
    0x37: "Mankey",       0x38: "Seel",          0x39: "Diglett",
    0x3A: "Tauros",       0x3E: "Farfetch'd",    0x3F: "Venonat",
    0x40: "Dragonite",    0x43: "Doduo",         0x44: "Poliwag",
    0x45: "Jynx",         0x46: "Moltres",       0x47: "Articuno",
    0x48: "Zapdos",       0x49: "Ditto",         0x4A: "Meowth",
    0x4B: "Krabby",       0x4F: "Vulpix",        0x50: "Ninetales",
    0x52: "Wigglytuff",   0x53: "Clefable",      0x54: "Pikachu",
    0x55: "Raichu",       0x58: "Pidgeotto",     0x59: "Pidgeot",
    0x5C: "Rattata",      0x5D: "Raticate",      0x60: "Nidorina",
    0x61: "Nidorino",     0x62: "Geodude",       0x68: "Poliwhirl",
    0x69: "Poliwrath",    0x6A: "Weedle",        0x6B: "Kakuna",
    0x6C: "Beedrill",     0x6F: "Dodrio",        0x70: "Squirtle",
    0x71: "Wartortle",    0x74: "Horsea",        0x75: "Seadra",
    0x7B: "Ponyta",       0x7C: "Rapidash",      0x7D: "Rattata",
    0x80: "Magikarp",     0x8B: "Jigglypuff",    0x8E: "Psyduck",
    0x8F: "Golduck",      0x95: "Tentacruel",    0x97: "Snorlax",
    0x99: "Bulbasaur",    0x9A: "Venusaur",      0x9B: "Parasect",
    0x9C: "Paras",        0xA1: "Golbat",        0xA2: "Zubat",
    0xA5: "Machamp",      0xA9: "Haunter",       0xAA: "Abra",
    0xAC: "Alakazam",     0xAD: "Pidgey",        0xAF: "Machop",
    0xB0: "Charmander",   0xB1: "Charmeleon",    0xB2: "Charizard",
    0xB5: "Oddish",       0xB6: "Gloom",         0xB7: "Vileplume",
    0xB8: "Bellsprout",   0xB9: "Weepinbell",    0xBA: "Victreebel",
}

# Move ID → display name (selected common moves)
_MOVES: dict[int, str] = {
    0x01: "Pound",        0x02: "Karate Chop",   0x04: "Comet Punch",
    0x05: "Mega Punch",   0x0A: "Scratch",       0x0C: "Vice Grip",
    0x0D: "Guillotine",   0x0E: "Razor Wind",    0x14: "SonicBoom",
    0x17: "Horn Attack",  0x1F: "Stomp",         0x21: "Tackle",
    0x22: "Body Slam",    0x23: "Wrap",          0x25: "Jump Kick",
    0x26: "Rolling Kick", 0x27: "Tail Whip",     0x29: "Leer",
    0x2A: "Bite",         0x2B: "Growl",         0x2C: "Roar",
    # Note: Growl in-game ID is 0x2D per disassembly
    0x2D: "Growl",        0x2E: "Sing",          0x2F: "Supersonic",
    0x30: "SonicBoom",    0x33: "DoubleSlap",    0x38: "Fire Punch",
    0x39: "Ice Punch",    0x3A: "ThunderPunch",  0x40: "Drill Peck",
    0x49: "Ember",        0x4A: "Flamethrower",  0x4D: "Water Gun",
    0x4E: "Hydro Pump",   0x4F: "Surf",          0x55: "Bubble",
    0x5C: "Thunderbolt",  0x5D: "Thunder Wave",  0x5E: "Thunder",
    0x5F: "Rock Throw",   0x60: "Earthquake",    0x63: "Hyper Beam",
    0x64: "Hi Jump Kick", 0x69: "Blizzard",      0x6A: "Psybeam",
    0x6B: "BubbleBeam",   0x6D: "Aurora Beam",   0x73: "Swords Dance",
    0x77: "Toxic",        0x79: "Hypnosis",      0x7A: "Meditate",
    0x7B: "Agility",      0x7C: "Quick Attack",  0x7D: "Rage",
    0x7E: "Teleport",     0x7F: "Night Shade",   0x80: "Mimic",
    0x81: "Screech",      0x82: "Double Team",   0x85: "Rest",
    0x86: "String Shot",  0x91: "Harden",        0x92: "Minimize",
    0x95: "Smokescreen",  0x96: "Confuse Ray",   0x97: "Withdraw",
    0x98: "Defense Curl", 0x99: "Tackle",        0x9C: "Sleep Powder",
    0x9D: "Petal Dance",  0x9E: "String Shot",   0xA1: "Vine Whip",
    0xA3: "Leech Seed",   0xA5: "Poison Powder",
}

# Valid button strings accepted by PyBoy
_VALID_BUTTONS = frozenset({"a", "b", "up", "down", "left", "right", "start", "select"})

# Auto-start tuning: clear intro/dialogue after loading battle save state
# so the agent can begin from an actionable turn.
_AUTO_START_MAX_PRESSES = 180
_AUTO_START_TICKS_PER_PRESS = 6

_AUTO_NEW_GAME_MAX_STEPS = 1200
_AUTO_NEW_GAME_TICKS_PER_STEP = 8

# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_rom() -> Path:
    """Return path to Pokemon Red ROM, or raise FileNotFoundError."""
    env_path = os.environ.get("POKEMON_RED_ROM")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p
    if _DEFAULT_ROM_PATH.is_file():
        return _DEFAULT_ROM_PATH
    raise FileNotFoundError(
        "Pokemon Red ROM not found.  Place it at:\n"
        f"  {_DEFAULT_ROM_PATH}\n"
        "or set the POKEMON_RED_ROM environment variable."
    )


def _find_state(name: str = "gary_battle") -> Path:
    p = _STATE_ROOT / f"{name}.state"
    if not p.is_file():
        raise FileNotFoundError(
            f"Save state '{name}.state' not found at {p}.\n"
            "Create it by calling:\n"
            "  from rlbridge.environments.predefined.pokemon_red import setup_save_state\n"
            "  setup_save_state()"
        )
    return p


def _species_name(sid: int) -> str:
    return _SPECIES.get(sid, f"Pokemon(0x{sid:02X})")


def _move_name(mid: int) -> str:
    if mid == 0:
        return "---"
    return _MOVES.get(mid, f"Move(0x{mid:02X})")


def _normalize_button(text: str) -> Optional[str]:
    """Map a natural-language action string to a PyBoy button name, or None."""
    t = text.strip().lower()
    # Direct match first
    if t in _VALID_BUTTONS:
        return t
    # Keyword matching for common natural language variants
    if t in ("confirm", "ok", "select move", "attack", "fight", "yes"):
        return "a"
    if t in ("cancel", "back", "no"):
        return "b"
    if "up" in t:
        return "up"
    if "down" in t:
        return "down"
    if "left" in t:
        return "left"
    if "right" in t:
        return "right"
    if "start" in t:
        return "start"
    if t in ("menu", "select"):
        return "select"
    if " a" in t or t.endswith(" a"):
        return "a"
    if " b" in t or t.endswith(" b"):
        return "b"
    return None  # ignore unrecognised input (safe no-op)


# ── Environment ───────────────────────────────────────────────────────────────

class PokemonRedEnvironment(RLIPEnvironment):
    """
    rlbridge wrapper for a single Pokemon Red battle episode via PyBoy.

    Each episode loads a PyBoy save state and runs until the battle
    finishes (player or enemy faints).  The observation is a rich text
    description of the battle state derived from emulator RAM.

    Parameters
    ----------
    variant:
        Short name used to locate the save state file under
        ``~/.rlbridge/pokemon_red/<variant>.state``.
    max_episode_steps:
        Maximum number of button presses before the episode is
        truncated.
    frames_per_step:
        Number of emulator frames to advance per ``step()`` call.
        24 frames (≈ 0.4 in-game seconds) gives responsive control
        while allowing short animations to complete.
    """

    BUTTONS = ["a", "b", "up", "down", "left", "right", "start", "select"]

    def __init__(
        self,
        variant: str = "gary_battle",
        max_episode_steps: int = 300,
        frames_per_step: int = 24,
    ) -> None:
        self._variant = variant
        self._max_episode_steps = max_episode_steps
        self._frames_per_step = frames_per_step

        self._rom_path: Optional[Path] = None
        self._state_path: Optional[Path] = None
        self._pyboy: Any = None
        self._steps: int = 0
        self._initialized: bool = False
        self._lock = threading.Lock()

    # ── env_id ────────────────────────────────────────────────────────────────

    @property
    def env_id(self) -> str:
        variant_slug = self._variant.replace("_", "-").title()
        return f"PokemonRed-{variant_slug}-v0"

    # ── Life-cycle ────────────────────────────────────────────────────────────

    def _boot_emulator(self) -> None:
        from pyboy import PyBoy  # noqa: PLC0415 (deferred import)

        if self._pyboy is not None:
            try:
                self._pyboy.stop()
            except Exception:
                pass
            self._pyboy = None

        _STATE_ROOT.mkdir(parents=True, exist_ok=True)
        if self._rom_path is None:
            self._rom_path = _find_rom()
        if self._variant == "gary_battle" and self._state_path is None:
            self._state_path = _find_state(self._variant)

        self._pyboy = PyBoy(
            str(self._rom_path),
            window="null",
            sound_emulated=False,
            log_level="CRITICAL",
        )
        self._pyboy.set_emulation_speed(0)  # maximum speed for training

    def _boot_to_new_game_start(self) -> int:
        """
        Auto-progress from ROM boot through title/intro/new-game setup.

        Returns number of injected button presses.
        """
        def _press_button(name: str) -> None:
            """Send a short button pulse (press+release) when supported."""
            # Preferred path: explicit press/release events (more reliable on title/menu screens).
            try:
                from pyboy.utils import WindowEvent  # noqa: PLC0415

                evt_map = {
                    "a": ("PRESS_BUTTON_A", "RELEASE_BUTTON_A"),
                    "b": ("PRESS_BUTTON_B", "RELEASE_BUTTON_B"),
                    "up": ("PRESS_ARROW_UP", "RELEASE_ARROW_UP"),
                    "down": ("PRESS_ARROW_DOWN", "RELEASE_ARROW_DOWN"),
                    "left": ("PRESS_ARROW_LEFT", "RELEASE_ARROW_LEFT"),
                    "right": ("PRESS_ARROW_RIGHT", "RELEASE_ARROW_RIGHT"),
                    "start": ("PRESS_BUTTON_START", "RELEASE_BUTTON_START"),
                    "select": ("PRESS_BUTTON_SELECT", "RELEASE_BUTTON_SELECT"),
                }
                press_name, release_name = evt_map[name]
                press_evt = getattr(WindowEvent, press_name)
                release_evt = getattr(WindowEvent, release_name)
                self._pyboy.send_input(press_evt)
                self._pyboy.tick()
                self._pyboy.send_input(release_evt)
                return
            except Exception:
                pass

            # Fallback path: convenience helper (works on older PyBoy APIs).
            self._pyboy.button(name)

        presses = 0

        def _tick_after_press() -> None:
            for _ in range(_AUTO_NEW_GAME_TICKS_PER_STEP):
                self._pyboy.tick()

        def _is_overworld_playable() -> bool:
            return self._u8(_ADDR_BATTLE_TYPE) == 0 and self._u8(_ADDR_MAP_ID) != 0

        def _do_press(name: str) -> bool:
            nonlocal presses
            _press_button(name)
            _tick_after_press()
            presses += 1
            return _is_overworld_playable()

        def _do_press_many(name: str, count: int) -> bool:
            for _ in range(count):
                if _do_press(name):
                    return True
            return False

        def _choose_preset_name(down_presses: int) -> bool:
            # Name menu cursor starts at NEW NAME; move to requested preset.
            if _do_press_many("up", 4):
                return True
            if _do_press_many("down", down_presses):
                return True
            if _do_press("a"):
                return True
            return False

        # 1) Clear title and intro text quickly.
        if _do_press_many("start", 80):
            return presses
        if _do_press_many("a", 220):
            return presses
        if _do_press_many("start", 40):
            return presses
        if _do_press_many("a", 80):
            return presses

        # 2) Deterministically pick names requested by user:
        #    player = RED (preset index 1), rival = GARY (preset index 2).
        for _ in range(6):
            if _choose_preset_name(down_presses=1):
                return presses
            if _do_press_many("a", 24):
                return presses
            if _choose_preset_name(down_presses=2):
                return presses
            if _do_press_many("a", 30):
                return presses

        # 3) Finish Oak intro and hand off when overworld control is available.
        while presses < _AUTO_NEW_GAME_MAX_STEPS:
            if _do_press("a"):
                return presses
            if presses % 20 == 0 and _do_press("start"):
                return presses

        return presses

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> ResetResult:
        with self._lock:
            self._boot_emulator()

            auto_start_presses = 0

            if self._variant == "gary_battle":
                # Restore emulator to the saved battle state
                with open(self._state_path, "rb") as fh:
                    self._pyboy.load_state(fh)
            else:
                # Start from cold boot and auto-create/enter a new game.
                auto_start_presses = self._boot_to_new_game_start()
                if not (self._u8(_ADDR_BATTLE_TYPE) == 0 and self._u8(_ADDR_MAP_ID) != 0):
                    bt = self._u8(_ADDR_BATTLE_TYPE)
                    mid = self._u8(_ADDR_MAP_ID)
                    raise RuntimeError(
                        "Pokemon Red auto-start could not reach playable overworld control. "
                        "Try increasing _AUTO_NEW_GAME_MAX_STEPS or verify ROM/version compatibility. "
                        f"Auto-start presses attempted: {auto_start_presses}. "
                        f"Observed battle_type={bt}, map_id={mid}."
                    )

            # Force fastest text speed and disable battle animations so the
            # agent doesn't have to wait through long display sequences.
            # D355: bit7=1 → animations off; low nibble 1 → fast text
            self._pyboy.memory[_ADDR_OPTIONS] = 0x81
            # D358: bits 0+1 set → no per-character delay override
            self._pyboy.memory[_ADDR_TEXT_SPEED_2] = 0x03

            # Advance a few frames to let the state stabilise
            for _ in range(10):
                self._pyboy.tick()

            # For battle variant, auto-progress opening dialogue so the agent
            # starts from an actionable battle state.
            if self._variant == "gary_battle":
                auto_start_presses += self._advance_to_playable_start()

            self._steps = 0
            self._initialized = True
            return ResetResult(
                observation=self._build_obs(),
                info={
                    **self._build_info(),
                    "auto_start_presses": auto_start_presses,
                    "playable_start_reached": (
                        self._is_playable_turn() if self._variant == "gary_battle"
                        else (self._u8(_ADDR_BATTLE_TYPE) == 0)
                    ),
                },
            )

    def step(self, action: Any) -> StepResult:
        with self._lock:
            if not self._initialized:
                raise RuntimeError("Call reset() before step().")

            cmd = _normalize_button(str(action))
            if cmd:
                self._pyboy.button(cmd)

            for _ in range(self._frames_per_step):
                self._pyboy.tick()

            self._steps += 1

            reward, terminated = self._check_battle_outcome()
            truncated = self._steps >= self._max_episode_steps

            return StepResult(
                observation=self._build_obs(),
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=self._build_info(),
            )

    def close(self) -> None:
        with self._lock:
            if self._pyboy is not None:
                try:
                    self._pyboy.stop()
                except Exception:
                    pass
                self._pyboy = None
            self._initialized = False

    # ── Memory helpers ────────────────────────────────────────────────────────

    def _u8(self, addr: int) -> int:
        return self._pyboy.memory[addr]

    def _u16(self, hi_addr: int) -> int:
        """Read a big-endian 16-bit value from two consecutive bytes."""
        return (self._pyboy.memory[hi_addr] << 8) | self._pyboy.memory[hi_addr + 1]

    def _is_playable_turn(self) -> bool:
        """True when trainer battle is active and the player can choose an action."""
        if self._u8(_ADDR_BATTLE_TYPE) != 2:
            return False
        if self._u8(_ADDR_BATTLE_TURN) != 0:
            return False
        return any(
            self._u8(addr) != 0
            for addr in (_ADDR_P_MOVE1, _ADDR_P_MOVE2, _ADDR_P_MOVE3, _ADDR_P_MOVE4)
        )

    def _advance_to_playable_start(self) -> int:
        """
        Auto-clear intro/dialogue so reset lands on the first actionable turn.

        Returns the number of injected button presses used.
        """
        presses = 0
        if self._is_playable_turn():
            return presses

        while presses < _AUTO_START_MAX_PRESSES and not self._is_playable_turn():
            self._pyboy.button("a")
            for _ in range(_AUTO_START_TICKS_PER_PRESS):
                self._pyboy.tick()
            presses += 1

        return presses

    # ── Battle outcome ────────────────────────────────────────────────────────

    def _check_battle_outcome(self) -> tuple[float, bool]:
        """Return (reward, terminated) based on current memory state."""
        if self._variant != "gary_battle":
            # Overworld free-play variant: no terminal battle reward.
            return 0.0, False

        battle_type = self._u8(_ADDR_BATTLE_TYPE)

        if battle_type == 0:
            # Battle has ended - determine winner by player HP
            player_hp = self._u16(_ADDR_P_HP_HI)
            reward = 1.0 if player_hp > 0 else -1.0
            return reward, True

        # Mid-battle: also catch the instant the player's Pokemon faints
        player_hp = self._u16(_ADDR_P_HP_HI)
        if player_hp == 0:
            return -1.0, True

        return 0.0, False

    # ── Observation ──────────────────────────────────────────────────────────

    def _build_obs(self) -> str:
        battle_type = self._u8(_ADDR_BATTLE_TYPE)

        if self._variant != "gary_battle":
            map_id = self._u8(_ADDR_MAP_ID)
            x = self._u8(_ADDR_PLAYER_X)
            y = self._u8(_ADDR_PLAYER_Y)
            status = "In battle" if battle_type != 0 else "Overworld"
            return "\n".join([
                "=== Pokemon Red Overworld ===",
                f"Status: {status}",
                f"Map ID: {map_id}",
                f"Position: x={x}, y={y}",
                "",
                "Buttons: a  b  up  down  left  right  start  select",
                "(Use d-pad to move; A/B for dialogue/menus.)",
            ])

        player_species = self._u8(_ADDR_P_SPECIES)
        player_name    = _species_name(player_species)
        player_hp      = self._u16(_ADDR_P_HP_HI)
        player_max_hp  = self._u16(_ADDR_P_MAX_HP_HI)
        player_level   = self._u8(_ADDR_P_LEVEL)

        enemy_species  = self._u8(_ADDR_E_SPECIES)
        enemy_name     = _species_name(enemy_species)
        enemy_hp       = self._u16(_ADDR_E_HP_HI)
        enemy_max_hp   = self._u16(_ADDR_E_MAX_HP_HI)
        enemy_level    = self._u8(_ADDR_E_LEVEL)

        # Moves available to the player
        moves = []
        for move_addr, pp_addr in [
            (_ADDR_P_MOVE1, _ADDR_P_PP1),
            (_ADDR_P_MOVE2, _ADDR_P_PP2),
            (_ADDR_P_MOVE3, _ADDR_P_PP3),
            (_ADDR_P_MOVE4, _ADDR_P_PP4),
        ]:
            mid = self._u8(move_addr)
            if mid != 0:
                pp  = self._u8(pp_addr)
                moves.append(f"{_move_name(mid)} (PP {pp})")

        turn_byte = self._u8(_ADDR_BATTLE_TURN)
        turn_txt  = "Player's turn" if turn_byte == 0 else "Opponent's turn"

        if battle_type == 0:
            status = "Battle has ended."
        elif battle_type == 2:
            status = f"Trainer battle active.  {turn_txt}."
        else:
            status = "Battle active."

        lines = [
            "=== Pokemon Battle ===",
            f"Status: {status}",
            "",
            f"Gary's {enemy_name}  (Lv.{enemy_level})  HP: {enemy_hp}/{enemy_max_hp}",
            f"  {'█' * min(20, int(20 * enemy_hp / max(enemy_max_hp, 1)))}{'░' * (20 - min(20, int(20 * enemy_hp / max(enemy_max_hp, 1))))}",
            "",
            f"Your {player_name}  (Lv.{player_level})  HP: {player_hp}/{player_max_hp}",
            f"  {'█' * min(20, int(20 * player_hp / max(player_max_hp, 1)))}{'░' * (20 - min(20, int(20 * player_hp / max(player_max_hp, 1))))}",
        ]
        if moves:
            lines += ["", "Your moves:"] + [f"  {m}" for m in moves]
        lines += [
            "",
            "Buttons: a  b  up  down  left  right  start  select",
            "(Press 'a' to confirm, 'b' to cancel, d-pad to navigate menus.)",
        ]
        return "\n".join(lines)

    def _build_info(self) -> dict[str, Any]:
        battle_type  = self._u8(_ADDR_BATTLE_TYPE)
        player_hp    = self._u16(_ADDR_P_HP_HI)
        player_maxhp = self._u16(_ADDR_P_MAX_HP_HI)
        enemy_hp     = self._u16(_ADDR_E_HP_HI)
        enemy_maxhp  = self._u16(_ADDR_E_MAX_HP_HI)
        return {
            "variant":      self._variant,
            "battle_type":   battle_type,
            "map_id":        self._u8(_ADDR_MAP_ID),
            "player_x":      self._u8(_ADDR_PLAYER_X),
            "player_y":      self._u8(_ADDR_PLAYER_Y),
            "player_hp":     player_hp,
            "player_max_hp": player_maxhp,
            "player_species": _species_name(self._u8(_ADDR_P_SPECIES)),
            "enemy_hp":      enemy_hp,
            "enemy_max_hp":  enemy_maxhp,
            "enemy_species": _species_name(self._u8(_ADDR_E_SPECIES)),
            "step":          self._steps,
            "buttons":       list(self.BUTTONS),
        }

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def observation_space(self) -> TextSpace:
        return TextSpace()

    @property
    def action_space(self) -> TextSpace:
        return TextSpace()

    @property
    def reward_range(self) -> tuple[float, float]:
        return (-1.0, 1.0)

    def render(self) -> RenderResult:
        if self._pyboy is None:
            return RenderResult(mode="ansi", text="[not initialised]")
        # Return RGB screen pixels encoded as ANSI text description
        return RenderResult(mode="ansi", text=self._build_obs())

    def sample_action(self) -> str:
        """
        Return a heuristic random action weighted toward 'a' (advance
        dialogue) since most frames in a Pokemon battle require pressing A.
        """
        import random
        weights = [0.55, 0.10, 0.10, 0.10, 0.05, 0.05, 0.025, 0.025]
        return random.choices(self.BUTTONS, weights=weights, k=1)[0]


# ── Factory ───────────────────────────────────────────────────────────────────

_VARIANT_META: dict[str, tuple[str, list[str], int, float]] = {
    "gary_battle": (
        "Pokemon Red: defeat Gary (Blue) in his first trainer battle in "
        "Oak's Lab.  Player and rival each have one Level-5 starter Pokemon.  "
        "Reward +1 for winning, -1 for losing.",
        ["pokemon-red", "game-boy", "rpg", "battle", "text"],
        300,
        1.0,
    ),
    "overworld_start": (
        "Pokemon Red overworld free-play from a fresh boot each episode. "
        "The environment auto-advances launch -> intro -> new game start so "
        "the agent can control movement between battles.",
        ["pokemon-red", "game-boy", "rpg", "overworld", "text"],
        1200,
        0.0,
    ),
}


class PokemonRedFactory(RLIPEnvironmentFactory):
    """Factory for a Pokemon Red battle-episode environment."""

    def __init__(self, variant: str = "gary_battle") -> None:
        if variant not in _VARIANT_META:
            raise ValueError(
                f"Unknown Pokemon Red variant {variant!r}.  "
                f"Available: {list(_VARIANT_META)}"
            )
        self._variant = variant
        desc, tags, max_steps, threshold = _VARIANT_META[variant]
        self._desc       = desc
        self._tags       = tags
        self._max_steps  = max_steps
        self._threshold  = threshold

    @property
    def env_info(self) -> EnvironmentInfo:
        slug = self._variant.replace("_", "-").title()
        return EnvironmentInfo(
            env_id=f"PokemonRed-{slug}-v0",
            description=self._desc,
            tags=self._tags,
            max_episode_steps=self._max_steps,
            reward_threshold=self._threshold,
            namespace="pokemon-red",
            render_modes=["ansi"],
        )

    def create(
        self,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> PokemonRedEnvironment:
        return PokemonRedEnvironment(
            variant=self._variant,
            max_episode_steps=self._max_steps,
        )


# ── Pre-built singletons ──────────────────────────────────────────────────────

POKEMON_RED_GARY_BATTLE_V0 = PokemonRedFactory("gary_battle")
POKEMON_RED_OVERWORLD_START_V0 = PokemonRedFactory("overworld_start")

ALL_POKEMON_RED_FACTORIES: list[PokemonRedFactory] = [
    POKEMON_RED_GARY_BATTLE_V0,
    POKEMON_RED_OVERWORLD_START_V0,
]

# ── Interactive setup helper ──────────────────────────────────────────────────

def setup_save_state(
    rom_path: Optional[Path | str] = None,
    output_path: Optional[Path | str] = None,
) -> Path:
    """
    Open Pokemon Red in a window and auto-save a state when Gary's
    first trainer battle starts (``wBattleType == 2``).

    Parameters
    ----------
    rom_path:
        Path to the Pokemon Red ROM.  Defaults to
        ``~/.rlbridge/pokemon_red/rom.gb`` or the ``POKEMON_RED_ROM``
        environment variable.
    output_path:
        Where to write the ``.state`` file.  Defaults to
        ``~/.rlbridge/pokemon_red/gary_battle.state``.

    Returns
    -------
    Path
        The path where the state was saved.
    """
    from pyboy import PyBoy  # noqa: PLC0415

    rom = Path(rom_path) if rom_path else _find_rom()
    out = Path(output_path) if output_path else _DEFAULT_STATE_PATH

    pyboy = PyBoy(str(rom), window="SDL2", log_level="WARNING")
    pyboy.set_emulation_speed(1)  # real-time so the user can play

    print("=" * 60)
    print("Pokemon Red - Gary Battle State Setup")
    print("=" * 60)
    print("Play until Gary says 'I'll take you on!' and the")
    print("battle screen appears.  The state is saved automatically.")
    print("(Close the window to abort.)")
    print("=" * 60)

    try:
        while True:
            if not pyboy.tick():
                print("Window closed - aborting setup.")
                return out

            battle_type = pyboy.memory[_ADDR_BATTLE_TYPE]
            if battle_type == 2:
                # Advance a few frames so the battle screen is fully loaded
                for _ in range(60):
                    pyboy.tick()
                out.parent.mkdir(parents=True, exist_ok=True)
                with open(out, "wb") as fh:
                    pyboy.save_state(fh)
                print(f"\nTrainer battle detected - state saved to:\n  {out}")
                break
    finally:
        pyboy.stop()

    return out
