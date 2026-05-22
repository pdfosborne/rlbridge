"""
rlbridge TextWorld Environments
============================
Wraps `TextWorld <https://github.com/microsoft/TextWorld>`_ text-adventure
games as rlbridge environments.

TextWorld generates fully-specified interactive-fiction (IF) games and
exposes them as text-in/text-out RL environments.  The observation at
every step is a narrative text string (room description + feedback from
the last command); the agent selects from a dynamic set of admissible
natural-language commands.

Game variants shipped with rlbridge
---------------------------------
``TextWorld-Take-v0``
    Single room.  One object on the floor.  Quest: *take <object>*.
    The simplest possible TextWorld game - good for smoke-testing pipelines.

``TextWorld-Navigate-v0``
    Two rooms connected by a door (locked or open).  One object in the
    second room.  Quest: navigate there and take it.

``TextWorld-TreasureHunt-v0``
    Three rooms in a line.  A key in room 2 unlocks a chest in room 3
    that contains the treasure.  Quest: take the treasure.

Game files are compiled once to ``~/.rlbridge/textworld/<variant>/game.z8``
and reused on subsequent calls; delete the directory to force recompilation.

Language translation
--------------------
TextWorld observations are already natural language, so the translator
simply normalises and returns the ``description`` + ``inventory`` fields
from the info dict.  The raw obs string (game output) is used as a
fallback when info is not available.

Usage
-----
::

    from rlbridge.environments.registry import registry

    factory = registry.get("TextWorld-Take-v0")
    env = factory.create()
    result = env.reset(seed=0)
    print(result.observation)          # "You are in the Kitchen. There is..."
    print(result.info["admissible_commands"])
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Optional

from ..base import rlbridgeEnvironment, rlbridgeEnvironmentFactory
from ...protocol.messages import (
    DiscreteSpace,
    EnvironmentInfo,
    RenderResult,
    ResetResult,
    StepResult,
    SuggestedHyperparameters,
    TextSpace,
)


# ── Cache directory ───────────────────────────────────────────────────────────

_CACHE_ROOT = Path.home() / ".rlbridge" / "textworld"

_REQUEST_INFOS_KWARGS: dict[str, bool] = dict(
    description=True,
    inventory=True,
    admissible_commands=True,
    score=True,
    won=True,
    lost=True,
    max_score=True,
)

_COMPILE_LOCK = threading.Lock()


# ── Game builders (pure GameMaker, no challenge modules needed) ───────────────

def _build_take_game(game_dir: Path) -> Path:
    """One room; one object on the floor; quest = take it."""
    import textworld
    import textworld.generator as M

    maker = M.GameMaker()
    room = maker.new_room("Kitchen")
    maker.set_player(room)
    obj = maker.new(type="o", name="old coin")
    room.add(obj)
    maker.set_quest_from_commands(["take old coin"])
    maker.build()

    opts = textworld.GameOptions()
    opts.path = str(game_dir)
    return Path(maker.compile(str(game_dir / "game")))


def _build_navigate_game(game_dir: Path) -> Path:
    """Two rooms; object in second room; quest = go there and take it."""
    import textworld
    import textworld.generator as M

    maker = M.GameMaker()
    bedroom = maker.new_room("Bedroom")
    kitchen = maker.new_room("Kitchen")
    maker.set_player(bedroom)
    maker.connect(bedroom.east, kitchen.west)

    obj = maker.new(type="o", name="rusty key")
    kitchen.add(obj)
    maker.set_quest_from_commands(["go east", "take rusty key"])
    maker.build()

    opts = textworld.GameOptions()
    opts.path = str(game_dir)
    return Path(maker.compile(str(game_dir / "game")))


def _build_treasure_hunt_game(game_dir: Path) -> Path:
    """Three rooms in a line; golden key in room 2, diamond in room 3."""
    import textworld
    import textworld.generator as M

    maker = M.GameMaker()
    r1 = maker.new_room("Entrance Hall")
    r2 = maker.new_room("Store Room")
    r3 = maker.new_room("Vault")
    maker.set_player(r1)
    maker.connect(r1.east, r2.west)
    maker.connect(r2.east, r3.west)

    key = maker.new(type="o", name="golden key")
    r2.add(key)
    treasure = maker.new(type="o", name="diamond")
    r3.add(treasure)

    maker.set_quest_from_commands([
        "go east",
        "take golden key",
        "go east",
        "take diamond",
    ])
    maker.build()

    opts = textworld.GameOptions()
    opts.path = str(game_dir)
    return Path(maker.compile(str(game_dir / "game")))


def _build_coin_collector_easy_game(game_dir: Path) -> Path:
    """Coin Collector level 1: single room, take the coin."""
    import textworld
    import textworld.generator
    import textworld.challenges.tw_coin_collector.coin_collector as cc

    opts = textworld.GameOptions()
    opts.seeds = 42
    opts.path = str(game_dir / "game")
    game = cc.make({"level": 1}, opts)
    game_file = textworld.generator.compile_game(game, opts)
    return Path(game_file)


def _build_coin_collector_medium_game(game_dir: Path) -> Path:
    """Coin Collector level 5: multi-room, navigate to find and take the coin."""
    import textworld
    import textworld.generator
    import textworld.challenges.tw_coin_collector.coin_collector as cc

    opts = textworld.GameOptions()
    opts.seeds = 42
    opts.path = str(game_dir / "game")
    game = cc.make({"level": 5}, opts)
    game_file = textworld.generator.compile_game(game, opts)
    return Path(game_file)


def _build_cooking_easy_game(game_dir: Path) -> Path:
    """
    Cooking Easy: single-room kitchen, one-ingredient recipe.
    Agent must read the cookbook, gather the ingredient, process it
    (chop/slice/dice as required), and prepare the meal.
    """
    import textworld
    import textworld.generator
    import textworld.challenges.tw_cooking.cooking as ck

    settings = {
        "recipe": 1, "take": 1, "go": 1,
        "open": 0, "cook": 0, "cut": 0, "drop": 0,
        "recipe_seed": 42, "split": "train",
    }
    opts = textworld.GameOptions()
    opts.seeds = 42
    opts.path = str(game_dir / "game")
    game = ck.make(settings, opts)
    game_file = textworld.generator.compile_game(game, opts)
    return Path(game_file)


def _build_cooking_medium_game(game_dir: Path) -> Path:
    """
    Cooking Medium: 6-room map, two-ingredient recipe.
    Agent must explore multiple rooms to find both ingredients,
    process them, and prepare the meal.
    """
    import textworld
    import textworld.generator
    import textworld.challenges.tw_cooking.cooking as ck

    settings = {
        "recipe": 2, "take": 2, "go": 6,
        "open": 0, "cook": 0, "cut": 0, "drop": 0,
        "recipe_seed": 42, "split": "train",
    }
    opts = textworld.GameOptions()
    opts.seeds = 42
    opts.path = str(game_dir / "game")
    game = ck.make(settings, opts)
    game_file = textworld.generator.compile_game(game, opts)
    return Path(game_file)


_BUILDERS = {
    "TextWorld-Take-v0":                   _build_take_game,
    "TextWorld-Navigate-v0":               _build_navigate_game,
    "TextWorld-TreasureHunt-v0":           _build_treasure_hunt_game,
    "TextWorld-CoinCollector-Easy-v0":     _build_coin_collector_easy_game,
    "TextWorld-CoinCollector-Medium-v0":   _build_coin_collector_medium_game,
    "TextWorld-Cooking-Easy-v0":           _build_cooking_easy_game,
    "TextWorld-Cooking-Medium-v0":         _build_cooking_medium_game,
}


def _get_or_compile_game(env_id: str) -> Path:
    """
    Return (and lazily compile if missing) the ``.z8`` game file for
    *env_id*.  Thread-safe; only compiles once per variant per process.
    """
    game_dir = _CACHE_ROOT / env_id
    game_file = game_dir / "game.z8"
    if game_file.exists():
        return game_file

    builder = _BUILDERS[env_id]
    with _COMPILE_LOCK:
        # Double-check inside the lock
        if game_file.exists():
            return game_file
        game_dir.mkdir(parents=True, exist_ok=True)
        compiled = builder(game_dir)
        # Compiled file may have a different suffix (.z8, .ulx, etc.)
        # Normalise to game.z8 via symlink/copy so callers always see the same path.
        if str(compiled) != str(game_file):
            compiled.rename(game_file)
    return game_file


# ── Environment wrapper ───────────────────────────────────────────────────────

class TextWorldEnvironment(rlbridgeEnvironment):
    """
    rlbridge wrapper around a compiled TextWorld game file.

    Observations are the raw text output from the interpreter.
    Actions are natural-language command strings drawn from
    ``info["admissible_commands"]``.

    Parameters
    ----------
    env_id:
        Registered rlbridge environment ID, e.g. ``"TextWorld-Take-v0"``.
    game_file:
        Path to the compiled ``.z8`` (or ``.ulx``) game file.
    max_episode_steps:
        Maximum steps per episode before truncation.
    """

    def __init__(
        self,
        env_id: str,
        game_file: Path,
        max_episode_steps: int = 50,
        known_max_score: int = 1,
    ) -> None:
        import textworld
        import textworld.gym as tw_gym

        self._env_id = env_id
        self._game_file = game_file
        self._max_episode_steps = max_episode_steps
        self._known_max_score = known_max_score
        self._lock = threading.Lock()
        self._initialized = False
        self._current_info: dict[str, Any] = {}

        tw_gym_id = tw_gym.register_game(
            str(game_file),
            max_episode_steps=max_episode_steps,
            request_infos=textworld.EnvInfos(**_REQUEST_INFOS_KWARGS),
        )
        self._tw_env = tw_gym.make(tw_gym_id)

    # ── env_id property (used by translator, protocol, etc.) ─────────────────

    @property
    def env_id(self) -> str:
        return self._env_id

    # ── Life-cycle ────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> ResetResult:
        with self._lock:
            # TextWorld gym doesn't use seed for reset (game is deterministic)
            obs, info = self._tw_env.reset()
            self._initialized = True
            self._current_info = info
            return ResetResult(
                observation=obs,
                info={
                    "admissible_commands": info.get("admissible_commands", []),
                    "description":         info.get("description", ""),
                    "inventory":           info.get("inventory", ""),
                    "score":               info.get("score", 0),
                    "max_score":           info.get("max_score", 1),
                },
            )

    def step(self, action: Any) -> StepResult:
        with self._lock:
            if not self._initialized:
                raise RuntimeError("Call reset() before step()")

            command = str(action)
            result = self._tw_env.step(command)
            # TextWorld gym (old API) returns (obs, reward, done, info)
            obs, reward, done, info = result
            self._current_info = info
            won = bool(info.get("won", False))
            lost = bool(info.get("lost", False))

            return StepResult(
                observation=obs,
                reward=float(reward),
                terminated=bool(done),
                truncated=False,
                info={
                    "admissible_commands": info.get("admissible_commands", []),
                    "description":         info.get("description", ""),
                    "inventory":           info.get("inventory", ""),
                    "score":               info.get("score", 0),
                    "max_score":           info.get("max_score", 1),
                    "won":                 won,
                    "lost":                lost,
                },
            )

    def close(self) -> None:
        try:
            self._tw_env.close()
        except Exception:
            pass
        self._initialized = False

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def observation_space(self) -> TextSpace:
        return TextSpace()

    @property
    def action_space(self) -> TextSpace:
        return TextSpace()

    @property
    def reward_range(self) -> tuple[float, float]:
        """Reward range inferred from the game's max_score (set at construction)."""
        return (0.0, float(self._known_max_score))

    def render(self) -> RenderResult:
        info = self._current_info
        lines: list[str] = []
        if info.get("description"):
            lines.append(info["description"])
        if info.get("inventory"):
            lines.append(f"\nInventory: {info['inventory']}")
        cmds = info.get("admissible_commands", [])
        if cmds:
            lines.append("\nAvailable commands:\n" + "\n".join(f"  > {c}" for c in cmds))
        score = info.get("score", "?")
        max_score = info.get("max_score", "?")
        lines.append(f"\nScore: {score}/{max_score}")
        return RenderResult(mode="ansi", text="\n".join(lines) if lines else "")

    def sample_action(self) -> Any:
        """Return a random admissible command, or 'look' as fallback."""
        import random
        cmds = self._current_info.get("admissible_commands", ["look"])
        return random.choice(cmds) if cmds else "look"


# ── Factory ───────────────────────────────────────────────────────────────────

# (description, tags, max_steps, reward_threshold, known_max_score)
_VARIANT_META: dict[str, tuple[str, list[str], int, float | None, int]] = {
    "TextWorld-Take-v0": (
        "Single-room TextWorld game: take the object to win.  "
        "The simplest possible text-adventure RL task.",
        ["textworld", "text-adventure", "simple", "discrete"],
        30, 1.0, 1,
    ),
    "TextWorld-Navigate-v0": (
        "Two-room TextWorld game: navigate to the second room and take "
        "the object there.  Tests basic room-transition reasoning.",
        ["textworld", "text-adventure", "navigation", "discrete"],
        50, 1.0, 1,
    ),
    "TextWorld-TreasureHunt-v0": (
        "Three-room TextWorld game: navigate east through two rooms, "
        "picking up a golden key in the Store Room and a diamond in the "
        "Vault.  Tests multi-step planning and sequential object collection.",
        ["textworld", "text-adventure", "multi-step", "discrete"],
        100, 1.0, 1,
    ),
    "TextWorld-CoinCollector-Easy-v0": (
        "TextWorld Coin Collector level 1: single room, find and take the coin.  "
        "Official TextWorld challenge - the simplest coin collector variant.",
        ["textworld", "text-adventure", "coin-collector", "simple", "discrete"],
        30, 1.0, 1,
    ),
    "TextWorld-CoinCollector-Medium-v0": (
        "TextWorld Coin Collector level 5: multi-room map, navigate to find "
        "and take the coin.  Tests room navigation under partial observability.",
        ["textworld", "text-adventure", "coin-collector", "navigation", "discrete"],
        100, 1.0, 1,
    ),
    "TextWorld-Cooking-Easy-v0": (
        "TextWorld Cooking challenge (easy): single kitchen, one-ingredient recipe.  "
        "Read the cookbook, find and prepare the ingredient, then prepare the meal.",
        ["textworld", "text-adventure", "cooking", "simple", "discrete"],
        50, 3.0, 3,
    ),
    "TextWorld-Cooking-Medium-v0": (
        "TextWorld Cooking challenge (medium): 6-room map, two-ingredient recipe.  "
        "Explore multiple rooms to collect both ingredients and prepare the meal.",
        ["textworld", "text-adventure", "cooking", "navigation", "multi-step", "discrete"],
        150, 4.0, 4,
    ),
}


_SUGGESTED_PARAMS: dict[str, SuggestedHyperparameters] = {
    "TextWorld-Take-v0": SuggestedHyperparameters(
        agent_type="tabular_q",
        n_episodes=200,
        max_steps=30,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99,
    ),
    "TextWorld-Navigate-v0": SuggestedHyperparameters(
        agent_type="tabular_q",
        n_episodes=300,
        max_steps=50,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.993,
    ),
    "TextWorld-TreasureHunt-v0": SuggestedHyperparameters(
        agent_type="tabular_q",
        n_episodes=500,
        max_steps=100,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.996,
    ),
    "TextWorld-CoinCollector-Easy-v0": SuggestedHyperparameters(
        agent_type="tabular_q",
        n_episodes=200,
        max_steps=30,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99,
    ),
    "TextWorld-CoinCollector-Medium-v0": SuggestedHyperparameters(
        agent_type="tabular_q",
        n_episodes=400,
        max_steps=100,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
    ),
    "TextWorld-Cooking-Easy-v0": SuggestedHyperparameters(
        agent_type="tabular_q",
        n_episodes=400,
        max_steps=50,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
    ),
    "TextWorld-Cooking-Medium-v0": SuggestedHyperparameters(
        agent_type="tabular_q",
        n_episodes=500,
        max_steps=150,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.997,
    ),
}


class TextWorldFactory(rlbridgeEnvironmentFactory):
    """Factory for a single compiled TextWorld game variant."""

    def __init__(self, env_id: str) -> None:
        if env_id not in _VARIANT_META:
            raise ValueError(
                f"Unknown TextWorld variant {env_id!r}.  "
                f"Available: {list(_VARIANT_META)}"
            )
        self._env_id = env_id
        desc, tags, max_steps, threshold, known_max_score = _VARIANT_META[env_id]
        self._desc = desc
        self._tags = tags
        self._max_steps = max_steps
        self._threshold = threshold
        self._known_max_score = known_max_score

    @property
    def env_info(self) -> EnvironmentInfo:
        return EnvironmentInfo(
            env_id=self._env_id,
            description=self._desc,
            tags=self._tags,
            max_episode_steps=self._max_steps,
            reward_threshold=self._threshold,
            namespace="textworld",
            render_modes=["ansi"],
            suggested_hyperparameters=_SUGGESTED_PARAMS.get(self._env_id),
        )

    def create(
        self,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> TextWorldEnvironment:
        game_file = _get_or_compile_game(self._env_id)
        return TextWorldEnvironment(
            env_id=self._env_id,
            game_file=game_file,
            max_episode_steps=self._max_steps,
            known_max_score=self._known_max_score,
        )


# ── Pre-built singletons ──────────────────────────────────────────────────────

TEXTWORLD_TAKE_V0                   = TextWorldFactory("TextWorld-Take-v0")
TEXTWORLD_NAVIGATE_V0               = TextWorldFactory("TextWorld-Navigate-v0")
TEXTWORLD_TREASUREHUNT_V0           = TextWorldFactory("TextWorld-TreasureHunt-v0")
TEXTWORLD_COINCOLLECTOR_EASY_V0     = TextWorldFactory("TextWorld-CoinCollector-Easy-v0")
TEXTWORLD_COINCOLLECTOR_MEDIUM_V0   = TextWorldFactory("TextWorld-CoinCollector-Medium-v0")
TEXTWORLD_COOKING_EASY_V0           = TextWorldFactory("TextWorld-Cooking-Easy-v0")
TEXTWORLD_COOKING_MEDIUM_V0         = TextWorldFactory("TextWorld-Cooking-Medium-v0")

ALL_TEXTWORLD_FACTORIES: list[TextWorldFactory] = [
    TEXTWORLD_TAKE_V0,
    TEXTWORLD_NAVIGATE_V0,
    TEXTWORLD_TREASUREHUNT_V0,
    TEXTWORLD_COINCOLLECTOR_EASY_V0,
    TEXTWORLD_COINCOLLECTOR_MEDIUM_V0,
    TEXTWORLD_COOKING_EASY_V0,
    TEXTWORLD_COOKING_MEDIUM_V0,
]
