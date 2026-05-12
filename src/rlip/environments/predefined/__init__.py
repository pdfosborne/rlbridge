"""Predefined RLIP environments — bundled, ready-to-use without extra installs."""

from .sailing import (
    SailingEnvironment,
    SailingFactory,
    SAILING_V0,
    SAILING_HARD_V0,
    ALL_SAILING_FACTORIES,
)
from .gridworld import (
    GridWorldEnv,
    GridWorldFactory,
    GRIDWORLD_V0,
    ALL_GRIDWORLD_FACTORIES,
)

# TextWorld is an optional dependency; skip registration if not installed.
try:
    from .textworld import (
        TextWorldEnvironment,
        TextWorldFactory,
        TEXTWORLD_TAKE_V0,
        TEXTWORLD_NAVIGATE_V0,
        TEXTWORLD_TREASUREHUNT_V0,
        TEXTWORLD_COINCOLLECTOR_EASY_V0,
        TEXTWORLD_COINCOLLECTOR_MEDIUM_V0,
        TEXTWORLD_COOKING_EASY_V0,
        TEXTWORLD_COOKING_MEDIUM_V0,
        ALL_TEXTWORLD_FACTORIES,
    )
    _TEXTWORLD_AVAILABLE = True
except ImportError:
    _TEXTWORLD_AVAILABLE = False
    TextWorldEnvironment = None  # type: ignore[assignment]
    TextWorldFactory = None      # type: ignore[assignment]
    ALL_TEXTWORLD_FACTORIES = [] # type: ignore[assignment]

__all__ = [
    "SailingEnvironment",
    "SailingFactory",
    "SAILING_V0",
    "SAILING_HARD_V0",
    "ALL_SAILING_FACTORIES",
    "GridWorldEnv",
    "GridWorldFactory",
    "GRIDWORLD_V0",
    "ALL_GRIDWORLD_FACTORIES",
    # TextWorld (optional)
    "TextWorldEnvironment",
    "TextWorldFactory",
    "ALL_TEXTWORLD_FACTORIES",
]
