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
]
