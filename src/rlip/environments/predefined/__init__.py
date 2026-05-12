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

# Chess (python-chess) is an optional dependency; skip if not installed.
try:
    from .chess_env import (
        ChessEnvironment,
        ChessFactory,
        CHESS_V0,
        CHESS_DISCRETE_V0,
        CHESS_SELFPLAY_V0,
        CHESS_FIRST_CAPTURE_V0,
        CHESS_FIRST_CAPTURE_DISCRETE_V0,
        ALL_CHESS_FACTORIES,
    )
    _CHESS_AVAILABLE = True
except ImportError:
    _CHESS_AVAILABLE = False
    ChessEnvironment = None         # type: ignore[assignment]
    ChessFactory = None             # type: ignore[assignment]
    ALL_CHESS_FACTORIES = []        # type: ignore[assignment]

# PyBoy/Pokemon Red is an optional dependency; skip if not installed.
try:
    from .pokemon_red import (
        PokemonRedEnvironment,
        PokemonRedFactory,
        POKEMON_RED_GARY_BATTLE_V0,
        ALL_POKEMON_RED_FACTORIES,
        setup_save_state as pokemon_red_setup_save_state,
    )
    _POKEMON_RED_AVAILABLE = True
except ImportError:
    _POKEMON_RED_AVAILABLE = False
    PokemonRedEnvironment = None       # type: ignore[assignment]
    PokemonRedFactory = None           # type: ignore[assignment]
    ALL_POKEMON_RED_FACTORIES = []     # type: ignore[assignment]

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
    # Chess (optional — requires python-chess)
    "ChessEnvironment",
    "ChessFactory",
    "CHESS_V0",
    "CHESS_DISCRETE_V0",
    "CHESS_SELFPLAY_V0",
    "CHESS_FIRST_CAPTURE_V0",
    "CHESS_FIRST_CAPTURE_DISCRETE_V0",
    "ALL_CHESS_FACTORIES",
    # Pokemon Red (optional — requires PyBoy + ROM)
    "PokemonRedEnvironment",
    "PokemonRedFactory",
    "POKEMON_RED_GARY_BATTLE_V0",
    "ALL_POKEMON_RED_FACTORIES",
    "pokemon_red_setup_save_state",
    # TextWorld (optional)
    "TextWorldEnvironment",
    "TextWorldFactory",
    "ALL_TEXTWORLD_FACTORIES",
]
