"""
RLIP Chess Environment
========================

Wraps `python-chess <https://python-chess.readthedocs.io/>`_ to provide a
text-in/text-out chess RL environment.

Actions are given as move strings in any of the following formats:

* **UCI** — ``"e2e4"``, ``"g1f3"``, ``"e7e8q"`` (promotion)
* **SAN** — ``"e4"``, ``"Nf3"``, ``"O-O"`` (castling), ``"exd5"``
* **Natural language** — ``"pawn e2 to e4"``, ``"knight to f3"`` — a fuzzy
  matcher attempts to infer the intended UCI move; the closest legal move is
  chosen.

Observations are a rich text description of the current board state including
the board diagram, material counts, legal moves, and game status.

Environment variants
--------------------
``Chess-v0``
    Full standard chess.  The opponent plays uniformly at random from all
    legal moves (a weak but unbiased adversary for bootstrapping agents).
    Player plays **White**.  Reward: ``+1`` win, ``-1`` loss, ``0`` draw,
    ``0`` per intermediate step.

``Chess-SelfPlay-v0``
    Full standard chess, **no opponent** — both sides are controlled by the
    agent.  Every call to ``step()`` applies one move, alternating between
    White and Black.  Reward is from White's perspective at the end of the
    game.

Usage
-----
::

    from rlip.environments.registry import registry

    env = registry.get("Chess-v0").create()
    result = env.reset(seed=42)
    print(result.observation)           # board diagram + legal moves
    result = env.step("e2e4")           # UCI
    result = env.step("e4")             # SAN also accepted
"""

from __future__ import annotations

import random
import re
import threading
from typing import Any, Optional

from ..base import RLIPEnvironment, RLIPEnvironmentFactory
from ...protocol.messages import (
    EnvironmentInfo,
    RenderResult,
    ResetResult,
    StepResult,
    TextSpace,
)

# ── Constants ────────────────────────────────────────────────────────────────

_PIECE_NAME: dict[str, str] = {
    "P": "Pawn",   "N": "Knight", "B": "Bishop",
    "R": "Rook",   "Q": "Queen",  "K": "King",
    "p": "pawn",   "n": "knight", "b": "bishop",
    "r": "rook",   "q": "queen",  "k": "king",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_move(text: str, board: Any) -> Optional[Any]:
    """
    Try to parse *text* as a legal move on *board*.

    Tries UCI then SAN, then a fuzzy keyword extractor.
    Returns a ``chess.Move`` or ``None`` if no legal match found.
    """
    import chess

    text = text.strip()
    if not text:
        return None

    # 1. Direct UCI parse (e.g. "e2e4")
    try:
        move = chess.Move.from_uci(text.lower())
        if move in board.legal_moves:
            return move
    except ValueError:
        pass

    # 2. SAN parse (e.g. "e4", "Nf3", "O-O")
    try:
        move = board.parse_san(text)
        if move in board.legal_moves:
            return move
    except (ValueError, chess.AmbiguousMoveError, chess.IllegalMoveError):
        pass

    # 3. Fuzzy: strip common filler words and retry UCI / SAN on what remains
    cleaned = re.sub(
        r"\b(move|play|go|put|place|takes?|captures?|to|from|the|a|an|"
        r"my|your|pawn|knight|bishop|rook|queen|king|on|at|piece)\b",
        " ",
        text.lower(),
    ).strip()
    # Extract square names (a-h)(1-8) from the remaining string
    squares = re.findall(r"[a-h][1-8]", cleaned)
    if len(squares) >= 2:
        uci_guess = squares[0] + squares[1]
        try:
            move = chess.Move.from_uci(uci_guess)
            if move in board.legal_moves:
                return move
        except ValueError:
            pass
    elif len(squares) == 1:
        # Might be just a destination square; find a unique piece that can go there
        dest = chess.parse_square(squares[0])
        candidates = [
            m for m in board.legal_moves if m.to_square == dest
        ]
        if len(candidates) == 1:
            return candidates[0]

    return None


def _material_balance(board: Any) -> str:
    """Return a one-line string showing piece counts for each side."""
    import chess

    wpieces = {p: 0 for p in "PNBRQ"}
    bpieces = {p: 0 for p in "pnbrq"}
    for piece in board.piece_map().values():
        sym = piece.symbol()
        if sym.upper() in wpieces:
            if sym.isupper():
                wpieces[sym] += 1
            else:
                bpieces[sym] += 1
    def fmt(pc: dict) -> str:
        return "  ".join(f"{_PIECE_NAME[k]}×{v}"
                         for k, v in pc.items() if v)
    return (
        f"White: {fmt(wpieces) or '—'}\n"
        f"Black: {fmt(bpieces) or '—'}"
    )


def _san_list(board: Any, limit: int = 20) -> list[str]:
    """Return up to *limit* legal moves in SAN notation."""
    moves = [board.san(m) for m in board.legal_moves]
    return sorted(moves)[:limit]


def _build_observation(board: Any, last_move_san: Optional[str]) -> str:
    """Build the text observation for the current board state."""
    import chess

    turn = "White" if board.turn == chess.WHITE else "Black"
    move_num = board.fullmove_number

    # Status line
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        status = f"CHECKMATE — {winner} wins!"
    elif board.is_stalemate():
        status = "STALEMATE — draw."
    elif board.is_insufficient_material():
        status = "DRAW — insufficient material."
    elif board.is_seventyfive_moves() or board.is_fifty_moves():
        status = "DRAW — 50-move rule."
    elif board.is_fivefold_repetition() or board.is_repetition(3):
        status = "DRAW — threefold repetition."
    elif board.is_check():
        status = f"CHECK — {turn} to move."
    else:
        status = f"{turn} to move."

    lines = [
        f"=== Chess  (Move {move_num}) ===",
        f"Status: {status}",
    ]

    if last_move_san:
        prev = "Black" if board.turn == chess.WHITE else "White"
        lines.append(f"Last move ({prev}): {last_move_san}")

    lines += [
        "",
        board.unicode(borders=True),
        "",
        "Material:",
        _material_balance(board),
        f"FEN: {board.fen()}",
    ]

    if not board.is_game_over():
        legal = _san_list(board)
        n_total = board.legal_moves.count()
        shown = legal[:20]
        lines += [
            "",
            f"Legal moves ({n_total} total, showing first 20 in SAN):",
            "  " + "  ".join(shown),
            "",
            "Enter a move in UCI (e.g. 'e2e4') or SAN (e.g. 'e4', 'Nf3', 'O-O').",
        ]

    return "\n".join(lines)


# ── Environment ───────────────────────────────────────────────────────────────

class ChessEnvironment(RLIPEnvironment):
    """
    Text-based chess RL environment powered by python-chess.

    Parameters
    ----------
    opponent:
        ``"random"`` — opponent plays a uniformly random legal move.
        ``"none"``   — no autonomous opponent; agent controls both sides
                       (self-play mode, alternating turns each step).
    player_color:
        ``"white"`` or ``"black"`` — the colour controlled by the agent
        when ``opponent != "none"``.  Ignored in self-play mode.
    max_episode_steps:
        Episode is truncated after this many half-moves (plies).
    """

    def __init__(
        self,
        opponent: str = "random",
        player_color: str = "white",
        max_episode_steps: int = 400,
    ) -> None:
        import chess  # deferred so ImportError surfaces clearly

        self._chess = chess
        self._opponent = opponent.lower()
        self._player_color = chess.WHITE if player_color.lower() == "white" else chess.BLACK
        self._max_episode_steps = max_episode_steps

        self._board: chess.Board = chess.Board()
        self._steps: int = 0
        self._last_move_san: Optional[str] = None
        self._initialized: bool = False
        self._lock = threading.Lock()
        self._rng = random.Random()

    # ── env_id ────────────────────────────────────────────────────────────────

    @property
    def env_id(self) -> str:
        if self._opponent == "none":
            return "Chess-SelfPlay-v0"
        return "Chess-v0"

    # ── Life-cycle ────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> ResetResult:
        with self._lock:
            self._board = self._chess.Board()
            self._steps = 0
            self._last_move_san = None
            self._initialized = True

            if seed is not None:
                self._rng.seed(seed)

            # If agent plays Black, let the random opponent make the first move
            if (
                self._opponent == "random"
                and self._player_color == self._chess.BLACK
            ):
                self._apply_opponent_move()

            return ResetResult(
                observation=_build_observation(self._board, self._last_move_san),
                info=self._build_info(),
            )

    def step(self, action: Any) -> StepResult:
        with self._lock:
            if not self._initialized:
                raise RuntimeError("Call reset() before step().")

            # --- Parse and apply the player's move ---
            move = _normalize_move(str(action), self._board)
            if move is None:
                # Invalid / illegal move: small penalty, no board change
                return StepResult(
                    observation=_build_observation(self._board, self._last_move_san),
                    reward=-0.01,
                    terminated=False,
                    truncated=False,
                    info={**self._build_info(), "invalid_move": str(action)},
                )

            self._last_move_san = self._board.san(move)
            self._board.push(move)
            self._steps += 1

            # Check if game ended after player's move
            if self._board.is_game_over():
                reward = self._terminal_reward()
                return StepResult(
                    observation=_build_observation(self._board, self._last_move_san),
                    reward=reward,
                    terminated=True,
                    truncated=False,
                    info=self._build_info(),
                )

            # --- Opponent responds (unless self-play) ---
            if self._opponent == "random":
                self._apply_opponent_move()

                if self._board.is_game_over():
                    reward = self._terminal_reward()
                    return StepResult(
                        observation=_build_observation(self._board, self._last_move_san),
                        reward=reward,
                        terminated=True,
                        truncated=False,
                        info=self._build_info(),
                    )

            truncated = self._steps >= self._max_episode_steps
            return StepResult(
                observation=_build_observation(self._board, self._last_move_san),
                reward=0.0,
                terminated=False,
                truncated=truncated,
                info=self._build_info(),
            )

    def close(self) -> None:
        self._initialized = False

    # ── Internals ─────────────────────────────────────────────────────────────

    def _apply_opponent_move(self) -> None:
        """Pick and push a random legal move for the opponent."""
        legal = list(self._board.legal_moves)
        if not legal:
            return
        move = self._rng.choice(legal)
        self._last_move_san = self._board.san(move)
        self._board.push(move)

    def _terminal_reward(self) -> float:
        """Reward from the agent's perspective."""
        outcome = self._board.outcome()
        if outcome is None:
            return 0.0
        if outcome.winner is None:
            return 0.0  # draw
        # In self-play mode, return from White's perspective
        if self._opponent == "none":
            return 1.0 if outcome.winner == self._chess.WHITE else -1.0
        return 1.0 if outcome.winner == self._player_color else -1.0

    def _build_info(self) -> dict[str, Any]:
        board = self._board
        outcome = board.outcome()
        return {
            "fen":              board.fen(),
            "turn":             "white" if board.turn == self._chess.WHITE else "black",
            "fullmove_number":  board.fullmove_number,
            "halfmove_clock":   board.halfmove_clock,
            "is_check":         board.is_check(),
            "is_game_over":     board.is_game_over(),
            "termination":      outcome.termination.name if outcome else None,
            "winner":           (
                ("white" if outcome.winner == self._chess.WHITE else "black")
                if outcome and outcome.winner is not None else
                ("draw" if outcome else None)
            ),
            "legal_move_count": board.legal_moves.count(),
            "step":             self._steps,
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
        return RenderResult(
            mode="ansi",
            text=_build_observation(self._board, self._last_move_san),
        )

    def sample_action(self) -> str:
        """Return a random legal move in UCI notation."""
        legal = list(self._board.legal_moves)
        if not legal:
            return "a1a1"  # no-op fallback (shouldn't happen)
        return self._rng.choice(legal).uci()


# ── Factory ───────────────────────────────────────────────────────────────────

_VARIANT_META: dict[str, tuple[str, list[str], int, float | None, dict[str, Any]]] = {
    "Chess-v0": (
        "Standard chess. Agent plays White against a uniform-random opponent. "
        "Reward +1 on win, -1 on loss, 0 on draw. "
        "Actions: UCI (e.g. 'e2e4') or SAN (e.g. 'e4', 'Nf3', 'O-O').",
        ["chess", "board-game", "strategy", "text", "two-player"],
        400,
        1.0,
        {"opponent": "random", "player_color": "white"},
    ),
    "Chess-SelfPlay-v0": (
        "Standard chess in self-play mode. "
        "The agent controls both White and Black, alternating each step. "
        "Reward is from White's perspective at game end. "
        "Actions: UCI (e.g. 'e2e4') or SAN (e.g. 'e4', 'Nf3', 'O-O').",
        ["chess", "board-game", "strategy", "text", "self-play"],
        400,
        None,
        {"opponent": "none", "player_color": "white"},
    ),
}


class ChessFactory(RLIPEnvironmentFactory):
    """Factory for a chess environment variant."""

    def __init__(self, env_id: str) -> None:
        if env_id not in _VARIANT_META:
            raise ValueError(
                f"Unknown Chess variant {env_id!r}.  "
                f"Available: {list(_VARIANT_META)}"
            )
        self._env_id = env_id
        desc, tags, max_steps, threshold, kwargs = _VARIANT_META[env_id]
        self._desc       = desc
        self._tags       = tags
        self._max_steps  = max_steps
        self._threshold  = threshold
        self._kwargs     = kwargs

    @property
    def env_info(self) -> EnvironmentInfo:
        return EnvironmentInfo(
            env_id=self._env_id,
            description=self._desc,
            tags=self._tags,
            max_episode_steps=self._max_steps,
            reward_threshold=self._threshold,
            namespace="chess",
            render_modes=["ansi"],
        )

    def create(
        self,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> ChessEnvironment:
        merged = {**self._kwargs, **kwargs}
        return ChessEnvironment(
            max_episode_steps=self._max_steps,
            **merged,
        )


# ── Pre-built singletons ──────────────────────────────────────────────────────

CHESS_V0            = ChessFactory("Chess-v0")
CHESS_SELFPLAY_V0   = ChessFactory("Chess-SelfPlay-v0")

ALL_CHESS_FACTORIES: list[ChessFactory] = [
    CHESS_V0,
    CHESS_SELFPLAY_V0,
]
