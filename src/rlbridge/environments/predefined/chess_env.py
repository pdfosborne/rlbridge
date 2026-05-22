"""
rlbridge Chess Environment
========================

Wraps `python-chess <https://python-chess.readthedocs.io/>`_ to provide a
text-in/text-out chess RL environment.

Actions are given as move strings in any of the following formats:

* **UCI** - ``"e2e4"``, ``"g1f3"``, ``"e7e8q"`` (promotion)
* **SAN** - ``"e4"``, ``"Nf3"``, ``"O-O"`` (castling), ``"exd5"``
* **Natural language** - ``"pawn e2 to e4"``, ``"knight to f3"`` - a fuzzy
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
    Full standard chess, **no opponent** - both sides are controlled by the
    agent.  Every call to ``step()`` applies one move, alternating between
    White and Black.  Reward is from White's perspective at the end of the
    game.

Usage
-----
::

    from rlbridge.environments.registry import registry

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
    DiscreteSpace,
    EnvironmentInfo,
    RenderResult,
    ResetResult,
    StepResult,
    SuggestedHyperparameters,
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

# Maximum legal moves in any chess position (theoretical max ≈ 218; 256 is a
# safe, power-of-two ceiling that also acts as the fixed Discrete(n) size).
_MAX_LEGAL_MOVES = 256


def _normalize_move(text: str, board: Any) -> Optional[Any]:
    """Try to parse *text* as a UCI or SAN move legal in *board*.

    Returns a ``chess.Move`` on success, ``None`` if the move is not legal or
    the text cannot be parsed.
    """
    import chess  # local import to avoid top-level dep at import time
    text = text.strip()
    # Try UCI first (e.g. "e2e4", "g1f3", "e7e8q")
    try:
        move = chess.Move.from_uci(text)
        if move in board.legal_moves:
            return move
    except (ValueError, chess.InvalidMoveError):
        pass
    # Try SAN (e.g. "e4", "Nf3", "O-O")
    try:
        move = board.parse_san(text)
        if move in board.legal_moves:
            return move
    except (ValueError, chess.InvalidMoveError, chess.AmbiguousMoveError, chess.IllegalMoveError):
        pass
    return None


# ── RGB-array board renderer ─────────────────────────────────────────────────

# Font paths tried in order; first one that loads wins.
_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]

_PIECE_GLYPH: dict[tuple, str] = {}

def _piece_glyph(piece_type: int, color: bool) -> str:
    """Return the Unicode chess glyph for *piece_type* / *color* (lazy init)."""
    global _PIECE_GLYPH
    if not _PIECE_GLYPH:
        import chess
        _PIECE_GLYPH = {
            (chess.PAWN,   chess.WHITE): "\u2659",  # ♙
            (chess.KNIGHT, chess.WHITE): "\u2658",  # ♘
            (chess.BISHOP, chess.WHITE): "\u2657",  # ♗
            (chess.ROOK,   chess.WHITE): "\u2656",  # ♖
            (chess.QUEEN,  chess.WHITE): "\u2655",  # ♕
            (chess.KING,   chess.WHITE): "\u2654",  # ♔
            (chess.PAWN,   chess.BLACK): "\u265f",  # ♟
            (chess.KNIGHT, chess.BLACK): "\u265e",  # ♞
            (chess.BISHOP, chess.BLACK): "\u265d",  # ♝
            (chess.ROOK,   chess.BLACK): "\u265c",  # ♜
            (chess.QUEEN,  chess.BLACK): "\u265b",  # ♛
            (chess.KING,   chess.BLACK): "\u265a",  # ♚
        }
    return _PIECE_GLYPH.get((piece_type, color), "?")


def _render_board_png(board: Any, sq_size: int = 72) -> bytes:
    """Render *board* as a PNG image using Pillow.  Returns PNG bytes."""
    import io
    import chess
    from PIL import Image, ImageDraw, ImageFont

    LIGHT = (240, 217, 181)
    DARK  = (181, 136,  99)
    HL    = (205, 210, 106)   # last-move highlight
    BG    = ( 40,  40,  40)   # margin background

    margin = sq_size // 2
    img_w  = 8 * sq_size + 2 * margin
    img    = Image.new("RGB", (img_w, img_w), BG)
    draw   = ImageDraw.Draw(img)

    # Squares highlighted for the last move
    hl_squares: set = set()
    try:
        last = board.peek()
        hl_squares = {last.from_square, last.to_square}
    except Exception:
        pass

    for rank in range(8):
        for file in range(8):
            sq  = chess.square(file, rank)
            x   = margin + file * sq_size
            y   = margin + (7 - rank) * sq_size
            col = HL if sq in hl_squares else (LIGHT if (file + rank) % 2 == 0 else DARK)
            draw.rectangle([x, y, x + sq_size - 1, y + sq_size - 1], fill=col)

    # Fonts
    font_piece: Any = None
    font_label: Any = None
    for fp in _FONT_PATHS:
        try:
            font_piece = ImageFont.truetype(fp, int(sq_size * 0.72))
            font_label = ImageFont.truetype(fp, int(sq_size * 0.26))
            break
        except Exception:
            pass
    if font_piece is None:
        font_piece = font_label = ImageFont.load_default()

    # Pieces
    for sq, piece in board.piece_map().items():
        file  = chess.square_file(sq)
        rank  = chess.square_rank(sq)
        x     = margin + file * sq_size
        y     = margin + (7 - rank) * sq_size
        glyph = _piece_glyph(piece.piece_type, piece.color)
        fill   = ( 20,  20,  20) if piece.color == chess.WHITE else (245, 245, 245)
        stroke = (250, 250, 250) if piece.color == chess.WHITE else ( 10,  10,  10)
        try:
            bb = draw.textbbox((0, 0), glyph, font=font_piece)
            tx = x + (sq_size - (bb[2] - bb[0])) // 2 - bb[0]
            ty = y + (sq_size - (bb[3] - bb[1])) // 2 - bb[1]
            draw.text((tx, ty), glyph, font=font_piece, fill=fill,
                      stroke_width=2, stroke_fill=stroke)
        except Exception:
            pass

    # Coordinate labels
    lbl = (180, 180, 180)
    for i in range(8):
        cx = margin + i * sq_size + sq_size // 2
        cy = margin + (7 - i) * sq_size + sq_size // 2
        try:
            draw.text((cx, img_w - margin // 2), "abcdefgh"[i], font=font_label,
                      fill=lbl, anchor="mm")
            draw.text((margin // 2, cy), str(i + 1), font=font_label,
                      fill=lbl, anchor="mm")
        except Exception:
            pass

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _sorted_legal_uci(board: Any) -> list[str]:
    """Return all legal moves as UCI strings in a deterministic sorted order."""
    return sorted(m.uci() for m in board.legal_moves)


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


def _build_observation(
    board: Any,
    last_move_san: Optional[str],
    first_capture_mode: bool = False,
    discrete_actions: bool = False,
    legal_uci: Optional[list[str]] = None,
) -> str:
    """Build the text observation for the current board state."""
    import chess

    turn = "White" if board.turn == chess.WHITE else "Black"
    move_num = board.fullmove_number

    # Status line
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        status = f"CHECKMATE - {winner} wins!"
    elif board.is_stalemate():
        status = "STALEMATE - draw."
    elif board.is_insufficient_material():
        status = "DRAW - insufficient material."
    elif board.is_seventyfive_moves() or board.is_fifty_moves():
        status = "DRAW - 50-move rule."
    elif board.is_fivefold_repetition() or board.is_repetition(3):
        status = "DRAW - threefold repetition."
    elif board.is_check():
        status = f"CHECK - {turn} to move."
    else:
        status = f"{turn} to move."

    goal_line = "  [First-Capture mode: make the first capture to win]" if first_capture_mode else ""
    lines = [
        f"=== Chess  (Move {move_num}) ==={goal_line}",
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
        if discrete_actions and legal_uci is not None:
            # Show index → SAN table so the agent (and human) can read off
            # which integer action corresponds to which move.
            n_total = len(legal_uci)
            shown   = legal_uci[:32]
            san_map = []
            for idx, uci in enumerate(shown):
                try:
                    san = board.san(board.parse_uci(uci))
                except Exception:
                    san = uci
                san_map.append(f"  {idx:3d}: {san}")
            lines += [
                "",
                f"Legal moves - {n_total} total  (action = index, wraps with modulo):",
                *san_map,
                *([". . ."] if n_total > 32 else []),
                "",
                f"Enter an integer 0–{n_total - 1} (or any int; wraps mod {n_total}).",
            ]
        else:
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
        ``"random"`` - opponent plays a uniformly random legal move.
        ``"none"``   - no autonomous opponent; agent controls both sides
                       (self-play mode, alternating turns each step).
    player_color:
        ``"white"`` or ``"black"`` - the colour controlled by the agent
        when ``opponent != "none"``.  Ignored in self-play mode.
    max_episode_steps:
        Episode is truncated after this many half-moves (plies).
    """

    def __init__(
        self,
        opponent: str = "random",
        player_color: str = "white",
        max_episode_steps: int = 400,
        first_capture: bool = False,
        discrete_actions: bool = False,
        render_mode: Optional[str] = None,
    ) -> None:
        import chess  # deferred so ImportError surfaces clearly

        self._chess = chess
        self._opponent = opponent.lower()
        self._player_color = chess.WHITE if player_color.lower() == "white" else chess.BLACK
        self._max_episode_steps = max_episode_steps
        self._first_capture = first_capture
        self._discrete_actions = discrete_actions
        self._render_mode = render_mode

        self._board: chess.Board = chess.Board()
        self._steps: int = 0
        self._last_move_san: Optional[str] = None
        self._legal_uci: list[str] = []   # current sorted legal-move list (discrete mode)
        self._initialized: bool = False
        self._lock = threading.Lock()
        self._rng = random.Random()

    # ── env_id ────────────────────────────────────────────────────────────────

    @property
    def env_id(self) -> str:
        if self._opponent == "none":
            return "Chess-SelfPlay-v0"
        if self._first_capture and self._discrete_actions:
            return "Chess-FirstCapture-Discrete-v0"
        if self._first_capture:
            return "Chess-FirstCapture-v0"
        if self._discrete_actions:
            return "Chess-Discrete-v0"
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
                self._apply_opponent_move_internal()

            if self._discrete_actions:
                self._legal_uci = _sorted_legal_uci(self._board)

            return ResetResult(
                observation=_build_observation(
                    self._board, self._last_move_san,
                    self._first_capture, self._discrete_actions,
                    self._legal_uci if self._discrete_actions else None,
                ),
                info=self._build_info(),
            )

    def step(self, action: Any) -> StepResult:
        with self._lock:
            if not self._initialized:
                raise RuntimeError("Call reset() before step().")

            # --- Resolve discrete integer action → chess.Move ---
            if self._discrete_actions:
                move = self._resolve_discrete_action(action)
            else:
                move = _normalize_move(str(action), self._board)

            if move is None:
                # Invalid / illegal move: small penalty, no board change
                return StepResult(
                    observation=_build_observation(
                        self._board, self._last_move_san,
                        self._first_capture, self._discrete_actions,
                        self._legal_uci if self._discrete_actions else None,
                    ),
                    reward=-0.01,
                    terminated=False,
                    truncated=False,
                    info={**self._build_info(), "invalid_move": str(action)},
                )

            player_captures = self._first_capture and self._board.is_capture(move)
            self._last_move_san = self._board.san(move)
            self._board.push(move)
            self._steps += 1

            # First-capture: agent captured first - win
            if player_captures:
                if self._discrete_actions:
                    self._legal_uci = _sorted_legal_uci(self._board)
                return StepResult(
                    observation=_build_observation(
                        self._board, self._last_move_san,
                        self._first_capture, self._discrete_actions,
                        self._legal_uci if self._discrete_actions else None,
                    ),
                    reward=1.0,
                    terminated=True,
                    truncated=False,
                    info={**self._build_info(), "first_capture": "player"},
                )

            # Check if game ended after player's move (checkmate / stalemate)
            if self._board.is_game_over():
                reward = self._terminal_reward()
                if self._discrete_actions:
                    self._legal_uci = []
                return StepResult(
                    observation=_build_observation(
                        self._board, self._last_move_san,
                        self._first_capture, self._discrete_actions,
                        self._legal_uci if self._discrete_actions else None,
                    ),
                    reward=reward,
                    terminated=True,
                    truncated=False,
                    info=self._build_info(),
                )

            # --- Opponent responds (unless self-play) ---
            if self._opponent == "random":
                opp_move = self._pick_opponent_move()
                if opp_move is not None:
                    opp_captures = self._first_capture and self._board.is_capture(opp_move)
                    self._last_move_san = self._board.san(opp_move)
                    self._board.push(opp_move)

                    # First-capture: opponent captured first - loss
                    if opp_captures:
                        if self._discrete_actions:
                            self._legal_uci = _sorted_legal_uci(self._board)
                        return StepResult(
                            observation=_build_observation(
                                self._board, self._last_move_san,
                                self._first_capture, self._discrete_actions,
                                self._legal_uci if self._discrete_actions else None,
                            ),
                            reward=-1.0,
                            terminated=True,
                            truncated=False,
                            info={**self._build_info(), "first_capture": "opponent"},
                        )

                    if self._board.is_game_over():
                        reward = self._terminal_reward()
                        if self._discrete_actions:
                            self._legal_uci = []
                        return StepResult(
                            observation=_build_observation(
                                self._board, self._last_move_san,
                                self._first_capture, self._discrete_actions,
                                self._legal_uci if self._discrete_actions else None,
                            ),
                            reward=reward,
                            terminated=True,
                            truncated=False,
                            info=self._build_info(),
                        )

            # Update legal move list for next step
            if self._discrete_actions:
                self._legal_uci = _sorted_legal_uci(self._board)

            truncated = self._steps >= self._max_episode_steps
            return StepResult(
                observation=_build_observation(
                    self._board, self._last_move_san,
                    self._first_capture, self._discrete_actions,
                    self._legal_uci if self._discrete_actions else None,
                ),
                reward=0.0,
                terminated=False,
                truncated=truncated,
                info=self._build_info(),
            )

    def close(self) -> None:
        self._initialized = False

    # ── Internals ─────────────────────────────────────────────────────────────

    def _pick_opponent_move(self) -> Optional[Any]:
        """Pick (but do not push) a random legal move for the opponent."""
        legal = list(self._board.legal_moves)
        if not legal:
            return None
        return self._rng.choice(legal)

    def _apply_opponent_move_internal(self) -> None:
        """Pick and push a random legal move (used only inside reset())."""
        move = self._pick_opponent_move()
        if move is not None:
            self._last_move_san = self._board.san(move)
            self._board.push(move)

    def _resolve_discrete_action(self, action: Any) -> Optional[Any]:
        """
        Map an integer *action* to a legal ``chess.Move``.

        The mapping is::

            move = legal_moves[ action % n_legal ]

        where ``legal_moves`` is the sorted-UCI list captured at the start of
        the current turn.  This means **every integer is always valid** - no
        action is ever truly illegal in discrete mode, which avoids the
        −0.01 penalty that confuses gradient-based agents early in training.

        Strings and UCI/SAN text are still accepted for human interop.
        """
        if not self._legal_uci:
            return None
        # Also accept string moves for interop with text-only tools
        if isinstance(action, str):
            return _normalize_move(action, self._board)
        try:
            idx = int(action) % len(self._legal_uci)
        except (TypeError, ValueError):
            return None
        uci = self._legal_uci[idx]
        try:
            return self._board.parse_uci(uci)
        except Exception:
            return None

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
        info: dict[str, Any] = {
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
        if self._discrete_actions and self._legal_uci:
            # Expose the index → UCI mapping so callers can inspect / mask actions
            info["legal_moves_uci"]  = list(self._legal_uci)
            info["n_legal"]          = len(self._legal_uci)
        return info

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def observation_space(self) -> TextSpace:
        return TextSpace()

    @property
    def action_space(self):
        if self._discrete_actions:
            return DiscreteSpace(n=_MAX_LEGAL_MOVES)
        return TextSpace()

    @property
    def reward_range(self) -> tuple[float, float]:
        return (-1.0, 1.0)

    def render(self) -> RenderResult:
        if self._render_mode == "rgb_array":
            import base64 as _b64
            png = _render_board_png(self._board)
            return RenderResult(
                mode="rgb_array",
                data=_b64.b64encode(png).decode("ascii"),
                width=8 * 72 + 72,
                height=8 * 72 + 72,
            )
        return RenderResult(
            mode="ansi",
            text=_build_observation(
                self._board, self._last_move_san,
                self._first_capture, self._discrete_actions,
                self._legal_uci if self._discrete_actions else None,
            ),
        )

    def sample_action(self) -> Any:
        """Return a random legal action.

        In discrete mode returns a random integer index in ``[0, n_legal)``.
        In text mode returns a random legal move in UCI notation.
        """
        if self._discrete_actions:
            n = len(self._legal_uci)
            if n == 0:
                return 0
            return self._rng.randint(0, n - 1)
        legal = list(self._board.legal_moves)
        if not legal:
            return "a1a1"
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
    "Chess-Discrete-v0": (
        f"Standard chess with a Discrete({_MAX_LEGAL_MOVES}) action space. "
        "Each integer action maps to legal_moves[action % n_legal] "
        "so every integer is always valid - no illegal-move penalty. "
        "Optimal for DQN/PPO agents; tabular_q also works via hashed text observations. "
        "info dict includes 'legal_moves_uci' and 'n_legal' at each step.",
        ["chess", "board-game", "strategy", "discrete", "two-player", "rl-ready"],
        400,
        1.0,
        {"opponent": "random", "player_color": "white", "discrete_actions": True},
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
    "Chess-FirstCapture-v0": (
        "Shortened chess: the episode ends as soon as any piece is captured. "
        "Agent plays White against a uniform-random opponent. "
        "Reward +1 if the agent makes the first capture, -1 if the opponent does. "
        "Episodes are much shorter than standard chess - ideal for quick training. "
        "Actions: UCI (e.g. 'e2e4') or SAN (e.g. 'e4', 'Nf3', 'O-O').",
        ["chess", "board-game", "strategy", "text", "two-player", "quick"],
        80,
        1.0,
        {"opponent": "random", "player_color": "white", "first_capture": True},
    ),
    "Chess-FirstCapture-Discrete-v0": (
        f"First-capture chess with a Discrete({_MAX_LEGAL_MOVES}) action space. "
        "Combines the short-episode first-capture rule with the integer action mapping. "
        "Ideal for rapidly training discrete-action RL agents on chess tactics. "
        "info dict includes 'legal_moves_uci' and 'n_legal' at each step.",
        ["chess", "board-game", "strategy", "discrete", "two-player", "quick", "rl-ready"],
        80,
        1.0,
        {"opponent": "random", "player_color": "white",
         "first_capture": True, "discrete_actions": True},
    ),
}


_SUGGESTED_PARAMS: dict[str, SuggestedHyperparameters] = {
    "Chess-v0": SuggestedHyperparameters(
        agent_type="dqn",
        n_episodes=5000,
        max_steps=400,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.997,
    ),
    "Chess-Discrete-v0": SuggestedHyperparameters(
        agent_type="dqn",
        n_episodes=5000,
        max_steps=400,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.997,
        hidden_size=128,
        lr=1e-3,
    ),
    "Chess-SelfPlay-v0": SuggestedHyperparameters(
        agent_type="dqn",
        n_episodes=5000,
        max_steps=400,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.997,
    ),
    "Chess-FirstCapture-v0": SuggestedHyperparameters(
        agent_type="dqn",
        n_episodes=5000,
        max_steps=80,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
    ),
    "Chess-FirstCapture-Discrete-v0": SuggestedHyperparameters(
        agent_type="dqn",
        n_episodes=5000,
        max_steps=80,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        hidden_size=128,
        lr=1e-3,
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
            render_modes=["ansi", "rgb_array"],
            suggested_hyperparameters=_SUGGESTED_PARAMS.get(self._env_id),
        )

    def create(
        self,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> ChessEnvironment:
        merged = {**self._kwargs, **kwargs}
        return ChessEnvironment(
            max_episode_steps=merged.pop("max_episode_steps", self._max_steps),
            render_mode=render_mode,
            **merged,
        )


# ── Pre-built singletons ──────────────────────────────────────────────────────

CHESS_V0                      = ChessFactory("Chess-v0")
CHESS_DISCRETE_V0             = ChessFactory("Chess-Discrete-v0")
CHESS_SELFPLAY_V0             = ChessFactory("Chess-SelfPlay-v0")
CHESS_FIRST_CAPTURE_V0        = ChessFactory("Chess-FirstCapture-v0")
CHESS_FIRST_CAPTURE_DISCRETE_V0 = ChessFactory("Chess-FirstCapture-Discrete-v0")

ALL_CHESS_FACTORIES: list[ChessFactory] = [
    CHESS_V0,
    CHESS_DISCRETE_V0,
    CHESS_SELFPLAY_V0,
    CHESS_FIRST_CAPTURE_V0,
    CHESS_FIRST_CAPTURE_DISCRETE_V0,
]
