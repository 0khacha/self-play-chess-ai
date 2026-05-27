"""
Label each move with a playing-style category.

Two labelling modes are supported:

* **Heuristic-only** (``config.STOCKFISH_PATH is None``): uses pure
  python-chess logic to classify moves as aggressive, defensive, or normal.
* **Stockfish-enhanced** (``config.STOCKFISH_PATH`` is set): combines engine
  evaluation with positional heuristics for higher accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import chess
from tqdm import tqdm

import config
from data.parser import GameRecord
from utils.helpers import (
    board_material_count,
    king_zone_squares,
    setup_logging,
)

logger = setup_logging("labeler", config.LOGS_DIR)

# ---------------------------------------------
# Piece values for sacrifice / trade detection
# ---------------------------------------------
_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,  # king can never be captured, but avoids KeyError
}


@dataclass(slots=True)
class LabeledSample:
    """A (fen, move) pair annotated with a style label."""

    fen: str
    """Board FEN *before* the move."""

    move_uci: str
    """Move in UCI format."""

    style: int
    """One of ``config.STYLE_NORMAL``, ``STYLE_AGGRESSIVE``, ``STYLE_DEFENSIVE``."""

    player_color: chess.Color
    """The target player's colour in this position."""


# -----------------------------------------------------------------------------
# Heuristic helpers
# -----------------------------------------------------------------------------

def _relative_rank(square: int, color: chess.Color) -> int:
    """Return the rank of *square* from *color*'s perspective (0-7)."""
    rank = chess.square_rank(square)
    return rank if color == chess.WHITE else 7 - rank


def _is_on_own_half(square: int, color: chess.Color) -> bool:
    """True if the square is on ranks 14 from *color*'s perspective."""
    return _relative_rank(square, color) <= 3


def _is_pawn_push_past_rank5(
    move: chess.Move,
    board: chess.Board,
    color: chess.Color,
) -> bool:
    """True if a pawn advances past rank 5 (relative to the player)."""
    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type != chess.PAWN:
        return False
    return _relative_rank(move.to_square, color) >= 5


def _is_sacrifice_or_trade_up(move: chess.Move, board: chess.Board) -> bool:
    """True if a capture where moved piece value >= captured piece value."""
    if not board.is_capture(move):
        return False

    moving_piece = board.piece_at(move.from_square)
    if moving_piece is None:
        return False

    # Handle en passant: captured piece is a pawn
    captured_piece = board.piece_at(move.to_square)
    if captured_piece is None:
        # Must be en passant
        captured_value = _PIECE_VALUES[chess.PAWN]
    else:
        captured_value = _PIECE_VALUES.get(captured_piece.piece_type, 0)

    moving_value = _PIECE_VALUES.get(moving_piece.piece_type, 0)
    return moving_value >= captured_value


def _moves_into_king_zone(
    move: chess.Move,
    board: chess.Board,
    opponent_color: chess.Color,
) -> bool:
    """True if a piece moves into the opponent king's zone."""
    zone = king_zone_squares(board, opponent_color)
    return move.to_square in zone


def _is_retreat(move: chess.Move, color: chess.Color) -> bool:
    """True if the piece retreats (moves closer to own back rank)."""
    from_rel = _relative_rank(move.from_square, color)
    to_rel = _relative_rank(move.to_square, color)
    return to_rel < from_rel


# -----------------------------------------------------------------------------
# Mode A: heuristic-only labelling
# -----------------------------------------------------------------------------

def _label_heuristic(record: GameRecord) -> int:
    """Classify a single move using only python-chess heuristics."""
    board = chess.Board(record.fen)
    move = chess.Move.from_uci(record.move_uci)
    color = record.player_color
    opponent = not color

    # -- Aggressive indicators --------------------------------------------
    gives_check = board.gives_check(move)

    if _is_sacrifice_or_trade_up(move, board):
        return config.STYLE_AGGRESSIVE

    if gives_check:
        return config.STYLE_AGGRESSIVE

    if _is_pawn_push_past_rank5(move, board, color):
        return config.STYLE_AGGRESSIVE

    if _moves_into_king_zone(move, board, opponent):
        return config.STYLE_AGGRESSIVE

    # -- Defensive indicators ---------------------------------------------
    if board.is_castling(move):
        return config.STYLE_DEFENSIVE

    if _is_retreat(move, color):
        return config.STYLE_DEFENSIVE

    # Piece stays on own half, no capture, no check
    if (
        not board.is_capture(move)
        and not gives_check
        and _is_on_own_half(move.to_square, color)
    ):
        return config.STYLE_DEFENSIVE

    # -- Normal -----------------------------------------------------------
    return config.STYLE_NORMAL


# -----------------------------------------------------------------------------
# Mode B: Stockfish-enhanced labelling
# -----------------------------------------------------------------------------

def _init_stockfish():
    """Initialise and return a Stockfish engine instance via ``chess.engine``."""
    import chess.engine  # imported lazily so users without stockfish don't pay

    engine = chess.engine.SimpleEngine.popen_uci(config.STOCKFISH_PATH)
    engine.configure({
        "Threads": config.STOCKFISH_THREADS,
        "Hash": config.STOCKFISH_HASH_MB,
    })
    logger.info(
        "Stockfish initialised (%s, depth=%d, threads=%d, hash=%dMB)",
        config.STOCKFISH_PATH,
        config.STOCKFISH_DEPTH,
        config.STOCKFISH_THREADS,
        config.STOCKFISH_HASH_MB,
    )
    return engine


def _eval_score_cp(
    engine,
    board: chess.Board,
    depth: int,
) -> Optional[int]:
    """Return the evaluation in centipawns from *board.turn*'s POV.

    Mate scores are clamped to 100_000 centipawns.
    Returns ``None`` on engine failure.
    """
    import chess.engine

    try:
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"].pov(board.turn)
        cp = score.score(mate_score=100_000)
        return cp
    except Exception as exc:
        logger.debug("Stockfish eval error: %s", exc)
        return None


def _label_stockfish(
    record: GameRecord,
    engine,
) -> int:
    """Classify a single move using Stockfish evaluation + heuristics."""
    board = chess.Board(record.fen)
    move = chess.Move.from_uci(record.move_uci)
    color = record.player_color

    # Evaluate *before* the move (from the player's POV)
    eval_before = _eval_score_cp(engine, board, config.STOCKFISH_DEPTH)

    # Apply the move and evaluate *after*
    board_after = board.copy()
    board_after.push(move)
    # After pushing, it's the opponent's turn, so we need POV of the player
    eval_after_raw = _eval_score_cp(engine, board_after, config.STOCKFISH_DEPTH)
    # Flip sign because eval_after_raw is from opponent's POV
    eval_after = -eval_after_raw if eval_after_raw is not None else None

    # Fall back to heuristic if engine failed
    if eval_before is None or eval_after is None:
        return _label_heuristic(record)

    eval_change = eval_after - eval_before  # negative = player's eval got worse

    gives_check = board.gives_check(move)
    is_capture = board.is_capture(move)

    # Material balance change for the player
    mat_before = board_material_count(board, color)
    mat_after = board_material_count(board_after, color)
    material_change = mat_after - mat_before  # negative = lost material

    # -- Aggressive -------------------------------------------------------
    if (
        eval_change < config.AGGRESSIVE_EVAL_DROP
        and (is_capture or gives_check)
    ):
        return config.STYLE_AGGRESSIVE

    if material_change < 0:
        # Player sacrificed material
        return config.STYLE_AGGRESSIVE

    # -- Defensive --------------------------------------------------------
    # Total material on the board before and after
    total_mat_before = (
        board_material_count(board, chess.WHITE)
        + board_material_count(board, chess.BLACK)
    )
    total_mat_after = (
        board_material_count(board_after, chess.WHITE)
        + board_material_count(board_after, chess.BLACK)
    )
    simplifies = total_mat_after < total_mat_before
    retreats = _is_retreat(move, color)

    if abs(eval_change) <= config.DEFENSIVE_EVAL_STABILITY and (
        simplifies or retreats
    ):
        return config.STYLE_DEFENSIVE

    # -- Normal -----------------------------------------------------------
    return config.STYLE_NORMAL


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def label_samples(records: list[GameRecord]) -> list[LabeledSample]:
    """Assign a style label to every record.

    Automatically selects **Mode A** (heuristic) or **Mode B** (Stockfish)
    based on whether ``config.STOCKFISH_PATH`` is configured.

    Parameters
    ----------
    records : list[GameRecord]
        Records produced by :func:`data.parser.parse_games`.

    Returns
    -------
    list[LabeledSample]
        One labelled sample per input record.
    """
    use_stockfish = config.STOCKFISH_PATH is not None
    engine = None

    if use_stockfish:
        logger.info("Labelling with Stockfish-enhanced mode (Mode B)")
        engine = _init_stockfish()
    else:
        logger.info("Labelling with heuristic-only mode (Mode A)")

    samples: list[LabeledSample] = []
    label_counts = {
        config.STYLE_NORMAL: 0,
        config.STYLE_AGGRESSIVE: 0,
        config.STYLE_DEFENSIVE: 0,
    }
    errors = 0

    try:
        for record in tqdm(records, desc="Labelling moves", unit="move"):
            try:
                if use_stockfish and engine is not None:
                    style = _label_stockfish(record, engine)
                else:
                    style = _label_heuristic(record)

                samples.append(
                    LabeledSample(
                        fen=record.fen,
                        move_uci=record.move_uci,
                        style=style,
                        player_color=record.player_color,
                    )
                )
                label_counts[style] += 1
            except Exception as exc:
                logger.debug("Labelling error for %s: %s", record.move_uci, exc)
                errors += 1
    finally:
        if engine is not None:
            try:
                engine.quit()
                logger.debug("Stockfish engine shut down")
            except Exception:
                pass

    logger.info(
        "Labelling complete  %d samples "
        "(Normal=%d, Aggressive=%d, Defensive=%d, errors=%d)",
        len(samples),
        label_counts[config.STYLE_NORMAL],
        label_counts[config.STYLE_AGGRESSIVE],
        label_counts[config.STYLE_DEFENSIVE],
        errors,
    )
    return samples
