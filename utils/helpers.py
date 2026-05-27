"""
Shared utility functions for the chess AI project.
"""
import os
import logging
import datetime
import chess
import chess.pgn


def setup_logging(name: str = "chess_ai", log_dir: str = None) -> logging.Logger:
    """Set up a logger with file and console handlers."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (optional)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(
            os.path.join(log_dir, f"{name}_{timestamp}.log"), encoding="utf-8"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def write_pgn(game: chess.pgn.Game, filepath: str) -> None:
    """Write a chess.pgn.Game object to a PGN file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        print(game, file=f)


def append_pgn(game: chess.pgn.Game, filepath: str) -> None:
    """Append a chess.pgn.Game to an existing PGN file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        print(game, file=f)
        print(file=f)  # blank line separator


def format_result(result: str) -> str:
    """Format a game result string for display."""
    mapping = {
        "1-0": "White wins",
        "0-1": "Black wins",
        "1/2-1/2": "Draw",
        "*": "Ongoing",
    }
    return mapping.get(result, result)


def board_material_count(board: chess.Board, color: chess.Color) -> int:
    """
    Count material value for a given color.
    P=1, N=3, B=3, R=5, Q=9
    """
    values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }
    total = 0
    for piece_type, value in values.items():
        total += len(board.pieces(piece_type, color)) * value
    return total


def material_balance(board: chess.Board) -> int:
    """
    Return material balance from White's perspective.
    Positive = White has more material.
    """
    return board_material_count(board, chess.WHITE) - board_material_count(
        board, chess.BLACK
    )


def king_zone_squares(board: chess.Board, color: chess.Color) -> set:
    """
    Get the squares surrounding a king (the 'king zone').
    Returns a set of square indices.
    """
    king_sq = board.king(color)
    if king_sq is None:
        return set()

    king_rank = chess.square_rank(king_sq)
    king_file = chess.square_file(king_sq)
    zone = set()
    for dr in [-1, 0, 1]:
        for df in [-1, 0, 1]:
            r, f = king_rank + dr, king_file + df
            if 0 <= r <= 7 and 0 <= f <= 7:
                zone.add(chess.square(f, r))
    return zone
