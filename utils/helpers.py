"""
Shared utility functions for the chess AI project.
"""
import os
import logging
import datetime
import chess


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
