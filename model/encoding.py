"""
Board and move encoding for the chess neural network.

Board encoding: 21 planes x 8x8 (perspective-relative)
Move encoding: AlphaZero-style 4672 move vocabulary (64 squares x 73 move types)
"""
from __future__ import annotations

import chess
import numpy as np
import torch

# ---------------------------------------------
# Direction vectors for queen-type moves
# ---------------------------------------------
# (delta_rank, delta_file)
QUEEN_DIRECTIONS = [
    (1, 0),    # North
    (1, 1),    # NE
    (0, 1),    # East
    (-1, 1),   # SE
    (-1, 0),   # South
    (-1, -1),  # SW
    (0, -1),   # West
    (1, -1),   # NW
]

# ---------------------------------------------
# Knight move deltas
# ---------------------------------------------
KNIGHT_DELTAS = [
    (2, 1), (2, -1), (-2, 1), (-2, -1),
    (1, 2), (1, -2), (-1, 2), (-1, -2),
]

# ---------------------------------------------
# Piece types in order
# ---------------------------------------------
PIECE_TYPES = [
    chess.PAWN, chess.KNIGHT, chess.BISHOP,
    chess.ROOK, chess.QUEEN, chess.KING,
]


def _flip_square(sq: int) -> int:
    """Flip a square vertically (for black's perspective). a1<->a8, etc."""
    return sq ^ 56  # XOR with 56 flips the rank bits


def encode_board(board: chess.Board) -> np.ndarray:
    """
    Encode a chess board as a 21x8x8 float32 tensor.
    Uses perspective-relative encoding (current player's POV).

    Planes:
        0-5:   Current player's pieces (P, N, B, R, Q, K)
        6-11:  Opponent's pieces (P, N, B, R, Q, K)
        12:    Constant 1s (side to move — always current player)
        13:    Current player kingside castling
        14:    Current player queenside castling
        15:    Opponent kingside castling
        16:    Opponent queenside castling
        17:    En passant square
        18:    Attack map — squares attacked by current player
        19:    Defense map — squares attacked by opponent
        20:    Last move — to-square of the most recent move
    """
    planes = np.zeros((21, 8, 8), dtype=np.float32)

    us = board.turn
    them = not us

    for i, pt in enumerate(PIECE_TYPES):
        # Current player's pieces -> planes 0-5
        for sq in board.pieces(pt, us):
            if us == chess.BLACK:
                sq = _flip_square(sq)
            r, f = chess.square_rank(sq), chess.square_file(sq)
            planes[i, r, f] = 1.0

        # Opponent's pieces -> planes 6-11
        for sq in board.pieces(pt, them):
            if us == chess.BLACK:
                sq = _flip_square(sq)
            r, f = chess.square_rank(sq), chess.square_file(sq)
            planes[i + 6, r, f] = 1.0

    # Plane 12: side to move (always 1 since perspective-relative)
    planes[12, :, :] = 1.0

    # Planes 13-16: castling rights
    planes[13, :, :] = float(board.has_kingside_castling_rights(us))
    planes[14, :, :] = float(board.has_queenside_castling_rights(us))
    planes[15, :, :] = float(board.has_kingside_castling_rights(them))
    planes[16, :, :] = float(board.has_queenside_castling_rights(them))

    # Plane 17: en passant
    if board.ep_square is not None:
        ep_sq = board.ep_square
        if us == chess.BLACK:
            ep_sq = _flip_square(ep_sq)
        r, f = chess.square_rank(ep_sq), chess.square_file(ep_sq)
        planes[17, r, f] = 1.0

    # Plane 18: attack map — squares attacked by current player
    for sq in chess.SQUARES:
        if board.is_attacked_by(us, sq):
            mapped_sq = _flip_square(sq) if us == chess.BLACK else sq
            r, f = chess.square_rank(mapped_sq), chess.square_file(mapped_sq)
            planes[18, r, f] = 1.0

    # Plane 19: defense map — squares attacked by opponent
    for sq in chess.SQUARES:
        if board.is_attacked_by(them, sq):
            mapped_sq = _flip_square(sq) if us == chess.BLACK else sq
            r, f = chess.square_rank(mapped_sq), chess.square_file(mapped_sq)
            planes[19, r, f] = 1.0

    # Plane 20: last move — to-square of the most recent move
    if board.move_stack:
        last_to = board.move_stack[-1].to_square
        if us == chess.BLACK:
            last_to = _flip_square(last_to)
        r, f = chess.square_rank(last_to), chess.square_file(last_to)
        planes[20, r, f] = 1.0

    return planes


def encode_board_tensor(board: chess.Board) -> torch.Tensor:
    """Encode board and return as a PyTorch tensor."""
    return torch.from_numpy(encode_board(board))


# ---------------------------------------------
# Move Encoding: AlphaZero-style (4672 moves)
# ---------------------------------------------
#
# Each move is encoded as: from_square * 73 + move_type_index
#
# Move type indices (73 total):
#   0-55:  Queen moves  8 directions x 7 distances
#          direction_idx * 7 + (distance - 1)
#   56-63: Knight moves  8 possible L-shapes
#   64-72: Underpromotions  3 pieces x 3 directions
#          64 + piece_idx * 3 + direction_idx
#          piece_idx: 0=knight, 1=bishop, 2=rook
#          direction_idx: 0=straight, 1=capture-left, 2=capture-right


def _direction_and_distance(from_sq: int, to_sq: int):
    """Get direction index and distance for a sliding move."""
    from_rank = chess.square_rank(from_sq)
    from_file = chess.square_file(from_sq)
    to_rank = chess.square_rank(to_sq)
    to_file = chess.square_file(to_sq)

    dr = to_rank - from_rank
    df = to_file - from_file

    distance = max(abs(dr), abs(df))
    if distance == 0:
        return None, None

    # Normalize to unit direction
    unit_dr = (1 if dr > 0 else -1 if dr < 0 else 0)
    unit_df = (1 if df > 0 else -1 if df < 0 else 0)

    try:
        dir_idx = QUEEN_DIRECTIONS.index((unit_dr, unit_df))
        return dir_idx, distance
    except ValueError:
        return None, None


def _knight_move_index(from_sq: int, to_sq: int):
    """Get knight move index (0-7) or None if not a knight move."""
    from_rank = chess.square_rank(from_sq)
    from_file = chess.square_file(from_sq)
    to_rank = chess.square_rank(to_sq)
    to_file = chess.square_file(to_sq)

    dr = to_rank - from_rank
    df = to_file - from_file

    try:
        return KNIGHT_DELTAS.index((dr, df))
    except ValueError:
        return None


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """
    Convert a chess.Move to an index in [0, 4671].
    Uses perspective-relative encoding: flips squares for black.
    """
    from_sq = move.from_square
    to_sq = move.to_square

    # Flip for black's perspective
    if board.turn == chess.BLACK:
        from_sq = _flip_square(from_sq)
        to_sq = _flip_square(to_sq)

    # Check underpromotions first
    if move.promotion is not None and move.promotion != chess.QUEEN:
        promo_map = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}
        piece_idx = promo_map[move.promotion]

        from_file = chess.square_file(from_sq)
        to_file = chess.square_file(to_sq)
        df = to_file - from_file

        if df == 0:
            dir_idx = 0   # straight
        elif df == -1:
            dir_idx = 1   # capture left
        else:
            dir_idx = 2   # capture right

        move_type = 64 + piece_idx * 3 + dir_idx
        return from_sq * 73 + move_type

    # Check knight moves
    k_idx = _knight_move_index(from_sq, to_sq)
    if k_idx is not None:
        move_type = 56 + k_idx
        return from_sq * 73 + move_type

    # Queen-type move (covers rook, bishop, queen, king, pawn moves)
    dir_idx, distance = _direction_and_distance(from_sq, to_sq)
    if dir_idx is not None and distance is not None:
        move_type = dir_idx * 7 + (distance - 1)
        return from_sq * 73 + move_type

    # Fallback: should never reach here for legal moves
    raise ValueError(f"Cannot encode move {move} from board {board.fen()}")


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """
    Convert an index in [0, 4671] back to a chess.Move.
    Must provide the board to determine perspective and validate.
    """
    from_sq = index // 73
    move_type = index % 73

    # Determine the target square based on move type
    if move_type < 56:
        # Queen-type move
        dir_idx = move_type // 7
        distance = (move_type % 7) + 1
        dr, df = QUEEN_DIRECTIONS[dir_idx]
        from_rank = chess.square_rank(from_sq)
        from_file = chess.square_file(from_sq)
        to_rank = from_rank + dr * distance
        to_file = from_file + df * distance
        if not (0 <= to_rank <= 7 and 0 <= to_file <= 7):
            return None
        to_sq = chess.square(to_file, to_rank)
        promotion = None

        # Check if this is a pawn reaching the last rank (auto-queen)
        if board.turn == chess.BLACK:
            actual_from = _flip_square(from_sq)
        else:
            actual_from = from_sq
        piece = board.piece_at(actual_from)
        if piece and piece.piece_type == chess.PAWN:
            if board.turn == chess.BLACK:
                actual_to = _flip_square(to_sq)
            else:
                actual_to = to_sq
            to_rank_actual = chess.square_rank(actual_to)
            if to_rank_actual == 7 or to_rank_actual == 0:
                promotion = chess.QUEEN

    elif move_type < 64:
        # Knight move
        k_idx = move_type - 56
        dr, df = KNIGHT_DELTAS[k_idx]
        from_rank = chess.square_rank(from_sq)
        from_file = chess.square_file(from_sq)
        to_rank = from_rank + dr
        to_file = from_file + df
        if not (0 <= to_rank <= 7 and 0 <= to_file <= 7):
            return None
        to_sq = chess.square(to_file, to_rank)
        promotion = None

    else:
        # Underpromotion
        up_idx = move_type - 64
        piece_idx = up_idx // 3
        dir_idx = up_idx % 3

        promo_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
        promotion = promo_pieces[piece_idx]

        from_file = chess.square_file(from_sq)
        from_rank = chess.square_rank(from_sq)

        if dir_idx == 0:
            to_file = from_file       # straight
        elif dir_idx == 1:
            to_file = from_file - 1   # capture left
        else:
            to_file = from_file + 1   # capture right

        to_rank = from_rank + 1  # always forward in relative perspective
        if not (0 <= to_file <= 7 and 0 <= to_rank <= 7):
            return None
        to_sq = chess.square(to_file, to_rank)

    # Flip back for black's perspective
    if board.turn == chess.BLACK:
        from_sq = _flip_square(from_sq)
        to_sq = _flip_square(to_sq)

    return chess.Move(from_sq, to_sq, promotion=promotion)


def get_legal_move_mask(board: chess.Board) -> torch.Tensor:
    """
    Create a binary mask of shape (4672,) where legal moves are 1.0
    and illegal moves are 0.0.
    """
    mask = torch.zeros(4672, dtype=torch.float32)
    for move in board.legal_moves:
        try:
            idx = move_to_index(move, board)
            mask[idx] = 1.0
        except (ValueError, IndexError):
            pass
    return mask


def get_legal_move_indices(board: chess.Board) -> dict:
    """
    Return a dict mapping move_index -> chess.Move for all legal moves.
    """
    mapping = {}
    for move in board.legal_moves:
        try:
            idx = move_to_index(move, board)
            mapping[idx] = move
        except (ValueError, IndexError):
            pass
    return mapping
