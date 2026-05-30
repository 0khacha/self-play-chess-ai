"""
PyTorch Dataset and DataLoader utilities for chess style training.

Key improvements over v1:
  - **Game-based splitting**: train/val split by *game*, not by sample,
    eliminating data leakage from correlated positions.
  - **Horizontal flip augmentation**: each position is also mirrored
    (kingside ↔ queenside) to double effective dataset size.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Optional

import chess
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import config
from data.labeler import LabeledSample
from data.parser import GameRecord
from model.encoding import encode_board, move_to_index
from utils.helpers import setup_logging

logger = setup_logging("dataset", config.LOGS_DIR)


# --------------------------------------------------------------------------
# Horizontal flip helpers
# --------------------------------------------------------------------------

def _flip_square_h(sq: int) -> int:
    """Mirror a square horizontally (swap a↔h files)."""
    rank = chess.square_rank(sq)
    file = chess.square_file(sq)
    return chess.square(7 - file, rank)


def _flip_move_h(move: chess.Move) -> chess.Move:
    """Mirror a move horizontally."""
    return chess.Move(
        _flip_square_h(move.from_square),
        _flip_square_h(move.to_square),
        promotion=move.promotion,
    )


def _flip_fen_h(fen: str) -> str:
    """Mirror a FEN string horizontally (swap files a↔h).

    Castling rights are mirrored and en-passant squares are flipped.
    """
    parts = fen.split(" ")
    # Flip piece placement
    rows = parts[0].split("/")
    flipped_rows = []
    for row in rows:
        expanded = ""
        for ch in row:
            if ch.isdigit():
                expanded += "." * int(ch)
            else:
                expanded += ch
        flipped_rows.append(expanded[::-1])  # reverse each rank

    # Compress back to FEN notation
    compressed = []
    for row in flipped_rows:
        fen_row = ""
        empty = 0
        for ch in row:
            if ch == ".":
                empty += 1
            else:
                if empty:
                    fen_row += str(empty)
                    empty = 0
                fen_row += ch
        if empty:
            fen_row += str(empty)
        compressed.append(fen_row)

    parts[0] = "/".join(compressed)

    # Flip castling rights (K↔Q, k↔q)
    if len(parts) > 2 and parts[2] != "-":
        castle_map = str.maketrans("KQkq", "QKqk")
        parts[2] = parts[2].translate(castle_map)

    # Flip en-passant square
    if len(parts) > 3 and parts[3] != "-":
        ep = parts[3]
        file_idx = ord(ep[0]) - ord("a")
        parts[3] = chr(ord("a") + 7 - file_idx) + ep[1]

    return " ".join(parts)


class ChessStyleDataset(Dataset):
    """Eagerly-encoded chess dataset with optional augmentation.

    Each item is a tuple of::

        (board_tensor, style_id, move_index)
    """

    def __init__(
        self,
        samples: list[LabeledSample],
        augment: bool = False,
        game_ids: list[str] | None = None,
    ) -> None:
        """Encode all samples and store resulting tensors.

        Parameters
        ----------
        samples : list[LabeledSample]
            Labelled move samples.
        augment : bool
            If True, also add horizontally-flipped copies.
        game_ids : list[str] or None
            Parallel list of game IDs (same length as *samples*).
            Used only for logging / debugging.
        """
        board_tensors: list[torch.Tensor] = []
        style_ids: list[int] = []
        move_indices: list[int] = []
        skipped = 0

        for sample in tqdm(samples, desc="Encoding samples", unit="sample"):
            try:
                board = chess.Board(sample.fen)
                bt = torch.from_numpy(encode_board(board))
                move = chess.Move.from_uci(sample.move_uci)
                mi = move_to_index(move, board)

                board_tensors.append(bt)
                style_ids.append(sample.style)
                move_indices.append(mi)

                # --- Horizontal flip augmentation ---
                if augment:
                    try:
                        flipped_fen = _flip_fen_h(sample.fen)
                        flipped_board = chess.Board(flipped_fen)
                        flipped_move = _flip_move_h(move)

                        # Only add if the flipped move is legal
                        if flipped_move in flipped_board.legal_moves:
                            bt_flip = torch.from_numpy(encode_board(flipped_board))
                            mi_flip = move_to_index(flipped_move, flipped_board)
                            board_tensors.append(bt_flip)
                            style_ids.append(sample.style)
                            move_indices.append(mi_flip)
                    except Exception:
                        pass  # skip bad flips silently

            except Exception as exc:
                logger.debug(
                    "Skipping sample (fen=%s, move=%s): %s",
                    sample.fen, sample.move_uci, exc,
                )
                skipped += 1

        if not board_tensors:
            logger.warning("Dataset is empty — no valid samples encoded")
            self.board_tensors = torch.empty(0, config.NUM_BOARD_PLANES, 8, 8)
            self.style_ids = torch.empty(0, dtype=torch.long)
            self.move_indices = torch.empty(0, dtype=torch.long)
        else:
            self.board_tensors = torch.stack(board_tensors)
            self.style_ids = torch.tensor(style_ids, dtype=torch.long)
            self.move_indices = torch.tensor(move_indices, dtype=torch.long)

        logger.info(
            "Dataset created — %d samples encoded (%d augmented), %d skipped",
            len(self.board_tensors),
            len(self.board_tensors) - len(samples) + skipped,
            skipped,
        )

    def __len__(self) -> int:
        return len(self.board_tensors)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.board_tensors[idx],
            self.style_ids[idx],
            self.move_indices[idx],
        )


# --------------------------------------------------------------------------
# Game-aware splitting
# --------------------------------------------------------------------------

def _group_samples_by_game(
    samples: list[LabeledSample],
    records: list[GameRecord] | None = None,
    game_ids: list[str] | None = None,
) -> dict[str, list[int]]:
    """Group sample indices by their game ID.

    If *game_ids* is provided it is used directly.  Otherwise, we use
    *records* (which must be the same length and order as *samples*).
    As a last resort we assign each sample to its own "game" (degrades
    to random splitting but never leaks).
    """
    n = len(samples)
    groups: dict[str, list[int]] = defaultdict(list)

    if game_ids and len(game_ids) == n:
        for i, gid in enumerate(game_ids):
            groups[gid].append(i)
    elif records and len(records) == n:
        for i, rec in enumerate(records):
            groups[rec.game_id].append(i)
    else:
        # Fallback: each sample is its own game
        for i in range(n):
            groups[f"__sample_{i}"].append(i)

    return dict(groups)


def build_dataloaders(
    samples: list[LabeledSample],
    batch_size: int | None = None,
    val_split: float | None = None,
    records: list[GameRecord] | None = None,
    game_ids: list[str] | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders with **game-based** splitting.

    All positions from the same game go entirely into train OR val — never
    both — eliminating the data-leakage that inflates training metrics.

    Parameters
    ----------
    samples : list[LabeledSample]
        Labelled samples.
    batch_size : int, optional
        Mini-batch size.  Defaults to ``config.BATCH_SIZE``.
    val_split : float, optional
        Fraction to hold out.  Defaults to ``config.VALIDATION_SPLIT``.
    records : list[GameRecord], optional
        If provided (same length/order as *samples*), used to extract game IDs.
    game_ids : list[str], optional
        Explicit list of game IDs, one per sample.
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if val_split is None:
        val_split = config.VALIDATION_SPLIT

    total = len(samples)
    if total == 0:
        logger.warning("No samples — returning empty DataLoaders")
        empty_ds = ChessStyleDataset([])
        empty_loader = DataLoader(empty_ds, batch_size=1)
        return empty_loader, empty_loader

    # --- Group by game and split -----------------------------------------
    groups = _group_samples_by_game(samples, records, game_ids)
    game_list = list(groups.keys())

    # Deterministic shuffle
    import random as _rng
    rng = _rng.Random(42)
    rng.shuffle(game_list)

    # Walk through games until we fill the val budget
    val_budget = max(1, int(total * val_split))
    val_indices: list[int] = []
    train_indices: list[int] = []
    val_count = 0

    for gid in game_list:
        idxs = groups[gid]
        if val_count < val_budget:
            val_indices.extend(idxs)
            val_count += len(idxs)
        else:
            train_indices.extend(idxs)

    # Build separate sample lists
    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]

    # Build datasets (augment training only)
    train_ds = ChessStyleDataset(train_samples, augment=True)
    val_ds = ChessStyleDataset(val_samples, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(config.DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(config.DEVICE.type == "cuda"),
    )

    # -- Print statistics -------------------------------------------------
    style_counts = Counter(s.style for s in samples)
    style_pct = {
        config.STYLE_NAMES.get(k, str(k)): f"{v} ({v / total * 100:.1f}%)"
        for k, v in sorted(style_counts.items())
    }

    n_games = len(game_list)
    n_val_games = sum(1 for gid in game_list if groups[gid][0] in set(val_indices))
    n_train_games = n_games - n_val_games

    print("\n+==========================================+")
    print("|         Dataset Statistics               |")
    print("+==========================================+")
    print(f"|  Total samples : {total:<23} |")
    print(f"|  Total games   : {n_games:<23} |")
    print(f"|  Training      : {len(train_ds):<7} ({n_train_games} games, +aug)  |")
    print(f"|  Validation    : {len(val_ds):<7} ({n_val_games} games)       |")
    print(f"|  Batch size    : {batch_size:<23} |")
    print("+==========================================+")
    print("|  Style distribution:                     |")
    for name, count_str in style_pct.items():
        label = f"    {name}"
        print(f"|  {label:<25} {count_str:<13}|")
    print("+==========================================+\n")

    logger.info(
        "DataLoaders ready — train=%d batches, val=%d batches "
        "(split by %d games)",
        len(train_loader), len(val_loader), n_games,
    )
    return train_loader, val_loader
