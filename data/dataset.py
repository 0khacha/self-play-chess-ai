"""
PyTorch Dataset and DataLoader utilities for chess style training.

Provides :class:`ChessStyleDataset` which pre-encodes all board positions
and moves at construction time, and :func:`build_dataloaders` for creating
train/val splits.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

import chess
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

import config
from data.labeler import LabeledSample
from model.encoding import encode_board, move_to_index
from utils.helpers import setup_logging

logger = setup_logging("dataset", config.LOGS_DIR)


class ChessStyleDataset(Dataset):
    """Lazily-indexed, eagerly-encoded chess dataset.

    All samples are encoded once during ``__init__`` so that training
    iterations incur no per-sample overhead.

    Each item is a tuple of::

        (board_tensor, style_id, move_index)

    Where:
        * ``board_tensor`` is a ``(18, 8, 8)`` float32 tensor,
        * ``style_id`` is a long scalar (0 / 1 / 2),
        * ``move_index`` is a long scalar in ``[0, 4671]``.
    """

    def __init__(self, samples: list[LabeledSample]) -> None:
        """Encode all samples and store resulting tensors.

        Samples that fail encoding (e.g. corrupt FEN, illegal move) are
        silently skipped with a warning.
        """
        board_tensors: list[torch.Tensor] = []
        style_ids: list[int] = []
        move_indices: list[int] = []
        skipped = 0

        for sample in tqdm(samples, desc="Encoding samples", unit="sample"):
            try:
                board = chess.Board(sample.fen)

                # Board tensor: (18, 8, 8)
                bt = torch.from_numpy(encode_board(board))

                # Move index: int in [0, 4671]
                move = chess.Move.from_uci(sample.move_uci)
                mi = move_to_index(move, board)

                board_tensors.append(bt)
                style_ids.append(sample.style)
                move_indices.append(mi)
            except Exception as exc:
                logger.debug(
                    "Skipping sample (fen=%s, move=%s): %s",
                    sample.fen,
                    sample.move_uci,
                    exc,
                )
                skipped += 1

        if not board_tensors:
            logger.warning("Dataset is empty  no valid samples encoded")
            self.board_tensors = torch.empty(0, 18, 8, 8)
            self.style_ids = torch.empty(0, dtype=torch.long)
            self.move_indices = torch.empty(0, dtype=torch.long)
        else:
            self.board_tensors = torch.stack(board_tensors)  # (N, 18, 8, 8)
            self.style_ids = torch.tensor(style_ids, dtype=torch.long)  # (N,)
            self.move_indices = torch.tensor(move_indices, dtype=torch.long)  # (N,)

        logger.info(
            "Dataset created  %d samples encoded, %d skipped",
            len(self.board_tensors),
            skipped,
        )

    def __len__(self) -> int:
        return len(self.board_tensors)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(board_tensor, style_id, move_index)``."""
        return (
            self.board_tensors[idx],
            self.style_ids[idx],
            self.move_indices[idx],
        )


def build_dataloaders(
    samples: list[LabeledSample],
    batch_size: Optional[int] = None,
    val_split: Optional[float] = None,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders from labelled samples.

    Parameters
    ----------
    samples : list[LabeledSample]
        Labelled samples produced by :func:`data.labeler.label_samples`.
    batch_size : int, optional
        Mini-batch size.  Defaults to ``config.BATCH_SIZE``.
    val_split : float, optional
        Fraction of data to hold out for validation.
        Defaults to ``config.VALIDATION_SPLIT``.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        ``(train_loader, val_loader)``
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if val_split is None:
        val_split = config.VALIDATION_SPLIT

    # Build the full dataset
    dataset = ChessStyleDataset(samples)
    total = len(dataset)

    if total == 0:
        logger.warning("No samples  returning empty DataLoaders")
        empty_loader = DataLoader(dataset, batch_size=1)
        return empty_loader, empty_loader

    # Random split
    val_size = max(1, int(total * val_split))
    train_size = total - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

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

    print("\n+==========================================+")
    print("|         Dataset Statistics               |")
    print("+==========================================+")
    print(f"|  Total samples : {total:<23} |")
    print(f"|  Training      : {train_size:<23} |")
    print(f"|  Validation    : {val_size:<23} |")
    print(f"|  Batch size    : {batch_size:<23} |")
    print("+==========================================+")
    print("|  Style distribution:                     |")
    for name, count_str in style_pct.items():
        label = f"    {name}"
        print(f"|  {label:<25} {count_str:<13}|")
    print("+==========================================+\n")

    logger.info(
        "DataLoaders ready  train=%d batches, val=%d batches",
        len(train_loader),
        len(val_loader),
    )
    return train_loader, val_loader
