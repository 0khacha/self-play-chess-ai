"""
Training loop for the style-conditioned chess policy network.

Features:
  - Top-1, top-3, top-5 accuracy tracking
  - Learning rate warmup + cosine annealing
  - Overfitting gap detection with warnings
  - Model checkpointing with early stopping
"""
from __future__ import annotations

import os
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from utils.helpers import setup_logging


logger = setup_logging("trainer")


def _topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, ks: list[int]) -> dict[int, int]:
    """Compute top-k correct counts for multiple k values.

    Returns a dict mapping k -> number of correct predictions.
    """
    maxk = max(ks)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    correct = pred.eq(targets.unsqueeze(1).expand_as(pred))

    results = {}
    for k in ks:
        results[k] = correct[:, :k].any(dim=1).sum().item()
    return results


class Trainer:
    """
    Handles training of the ChessStyleNetwork with:
    - Cross-entropy loss on move prediction
    - Adam optimizer with LR warmup + cosine annealing
    - Top-1, top-3, top-5 accuracy tracking
    - Validation tracking and early stopping
    - Overfitting gap detection
    - Model checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = None,
        weight_decay: float = None,
        num_epochs: int = None,
        patience: int = None,
        device: torch.device = None,
        checkpoint_dir: str = None,
        checkpoint_name: str = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr or config.LEARNING_RATE
        self.weight_decay = weight_decay or config.WEIGHT_DECAY
        self.num_epochs = num_epochs or config.NUM_EPOCHS
        self.patience = patience or config.EARLY_STOPPING_PATIENCE
        self.device = device or config.DEVICE
        self.checkpoint_dir = checkpoint_dir or config.MODELS_DIR
        self.checkpoint_name = checkpoint_name or config.CHECKPOINT_NAME

        # Top-K values to track
        self.top_ks = getattr(config, "TOP_K_ACCURACIES", [1, 3, 5])
        self.warmup_epochs = getattr(config, "LR_WARMUP_EPOCHS", 3)

        # Move model to device
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Scheduler: warmup + cosine annealing
        # We'll handle warmup manually and use cosine for the rest
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.num_epochs - self.warmup_epochs),
        )

        # Loss function (with label smoothing to reduce overfitting)
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=getattr(config, "LABEL_SMOOTHING", 0.0)
        )

        # Tracking
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.history = []

    def _get_warmup_lr(self, epoch: int) -> float:
        """Compute the linearly-warmed-up learning rate for the given epoch."""
        if epoch <= self.warmup_epochs and self.warmup_epochs > 0:
            return self.lr * (epoch / self.warmup_epochs)
        return self.lr

    def _apply_warmup(self, epoch: int) -> None:
        """Apply linear warmup if we're in the warmup phase."""
        if epoch <= self.warmup_epochs:
            warmup_lr = self._get_warmup_lr(epoch)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = warmup_lr

    def _extract_logits(self, model_output):
        """Extract policy logits from model output (handles tuple or plain tensor)."""
        if isinstance(model_output, tuple):
            return model_output[0]  # (policy_logits, value)
        return model_output

    def train_epoch(self) -> dict:
        """Run one training epoch. Returns metrics dict."""
        self.model.train()
        total_loss = 0.0
        correct_counts = {k: 0 for k in self.top_ks}
        total = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            board_tensors, style_ids, move_indices = batch
            board_tensors = board_tensors.to(self.device)
            style_ids = style_ids.to(self.device)
            move_indices = move_indices.to(self.device)

            # Forward pass
            output = self.model(board_tensors, style_ids)
            logits = self._extract_logits(output)

            # Loss
            loss = self.criterion(logits, move_indices)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Track metrics
            batch_size = board_tensors.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size

            # Top-K accuracy
            tk = _topk_accuracy(logits, move_indices, self.top_ks)
            for k in self.top_ks:
                correct_counts[k] += tk[k]

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                top1=f"{correct_counts[1] / total:.3f}" if 1 in correct_counts else "",
            )

        avg_loss = total_loss / total
        accuracies = {k: correct_counts[k] / total for k in self.top_ks}
        return {"loss": avg_loss, **{f"top{k}_acc": accuracies[k] for k in self.top_ks}}

    @torch.no_grad()
    def validate(self) -> dict:
        """Run validation. Returns metrics dict."""
        self.model.eval()
        total_loss = 0.0
        correct_counts = {k: 0 for k in self.top_ks}
        total = 0

        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            board_tensors, style_ids, move_indices = batch
            board_tensors = board_tensors.to(self.device)
            style_ids = style_ids.to(self.device)
            move_indices = move_indices.to(self.device)

            output = self.model(board_tensors, style_ids)
            logits = self._extract_logits(output)
            loss = self.criterion(logits, move_indices)

            batch_size = board_tensors.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size

            tk = _topk_accuracy(logits, move_indices, self.top_ks)
            for k in self.top_ks:
                correct_counts[k] += tk[k]

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracies = {k: correct_counts[k] / total if total > 0 else 0.0 for k in self.top_ks}
        return {"loss": avg_loss, **{f"top{k}_acc": accuracies[k] for k in self.top_ks}}

    def save_checkpoint(self, filepath: str = None, extra: dict = None):
        """Save model checkpoint."""
        if filepath is None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            filepath = os.path.join(self.checkpoint_dir, self.checkpoint_name)

        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }
        if extra:
            state.update(extra)

        torch.save(state, filepath)
        logger.info(f"Checkpoint saved to {filepath}")

    def save_training_log(self):
        """Save training history to CSV."""
        log_path = os.path.join(config.LOGS_DIR, "training_log.csv")

        # Build header dynamically based on top-k values
        topk_train_cols = [f"train_top{k}_acc" for k in self.top_ks]
        topk_val_cols = [f"val_top{k}_acc" for k in self.top_ks]

        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["epoch", "train_loss"] + topk_train_cols +
                ["val_loss"] + topk_val_cols +
                ["lr", "time_sec", "overfit_gap"]
            )
            for row in self.history:
                train_topk = [f"{row.get(f'train_top{k}_acc', 0):.4f}" for k in self.top_ks]
                val_topk = [f"{row.get(f'val_top{k}_acc', 0):.4f}" for k in self.top_ks]
                writer.writerow(
                    [row["epoch"], f"{row['train_loss']:.6f}"] + train_topk +
                    [f"{row['val_loss']:.6f}"] + val_topk +
                    [f"{row['lr']:.8f}", f"{row['time']:.1f}",
                     f"{row.get('overfit_gap', 0):.4f}"]
                )
        logger.info(f"Training log saved to {log_path}")

    def train(self) -> list[dict]:
        """
        Run the full training loop.
        Returns the final training history.
        """
        logger.info("=" * 60)
        logger.info("Starting training")
        logger.info(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Epochs: {self.num_epochs}")
        logger.info(f"   Warmup epochs: {self.warmup_epochs}")
        logger.info(f"   Batch size: {self.train_loader.batch_size}")
        logger.info(f"   Train samples: {len(self.train_loader.dataset):,}")
        logger.info(f"   Val samples: {len(self.val_loader.dataset):,}")
        logger.info(f"   Learning rate: {self.lr}")
        logger.info(f"   Top-K tracking: {self.top_ks}")
        logger.info("=" * 60)

        for epoch in range(1, self.num_epochs + 1):
            epoch_start = time.time()

            # Apply warmup LR
            if epoch <= self.warmup_epochs:
                self._apply_warmup(epoch)

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Step scheduler (only after warmup phase)
            if epoch > self.warmup_epochs:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - epoch_start

            # Overfitting gap
            train_top1 = train_metrics.get("top1_acc", 0)
            val_top1 = val_metrics.get("top1_acc", 0)
            overfit_gap = train_top1 - val_top1

            # Record history
            record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "lr": current_lr,
                "time": elapsed,
                "overfit_gap": overfit_gap,
            }
            for k in self.top_ks:
                record[f"train_top{k}_acc"] = train_metrics.get(f"top{k}_acc", 0)
                record[f"val_top{k}_acc"] = val_metrics.get(f"top{k}_acc", 0)
            self.history.append(record)

            # Log
            topk_str = " | ".join(
                f"Top-{k}: {train_metrics.get(f'top{k}_acc', 0):.3f}/{val_metrics.get(f'top{k}_acc', 0):.3f}"
                for k in self.top_ks
            )
            logger.info(
                f"Epoch {epoch:3d}/{self.num_epochs} | "
                f"Loss: {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
                f"{topk_str} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Overfitting warning
            if overfit_gap > 0.25:
                logger.warning(
                    f"  ⚠ Overfitting detected: train-val gap = {overfit_gap:.3f} "
                    f"(train={train_top1:.3f}, val={val_top1:.3f})"
                )

            # Checkpointing (best model by val loss)
            val_loss = val_metrics["loss"]
            if val_loss < self.best_val_loss:
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint(extra={"epoch": epoch})
                logger.info(
                    f"  ✓ New best model saved (val loss improved by {improvement:.4f})"
                )
            else:
                self.epochs_without_improvement += 1
                logger.info(
                    f"  No improvement for {self.epochs_without_improvement}/{self.patience} epochs"
                )

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                logger.info(
                    f"Early stopping after {epoch} epochs "
                    f"(no improvement for {self.patience} epochs)"
                )
                break

        # Save training log
        self.save_training_log()

        # Final summary
        best_epoch = min(self.history, key=lambda r: r["val_loss"])
        logger.info("=" * 60)
        logger.info("Training complete")
        logger.info(f"   Best val loss: {self.best_val_loss:.4f} (epoch {best_epoch['epoch']})")
        for k in self.top_ks:
            logger.info(f"   Best val top-{k} acc: {best_epoch.get(f'val_top{k}_acc', 0):.4f}")
        logger.info(
            f"   Model saved to: {os.path.join(self.checkpoint_dir, self.checkpoint_name)}"
        )
        logger.info("=" * 60)

        return self.history
