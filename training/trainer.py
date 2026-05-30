"""
Training loop for the style-conditioned chess policy network.
"""
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


class Trainer:
    """
    Handles training of the ChessStyleNetwork with:
    - Cross-entropy loss on move prediction
    - Adam optimizer with cosine annealing
    - Validation tracking and early stopping
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

        # Move model to device
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs,
        )

        # Loss function (with label smoothing to reduce overfitting)
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=getattr(config, "LABEL_SMOOTHING", 0.0)
        )

        # Tracking
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.history = []

    def train_epoch(self) -> tuple:
        """Run one training epoch. Returns (avg_loss, accuracy)."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            board_tensors, style_ids, move_indices = batch
            board_tensors = board_tensors.to(self.device)
            style_ids = style_ids.to(self.device)
            move_indices = move_indices.to(self.device)

            # Forward pass
            logits = self.model(board_tensors, style_ids)

            # Loss
            loss = self.criterion(logits, move_indices)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Track metrics
            total_loss += loss.item() * board_tensors.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == move_indices).sum().item()
            total += board_tensors.size(0)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{correct / total:.3f}",
            )

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self) -> tuple:
        """Run validation. Returns (avg_loss, accuracy)."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            board_tensors, style_ids, move_indices = batch
            board_tensors = board_tensors.to(self.device)
            style_ids = style_ids.to(self.device)
            move_indices = move_indices.to(self.device)

            logits = self.model(board_tensors, style_ids)
            loss = self.criterion(logits, move_indices)

            total_loss += loss.item() * board_tensors.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == move_indices).sum().item()
            total += board_tensors.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy

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
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "train_acc",
                    "val_loss",
                    "val_acc",
                    "lr",
                    "time_sec",
                ]
            )
            for row in self.history:
                writer.writerow(
                    [
                        row["epoch"],
                        f"{row['train_loss']:.6f}",
                        f"{row['train_acc']:.4f}",
                        f"{row['val_loss']:.6f}",
                        f"{row['val_acc']:.4f}",
                        f"{row['lr']:.8f}",
                        f"{row['time']:.1f}",
                    ]
                )
        logger.info(f"Training log saved to {log_path}")

    def train(self) -> dict:
        """
        Run the full training loop.
        Returns the final training history.
        """
        logger.info("=" * 60)
        logger.info("Starting training")
        logger.info(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Epochs: {self.num_epochs}")
        logger.info(f"   Batch size: {self.train_loader.batch_size}")
        logger.info(f"   Train samples: {len(self.train_loader.dataset):,}")
        logger.info(f"   Val samples: {len(self.val_loader.dataset):,}")
        logger.info(f"   Learning rate: {self.lr}")
        logger.info("=" * 60)

        for epoch in range(1, self.num_epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            # Step scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            elapsed = time.time() - epoch_start

            # Record history
            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": current_lr,
                    "time": elapsed,
                }
            )

            # Log
            logger.info(
                f"Epoch {epoch:3d}/{self.num_epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Checkpointing (best model)
            if val_loss < self.best_val_loss:
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint(extra={"epoch": epoch})
                logger.info(
                    f"  New best model saved (val loss improved by {improvement:.4f})"
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

        logger.info("=" * 60)
        logger.info("Training complete")
        logger.info(f"   Best val loss: {self.best_val_loss:.4f}")
        logger.info(
            f"   Model saved to: {os.path.join(self.checkpoint_dir, self.checkpoint_name)}"
        )
        logger.info("=" * 60)

        return self.history
