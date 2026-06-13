"""
Train a clone model for a specific Chess.com username.

Usage:
    python train_for_user.py                    # trains for default (0khacha)
    python train_for_user.py --user hikaru      # trains for any username

The trained model is saved to:
    output/models/clone_<username>.pt

It is automatically used by CloneAgent when that username is loaded.
"""

import argparse
import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils.helpers import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Train a chess clone model for a Chess.com user")
    parser.add_argument("--user", default=config.CHESS_COM_USERNAME,
                        help="Chess.com username to train on")
    parser.add_argument("--epochs", type=int, default=40,
                        help="Max training epochs (default: 40)")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate (default: 5e-4)")
    args = parser.parse_args()

    username = args.user.strip().lower()
    logger   = setup_logging("train_for_user", config.LOGS_DIR)

    model_name = f"clone_{username}.pt"
    model_path = os.path.join(config.MODELS_DIR, model_name)

    print()
    print("=" * 60)
    print(f"  Clone Model Training  ->  {username}")
    print("=" * 60)
    print(f"  Device   : {config.DEVICE}")
    print(f"  Save to  : {model_path}")
    print("=" * 60)
    print()

    t0 = time.time()

    # ── Step 1: Fetch games ──────────────────────────────────────────────
    print("Step 1/4 — Fetching games from Chess.com...")
    import requests
    from data.fetcher import fetch_all_games
    from data.parser  import parse_games

    headers      = {"User-Agent": config.API_USER_AGENT}
    archives_url = f"https://api.chess.com/pub/player/{username}/games/archives"
    resp = requests.get(archives_url, headers=headers, timeout=30)
    resp.raise_for_status()
    archives = resp.json().get("archives", [])
    print(f"  Found {len(archives)} monthly archives")

    games = fetch_all_games(archives, username)
    print(f"  Fetched {len(games)} games\n")

    if not games:
        print("ERROR: No games found. Check the username.")
        sys.exit(1)

    # ── Step 2: Parse ────────────────────────────────────────────────────
    print("Step 2/4 — Parsing positions...")
    records = parse_games(games, username)
    print(f"  Extracted {len(records)} position-move samples\n")

    if len(records) < 500:
        print(f"WARNING: Only {len(records)} samples — model quality may be poor.")
        print("  Consider a username with more games.\n")

    # ── Step 3: Label & dataset ──────────────────────────────────────────
    print("Step 3/4 — Building dataset...")
    from data.labeler import label_samples
    from data.dataset import build_dataloaders

    labeled = label_samples(records)

    # Use lighter augmentation & bigger val split to combat overfitting
    train_loader, val_loader = build_dataloaders(
        labeled,
        records=records,
        batch_size=256,
        val_split=0.20,   # 20% val — more honest evaluation
    )
    print(f"  Train: {len(train_loader.dataset):,}  Val: {len(val_loader.dataset):,}\n")

    # ── Step 4: Train ────────────────────────────────────────────────────
    print("Step 4/4 — Training model...")
    from model.network   import ChessStyleNetwork
    from training.trainer import Trainer

    model = ChessStyleNetwork.from_config()
    print(f"  Parameters: {model.count_parameters():,}\n")

    trainer = Trainer(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        lr           = args.lr,
        weight_decay = 5e-4,       # moderate regularisation
        num_epochs   = args.epochs,
        patience     = 10,
        checkpoint_name = model_name,
    )
    trainer.train()

    # ── Done ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"  Done in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"  Model: {model_path}")
    print()
    print("  The server (play.py) will automatically use this model")
    print(f"  when '{username}' is loaded in the UI.")
    print("=" * 60)
    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
