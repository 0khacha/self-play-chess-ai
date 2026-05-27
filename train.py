"""
Play Against Yourself AI -- Training Entry Point

Usage:
    python train.py

This script:
1. Fetches games from Chess.com archives
2. Parses PGN files and extracts player moves
3. Labels each move with a playing style (Normal/Aggressive/Defensive)
4. Trains a style-conditioned neural network to predict moves
5. Saves the trained model to output/models/
"""
import sys
import os
import time

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils.helpers import setup_logging


def main():
    logger = setup_logging("train", log_dir=config.LOGS_DIR)

    print()
    print("=" * 60)
    print("PLAY AGAINST YOURSELF AI -- Training Pipeline")
    print("=" * 60)
    print(f"  Player:    {config.CHESS_COM_USERNAME}")
    print(f"  Archives:  {len(config.CHESS_COM_ARCHIVES)}")
    print(f"  Device:    {config.DEVICE}")
    if config.STOCKFISH_PATH:
        print(f"  Stockfish: {config.STOCKFISH_PATH}")
    else:
        print("  Stockfish: not configured (using heuristic labeling)")
    print("=" * 60)
    print()

    total_start = time.time()

    # --- Step 1: Fetch games ------------------------------------------
    print("Step 1/5: Fetching games from Chess.com...")
    from data.fetcher import fetch_all_games

    games = fetch_all_games()
    print(f"   Fetched {len(games)} games\n")

    if not games:
        print("ERROR: No games found. Check username and archives in config.py.")
        sys.exit(1)

    # --- Step 2: Parse PGN -------------------------------------------
    print("Step 2/5: Parsing PGN and extracting positions...")
    from data.parser import parse_games

    records = parse_games(games)
    print(f"   Extracted {len(records)} position-move samples\n")

    if not records:
        print("ERROR: No valid positions extracted.")
        sys.exit(1)

    # --- Step 3: Label styles ----------------------------------------
    print("Step 3/5: Labeling move styles...")
    from data.labeler import label_samples

    labeled = label_samples(records)
    print(f"   Labeled {len(labeled)} samples")

    # Print style distribution
    from collections import Counter

    style_counts = Counter(s.style for s in labeled)
    for style_id, name in config.STYLE_NAMES.items():
        count = style_counts.get(style_id, 0)
        pct = count / len(labeled) * 100 if labeled else 0
        print(f"      {name}: {count:,} ({pct:.1f}%)")
    print()

    # --- Step 4: Build dataset ---------------------------------------
    print("Step 4/5: Building training dataset...")
    from data.dataset import build_dataloaders

    train_loader, val_loader = build_dataloaders(labeled)
    print(f"   Train: {len(train_loader.dataset):,} samples, Val: {len(val_loader.dataset):,} samples\n")

    # --- Step 5: Train model -----------------------------------------
    print("Step 5/5: Training the model...")
    from model.network import ChessStyleNetwork
    from training.trainer import Trainer

    model = ChessStyleNetwork.from_config()
    print(f"   Model parameters: {model.count_parameters():,}")
    print()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    history = trainer.train()

    # --- Done --------------------------------------------------------
    total_time = time.time() - total_start
    print()
    print("=" * 60)
    print(f"Training complete in {total_time:.0f}s ({total_time / 60:.1f} min)")
    print(f"   Model saved to: {os.path.join(config.MODELS_DIR, config.CHECKPOINT_NAME)}")
    print()
    print("   Next step: Run self-play tournament:")
    print("   > python self_play.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
