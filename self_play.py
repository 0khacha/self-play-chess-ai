"""
Play Against Yourself AI -- Self-Play Tournament Entry Point

Usage:
    python self_play.py

This script:
1. Loads the trained model from output/models/
2. Creates 3 AI agents (Normal, Aggressive, Defensive)
3. Runs a round-robin self-play tournament
4. Outputs PGN files and statistics
"""
import sys
import os
import time

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils.helpers import setup_logging


def main():
    logger = setup_logging("self_play", log_dir=config.LOGS_DIR)

    model_path = os.path.join(config.MODELS_DIR, config.CHECKPOINT_NAME)

    print()
    print("=" * 60)
    print("PLAY AGAINST YOURSELF AI -- Self-Play Tournament")
    print("=" * 60)
    print(f"  Model:       {model_path}")
    print(f"  Device:      {config.DEVICE}")
    print(f"  Temperature: {config.SELFPLAY_TEMPERATURE}")
    print(f"  Games/pair:  {config.SELFPLAY_GAMES_PER_MATCHUP} per color ({config.SELFPLAY_GAMES_PER_MATCHUP * 2} total)")
    print(f"  Max moves:   {config.SELFPLAY_MAX_MOVES} per side")
    print("=" * 60)
    print()

    # Check model exists
    if not os.path.isfile(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("   Run 'python train.py' first to train the model.")
        sys.exit(1)

    total_start = time.time()

    # --- Run Tournament ----------------------------------------------
    print("Starting round-robin tournament...\n")
    print("  Matchups:")
    print("    Normal      vs  Aggressive")
    print("    Aggressive  vs  Defensive")
    print("    Defensive   vs  Normal")
    print()

    from selfplay.tournament import run_tournament

    results = run_tournament(model_path=model_path)

    # --- Print Statistics --------------------------------------------
    print()
    from selfplay.stats import print_tournament_stats, plot_style_wins

    print_tournament_stats(results)

    # --- Generate Chart ----------------------------------------------
    try:
        chart_path = os.path.join(config.LOGS_DIR, "style_wins.png")
        plot_style_wins(results, save_path=chart_path)
        print(f"\nStyle wins chart saved to: {chart_path}")
    except Exception as e:
        logger.warning(f"Could not generate chart: {e}")

    # --- Summary -----------------------------------------------------
    total_time = time.time() - total_start
    print()
    print("=" * 60)
    print(f"Tournament complete in {total_time:.0f}s ({total_time / 60:.1f} min)")
    print(f"   PGN files saved to: {config.GAMES_DIR}")
    print(f"   Logs saved to:      {config.LOGS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
