"""
Tournament statistics, analysis, and visualisation.

Provides functions to compute per-matchup and overall style scores,
print formatted result tables, and optionally generate matplotlib
bar charts.
"""
from __future__ import annotations

import os
from typing import Any

import config
from selfplay.tournament import TournamentResults, MatchupResult


# ---------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------

def analyze_tournament(results: TournamentResults) -> dict[str, Any]:
    """Compute comprehensive statistics from tournament results.

    Returns
    -------
    dict
        Keys:
        - ``matchups``  : list of per-matchup stat dicts
        - ``style_wins``  : dict mapping style_id -> total wins
        - ``style_draws`` : dict mapping style_id -> total draws
        - ``style_games`` : dict mapping style_id -> total games played
        - ``style_win_rate`` : dict mapping style_id -> win rate (float)
        - ``avg_game_length`` : average half-moves per game
        - ``total_games``  : total games played
        - ``dominant_style`` : style_id with the most wins
    """
    # Accumulators per style
    style_wins: dict[int, int] = {s: 0 for s in config.STYLE_NAMES}
    style_draws: dict[int, int] = {s: 0 for s in config.STYLE_NAMES}
    style_games: dict[int, int] = {s: 0 for s in config.STYLE_NAMES}

    matchup_stats: list[dict[str, Any]] = []
    total_half_moves = 0
    total_games = 0

    for matchup in results.matchups:
        num_games = matchup.wins_a + matchup.wins_b + matchup.draws
        win_rate_a = matchup.wins_a / num_games if num_games else 0.0
        win_rate_b = matchup.wins_b / num_games if num_games else 0.0
        draw_rate = matchup.draws / num_games if num_games else 0.0

        # Average game length for this matchup
        matchup_half_moves = sum(g.move_count for g in matchup.games)
        avg_len = matchup_half_moves / num_games if num_games else 0.0

        matchup_stats.append({
            "style_a": matchup.style_a,
            "style_b": matchup.style_b,
            "label_a": config.STYLE_NAMES[matchup.style_a],
            "label_b": config.STYLE_NAMES[matchup.style_b],
            "wins_a": matchup.wins_a,
            "wins_b": matchup.wins_b,
            "draws": matchup.draws,
            "total": num_games,
            "win_rate_a": win_rate_a,
            "win_rate_b": win_rate_b,
            "draw_rate": draw_rate,
            "avg_game_length": avg_len,
        })

        # Roll up into overall style tallies
        style_wins[matchup.style_a] += matchup.wins_a
        style_wins[matchup.style_b] += matchup.wins_b

        style_draws[matchup.style_a] += matchup.draws
        style_draws[matchup.style_b] += matchup.draws

        style_games[matchup.style_a] += num_games
        style_games[matchup.style_b] += num_games

        total_half_moves += matchup_half_moves
        total_games += num_games

    avg_game_length = total_half_moves / total_games if total_games else 0.0

    style_win_rate: dict[int, float] = {}
    for s in config.STYLE_NAMES:
        style_win_rate[s] = (
            style_wins[s] / style_games[s] if style_games[s] else 0.0
        )

    dominant_style = max(style_wins, key=style_wins.get)  # type: ignore[arg-type]

    return {
        "matchups": matchup_stats,
        "style_wins": style_wins,
        "style_draws": style_draws,
        "style_games": style_games,
        "style_win_rate": style_win_rate,
        "avg_game_length": avg_game_length,
        "total_games": total_games,
        "dominant_style": dominant_style,
    }


# ---------------------------------------------------------------------
# Formatted printing
# ---------------------------------------------------------------------

def print_tournament_stats(results: TournamentResults) -> None:
    """Print a human-readable summary table of tournament results."""
    stats = analyze_tournament(results)

    print()
    print("=" * 70)
    print("Tournament Results Summary")
    print("=" * 70)
    print()

    # -- Per-matchup table ---------------------------------------------
    header = (
        f"{'Matchup':<36s} {'W-A':>4s} {'W-B':>4s} "
        f"{'Draw':>4s} {'Games':>5s} {'Avg Len':>7s}"
    )
    print(header)
    print("-" * 70)

    for m in stats["matchups"]:
        label = f"{m['label_a']} vs {m['label_b']}"
        print(
            f"{label:<36s} {m['wins_a']:>4d} {m['wins_b']:>4d} "
            f"{m['draws']:>4d} {m['total']:>5d} {m['avg_game_length']:>7.1f}"
        )

    print("-" * 70)
    print(f"{'Total':>36s} {'':>4s} {'':>4s} {'':>4s} "
          f"{stats['total_games']:>5d} {stats['avg_game_length']:>7.1f}")
    print()

    # -- Style dominance ranking ---------------------------------------
    print("Style Ranking (by total wins)")
    print("-" * 40)

    # Sort styles by wins descending
    ranking = sorted(
        stats["style_wins"].items(),
        key=lambda kv: kv[1],
        reverse=True,
    )

    medals = ["1st", "2nd", "3rd"]
    for rank, (style_id, wins) in enumerate(ranking):
        medal = medals[rank] if rank < len(medals) else "   "
        name = config.STYLE_NAMES[style_id]
        games = stats["style_games"][style_id]
        draws = stats["style_draws"][style_id]
        losses = games - wins - draws
        wr = stats["style_win_rate"][style_id]
        print(
            f"  {medal}  {name:<20s}  "
            f"{wins}W / {losses}L / {draws}D  "
            f"(WR: {wr:.1%})"
        )

    print()

    # -- Average game lengths ------------------------------------------
    print("Average Game Length")
    print("-" * 40)
    for m in stats["matchups"]:
        label = f"{m['label_a']} vs {m['label_b']}"
        print(f"  {label:<36s}  {m['avg_game_length']:.1f} half-moves")
    print(f"  {'Overall':<36s}  {stats['avg_game_length']:.1f} half-moves")
    print()

    # -- Dominant style callout ----------------------------------------
    dom = stats["dominant_style"]
    dom_name = config.STYLE_NAMES[dom]
    dom_wins = stats["style_wins"][dom]
    print(f"Dominant style: {dom_name} with {dom_wins} total wins")
    print()


# ---------------------------------------------------------------------
# Matplotlib visualisation
# ---------------------------------------------------------------------

def plot_style_wins(
    results: TournamentResults,
    save_path: str | None = None,
) -> None:
    """Create and save a bar chart of total wins per style.

    Parameters
    ----------
    results:
        Tournament results to visualise.
    save_path:
        File path for the saved PNG.  Defaults to
        ``config.LOGS_DIR / style_wins.png``.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib is not installed  skipping chart generation.")
        return

    stats = analyze_tournament(results)

    if save_path is None:
        save_path = os.path.join(config.LOGS_DIR, "style_wins.png")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # -- Data preparation ----------------------------------------------
    style_ids = [config.STYLE_NORMAL, config.STYLE_AGGRESSIVE, config.STYLE_DEFENSIVE]
    labels = [config.STYLE_NAMES[s].split()[-1] for s in style_ids]  # 'Normal', 
    wins = [stats["style_wins"][s] for s in style_ids]
    colors = ["#2ecc71", "#e74c3c", "#3498db"]  # green, red, blue

    # -- Create chart --------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, wins, color=colors, edgecolor="white", linewidth=1.2)

    # Value labels on top of each bar
    for bar, w in zip(bars, wins):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(w),
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=13,
        )

    ax.set_ylabel("Total Wins", fontsize=12)
    ax.set_title("Self-Play Tournament  Wins by Style", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(wins) * 1.2 if max(wins) > 0 else 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add a subtle grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"Style wins chart saved -> {save_path}")
