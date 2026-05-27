"""
Round-robin tournament manager for self-play between style agents.

Creates three ChessAgent instances (Normal, Aggressive, Defensive) from the
same trained model and runs every pairwise matchup with each side playing
both colors for *num_games* games.
"""
from __future__ import annotations

import os
import datetime
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import config
from model.inference import ChessAgent
from selfplay.engine import play_game, GameResult
from utils.helpers import append_pgn, setup_logging

if TYPE_CHECKING:
    pass  # all runtime imports above

logger = setup_logging("tournament", config.LOGS_DIR)

# -- Matchup pairs (style-id tuples) ----------------------------------
_MATCHUP_PAIRS: list[tuple[int, int]] = [
    (config.STYLE_NORMAL, config.STYLE_AGGRESSIVE),
    (config.STYLE_AGGRESSIVE, config.STYLE_DEFENSIVE),
    (config.STYLE_DEFENSIVE, config.STYLE_NORMAL),
]


# -- Dataclasses ------------------------------------------------------
@dataclass
class MatchupResult:
    """Aggregated results for one pair of styles across all games."""

    style_a: int
    style_b: int
    games: list[GameResult] = field(default_factory=list)
    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0


@dataclass
class TournamentResults:
    """Container for a full round-robin tournament."""

    matchups: list[MatchupResult] = field(default_factory=list)
    total_games: int = 0


# -- Helpers ----------------------------------------------------------
def _style_label(style_id: int) -> str:
    """Return the display name for a style id (e.g. ' Normal')."""
    return config.STYLE_NAMES.get(style_id, f"Style {style_id}")


def _record_result(
    matchup: MatchupResult,
    game_result: GameResult,
    *,
    white_is_a: bool,
) -> None:
    """Tally a single game result into *matchup* counters."""
    r = game_result.result
    if r == "1/2-1/2":
        matchup.draws += 1
    elif r == "1-0":
        # White won
        if white_is_a:
            matchup.wins_a += 1
        else:
            matchup.wins_b += 1
    elif r == "0-1":
        # Black won
        if white_is_a:
            matchup.wins_b += 1
        else:
            matchup.wins_a += 1


# -- Main entry point ------------------------------------------------
def run_tournament(
    model_path: str | None = None,
    num_games: int | None = None,
    temperature: float | None = None,
) -> TournamentResults:
    """Execute a full round-robin self-play tournament.

    Parameters
    ----------
    model_path:
        Path to the trained model checkpoint.  Defaults to
        ``config.MODELS_DIR / config.CHECKPOINT_NAME``.
    num_games:
        Number of games *per color* for each matchup (total per matchup
        is ``2 x num_games``).  Defaults to
        ``config.SELFPLAY_GAMES_PER_MATCHUP``.
    temperature:
        Sampling temperature for move selection.  Defaults to
        ``config.SELFPLAY_TEMPERATURE``.

    Returns
    -------
    TournamentResults
        Full results with per-matchup breakdowns.
    """
    # -- Resolve defaults ----------------------------------------------
    if model_path is None:
        model_path = os.path.join(config.MODELS_DIR, config.CHECKPOINT_NAME)
    if num_games is None:
        num_games = config.SELFPLAY_GAMES_PER_MATCHUP
    if temperature is None:
        temperature = config.SELFPLAY_TEMPERATURE

    # -- Build agents --------------------------------------------------
    agents: dict[int, ChessAgent] = {}
    for style_id in (config.STYLE_NORMAL, config.STYLE_AGGRESSIVE, config.STYLE_DEFENSIVE):
        agents[style_id] = ChessAgent(
            model_path=model_path,
            style=style_id,
            device=config.DEVICE,
            temperature=temperature,
        )
        logger.info("Created agent: %s", _style_label(style_id))

    # -- Tournament header ---------------------------------------------
    total_matchup_games = num_games * 2  # each pair plays num_games per color
    total_games = len(_MATCHUP_PAIRS) * total_matchup_games

    print()
    print("=" * 60)
    print("Self-Play Round-Robin Tournament")
    print(f"   Model  : {model_path}")
    print(f"   Games  : {num_games} per color x {len(_MATCHUP_PAIRS)} matchups "
          f"= {total_games} total")
    print(f"   Temp   : {temperature}")
    print(f"   Date   : {datetime.date.today().isoformat()}")
    print("=" * 60)
    print()

    tournament = TournamentResults()
    game_counter = 0

    # -- Iterate over matchup pairs ------------------------------------
    for pair_idx, (style_a, style_b) in enumerate(_MATCHUP_PAIRS, start=1):
        label_a = _style_label(style_a)
        label_b = _style_label(style_b)

        print(f"-- Matchup {pair_idx}/{len(_MATCHUP_PAIRS)}: "
              f"{label_a} vs {label_b} --")

        matchup = MatchupResult(style_a=style_a, style_b=style_b)

        # PGN file for this matchup
        safe_a = config.STYLE_NAMES[style_a].split()[-1]  # e.g. 'Normal'
        safe_b = config.STYLE_NAMES[style_b].split()[-1]
        pgn_filename = f"{safe_a}_vs_{safe_b}_{datetime.date.today().isoformat()}.pgn"
        pgn_path = os.path.join(config.GAMES_DIR, pgn_filename)

        # Clear the file if it already exists (fresh tournament)
        if os.path.exists(pgn_path):
            os.remove(pgn_path)

        # -- Phase 1: style_a as White
        print(f"   [White: {label_a}]  [Black: {label_b}]   ", end="", flush=True)
        for g in range(1, num_games + 1):
            game_counter += 1
            game_result = play_game(
                white_agent=agents[style_a],
                black_agent=agents[style_b],
            )
            matchup.games.append(game_result)
            _record_result(matchup, game_result, white_is_a=True)
            append_pgn(game_result.pgn_game, pgn_path)

            # Progress dot
            _print_result_dot(game_result.result, is_white_a=True)

        print()  # newline after dots

        # -- Phase 2: style_b as White
        print(f"   [White: {label_b}]  [Black: {label_a}]   ", end="", flush=True)
        for g in range(1, num_games + 1):
            game_counter += 1
            game_result = play_game(
                white_agent=agents[style_b],
                black_agent=agents[style_a],
            )
            matchup.games.append(game_result)
            _record_result(matchup, game_result, white_is_a=False)
            append_pgn(game_result.pgn_game, pgn_path)

            _print_result_dot(game_result.result, is_white_a=False)

        print()  # newline after dots

        # -- Matchup summary
        total_m = matchup.wins_a + matchup.wins_b + matchup.draws
        print(f"   {label_a}: {matchup.wins_a}W  |  "
              f"{label_b}: {matchup.wins_b}W  |  "
              f"Draws: {matchup.draws}  ({total_m} games)")
        print(f"   PGN saved -> {pgn_path}")
        print()

        logger.info(
            "Matchup %s vs %s complete: %dW-%dW-%dD",
            label_a, label_b,
            matchup.wins_a, matchup.wins_b, matchup.draws,
        )

        tournament.matchups.append(matchup)

    tournament.total_games = game_counter

    # -- Final banner
    print("=" * 60)
    print(f"Tournament complete -- {tournament.total_games} games played.")
    print("=" * 60)
    print()

    return tournament


def _print_result_dot(result: str, *, is_white_a: bool) -> None:
    """Print a single character for quick visual progress.

    W = style_a win, L = style_b win, D = draw.
    """
    if result == "1/2-1/2":
        print("D", end="", flush=True)
    elif result == "1-0":
        # White won
        print("W" if is_white_a else "L", end="", flush=True)
    elif result == "0-1":
        # Black won
        print("L" if is_white_a else "W", end="", flush=True)
