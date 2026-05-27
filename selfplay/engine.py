"""
Single game engine for self-play between two ChessAgent instances.

Handles move alternation, legal move enforcement, all standard
game-ending conditions, PGN generation, and move logging.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import chess
import chess.pgn

import config

if TYPE_CHECKING:
    from model.inference import ChessAgent


@dataclass
class GameResult:
    """Container for the outcome and metadata of a single self-play game."""

    pgn_game: chess.pgn.Game
    result: str  # '1-0', '0-1', '1/2-1/2'
    move_count: int  # total half-moves played
    white_style: str  # e.g., ' Normal'
    black_style: str
    move_log: list[tuple[int, str, str]] = field(default_factory=list)
    termination: str = ""  # e.g., 'checkmate', 'stalemate', 


def play_game(
    white_agent: "ChessAgent",
    black_agent: "ChessAgent",
    max_moves: int | None = None,
) -> GameResult:
    """Play a single chess game between *white_agent* and *black_agent*.

    Parameters
    ----------
    white_agent:
        Agent controlling the white pieces.
    black_agent:
        Agent controlling the black pieces.
    max_moves:
        Maximum number of **full** moves (per side).  When ``None``,
        defaults to ``config.SELFPLAY_MAX_MOVES``.

    Returns
    -------
    GameResult
        Dataclass capturing the PGN, result string, termination reason,
        move log, and style metadata.
    """
    if max_moves is None:
        max_moves = config.SELFPLAY_MAX_MOVES

    # Maximum *half-moves* before we declare a draw by move limit.
    max_half_moves: int = max_moves * 2

    # -- Initialise board and PGN --------------------------------------
    board = chess.Board()

    pgn_game = chess.pgn.Game()
    pgn_game.headers["Event"] = "Self-Play Tournament"
    pgn_game.headers["Site"] = "SelfPlayChessAI"
    pgn_game.headers["Date"] = datetime.date.today().strftime("%Y.%m.%d")
    pgn_game.headers["Round"] = "-"
    pgn_game.headers["White"] = white_agent.style_name
    pgn_game.headers["Black"] = black_agent.style_name
    pgn_game.headers["Result"] = "*"  # placeholder until game ends

    # The PGN node pointer  initially the game root.
    node: chess.pgn.GameNode = pgn_game

    move_log: list[tuple[int, str, str]] = []
    half_move_count: int = 0
    result: str = "*"
    termination: str = ""

    # -- Game loop -----------------------------------------------------
    while not board.is_game_over(claim_draw=False):
        # Determine active agent
        if board.turn == chess.WHITE:
            active_agent = white_agent
            color_label = "white"
        else:
            active_agent = black_agent
            color_label = "black"

        # Full-move number (1-indexed, increments after Black's move)
        full_move_number = board.fullmove_number

        # -- Select and apply move -------------------------------------
        move: chess.Move = active_agent.select_move(board)

        # Safety: ensure legality (should always be legal, but guard)
        if move not in board.legal_moves:
            # If the agent returns an illegal move, pick a random legal one.
            move = list(board.legal_moves)[0]

        # Append to PGN tree
        node = node.add_variation(move)

        # Record in move log
        move_log.append((full_move_number, color_label, move.uci()))

        # Push onto the internal board
        board.push(move)
        half_move_count += 1

        # -- Check game-ending conditions ------------------------------
        if board.is_checkmate():
            # The side that just moved delivered checkmate.
            result = "1-0" if board.turn == chess.BLACK else "0-1"
            termination = "checkmate"
            break

        if board.is_stalemate():
            result = "1/2-1/2"
            termination = "stalemate"
            break

        if board.is_insufficient_material():
            result = "1/2-1/2"
            termination = "insufficient material"
            break

        if board.can_claim_threefold_repetition():
            result = "1/2-1/2"
            termination = "draw by repetition"
            break

        if board.can_claim_fifty_moves():
            result = "1/2-1/2"
            termination = "draw by 50-move rule"
            break

        if half_move_count >= max_half_moves:
            result = "1/2-1/2"
            termination = "draw by move limit"
            break

    # Handle python-chess native game-over (e.g. five-fold repetition,
    # 75-move rule) that can bypass the claim flags above.
    if result == "*":
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            if outcome.winner is None:
                result = "1/2-1/2"
                termination = outcome.termination.name.lower().replace("_", " ")
            elif outcome.winner == chess.WHITE:
                result = "1-0"
                termination = outcome.termination.name.lower().replace("_", " ")
            else:
                result = "0-1"
                termination = outcome.termination.name.lower().replace("_", " ")
        else:
            # Absolute fallback  should not happen.
            result = "1/2-1/2"
            termination = "unknown"

    # -- Finalise PGN --------------------------------------------------
    pgn_game.headers["Result"] = result
    pgn_game.headers["Termination"] = termination

    # Add termination as a comment on the last move node.
    if node is not pgn_game:
        node.comment = termination

    return GameResult(
        pgn_game=pgn_game,
        result=result,
        move_count=half_move_count,
        white_style=white_agent.style_name,
        black_style=black_agent.style_name,
        move_log=move_log,
        termination=termination,
    )
