"""
Parse PGN strings from Chess.com game dicts into structured records.

Each game is walked move-by-move, and for every move made by the target
player, a ``GameRecord`` is emitted containing the FEN *before* the move
and the move in UCI notation.
"""
import io
from dataclasses import dataclass
from typing import Optional

import chess
import chess.pgn
from tqdm import tqdm

import config
from utils.helpers import setup_logging

logger = setup_logging("parser", config.LOGS_DIR)


@dataclass(slots=True)
class GameRecord:
    """A single labelled (fen, move) sample extracted from a game."""

    fen: str
    """Board FEN *before* the move was played."""

    move_uci: str
    """Move in UCI format (e.g. ``'e2e4'``)."""

    player_color: chess.Color
    """``chess.WHITE`` or ``chess.BLACK``  the target player's colour."""

    move_number: int
    """Full-move number (1-based, matching PGN numbering)."""

    game_id: str
    """Opaque identifier used to group records from the same game."""

    result: str
    """Game result string: ``'1-0'``, ``'0-1'``, ``'1/2-1/2'``, or ``'*'``."""


def _game_id_from_dict(game_dict: dict, index: int) -> str:
    """Build a human-readable game ID.

    Prefers the Chess.com URL; falls back to a sequential index.
    """
    url = game_dict.get("url", "")
    if url:
        # Keep just the trailing numeric game ID from the URL
        return url.rstrip("/").rsplit("/", 1)[-1]
    return f"game_{index}"


def parse_games(
    games: list[dict],
    username: Optional[str] = None,
) -> list[GameRecord]:
    """Parse a list of Chess.com game dicts into ``GameRecord`` samples.

    Parameters
    ----------
    games : list[dict]
        Game dicts produced by :func:`data.fetcher.fetch_all_games`.
        Each must contain at least ``'pgn'``, ``'white'``, and ``'black'``.
    username : str, optional
        The target player's Chess.com username (case-insensitive).
        Defaults to ``config.CHESS_COM_USERNAME``.

    Returns
    -------
    list[GameRecord]
        One record per move made by the target player, across all games.
    """
    if username is None:
        username = config.CHESS_COM_USERNAME
    username_lower = username.lower()

    records: list[GameRecord] = []
    skipped_short = 0
    skipped_user_missing = 0
    parse_errors = 0

    for idx, game_dict in enumerate(
        tqdm(games, desc="Parsing PGNs", unit="game")
    ):
        pgn_str = game_dict.get("pgn", "")
        if not pgn_str:
            parse_errors += 1
            continue

        # --- read game ---------------------------------------------------
        try:
            pgn_io = io.StringIO(pgn_str)
            game = chess.pgn.read_game(pgn_io)
        except Exception as exc:
            logger.debug("PGN parse error in game %d: %s", idx, exc)
            parse_errors += 1
            continue

        if game is None:
            parse_errors += 1
            continue

        # --- determine player colour ------------------------------------
        white_name = str(game_dict.get("white", "")).lower()
        black_name = str(game_dict.get("black", "")).lower()

        # Also check PGN headers as fallback
        if white_name == username_lower:
            player_color = chess.WHITE
        elif black_name == username_lower:
            player_color = chess.BLACK
        else:
            # Try PGN headers
            pgn_white = game.headers.get("White", "").lower()
            pgn_black = game.headers.get("Black", "").lower()
            if pgn_white == username_lower:
                player_color = chess.WHITE
            elif pgn_black == username_lower:
                player_color = chess.BLACK
            else:
                skipped_user_missing += 1
                continue

        # --- walk through moves -----------------------------------------
        board = game.board()
        result = game.headers.get("Result", "*")
        gid = _game_id_from_dict(game_dict, idx)

        total_half_moves = 0
        game_records_buf: list[GameRecord] = []

        node = game
        while node.variations:
            next_node = node.variation(0)
            move = next_node.move
            total_half_moves += 1

            # Record only the target player's moves
            if board.turn == player_color:
                fen = board.fen()
                move_uci = move.uci()
                move_number = board.fullmove_number

                game_records_buf.append(
                    GameRecord(
                        fen=fen,
                        move_uci=move_uci,
                        player_color=player_color,
                        move_number=move_number,
                        game_id=gid,
                        result=result,
                    )
                )

            board.push(move)
            node = next_node

        # Skip aborted games (fewer than 5 total half-moves)
        if total_half_moves < 5:
            skipped_short += 1
            continue

        records.extend(game_records_buf)

    logger.info(
        "Parsing complete  %d records from %d games "
        "(skipped: %d short, %d user-missing, %d parse-errors)",
        len(records),
        len(games),
        skipped_short,
        skipped_user_missing,
        parse_errors,
    )
    return records
