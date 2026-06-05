"""
Book Agent – mimic a Chess.com player's opening/middlegame tendencies.

PlayerBook  fetches a player's games, builds a position → move frequency map.
BookAgent   plays historical moves when the position is in the book, otherwise
            falls back to Stockfish (strength-clamped to the player's rating).
"""

import os, sys, logging, random
import chess, chess.engine
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from data.fetcher import fetch_all_games
from data.parser import parse_games

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PlayerBook
# ---------------------------------------------------------------------------

class PlayerBook:
    """Position → move frequency map built from a Chess.com player's games."""

    def __init__(self, username: str) -> None:
        self.username = username
        self.book: dict[str, dict[str, int]] = {}
        self.rating: int | None = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> dict:
        """Fetch games, parse them, and populate the book.

        Returns a stats dict: {"games", "positions", "username", "rating"}.
        """
        # 1. Fetch archive list from Chess.com
        archives_url = f"https://api.chess.com/pub/player/{self.username}/games/archives"
        headers = {"User-Agent": config.API_USER_AGENT}

        resp = requests.get(archives_url, headers=headers, timeout=30)
        resp.raise_for_status()
        archives = resp.json().get("archives", [])
        logger.info("Found %d archives for '%s'", len(archives), self.username)

        # 2. Fetch all games
        games = fetch_all_games(archives, self.username)
        logger.info("Fetched %d games for '%s'", len(games), self.username)

        # 3. Parse into GameRecord objects
        records = parse_games(games, self.username)
        logger.info("Parsed %d position records", len(records))

        # 4. Build the book: fen_key → {move_uci: count}
        self.book.clear()
        for rec in records:
            key = self._fen_key(rec.fen)
            if key not in self.book:
                self.book[key] = {}
            move = rec.move_uci
            self.book[key][move] = self.book[key].get(move, 0) + 1

        # 5. Try to fetch the player's rating
        self.rating = self._fetch_rating(headers)

        stats = {
            "games": len(games),
            "positions": len(self.book),
            "username": self.username,
            "rating": self.rating,
        }
        logger.info("Book built: %s", stats)
        return stats

    # ------------------------------------------------------------------
    # Move selection
    # ------------------------------------------------------------------

    def get_move(self, board: chess.Board) -> chess.Move | None:
        """Return a weighted-random historical move, or None if not in book."""
        key = self._fen_key(board.fen())
        entry = self.book.get(key)
        if not entry:
            return None

        # Weighted random selection based on move frequency
        moves_uci = list(entry.keys())
        weights = list(entry.values())
        chosen_uci = random.choices(moves_uci, weights=weights, k=1)[0]

        try:
            move = chess.Move.from_uci(chosen_uci)
        except ValueError:
            return None

        # Validate the move is legal in the current position
        if move in board.legal_moves:
            return move

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fen_key(fen: str) -> str:
        """Return the first 4 space-separated parts of a FEN string.

        This captures piece placement, active colour, castling rights,
        and en-passant square – enough to identify a position without
        half-move / full-move clocks.
        """
        return " ".join(fen.split()[:4])

    def _fetch_rating(self, headers: dict) -> int | None:
        """Try to get the player's rapid or blitz rating from Chess.com."""
        stats_url = f"https://api.chess.com/pub/player/{self.username}/stats"
        try:
            resp = requests.get(stats_url, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            # Try rapid first, then blitz
            for category in ("chess_rapid", "chess_blitz"):
                cat_data = data.get(category, {})
                last = cat_data.get("last", {})
                rating = last.get("rating")
                if rating is not None:
                    logger.info("Found %s rating: %d", category, rating)
                    return int(rating)

            logger.info("No rapid/blitz rating found for '%s'", self.username)
            return None
        except Exception as exc:
            logger.warning("Failed to fetch rating for '%s': %s", self.username, exc)
            return None


# ---------------------------------------------------------------------------
# BookAgent
# ---------------------------------------------------------------------------

class BookAgent:
    """Chess agent that plays a real player's moves, falling back to Stockfish."""

    def __init__(self, player_book: PlayerBook, stockfish_path: str) -> None:
        self.book = player_book

        # Start Stockfish
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

        # Configure Stockfish strength based on the player's rating
        if self.book.rating is not None:
            clamped_elo = max(1320, min(3190, self.book.rating))
            self.engine.configure({
                "UCI_LimitStrength": True,
                "UCI_Elo": clamped_elo,
            })
            logger.info("Stockfish ELO clamped to %d (player rating: %d)",
                        clamped_elo, self.book.rating)
        else:
            self.engine.configure({"Skill Level": 10})
            logger.info("No rating available; Stockfish Skill Level set to 10")

    def select_move(self, board: chess.Board) -> chess.Move:
        """Select a move: book first, then Stockfish fallback."""
        # Try the opening book
        book_move = self.book.get_move(board)
        if book_move is not None:
            return book_move

        # Fallback to Stockfish
        result = self.engine.play(board, chess.engine.Limit(time=0.5))
        return result.move

    def close(self) -> None:
        """Shut down the Stockfish engine."""
        try:
            self.engine.quit()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()
