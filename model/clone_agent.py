"""
CloneAgent – play like a real Chess.com player using their archive games.

Architecture:
  1. PlayerBook   – exact FEN → player's historical moves (weighted random).
  2. NeuralFallback – ResNet policy network scores all legal moves.
  3. TacticalFilter – SEE-based blunder detection (no Stockfish required).

Flow: book hit → play it.
      book miss → neural top-K candidates → filter blunders → best safe move.
"""

from __future__ import annotations

import os
import sys
import logging
import random
from typing import Optional

import chess
import requests
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from data.fetcher import fetch_all_games
from data.parser import parse_games

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Piece values for SEE (centipawns → simple integer)
# ---------------------------------------------------------------------------
_SEE_VALUE = {
    chess.PAWN:   1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK:   5,
    chess.QUEEN:  9,
    chess.KING:   100,
}


# ---------------------------------------------------------------------------
# PlayerBook  (position → move frequency)
# ---------------------------------------------------------------------------

class PlayerBook:
    """Builds a FEN-key → {move_uci: count} map from a Chess.com player's games."""

    def __init__(self, username: str) -> None:
        self.username = username
        self.book: dict[str, dict[str, int]] = {}
        self.rating: Optional[int] = None

    # ------------------------------------------------------------------
    def build(self) -> dict:
        """Fetch games, parse them, and populate the book. Returns stats dict."""
        headers = {"User-Agent": config.API_USER_AGENT}

        # 1. Archives list
        archives_url = f"https://api.chess.com/pub/player/{self.username}/games/archives"
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

        # 4. Build book
        self.book.clear()
        for rec in records:
            key = self._fen_key(rec.fen)
            entry = self.book.setdefault(key, {})
            entry[rec.move_uci] = entry.get(rec.move_uci, 0) + 1

        # 5. Rating
        self.rating = self._fetch_rating(headers)

        stats = {
            "games":     len(games),
            "positions": len(self.book),
            "username":  self.username,
            "rating":    self.rating,
        }
        logger.info("Book built: %s", stats)
        return stats

    def get_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Weighted-random historical move, or None if position not in book."""
        key = self._fen_key(board.fen())
        entry = self.book.get(key)
        if not entry:
            return None

        uci_list = list(entry.keys())
        weights  = list(entry.values())
        chosen   = random.choices(uci_list, weights=weights, k=1)[0]

        try:
            move = chess.Move.from_uci(chosen)
        except ValueError:
            return None

        return move if move in board.legal_moves else None

    @staticmethod
    def _fen_key(fen: str) -> str:
        """First 4 FEN fields (position, turn, castling, en passant)."""
        return " ".join(fen.split()[:4])

    def _fetch_rating(self, headers: dict) -> Optional[int]:
        stats_url = f"https://api.chess.com/pub/player/{self.username}/stats"
        try:
            resp = requests.get(stats_url, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            for cat in ("chess_rapid", "chess_blitz", "chess_bullet"):
                rating = data.get(cat, {}).get("last", {}).get("rating")
                if rating is not None:
                    logger.info("Found %s rating: %d", cat, rating)
                    return int(rating)
            return None
        except Exception as exc:
            logger.warning("Failed to fetch rating: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Tactical safety check (SEE-based, no engine needed)
# ---------------------------------------------------------------------------

def _see(board: chess.Board, square: int, attacker_color: chess.Color) -> int:
    """
    Simplified Static Exchange Evaluation.
    Returns the material gain the attacker can expect by capturing on `square`.
    Positive = good for attacker.
    """
    # Find cheapest attacker of the given color
    attackers = board.attackers(attacker_color, square)
    if not attackers:
        return 0

    # Pick the least valuable attacker
    least_sq = min(attackers, key=lambda s: _SEE_VALUE[board.piece_at(s).piece_type])
    captured_piece = board.piece_at(square)
    if captured_piece is None:
        return 0

    capture_gain = _SEE_VALUE[captured_piece.piece_type]

    # Simulate the capture
    board.push(chess.Move(least_sq, square))
    # Recursively: opponent re-captures
    recapture_loss = _see(board, square, not attacker_color)
    board.pop()

    return max(0, capture_gain - recapture_loss)


def is_tactically_safe(board: chess.Board, move: chess.Move) -> bool:
    """
    Returns True if `move` does NOT immediately blunder material.

    A move is considered a blunder if, after making it, any of our pieces
    can be captured for a net material gain by the opponent.

    We skip pawn captures of pawns (common tactical patterns)
    and allow trading equal pieces.
    """
    board.push(move)
    us   = not board.turn   # we just moved
    them = board.turn

    safe = True
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None or piece.color != us:
            continue
        if piece.piece_type == chess.KING:
            continue

        if board.is_attacked_by(them, sq):
            gain = _see(board, sq, them)
            # Opponent can win material by capturing here
            if gain > 0:
                safe = False
                break

    board.pop()
    return safe


# ---------------------------------------------------------------------------
# Neural policy fallback
# ---------------------------------------------------------------------------

class NeuralPolicy:
    """
    Loads the trained ChessStyleNetwork and scores legal moves.
    Silently disabled if the checkpoint is missing.
    """

    def __init__(self, model_path: str) -> None:
        self.enabled = False
        self.model   = None
        self.device  = torch.device(config.DEVICE if hasattr(config, "DEVICE") else "cpu")

        if not os.path.exists(model_path):
            logger.warning("Neural model not found at %s — using book-only fallback", model_path)
            return

        try:
            from model.network import ChessStyleNetwork
            from model.encoding import move_to_index

            self._move_to_index = move_to_index
            model = ChessStyleNetwork.from_config()
            ckpt  = torch.load(model_path, map_location=self.device, weights_only=False)

            # Support both raw state_dict and wrapped checkpoints
            state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(state, strict=False)
            model.to(self.device)
            model.eval()

            self.model   = model
            self.enabled = True
            logger.info("Neural policy loaded from %s", model_path)
        except Exception as exc:
            logger.warning("Could not load neural model: %s — using book-only fallback", exc)

    def score_moves(self, board: chess.Board) -> list[tuple[chess.Move, float]]:
        """
        Returns a list of (move, score) for all legal moves, sorted descending.
        Falls back to random uniform scores if the model is not available.
        """
        legal = list(board.legal_moves)

        if not self.enabled or self.model is None:
            # No model: uniform random scores
            return [(m, random.random()) for m in legal]

        from model.encoding import encode_board_tensor

        try:
            board_t  = encode_board_tensor(board).unsqueeze(0).to(self.device)
            style_t  = torch.zeros(1, dtype=torch.long, device=self.device)  # "Normal" style

            with torch.no_grad():
                logits = self.model(board_t, style_t).squeeze(0)   # (4672,)

            # Mask illegal moves
            scored = []
            for move in legal:
                try:
                    idx   = self._move_to_index(move, board)
                    score = logits[idx].item()
                except Exception:
                    score = -1e9
                scored.append((move, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            return scored

        except Exception as exc:
            logger.warning("Neural scoring failed: %s", exc)
            return [(m, random.random()) for m in legal]


# ---------------------------------------------------------------------------
# CloneAgent  (main entry point)
# ---------------------------------------------------------------------------

class CloneAgent:
    """
    Play like a specific Chess.com player.

    Priority order:
      1. Book move  – their actual historical move from this exact position.
      2. Neural move – best safe move from the policy network (no Stockfish).
      3. Forced     – if all moves are "blunders", pick the best-scoring one.
    """

    TOP_K = 5   # neural candidates to evaluate for tactical safety

    def __init__(self, player_book: PlayerBook, model_path: str) -> None:
        self.book = player_book

        # Look for a per-user trained model first
        username    = player_book.username.lower()
        user_model  = os.path.join(os.path.dirname(model_path), f"clone_{username}.pt")

        if os.path.exists(user_model):
            logger.info("Using per-user model: %s", user_model)
            chosen_path = user_model
        else:
            logger.info("No per-user model found; using generic model: %s", model_path)
            chosen_path = model_path

        self.neural = NeuralPolicy(chosen_path)

    def select_move(self, board: chess.Board) -> chess.Move:
        # ── 1. Book lookup ──────────────────────────────────────────────
        book_move = self.book.get_move(board)
        if book_move is not None:
            logger.debug("Book move: %s", board.san(book_move))
            return book_move

        # ── 2. Neural scoring ───────────────────────────────────────────
        scored = self.neural.score_moves(board)

        if not scored:
            return next(iter(board.legal_moves))

        # ── 3. Tactical filter on top-K candidates ─────────────────────
        top_k = [m for m, _ in scored[: self.TOP_K]]
        safe  = [m for m in top_k if is_tactically_safe(board, m)]

        if safe:
            chosen = safe[0]
            logger.debug("Neural (safe, top-%d): %s", len(top_k), board.san(chosen))
            return chosen

        # ── 4. Expand search: filter all legal moves ────────────────────
        safe_all = [m for m, _ in scored if is_tactically_safe(board, m)]
        if safe_all:
            chosen = safe_all[0]
            logger.debug("Neural (safe, all legal): %s", board.san(chosen))
            return chosen

        # ── 5. Forced / zugzwang – just play the highest-scored move ────
        chosen = scored[0][0]
        logger.debug("Forced move (no safe option): %s", board.san(chosen))
        return chosen
