"""
Hybrid chess agent: neural style model + Stockfish blunder filter.

The trained model picks moves that match the player's personal style,
while Stockfish ensures no dumb/blunder moves slip through.

Strategy:
  1. Ask Stockfish for the top N moves with evaluations.
  2. Score each of those moves with the neural model (style match).
  3. Filter out any move that is a blunder (loses > threshold vs best).
  4. From the remaining, pick the move the model likes most.

Result: plays like the user, but never hangs a piece or blunders.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import chess
import chess.engine
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from model.network import ChessStyleNetwork
from model.encoding import encode_board_tensor, get_legal_move_mask, move_to_index

logger = logging.getLogger(__name__)


class HybridChessAgent:
    """Neural style model + Stockfish for blunder-free, style-aware play.

    Parameters
    ----------
    model_path : str
        Path to the trained ``.pt`` checkpoint.
    style : int
        Playing style (0=Normal, 1=Aggressive, 2=Defensive).
    stockfish_path : str
        Path to the Stockfish binary.
    stockfish_depth : int
        Search depth for Stockfish (8-12 recommended).
    blunder_threshold : int
        Max centipawn loss vs best move before a move is rejected (default 150cp).
    stockfish_multipv : int
        How many candidate moves to get from Stockfish (default 10).
    device : torch.device, optional
        Torch device for the neural model.
    """

    def __init__(
        self,
        model_path: str,
        style: int,
        stockfish_path: str,
        stockfish_depth: int = 10,
        blunder_threshold: int = 150,
        stockfish_multipv: int = 10,
        device: Optional[torch.device] = None,
    ) -> None:
        self.style = style
        self.device = device or config.DEVICE
        self.stockfish_depth = stockfish_depth
        self.blunder_threshold = blunder_threshold
        self.stockfish_multipv = stockfish_multipv

        # -- Load neural model ---
        self.model = ChessStyleNetwork.from_config()
        self._load_checkpoint(model_path)
        self.model.to(self.device)
        self.model.eval()

        # -- Start Stockfish engine ---
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.engine.configure({"Threads": 2, "Hash": 128})

        logger.info(
            "HybridChessAgent ready | style=%s depth=%d threshold=%dcp multipv=%d",
            self.style_name, stockfish_depth, blunder_threshold, stockfish_multipv,
        )

    def _load_checkpoint(self, model_path: str) -> None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)

    @staticmethod
    def _total_material(board: chess.Board) -> int:
        """Count total non-king material on the board (both sides)."""
        values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                  chess.ROOK: 5, chess.QUEEN: 9}
        total = 0
        for pt, val in values.items():
            total += len(board.pieces(pt, chess.WHITE)) * val
            total += len(board.pieces(pt, chess.BLACK)) * val
        return total

    def _get_phase_params(self, board: chess.Board) -> tuple[int, int, float]:
        """Return (depth, blunder_threshold, stockfish_weight) based on game phase.

        Endgames need more precise play (deeper search, tighter threshold,
        Stockfish eval weighted much more than the neural style score).
        """
        mat = self._total_material(board)

        if mat <= 12:
            # Deep endgame (e.g. R+P vs R, or K+P vs K)
            return 18, 30, 10.0
        elif mat <= 24:
            # Late middlegame / early endgame
            return 14, 75, 5.0
        else:
            # Opening / middlegame — style matters most
            return self.stockfish_depth, self.blunder_threshold, 1.0

    @torch.no_grad()
    def select_move(self, board: chess.Board) -> chess.Move:
        """Select a move: Stockfish-safe + style-matched.

        Pipeline:
          1. Determine game phase (opening/middle/endgame) from material count.
          2. Get Stockfish's top N moves with evaluations (deeper in endgames).
          3. Filter out blunders (tighter threshold in endgames).
          4. Score remaining moves with the neural model.
          5. Combine scores: neural style + Stockfish eval (Stockfish weighted
             more heavily in endgames where precision matters).
          6. Pick the move with highest combined score.
        """
        legal_moves = list(board.legal_moves)
        if len(legal_moves) == 1:
            return legal_moves[0]

        # --- Phase-dependent parameters ---
        depth, threshold, sf_weight = self._get_phase_params(board)

        # --- Step 1: Get Stockfish candidates ---
        try:
            analysis = self.engine.analyse(
                board,
                chess.engine.Limit(depth=depth),
                multipv=min(self.stockfish_multipv, len(legal_moves)),
            )
        except Exception as exc:
            logger.warning("Stockfish analysis failed: %s — falling back to model", exc)
            return self._model_only_move(board)

        # Extract moves and their evaluations
        candidates = []
        for info in analysis:
            move = info.get("pv", [None])[0]
            score = info.get("score")
            if move and score:
                cp = score.pov(board.turn).score(mate_score=100_000)
                candidates.append((move, cp))

        if not candidates:
            return self._model_only_move(board)

        # --- Step 2: Filter blunders ---
        best_eval = candidates[0][1]  # Stockfish's best move eval
        safe_moves = [
            (move, cp) for move, cp in candidates
            if (best_eval - cp) <= threshold
        ]

        # If all moves are "blunders" (forced position), keep the best one
        if not safe_moves:
            safe_moves = [candidates[0]]

        # --- Step 3: Score with neural model ---
        board_tensor = encode_board_tensor(board).unsqueeze(0).to(self.device)
        style_tensor = torch.tensor([self.style], dtype=torch.long, device=self.device)
        logits = self.model(board_tensor, style_tensor).squeeze(0)

        # Get model's preference for each safe move
        best_move = safe_moves[0][0]
        best_score = float("-inf")

        for move, cp in safe_moves:
            try:
                idx = move_to_index(move, board)
                model_score = logits[idx].item()
                # Combine: neural style + Stockfish eval
                # sf_weight scales with game phase (1x middle, 5-10x endgame)
                combined = model_score + sf_weight * (cp / 100.0)
                if combined > best_score:
                    best_score = combined
                    best_move = move
            except Exception:
                continue

        return best_move

    def _model_only_move(self, board: chess.Board) -> chess.Move:
        """Fallback: use the neural model alone (greedy)."""
        import random

        board_tensor = encode_board_tensor(board).unsqueeze(0).to(self.device)
        style_tensor = torch.tensor([self.style], dtype=torch.long, device=self.device)
        logits = self.model(board_tensor, style_tensor).squeeze(0)

        legal_mask = get_legal_move_mask(board).to(self.device)
        logits = logits.masked_fill(legal_mask == 0, float("-inf"))

        from model.encoding import index_to_move
        chosen_index = logits.argmax().item()
        move = index_to_move(chosen_index, board)

        if move and move in board.legal_moves:
            return move
        return random.choice(list(board.legal_moves))

    @property
    def style_name(self) -> str:
        return config.STYLE_NAMES[self.style]

    def close(self):
        """Shut down the Stockfish engine."""
        try:
            self.engine.quit()
        except Exception:
            pass

    def __del__(self):
        self.close()

    def __repr__(self) -> str:
        return (
            f"HybridChessAgent(style={self.style_name!r}, "
            f"depth={self.stockfish_depth}, "
            f"threshold={self.blunder_threshold}cp)"
        )
