"""
Chess agent that uses the trained style-conditioned network to select moves.

Supports temperature-controlled sampling (stochastic play) and greedy
selection (temperature -> 0). Illegal moves are masked before sampling.
"""
from __future__ import annotations

import logging
import os
import random
import sys
from typing import Optional

import chess
import torch
import torch.nn.functional as F

# Allow importing config and sibling modules from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from model.network import ChessStyleNetwork
from model.encoding import encode_board_tensor, get_legal_move_mask, index_to_move

logger = logging.getLogger(__name__)


class ChessAgent:
    """
    An agent that plays chess by querying a trained :class:`ChessStyleNetwork`.

    Args:
        model_path:  Path to the saved model checkpoint (``.pt`` file).
        style:       Playing-style index (0 = Normal, 1 = Aggressive, 2 = Defensive).
        device:      Torch device to run inference on.  Falls back to ``config.DEVICE``.
        temperature: Softmax temperature for move sampling.
                     Values near 0 -> greedy (argmax); higher -> more exploration.
    """

    # Moves with probability below this threshold are treated as near-zero
    _GREEDY_THRESHOLD: float = 1e-4

    def __init__(
        self,
        model_path: str,
        style: int,
        device: Optional[torch.device] = None,
        temperature: float = 0.8,
    ) -> None:
        if style not in config.STYLE_NAMES:
            raise ValueError(
                f"Invalid style {style}. Must be one of {list(config.STYLE_NAMES.keys())}."
            )

        self.style = style
        self.device = device if device is not None else config.DEVICE
        self.temperature = temperature

        # -- Load the model ---------------------------------------
        self.model = ChessStyleNetwork.from_config()
        self._load_checkpoint(model_path)
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "ChessAgent ready  |  style=%s  device=%s  temperature=%.2f  params=%s",
            self.style_name,
            self.device,
            self.temperature,
            f"{self.model.count_parameters():,}",
        )

    # ----------------------------------------------------------
    # Checkpoint loading
    # ----------------------------------------------------------

    def _load_checkpoint(self, model_path: str) -> None:
        """
        Load model weights from a checkpoint file.

        The checkpoint may be a raw ``state_dict`` or a dict containing a
        ``"model_state_dict"`` key (as saved by typical training loops).

        Args:
            model_path: Absolute or relative path to the ``.pt`` checkpoint.

        Raises:
            FileNotFoundError: If *model_path* does not exist.
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            # Assume the file *is* the state dict directly
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        logger.info("Loaded checkpoint from %s", model_path)

    # ----------------------------------------------------------
    # Move selection
    # ----------------------------------------------------------

    @torch.no_grad()
    def select_move(self, board: chess.Board) -> chess.Move:
        """
        Select a move for the current board position.

        Pipeline:
            1. Encode the board -> (1, 18, 8, 8) tensor.
            2. Forward pass with the style token -> (1, 4672) logits.
            3. Mask illegal moves (set logits to -).
            4. Apply temperature scaling.
            5. Convert to probabilities via softmax and sample.
            6. Map the sampled index back to a ``chess.Move``.
            7. Validate legality; fall back to a random legal move if needed.

        Args:
            board: The current chess position.

        Returns:
            The chosen ``chess.Move``.
        """
        # 1. Encode board -> (1, 18, 8, 8)
        board_tensor = encode_board_tensor(board).unsqueeze(0).to(self.device)

        # 2. Style tensor -> (1,)
        style_tensor = torch.tensor([self.style], dtype=torch.long, device=self.device)

        # 3. Forward pass -> logits (1, 4672)
        logits = self.model(board_tensor, style_tensor).squeeze(0)  # (4672,)

        # 4. Mask illegal moves with -inf
        legal_mask = get_legal_move_mask(board).to(self.device)  # (4672,)
        logits = logits.masked_fill(legal_mask == 0, float("-inf"))

        # 5. Temperature scaling
        if self.temperature < self._GREEDY_THRESHOLD:
            # Greedy: pick the move with the highest logit
            chosen_index = logits.argmax().item()
        else:
            scaled_logits = logits / self.temperature
            probs = F.softmax(scaled_logits, dim=0)

            # Sample from the distribution
            chosen_index = torch.multinomial(probs, num_samples=1).item()

        # 6. Map index -> chess.Move
        move = index_to_move(chosen_index, board)

        # 7. Validate legality (should always pass, but safety first)
        if move is not None and move in board.legal_moves:
            return move

        # Fallback: pick a random legal move (should be unreachable)
        logger.warning(
            "Sampled move index %d mapped to invalid/illegal move %s  "
            "falling back to random legal move. FEN: %s",
            chosen_index,
            move,
            board.fen(),
        )
        return random.choice(list(board.legal_moves))

    # ----------------------------------------------------------
    # Properties
    # ----------------------------------------------------------

    @property
    def style_name(self) -> str:
        """Human-readable name (with emoji) for the current playing style."""
        return config.STYLE_NAMES[self.style]

    def __repr__(self) -> str:
        return (
            f"ChessAgent(style={self.style_name!r}, "
            f"device={self.device}, "
            f"temperature={self.temperature:.2f})"
        )
