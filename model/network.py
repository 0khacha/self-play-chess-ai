"""
Style-conditioned residual CNN for chess move prediction.

Architecture:
    1. Style embedding:  (batch,) int  ->  (batch, style_embed_dim, 8, 8)
    2. Concatenation:    board (batch, 18, 8, 8) + style -> (batch, 50, 8, 8)
    3. Initial conv:     50 -> 128 channels, 3x3, pad 1, BN, ReLU
    4. Residual tower:   6 x ResidualBlock (128 -> 128)
    5. Policy head:      Conv 128->32, 1x1, BN, ReLU -> FC(2048 -> 4672)
"""
from __future__ import annotations

import sys
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow importing config from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class ResidualBlock(nn.Module):
    """
    A single residual block with two convolutional layers and a skip connection.

    Structure:
        x -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+x) -> ReLU
    """

    def __init__(self, num_filters: int = config.NUM_FILTERS) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.

        Args:
            x: Input tensor of shape (batch, num_filters, 8, 8).

        Returns:
            Output tensor of the same shape with the skip connection applied.
        """
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = F.relu(out)
        return out


class ChessStyleNetwork(nn.Module):
    """
    Style-conditioned residual CNN that predicts a policy (move distribution)
    given a board state and a playing-style token.

    Args:
        num_planes:      Number of input board planes (default 18).
        num_styles:      Number of distinct playing styles (default 3).
        style_embed_dim: Dimensionality of the style embedding vector (default 32).
        num_filters:     Number of convolutional filters in the residual tower (default 128).
        num_res_blocks:  Number of residual blocks in the tower (default 6).
        policy_filters:  Number of filters in the policy-head 1x1 conv (default 32).
        move_vocab_size: Size of the move vocabulary / output logits (default 4672).
    """

    def __init__(
        self,
        num_planes: int = config.NUM_BOARD_PLANES,
        num_styles: int = config.NUM_STYLES,
        style_embed_dim: int = config.STYLE_EMBED_DIM,
        num_filters: int = config.NUM_FILTERS,
        num_res_blocks: int = config.NUM_RESIDUAL_BLOCKS,
        policy_filters: int = config.POLICY_HEAD_FILTERS,
        move_vocab_size: int = config.MOVE_VOCAB_SIZE,
    ) -> None:
        super().__init__()

        self.num_planes = num_planes
        self.num_styles = num_styles
        self.style_embed_dim = style_embed_dim
        self.num_filters = num_filters
        self.num_res_blocks = num_res_blocks
        self.policy_filters = policy_filters
        self.move_vocab_size = move_vocab_size

        # -- Style embedding --------------------------------------
        self.style_embedding = nn.Embedding(num_styles, style_embed_dim)

        # -- Initial convolution (board planes + style planes -> filters) --
        in_channels = num_planes + style_embed_dim  # 18 + 32 = 50
        self.input_conv = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(num_filters)

        # -- Residual tower ---------------------------------------
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # -- Policy head ------------------------------------------
        self.policy_conv = nn.Conv2d(num_filters, policy_filters, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_filters)
        self.policy_fc = nn.Linear(
            policy_filters * config.BOARD_SIZE * config.BOARD_SIZE,  # 32 * 8 * 8 = 2048
            move_vocab_size,
        )

    def forward(self, board_tensor: torch.Tensor, style_id: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: board representation + style -> move logits.

        Args:
            board_tensor: Float tensor of shape (batch, num_planes, 8, 8).
            style_id:     Long tensor of shape (batch,) with style indices in [0, num_styles).

        Returns:
            Policy logits of shape (batch, move_vocab_size).
        """
        batch_size = board_tensor.size(0)

        # Style embedding: (batch,) -> (batch, style_embed_dim)
        style_embed = self.style_embedding(style_id)

        # Broadcast style embedding spatially: -> (batch, style_embed_dim, 8, 8)
        style_planes = style_embed.unsqueeze(-1).unsqueeze(-1).expand(
            batch_size, self.style_embed_dim, config.BOARD_SIZE, config.BOARD_SIZE
        )

        # Concatenate board planes with style planes along channel dim
        # (batch, num_planes, 8, 8) + (batch, style_embed_dim, 8, 8) -> (batch, 50, 8, 8)
        x = torch.cat([board_tensor, style_planes], dim=1)

        # Initial convolution block
        x = F.relu(self.input_bn(self.input_conv(x)))

        # Residual tower
        x = self.residual_tower(x)

        # Policy head
        x = F.relu(self.policy_bn(self.policy_conv(x)))
        x = x.view(batch_size, -1)  # flatten to (batch, policy_filters * 8 * 8)
        logits = self.policy_fc(x)

        return logits

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters in the network."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls) -> "ChessStyleNetwork":
        """
        Create a ChessStyleNetwork with all hyperparameters sourced from the
        project config module.

        Returns:
            A new ChessStyleNetwork instance with default config values.
        """
        return cls(
            num_planes=config.NUM_BOARD_PLANES,
            num_styles=config.NUM_STYLES,
            style_embed_dim=config.STYLE_EMBED_DIM,
            num_filters=config.NUM_FILTERS,
            num_res_blocks=config.NUM_RESIDUAL_BLOCKS,
            policy_filters=config.POLICY_HEAD_FILTERS,
            move_vocab_size=config.MOVE_VOCAB_SIZE,
        )

    def __repr__(self) -> str:
        return (
            f"ChessStyleNetwork("
            f"planes={self.num_planes}, "
            f"styles={self.num_styles}, "
            f"embed={self.style_embed_dim}, "
            f"filters={self.num_filters}, "
            f"res_blocks={self.num_res_blocks}, "
            f"policy_filters={self.policy_filters}, "
            f"vocab={self.move_vocab_size}, "
            f"params={self.count_parameters():,})"
        )
