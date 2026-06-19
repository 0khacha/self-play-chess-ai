"""
Style-conditioned residual CNN with Squeeze-and-Excitation attention
for chess move prediction.

Architecture:
    1. Style embedding:  (batch,) int  ->  (batch, style_embed_dim, 8, 8)
    2. Concatenation:    board (batch, 21, 8, 8) + style -> (batch, 37, 8, 8)
    3. Initial conv:     37 -> 256 channels, 3x3, pad 1, BN, ReLU
    4. Residual tower:   10 x SE-ResidualBlock (256 -> 256)
    5. Policy head:      Conv 256->32, 1x1, BN, ReLU -> Dropout(0.3) -> FC(2048 -> 4672)
    6. Value head:       Conv 256->32, 1x1, BN, ReLU -> FC(2048 -> 256) -> ReLU -> FC(256 -> 1) -> Tanh
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


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.

    Structure:
        x -> GlobalAvgPool -> FC(C, C//ratio) -> ReLU -> FC(C//ratio, C) -> Sigmoid -> x * attention
    """

    def __init__(self, channels: int, ratio: int = config.SE_RATIO) -> None:
        super().__init__()
        mid = max(channels // ratio, 1)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        # Squeeze: (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
        s = self.squeeze(x).view(b, c)
        # Excitation
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        # Scale: (B, C) -> (B, C, 1, 1)
        return x * s.view(b, c, 1, 1)


class ResidualBlock(nn.Module):
    """
    A single residual block with SE attention and skip connection.

    Structure:
        x -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> SE -> (+x) -> ReLU
    """

    def __init__(self, num_filters: int = config.NUM_FILTERS,
                 se_ratio: int = config.SE_RATIO) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.se = SEBlock(num_filters, se_ratio)

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
        out = self.se(out)
        out = out + residual
        out = F.relu(out)
        return out


class ChessStyleNetwork(nn.Module):
    """
    Style-conditioned residual CNN with SE attention that predicts a policy
    (move distribution) and a position value given a board state and a
    playing-style token.

    Args:
        num_planes:      Number of input board planes (default 21).
        num_styles:      Number of distinct playing styles (default 3).
        style_embed_dim: Dimensionality of the style embedding vector (default 16).
        num_filters:     Number of convolutional filters in the residual tower (default 256).
        num_res_blocks:  Number of residual blocks in the tower (default 10).
        policy_filters:  Number of filters in the policy-head 1x1 conv (default 32).
        value_filters:   Number of filters in the value-head 1x1 conv (default 32).
        move_vocab_size: Size of the move vocabulary / output logits (default 4672).
        se_ratio:        SE block reduction ratio (default 4).
    """

    def __init__(
        self,
        num_planes: int = config.NUM_BOARD_PLANES,
        num_styles: int = config.NUM_STYLES,
        style_embed_dim: int = config.STYLE_EMBED_DIM,
        num_filters: int = config.NUM_FILTERS,
        num_res_blocks: int = config.NUM_RESIDUAL_BLOCKS,
        policy_filters: int = config.POLICY_HEAD_FILTERS,
        value_filters: int = config.VALUE_HEAD_FILTERS,
        move_vocab_size: int = config.MOVE_VOCAB_SIZE,
        se_ratio: int = config.SE_RATIO,
    ) -> None:
        super().__init__()

        self.num_planes = num_planes
        self.num_styles = num_styles
        self.style_embed_dim = style_embed_dim
        self.num_filters = num_filters
        self.num_res_blocks = num_res_blocks
        self.policy_filters = policy_filters
        self.value_filters = value_filters
        self.move_vocab_size = move_vocab_size
        self.se_ratio = se_ratio

        # -- Style embedding --------------------------------------
        self.style_embedding = nn.Embedding(num_styles, style_embed_dim)

        # -- Initial convolution (board planes + style planes -> filters) --
        in_channels = num_planes + style_embed_dim
        self.input_conv = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(num_filters)

        # -- Residual tower with SE attention ---------------------
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(num_filters, se_ratio) for _ in range(num_res_blocks)]
        )

        # -- Policy head ------------------------------------------
        self.policy_conv = nn.Conv2d(num_filters, policy_filters, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_filters)
        self.policy_dropout = nn.Dropout(0.3)
        self.policy_fc = nn.Linear(
            policy_filters * config.BOARD_SIZE * config.BOARD_SIZE,  # 32 * 8 * 8 = 2048
            move_vocab_size,
        )

        # -- Value head -------------------------------------------
        self.value_conv = nn.Conv2d(num_filters, value_filters, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(value_filters)
        self.value_fc1 = nn.Linear(
            value_filters * config.BOARD_SIZE * config.BOARD_SIZE,  # 32 * 8 * 8 = 2048
            256,
        )
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, board_tensor: torch.Tensor, style_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: board representation + style -> move logits + value.

        Args:
            board_tensor: Float tensor of shape (batch, num_planes, 8, 8).
            style_id:     Long tensor of shape (batch,) with style indices in [0, num_styles).

        Returns:
            Tuple of:
                - Policy logits of shape (batch, move_vocab_size).
                - Value estimate of shape (batch, 1) in [-1, 1].
        """
        batch_size = board_tensor.size(0)

        # Style embedding: (batch,) -> (batch, style_embed_dim)
        style_embed = self.style_embedding(style_id)

        # Broadcast style embedding spatially: -> (batch, style_embed_dim, 8, 8)
        style_planes = style_embed.unsqueeze(-1).unsqueeze(-1).expand(
            batch_size, self.style_embed_dim, config.BOARD_SIZE, config.BOARD_SIZE
        )

        # Concatenate board planes with style planes along channel dim
        # (batch, num_planes, 8, 8) + (batch, style_embed_dim, 8, 8)
        x = torch.cat([board_tensor, style_planes], dim=1)

        # Initial convolution block
        x = F.relu(self.input_bn(self.input_conv(x)))

        # Residual tower with SE attention
        x = self.residual_tower(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(batch_size, -1)
        p = self.policy_dropout(p)
        policy_logits = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(batch_size, -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters in the network."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls) -> ChessStyleNetwork:
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
            value_filters=config.VALUE_HEAD_FILTERS,
            move_vocab_size=config.MOVE_VOCAB_SIZE,
            se_ratio=config.SE_RATIO,
        )

    def __repr__(self) -> str:
        return (
            f"ChessStyleNetwork("
            f"planes={self.num_planes}, "
            f"styles={self.num_styles}, "
            f"embed={self.style_embed_dim}, "
            f"filters={self.num_filters}, "
            f"res_blocks={self.num_res_blocks}, "
            f"SE_ratio={self.se_ratio}, "
            f"policy_filters={self.policy_filters}, "
            f"value_filters={self.value_filters}, "
            f"vocab={self.move_vocab_size}, "
            f"params={self.count_parameters():,})"
        )
