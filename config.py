"""
Global configuration for Play Against Yourself AI.
"""
import os
import torch

# ---------------------------------------------
# Project Paths
# ---------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
GAMES_DIR = os.path.join(OUTPUT_DIR, "games")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
RAW_DATA_DIR = os.path.join(OUTPUT_DIR, "raw")

for d in [OUTPUT_DIR, MODELS_DIR, GAMES_DIR, LOGS_DIR, RAW_DATA_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------
# Chess.com API
# ---------------------------------------------
CHESS_COM_USERNAME = "0khacha"
API_USER_AGENT = "SelfPlayChessAI/1.0 (chess-ml-research)"
API_RATE_LIMIT_SECONDS = 1.0

# ---------------------------------------------
# Stockfish (optional, used for move labeling)
# ---------------------------------------------
# Set to None to use heuristic-only labeling.
STOCKFISH_PATH = os.path.join(PROJECT_ROOT, "stockfish", "stockfish-windows-x86-64-avx2.exe")
STOCKFISH_DEPTH = 12
STOCKFISH_THREADS = 2
STOCKFISH_HASH_MB = 256

# ---------------------------------------------
# Style Labeling Thresholds (centipawns)
# ---------------------------------------------
AGGRESSIVE_EVAL_DROP = -50
AGGRESSIVE_MATERIAL_SACRIFICE = True
DEFENSIVE_EVAL_STABILITY = 30
DEFENSIVE_SIMPLIFICATION = True

# ---------------------------------------------
# Board Encoding
# ---------------------------------------------
NUM_BOARD_PLANES = 18   # 12 piece + turn + 4 castling + en passant
BOARD_SIZE = 8

# ---------------------------------------------
# Move Encoding (AlphaZero-style)
# ---------------------------------------------
NUM_MOVE_TYPES = 73     # 56 queen + 8 knight + 9 underpromotion
NUM_SQUARES = 64
MOVE_VOCAB_SIZE = NUM_SQUARES * NUM_MOVE_TYPES  # 4672

# ---------------------------------------------
# Style Tokens
# ---------------------------------------------
STYLE_NORMAL = 0
STYLE_AGGRESSIVE = 1
STYLE_DEFENSIVE = 2
NUM_STYLES = 3

STYLE_NAMES = {
    STYLE_NORMAL: "Normal",
    STYLE_AGGRESSIVE: "Aggressive",
    STYLE_DEFENSIVE: "Defensive",
}

# ---------------------------------------------
# Model Architecture
# ---------------------------------------------
STYLE_EMBED_DIM = 8
NUM_FILTERS = 128
NUM_RESIDUAL_BLOCKS = 6
POLICY_HEAD_FILTERS = 32

# ---------------------------------------------
# Training Hyperparameters
# ---------------------------------------------
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 8
VALIDATION_SPLIT = 0.15
LABEL_SMOOTHING = 0.1
CHECKPOINT_NAME = "chess_style_model.pt"

# Data filtering
MIN_GAME_HALFMOVES = 14
SKIP_OPENING_MOVES = 6
MIN_TIME_CONTROL_SECONDS = 180

# ---------------------------------------------
# Device
# ---------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
