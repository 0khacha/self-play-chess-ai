"""
Global configuration for Play Against Yourself AI.
"""
import os
import torch

# ---------------------------------------------
# Version
# ---------------------------------------------
VERSION = "2.0.0"

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
API_USER_AGENT = "SelfPlayChessAI/2.0 (chess-ml-research)"
API_RATE_LIMIT_SECONDS = 1.0

# ---------------------------------------------
# Stockfish (optional, used for move labeling)
# ---------------------------------------------
# Auto-detect: only set if the binary actually exists.
_STOCKFISH_CANDIDATE = os.path.join(
    PROJECT_ROOT, "stockfish", "stockfish-windows-x86-64-avx2.exe"
)
STOCKFISH_PATH = _STOCKFISH_CANDIDATE if os.path.isfile(_STOCKFISH_CANDIDATE) else None
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
NUM_BOARD_PLANES = 21   # 12 piece + turn + 4 castling + en passant + 2 attack maps + last move
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
STYLE_EMBED_DIM = 16
NUM_FILTERS = 256
NUM_RESIDUAL_BLOCKS = 10
POLICY_HEAD_FILTERS = 32
VALUE_HEAD_FILTERS = 32
SE_RATIO = 4                # Squeeze-and-Excitation reduction ratio

# ---------------------------------------------
# Training Hyperparameters
# ---------------------------------------------
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 80
EARLY_STOPPING_PATIENCE = 12
VALIDATION_SPLIT = 0.15
LABEL_SMOOTHING = 0.1
CHECKPOINT_NAME = "chess_style_model.pt"

# Top-K accuracy tracking (for chess, top-3/top-5 are more meaningful)
TOP_K_ACCURACIES = [1, 3, 5]

# Learning rate warmup
LR_WARMUP_EPOCHS = 3

# Data filtering
MIN_GAME_HALFMOVES = 14
SKIP_OPENING_MOVES = 6
MIN_TIME_CONTROL_SECONDS = 180

# ---------------------------------------------
# Inference
# ---------------------------------------------
NEURAL_TOP_K = 8            # candidates for tactical filtering
NEURAL_TEMPERATURE = 1.2    # softmax temperature for move sampling

# ---------------------------------------------
# Device
# ---------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
