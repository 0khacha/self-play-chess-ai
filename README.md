# ♟️ Play Against Yourself AI

Train a personalized chess AI from your **Chess.com** game history, then play against it in a sleek dark-themed web interface. The AI learns your playing style and uses **Stockfish** as a tactical backbone to ensure it never makes dumb moves.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![python-chess](https://img.shields.io/badge/python--chess-1.10%2B-green)](https://python-chess.readthedocs.io)

---

## ✨ Features

- **🧠 Style-Conditioned Neural Network** — A single ResNet model that adapts its play based on a style token (Normal / Aggressive / Defensive)
- **🐟 Hybrid Engine** — Combines your trained neural model with Stockfish: your style + no blunders
- **🎮 Web Interface** — Dark-themed, responsive chess board with click-to-move, move history, captured pieces, and game-over modals
- **⚔️ Self-Play Tournament** — Round-robin between all 3 style variants with full PGN output and statistics
- **📊 Smart Data Pipeline** — Game-based train/val splitting (no data leakage), opening book filtering, bullet game exclusion, horizontal flip augmentation

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/0khacha/self-play-chess-ai.git
cd self-play-chess-ai
pip install -r requirements.txt
```

### 2. Download Stockfish (recommended)

Download from **https://stockfishchess.org/download/** and place the binary in the project:

```bash
# Windows (PowerShell)
Invoke-WebRequest -Uri "https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-windows-x86-64-avx2.zip" -OutFile stockfish.zip
Expand-Archive stockfish.zip -DestinationPath stockfish

# Linux/Mac
wget https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64-avx2.tar
tar xf stockfish-ubuntu-x86-64-avx2.tar -C stockfish
```

Then set the path in `config.py`:
```python
STOCKFISH_PATH = os.path.join(PROJECT_ROOT, "stockfish", "stockfish", "stockfish-windows-x86-64-avx2.exe")
# Or for Linux: "/path/to/stockfish/stockfish-ubuntu-x86-64-avx2"
```

### 3. Configure your username

Edit `config.py`:
```python
CHESS_COM_USERNAME = "your_chess_com_username"

CHESS_COM_ARCHIVES = [
    "https://api.chess.com/pub/player/your_username/games/2025/01",
    "https://api.chess.com/pub/player/your_username/games/2025/02",
    # add more months as needed
]
```

### 4. Train the model

```bash
python train.py
```

This will:
1. Fetch all your games from Chess.com (cached after first run)
2. Filter out bullet games (< 3 min) and skip opening book moves
3. Label each move as Normal / Aggressive / Defensive
4. Train a style-conditioned neural network with augmentation
5. Save the best model to `output/models/chess_style_model.pt`

### 5. Play against your AI!

```bash
python play.py
```

Open **http://localhost:5000** in your browser. Pick a style, pick your color, and start playing!

### 6. (Optional) Run a self-play tournament

```bash
python self_play.py
```

---

## 🎮 Web Interface

The web UI features:
- **Dark theme** with glassmorphism panels and ambient lighting
- **Wooden board frame** with crisp piece rendering
- **Click-to-move** with legal move hints (dots for empty squares, rings for captures)
- **3 AI personalities** — Normal, Aggressive, Defensive
- **Move history** panel with algebraic notation
- **Captured pieces** display with material advantage
- **Game-over modal** with play-again option
- **AI thinking indicator** with pulsing glow animation

---

## 🏗️ Architecture

### Hybrid Engine (Neural + Stockfish)

```
Position
    │
    ├── Stockfish (depth 10) ──→ Top 10 safe moves + evaluations
    │                               │
    │                               ▼
    │                        Filter blunders
    │                        (> 150cp loss)
    │                               │
    └── Neural Model ──────→ Style scores for each safe move
                                    │
                                    ▼
                            Pick highest combined score
                            (style match + tactical quality)
```

**Result:** Plays like you, but never hangs a piece.

### Neural Network

```
Input: board (18×8×8) + style token (0/1/2)
    │
    ├── Style Embedding: nn.Embedding(3, 8) → broadcast to 8×8×8
    │
    ▼
Concatenate → 26×8×8
    │
Conv 26→128, 3×3 + BN + ReLU
    │
6× Residual Blocks (128→128)
    │
Policy Head: Conv 128→32 (1×1) + Dropout(0.3) + FC(2048→4672)
    │
Logits (4672) → masked to legal moves
```

### Board Encoding — 18 planes × 8×8

| Planes | Content |
|--------|---------|
| 0–5    | Current player's pieces (P, N, B, R, Q, K) |
| 6–11   | Opponent's pieces (P, N, B, R, Q, K) |
| 12     | Side to move |
| 13–14  | Current player's castling rights |
| 15–16  | Opponent's castling rights |
| 17     | En passant target square |

### Move Encoding — AlphaZero-style 4672

- 64 from-squares × 73 move types = **4672 total**
- 56 queen-type moves (8 directions × 7 distances)
- 8 knight moves + 9 underpromotions

---

## 📂 Project Structure

```
SelfPlayChessAI/
├── train.py                  # Training pipeline entry point
├── self_play.py              # Self-play tournament runner
├── play.py                   # Web server (Flask)
├── config.py                 # All settings in one place
│
├── data/
│   ├── fetcher.py            # Chess.com API with caching + bullet filtering
│   ├── parser.py             # PGN → (FEN, move) with opening skip
│   ├── labeler.py            # Multi-indicator style labeling
│   └── dataset.py            # Game-based split + flip augmentation
│
├── model/
│   ├── encoding.py           # Board → 18×8×8 tensor + 4672-move vocab
│   ├── network.py            # Style-conditioned ResNet + Dropout
│   ├── inference.py          # Pure neural agent with top-k sampling
│   └── hybrid_agent.py       # Neural + Stockfish hybrid agent
│
├── training/
│   └── trainer.py            # Training loop with label smoothing
│
├── selfplay/
│   ├── engine.py             # Single game player
│   ├── tournament.py         # Round-robin manager
│   └── stats.py              # Statistics + charts
│
├── static/
│   ├── index.html            # Chess board UI
│   ├── style.css             # Dark theme stylesheet
│   └── app.js                # Client-side game logic
│
└── output/
    ├── models/               # Saved checkpoints (.pt)
    ├── games/                # Self-play PGN files
    ├── logs/                 # Training logs + charts
    └── raw/                  # Cached API responses
```

---

## ⚙️ Configuration

All settings are in [`config.py`](config.py):

| Setting | Default | Description |
|---------|---------|-------------|
| `CHESS_COM_USERNAME` | `"0khacha"` | Your Chess.com username |
| `STOCKFISH_PATH` | Auto-detected | Path to Stockfish binary |
| `STYLE_EMBED_DIM` | `8` | Style embedding dimensions |
| `NUM_FILTERS` | `128` | CNN filter count |
| `NUM_RESIDUAL_BLOCKS` | `6` | Residual tower depth |
| `BATCH_SIZE` | `256` | Training batch size |
| `LEARNING_RATE` | `1e-3` | Adam learning rate |
| `WEIGHT_DECAY` | `1e-3` | L2 regularization strength |
| `LABEL_SMOOTHING` | `0.1` | Cross-entropy label smoothing |
| `VALIDATION_SPLIT` | `0.15` | Fraction held for validation |
| `EARLY_STOPPING_PATIENCE` | `8` | Epochs before stopping |
| `INFERENCE_TOP_K` | `5` | Top-k sampling for neural-only mode |
| `MIN_TIME_CONTROL_SECONDS` | `180` | Min time control (filters bullet) |
| `SKIP_OPENING_MOVES` | `6` | Opening moves to skip |

---

## 🧪 Anti-Overfitting Measures

| Technique | Implementation |
|-----------|---------------|
| **Game-based split** | Train/val split by game, not sample — no data leakage |
| **Horizontal flip augmentation** | Mirror positions kingside↔queenside to 2× data |
| **Dropout (0.3)** | Before the policy FC layer |
| **Label smoothing (0.1)** | Prevents overconfident predictions |
| **Weight decay (1e-3)** | L2 regularization on all parameters |
| **Multi-indicator labeling** | Requires 2+ signals for aggressive/defensive labels |
| **Opening filtering** | First 6 moves excluded (book territory) |
| **Bullet filtering** | Games < 3 min excluded (hasty moves ≠ real style) |

---

## 📄 License

MIT — see [LICENSE](LICENSE)
