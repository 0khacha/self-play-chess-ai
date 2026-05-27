# Play Against Yourself AI

Train 3 personalized chess AI variants from your Chess.com history and watch them compete against each other.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![python-chess](https://img.shields.io/badge/python--chess-1.10%2B-green)](https://python-chess.readthedocs.io)

---

## What Is This?

This project trains **3 distinct AI versions of you** based on your real Chess.com game history:

| Agent | Style | Behavior |
|-------|-------|----------|
| **Normal** | Balanced | Plays like your average game |
| **Aggressive** | Attacking | Sacrifices material, attacks the king, takes risks |
| **Defensive** | Positional | Castles early, retreats safely, avoids complications |

All three agents share one **style-conditioned neural network**  a single model that adapts its
play based on a style token injected at inference time.

It then runs a **self-play round-robin tournament**:

```
Normal      vs  Aggressive
Aggressive  vs  Defensive
Defensive   vs  Normal
```

And outputs full PGN files, win/loss stats, and style dominance rankings.

> **Want to play against the trained model?**
> The model in this repo was trained on games from Chess.com user **[0khacha](https://www.chess.com/member/0khacha)**.
> Challenge them on Chess.com if you want to see how the real player compares!

---

## Project Structure

```
SelfPlayChessAI/
+-- train.py                  # Run this first
+-- self_play.py              # Run this second
+-- config.py                 # All settings in one place
|
+-- data/
|   +-- fetcher.py            # Downloads games from Chess.com API (with caching)
|   +-- parser.py             # PGN -> (FEN, move) pairs for your moves only
|   +-- labeler.py            # Labels each move: Normal / Aggressive / Defensive
|   +-- dataset.py            # PyTorch Dataset + DataLoader builder
|
+-- model/
|   +-- encoding.py           # Board -> 18x8x8 tensor + 4672-move vocabulary
|   +-- network.py            # Style-conditioned Residual CNN (11.4M params)
|   +-- inference.py          # ChessAgent: model -> legal move selection
|
+-- training/
|   +-- trainer.py            # Training loop (Adam, cosine LR, early stopping)
|
+-- selfplay/
|   +-- engine.py             # Plays one game between two agents
|   +-- tournament.py         # Round-robin tournament manager
|   +-- stats.py              # Statistics, rankings, matplotlib charts
|
+-- utils/
|   +-- helpers.py            # Logging, PGN I/O, material helpers
|
+-- output/
    +-- models/               # Saved model checkpoints
    +-- games/                # Self-play PGN files
    +-- logs/                 # Training logs + style wins chart
    +-- raw/                  # Cached Chess.com API responses
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/SelfPlayChessAI.git
cd SelfPlayChessAI
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional but recommended) Install Stockfish

Download Stockfish from **https://stockfishchess.org/download/** and set the path in `config.py`:

```python
# config.py
STOCKFISH_PATH = r"C:\path\to\stockfish.exe"   # Windows
# STOCKFISH_PATH = "/usr/local/bin/stockfish"   # Linux/Mac
```

Without Stockfish, the system uses **heuristic-only** style labeling (still works well).

---

## Usage

### Step 1  Configure your username and archives

Edit `config.py`:
```python
CHESS_COM_USERNAME = "your_chess_com_username"

CHESS_COM_ARCHIVES = [
    "https://api.chess.com/pub/player/your_username/games/2025/01",
    "https://api.chess.com/pub/player/your_username/games/2025/02",
    # add more months as needed
]
```

### Step 2  Train the model
```bash
python train.py
```

This will:
1. Fetch all your games from Chess.com (cached after first run)
2. Parse every PGN and extract your moves
3. Label each move as Normal / Aggressive / Defensive
4. Build a PyTorch dataset
5. Train the style-conditioned neural network
6. Save the best model to `output/models/chess_style_model.pt`

**Example output:**
```
============================================================
PLAY AGAINST YOURSELF AI -- Training Pipeline
============================================================
  Player:    0khacha
  Archives:  14
  Device:    cuda

Step 1/5: Fetching games from Chess.com...
   Fetched 847 games

Step 2/5: Parsing PGN and extracting positions...
   Extracted 18,342 position-move samples

Step 3/5: Labeling move styles...
   Labeled 18,342 samples
      Normal:     10,124 (55.2%)
      Aggressive:  4,891 (26.7%)
      Defensive:   3,327 (18.1%)

Step 5/5: Training the model...
   Model parameters: 11,407,584

Epoch   1/50 | Train Loss: 8.4523  Acc: 0.021 | Val Loss: 8.3901  Acc: 0.023
Epoch   2/50 | Train Loss: 7.9142  Acc: 0.038 | Val Loss: 7.8830  Acc: 0.041
...
```

### Step 3  Run the self-play tournament
```bash
python self_play.py
```

**Example output:**
```
============================================================
Self-Play Round-Robin Tournament
============================================================

Matchup 1/3: Normal vs Aggressive
   [White: Normal]      [Black: Aggressive]   W W D W L W W W D W
   [White: Aggressive]  [Black: Normal]       L W D W W L D W W L
   Normal: 12W  |  Aggressive: 6W  |  Draws: 2  (20 games)

Matchup 2/3: Aggressive vs Defensive
   ...

============================================================
Tournament Results Summary

Matchup                               W-A   W-B  Draw  Games  Avg Len
----------------------------------------------------------------------
Normal      vs  Aggressive             12     6     2     20     47.3
Aggressive  vs  Defensive               8    10     2     20     43.1
Defensive   vs  Normal                  7     9     4     20     61.8

Style Ranking (by total wins)
----------------------------------------
  1st  Normal         21W / 13L / 6D   (WR: 52.5%)
  2nd  Defensive      17W / 15L / 8D   (WR: 42.5%)
  3rd  Aggressive     14W / 16L / 10D  (WR: 35.0%)

Dominant style: Normal with 21 total wins
```

---

## Model Architecture

```
Input: board_tensor (18x8x8) + style_token (0/1/2)
                    |
         +----------+-----------+
         |  Style Embedding     |   nn.Embedding(3, 32)
         |  32-dim -> 32x8x8   |   broadcast spatially
         +----------+-----------+
                    | concat
         +----------+-----------+
         |  (18+32)x8x8 = 50   |
         +----------+-----------+
                    |
         +----------+-----------+
         |  Conv 50->128, 3x3  |   + BatchNorm + ReLU
         +----------+-----------+
                    |
         +----------+-----------+
         |  6x Residual Blocks  |   128->128 with skip connections
         +----------+-----------+
                    |
         +----------+-----------+
         |  Policy Head         |   Conv 128->32 (1x1) + FC(2048->4672)
         +----------+-----------+
                    |
              Logits (4672,)    <- masked to legal moves only
```

**Total parameters: 11,407,584**

### Board Encoding  18 planes x 8x8 (perspective-relative)

| Planes | Content |
|--------|---------|
| 0-5  | Current player's pieces (P, N, B, R, Q, K) |
| 6-11 | Opponent's pieces (P, N, B, R, Q, K) |
| 12   | Side to move (always 1  relative encoding) |
| 13-14 | Current player's castling rights (kingside, queenside) |
| 15-16 | Opponent's castling rights (kingside, queenside) |
| 17   | En passant target square |

### Move Encoding  AlphaZero-style 4672

- 64 from-squares x 73 move types = **4672 total**
- 56 queen-type moves (8 directions x 7 distances)
- 8 knight moves
- 9 underpromotions (3 piece types x 3 directions)
- Queen promotions encoded as queen-type moves

---

## Style Labeling

### Without Stockfish (heuristic-only)

| Style | Trigger Conditions |
|-------|-------------------|
| **Aggressive** | Sacrifice/trade-up capture . Giving check . Pawn push past rank 5 . Move into opponent's king zone |
| **Defensive** | Castling . Retreat move . Quiet move staying on own half |
| **Normal** | Everything else |

### With Stockfish (enhanced)

Uses centipawn evaluation change and material balance:

| Style | Engine Signals |
|-------|---------------|
| **Aggressive** | Eval drops >50cp with capture/check . Material sacrifice |
| **Defensive** | Eval change <30cp with simplification or retreat |
| **Normal** | Stable eval without strong positional signal |

---

## Configuration Reference

All settings are in [`config.py`](config.py):

| Setting | Default | Description |
|---------|---------|-------------|
| `CHESS_COM_USERNAME` | `"0khacha"` | Your Chess.com username |
| `CHESS_COM_ARCHIVES` | 14 months | Archive URLs to fetch |
| `STOCKFISH_PATH` | `None` | Path to Stockfish binary |
| `STOCKFISH_DEPTH` | `12` | Search depth for evaluation |
| `NUM_FILTERS` | `128` | CNN filter count |
| `NUM_RESIDUAL_BLOCKS` | `6` | Residual tower depth |
| `BATCH_SIZE` | `256` | Training batch size |
| `LEARNING_RATE` | `1e-3` | Adam learning rate |
| `NUM_EPOCHS` | `50` | Max training epochs |
| `EARLY_STOPPING_PATIENCE` | `5` | Epochs without improvement before stopping |
| `SELFPLAY_GAMES_PER_MATCHUP` | `10` | Games per color per matchup |
| `SELFPLAY_TEMPERATURE` | `0.8` | Move sampling temperature |
| `SELFPLAY_MAX_MOVES` | `200` | Max moves per side per game |

---

## Output Files

| File | Description |
|------|-------------|
| `output/models/chess_style_model.pt` | Best trained model checkpoint |
| `output/games/Normal_vs_Aggressive_YYYY-MM-DD.pgn` | PGN of self-play games |
| `output/logs/training_log.csv` | Per-epoch loss/accuracy |
| `output/logs/style_wins.png` | Bar chart of wins per style |
| `output/raw/*.json` | Cached Chess.com API responses |

---

## Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >=2.0 | Neural network (CUDA recommended) |
| `python-chess` | >=1.10 | PGN parsing, board logic, legal moves |
| `requests` | >=2.31 | Chess.com API fetching |
| `numpy` | >=1.24 | Array operations |
| `tqdm` | >=4.65 | Progress bars |
| `matplotlib` | >=3.7 | Style wins chart |
| `stockfish` | >=3.28 | Stockfish Python wrapper (optional) |

---

## Roadmap

- [ ] MCTS (Monte Carlo Tree Search) integration for self-play improvement
- [ ] Value head addition (predict game outcome from position)
- [ ] Web UI to play against your AI in the browser
- [ ] Support for multiple player archives (opponent modeling)
- [ ] Lichess API support
- [ ] Docker container for reproducible training

---

## License

MIT  see [LICENSE](LICENSE)
