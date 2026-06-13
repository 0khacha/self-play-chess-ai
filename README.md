# Play Against Yourself AI

> Challenge a clone of any Chess.com player -- built from their real game history.

Enter any Chess.com username, and the AI builds a playing profile from their archived games.
It replays their actual moves in known positions and uses a neural network with tactical safety filters for novel ones.

**No Stockfish required.** Everything runs locally.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![Flask](https://img.shields.io/badge/backend-Flask-lightgrey)
![PyTorch](https://img.shields.io/badge/model-PyTorch-red)

---

## How It Works

```
User enters a Chess.com username
          |
          v
    +-------------+
    | PlayerBook  |  Fetches all archived games via Chess.com API
    |             |  Builds FEN -> {move: frequency} map
    +------+------+
           |
           v
    +------+------+
    | CloneAgent  |  Plays the game move-by-move
    +------+------+
           |
     +-----+-----+
     |           |
  Book hit?   Book miss?
     |           |
  Weighted    Neural policy network
  random      scores all legal moves
  from the         |
  player's    SEE tactical filter
  history     (removes blunders)
                   |
              Best safe move
```

### Move Selection Priority

| Priority | Source | Description |
|----------|--------|-------------|
| 1 | **Position Book** | Exact FEN match -- plays the player's actual historical move (weighted by frequency) |
| 2 | **Neural Network** | ResNet policy network scores all 4672 legal move encodings |
| 3 | **SEE Tactical Filter** | Static Exchange Evaluation removes moves that hang material |
| 4 | **Forced Move** | If all moves lose material, plays the highest-scored one anyway |

---

## Features

- **Clone any Chess.com player** -- enter a username, play against their style
- **Chess clocks** -- 10min / 5min / 3min / 1min time controls
- **No engine dependency** -- pure neural network + book + SEE filter
- **Per-user model training** -- train a dedicated model for better accuracy
- **Dark premium UI** -- glassmorphism, animations, responsive layout
- **Castling support** -- click the rook or the king's destination square
- **Full game tracking** -- move list, captured pieces, material diff

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.10+
- `python-chess` >= 1.10.0
- `torch` >= 2.0.0
- `flask` >= 3.0.0
- `requests`, `numpy`, `tqdm`, `matplotlib`

### 2. Run the server

```bash
python play.py
```

Open **http://localhost:5000** in your browser.

The default player (`0khacha`) is pre-loaded automatically. Enter any Chess.com username and click **Load** to play against someone else.

### 3. (Optional) Train a dedicated model

For better accuracy on novel positions, train a model specifically for a player:

```bash
# Train for the default user
python train_for_user.py

# Train for any Chess.com username
python train_for_user.py --user hikaru --epochs 40
```

The trained model is saved as `output/models/clone_<username>.pt` and automatically used by the server when that username is loaded.

---

## Project Structure

```
SelfPlayChessAI/
|-- play.py                  # Flask web server (main entry point)
|-- train_for_user.py        # Train a per-user clone model
|-- config.py                # All configuration and hyperparameters
|-- requirements.txt
|
|-- model/
|   |-- clone_agent.py       # CloneAgent: book + neural + SEE filter
|   |-- network.py           # ChessStyleNetwork (ResNet CNN)
|   |-- encoding.py          # Board encoding (18 planes) + AlphaZero move encoding (4672)
|
|-- data/
|   |-- fetcher.py           # Chess.com API game fetcher with caching
|   |-- parser.py            # PGN parser -> GameRecord objects
|   |-- labeler.py           # Style labeling (Normal/Aggressive/Defensive)
|   |-- dataset.py           # PyTorch dataset with game-based splitting
|
|-- training/
|   |-- trainer.py           # Training loop with early stopping
|
|-- static/
|   |-- index.html           # Game UI
|   |-- style.css            # Premium dark theme CSS
|   |-- app.js               # Chess game logic + board rendering
|
|-- utils/
|   |-- helpers.py           # Logging setup + board utilities
|
|-- output/
|   |-- models/              # Trained .pt model files
|   |-- logs/                # Training logs
```

---

## Architecture

### Neural Network

**ChessStyleNetwork** -- a residual CNN conditioned on playing style:

| Component | Details |
|-----------|---------|
| Input | 18-plane 8x8 board encoding (perspective-relative) |
| Style embedding | 3 styles (Normal, Aggressive, Defensive) -> 8-dim -> broadcast to 8x8 |
| Initial conv | 26 -> 128 channels, 3x3, BatchNorm, ReLU |
| Residual tower | 6 ResidualBlocks (128 channels each) |
| Policy head | Conv 128->32, 1x1 -> Dropout(0.3) -> FC(2048 -> 4672) |
| Output | 4672 logits (AlphaZero move encoding) |
| Parameters | ~1.6M |

### Board Encoding (18 planes)

| Planes | Content |
|--------|---------|
| 0-5 | Current player's pieces (P, N, B, R, Q, K) |
| 6-11 | Opponent's pieces |
| 12 | Side to move (constant 1) |
| 13-16 | Castling rights (KQkq) |
| 17 | En passant square |

### Tactical Filter (SEE)

Static Exchange Evaluation prevents the neural network from making blunders:

1. Score all legal moves with the neural network
2. Take the top-5 candidates
3. For each candidate, simulate the move and check if any of our pieces can be captured for a net material gain
4. Play the highest-scoring **safe** move
5. If no move is safe (zugzwang), play the highest-scored one anyway

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serve the game UI |
| `POST` | `/api/load_player` | Load a Chess.com player's games. Body: `{"username": "..."}` |
| `POST` | `/api/start` | Start a new game. Body: `{"playerColor": "white"\|"black"}` |
| `POST` | `/api/move` | Make a move. Body: `{"fen": "...", "move": "e2e4"}` |

---

## Configuration

All settings are in [`config.py`](config.py):

| Setting | Default | Description |
|---------|---------|-------------|
| `CHESS_COM_USERNAME` | `0khacha` | Default player to pre-load |
| `NUM_FILTERS` | 128 | ResNet channel width |
| `NUM_RESIDUAL_BLOCKS` | 6 | Depth of the residual tower |
| `BATCH_SIZE` | 256 | Training batch size |
| `LEARNING_RATE` | 1e-3 | Adam learning rate |
| `LABEL_SMOOTHING` | 0.1 | Cross-entropy label smoothing |
| `NUM_EPOCHS` | 50 | Max training epochs |
| `EARLY_STOPPING_PATIENCE` | 8 | Stop after N epochs without improvement |

---

## License

MIT License -- see [LICENSE](LICENSE).

---

Built with <3 by [0khacha](https://github.com/0khacha)
