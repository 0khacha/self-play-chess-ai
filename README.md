# Play Against Yourself AI

> Challenge a clone of any Chess.com player — built from their real game history.

Enter any Chess.com username, and the AI builds a playing profile from their archived games.
It replays their actual moves in known positions and uses a neural network with tactical safety filters for novel ones.

**No Stockfish required.** Everything runs locally.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![Flask](https://img.shields.io/badge/backend-Flask-lightgrey)
![PyTorch](https://img.shields.io/badge/model-PyTorch-red)
![Version](https://img.shields.io/badge/version-2.0.0-purple)

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
| 1 | **Position Book** | Exact FEN match — plays the player's actual historical move (weighted by frequency) |
| 2 | **Neural Network** | ResNet + SE-attention policy network scores all 4672 legal move encodings |
| 3 | **SEE Tactical Filter** | Static Exchange Evaluation removes moves that hang material |
| 4 | **Temperature Sampling** | Softmax with temperature for human-like variation |
| 5 | **Forced Move** | If all moves lose material, plays the highest-scored one anyway |

---

## Features

- **Clone any Chess.com player** — enter a username, play against their style
- **Chess clocks** — 10min / 5min / 3min / 1min time controls
- **No engine dependency** — pure neural network + book + SEE filter
- **Per-user model training** — train a dedicated model for better accuracy
- **Dark premium UI** — glassmorphism, animations, responsive layout
- **Move sounds** — Web Audio API sound effects (move, capture, check, castle)
- **Drag & drop** — move pieces by clicking or dragging
- **Keyboard shortcuts** — `N` new game, `F` flip, `Z` undo, `M` mute, `R` resign
- **Undo moves** — take back your last move
- **Resign & draw** — resign or offer a draw mid-game
- **PGN export** — download your game as a PGN file
- **Premove support** — queue a move while the AI is thinking
- **Castling support** — click the rook or the king's destination square
- **Full game tracking** — move list, captured pieces, material diff
- **Style-aware AI** — automatic style detection (Normal / Aggressive / Defensive)

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
- `requests`, `numpy`, `tqdm`

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

# Resume training from a checkpoint
python train_for_user.py --user hikaru --resume
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
|   |-- network.py           # ChessStyleNetwork (ResNet + SE-attention CNN)
|   |-- encoding.py          # Board encoding (21 planes) + AlphaZero move encoding (4672)
|
|-- data/
|   |-- fetcher.py           # Chess.com API game fetcher with caching
|   |-- parser.py            # PGN parser -> GameRecord objects
|   |-- labeler.py           # Style labeling (Normal/Aggressive/Defensive)
|   |-- dataset.py           # PyTorch dataset with game-based splitting
|
|-- training/
|   |-- trainer.py           # Training loop with early stopping + top-k accuracy
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

**ChessStyleNetwork** — a residual CNN with Squeeze-and-Excitation attention, conditioned on playing style:

| Component | Details |
|-----------|---------|
| Input | 21-plane 8x8 board encoding (perspective-relative) |
| Style embedding | 3 styles (Normal, Aggressive, Defensive) → 16-dim → broadcast to 8×8 |
| Initial conv | 37 → 256 channels, 3×3, BatchNorm, ReLU |
| Residual tower | 10 SE-ResidualBlocks (256 channels each, SE ratio 4) |
| Policy head | Conv 256→32, 1×1 → Dropout(0.3) → FC(2048 → 4672) |
| Value head | Conv 256→32, 1×1 → FC(2048 → 256 → 1) → Tanh |
| Output | 4672 policy logits + scalar value in [-1, 1] |
| Parameters | ~6.5M |

### Board Encoding (21 planes)

| Planes | Content |
|--------|---------|
| 0-5 | Current player's pieces (P, N, B, R, Q, K) |
| 6-11 | Opponent's pieces |
| 12 | Side to move (constant 1) |
| 13-16 | Castling rights (KQkq) |
| 17 | En passant square |
| 18 | Attack map (squares attacked by current player) |
| 19 | Defense map (squares attacked by opponent) |
| 20 | Last move (to-square of the previous move) |

### Tactical Filter (SEE)

Static Exchange Evaluation prevents the neural network from making blunders:

1. Score all legal moves with the neural network
2. Take the top-8 candidates
3. For each candidate, simulate captures and check net material gain/loss
4. Play the highest-scoring **safe** move with temperature sampling
5. If no move is safe (zugzwang), play the highest-scored one anyway

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serve the game UI |
| `POST` | `/api/load_player` | Load a Chess.com player's games. Body: `{"username": "..."}` |
| `POST` | `/api/start` | Start a new game. Body: `{"playerColor": "white"\|"black"}` |
| `POST` | `/api/move` | Make a move. Body: `{"fen": "...", "move": "e2e4"}` |
| `POST` | `/api/undo` | Undo the last move pair (player + AI) |
| `POST` | `/api/resign` | Resign the current game |
| `POST` | `/api/export_pgn` | Export the current game as PGN |

---

## Configuration

All settings are in [`config.py`](config.py):

| Setting | Default | Description |
|---------|---------|-------------|
| `CHESS_COM_USERNAME` | `0khacha` | Default player to pre-load |
| `NUM_FILTERS` | 256 | ResNet channel width |
| `NUM_RESIDUAL_BLOCKS` | 10 | Depth of the residual tower |
| `SE_RATIO` | 4 | Squeeze-and-Excitation reduction ratio |
| `BATCH_SIZE` | 256 | Training batch size |
| `LEARNING_RATE` | 1e-3 | Adam learning rate |
| `LABEL_SMOOTHING` | 0.1 | Cross-entropy label smoothing |
| `NUM_EPOCHS` | 80 | Max training epochs |
| `EARLY_STOPPING_PATIENCE` | 12 | Stop after N epochs without improvement |
| `NEURAL_TEMPERATURE` | 1.2 | Softmax temperature for move selection |

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `N` | New game |
| `F` | Flip board |
| `Z` / `Ctrl+Z` | Undo last move |
| `M` | Toggle sound |
| `R` | Resign |
| `Esc` | Deselect piece |

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

MIT License — see [LICENSE](LICENSE).

---

Built with <3 by [0khacha](https://github.com/0khacha)
