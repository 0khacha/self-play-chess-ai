"""
Play Against Yourself AI - Web Interface

Uses the HybridChessAgent (neural style model + Stockfish blunder filter)
for smart, style-aware play that never makes dumb moves.

Run:
    python play.py

Then open http://localhost:5000 in your browser.
"""

import os
import sys

from flask import Flask, request, jsonify, send_from_directory
import chess

import config

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder="static", static_url_path="/static")

# Cache agents so the model + engine are loaded only once per style.
_agents: dict[int, object] = {}


def _get_agent(style: int):
    """Get or create a cached agent for the given style.

    Uses HybridChessAgent (neural + Stockfish) if Stockfish is available,
    otherwise falls back to the pure neural ChessAgent.
    """
    if style not in _agents:
        model_path = os.path.join(config.MODELS_DIR, config.CHECKPOINT_NAME)

        if config.STOCKFISH_PATH and os.path.isfile(config.STOCKFISH_PATH):
            from model.hybrid_agent import HybridChessAgent
            _agents[style] = HybridChessAgent(
                model_path=model_path,
                style=style,
                stockfish_path=config.STOCKFISH_PATH,
                stockfish_depth=10,
                blunder_threshold=150,
                stockfish_multipv=10,
            )
        else:
            from model.inference import ChessAgent
            _agents[style] = ChessAgent(
                model_path=model_path,
                style=style,
                temperature=0.5,
            )
    return _agents[style]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _legal_moves(board: chess.Board) -> list[dict]:
    """Return legal moves in a JSON-friendly format."""
    moves = []
    for m in board.legal_moves:
        moves.append(
            {
                "from": chess.square_name(m.from_square),
                "to": chess.square_name(m.to_square),
                "uci": m.uci(),
                "capture": board.is_capture(m),
                "promotion": m.promotion is not None,
            }
        )
    return moves


def _board_state(board: chess.Board) -> dict:
    """Snapshot the board into a JSON-serialisable dict."""
    game_over = board.is_game_over(claim_draw=True)
    outcome = board.outcome(claim_draw=True) if game_over else None
    return {
        "fen": board.fen(),
        "legalMoves": _legal_moves(board) if not game_over else [],
        "isCheck": board.is_check(),
        "gameOver": game_over,
        "result": outcome.result() if outcome else None,
        "termination": (
            outcome.termination.name.replace("_", " ").title() if outcome else None
        ),
    }


def _move_info(move: chess.Move, san: str) -> dict:
    return {
        "from": chess.square_name(move.from_square),
        "to": chess.square_name(move.to_square),
        "uci": move.uci(),
        "san": san,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/start", methods=["POST"])
def start_game():
    """Start a new game.  If the player chose black, the AI plays first."""
    data = request.json or {}
    style = int(data.get("style", 0))
    player_color = data.get("playerColor", "white")

    board = chess.Board()
    resp = _board_state(board)
    resp["aiMove"] = None

    if player_color == "black":
        agent = _get_agent(style)
        ai_move = agent.select_move(board)
        san = board.san(ai_move)
        board.push(ai_move)
        resp = _board_state(board)
        resp["aiMove"] = _move_info(ai_move, san)

    return jsonify(resp)


@app.route("/api/move", methods=["POST"])
def make_move():
    """Accept the player's move, validate it, then let the AI respond."""
    data = request.json
    fen = data["fen"]
    move_uci = data["move"]
    style = int(data.get("style", 0))

    board = chess.Board(fen)

    # --- validate player move ---
    try:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return jsonify({"success": False, "error": "Illegal move"}), 400
    except Exception:
        return jsonify({"success": False, "error": "Invalid move string"}), 400

    player_san = board.san(move)
    board.push(move)

    resp: dict = {"success": True, "playerMove": _move_info(move, player_san), "aiMove": None}
    resp.update(_board_state(board))

    # If the game ended with the player's move, return immediately.
    if board.is_game_over(claim_draw=True):
        return jsonify(resp)

    # --- AI responds ---
    agent = _get_agent(style)
    ai_move = agent.select_move(board)
    ai_san = board.san(ai_move)
    board.push(ai_move)

    resp["aiMove"] = _move_info(ai_move, ai_san)
    resp.update(_board_state(board))
    return jsonify(resp)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model_path = os.path.join(config.MODELS_DIR, config.CHECKPOINT_NAME)
    if not os.path.exists(model_path):
        print(f"Error: trained model not found at {model_path}")
        print("Run  python train.py  first to train the model.")
        sys.exit(1)

    mode = "HYBRID (Neural + Stockfish)" if (
        config.STOCKFISH_PATH and os.path.isfile(config.STOCKFISH_PATH)
    ) else "Neural only"
    print(f"\n  Mode: {mode}")

    print("  Loading AI agents...")
    for s in range(3):
        _get_agent(s)
        print(f"    [OK] {config.STYLE_NAMES[s]}")

    print("\n  >> Open http://localhost:5000 in your browser\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
