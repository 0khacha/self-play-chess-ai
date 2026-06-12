"""
Play Against Yourself AI – Web Interface

Uses CloneAgent (player book + neural network + SEE tactical filter)
to mimic a Chess.com player's style. No Stockfish required.

Run:
    python play.py

Then open http://localhost:5000 in your browser.
"""

import os
import sys
import logging

from flask import Flask, request, jsonify, send_from_directory
import chess

import config
from model.clone_agent import PlayerBook, CloneAgent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder="static", static_url_path="/static")

# Global agent state
_current_agent: CloneAgent | None = None
_current_username: str | None = None

# Neural model path (shared across all player loads)
_MODEL_PATH = os.path.join(config.MODELS_DIR, config.CHECKPOINT_NAME)


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
                "to":   chess.square_name(m.to_square),
                "uci":  m.uci(),
                "capture":   board.is_capture(m),
                "promotion": m.promotion is not None,
            }
        )
    return moves


def _board_state(board: chess.Board) -> dict:
    """Snapshot the board into a JSON-serialisable dict."""
    game_over = board.is_game_over(claim_draw=True)
    outcome   = board.outcome(claim_draw=True) if game_over else None
    return {
        "fen":        board.fen(),
        "legalMoves": _legal_moves(board) if not game_over else [],
        "isCheck":    board.is_check(),
        "gameOver":   game_over,
        "result":     outcome.result() if outcome else None,
        "termination": (
            outcome.termination.name.replace("_", " ").title() if outcome else None
        ),
    }


def _move_info(move: chess.Move, san: str) -> dict:
    return {
        "from": chess.square_name(move.from_square),
        "to":   chess.square_name(move.to_square),
        "uci":  move.uci(),
        "san":  san,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/load_player", methods=["POST"])
def load_player():
    """Load a Chess.com player's archive and create a CloneAgent."""
    global _current_agent, _current_username

    data     = request.json or {}
    username = data.get("username", "").strip()
    if not username:
        return jsonify({"success": False, "error": "Username is required"}), 400

    try:
        # Build the position book for this player
        book  = PlayerBook(username)
        stats = book.build()

        # Create the clone agent (neural model shared, book is per-player)
        _current_agent   = CloneAgent(book, _MODEL_PATH)
        _current_username = username

        return jsonify({
            "success":   True,
            "username":  stats["username"],
            "games":     stats["games"],
            "positions": stats["positions"],
            "rating":    stats["rating"],
            "neural":    _current_agent.neural.enabled,
        })
    except Exception as exc:
        logger.exception("Failed to load player '%s'", username)
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/start", methods=["POST"])
def start_game():
    """Start a new game.  If the player chose black, the AI plays first."""
    data         = request.json or {}
    player_color = data.get("playerColor", "white")

    if _current_agent is None:
        return jsonify({"success": False, "error": "No player loaded"}), 400

    board      = chess.Board()
    resp       = _board_state(board)
    resp["aiMove"] = None

    if player_color == "black":
        ai_move = _current_agent.select_move(board)
        san     = board.san(ai_move)
        board.push(ai_move)
        resp = _board_state(board)
        resp["aiMove"] = _move_info(ai_move, san)

    return jsonify(resp)


@app.route("/api/move", methods=["POST"])
def make_move():
    """Accept the player's move, validate it, then let the AI respond."""
    data     = request.json
    fen      = data["fen"]
    move_uci = data["move"]

    if _current_agent is None:
        return jsonify({"success": False, "error": "No player loaded"}), 400

    board = chess.Board(fen)

    try:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return jsonify({"success": False, "error": "Illegal move"}), 400
    except Exception:
        return jsonify({"success": False, "error": "Invalid move string"}), 400

    player_san = board.san(move)
    board.push(move)

    resp = {"success": True, "playerMove": _move_info(move, player_san), "aiMove": None}
    resp.update(_board_state(board))

    if board.is_game_over(claim_draw=True):
        return jsonify(resp)

    ai_move = _current_agent.select_move(board)
    ai_san  = board.san(ai_move)
    board.push(ai_move)

    resp["aiMove"] = _move_info(ai_move, ai_san)
    resp.update(_board_state(board))
    return jsonify(resp)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    print()
    print("=" * 55)
    print("  Play Against Yourself AI")
    print("=" * 55)

    # Neural model status
    if os.path.exists(_MODEL_PATH):
        print(f"  Neural model : {_MODEL_PATH}")
    else:
        print(f"  Neural model : NOT FOUND – book-only fallback")
        print(f"    (run  python train.py  to train the model)")

    # Pre-load the default player
    default_user = config.CHESS_COM_USERNAME
    print(f"  Pre-loading  : {default_user} ...")

    try:
        book  = PlayerBook(default_user)
        stats = book.build()
        _current_agent   = CloneAgent(book, _MODEL_PATH)
        _current_username = default_user
        neural_ok = _current_agent.neural.enabled
        print(f"    [OK] {stats['games']} games, {stats['positions']} positions, "
              f"rating={stats['rating']}, neural={'on' if neural_ok else 'off'}")
    except Exception as exc:
        print(f"    [WARN] {exc}")
        print("    Load a player via the UI instead.")

    print()
    print("  >> Open http://localhost:5000 in your browser")
    print("=" * 55)
    print()

    app.run(host="0.0.0.0", port=5000, debug=False)
