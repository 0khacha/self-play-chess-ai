"""
Play Against Yourself AI – Web Interface

Uses CloneAgent (player book + neural network + SEE tactical filter)
to mimic a Chess.com player's style. No Stockfish required.

Run:
    python play.py

Then open http://localhost:5000 in your browser.
"""

from __future__ import annotations

import io
import os
import sys
import logging
from datetime import date
from typing import Optional

from flask import Flask, request, jsonify, send_from_directory
import chess
import chess.pgn

import config
from model.clone_agent import PlayerBook, CloneAgent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder="static", static_url_path="/static")

# Global agent state
_current_agent: Optional[CloneAgent] = None
_current_username: Optional[str] = None

# Board / game tracking  (session-based, single-player server)
_current_board: Optional[chess.Board] = None
_move_history: list[chess.Move] = []
_game_id: int = 0
_player_color: Optional[str] = None

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


def _build_pgn(
    moves: list[chess.Move],
    player_color: str,
    username: str,
    result: str,
) -> str:
    """Build a PGN string from the recorded move history."""
    game = chess.pgn.Game()

    # Headers
    game.headers["Event"] = "Play Against Yourself AI"
    game.headers["Site"] = "localhost"
    game.headers["Date"] = date.today().strftime("%Y.%m.%d")

    if player_color == "white":
        game.headers["White"] = username or "Player"
        game.headers["Black"] = f"CloneAI ({_current_username or 'unknown'})"
    else:
        game.headers["White"] = f"CloneAI ({_current_username or 'unknown'})"
        game.headers["Black"] = username or "Player"

    game.headers["Result"] = result

    # Replay moves onto the game node tree
    node = game
    board = chess.Board()
    for move in moves:
        if move in board.legal_moves:
            node = node.add_variation(move)
            board.push(move)

    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    return game.accept(exporter)


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
    global _current_board, _move_history, _game_id, _player_color

    data         = request.json or {}
    player_color = data.get("playerColor", "white")

    if _current_agent is None:
        return jsonify({"success": False, "error": "No player loaded"}), 400

    # Initialise fresh board with full history tracking
    _game_id += 1
    _current_board = chess.Board()
    _move_history = []
    _player_color = player_color

    board = _current_board
    resp  = _board_state(board)
    resp["aiMove"] = None
    resp["gameId"] = _game_id

    if player_color == "black":
        ai_move = _current_agent.select_move(board)
        san     = board.san(ai_move)
        board.push(ai_move)
        _move_history.append(ai_move)
        resp = _board_state(board)
        resp["aiMove"] = _move_info(ai_move, san)
        resp["gameId"] = _game_id

    return jsonify(resp)


@app.route("/api/move", methods=["POST"])
def make_move():
    """Accept the player's move, validate it, then let the AI respond."""
    global _current_board, _move_history

    data     = request.json
    move_uci = data["move"]

    if _current_agent is None:
        return jsonify({"success": False, "error": "No player loaded"}), 400

    # Use the tracked board (full history) for proper draw detection.
    # Fall back to FEN reconstruction if no tracked board exists (e.g. a
    # client reconnected with only a FEN).
    if _current_board is None:
        fen = data.get("fen")
        if fen is None:
            return jsonify({"success": False, "error": "No active game"}), 400
        _current_board = chess.Board(fen)
        _move_history = []

    board = _current_board

    try:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return jsonify({"success": False, "error": "Illegal move"}), 400
    except Exception:
        return jsonify({"success": False, "error": "Invalid move string"}), 400

    player_san = board.san(move)
    board.push(move)
    _move_history.append(move)

    resp = {"success": True, "playerMove": _move_info(move, player_san), "aiMove": None}
    resp.update(_board_state(board))

    if board.is_game_over(claim_draw=True):
        return jsonify(resp)

    ai_move = _current_agent.select_move(board)
    ai_san  = board.san(ai_move)
    board.push(ai_move)
    _move_history.append(ai_move)

    resp["aiMove"] = _move_info(ai_move, ai_san)
    resp.update(_board_state(board))
    return jsonify(resp)


@app.route("/api/undo", methods=["POST"])
def undo_move():
    """
    Undo the last TWO half-moves (player move + AI response) so it is
    the player's turn again.  Returns the resulting board state.
    """
    global _current_board, _move_history

    if _current_board is None:
        return jsonify({"success": False, "error": "No active game"}), 400

    board = _current_board

    # We need at least two moves on the stack to undo a full pair
    moves_to_undo = min(2, len(_move_history))
    if moves_to_undo == 0:
        return jsonify({"success": False, "error": "Nothing to undo"}), 400

    for _ in range(moves_to_undo):
        board.pop()
        _move_history.pop()

    resp = {"success": True}
    resp.update(_board_state(board))
    return jsonify(resp)


@app.route("/api/resign", methods=["POST"])
def resign_game():
    """
    The player resigns.  Returns the result string based on the player's
    colour (the side that resigned loses).
    """
    if _current_board is None:
        return jsonify({"success": False, "error": "No active game"}), 400

    # Determine result: the player loses
    if _player_color == "white":
        result = "0-1"
    else:
        result = "1-0"

    return jsonify({
        "success": True,
        "result": result,
        "termination": "Resignation",
    })


@app.route("/api/export_pgn", methods=["POST"])
def export_pgn():
    """
    Generate and return a PGN string for the current (or just-finished) game.

    Expected body keys (all optional – server-side state is preferred):
      - ``username``    – human player's name for PGN headers.
      - ``playerColor`` – ``"white"`` or ``"black"``.
    """
    data = request.json or {}
    username = data.get("username", "Player")
    player_color = data.get("playerColor", _player_color or "white")

    if not _move_history:
        return jsonify({"success": False, "error": "No moves to export"}), 400

    # Determine the result
    board = _current_board
    if board is not None and board.is_game_over(claim_draw=True):
        outcome = board.outcome(claim_draw=True)
        result = outcome.result() if outcome else "*"
    else:
        result = "*"

    pgn_str = _build_pgn(_move_history, player_color, username, result)

    return jsonify({"success": True, "pgn": pgn_str})


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
        print(f"    (run  python train_for_user.py  to train the model)")

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
