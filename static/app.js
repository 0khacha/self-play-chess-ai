/* ================================================================
   Play Against Yourself AI — Client-side Game Logic  (v2)
   ================================================================ */

/* ─── Use filled glyphs for BOTH colours ───
   White pieces get a stacked text-shadow outline via CSS,
   giving crisp, platform-consistent rendering on Windows.  */
const PIECE_CHAR = {
  K: "\u265A", Q: "\u265B", R: "\u265C", B: "\u265D", N: "\u265E", P: "\u265F",
  k: "\u265A", q: "\u265B", r: "\u265C", b: "\u265D", n: "\u265E", p: "\u265F",
};

const STYLE_NAMES  = { 0: "Normal", 1: "Aggressive", 2: "Defensive" };
const PIECE_VALUES = { K:0,Q:9,R:5,B:3,N:3,P:1, k:0,q:9,r:5,b:3,n:3,p:1 };
const START_COUNTS = { K:1,Q:1,R:2,B:2,N:2,P:8, k:1,q:1,r:2,b:2,n:2,p:8 };
const START_FEN    = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// ─── Main Game Class ─────────────────────────────────────────────
class ChessGame {
  constructor() {
    this.fen            = START_FEN;
    this.legalMoves     = [];
    this.selectedSquare = null;
    this.playerColor    = "white";
    this.aiStyle        = 0;
    this.flipped        = false;
    this.gameOver       = false;
    this.gameStarted    = false;
    this.aiThinking     = false;
    this.isCheck        = false;
    this.lastMove       = null;           // { from, to }
    this.moveList       = [];             // [{ san, color }, ...]
    this.pendingPromo   = null;           // { from, to } awaiting choice

    // DOM cache
    this.boardEl   = document.getElementById("board");
    this.wrapperEl = document.getElementById("board-wrapper");
    this.statusEl  = document.getElementById("status");
    this.movesEl   = document.getElementById("moves");

    this._bind();
    this._renderBoard();
    this._renderPlayers();
  }

  // ──────────────── Event binding ────────────────
  _bind() {
    // Style buttons
    for (const b of document.querySelectorAll(".style-btn"))
      b.addEventListener("click", () => this._setStyle(+b.dataset.style));

    // Colour buttons
    for (const b of document.querySelectorAll(".color-btn"))
      b.addEventListener("click", () => this._setColor(b.dataset.color));

    // Actions
    document.getElementById("new-game").addEventListener("click", () => this.newGame());
    document.getElementById("flip-board").addEventListener("click", () => this._flip());
    document.getElementById("go-play-again").addEventListener("click", () => {
      document.getElementById("game-over-modal").classList.remove("active");
      this.newGame();
    });

    // Board clicks
    this.boardEl.addEventListener("click",       (e) => this._onSquareClick(e));
    this.boardEl.addEventListener("contextmenu", (e) => { e.preventDefault(); this.selectedSquare = null; this._renderBoard(); });
  }

  // ──────────────── Settings ────────────────
  _setStyle(s) {
    this.aiStyle = s;
    for (const b of document.querySelectorAll(".style-btn"))
      b.classList.toggle("active", +b.dataset.style === s);
    this._renderPlayers();
  }

  _setColor(c) {
    this.playerColor = c;
    for (const b of document.querySelectorAll(".color-btn"))
      b.classList.toggle("active", b.dataset.color === c);
  }

  _flip() {
    this.flipped = !this.flipped;
    this._renderBoard();
    this._renderPlayers();
    this._renderCaptured();
  }

  // ──────────────── New Game ────────────────
  async newGame() {
    this.moveList       = [];
    this.lastMove       = null;
    this.selectedSquare = null;
    this.gameOver       = false;
    this.gameStarted    = true;
    this.isCheck        = false;
    this.pendingPromo   = null;
    this.flipped        = this.playerColor === "black";

    this._setStatus("Starting game\u2026", "thinking");

    try {
      const data = await this._api("/api/start", {
        style: this.aiStyle,
        playerColor: this.playerColor,
      });

      this.fen        = data.fen;
      this.legalMoves = data.legalMoves;
      this.gameOver   = data.gameOver;
      this.isCheck    = data.isCheck;

      if (data.aiMove) {
        this.lastMove = { from: data.aiMove.from, to: data.aiMove.to };
        this.moveList.push({ san: data.aiMove.san, color: "white" });
      }

      this._renderAll();
      this._updateStatus(data);
    } catch {
      this._setStatus("Connection error", "game-over");
    }
  }

  // ──────────────── Board clicks ────────────────
  _onSquareClick(e) {
    if (!this.gameStarted || this.gameOver || this.aiThinking) return;
    const el = e.target.closest(".square");
    if (!el) return;
    const sq = el.dataset.square;

    if (this.selectedSquare) {
      // Clicking the same square → deselect
      if (sq === this.selectedSquare) {
        this.selectedSquare = null;
        return this._renderBoard();
      }

      // Legal move to this square?
      const hits = this.legalMoves.filter(m => m.from === this.selectedSquare && m.to === sq);
      if (hits.length) {
        if (hits.some(m => m.promotion)) {          // promotion
          this.pendingPromo = { from: this.selectedSquare, to: sq };
          return this._showPromoModal();
        }
        return this._sendMove(hits[0].uci);
      }

      // Click on another own piece → reselect
      const p = this._pieceAt(sq);
      if (p && this._own(p)) {
        this.selectedSquare = sq;
      } else {
        this.selectedSquare = null;
      }
      this._renderBoard();
    } else {
      const p = this._pieceAt(sq);
      if (p && this._own(p) && this.legalMoves.some(m => m.from === sq)) {
        this.selectedSquare = sq;
        this._renderBoard();
      }
    }
  }

  // ──────────────── Send move ────────────────
  async _sendMove(uci) {
    this.selectedSquare = null;
    this.aiThinking = true;
    this._setStatus("AI is thinking\u2026", "thinking");
    this.wrapperEl.classList.add("thinking");
    this._renderBoard();   // show cleared selection immediately

    try {
      const data = await this._api("/api/move", {
        fen: this.fen, move: uci, style: this.aiStyle,
      });
      if (!data.success) {
        this._setStatus("Illegal move", "game-over");
        this.aiThinking = false;
        this.wrapperEl.classList.remove("thinking");
        return;
      }

      // Player move
      this.moveList.push({ san: data.playerMove.san, color: this.playerColor });

      // AI move
      if (data.aiMove) {
        const aiCol = this.playerColor === "white" ? "black" : "white";
        this.moveList.push({ san: data.aiMove.san, color: aiCol });
        this.lastMove = { from: data.aiMove.from, to: data.aiMove.to };
      } else {
        this.lastMove = { from: data.playerMove.from, to: data.playerMove.to };
      }

      this.fen        = data.fen;
      this.legalMoves = data.legalMoves;
      this.gameOver   = data.gameOver;
      this.isCheck    = data.isCheck;
    } catch {
      this._setStatus("Connection error", "game-over");
    }

    this.aiThinking = false;
    this.wrapperEl.classList.remove("thinking");
    this._renderAll();
    if (this.gameOver) {
      const d = { result: this._resultFromMoves(), termination: "" };
      // grab latest from server (already set above)
      d.result      = this.moveList.length ? this._latestResult : "1/2-1/2";
      d.termination = this._latestTermination || "";
      setTimeout(() => this._showGameOver(d), 500);
    }
  }

  // Override _sendMove's game-over handling to use server data:
  async _sendMoveReal() {} // placeholder; the logic above handles it inline.

  // Patch: capture result/termination from server response
  async _api(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    // stash server result info
    if (data.result)      this._latestResult      = data.result;
    if (data.termination) this._latestTermination  = data.termination;
    return data;
  }

  // ──────────────── Promotion modal ────────────────
  _showPromoModal() {
    const modal = document.getElementById("promotion-modal");
    const box   = document.getElementById("promotion-options");
    const white = this.playerColor === "white";
    const chars = white
      ? [["q","♛"],["r","♜"],["b","♝"],["n","♞"]]
      : [["q","♛"],["r","♜"],["b","♝"],["n","♞"]];
    const cls = white ? "white-piece" : "black-piece";

    box.innerHTML = chars.map(([k, ch]) =>
      `<button class="promo-btn ${cls}" data-piece="${k}">${ch}</button>`
    ).join("");

    for (const btn of box.querySelectorAll(".promo-btn")) {
      btn.addEventListener("click", () => {
        const uci = this.pendingPromo.from + this.pendingPromo.to + btn.dataset.piece;
        this.pendingPromo = null;
        modal.classList.remove("active");
        this._sendMove(uci);
      });
    }
    modal.classList.add("active");
  }

  // ──────────────── Game over modal ────────────────
  _showGameOver(data) {
    const modal  = document.getElementById("game-over-modal");
    const icon   = document.getElementById("go-icon");
    const title  = document.getElementById("go-title");
    const detail = document.getElementById("go-detail");

    const r = data.result || this._latestResult || "1/2-1/2";
    const won = (this.playerColor === "white" && r === "1-0")
             || (this.playerColor === "black" && r === "0-1");
    const draw = r === "1/2-1/2";

    icon.textContent  = won ? "🎉" : draw ? "🤝" : "💀";
    title.textContent = won ? "You Win!" : draw ? "Draw" : "AI Wins";
    detail.textContent = data.termination || this._latestTermination || "";

    this._updateStatus({ gameOver: true, result: r, termination: data.termination || this._latestTermination, isCheck: false });
    modal.classList.add("active");
  }

  // ──────────────── Rendering ────────────────
  _renderAll() {
    this._renderBoard();
    this._renderMoves();
    this._renderPlayers();
    this._renderCaptured();
    if (!this.gameOver) this._updateStatus({ gameOver: false, isCheck: this.isCheck });
  }

  _renderBoard() {
    const pieces    = this._parseFEN();
    const turnChar  = this.fen.split(" ")[1];             // 'w' or 'b'
    const kingInChk = this.isCheck ? this._findKing(turnChar === "w" ? "K" : "k") : null;

    // Unique legal targets from selected square
    const targets = new Set(), captures = new Set();
    if (this.selectedSquare) {
      for (const m of this.legalMoves) {
        if (m.from === this.selectedSquare) {
          targets.add(m.to);
          if (m.capture) captures.add(m.to);
        }
      }
    }

    let html = "";
    for (let row = 0; row < 8; row++) {
      for (let col = 0; col < 8; col++) {
        const rank = this.flipped ? row     : 7 - row;
        const file = this.flipped ? 7 - col : col;
        const sq   = "abcdefgh"[file] + (rank + 1);
        const light = (rank + file) % 2 === 1;

        const cls = ["square", light ? "light" : "dark"];
        if (sq === this.selectedSquare) cls.push("selected");
        if (this.lastMove && (sq === this.lastMove.from || sq === this.lastMove.to)) cls.push("last-move");
        if (kingInChk === sq) cls.push("in-check");
        if (targets.has(sq)) cls.push("legal-target");

        const piece = pieces[rank]?.[file] ?? null;
        if (piece) {
          cls.push("has-piece");
          if (this._own(piece)) cls.push("own-piece");
        }

        html += `<div class="${cls.join(" ")}" data-square="${sq}">`;

        if (piece) {
          const pcls = piece === piece.toUpperCase() ? "white-piece" : "black-piece";
          const anim = this.lastMove && sq === this.lastMove.to && !this.aiThinking ? " piece-animated" : "";
          html += `<span class="piece ${pcls}${anim}">${PIECE_CHAR[piece]}</span>`;
        }

        // Hints
        if (targets.has(sq) && sq !== this.selectedSquare) {
          html += captures.has(sq) || piece
            ? `<div class="capture-hint"></div>`
            : `<div class="move-hint"></div>`;
        }

        // Coords
        if (col === 0) html += `<span class="coord coord-rank">${rank + 1}</span>`;
        if (row === 7) html += `<span class="coord coord-file">${"abcdefgh"[file]}</span>`;

        html += `</div>`;
      }
    }
    this.boardEl.innerHTML = html;
  }

  _renderMoves() {
    if (!this.moveList.length) {
      this.movesEl.innerHTML = `<div class="moves-empty">No moves yet</div>`;
      return;
    }
    let h = "";
    for (let i = 0; i < this.moveList.length; i += 2) {
      const n = (i >> 1) + 1;
      const w = this.moveList[i];
      const b = this.moveList[i + 1];
      h += `<div class="move-row">
        <span class="move-num">${n}.</span>
        <span class="move-san">${w.san}</span>
        <span class="move-san${b ? "" : " empty"}">${b ? b.san : ""}</span>
      </div>`;
    }
    this.movesEl.innerHTML = h;
    this.movesEl.scrollTop = this.movesEl.scrollHeight;
  }

  _renderPlayers() {
    const name = STYLE_NAMES[this.aiStyle];
    const topCol    = this.flipped ? "white" : "black";
    const bottomCol = this.flipped ? "black" : "white";
    const isTopPlayer = this.playerColor === topCol;

    document.getElementById("top-icon").textContent    = topCol === "white" ? "♔" : "♚";
    document.getElementById("bottom-icon").textContent  = bottomCol === "white" ? "♔" : "♚";
    document.getElementById("top-name").textContent    = isTopPlayer ? "You" : `AI \u00B7 ${name}`;
    document.getElementById("bottom-name").textContent = isTopPlayer ? `AI \u00B7 ${name}` : "You";
  }

  _renderCaptured() {
    if (!this.fen) return;
    const counts = {};
    for (const ch of this.fen.split(" ")[0]) {
      if (START_COUNTS[ch] !== undefined) counts[ch] = (counts[ch] || 0) + 1;
    }

    const capW = [], capB = [];
    for (const [p, n] of Object.entries(START_COUNTS)) {
      const d = n - (counts[p] || 0);
      for (let i = 0; i < d; i++) (p === p.toUpperCase() ? capW : capB).push(p);
    }

    const valSort = (a, b) => PIECE_VALUES[b.toUpperCase()] - PIECE_VALUES[a.toUpperCase()];
    capW.sort(valSort); capB.sort(valSort);

    const wVal = capB.reduce((s, p) => s + PIECE_VALUES[p], 0);
    const bVal = capW.reduce((s, p) => s + PIECE_VALUES[p], 0);
    const adv  = wVal - bVal;

    const topCol = this.flipped ? "white" : "black";
    const topC   = document.getElementById("top-captured");
    const botC   = document.getElementById("bottom-captured");
    const topD   = document.getElementById("top-diff");
    const botD   = document.getElementById("bottom-diff");

    const render = arr => arr.map(p => `<span>${PIECE_CHAR[p]}</span>`).join("");

    if (topCol === "black") {
      topC.innerHTML = render(capW);
      botC.innerHTML = render(capB);
      topD.textContent = adv < 0 ? `+${-adv}` : "";
      botD.textContent = adv > 0 ? `+${adv}` : "";
    } else {
      topC.innerHTML = render(capB);
      botC.innerHTML = render(capW);
      topD.textContent = adv > 0 ? `+${adv}` : "";
      botD.textContent = adv < 0 ? `+${-adv}` : "";
    }
  }

  // ──────────────── Status ────────────────
  _updateStatus(data) {
    if (!data) return;
    if (data.gameOver) {
      const r = data.result;
      const won = (this.playerColor === "white" && r === "1-0")
               || (this.playerColor === "black" && r === "0-1");
      const txt = won ? `You win! ${data.termination || ""}` :
                  r === "1/2-1/2" ? `Draw \u2014 ${data.termination || ""}` :
                  `AI wins! ${data.termination || ""}`;
      return this._setStatus(txt.trim(), "game-over");
    }
    if (data.isCheck) return this._setStatus("Check!", "in-check");
    this._setStatus("Your turn", "your-turn");
  }

  _setStatus(text, cls) {
    this.statusEl.className = "status " + (cls || "");
    this.statusEl.querySelector(".status-text").textContent = text;
  }

  // ──────────────── Helpers ────────────────
  _parseFEN() {
    const ranks = this.fen.split(" ")[0].split("/");
    const board = [];
    for (let r = 0; r < 8; r++) {
      const row = [];
      for (const ch of ranks[7 - r]) {
        if (ch >= "1" && ch <= "8") for (let i = 0; i < +ch; i++) row.push(null);
        else row.push(ch);
      }
      board.push(row);
    }
    return board;
  }

  _pieceAt(sq) {
    const f = sq.charCodeAt(0) - 97, r = +sq[1] - 1;
    return this._parseFEN()[r]?.[f] ?? null;
  }

  _own(p) { return this.playerColor === "white" ? p === p.toUpperCase() : p === p.toLowerCase(); }

  _findKing(k) {
    const board = this._parseFEN();
    for (let r = 0; r < 8; r++)
      for (let f = 0; f < 8; f++)
        if (board[r][f] === k) return "abcdefgh"[f] + (r + 1);
    return null;
  }
}

// ─── Bootstrap ───
document.addEventListener("DOMContentLoaded", () => { window.game = new ChessGame(); });
