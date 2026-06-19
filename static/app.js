/* ================================================================
   Play Against Yourself AI — Client-side Game Logic  (v4)
   ================================================================
   Features: sounds, undo, resign, PGN export, keyboard shortcuts,
   drag & drop, premove, FEN cache optimisation.
   ================================================================ */

// ─── Sound Manager (Web Audio API) ──────────────────────────────
class SoundManager {
  constructor() {
    this.enabled = true;
    this.ctx = null;
  }
  _ensure() {
    if (!this.ctx) {
      this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (this.ctx.state === "suspended") this.ctx.resume();
  }
  _tone(freq, dur, type = "sine", vol = 0.15) {
    if (!this.enabled) return;
    this._ensure();
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.type = type;
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(vol, this.ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + dur);
    osc.connect(gain).connect(this.ctx.destination);
    osc.start(); osc.stop(this.ctx.currentTime + dur);
  }
  move()    { this._tone(600, 0.08, "sine", 0.12); }
  capture() { this._tone(300, 0.15, "triangle", 0.18); }
  check()   { this._tone(880, 0.12, "square", 0.10); setTimeout(() => this._tone(1100, 0.10, "square", 0.08), 100); }
  castle()  { this._tone(500, 0.06, "sine", 0.10); setTimeout(() => this._tone(650, 0.06, "sine", 0.10), 70); }
  gameOver(){ this._tone(440, 0.3, "sine", 0.12); setTimeout(() => this._tone(554, 0.3, "sine", 0.10), 150); setTimeout(() => this._tone(659, 0.5, "sine", 0.10), 300); }
  illegal() { this._tone(150, 0.15, "sawtooth", 0.08); }
  toggle()  { this.enabled = !this.enabled; return this.enabled; }
}

const PIECE_CHAR = {
  K: "\u265A", Q: "\u265B", R: "\u265C", B: "\u265D", N: "\u265E", P: "\u265F",
  k: "\u265A", q: "\u265B", r: "\u265C", b: "\u265D", n: "\u265E", p: "\u265F",
};

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
    this.flipped        = false;
    this.gameOver       = false;
    this.gameStarted    = false;
    this.aiThinking     = false;
    this.isCheck        = false;
    this.lastMove       = null;
    this.moveList       = [];
    this.pendingPromo   = null;
    this.playerLoaded   = false;
    this.loadedUsername  = null;

    // FEN cache
    this._cachedFen     = null;
    this._cachedBoard   = null;

    // Sound
    this.sound = new SoundManager();

    // Drag state
    this._dragging      = false;
    this._dragSquare    = null;
    this._dragGhost     = null;

    // Premove
    this._premove       = null;

    // Timer state
    this.timeMinutes    = 10;
    this.playerTime     = 10 * 60;
    this.aiTime         = 10 * 60;
    this.clockInterval  = null;
    this.clockSide      = null;

    // DOM cache
    this.boardEl    = document.getElementById("board");
    this.wrapperEl  = document.getElementById("board-wrapper");
    this.statusEl   = document.getElementById("status");
    this.movesEl    = document.getElementById("moves");
    this.topClock   = document.getElementById("top-clock");
    this.botClock   = document.getElementById("bottom-clock");
    this.thinkingEl = document.getElementById("thinking-overlay");

    this._bind();
    this._renderBoard();
    this._renderPlayers();
    this._renderClocks();
  }

  // ──────────────── Event binding ────────────────
  _bind() {
    // Colour buttons
    for (const b of document.querySelectorAll(".seg-btn"))
      b.addEventListener("click", () => this._setColor(b.dataset.color));

    // Load player button
    document.getElementById("load-player").addEventListener("click", () => this._loadPlayer());

    // Enter key on username input
    document.getElementById("username-input").addEventListener("keydown", (e) => {
      if (e.key === "Enter") this._loadPlayer();
    });

    // Time control buttons
    for (const b of document.querySelectorAll(".time-btn")) {
      b.addEventListener("click", () => {
        this.timeMinutes = +b.dataset.minutes;
        for (const x of document.querySelectorAll(".time-btn"))
          x.classList.toggle("active", x === b);
        if (!this.gameStarted) {
          this.playerTime = this.timeMinutes * 60;
          this.aiTime     = this.timeMinutes * 60;
          this._renderClocks();
        }
      });
    }

    // Actions
    document.getElementById("new-game").addEventListener("click", () => this.newGame());
    document.getElementById("flip-board").addEventListener("click", () => this._flip());
    document.getElementById("undo-btn").addEventListener("click", () => this._undo());
    document.getElementById("resign-btn").addEventListener("click", () => this._resign());
    document.getElementById("export-pgn").addEventListener("click", () => this._exportPGN());
    document.getElementById("go-play-again").addEventListener("click", () => {
      document.getElementById("game-over-modal").classList.remove("active");
      this.newGame();
    });

    // Sound toggle
    document.getElementById("sound-toggle").addEventListener("click", () => {
      const on = this.sound.toggle();
      document.getElementById("sound-toggle").textContent = on ? "🔊" : "🔇";
      document.getElementById("sound-toggle").classList.toggle("muted", !on);
    });

    // Board clicks
    this.boardEl.addEventListener("click",       (e) => this._onSquareClick(e));
    this.boardEl.addEventListener("contextmenu", (e) => { e.preventDefault(); this.selectedSquare = null; this._premove = null; this._renderBoard(); });

    // Drag and drop
    this.boardEl.addEventListener("mousedown", (e) => this._onDragStart(e));
    document.addEventListener("mousemove", (e) => this._onDragMove(e));
    document.addEventListener("mouseup", (e) => this._onDragEnd(e));

    // Keyboard shortcuts
    document.addEventListener("keydown", (e) => this._onKeyDown(e));
  }

  // ──────────────── Keyboard shortcuts ────────────────
  _onKeyDown(e) {
    // Don't intercept when typing in input
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

    const key = e.key.toLowerCase();
    if (key === "n") this.newGame();
    else if (key === "f") this._flip();
    else if (key === "z" || (e.ctrlKey && key === "z")) { e.preventDefault(); this._undo(); }
    else if (key === "m") document.getElementById("sound-toggle").click();
    else if (key === "r") this._resign();
    else if (key === "escape") { this.selectedSquare = null; this._premove = null; this._renderBoard(); }
  }

  // ──────────────── Drag and Drop ────────────────
  _onDragStart(e) {
    if (!this.gameStarted || this.gameOver) return;
    const el = e.target.closest(".square");
    if (!el) return;
    const sq = el.dataset.square;
    const p = this._pieceAt(sq);
    if (!p || !this._own(p)) return;
    if (!this.legalMoves.some(m => m.from === sq) && !this.aiThinking) return;

    e.preventDefault();
    this._dragging = true;
    this._dragSquare = sq;
    this.selectedSquare = sq;
    this._renderBoard();

    // Create ghost
    const pcls = p === p.toUpperCase() ? "white-piece" : "black-piece";
    const ghost = document.createElement("div");
    ghost.className = `drag-ghost piece ${pcls}`;
    ghost.textContent = PIECE_CHAR[p];
    ghost.style.left = e.clientX + "px";
    ghost.style.top = e.clientY + "px";
    document.body.appendChild(ghost);
    this._dragGhost = ghost;
  }

  _onDragMove(e) {
    if (!this._dragging || !this._dragGhost) return;
    this._dragGhost.style.left = e.clientX + "px";
    this._dragGhost.style.top = e.clientY + "px";
  }

  _onDragEnd(e) {
    if (!this._dragging) return;
    this._dragging = false;
    if (this._dragGhost) {
      this._dragGhost.remove();
      this._dragGhost = null;
    }

    // Find the square under cursor
    const el = document.elementFromPoint(e.clientX, e.clientY);
    const sqEl = el ? el.closest(".square") : null;
    if (!sqEl) { this.selectedSquare = null; this._renderBoard(); return; }

    const toSq = sqEl.dataset.square;
    if (toSq === this._dragSquare) {
      // Clicked same square — keep selected for click-click
      return;
    }

    // Try to make the move
    this._tryMove(this._dragSquare, toSq);
    this._dragSquare = null;
  }

  // ──────────────── Clock helpers ────────────────
  _formatTime(secs) {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }

  _renderClocks() {
    const playerIsBottom = !this.flipped;
    const playerClock = playerIsBottom ? this.botClock : this.topClock;
    const aiClock     = playerIsBottom ? this.topClock : this.botClock;

    if (!playerClock || !aiClock) return;

    playerClock.textContent = this._formatTime(this.playerTime);
    aiClock.textContent     = this._formatTime(this.aiTime);

    playerClock.classList.toggle("active-clock", this.clockSide === "player");
    aiClock.classList.toggle("active-clock",     this.clockSide === "ai");

    playerClock.classList.toggle("low-time", this.playerTime < 30 && this.clockSide === "player");
    aiClock.classList.toggle("low-time",     this.aiTime < 30 && this.clockSide === "ai");
  }

  _stopClock() {
    if (this.clockInterval) {
      clearInterval(this.clockInterval);
      this.clockInterval = null;
    }
    this.clockSide = null;
    this._renderClocks();
  }

  _startClock(side) {
    this._stopClock();
    this.clockSide = side;
    this.clockInterval = setInterval(() => {
      if (side === "player") {
        this.playerTime = Math.max(0, this.playerTime - 1);
        if (this.playerTime === 0) { this._stopClock(); this._onTimeout("player"); }
      } else {
        this.aiTime = Math.max(0, this.aiTime - 1);
        if (this.aiTime === 0) { this._stopClock(); this._onTimeout("ai"); }
      }
      this._renderClocks();
    }, 1000);
    this._renderClocks();
  }

  _onTimeout(loser) {
    this.gameOver = true;
    const playerLost = loser === "player";
    const result = playerLost
      ? (this.playerColor === "white" ? "0-1" : "1-0")
      : (this.playerColor === "white" ? "1-0" : "0-1");
    this._latestResult      = result;
    this._latestTermination = "Timeout";
    this.sound.gameOver();
    setTimeout(() => this._showGameOver({ result, termination: "Timeout" }), 200);
  }

  // ──────────────── Load Player ────────────────
  async _loadPlayer() {
    const input = document.getElementById("username-input");
    const btn   = document.getElementById("load-player");
    const stats = document.getElementById("player-stats");
    const username = input.value.trim();

    if (!username) return;

    btn.classList.add("loading");
    btn.textContent = "Loading…";
    stats.querySelector(".stats-text").textContent = `Fetching games for ${username}…`;
    stats.querySelector(".stats-text").className = "stats-text";

    try {
      const data = await this._api("/api/load_player", { username });

      if (!data.success) {
        stats.querySelector(".stats-text").textContent = `Error: ${data.error}`;
        stats.querySelector(".stats-text").className = "stats-text";
        return;
      }

      this.playerLoaded  = true;
      this.loadedUsername = data.username;

      const ratingStr = data.rating ? ` (${data.rating})` : "";
      stats.querySelector(".stats-text").innerHTML =
        `<span class="stat-highlight">${data.username}${ratingStr}</span> loaded — ` +
        `<span class="stat-highlight">${data.games}</span> games, ` +
        `<span class="stat-highlight">${data.positions}</span> positions in book`;
      stats.querySelector(".stats-text").className = "stats-text loaded";

      this._setStatus("Press New Game to play!", "your-turn");
      this._renderPlayers();
    } catch (e) {
      stats.querySelector(".stats-text").textContent = `Connection error — is the server running?`;
      stats.querySelector(".stats-text").className = "stats-text";
    } finally {
      btn.classList.remove("loading");
      btn.textContent = "Load";
    }
  }

  // ──────────────── Settings ────────────────
  _setColor(c) {
    this.playerColor = c;
    for (const b of document.querySelectorAll(".seg-btn"))
      b.classList.toggle("active", b.dataset.color === c);
  }

  _flip() {
    this.flipped = !this.flipped;
    this._renderBoard();
    this._renderPlayers();
    this._renderCaptured();
    this._renderClocks();
  }

  // ──────────────── New Game ────────────────
  async newGame() {
    if (!this.playerLoaded) {
      this._setStatus("Load a player first!", "game-over");
      return;
    }

    this._stopClock();
    this.playerTime = this.timeMinutes * 60;
    this.aiTime     = this.timeMinutes * 60;

    this.moveList       = [];
    this.lastMove       = null;
    this.selectedSquare = null;
    this.gameOver       = false;
    this.gameStarted    = true;
    this.isCheck        = false;
    this.pendingPromo   = null;
    this._premove       = null;
    this._cachedFen     = null;
    this._cachedBoard   = null;
    this.flipped        = this.playerColor === "black";

    this._setStatus("Starting game…", "thinking");
    this._renderClocks();

    try {
      const data = await this._api("/api/start", {
        playerColor: this.playerColor,
      });

      this.fen        = data.fen;
      this.legalMoves = data.legalMoves;
      this.gameOver   = data.gameOver;
      this.isCheck    = data.isCheck;

      if (data.aiMove) {
        this.lastMove = { from: data.aiMove.from, to: data.aiMove.to };
        this.moveList.push({ san: data.aiMove.san, color: "white" });
        this.sound.move();
        this._startClock("player");
      } else {
        this._startClock("player");
      }

      this._renderAll();
      this._updateStatus(data);
    } catch {
      this._setStatus("Connection error", "game-over");
    }
  }

  // ──────────────── Board clicks ────────────────
  _onSquareClick(e) {
    if (!this.gameStarted || this.gameOver || this._dragging) return;
    const el = e.target.closest(".square");
    if (!el) return;
    const sq = el.dataset.square;

    // If AI is thinking, allow premove
    if (this.aiThinking) {
      this._handlePremove(sq);
      return;
    }

    if (this.selectedSquare) {
      if (sq === this.selectedSquare) {
        this.selectedSquare = null;
        return this._renderBoard();
      }

      // Try the move
      if (this._tryMove(this.selectedSquare, sq)) return;

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

  _tryMove(fromSq, toSq) {
    let hits = this.legalMoves.filter(m => m.from === fromSq && m.to === toSq);

    // Castling via rook click
    if (!hits.length) {
      const selPiece = this._pieceAt(fromSq);
      const clickPiece = this._pieceAt(toSq);
      if (selPiece && clickPiece && this._own(clickPiece)
          && selPiece.toUpperCase() === "K" && clickPiece.toUpperCase() === "R") {
        const rookFile = toSq.charCodeAt(0) - 97;
        const kingFile = fromSq.charCodeAt(0) - 97;
        const rank = fromSq[1];
        const castleTo = rookFile > kingFile ? "g" + rank : "c" + rank;
        hits = this.legalMoves.filter(m => m.from === fromSq && m.to === castleTo);
      }
    }

    if (hits.length) {
      if (hits.some(m => m.promotion)) {
        this.pendingPromo = { from: fromSq, to: toSq };
        return this._showPromoModal(), true;
      }
      this._sendMove(hits[0].uci);
      return true;
    }
    return false;
  }

  // ──────────────── Premove ────────────────
  _handlePremove(sq) {
    if (!this._premove) {
      const p = this._pieceAt(sq);
      if (p && this._own(p)) {
        this._premove = { from: sq, to: null };
        this._renderBoard();
      }
    } else if (!this._premove.to) {
      if (sq === this._premove.from) {
        this._premove = null;
      } else {
        this._premove.to = sq;
      }
      this._renderBoard();
    } else {
      this._premove = null;
      this._renderBoard();
    }
  }

  _executePremove() {
    if (!this._premove || !this._premove.to) { this._premove = null; return; }
    const { from, to } = this._premove;
    this._premove = null;

    let hits = this.legalMoves.filter(m => m.from === from && m.to === to);
    if (hits.length && !hits.some(m => m.promotion)) {
      this._sendMove(hits[0].uci);
    }
  }

  // ──────────────── Send move ────────────────
  async _sendMove(uci) {
    this.selectedSquare = null;
    this.aiThinking = true;
    this._setStatus("AI is thinking…", "thinking");
    this.wrapperEl.classList.add("thinking");
    if (this.thinkingEl) this.thinkingEl.classList.add("active");
    this._renderBoard();

    this._startClock("ai");

    try {
      const data = await this._api("/api/move", {
        fen: this.fen, move: uci,
      });
      if (!data.success) {
        this._setStatus("Illegal move", "game-over");
        this.aiThinking = false;
        this.wrapperEl.classList.remove("thinking");
        if (this.thinkingEl) this.thinkingEl.classList.remove("active");
        this._startClock("player");
        this.sound.illegal();
        return;
      }

      // Player move sound
      const playerMoveData = data.playerMove;
      if (playerMoveData) {
        this.moveList.push({ san: playerMoveData.san, color: this.playerColor });
        if (uci.includes("O") || (playerMoveData.san && (playerMoveData.san.startsWith("O")))) {
          this.sound.castle();
        } else if (data.isCheck) {
          this.sound.check();
        } else {
          this.sound.move();
        }
      }

      // AI move
      if (data.aiMove) {
        const aiCol = this.playerColor === "white" ? "black" : "white";
        this.moveList.push({ san: data.aiMove.san, color: aiCol });
        this.lastMove = { from: data.aiMove.from, to: data.aiMove.to };

        // AI move sound
        if (data.aiMove.san && data.aiMove.san.includes("x")) {
          this.sound.capture();
        } else {
          this.sound.move();
        }
      } else {
        this.lastMove = { from: playerMoveData.from, to: playerMoveData.to };
      }

      this.fen        = data.fen;
      this.legalMoves = data.legalMoves;
      this.gameOver   = data.gameOver;
      this.isCheck    = data.isCheck;
      this._cachedFen = null;
    } catch {
      this._setStatus("Connection error", "game-over");
    }

    this.aiThinking = false;
    this.wrapperEl.classList.remove("thinking");
    if (this.thinkingEl) this.thinkingEl.classList.remove("active");
    this._renderAll();

    if (this.gameOver) {
      this._stopClock();
      this.sound.gameOver();
      setTimeout(() => this._showGameOver({
        result: this._latestResult || "1/2-1/2",
        termination: this._latestTermination || "",
      }), 500);
    } else {
      this._startClock("player");
      if (this.isCheck) this.sound.check();

      // Execute premove if queued
      if (this._premove && this._premove.to) {
        setTimeout(() => this._executePremove(), 100);
      }
    }
  }

  // ──────────────── Undo ────────────────
  async _undo() {
    if (!this.gameStarted || this.gameOver || this.aiThinking) return;
    try {
      const data = await this._api("/api/undo", {});
      if (data.success) {
        this.fen        = data.fen;
        this.legalMoves = data.legalMoves || [];
        this.isCheck    = data.isCheck || false;
        this.gameOver   = false;
        this._cachedFen = null;
        this.lastMove   = null;

        // Remove last 2 moves from move list (player + AI)
        const undone = data.undone || 0;
        for (let i = 0; i < undone; i++) {
          if (this.moveList.length > 0) this.moveList.pop();
        }

        this._renderAll();
        this._setStatus("Your turn", "your-turn");
        this.sound.move();
      }
    } catch {
      this._setStatus("Undo failed", "game-over");
    }
  }

  // ──────────────── Resign ────────────────
  async _resign() {
    if (!this.gameStarted || this.gameOver || this.aiThinking) return;
    try {
      const data = await this._api("/api/resign", {});
      if (data.success) {
        this.gameOver = true;
        this._stopClock();
        this.sound.gameOver();
        this._showGameOver({ result: data.result, termination: "Resignation" });
      }
    } catch {
      this._setStatus("Resign failed", "game-over");
    }
  }

  // ──────────────── PGN Export ────────────────
  async _exportPGN() {
    try {
      const data = await this._api("/api/export_pgn", {
        moves: this.moveList,
        playerColor: this.playerColor,
        username: this.loadedUsername || "AI",
      });
      if (data.success && data.pgn) {
        const blob = new Blob([data.pgn], { type: "application/x-chess-pgn" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `game_${this.loadedUsername || "ai"}_${Date.now()}.pgn`;
        a.click();
        URL.revokeObjectURL(url);
      }
    } catch {
      this._setStatus("Export failed", "game-over");
    }
  }

  // API helper
  async _api(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (data.result)      this._latestResult      = data.result;
    if (data.termination) this._latestTermination  = data.termination;
    return data;
  }

  // ──────────────── Promotion modal ────────────────
  _showPromoModal() {
    const modal = document.getElementById("promotion-modal");
    const box   = document.getElementById("promotion-options");
    const white = this.playerColor === "white";
    const chars = [["q","♛"],["r","♜"],["b","♝"],["n","♞"]];
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
    title.textContent = won ? "You Win!" : draw ? "Draw" : `${this.loadedUsername || "AI"} Wins`;
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
    this._renderClocks();
    if (!this.gameOver) this._updateStatus({ gameOver: false, isCheck: this.isCheck });
  }

  _renderBoard() {
    const pieces    = this._getBoard();
    const turnChar  = this.fen.split(" ")[1];
    const kingInChk = this.isCheck ? this._findKing(turnChar === "w" ? "K" : "k") : null;

    const targets = new Set(), captures = new Set();
    if (this.selectedSquare && !this.aiThinking) {
      for (const m of this.legalMoves) {
        if (m.from === this.selectedSquare) {
          targets.add(m.to);
          if (m.capture) captures.add(m.to);
        }
      }
    }

    // Premove squares
    const premoveSquares = new Set();
    if (this._premove) {
      premoveSquares.add(this._premove.from);
      if (this._premove.to) premoveSquares.add(this._premove.to);
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
        if (premoveSquares.has(sq)) cls.push("premove");

        const piece = pieces[rank]?.[file] ?? null;
        if (piece) {
          cls.push("has-piece");
          if (this._own(piece)) cls.push("own-piece");
        }

        html += `<div class="${cls.join(" ")}" data-square="${sq}">`;

        if (piece) {
          const pcls = piece === piece.toUpperCase() ? "white-piece" : "black-piece";
          const anim = this.lastMove && sq === this.lastMove.to && !this.aiThinking ? " piece-animated" : "";
          const dragCls = this._dragging && sq === this._dragSquare ? " dragging" : "";
          html += `<span class="piece ${pcls}${anim}${dragCls}">${PIECE_CHAR[piece]}</span>`;
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
    const name = this.loadedUsername || "AI";
    const topCol    = this.flipped ? "white" : "black";
    const bottomCol = this.flipped ? "black" : "white";
    const isTopPlayer = this.playerColor === topCol;

    document.getElementById("top-icon").textContent    = topCol === "white" ? "♔" : "♚";
    document.getElementById("bottom-icon").textContent  = bottomCol === "white" ? "♔" : "♚";
    document.getElementById("top-name").textContent    = isTopPlayer ? "You" : name;
    document.getElementById("bottom-name").textContent = isTopPlayer ? name : "You";
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
                  r === "1/2-1/2" ? `Draw — ${data.termination || ""}` :
                  `${this.loadedUsername || "AI"} wins! ${data.termination || ""}`;
      return this._setStatus(txt.trim(), "game-over");
    }
    if (data.isCheck) return this._setStatus("Check!", "in-check");
    this._setStatus("Your turn", "your-turn");
  }

  _setStatus(text, cls) {
    this.statusEl.className = "status-row " + (cls || "");
    this.statusEl.querySelector(".status-text").textContent = text;
  }

  // ──────────────── Helpers (with FEN caching) ────────────────
  _getBoard() {
    if (this._cachedFen === this.fen && this._cachedBoard) return this._cachedBoard;
    this._cachedFen = this.fen;
    this._cachedBoard = this._parseFEN();
    return this._cachedBoard;
  }

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
    return this._getBoard()[r]?.[f] ?? null;
  }

  _own(p) { return this.playerColor === "white" ? p === p.toUpperCase() : p === p.toLowerCase(); }

  _findKing(k) {
    const board = this._getBoard();
    for (let r = 0; r < 8; r++)
      for (let f = 0; f < 8; f++)
        if (board[r][f] === k) return "abcdefgh"[f] + (r + 1);
    return null;
  }
}

// ─── Bootstrap ───
document.addEventListener("DOMContentLoaded", () => {
  window.game = new ChessGame();
  // Auto-load the default player
  const input = document.getElementById("username-input");
  if (input && input.value.trim()) {
    window.game._loadPlayer();
  }
});
