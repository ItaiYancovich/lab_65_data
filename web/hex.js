// Hex game engine -- a faithful port of alphazero_hex/hex_game.py.
// Board indexing, neighbour offsets, win-condition and canonical encoding all
// mirror the Python engine exactly so the trained network sees the same
// representation it was trained on.
'use strict';

const EMPTY = 0, BLACK = 1, WHITE = 2;
const OFFSETS = [[-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0]];

function neighbourTable(n) {
  const table = [];
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const nbs = [];
      for (const [dr, dc] of OFFSETS) {
        const rr = r + dr, cc = c + dc;
        if (rr >= 0 && rr < n && cc >= 0 && cc < n) nbs.push(rr * n + cc);
      }
      table.push(nbs);
    }
  }
  return table;
}
const _neiCache = new Map();
function getNei(n) {
  if (!_neiCache.has(n)) _neiCache.set(n, neighbourTable(n));
  return _neiCache.get(n);
}

function other(p) { return p === BLACK ? WHITE : BLACK; }

class HexBoard {
  constructor(n) {
    this.n = n;
    this.ncells = n * n;
    this.board = new Uint8Array(this.ncells);
    // union-find over cells + 4 virtual edge nodes: TOP, BOTTOM, LEFT, RIGHT
    this.TOP = this.ncells; this.BOTTOM = this.ncells + 1;
    this.LEFT = this.ncells + 2; this.RIGHT = this.ncells + 3;
    this.parent = new Int32Array(this.ncells + 4);
    for (let i = 0; i < this.parent.length; i++) this.parent[i] = i;
    this.toMove = BLACK;
    this.winner = 0;
    this.moveCount = 0;
    this.nei = getNei(n);
  }

  copy() {
    const b = Object.create(HexBoard.prototype);
    b.n = this.n; b.ncells = this.ncells;
    b.board = this.board.slice();
    b.TOP = this.TOP; b.BOTTOM = this.BOTTOM; b.LEFT = this.LEFT; b.RIGHT = this.RIGHT;
    b.parent = this.parent.slice();
    b.toMove = this.toMove; b.winner = this.winner; b.moveCount = this.moveCount;
    b.nei = this.nei;
    return b;
  }

  find(x) {
    const p = this.parent;
    while (p[x] !== x) { p[x] = p[p[x]]; x = p[x]; }
    return x;
  }

  union(a, b) {
    const ra = this.find(a), rb = this.find(b);
    if (ra !== rb) { if (ra < rb) this.parent[ra] = rb; else this.parent[rb] = ra; }
  }

  legalMoves() {
    const out = [];
    for (let i = 0; i < this.ncells; i++) if (this.board[i] === EMPTY) out.push(i);
    return out;
  }

  isLegal(m) { return m >= 0 && m < this.ncells && this.board[m] === EMPTY && this.winner === 0; }

  play(move) {
    if (this.winner) throw new Error('game already decided');
    if (this.board[move] !== EMPTY) throw new Error(`cell ${move} occupied`);
    const player = this.toMove, n = this.n;
    this.board[move] = player;
    const r = (move / n) | 0, c = move % n;
    for (const nb of this.nei[move]) if (this.board[nb] === player) this.union(move, nb);
    if (player === BLACK) {
      if (r === 0) this.union(move, this.TOP);
      if (r === n - 1) this.union(move, this.BOTTOM);
      if (this.find(this.TOP) === this.find(this.BOTTOM)) this.winner = BLACK;
    } else {
      if (c === 0) this.union(move, this.LEFT);
      if (c === n - 1) this.union(move, this.RIGHT);
      if (this.find(this.LEFT) === this.find(this.RIGHT)) this.winner = WHITE;
    }
    this.moveCount++;
    this.toMove = other(player);
  }

  isTerminal() { return this.winner !== 0; }

  // Cells where `player` moving right now would complete a connection.
  // Reads straight off the union-find, mirroring heuristics.immediate_wins.
  immediateWins(player) {
    if (this.winner) return [];
    const n = this.n;
    let aRoot, bRoot;
    if (player === BLACK) { aRoot = this.find(this.TOP); bRoot = this.find(this.BOTTOM); }
    else { aRoot = this.find(this.LEFT); bRoot = this.find(this.RIGHT); }
    if (aRoot === bRoot) return [];
    const out = [];
    for (let cell = 0; cell < this.ncells; cell++) {
      if (this.board[cell]) continue;
      const r = (cell / n) | 0, c = cell % n;
      let hitsA = player === BLACK ? r === 0 : c === 0;
      let hitsB = player === BLACK ? r === n - 1 : c === n - 1;
      for (const nb of this.nei[cell]) {
        if (this.board[nb] !== player) continue;
        const root = this.find(nb);
        if (root === aRoot) hitsA = true; else if (root === bRoot) hitsB = true;
      }
      if (hitsA && hitsB) out.push(cell);
    }
    return out;
  }

  // Canonical (side-to-move) 3 x n x n planes, matching HexBoard.canonical_planes.
  canonicalPlanes() {
    const n = this.n, planes = new Float32Array(3 * n * n);
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const cell = r * n + c;
        const v = this.board[cell];
        let own, opp, or_, oc;
        if (this.toMove === BLACK) { or_ = r; oc = c; own = v === BLACK; opp = v === WHITE; }
        else { or_ = c; oc = r; own = v === WHITE; opp = v === BLACK; } // transpose
        const idx = or_ * n + oc;
        if (own) planes[idx] = 1;
        if (opp) planes[n * n + idx] = 1;
      }
    }
    planes.fill(1, 2 * n * n);
    return planes;
  }

  toCanonicalMove(move) {
    if (this.toMove === BLACK) return move;
    const n = this.n, r = (move / n) | 0, c = move % n;
    return c * n + r;
  }
  fromCanonicalMove(move) { return this.toCanonicalMove(move); } // involution
}

function moveToStr(move, n) {
  const r = (move / n) | 0, c = move % n;
  return String.fromCharCode(97 + c) + (r + 1);
}
function strToMove(text, n) {
  text = text.trim().toLowerCase();
  const c = text.charCodeAt(0) - 97;
  const r = parseInt(text.slice(1), 10) - 1;
  return r * n + c;
}
function rotate180Move(move, n) {
  const r = (move / n) | 0, c = move % n;
  return (n - 1 - r) * n + (n - 1 - c);
}

if (typeof module !== 'undefined') {
  module.exports = { HexBoard, EMPTY, BLACK, WHITE, other, getNei, moveToStr, strToMove, rotate180Move };
}
