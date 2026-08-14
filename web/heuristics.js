// Classical Hex knowledge -- a port of alphazero_hex/heuristics.py.
// Two-distance (Anshelevich) connection strength, bridge detection, and the
// move-scoring used by the rule-based and alpha-beta agents.
'use strict';

// Node: each file is an isolated module scope, so getNei needs importing.
// Browser: all <script> tags share one global scope, so hex.js's top-level
// `const getNei` is already visible here -- this block must do nothing then,
// and in particular must not `var`-declare anything (var hoists to the
// shared global scope regardless of whether this branch runs, which would
// collide with hex.js's `const` of the same name and throw a SyntaxError).
if (typeof require !== 'undefined' && typeof getNei === 'undefined') {
  Object.assign(globalThis, require('./hex.js'));
}

const INF = 1e6;

function paddedNeighbours(n) {
  const nei = getNei(n), ncells = n * n, out = [];
  for (let i = 0; i < ncells; i++) {
    const row = new Int32Array(6).fill(ncells);
    nei[i].forEach((v, k) => { row[k] = v; });
    out.push(row);
  }
  return out;
}
const _padCache = new Map();
function getPadded(n) {
  if (!_padCache.has(n)) _padCache.set(n, paddedNeighbours(n));
  return _padCache.get(n);
}

const BRIDGE_OFFSETS = [
  [[-1, -1], [[-1, 0], [0, -1]]],
  [[1, 1], [[1, 0], [0, 1]]],
  [[-2, 1], [[-1, 0], [-1, 1]]],
  [[2, -1], [[1, 0], [1, -1]]],
  [[-1, 2], [[-1, 1], [0, 1]]],
  [[1, -2], [[1, -1], [0, -1]]],
];

function bridgeTable(n) {
  const table = [];
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const entries = [];
      for (const [[dr, dc], carriers] of BRIDGE_OFFSETS) {
        const pr = r + dr, pc = c + dc;
        if (pr < 0 || pr >= n || pc < 0 || pc >= n) continue;
        let ok = true;
        const cells = [];
        for (const [cr, cc] of carriers) {
          const ar = r + cr, ac = c + cc;
          if (ar < 0 || ar >= n || ac < 0 || ac >= n) { ok = false; break; }
          cells.push(ar * n + ac);
        }
        if (ok) entries.push([pr * n + pc, cells[0], cells[1]]);
      }
      table.push(entries);
    }
  }
  return table;
}
const _bridgeCache = new Map();
function getBridgeTable(n) {
  if (!_bridgeCache.has(n)) _bridgeCache.set(n, bridgeTable(n));
  return _bridgeCache.get(n);
}

// Two-distance from one of `player`'s edges to every cell.
// source in {'top','bottom','left','right'}.
function twoDistance(boardFlat, n, player, source, opp) {
  const ncells = n * n, nei = getPadded(n);
  const isOwn = new Uint8Array(ncells), isOpp = new Uint8Array(ncells);
  for (let i = 0; i < ncells; i++) { isOwn[i] = boardFlat[i] === player ? 1 : 0; isOpp[i] = boardFlat[i] === opp ? 1 : 0; }

  const src = new Float64Array(ncells).fill(INF);
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    const i = r * n + c;
    if ((source === 'top' && r === 0) || (source === 'bottom' && r === n - 1) ||
        (source === 'left' && c === 0) || (source === 'right' && c === n - 1)) src[i] = 0;
  }
  for (let i = 0; i < ncells; i++) if (isOpp[i]) src[i] = INF;

  // Two candidate slots for the source edge: a board edge cannot be cut, so
  // touching it already counts as a redundant (doubly-connected) route --
  // mirrors the padding in heuristics.two_distance.
  let d = new Float64Array(ncells).fill(INF);
  const cand = new Float64Array(8);
  for (let iter = 0; iter < 2 * n + 4; iter++) {
    const nd = new Float64Array(ncells);
    let changed = false;
    for (let i = 0; i < ncells; i++) {
      if (isOpp[i]) { nd[i] = INF; if (d[i] !== INF) changed = true; continue; }
      const row = nei[i];
      for (let k = 0; k < 6; k++) cand[k] = row[k] === ncells ? INF : d[row[k]];
      cand[6] = src[i]; cand[7] = src[i];
      const arr = Array.from(cand).sort((a, b) => a - b);
      let val = isOwn[i] ? arr[0] : arr[1] + 1.0;
      if (val > d[i]) val = d[i]; // min with the previous iteration
      if (val !== d[i]) changed = true;
      nd[i] = val;
    }
    d = nd;
    if (!changed) break;
  }
  for (let i = 0; i < ncells; i++) if (d[i] > INF) d[i] = INF;
  return d;
}

function edgesFor(player, BLACK) { return player === BLACK ? ['top', 'bottom'] : ['left', 'right']; }

function connectionCost(boardFlat, n, player, BLACK, WHITE, EMPTY) {
  const opp = player === BLACK ? WHITE : BLACK;
  const [e1, e2] = edgesFor(player, BLACK);
  const d1 = twoDistance(boardFlat, n, player, e1, opp);
  const d2 = twoDistance(boardFlat, n, player, e2, opp);
  const ncells = n * n, total = new Float64Array(ncells);
  let best = INF;
  for (let i = 0; i < ncells; i++) {
    let t = d1[i] + d2[i];
    if (boardFlat[i] === EMPTY) t -= 1.0;
    if (t > INF) t = INF;
    total[i] = t;
    if (t < best) best = t;
  }
  return { potential: best, cost: total };
}

const _centralityCache = new Map();
function centrality(n) {
  if (_centralityCache.has(n)) return _centralityCache.get(n);
  const out = new Float64Array(n * n), mid = (n - 1) / 2;
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    const dist = Math.max(Math.abs(r - mid), Math.abs(c - mid)) / Math.max(mid, 1e-9);
    out[r * n + c] = 1.0 - dist;
  }
  _centralityCache.set(n, out);
  return out;
}

function moveScores(board, player, BLACK, WHITE, EMPTY, defenceWeight = 1.0) {
  const n = board.n, flat = board.board, opp = player === BLACK ? WHITE : BLACK;
  const my = connectionCost(flat, n, player, BLACK, WHITE, EMPTY);
  const op = connectionCost(flat, n, opp, BLACK, WHITE, EMPTY);
  const cen = centrality(n), ncells = n * n, score = new Float64Array(ncells);
  for (let i = 0; i < ncells; i++) {
    score[i] = -Math.min(my.cost[i], 1e5) - defenceWeight * Math.min(op.cost[i], 1e5) + 0.15 * cen[i];
    if (flat[i] !== EMPTY) score[i] = -Infinity;
  }
  return score;
}

// Static evaluation in [-1, 1] from `player`'s point of view, used as the
// leaf heuristic by the alpha-beta searcher. Mirrors heuristics.evaluate.
function evaluate(board, player, BLACK, WHITE) {
  if (board.winner) return board.winner === player ? 1.0 : -1.0;
  const n = board.n, opp = player === BLACK ? WHITE : BLACK;
  const my = connectionCost(board.board, n, player, BLACK, WHITE, 0);
  const op = connectionCost(board.board, n, opp, BLACK, WHITE, 0);
  if (my.potential >= INF) return -1.0;
  if (op.potential >= INF) return 1.0;
  const diff = op.potential - my.potential;
  return Math.tanh(diff / 3.0) * 0.99;
}

function brokenBridgeResponse(board, player, lastMove, BLACK, WHITE, EMPTY) {
  if (lastMove == null) return null;
  const n = board.n, flat = board.board, opp = player === BLACK ? WHITE : BLACK;
  if (flat[lastMove] !== opp) return null;
  const bt = getBridgeTable(n);
  for (let cell = 0; cell < n * n; cell++) {
    if (flat[cell] !== player) continue;
    for (const [partner, ca, cb] of bt[cell]) {
      if (flat[partner] !== player) continue;
      if (lastMove === ca && flat[cb] === EMPTY) return cb;
      if (lastMove === cb && flat[ca] === EMPTY) return ca;
    }
  }
  return null;
}

if (typeof module !== 'undefined') {
  module.exports = { twoDistance, connectionCost, moveScores, brokenBridgeResponse, getBridgeTable, INF, centrality, evaluate };
}
