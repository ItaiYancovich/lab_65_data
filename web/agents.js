// Agents -- a port of alphazero_hex/agents/*.py to run entirely client-side.
// Every agent exposes `async selectMove(board, lastMove, onProgress)`.
// Search agents yield to the event loop periodically (setTimeout(0)) so the
// page stays responsive and the "thinking" indicator can animate.
'use strict';

// See the matching comment in heuristics.js: this must use globalThis
// assignment, not var/let/const, or it breaks the browser's shared script
// scope even when the branch never runs.
if (typeof require !== 'undefined' && typeof BLACK === 'undefined') {
  Object.assign(globalThis, require('./hex.js'));
  Object.assign(globalThis, require('./heuristics.js'));
  Object.assign(globalThis, require('./net.js'));
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function choice(rng, arr) { return arr[Math.floor(rng() * arr.length)]; }
function yieldToUI() { return new Promise((r) => setTimeout(r, 0)); }

// ------------------------------------------------------------------ random
class RandomAgent {
  constructor(seed) { this.rng = mulberry32(seed); this.name = 'Random'; }
  async selectMove(board) { return choice(this.rng, board.legalMoves()); }
}

// ------------------------------------------------------------- rule-based
class RuleBasedAgent {
  constructor(seed, noise = 0.05) { this.rng = mulberry32(seed); this.noise = noise; this.name = 'Rule-based (two-distance + bridges)'; }
  async selectMove(board, lastMove) {
    const me = board.toMove, opp = other(me);
    const wins = board.immediateWins(me);
    if (wins.length) return wins[0];
    const threats = board.immediateWins(opp);
    if (threats.length === 1) return threats[0];
    const save = brokenBridgeResponse(board, me, lastMove, BLACK, WHITE, EMPTY);
    if (save != null) return save;
    const scores = moveScores(board, me, BLACK, WHITE, EMPTY, 1.0);
    let best = -1, bestScore = -Infinity;
    for (let i = 0; i < scores.length; i++) {
      if (scores[i] === -Infinity) continue;
      const s = scores[i] + (this.noise > 0 ? gaussian(this.rng) * this.noise : 0);
      if (s > bestScore) { bestScore = s; best = i; }
    }
    return best;
  }
}
function gaussian(rng) {
  const u1 = Math.max(rng(), 1e-12), u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ------------------------------------------------ smart brute force (alpha-beta)
class MinimaxAgent {
  constructor(seed, timeBudgetMs = 1000, beam = 8, maxDepth = 10) {
    this.rng = mulberry32(seed); this.timeBudgetMs = timeBudgetMs; this.beam = beam; this.maxDepth = maxDepth;
    this.name = `Brute-force alpha-beta (${(timeBudgetMs / 1000).toFixed(1)}s/move)`;
    this.tt = new Map();
    this.nodes = 0;
  }

  async selectMove(board, lastMove, onProgress) {
    const deadline = Date.now() + this.timeBudgetMs;
    const me = board.toMove, opp = other(me);
    const wins = board.immediateWins(me);
    if (wins.length) return wins[0];
    const threats = board.immediateWins(opp);
    if (threats.length === 1) return threats[0];
    if (threats.length > 1) {
      const scores = moveScores(board, me, BLACK, WHITE, EMPTY);
      return threats.reduce((a, b) => (scores[b] > scores[a] ? b : a));
    }

    const scores0 = moveScores(board, me, BLACK, WHITE, EMPTY);
    let best = 0;
    for (let i = 1; i < scores0.length; i++) if (scores0[i] > scores0[best]) best = i;
    this.nodes = 0;
    for (let depth = 1; depth <= this.maxDepth; depth++) {
      let result;
      try {
        result = await this._root(board, depth, deadline, onProgress);
      } catch (e) {
        if (e === TIMEOUT) break;
        throw e;
      }
      if (result.move != null) { best = result.move; if (onProgress) onProgress({ depth, nodes: this.nodes }); }
      if (Math.abs(result.value) >= 0.999) break;
      if (Date.now() >= deadline) break;
    }
    return best;
  }

  async _root(board, depth, deadline) {
    let alpha = -2, beta = 2, bestMove = null, bestValue = -2;
    for (const move of this._candidates(board, depth)) {
      const child = board.copy();
      child.play(move);
      const value = -(await this._negamax(child, depth - 1, -beta, -alpha, deadline));
      if (value > bestValue) { bestValue = value; bestMove = move; }
      alpha = Math.max(alpha, value);
    }
    return { value: bestValue, move: bestMove };
  }

  async _negamax(board, depth, alpha, beta, deadline) {
    this.nodes++;
    if (this.nodes % 512 === 0) {
      if (Date.now() >= deadline) throw TIMEOUT;
      await yieldToUI();
    }
    if (board.winner) return -1.0;
    if (depth <= 0) return evaluate(board, board.toMove, BLACK, WHITE);

    const key = board.board.join('') + ':' + board.toMove;
    const hit = this.tt.get(key);
    let ttMove = null;
    if (hit) {
      ttMove = hit.move;
      if (hit.depth >= depth) {
        if (hit.flag === 0) return hit.value;
        if (hit.flag === -1 && hit.value > alpha) alpha = hit.value;
        else if (hit.flag === 1 && hit.value < beta) beta = hit.value;
        if (alpha >= beta) return hit.value;
      }
    }

    const me = board.toMove, opp = other(me);
    const wins = board.immediateWins(me);
    if (wins.length) return 1.0;
    const threats = board.immediateWins(opp);
    let moves;
    if (threats.length === 1) moves = threats;
    else if (threats.length > 1) return -1.0;
    else moves = this._candidates(board, depth, ttMove);

    const origAlpha = alpha;
    let bestValue = -2, bestMove = null;
    for (const move of moves) {
      const child = board.copy();
      child.play(move);
      const value = -(await this._negamax(child, depth - 1, -beta, -alpha, deadline));
      if (value > bestValue) { bestValue = value; bestMove = move; }
      alpha = Math.max(alpha, value);
      if (alpha >= beta) break;
    }
    let flag = 0;
    if (bestValue <= origAlpha) flag = 1; else if (bestValue >= beta) flag = -1;
    if (this.tt.size < 200000) this.tt.set(key, { depth, flag, value: bestValue, move: bestMove });
    return bestValue;
  }

  _candidates(board, depth, ttMove) {
    const scores = moveScores(board, board.toMove, BLACK, WHITE, EMPTY);
    const width = depth >= 3 ? this.beam : Math.max(4, (this.beam / 2) | 0);
    const idx = [];
    for (let i = 0; i < scores.length; i++) if (Number.isFinite(scores[i])) idx.push(i);
    idx.sort((a, b) => scores[b] - scores[a]);
    let moves = idx.slice(0, width);
    if (ttMove != null) {
      moves = moves.filter((m) => m !== ttMove);
      if (board.board[ttMove] === EMPTY) moves.unshift(ttMove);
    }
    return moves;
  }
}
const TIMEOUT = Symbol('timeout');

// -------------------------------------------------------- classic rollout MCTS
function randomRollout(board, rng) {
  const n = board.n, arr = board.board.slice();
  const empties = [];
  for (let i = 0; i < arr.length; i++) if (arr[i] === EMPTY) empties.push(i);
  if (empties.length === 0) return floodWinner(arr, n) === board.toMove ? 1 : -1;
  for (let i = empties.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [empties[i], empties[j]] = [empties[j], empties[i]]; }
  const me = board.toMove, opp = other(me);
  for (let k = 0; k < empties.length; k++) arr[empties[k]] = (k % 2 === 0) ? me : opp;
  return floodWinner(arr, n) === me ? 1 : -1;
}
function floodWinner(arr, n) {
  const nei = getNei(n);
  const seen = new Uint8Array(n * n);
  const stack = [];
  for (let c = 0; c < n; c++) if (arr[c] === BLACK) { stack.push(c); seen[c] = 1; }
  while (stack.length) {
    const cur = stack.pop();
    if (cur >= (n - 1) * n) return BLACK;
    for (const nb of nei[cur]) if (!seen[nb] && arr[nb] === BLACK) { seen[nb] = 1; stack.push(nb); }
  }
  return WHITE;
}

class RolloutMCTSAgent {
  constructor(seed, timeBudgetMs = 1000, cUct = 1.0) {
    this.rng = mulberry32(seed); this.timeBudgetMs = timeBudgetMs; this.cUct = cUct;
    this.name = `Classic MCTS rollouts (${(timeBudgetMs / 1000).toFixed(1)}s/move)`;
  }
  async selectMove(board, lastMove, onProgress) {
    const me = board.toMove, opp = other(me);
    const wins = board.immediateWins(me);
    if (wins.length) return wins[0];
    const threats = board.immediateWins(opp);
    if (threats.length === 1) return threats[0];

    const moves = board.legalMoves();
    const N = new Float64Array(moves.length), W = new Float64Array(moves.length);
    const children = new Array(moves.length).fill(null);
    const deadline = Date.now() + this.timeBudgetMs;
    let sims = 0;
    while (Date.now() < deadline) {
      this._simulate(board, moves, N, W, children);
      sims++;
      if (sims % 200 === 0) { if (onProgress) onProgress({ sims }); await yieldToUI(); }
    }
    if (onProgress) onProgress({ sims, done: true });
    let best = 0;
    for (let i = 1; i < N.length; i++) if (N[i] > N[best]) best = i;
    return moves[best];
  }

  _simulate(rootBoard, moves, N, W, children) {
    const path = [];
    let board = rootBoard.copy();
    let curMoves = moves, curN = N, curW = W, curChildren = children;
    while (true) {
      const a = this._uctSelect(curN, curW);
      const move = curMoves[a];
      path.push([curN, curW, a]);
      board.play(move);
      if (board.isTerminal()) { this._backup(path, -1.0); return; }
      if (curChildren[a] === null) {
        const nm = board.legalMoves();
        curChildren[a] = { moves: nm, N: new Float64Array(nm.length), W: new Float64Array(nm.length), children: new Array(nm.length).fill(null) };
        const value = randomRollout(board, this.rng);
        this._backup(path, value);
        return;
      }
      const nxt = curChildren[a];
      curMoves = nxt.moves; curN = nxt.N; curW = nxt.W; curChildren = nxt.children;
    }
  }
  _uctSelect(N, W) {
    let total = 0;
    for (let i = 0; i < N.length; i++) total += N[i];
    const unvisited = [];
    for (let i = 0; i < N.length; i++) if (N[i] === 0) unvisited.push(i);
    if (unvisited.length) return unvisited[Math.floor(this.rng() * unvisited.length)];
    let best = 0, bestScore = -Infinity;
    const logTotal = Math.log(total);
    for (let i = 0; i < N.length; i++) {
      const q = W[i] / N[i], u = this.cUct * Math.sqrt(2 * logTotal / N[i]);
      if (q + u > bestScore) { bestScore = q + u; best = i; }
    }
    return best;
  }
  _backup(path, value) {
    let v = value;
    for (let i = path.length - 1; i >= 0; i--) { v = -v; const [N, W, a] = path[i]; N[a] += 1; W[a] += v; }
  }
}

// --------------------------------------------------------------- AlphaZero
class PUCTSearch {
  constructor(rootState, net, cPuct, rng) {
    this.root = rootState; this.net = net; this.cPuct = cPuct; this.rng = rng;
    this.nodes = [{ moves: null, P: null, N: null, W: null, child: null, term: null }];
  }
  isExpanded(i) { return this.nodes[i].moves !== null; }

  // Descend to a leaf needing evaluation. Returns {state, node} or null if
  // that leaf is terminal (already backed up) and the caller should call again.
  descend() {
    let state = this.root.copy(), node = 0, path = [];
    while (true) {
      const nd = this.nodes[node];
      if (nd.term !== null) { this._backup(path, nd.term); return null; }
      if (!this.isExpanded(node)) return { state, node, path };
      const a = this._select(node);
      const move = nd.moves[a];
      path.push([node, a]);
      state.play(move);
      let nxt = nd.child[a];
      if (nxt < 0) {
        nxt = this.nodes.length;
        this.nodes.push({ moves: null, P: null, N: null, W: null, child: null, term: null });
        nd.child[a] = nxt;
        if (state.isTerminal()) this.nodes[nxt].term = -1.0;
      }
      node = nxt;
    }
  }

  _select(i) {
    const nd = this.nodes[i];
    let total = 0;
    for (let k = 0; k < nd.N.length; k++) total += nd.N[k];
    const sqrtTotal = total > 0 ? Math.sqrt(total) : 1.0;
    let parentQ = 0;
    if (total > 0) { let sw = 0; for (let k = 0; k < nd.W.length; k++) sw += nd.W[k]; parentQ = sw / total; }
    let best = 0, bestScore = -Infinity;
    for (let k = 0; k < nd.N.length; k++) {
      const q = nd.N[k] > 0 ? nd.W[k] / nd.N[k] : parentQ - 0.25;
      const u = this.cPuct * nd.P[k] * sqrtTotal / (1 + nd.N[k]);
      const s = q + u;
      if (s > bestScore) { bestScore = s; best = k; }
    }
    return best;
  }

  expand(node, path, state, logits, value) {
    const legal = state.legalMoves();
    if (legal.length === 0) { this.nodes[node].term = 0; this._backup(path, 0); return; }
    const canonIdx = legal.map((m) => state.toCanonicalMove(m));
    const probs = maskedSoftmax(logits, canonIdx);
    const nd = this.nodes[node];
    nd.moves = legal; nd.P = Float32Array.from(probs); nd.N = new Float64Array(legal.length);
    nd.W = new Float64Array(legal.length); nd.child = new Int32Array(legal.length).fill(-1);
    if (node === 0) this._applyRootNoise();
    this._backup(path, value);
  }

  _applyRootNoise() {
    const nd = this.nodes[0], k = nd.P.length;
    if (k <= 1) return;
    const alpha = Math.max(0.03, 10.0 / (this.root.n * this.root.n));
    const gamma = new Float64Array(k);
    let sum = 0;
    for (let i = 0; i < k; i++) { gamma[i] = sampleGamma(alpha, this.rng); sum += gamma[i]; }
    const eps = 0.25;
    for (let i = 0; i < k; i++) nd.P[i] = (1 - eps) * nd.P[i] + eps * (gamma[i] / sum);
  }

  _backup(path, value) {
    let v = value;
    for (let i = path.length - 1; i >= 0; i--) { v = -v; const [node, a] = path[i]; const nd = this.nodes[node]; nd.N[a] += 1; nd.W[a] += v; }
  }

  rootVisits() { return { moves: this.nodes[0].moves, N: this.nodes[0].N }; }
}
// Marsaglia-Tsang gamma sampler (alpha < 1 boosted via alpha+1 then power trick).
function sampleGamma(alpha, rng) {
  if (alpha < 1) { const u = Math.max(rng(), 1e-12); return sampleGamma(alpha + 1, rng) * Math.pow(u, 1 / alpha); }
  const d = alpha - 1 / 3, c = 1 / Math.sqrt(9 * d);
  while (true) {
    let x, v;
    do { x = gaussian(rng); v = 1 + c * x; } while (v <= 0);
    v = v * v * v;
    const u = Math.max(rng(), 1e-12);
    if (Math.log(u) < 0.5 * x * x + d - d * v + d * Math.log(v)) return d * v;
  }
}

class AlphaZeroAgent {
  constructor(seed, net, simulations = 150, cPuct = 1.6, addNoise = false) {
    this.rng = mulberry32(seed); this.net = net; this.simulations = simulations; this.cPuct = cPuct;
    this.addNoise = addNoise;
    this.name = `AlphaZero (${simulations} sims)`;
    this.lastValue = 0;
  }
  async selectMove(board, lastMove, onProgress) {
    const search = new PUCTSearch(board.copy(), this.net, this.cPuct, this.rng);
    let sims = 0;
    while (sims < this.simulations) {
      const leaf = search.descend();
      if (leaf === null) { sims++; continue; }
      const planes = leaf.state.canonicalPlanes();
      const { logits, value } = this.net.forward(planes, leaf.state.n);
      search.expand(leaf.node, leaf.path, leaf.state, logits, value);
      sims++;
      if (sims % 5 === 0) { if (onProgress) onProgress({ sims, total: this.simulations }); await yieldToUI(); }
    }
    if (onProgress) onProgress({ sims, total: this.simulations, done: true });
    const { moves, N } = search.rootVisits();
    let best = 0;
    for (let i = 1; i < N.length; i++) if (N[i] > N[best]) best = i;
    let total = 0, wsum = 0;
    for (let i = 0; i < N.length; i++) total += N[i];
    for (let i = 0; i < N.length; i++) wsum += search.nodes[0].W[i];
    this.lastValue = total > 0 ? wsum / total : 0;
    return moves[best];
  }
}

class PolicyOnlyAgent {
  constructor(seed, net) { this.rng = mulberry32(seed); this.net = net; this.name = 'AlphaZero policy only (no search)'; }
  async selectMove(board) {
    const planes = board.canonicalPlanes();
    const { logits } = this.net.forward(planes, board.n);
    const legal = board.legalMoves();
    const canonIdx = legal.map((m) => board.toCanonicalMove(m));
    const probs = maskedSoftmax(logits, canonIdx);
    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
    return legal[best];
  }
}

if (typeof module !== 'undefined') {
  module.exports = { RandomAgent, RuleBasedAgent, MinimaxAgent, RolloutMCTSAgent, AlphaZeroAgent, PolicyOnlyAgent, mulberry32 };
}
