'use strict';
/* Hex Lab -- browser-side controller: board rendering, seat/agent wiring,
   game loop, undo, and the reference standings card. Depends on hex.js,
   heuristics.js, net.js and agents.js already being loaded (script tags),
   and on window.NET_JSON holding the exported, BN-folded network weights. */

const N = 11;
const AGENT_LABELS = {
  human: 'Human (you)',
  random: 'Random',
  rule: 'Rule-based',
  minimax: 'Brute-force alpha-beta',
  rollout: 'Classic MCTS rollouts',
  policy: 'AlphaZero -- policy only',
  az: 'AlphaZero -- with search',
};
const TOURNAMENT_STANDINGS = [
  { agent: 'AlphaZero (400 sims)', elo: 2653, err: 177, wr: 99.3 },
  { agent: 'AlphaZero policy only', elo: 2137, err: 137, wr: 80.7 },
  { agent: 'Classic MCTS rollouts', elo: 1527, err: 96, wr: 58.7 },
  { agent: 'Brute-force alpha-beta', elo: 1134, err: 96, wr: 40.0 },
  { agent: 'Rule-based', elo: 726, err: 110, wr: 21.3 },
  { agent: 'Random', elo: 0, err: 227, wr: 0.0 },
];

let netInstance = null;
function getNet() {
  if (!netInstance) netInstance = new HexNetJS(window.NET_JSON);
  return netInstance;
}

let seedCounter = 1;
function nextSeed() { return (seedCounter = (seedCounter * 2654435761 + 1) >>> 0); }

function makeAgent(kind, strength) {
  switch (kind) {
    case 'human': return null;
    case 'random': return new RandomAgent(nextSeed());
    case 'rule': return new RuleBasedAgent(nextSeed(), 0.05);
    case 'minimax': return new MinimaxAgent(nextSeed(), strength, 8);
    case 'rollout': return new RolloutMCTSAgent(nextSeed(), strength);
    case 'policy': return new PolicyOnlyAgent(nextSeed(), getNet());
    case 'az': return new AlphaZeroAgent(nextSeed(), getNet(), strength);
    default: throw new Error('unknown agent ' + kind);
  }
}

// --------------------------------------------------------------- game state
const state = {
  board: new HexBoard(N),
  lastMove: null,
  history: [], // list of played moves, for undo/replay
  seats: {
    [BLACK]: { kind: 'human', strengthParam: null, agent: null },
    [WHITE]: { kind: 'az', strengthParam: 140, agent: null },
  },
  gen: 0,       // bumped on every reset so stray async agent moves are dropped
  thinking: false,
  gameOver: false,
};
rebuildAgents();

function rebuildAgents() {
  for (const color of [BLACK, WHITE]) {
    const seat = state.seats[color];
    seat.agent = makeAgent(seat.kind, seat.strengthParam);
  }
}

// ------------------------------------------------------------- board layout
const HEX_SIZE = 26;
function axialToPixel(r, c) {
  return [HEX_SIZE * Math.sqrt(3) * (c + r / 2), HEX_SIZE * 1.5 * r];
}
function hexPoints(cx, cy, size) {
  const pts = [];
  for (let i = 0; i < 6; i++) {
    const a = (Math.PI / 180) * (60 * i - 30);
    pts.push(`${(cx + size * Math.cos(a)).toFixed(2)},${(cy + size * Math.sin(a)).toFixed(2)}`);
  }
  return pts.join(' ');
}
function edgeMidDir(i) {
  const a1 = (Math.PI / 180) * (60 * i - 30), a2 = (Math.PI / 180) * (60 * ((i + 1) % 6) - 30);
  const mx = (Math.cos(a1) + Math.cos(a2)) / 2, my = (Math.sin(a1) + Math.sin(a2)) / 2;
  const len = Math.hypot(mx, my);
  return [mx / len, my / len];
}
// OFFSETS reuses hex.js's global (see the require-shim comments there for why
// app.js must not redeclare it with const/let/var in this shared scope).

function buildBoardSvg() {
  const svg = document.getElementById('board-svg');
  svg.innerHTML = '';
  const cellPolys = [];
  const centers = [];
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
    const [x, y] = axialToPixel(r, c);
    centers[r * N + c] = [x, y];
    minX = Math.min(minX, x - HEX_SIZE); maxX = Math.max(maxX, x + HEX_SIZE);
    minY = Math.min(minY, y - HEX_SIZE); maxY = Math.max(maxY, y + HEX_SIZE);
  }
  const padL = 34, padT = 26, padR = 14, padB = 14;
  const vbX = minX - padL, vbY = minY - padT, vbW = (maxX - minX) + padL + padR, vbH = (maxY - minY) + padT + padB;
  svg.setAttribute('viewBox', `${vbX} ${vbY} ${vbW} ${vbH}`);
  svg.setAttribute('role', 'grid');
  svg.setAttribute('aria-label', `Hex board, ${N} by ${N}`);

  const edgesGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  const cellsGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  const labelsGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  svg.append(edgesGroup, cellsGroup, labelsGroup);

  // Column labels (a..k) above row 0, row labels (1..N) left of column 0.
  for (let c = 0; c < N; c++) {
    const [x, y] = centers[c];
    const t = svgText(x, y - HEX_SIZE - 8, String.fromCharCode(97 + c));
    t.setAttribute('text-anchor', 'middle');
    labelsGroup.appendChild(t);
  }
  for (let r = 0; r < N; r++) {
    const [x, y] = centers[r * N];
    const t = svgText(x - HEX_SIZE * 1.55, y + 3, String(r + 1));
    t.setAttribute('text-anchor', 'middle');
    labelsGroup.appendChild(t);
  }

  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      const cell = r * N + c, [cx, cy] = centers[cell];

      // outward boundary edges -> coloured goal markers (see app design notes)
      for (const [dr, dc] of OFFSETS) {
        const rr = r + dr, cc = c + dc;
        const rowOut = rr < 0 || rr >= N, colOut = cc < 0 || cc >= N;
        if (rowOut === colOut) continue; // interior neighbour, or off past a corner
        const [nx, ny] = axialToPixel(rr, cc);
        let dx = nx - cx, dy = ny - cy; const len = Math.hypot(dx, dy); dx /= len; dy /= len;
        let bestI = 0, bestDot = -Infinity;
        for (let i = 0; i < 6; i++) {
          const [ex, ey] = edgeMidDir(i);
          const dot = ex * dx + ey * dy;
          if (dot > bestDot) { bestDot = dot; bestI = i; }
        }
        const a1 = (Math.PI / 180) * (60 * bestI - 30), a2 = (Math.PI / 180) * (60 * ((bestI + 1) % 6) - 30);
        const x1 = cx + (HEX_SIZE + 1.5) * Math.cos(a1), y1 = cy + (HEX_SIZE + 1.5) * Math.sin(a1);
        const x2 = cx + (HEX_SIZE + 1.5) * Math.cos(a2), y2 = cy + (HEX_SIZE + 1.5) * Math.sin(a2);
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', x1.toFixed(2)); line.setAttribute('y1', y1.toFixed(2));
        line.setAttribute('x2', x2.toFixed(2)); line.setAttribute('y2', y2.toFixed(2));
        line.setAttribute('stroke-width', '4');
        line.setAttribute('class', 'edge-mark ' + (rowOut ? 'black-edge' : 'white-edge'));
        edgesGroup.appendChild(line);
      }

      const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
      poly.setAttribute('points', hexPoints(cx, cy, HEX_SIZE - 1));
      poly.setAttribute('class', 'hex-fill empty');
      poly.dataset.cell = cell;
      const wrap = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      wrap.setAttribute('class', 'hexcell');
      wrap.setAttribute('role', 'button');
      wrap.setAttribute('tabindex', '0');
      wrap.setAttribute('aria-label', moveToStr(cell, N));
      wrap.dataset.cell = cell;
      wrap.appendChild(poly);
      wrap.addEventListener('click', () => onCellClick(cell));
      wrap.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onCellClick(cell); }
      });
      cellsGroup.appendChild(wrap);
      cellPolys[cell] = poly;
    }
  }
  return cellPolys;
}
function svgText(x, y, str) {
  const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  t.setAttribute('x', x.toFixed(2)); t.setAttribute('y', y.toFixed(2));
  t.setAttribute('class', 'coord-label');
  t.textContent = str;
  return t;
}

let cellPolys = buildBoardSvg();

function renderBoard() {
  const winningSet = state.gameOver ? winningCells(state.board) : null;
  for (let i = 0; i < N * N; i++) {
    const v = state.board.board[i];
    const poly = cellPolys[i];
    let cls = 'hex-fill ';
    cls += v === BLACK ? 'black' : v === WHITE ? 'white' : 'empty';
    if (winningSet && winningSet.has(i)) cls += ' winning';
    if (i === state.lastMove) cls += ' last-move';
    poly.setAttribute('class', cls);
    const wrap = poly.parentElement;
    const playable = !state.gameOver && !state.thinking && v === EMPTY &&
      state.seats[state.board.toMove].kind === 'human';
    wrap.classList.toggle('playable', playable);
    const coord = moveToStr(i, N);
    wrap.setAttribute('aria-label', v === EMPTY ? coord : `${coord}, ${v === BLACK ? 'black' : 'white'}`);
  }
}

function winningCells(board) {
  // Highlight the connected chain that actually links the winner's two edges.
  const n = board.n, winner = board.winner, set = new Set();
  if (!winner) return set;
  const startEdge = winner === BLACK
    ? [...Array(n).keys()]
    : [...Array(n).keys()].map((c) => c * n);
  const targetRow = winner === BLACK ? n - 1 : null;
  const seen = new Uint8Array(n * n);
  const stack = [];
  for (const s of startEdge) if (board.board[s] === winner) { stack.push(s); seen[s] = 1; }
  let reached = false;
  const nei = getNei(n);
  while (stack.length) {
    const cur = stack.pop();
    set.add(cur);
    const r = (cur / n) | 0, c = cur % n;
    if ((winner === BLACK && r === n - 1) || (winner === WHITE && c === n - 1)) reached = true;
    for (const nb of nei[cur]) if (!seen[nb] && board.board[nb] === winner) { seen[nb] = 1; stack.push(nb); }
  }
  return reached ? set : new Set();
}

// ------------------------------------------------------------------- loop
async function onCellClick(cell) {
  if (state.gameOver || state.thinking) return;
  const seat = state.seats[state.board.toMove];
  if (seat.kind !== 'human') return;
  if (!state.board.isLegal(cell)) return;
  playMove(cell);
  await stepLoop();
}

function playMove(move) {
  state.board.play(move);
  state.lastMove = move;
  state.history.push(move);
  if (state.board.isTerminal()) state.gameOver = true;
  renderBoard();
  renderMoveLog();
  renderStatus();
}

async function stepLoop() {
  const myGen = state.gen;
  while (!state.gameOver) {
    const seat = state.seats[state.board.toMove];
    if (seat.kind === 'human') { renderStatus(); return; }
    state.thinking = true;
    renderBoard();
    renderStatus();
    const lastMove = state.lastMove;
    const board = state.board;
    let move;
    try {
      move = await seat.agent.selectMove(board, lastMove, (progress) => {
        if (state.gen !== myGen) return;
        renderThinkingProgress(seat, progress);
      });
    } catch (e) {
      console.error(e);
      state.thinking = false;
      renderStatus('An agent hit an error -- see console. Try New Game.');
      return;
    }
    if (state.gen !== myGen) return; // a reset happened while we were awaiting
    state.thinking = false;
    if (seat.kind === 'az' && typeof seat.agent.lastValue === 'number') {
      updateEvalBar(state.board.toMove, seat.agent.lastValue);
    }
    playMove(move);
  }
  renderStatus();
}

// ------------------------------------------------------------------- UI
const els = {};
function cacheEls() {
  ['status-dot', 'status-text', 'status-sub', 'eval-row', 'eval-fill', 'eval-caption',
   'banner', 'banner-text', 'movelog', 'new-game', 'swap-seats', 'undo',
   'seat-black-kind', 'seat-white-kind', 'seat-black-strength', 'seat-white-strength',
   'seat-black-strength-row', 'seat-white-strength-row', 'seat-black-strength-val',
   'seat-white-strength-val'].forEach((id) => { els[id] = document.getElementById(id); });
}

function updateControlsDisabled() {
  const busy = state.thinking;
  for (const key of ['black', 'white']) {
    els[`seat-${key}-kind`].disabled = busy;
    els[`seat-${key}-strength`].disabled = busy;
  }
  els['undo'].disabled = busy || state.history.length === 0;
  els['swap-seats'].disabled = busy;
}

function renderStatus(errorMsg) {
  updateControlsDisabled();
  const dot = els['status-dot'], text = els['status-text'], sub = els['status-sub'];
  dot.className = 'status-dot';
  if (errorMsg) {
    text.innerHTML = errorMsg;
    sub.textContent = '';
    return;
  }
  if (state.gameOver) {
    const winner = state.board.winner;
    dot.classList.add(winner === BLACK ? 'black' : 'white');
    const seat = state.seats[winner];
    const label = seat.kind === 'human' ? 'You' : AGENT_LABELS[seat.kind];
    text.innerHTML = `<b>${winner === BLACK ? 'Black' : 'White'}</b> wins -- ${label} connected ` +
      (winner === BLACK ? 'top to bottom' : 'left to right') + ` in ${state.board.moveCount} moves.`;
    sub.textContent = '';
    showBanner(`${winner === BLACK ? 'Black' : 'White'} wins`);
    return;
  }
  hideBanner();
  const toMove = state.board.toMove;
  const seat = state.seats[toMove];
  dot.classList.add(toMove === BLACK ? 'black' : 'white');
  if (state.thinking) dot.classList.add('thinking');
  const who = seat.kind === 'human' ? 'Your move' : `${AGENT_LABELS[seat.kind]} is thinking`;
  text.innerHTML = `<b>${toMove === BLACK ? 'Black' : 'White'}</b> -- ${who}${state.thinking ? '&hellip;' : ''}`;
  if (!state.thinking) sub.textContent = '';
}

function renderThinkingProgress(seat, progress) {
  const sub = els['status-sub'];
  if (progress.total != null) sub.textContent = `simulations ${progress.sims} / ${progress.total}`;
  else if (progress.sims != null) sub.textContent = `rollouts ${progress.sims.toLocaleString()}`;
  else if (progress.depth != null) sub.textContent = `depth ${progress.depth}, ${progress.nodes.toLocaleString()} nodes`;
}

function updateEvalBar(searchedColor, value) {
  // value is P(searchedColor wins) roughly, in [-1,1] from that colour's POV.
  const blackWinProb = searchedColor === BLACK ? (value + 1) / 2 : 1 - (value + 1) / 2;
  els['eval-row'].style.display = 'flex';
  els['eval-fill'].style.width = `${(blackWinProb * 100).toFixed(1)}%`;
  els['eval-caption'].textContent =
    `AlphaZero's estimate: Black ${(blackWinProb * 100).toFixed(0)}% / White ${((1 - blackWinProb) * 100).toFixed(0)}%`;
}

function showBanner(msg) {
  els['banner'].classList.add('show');
  els['banner-text'].textContent = msg;
}
function hideBanner() { els['banner'].classList.remove('show'); }

function renderMoveLog() {
  const el = els['movelog'];
  if (!state.history.length) { el.innerHTML = '<div class="empty">No moves yet.</div>'; return; }
  el.innerHTML = state.history.map((m, i) => {
    const color = i % 2 === 0 ? 'b' : 'w';
    return `<div class="mv ${color}"><span class="n">${i + 1}.</span><span class="c">${moveToStr(m, N)}</span></div>`;
  }).join('');
  el.scrollTop = el.scrollHeight;
}

function renderStandings() {
  const rows = TOURNAMENT_STANDINGS.map((r, i) => `
    <tr>
      <td class="num">${i + 1}</td>
      <td class="name">${r.agent}</td>
      <td class="num">${r.elo}</td>
      <td class="num">${r.wr.toFixed(1)}%</td>
    </tr>`).join('');
  document.getElementById('standings-body').innerHTML = rows;
}

// --------------------------------------------------------------- controls
function strengthConfigFor(kind) {
  // {min, max, step, default, unit, toParam}
  if (kind === 'az') return { min: 30, max: 400, step: 10, def: 140, unit: 'sims', toParam: (v) => v };
  if (kind === 'minimax') return { min: 300, max: 3000, step: 100, def: 1000, unit: 'ms', toParam: (v) => v };
  if (kind === 'rollout') return { min: 300, max: 3000, step: 100, def: 1000, unit: 'ms', toParam: (v) => v };
  return null;
}

function setupSeatControls(color) {
  const key = color === BLACK ? 'black' : 'white';
  const kindSel = els[`seat-${key}-kind`];
  const strengthInput = els[`seat-${key}-strength`];
  const strengthRow = els[`seat-${key}-strength-row`];
  const strengthVal = els[`seat-${key}-strength-val`];

  kindSel.value = state.seats[color].kind;
  const onKindChange = () => {
    const kind = kindSel.value;
    const cfg = strengthConfigFor(kind);
    state.seats[color].kind = kind;
    if (cfg) {
      strengthRow.style.display = 'flex';
      strengthInput.min = cfg.min; strengthInput.max = cfg.max; strengthInput.step = cfg.step;
      strengthInput.value = cfg.def;
      strengthVal.textContent = cfg.unit === 'ms' ? `${(cfg.def / 1000).toFixed(1)}s` : `${cfg.def} ${cfg.unit}`;
      state.seats[color].strengthParam = cfg.unit === 'ms' ? cfg.def : cfg.def;
    } else {
      strengthRow.style.display = 'none';
      state.seats[color].strengthParam = null;
    }
    rebuildAgents();
  };
  kindSel.addEventListener('change', onKindChange);
  strengthInput.addEventListener('input', () => {
    const cfg = strengthConfigFor(kindSel.value);
    const v = Number(strengthInput.value);
    strengthVal.textContent = cfg.unit === 'ms' ? `${(v / 1000).toFixed(1)}s` : `${v} ${cfg.unit}`;
    state.seats[color].strengthParam = v;
    rebuildAgents();
  });
  onKindChange();
}

function resetGame() {
  state.gen++;
  state.board = new HexBoard(N);
  state.lastMove = null;
  state.history = [];
  state.gameOver = false;
  state.thinking = false;
  els['eval-row'].style.display = 'none'; els['eval-caption'].textContent = '';
  hideBanner();
  rebuildAgents();
  renderBoard();
  renderMoveLog();
  renderStatus();
  stepLoop();
}

function swapSeats() {
  const b = state.seats[BLACK], w = state.seats[WHITE];
  state.seats[BLACK] = w; state.seats[WHITE] = b;
  document.getElementById('seat-black-kind').value = state.seats[BLACK].kind;
  document.getElementById('seat-white-kind').value = state.seats[WHITE].kind;
  document.getElementById('seat-black-kind').dispatchEvent(new Event('change'));
  document.getElementById('seat-white-kind').dispatchEvent(new Event('change'));
  // Restore the strength params the swap just clobbered via the default-reset.
  state.seats[BLACK].strengthParam = b.strengthParam;
  state.seats[WHITE].strengthParam = w.strengthParam;
  rebuildAgents();
  resetGame();
}

function undo() {
  if (state.thinking || !state.history.length) return;
  state.gen++;
  // Undo returns control to a human: pop the trailing agent replies plus the
  // human move that prompted them, not just one ply (which would hand the
  // move straight back to the same agent and look like nothing happened).
  // A spectated agent-vs-agent game has no such point, so step back one ply.
  const anyHuman = state.seats[BLACK].kind === 'human' || state.seats[WHITE].kind === 'human';
  if (!anyHuman) {
    state.history.pop();
  } else {
    while (state.history.length) {
      const idx = state.history.length - 1;
      const mover = idx % 2 === 0 ? BLACK : WHITE;
      const wasHuman = state.seats[mover].kind === 'human';
      state.history.pop();
      if (wasHuman) break;
    }
  }
  const moves = state.history.slice();
  state.board = new HexBoard(N);
  for (const m of moves) state.board.play(m);
  state.lastMove = moves.length ? moves[moves.length - 1] : null;
  state.gameOver = false;
  els['eval-row'].style.display = 'none'; els['eval-caption'].textContent = '';
  hideBanner();
  renderBoard();
  renderMoveLog();
  renderStatus();
  stepLoop();
}

function initThemeToggle() {
  const root = document.documentElement;
  const buttons = document.querySelectorAll('.theme-toggle button');
  const apply = (mode) => {
    if (mode === 'system') root.removeAttribute('data-theme');
    else root.setAttribute('data-theme', mode);
    buttons.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.mode === mode)));
  };
  buttons.forEach((b) => b.addEventListener('click', () => apply(b.dataset.mode)));
  apply('system');
}

function main() {
  cacheEls();
  renderStandings();
  setupSeatControls(BLACK);
  setupSeatControls(WHITE);
  document.getElementById('new-game').addEventListener('click', resetGame);
  document.getElementById('swap-seats').addEventListener('click', swapSeats);
  document.getElementById('undo').addEventListener('click', undo);
  document.getElementById('banner-newgame').addEventListener('click', resetGame);
  initThemeToggle();
  renderBoard();
  renderMoveLog();
  renderStatus();
  stepLoop();
}

document.addEventListener('DOMContentLoaded', main);
