// Node-side sanity checks for hex.js / heuristics.js, run standalone (no
// Python here) -- structural checks mirroring tests/test_hex.py.
'use strict';
const { HexBoard, BLACK, WHITE, getNei, rotate180Move } = require('./hex.js');
const { connectionCost, moveScores } = require('./heuristics.js');

function assert(cond, msg) { if (!cond) throw new Error('FAIL: ' + msg); }

function testNeighbours() {
  const n = 5, nei = getNei(n);
  assert(new Set(nei[0]).size === 2 && nei[0].includes(1) && nei[0].includes(n), 'corner neighbours');
  const centre = 2 * n + 2;
  assert(nei[centre].length === 6, 'centre has 6 neighbours');
  console.log('ok neighbours');
}

function testBlackVerticalWin() {
  const n = 5, b = new HexBoard(n);
  const black = [0, 1, 2, 3, 4].map(r => r * n + 2);
  const white = [0, 4, n, n + 4, 2 * n];
  for (let i = 0; i < n; i++) {
    b.play(black[i]);
    if (b.isTerminal()) break;
    b.play(white[i]);
  }
  assert(b.winner === BLACK, 'black should win vertical chain');
  console.log('ok black vertical win');
}

function testNoDraws() {
  let rng = mulberry32(42);
  for (let trial = 0; trial < 100; trial++) {
    const n = 7, b = new HexBoard(n);
    const order = shuffle([...Array(n * n).keys()], rng);
    for (const m of order) {
      b.play(m);
      if (b.isTerminal()) break;
    }
    assert(b.winner === BLACK || b.winner === WHITE, 'no draws');
  }
  console.log('ok no draws (100 random games)');
}

function testImmediateWinsMatchesBruteForce() {
  let rng = mulberry32(7);
  let checked = 0;
  for (let g = 0; g < 20; g++) {
    const n = 7, b = new HexBoard(n);
    const order = shuffle([...Array(n * n).keys()], rng);
    for (const m of order) {
      if (b.isTerminal()) break;
      for (const player of [BLACK, WHITE]) {
        const fast = new Set(b.immediateWins(player));
        const brute = new Set();
        for (const cand of b.legalMoves()) {
          const probe = b.copy();
          probe.toMove = player;
          probe.play(cand);
          if (probe.winner === player) brute.add(cand);
        }
        assert(fast.size === brute.size && [...fast].every(x => brute.has(x)), `mismatch at move ${m}`);
        checked++;
      }
      b.play(m);
    }
  }
  console.log(`ok immediate wins match brute force (${checked} positions)`);
}

function testTwoDistanceEmptyBoard() {
  const n = 7, b = new HexBoard(n);
  const { potential } = connectionCost(b.board, n, BLACK, BLACK, WHITE, 0);
  assert(potential === n, `expected potential ${n}, got ${potential}`);
  const scores = moveScores(b, BLACK, BLACK, WHITE, 0);
  let best = 0;
  for (let i = 1; i < scores.length; i++) if (scores[i] > scores[best]) best = i;
  const r = (best / n) | 0, c = best % n;
  assert(r >= 1 && r <= n - 2 && c >= 1 && c <= n - 2, `expected central best cell, got (${r},${c})`);
  console.log(`ok two-distance empty board (potential=${potential}, best=(${r},${c}))`);
}

function testRotation() {
  const n = 11;
  for (let m = 0; m < n * n; m++) assert(rotate180Move(rotate180Move(m, n), n) === m, 'rotation involution');
  console.log('ok rotation involution');
}

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function shuffle(arr, rng) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

testNeighbours();
testBlackVerticalWin();
testNoDraws();
testImmediateWinsMatchesBruteForce();
testTwoDistanceEmptyBoard();
testRotation();
console.log('\nall JS engine tests passed');
