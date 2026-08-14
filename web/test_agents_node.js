'use strict';
if (typeof atob === 'undefined') global.atob = (b64) => Buffer.from(b64, 'base64').toString('binary');
const fs = require('fs');
const { HexBoard, BLACK, WHITE } = require('./hex.js');
const { HexNetJS } = require('./net.js');
const { RandomAgent, RuleBasedAgent, MinimaxAgent, RolloutMCTSAgent, AlphaZeroAgent, PolicyOnlyAgent } = require('./agents.js');

const netJson = JSON.parse(fs.readFileSync(__dirname + '/../runs/az_hex/net.json', 'utf8'));
const net = new HexNetJS(netJson);

async function playGame(black, white, n, opening) {
  const board = new HexBoard(n);
  let last = null;
  if (opening != null) { board.play(opening); last = opening; }
  let plies = 0;
  while (!board.isTerminal() && plies < n * n) {
    const agent = board.toMove === BLACK ? black : white;
    const move = await agent.selectMove(board, last);
    if (!board.isLegal(move)) throw new Error(`${agent.name} played illegal move ${move} (plies=${plies})`);
    board.play(move); last = move; plies++;
  }
  if (!board.isTerminal()) throw new Error('game did not terminate');
  return board.winner === BLACK ? black.name : white.name;
}

async function main() {
  const n = 7;
  console.log(`playing every agent pair on ${n}x${n}...`);

  const pairs = [
    [new RandomAgent(1), new RuleBasedAgent(2, 0.05)],
    [new RuleBasedAgent(3, 0.05), new MinimaxAgent(4, 300, 5)],
    [new MinimaxAgent(5, 300, 5), new RolloutMCTSAgent(6, 300)],
    [new RolloutMCTSAgent(7, 300), new PolicyOnlyAgent(8, net)],
    [new PolicyOnlyAgent(9, net), new AlphaZeroAgent(10, net, 24)],
    [new AlphaZeroAgent(11, net, 24), new RandomAgent(12)],
  ];
  for (const [a, b] of pairs) {
    const t0 = Date.now();
    const winner = await playGame(a, b, n);
    console.log(`  ${a.name} vs ${b.name}: ${winner} won (${((Date.now() - t0) / 1000).toFixed(1)}s)`);
  }

  // Rule-based should crush random over several games (mirrors test_agents.py).
  let wins = 0;
  const games = 8;
  for (let i = 0; i < games; i++) {
    const rb = new RuleBasedAgent(100 + i, 0.05), rnd = new RandomAgent(200 + i);
    const winner = i % 2 === 0 ? await playGame(rb, rnd, 7) : await playGame(rnd, rb, 7);
    if (winner === rb.name) wins++;
  }
  if (wins < games - 1) throw new Error(`rule-based only won ${wins}/${games} vs random`);
  console.log(`ok rule-based beats random ${wins}/${games}`);

  // AlphaZero (even with a tiny budget) should beat random convincingly.
  wins = 0;
  for (let i = 0; i < 6; i++) {
    const az = new AlphaZeroAgent(300 + i, net, 40), rnd = new RandomAgent(400 + i);
    const winner = i % 2 === 0 ? await playGame(az, rnd, 9) : await playGame(rnd, az, 9);
    if (winner === az.name) wins++;
  }
  if (wins < 5) throw new Error(`AlphaZero only won ${wins}/6 vs random`);
  console.log(`ok AlphaZero beats random ${wins}/6 on 9x9`);

  console.log('\nall JS agent tests passed');
}

main().catch((e) => { console.error(e); process.exit(1); });
