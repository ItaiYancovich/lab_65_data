# AlphaZero for Hex 11x11

A from-scratch implementation of the AlphaGo Zero / AlphaZero algorithm, trained
by pure self-play on 11x11 Hex, then measured against classical game-playing
opponents: a rule-based expert, a "smart brute force" alpha-beta searcher, and
pre-neural Monte-Carlo tree search with random rollouts.

Everything here is trained on **four CPU cores, no GPU**, starting from random
weights and the rules of the game alone. No human games, no opening book, no
hand-crafted features in the network's input.

## Quick start

```bash
pip install numpy torch --index-url https://download.pytorch.org/whl/cpu

python3 train_hex.py --run-dir runs/az_hex          # full curriculum
python3 run_tournament.py --ckpt runs/az_hex/final.pt --games-per-pair 20
python3 play_hex.py --ckpt runs/az_hex/final.pt     # play it yourself

python3 tests/test_hex.py && python3 tests/test_mcts.py \
  && python3 tests/test_agents.py && python3 tests/test_arena.py
```

## The game

Hex is played on an `n x n` rhombus of hexagons. Black connects top to bottom,
White connects left to right. Two properties shape the whole design:

* **Hex can never be drawn.** Once the board is full, exactly one player owns a
  crossing connection. So the value target is always a clean +-1, and a random
  playout never needs an evaluation function -- fill the board and read off the
  winner.
* **The first player has a large advantage** on 11x11. Every comparison in this
  repo therefore plays each opening from both sides, so colour cancels exactly.

## The algorithm

The AlphaZero loop, unchanged in substance from the paper:

1. **Self-play.** Play games with PUCT Monte-Carlo tree search guided by the
   current network. Dirichlet noise at the root and temperature sampling early
   in the game supply exploration.
2. **Record.** For every position store the search's visit distribution `pi`
   and, when the game ends, the outcome `z` from that position's point of view.
3. **Train.** Fit the policy head to `pi` by cross-entropy and the value head to
   `z` by mean-squared error.
4. Repeat with the improved network.

Search is the policy improvement operator; the network distils the search back
into a single forward pass; the stronger network makes the next search better.

### Design choices specific to Hex

**Canonical side-to-move encoding.** When White is to move, the board is
transposed and the colours swapped. White connecting left-right becomes "the
player to move connecting top-bottom", so the network only ever learns one
point of view. This halves the amount of self-play needed for a given strength.

**A fully-convolutional, size-agnostic network.** The policy head is a 1x1
convolution emitting one logit per cell and the value head pools globally, so
the same weights run on any board size.

**A board-size curriculum: 5x5 -> 7x7 -> 9x9 -> 11x11.** This is the single
biggest reason the run is feasible on a CPU. A 5x5 game is ~15 plies instead of
~70 and each position is ~4x cheaper to evaluate, so early games cost roughly
25x less. Hex tactics -- bridges, ladders, edge templates -- are *local*
patterns, and a convolutional network carries them straight across board sizes.
The 11x11 stage therefore begins from a player that already understands the
tactics and only has to learn large-board strategy.

**180-degree symmetry augmentation.** Rotating an `n x n` Hex board by 180
degrees maps top-bottom to top-bottom and left-right to left-right, so it
preserves both players' goals. (Transposing does *not*: it swaps the two
players' objectives, so it is not a usable augmentation under the canonical
encoding.)

### Making it fast enough on four cores

| Technique | Effect |
|---|---|
| Batched self-play: many games in flight, all their MCTS leaves evaluated in one forward pass | the dominant win; a batch of 1 leaves most of the CPU idle |
| Multi-process self-play (4 workers, 1 torch thread each) | ~2.4x over a single 4-thread process |
| BatchNorm folded into convolutions for inference (`torch.jit.optimize_for_inference`) | ~1.4x |
| Virtual-loss leaf batching inside a single search | lets the agent batch at *play* time too |
| Incremental union-find win detection | O(alpha) per move instead of a flood fill |
| Vectorized PUCT selection (numpy over all children at once) | keeps Python overhead near ~35% of wall clock |

Measured self-play throughput on 11x11 at 160 simulations per move: about
**3,800-4,200 network evaluations per second**, roughly 1,000-1,400 complete
games per hour.

## The opponents

| Agent | What it is |
|---|---|
| `random` | uniform legal move; the floor of the rating scale |
| `rule-based` | Anshelevich **two-distance** connection strength, plus win-now / block-the-only-threat / restore-a-broken-bridge responses. No search. |
| `brute-force alpha-beta` | **Smart brute force**: negamax with alpha-beta, iterative deepening under a wall-clock budget, a transposition table, forced-move extensions, and beam pruning to the top-k cells by two-distance. Plain minimax is hopeless at branching factor 121, so the pruning is what makes it a real opponent. |
| `classic MCTS rollouts` | UCT with uniformly random playouts -- the strongest approach known for Hex before neural networks, and unusually effective here because a filled Hex board always has a winner. |
| `AlphaZero policy only` | the trained network's policy head, no search. Isolates what the search contributes. |
| `AlphaZero` | the trained network plus PUCT search. |

**Two-distance**, used by both hand-written agents, is the standard Hex
connection measure: an empty cell is charged the *second* smallest neighbour
distance plus one, so a cell only becomes cheap when it has two independent
routes through -- exactly the redundancy that makes a Hex connection hard to
cut. Ordinary shortest path badly overrates a connection hanging on one cell.

## Layout

```
alphazero_hex/
  hex_game.py    rules, union-find win detection, canonical encoding
  net.py         size-agnostic residual policy/value network
  mcts.py        PUCT search; coroutine-style so leaves can be batched
  evaluator.py   batched network evaluation (JIT, legality masking)
  selfplay.py    many games in flight, one forward pass per round trip
  train.py       the AlphaZero iteration and the board-size curriculum
  heuristics.py  two-distance, bridges, fast win detection
  arena.py       colour-balanced matches and Bradley-Terry Elo
  tournament.py  parallel round-robin and report generation
  registry.py    picklable agent specs for worker processes
  agents/        random, rule-based, alpha-beta, rollout MCTS, AlphaZero
tests/           engine, search, agent and rating tests
```

## Results

See [`runs/tournament/report.md`](runs/tournament/report.md) and
`RESULTS.md` for the tournament tables and what they show.
