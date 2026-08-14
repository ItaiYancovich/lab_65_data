# AlphaZero for Hex 11x11 -- tournament results

450 games, 30 per pairing, colours balanced (every opening played from both sides).
Average game length 38 plies. Black (first player) won 50.0% of all games.

## Standings

| # | Agent | W | L | Win % | Elo | +/- | s/move |
|---|-------|---|---|-------|-----|-----|--------|
| 1 | AlphaZero (400 sims) | 149 | 1 | 99.3% | 2653 | 177 | 0.47 |
| 2 | AlphaZero policy only (no search) | 121 | 29 | 80.7% | 2137 | 137 | 0.00 |
| 3 | classic MCTS rollouts (1.0s/move) | 88 | 62 | 58.7% | 1527 | 96 | 0.98 |
| 4 | brute-force alpha-beta (1.0s/move) | 60 | 90 | 40.0% | 1134 | 96 | 0.79 |
| 5 | rule-based (two-distance+bridges) | 32 | 118 | 21.3% | 726 | 110 | 0.00 |
| 6 | random | 0 | 150 | 0.0% | 0 | 227 | 0.00 |

Elo is a Bradley-Terry maximum-likelihood fit, anchored at `random` = 0.

## Head-to-head (row's wins against column)

| |random|rule-based|brute-force alpha-|classic MCTS rollo|AlphaZero policy o|AlphaZero|
|---|---|---|---|---|---|---|
|random|--|0-30|0-30|0-30|0-30|0-30|
|rule-based|30-0|--|1-29|1-29|0-30|0-30|
|brute-force alpha-|30-0|29-1|--|1-29|0-30|0-30|
|classic MCTS rollo|30-0|29-1|29-1|--|0-30|0-30|
|AlphaZero policy o|30-0|30-0|30-0|30-0|--|1-29|
|AlphaZero|30-0|30-0|30-0|30-0|29-1|--|
