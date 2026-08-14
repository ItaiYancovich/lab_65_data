# AlphaZero for Hex 11x11 -- curriculum ladder

240 games, 16 per pairing, colours balanced (every opening played from both sides).
Average game length 40 plies. Black (first player) won 54.2% of all games.

## Standings

| # | Agent | W | L | Win % | Elo | +/- | s/move |
|---|-------|---|---|-------|-----|-----|--------|
| 1 | AlphaZero final (11x11, 30 iters) | 72 | 8 | 90.0% | 2548 | 106 | 0.61 |
| 2 | AlphaZero after 11x11 (20 iters) | 68 | 12 | 85.0% | 2478 | 103 | 0.63 |
| 3 | AlphaZero after 9x9 stage | 50 | 30 | 62.5% | 2173 | 94 | 0.63 |
| 4 | AlphaZero after 7x7 stage | 34 | 46 | 42.5% | 1850 | 94 | 0.62 |
| 5 | AlphaZero after 5x5 stage | 16 | 64 | 20.0% | 1334 | 136 | 0.65 |
| 6 | rule-based (two-distance+bridges) | 0 | 80 | 0.0% | 726 | 239 | 0.00 |

Elo is a Bradley-Terry maximum-likelihood fit, anchored at `rule-based (two-distance+bridges)` = 0.

## Head-to-head (row's wins against column)

| |rule-based|AlphaZero after 5x|AlphaZero after 7x|AlphaZero after 9x|AlphaZero after 11|AlphaZero final|
|---|---|---|---|---|---|---|
|rule-based|--|0-16|0-16|0-16|0-16|0-16|
|AlphaZero after 5x|16-0|--|0-16|0-16|0-16|0-16|
|AlphaZero after 7x|16-0|16-0|--|1-15|1-15|0-16|
|AlphaZero after 9x|16-0|16-0|15-1|--|2-14|1-15|
|AlphaZero after 11|16-0|16-0|15-1|14-2|--|7-9|
|AlphaZero final|16-0|16-0|16-0|15-1|9-7|--|
