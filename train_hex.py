#!/usr/bin/env python3
"""Train the AlphaZero Hex agent.

Examples
--------
    python3 train_hex.py --run-dir runs/az_hex            # full curriculum
    python3 train_hex.py --preset quick --run-dir runs/q  # a few minutes
"""

from __future__ import annotations

import argparse

from alphazero_hex.net import NetConfig
from alphazero_hex.train import Stage, TrainConfig, Trainer, default_curriculum


def quick_curriculum() -> list[Stage]:
    return [
        Stage(board_size=5, iterations=4, games_per_iter=128, simulations=64,
              games_in_flight=32, lr=2e-3),
        Stage(board_size=7, iterations=3, games_per_iter=96, simulations=64,
              games_in_flight=24, lr=2e-3),
        Stage(board_size=11, iterations=3, games_per_iter=48, simulations=64,
              games_in_flight=12, lr=1e-3),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="runs/az_hex")
    ap.add_argument("--preset", choices=["full", "quick", "board11-only"], default="full")
    ap.add_argument("--init-from", default=None,
                    help="checkpoint to continue training from (must match --channels/--blocks)")
    ap.add_argument("--lr", type=float, default=None, help="override every stage's learning rate")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--iters-11", type=int, default=None,
                    help="override the number of 11x11 iterations")
    args = ap.parse_args()

    if args.preset == "quick":
        stages = quick_curriculum()
    elif args.preset == "board11-only":
        stages = [Stage(board_size=11, iterations=args.iters_11 or 40, games_per_iter=128,
                        simulations=160, games_in_flight=16, lr=1e-3)]
    else:
        stages = default_curriculum()
        if args.iters_11 is not None:
            stages[-1].iterations = args.iters_11

    if args.lr is not None:
        for s in stages:
            s.lr = args.lr

    cfg = TrainConfig(
        run_dir=args.run_dir,
        net=NetConfig(channels=args.channels, blocks=args.blocks),
        stages=stages,
        workers=args.workers,
        seed=args.seed,
        init_from=args.init_from,
        eval_every=args.eval_every,
    )
    trainer = Trainer(cfg)
    try:
        trainer.run()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
