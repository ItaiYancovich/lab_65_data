#!/usr/bin/env python3
"""Match the trained AlphaZero agent against the classical Hex engines.

    python3 run_tournament.py --ckpt runs/az_hex/final.pt --games-per-pair 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

from alphazero_hex.registry import default_field, spec
from alphazero_hex.tournament import format_report, run_tournament


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="runs/az_hex/final.pt")
    ap.add_argument("--board-size", type=int, default=11)
    ap.add_argument("--games-per-pair", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="runs/tournament")
    ap.add_argument("--az-sims", type=int, default=400)
    ap.add_argument("--minimax-time", type=float, default=1.0)
    ap.add_argument("--rollout-time", type=float, default=1.0,
                    help="seconds per move for the classic rollout-MCTS agent")
    ap.add_argument("--rollout-sims", type=int, default=200_000,
                    help="simulation cap for the rollout agent (the time budget usually binds first)")
    ap.add_argument("--include-random", action="store_true", default=True)
    ap.add_argument("--scaling", action="store_true",
                    help="also enter AlphaZero at several simulation counts")
    ap.add_argument("--ladder", action="store_true",
                    help="enter each curriculum checkpoint, to rate training progress")
    args = ap.parse_args()

    field = default_field(args.ckpt)
    # Apply CLI overrides to the standard field.
    field = [
        spec("random", "random"),
        spec("rule", "rule-based (two-distance+bridges)", noise=0.05),
        spec("minimax", f"brute-force alpha-beta ({args.minimax_time}s/move)",
             time_budget=args.minimax_time, beam=8),
        spec("rollout", f"classic MCTS rollouts ({args.rollout_time}s/move)",
             simulations=args.rollout_sims, time_budget=args.rollout_time),
        spec("policy", "AlphaZero policy only (no search)", ckpt=args.ckpt),
        spec("az", f"AlphaZero ({args.az_sims} sims)", ckpt=args.ckpt,
             simulations=args.az_sims),
    ]
    if args.scaling:
        field += [
            spec("az", "AlphaZero (100 sims)", ckpt=args.ckpt, simulations=100),
            spec("az", "AlphaZero (800 sims)", ckpt=args.ckpt, simulations=800),
        ]
    if args.ladder:
        # Rate each curriculum checkpoint at identical search, so the Elo gaps
        # measure what training added rather than what search added.
        ckpt_dir = Path(args.ckpt).parent
        final = Path(args.ckpt).resolve()
        for p in sorted(ckpt_dir.glob("stage_*.pt")):
            if p.resolve() == final:
                continue
            label = p.stem.replace("stage_", "").replace("_", " ")
            field.append(spec("az", f"AlphaZero @ {label} ({args.az_sims} sims)",
                              ckpt=str(p), simulations=args.az_sims))

    summary = run_tournament(
        field,
        board_size=args.board_size,
        games_per_pair=args.games_per_pair,
        workers=args.workers,
        seed=args.seed,
        out_dir=args.out_dir,
    )
    report = format_report(summary)
    out = Path(args.out_dir) / "report.md"
    out.write_text(report)
    print("\n" + report)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
