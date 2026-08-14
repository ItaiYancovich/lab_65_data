#!/usr/bin/env python3
"""Two follow-up studies on the trained agent.

**Ladder** -- every curriculum checkpoint at identical search strength, so the
Elo gaps measure what *training* added rather than what search added.

**Search scaling** -- the final network at several simulation counts, which
measures what the *search* adds on top of a fixed network. AlphaZero's
signature property is that strength keeps climbing with search depth.

Both are anchored on the rule-based agent at the Elo it earned in the main
tournament, so all three tables share one scale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alphazero_hex.registry import spec
from alphazero_hex.tournament import format_report, run_tournament

RULE_LABEL = "rule-based (two-distance+bridges)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="runs/az_hex")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ladder-games", type=int, default=16)
    ap.add_argument("--scaling-games", type=int, default=20)
    ap.add_argument("--main-summary", default="runs/tournament/summary.json",
                    help="used to anchor these tables to the main tournament's Elo scale")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    final = run_dir / "final.pt"

    anchor_rating = 0.0
    main_path = Path(args.main_summary)
    if main_path.exists():
        main = json.loads(main_path.read_text())
        anchor_rating = main["elo"].get(RULE_LABEL, 0.0)
        print(f"anchoring on '{RULE_LABEL}' = {anchor_rating:.0f} Elo "
              f"(from the main tournament)\n")

    # ------------------------------------------------------------- ladder
    rungs = [
        ("stage_5.pt", "after 5x5 stage"),
        ("stage_7.pt", "after 7x7 stage"),
        ("stage_9.pt", "after 9x9 stage"),
        ("stage_11a_after20iters.pt", "after 11x11 (20 iters)"),
    ]
    ladder = [spec("rule", RULE_LABEL, noise=0.05)]
    for fname, label in rungs:
        p = run_dir / fname
        if p.exists():
            ladder.append(spec("az", f"AlphaZero {label}", ckpt=str(p), simulations=400))
    ladder.append(spec("az", "AlphaZero final (11x11, 30 iters)",
                       ckpt=str(final), simulations=400))

    print("=== curriculum ladder (identical 400-simulation search throughout) ===")
    s1 = run_tournament(ladder, board_size=11, games_per_pair=args.ladder_games,
                        workers=args.workers, seed=11,
                        out_dir=Path(args.out_dir) / "ladder",
                        anchor=RULE_LABEL, anchor_rating=anchor_rating)
    r1 = format_report(s1).replace("tournament results", "curriculum ladder")
    (Path(args.out_dir) / "ladder" / "report.md").write_text(r1)
    print("\n" + r1)

    # ------------------------------------------------------ search scaling
    scaling = [spec("rule", RULE_LABEL, noise=0.05)]
    for sims in (25, 50, 100, 200, 400, 800):
        scaling.append(spec("az", f"AlphaZero {sims} sims", ckpt=str(final), simulations=sims))

    print("\n=== search scaling (one fixed network, varying simulations) ===")
    s2 = run_tournament(scaling, board_size=11, games_per_pair=args.scaling_games,
                        workers=args.workers, seed=22,
                        out_dir=Path(args.out_dir) / "scaling",
                        anchor=RULE_LABEL, anchor_rating=anchor_rating)
    r2 = format_report(s2).replace("tournament results", "search scaling")
    (Path(args.out_dir) / "scaling" / "report.md").write_text(r2)
    print("\n" + r2)


if __name__ == "__main__":
    main()
