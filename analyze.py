#!/usr/bin/env python3
"""Summarise a training run's log into learning curves.

    python3 analyze.py --run-dir runs/az_hex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_log(run_dir: Path) -> list[dict]:
    records = []
    with open(run_dir / "log.jsonl") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("event") == "start":
                continue
            records.append(rec)
    return records


def summarise(records: list[dict]) -> dict:
    curves = {
        "iteration": [], "stage": [], "policy_loss": [], "value_loss": [],
        "policy_top1": [], "avg_plies": [], "games": [], "selfplay_sec": [],
        "train_sec": [], "positions": [],
    }
    evals = {"iteration": [], "stage": [], "vs_random": [], "vs_rule_based": [],
             "vs_start_checkpoint": []}
    for r in records:
        curves["iteration"].append(r["iteration"])
        curves["stage"].append(r["stage"])
        curves["policy_loss"].append(r["policy_loss"])
        curves["value_loss"].append(r["value_loss"])
        curves["policy_top1"].append(r["policy_top1_agreement"])
        curves["avg_plies"].append(r["avg_plies"])
        curves["games"].append(r["games"])
        curves["positions"].append(r["new_positions"])
        curves["selfplay_sec"].append(r["selfplay_sec"])
        curves["train_sec"].append(r["train_sec"])
        if "vs_rule_based" in r:
            evals["iteration"].append(r["iteration"])
            evals["stage"].append(r["stage"])
            evals["vs_random"].append(r["vs_random"])
            evals["vs_rule_based"].append(r["vs_rule_based"])
            evals["vs_start_checkpoint"].append(r.get("vs_start_checkpoint"))

    stages = {}
    for i, s in enumerate(curves["stage"]):
        st = stages.setdefault(s, {"iters": 0, "games": 0, "positions": 0, "seconds": 0.0})
        st["iters"] += 1
        st["games"] += curves["games"][i]
        st["positions"] += curves["positions"][i]
        st["seconds"] += curves["selfplay_sec"][i] + curves["train_sec"][i]

    return {
        "curves": curves,
        "evals": evals,
        "per_stage": stages,
        "total_games": sum(curves["games"]),
        "total_positions": sum(curves["positions"]),
        "total_hours": sum(curves["selfplay_sec"] + curves["train_sec"]) / 3600.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/az_hex")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    summary = summarise(load_log(run_dir))
    out = Path(args.out) if args.out else run_dir / "curves.json"
    out.write_text(json.dumps(summary, indent=2))

    print(f"total: {summary['total_games']} self-play games, "
          f"{summary['total_positions']} positions, {summary['total_hours']:.2f} h")
    print("\nper board size:")
    for size, st in sorted(summary["per_stage"].items()):
        print(f"  {size}x{size}: {st['iters']:3d} iters  {st['games']:6d} games  "
              f"{st['positions']:8d} positions  {st['seconds'] / 3600:.2f} h")
    if summary["evals"]["iteration"]:
        print("\nperiodic evaluation (win rate as AlphaZero):")
        e = summary["evals"]
        for i in range(len(e["iteration"])):
            print(f"  it {e['iteration'][i]:3d} ({e['stage'][i]}x{e['stage'][i]}): "
                  f"vs random {e['vs_random'][i]:.2f}   vs rule-based {e['vs_rule_based'][i]:.2f}")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
