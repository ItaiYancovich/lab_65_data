"""Parallel round-robin tournament and report generation."""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .arena import balanced_openings, elo_standard_errors, fit_elo, play_game
from .registry import AgentSpec, build_agent


def _play_one(task):
    """Worker entry point: play a single game from a pair of specs."""
    black_spec, white_spec, board_size, opening, seed = task
    torch.set_num_threads(1)
    black = build_agent(black_spec, seed, board_size)
    white = build_agent(white_spec, seed + 1, board_size)
    rec = play_game(black, white, board_size, opening=opening, record_moves=True)
    return {
        "black": black_spec.label,
        "white": white_spec.label,
        "winner": rec.winner,
        "plies": rec.plies,
        "opening": opening,
        "moves": rec.moves,
        "black_seconds": rec.black_seconds,
        "white_seconds": rec.white_seconds,
        "black_moves": rec.black_moves,
        "white_moves": rec.white_moves,
    }


def build_schedule(specs: list[AgentSpec], board_size: int, games_per_pair: int, seed: int = 0):
    """Every ordered pair, every opening played from both sides."""
    rng = np.random.default_rng(seed)
    pairs = max(1, games_per_pair // 2)
    openings = balanced_openings(board_size, pairs, rng)
    tasks = []
    for a, b in itertools.combinations(specs, 2):
        for k in range(pairs):
            op = openings[k % len(openings)]
            s = int(rng.integers(1 << 30))
            tasks.append((a, b, board_size, op, s))
            tasks.append((b, a, board_size, op, s + 7))
    return tasks


def run_tournament(
    specs: list[AgentSpec],
    board_size: int = 11,
    games_per_pair: int = 20,
    workers: int = 4,
    seed: int = 0,
    out_dir: str | Path = "runs/tournament",
    verbose: bool = True,
    anchor: str | None = None,
    anchor_rating: float = 0.0,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = build_schedule(specs, board_size, games_per_pair, seed)
    if verbose:
        print(f"tournament: {len(specs)} agents, {len(tasks)} games on "
              f"{board_size}x{board_size}, {workers} workers", flush=True)

    t0 = time.time()
    games: list[dict] = []
    if workers > 1:
        import concurrent.futures as cf
        import multiprocessing as mp

        ctx = mp.get_context("fork")
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            for i, res in enumerate(pool.map(_play_one, tasks, chunksize=1)):
                games.append(res)
                if verbose and (i + 1) % 10 == 0:
                    el = time.time() - t0
                    rate = (i + 1) / el
                    print(f"  {i + 1}/{len(tasks)} games ({el:.0f}s, "
                          f"eta {(len(tasks) - i - 1) / max(rate, 1e-9) / 60:.0f} min)", flush=True)
    else:
        for i, task in enumerate(tasks):
            games.append(_play_one(task))

    summary = summarise(games, specs, anchor=anchor, anchor_rating=anchor_rating)
    summary["board_size"] = board_size
    summary["games_per_pair"] = games_per_pair
    summary["total_seconds"] = time.time() - t0
    with open(out_dir / "games.json", "w") as fh:
        json.dump(games, fh)
    with open(out_dir / "summary.json", "w") as fh:
        json.dump({k: v for k, v in summary.items() if k != "games"}, fh, indent=2)
    return summary


def summarise(
    games: list[dict],
    specs: list[AgentSpec],
    anchor: str | None = None,
    anchor_rating: float = 0.0,
) -> dict:
    labels = [s.label for s in specs]
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    wins = np.zeros((k, k), dtype=int)  # wins[i][j] = i beat j
    black_wins = 0
    seconds = {l: 0.0 for l in labels}
    move_counts = {l: 0 for l in labels}
    plies = []

    for g in games:
        bi, wi = idx[g["black"]], idx[g["white"]]
        if g["winner"] == g["black"]:
            wins[bi, wi] += 1
            black_wins += 1
        else:
            wins[wi, bi] += 1
        seconds[g["black"]] += g["black_seconds"]
        seconds[g["white"]] += g["white_seconds"]
        move_counts[g["black"]] += g["black_moves"]
        move_counts[g["white"]] += g["white_moves"]
        plies.append(g["plies"])

    results = []
    for i in range(k):
        for j in range(i + 1, k):
            results.append((labels[i], labels[j], int(wins[i, j]), int(wins[j, i])))
    # Anchoring on a shared agent puts separate tournaments on one scale.
    elo = fit_elo(results, anchor=anchor or labels[0], anchor_rating=anchor_rating)
    errs = elo_standard_errors(results, elo)

    table = []
    for i, label in enumerate(labels):
        w = int(wins[i].sum())
        l = int(wins[:, i].sum())
        n = w + l
        table.append({
            "agent": label,
            "wins": w,
            "losses": l,
            "games": n,
            "win_rate": w / max(n, 1),
            "elo": elo[label],
            "elo_err": errs[label],
            "sec_per_move": seconds[label] / max(move_counts[label], 1),
        })
    table.sort(key=lambda r: -r["elo"])
    return {
        "table": table,
        "matrix": wins.tolist(),
        "labels": labels,
        "elo": elo,
        "black_win_rate": black_wins / max(len(games), 1),
        "avg_plies": float(np.mean(plies)) if plies else 0.0,
        "n_games": len(games),
    }


def format_report(summary: dict) -> str:
    """Markdown report of the tournament."""
    labels = summary["labels"]
    matrix = np.array(summary["matrix"])
    n = summary["board_size"]
    lines = [
        f"# AlphaZero for Hex {n}x{n} -- tournament results",
        "",
        f"{summary['n_games']} games, {summary['games_per_pair']} per pairing, "
        f"colours balanced (every opening played from both sides).",
        f"Average game length {summary['avg_plies']:.0f} plies. "
        f"Black (first player) won {summary['black_win_rate'] * 100:.1f}% of all games.",
        "",
        "## Standings",
        "",
        "| # | Agent | W | L | Win % | Elo | +/- | s/move |",
        "|---|-------|---|---|-------|-----|-----|--------|",
    ]
    for i, row in enumerate(summary["table"], 1):
        lines.append(
            f"| {i} | {row['agent']} | {row['wins']} | {row['losses']} | "
            f"{row['win_rate'] * 100:.1f}% | {row['elo']:.0f} | "
            f"{row['elo_err']:.0f} | {row['sec_per_move']:.2f} |"
        )
    lines += ["", "Elo is a Bradley-Terry maximum-likelihood fit, anchored at "
              f"`{labels[0]}` = 0.", "", "## Head-to-head (row's wins against column)", ""]
    short = [l.split(" (")[0][:18] for l in labels]
    lines.append("| |" + "|".join(short) + "|")
    lines.append("|---|" + "|".join("---" for _ in short) + "|")
    for i, label in enumerate(short):
        cells = []
        for j in range(len(labels)):
            if i == j:
                cells.append("--")
            else:
                total = matrix[i, j] + matrix[j, i]
                cells.append(f"{matrix[i, j]}-{matrix[j, i]}" if total else "-")
        lines.append(f"|{label}|" + "|".join(cells) + "|")
    return "\n".join(lines) + "\n"
