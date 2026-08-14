"""AlphaZero training loop for Hex, with a board-size curriculum.

Each iteration is the AlphaZero cycle: generate self-play games with the
current network, append them to a replay buffer, and fit the network to the
search's visit counts (policy) and the game outcomes (value).

Two things make this feasible on four CPU cores:

* **Batched, multi-process self-play.**  Games are played concurrently and
  their leaf positions evaluated in one forward pass per round trip.
* **A board-size curriculum.**  The network is fully convolutional, so the
  weights learned on 5x5 transfer straight to 7x7, 9x9 and finally 11x11.
  Hex tactics (bridges, ladders, edge templates) are local patterns; learning
  them where games are 20 plies long instead of 70 is far cheaper, and the
  11x11 run then starts from a player that already understands them.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .evaluator import BatchEvaluator
from .hex_game import BLACK, HexBoard
from .mcts import MCTSConfig
from .net import HexNet, NetConfig, count_parameters, save_checkpoint
from .selfplay import Example, SelfPlayConfig, run_self_play, augment


@dataclass
class Stage:
    """One rung of the board-size curriculum."""

    board_size: int
    iterations: int
    games_per_iter: int
    simulations: int
    games_in_flight: int = 32
    lr: float = 2e-3
    temp_moves: int = 0  # 0 -> board_size
    time_budget: float | None = None  # seconds per iteration of self-play


@dataclass
class TrainConfig:
    run_dir: str = "runs/az_hex"
    net: NetConfig = field(default_factory=lambda: NetConfig(channels=64, blocks=5))
    stages: list[Stage] = field(default_factory=list)
    batch_size: int = 256
    sample_reuse: float = 3.0  # each new position is seen ~this many times
    min_train_steps: int = 40
    max_train_steps: int = 400
    buffer_positions: int = 160_000
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    workers: int = 4
    seed: int = 0
    init_from: str | None = None  # checkpoint to continue training from
    eval_every: int = 6
    eval_games: int = 10
    eval_simulations: int = 96


# --------------------------------------------------------------------- data
def examples_to_tensors(batch: list[Example], rng: np.random.Generator):
    """Collate examples into tensors, applying the 180-degree symmetry."""
    n = batch[0].canon_board.shape[0]
    boards = np.empty((len(batch), n, n), dtype=np.uint8)
    pis = np.empty((len(batch), n * n), dtype=np.float32)
    zs = np.empty(len(batch), dtype=np.float32)
    flip = rng.random(len(batch)) < 0.5
    for i, ex in enumerate(batch):
        if flip[i]:
            boards[i] = ex.canon_board[::-1, ::-1]
            pis[i] = ex.pi.reshape(n, n)[::-1, ::-1].reshape(-1)
        else:
            boards[i] = ex.canon_board
            pis[i] = ex.pi
        zs[i] = ex.z
    planes = np.zeros((len(batch), 3, n, n), dtype=np.float32)
    planes[:, 0] = boards == 1
    planes[:, 1] = boards == 2
    planes[:, 2] = 1.0
    return (
        torch.from_numpy(planes),
        torch.from_numpy(pis),
        torch.from_numpy(zs),
    )


# ------------------------------------------------------------- self-play MP
def _worker_init():
    # One thread per worker: the parallelism comes from running four
    # independent self-play processes, not from threading one forward pass.
    torch.set_num_threads(1)


def _worker_selfplay(payload):
    net_cfg, state_dict, sp_cfg, num_games, seed = payload
    torch.set_num_threads(1)
    net = HexNet(NetConfig(**net_cfg))
    net.load_state_dict(state_dict)
    net.eval()
    ev = BatchEvaluator(net, board_size=sp_cfg.board_size)
    return run_self_play(ev, sp_cfg, num_games, np.random.default_rng(seed))


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.run_dir = Path(cfg.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.net = HexNet(cfg.net)
        if cfg.init_from:
            blob = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
            if blob["cfg"] != cfg.net.to_dict():
                raise ValueError(
                    f"checkpoint architecture {blob['cfg']} does not match "
                    f"configured {cfg.net.to_dict()}"
                )
            self.net.load_state_dict(blob["state_dict"])
            self.iteration_offset = int(blob.get("extra", {}).get("iteration", 0))
            print(f"continuing from {cfg.init_from} "
                  f"(iteration {self.iteration_offset})", flush=True)
        else:
            self.iteration_offset = 0
        self.opt = torch.optim.AdamW(
            self.net.parameters(), lr=2e-3, weight_decay=cfg.weight_decay
        )
        self.buffer: deque[Example] = deque(maxlen=cfg.buffer_positions)
        self.iteration = self.iteration_offset
        self.log_path = self.run_dir / "log.jsonl"
        self.pool = None
        if cfg.workers > 1:
            import concurrent.futures as cf
            import multiprocessing as mp

            self.pool = cf.ProcessPoolExecutor(
                max_workers=cfg.workers,
                mp_context=mp.get_context("fork"),
                initializer=_worker_init,
            )

    # ------------------------------------------------------------- logging
    def log(self, record: dict) -> None:
        record["iteration"] = self.iteration
        record["time"] = time.time()
        with open(self.log_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")

    # ----------------------------------------------------------- self-play
    def generate(self, stage: Stage) -> dict:
        sp_cfg = SelfPlayConfig(
            board_size=stage.board_size,
            games_in_flight=stage.games_in_flight,
            temp_moves=stage.temp_moves,
            mcts=MCTSConfig(simulations=stage.simulations),
        )
        if self.pool is None:
            torch.set_num_threads(os.cpu_count() or 4)
            ev = BatchEvaluator(self.net, board_size=stage.board_size)
            examples, stats = run_self_play(
                ev, sp_cfg, stage.games_per_iter, self.rng, time_budget=stage.time_budget
            )
        else:
            state_dict = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
            per_worker = max(1, math.ceil(stage.games_per_iter / self.cfg.workers))
            payloads = [
                (
                    self.cfg.net.to_dict(),
                    state_dict,
                    sp_cfg,
                    per_worker,
                    int(self.rng.integers(1 << 30)),
                )
                for _ in range(self.cfg.workers)
            ]
            results = list(self.pool.map(_worker_selfplay, payloads))
            examples = [e for r in results for e in r[0]]
            stats = {
                "games": sum(r[1]["games"] for r in results),
                "positions": sum(r[1]["positions"] for r in results),
                "seconds": max(r[1]["seconds"] for r in results),
                "avg_plies": float(np.mean([r[1]["avg_plies"] for r in results])),
                "black_win_rate": float(np.mean([r[1]["black_win_rate"] for r in results])),
                "avg_batch": float(np.mean([r[1]["avg_batch"] for r in results])),
            }
            stats["games_per_sec"] = stats["games"] / max(stats["seconds"], 1e-9)
        self.buffer.extend(examples)
        stats["new_positions"] = len(examples)
        stats["buffer"] = len(self.buffer)
        return stats

    # ------------------------------------------------------------ training
    def train_steps(self, stage: Stage, new_positions: int) -> dict:
        steps = int(new_positions * self.cfg.sample_reuse / self.cfg.batch_size)
        steps = int(np.clip(steps, self.cfg.min_train_steps, self.cfg.max_train_steps))
        for g in self.opt.param_groups:
            g["lr"] = stage.lr

        self.net.train()
        torch.set_num_threads(os.cpu_count() or 4)
        buf = list(self.buffer)
        p_losses, v_losses, accs = [], [], []
        for _ in range(steps):
            idx = self.rng.integers(0, len(buf), size=min(self.cfg.batch_size, len(buf)))
            batch = [buf[int(i)] for i in idx]
            x, pi, z = examples_to_tensors(batch, self.rng)
            logits, value = self.net(x)
            # Cross-entropy against the full search distribution (not a label):
            # this is the AlphaZero policy target.
            logp = F.log_softmax(logits, dim=1)
            policy_loss = -(pi * logp).sum(dim=1).mean()
            value_loss = F.mse_loss(value, z)
            loss = policy_loss + self.cfg.value_loss_weight * value_loss
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
            self.opt.step()
            with torch.no_grad():
                p_losses.append(policy_loss.item())
                v_losses.append(value_loss.item())
                accs.append((logits.argmax(1) == pi.argmax(1)).float().mean().item())
        self.net.eval()
        return {
            "train_steps": steps,
            "policy_loss": float(np.mean(p_losses)),
            "value_loss": float(np.mean(v_losses)),
            "policy_top1_agreement": float(np.mean(accs)),
        }

    # ---------------------------------------------------------- evaluation
    def quick_eval(self, stage: Stage) -> dict:
        """Track progress against fixed reference opponents."""
        from .agents.az_agent import AlphaZeroAgent
        from .agents.base import RandomAgent
        from .agents.rule_based import RuleBasedAgent
        from .arena import play_match

        torch.set_num_threads(os.cpu_count() or 4)
        ev = BatchEvaluator(self.net, board_size=stage.board_size)
        az = AlphaZeroAgent(ev, simulations=self.cfg.eval_simulations, seed=int(self.rng.integers(1 << 30)))
        out = {}
        for name, opp in (
            ("vs_random", RandomAgent(seed=int(self.rng.integers(1 << 30)))),
            ("vs_rule_based", RuleBasedAgent(seed=int(self.rng.integers(1 << 30)), noise=0.05)),
        ):
            res = play_match(az, opp, stage.board_size, self.cfg.eval_games,
                             rng=np.random.default_rng(int(self.rng.integers(1 << 30))))
            out[name] = res["a_win_rate"]
        return out

    # --------------------------------------------------------------- driver
    def run(self) -> None:
        cfg = self.cfg
        print(f"network: {count_parameters(self.net) / 1e3:.0f}k parameters "
              f"({cfg.net.channels}ch x {cfg.net.blocks} blocks)", flush=True)
        self.log({"event": "start", "config": {
            "net": cfg.net.to_dict(),
            "stages": [asdict(s) for s in cfg.stages],
            "workers": cfg.workers,
        }})
        prev_size = None
        for si, stage in enumerate(cfg.stages):
            print(f"\n=== stage {si + 1}/{len(cfg.stages)}: {stage.board_size}x{stage.board_size}, "
                  f"{stage.iterations} iters x {stage.games_per_iter} games @ {stage.simulations} sims, "
                  f"lr {stage.lr} ===", flush=True)
            if prev_size is not None and prev_size != stage.board_size:
                # Positions from a smaller board are a different game; keep the
                # weights (that is the point of the curriculum), drop the data.
                self.buffer.clear()
            prev_size = stage.board_size
            for it in range(stage.iterations):
                self.iteration += 1
                t0 = time.time()
                sp = self.generate(stage)
                t1 = time.time()
                tr = self.train_steps(stage, sp["new_positions"])
                t2 = time.time()
                record = {"stage": stage.board_size, **sp, **tr,
                          "selfplay_sec": t1 - t0, "train_sec": t2 - t1}
                if cfg.eval_every and (it + 1) % cfg.eval_every == 0:
                    record.update(self.quick_eval(stage))
                self.log(record)
                msg = (f"  it {self.iteration:3d} | {stage.board_size}x{stage.board_size} "
                       f"| games {sp['games']:4d} pos {sp['new_positions']:6d} "
                       f"| plies {sp['avg_plies']:5.1f} | P {tr['policy_loss']:.3f} "
                       f"V {tr['value_loss']:.3f} top1 {tr['policy_top1_agreement']:.2f} "
                       f"| sp {t1 - t0:5.0f}s tr {t2 - t1:4.0f}s")
                if "vs_rule_based" in record:
                    msg += f" | vs rand {record['vs_random']:.2f} vs rules {record['vs_rule_based']:.2f}"
                print(msg, flush=True)
                save_checkpoint(self.run_dir / "latest.pt", self.net,
                                {"iteration": self.iteration, "stage": stage.board_size})
            save_checkpoint(self.run_dir / f"stage_{stage.board_size}.pt", self.net,
                            {"iteration": self.iteration, "stage": stage.board_size})
        save_checkpoint(self.run_dir / "final.pt", self.net, {"iteration": self.iteration})
        print("\ntraining complete ->", self.run_dir / "final.pt", flush=True)

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)


def default_curriculum() -> list[Stage]:
    """Cheap tactics first, then scale the board up.

    Sizes are ordered by cost per game: a 5x5 game is ~20 plies and evaluates
    ~25x faster than an 11x11 one, so the early stages buy bridge and
    edge-template knowledge for almost nothing.  ``games_in_flight`` is
    per worker and is kept at or below ``games_per_iter / workers`` so every
    worker actually has a full batch of positions to evaluate at once.

    The final size is split into two stages purely to anneal the learning rate.
    """
    return [
        Stage(board_size=5, iterations=14, games_per_iter=256, simulations=96,
              games_in_flight=32, lr=2e-3),
        Stage(board_size=7, iterations=12, games_per_iter=192, simulations=128,
              games_in_flight=24, lr=2e-3),
        Stage(board_size=9, iterations=10, games_per_iter=128, simulations=128,
              games_in_flight=16, lr=1.5e-3),
        Stage(board_size=11, iterations=20, games_per_iter=128, simulations=160,
              games_in_flight=16, lr=1e-3),
        Stage(board_size=11, iterations=10, games_per_iter=128, simulations=160,
              games_in_flight=16, lr=4e-4),
    ]
