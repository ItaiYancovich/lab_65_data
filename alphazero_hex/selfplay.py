"""Vectorized self-play.

Many games are advanced concurrently and the leaf positions they produce are
evaluated in one batched forward pass.  On a CPU this is worth roughly an order
of magnitude over playing games one at a time, because a batch-of-1 convolution
leaves almost all of the machine idle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .evaluator import BatchEvaluator
from .hex_game import BLACK, HexBoard, rotate180_move
from .mcts import MCTSConfig, Search


@dataclass
class SelfPlayConfig:
    board_size: int = 11
    games_in_flight: int = 48
    temp_moves: int = 0  # 0 -> board_size (about one row's worth of plies)
    temp: float = 1.0
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    tree_reuse: bool = True
    random_opening_prob: float = 0.25  # play ply 1 uniformly at random sometimes

    def temperature_moves(self) -> int:
        return self.temp_moves if self.temp_moves > 0 else self.board_size


@dataclass
class Example:
    canon_board: np.ndarray  # uint8 (n, n): 0 empty, 1 side-to-move, 2 opponent
    pi: np.ndarray  # float32 (n*n,) in the canonical frame
    z: float = 0.0  # filled in when the game ends
    player: int = 0  # real colour of the side to move at this position


class SelfPlayGame:
    """A single game being played by MCTS, driven one evaluation at a time."""

    def __init__(self, cfg: SelfPlayConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.state = HexBoard(cfg.board_size)
        self.examples: list[Example] = []
        self.search: Search | None = None
        self.finished = False
        self.winner = 0
        if rng.random() < cfg.random_opening_prob:
            self.state.play(int(rng.integers(self.state.ncells)))

    # ---------------------------------------------------------------- driver
    def prepare(self) -> HexBoard | None:
        """Advance until a network evaluation is needed; ``None`` when done."""
        while not self.finished:
            if self.search is None:
                self.search = Search(self.state.copy(), self.cfg.mcts, self.rng)
            leaf = self.search.next_leaf()
            if leaf is not None:
                return leaf[0]
            self._play_move()
        return None

    def receive(self, priors: np.ndarray, value: float) -> None:
        self.search.expand(self.search._pending_node, priors, float(value))

    # ------------------------------------------------------------ internals
    def _play_move(self) -> None:
        search = self.search
        n = self.state.n
        ply = self.state.move_count
        temp = self.cfg.temp if ply < self.cfg.temperature_moves() else 0.0
        moves, probs = search.policy_target(self.cfg.temp)

        # Training target always uses the visit distribution at temperature 1;
        # only the *played* move is sharpened later in the game.
        pi = np.zeros(n * n, dtype=np.float32)
        for m, p in zip(moves, probs):
            pi[self.state.to_canonical_move(int(m))] = p
        canon = self.state.array()
        canon = canon if self.state.to_move == BLACK else canon.T
        if self.state.to_move != BLACK:
            canon = np.where(canon == 0, 0, 3 - canon)  # swap colours
        self.examples.append(Example(canon.astype(np.uint8).copy(), pi, player=self.state.to_move))

        if temp <= 1e-3:
            choice = int(np.argmax(search.N[0]))
        else:
            _, play_probs = search.policy_target(temp)
            p64 = play_probs.astype(np.float64)
            p64 /= p64.sum()  # renormalise in float64; np.random.choice is strict
            choice = int(self.rng.choice(len(moves), p=p64))
        move = int(moves[choice])

        self.state.play(move)
        if self.state.is_terminal():
            self._finish()
            return
        self.search = search.child_search(move) if self.cfg.tree_reuse else None
        if self.search is None:
            self.search = Search(self.state.copy(), self.cfg.mcts, self.rng)

    def _finish(self) -> None:
        self.finished = True
        self.winner = self.state.winner
        # Each example is stored from the point of view of the side to move at
        # that position, so its target is simply "did that colour go on to win".
        for ex in self.examples:
            ex.z = 1.0 if ex.player == self.winner else -1.0


def run_self_play(
    evaluator: BatchEvaluator,
    cfg: SelfPlayConfig,
    num_games: int,
    rng: np.random.Generator,
    time_budget: float | None = None,
    log_every: int = 0,
) -> tuple[list[Example], dict]:
    """Play ``num_games`` games, returning training examples and stats."""
    start = time.time()
    games = [SelfPlayGame(cfg, rng) for _ in range(min(cfg.games_in_flight, num_games))]
    started = len(games)
    completed = 0
    examples: list[Example] = []
    black_wins = 0
    total_plies = 0
    nn_batches = 0
    nn_positions = 0

    while games:
        pending_games = []
        pending_states = []
        for g in games:
            st = g.prepare()
            if st is not None:
                pending_games.append(g)
                pending_states.append(st)

        if pending_states:
            priors, values = evaluator.evaluate(pending_states)
            nn_batches += 1
            nn_positions += len(pending_states)
            for i, g in enumerate(pending_games):
                g.receive(priors[i], values[i])

        # Retire finished games, refill the pool.
        still: list[SelfPlayGame] = []
        for g in games:
            if g.finished:
                examples.extend(g.examples)
                completed += 1
                total_plies += len(g.examples)
                black_wins += 1 if g.winner == BLACK else 0
                if log_every and completed % log_every == 0:
                    el = time.time() - start
                    print(
                        f"    self-play {completed}/{num_games} games "
                        f"({el:.0f}s, {completed / max(el, 1e-9):.2f} g/s, "
                        f"{len(examples)} positions)",
                        flush=True,
                    )
                out_of_time = time_budget is not None and (time.time() - start) > time_budget
                if started < num_games and not out_of_time:
                    started += 1
                    still.append(SelfPlayGame(cfg, rng))
            else:
                still.append(g)
        games = still

    elapsed = time.time() - start
    stats = {
        "games": completed,
        "positions": len(examples),
        "seconds": elapsed,
        "games_per_sec": completed / max(elapsed, 1e-9),
        "avg_plies": total_plies / max(completed, 1),
        "black_win_rate": black_wins / max(completed, 1),
        "nn_batches": nn_batches,
        "avg_batch": nn_positions / max(nn_batches, 1),
    }
    return examples, stats


def augment(ex: Example) -> list[Example]:
    """Hex on a rhombus is invariant under a 180 degree rotation."""
    n = ex.canon_board.shape[0]
    rot_board = ex.canon_board[::-1, ::-1].copy()
    rot_pi = np.zeros_like(ex.pi)
    nz = np.flatnonzero(ex.pi)
    for m in nz:
        rot_pi[rotate180_move(int(m), n)] = ex.pi[m]
    return [ex, Example(rot_board, rot_pi, ex.z, ex.player)]
