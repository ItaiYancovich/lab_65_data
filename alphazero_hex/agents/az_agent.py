"""The trained agent: network + PUCT search (and a search-free ablation)."""

from __future__ import annotations

import numpy as np

from ..evaluator import BatchEvaluator
from ..hex_game import HexBoard
from ..mcts import MCTSConfig, Search
from .base import Agent


class AlphaZeroAgent(Agent):
    """AlphaZero at play time: PUCT search guided by the learned net."""

    def __init__(
        self,
        evaluator: BatchEvaluator,
        simulations: int = 400,
        c_puct: float = 1.6,
        temperature: float = 0.0,
        seed: int | None = None,
        name: str | None = None,
    ):
        self.evaluator = evaluator
        self.cfg = MCTSConfig(simulations=simulations, c_puct=c_puct, add_noise=False)
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)
        self.name = name or f"alphazero({simulations}sims)"
        self.last_value = 0.0

    def select_move(self, board: HexBoard, last_move: int | None = None) -> int:
        search = Search(board.copy(), self.cfg, self.rng)
        # Batch every leaf the search asks for; on a CPU a batch of one is
        # wasteful, so drain as many as the tree will hand over per round trip.
        while True:
            leaf = search.next_leaf()
            if leaf is None:
                break
            state, node = leaf
            priors, values = self.evaluator.evaluate([state])
            search.expand(node, priors[0], float(values[0]))
        self.last_value = search.root_value()
        moves, probs = search.policy_target(self.temperature)
        if self.temperature <= 1e-3:
            return int(moves[int(np.argmax(probs))])
        p = probs.astype(np.float64)
        p /= p.sum()
        return int(self.rng.choice(moves, p=p))


class PolicyOnlyAgent(Agent):
    """Network policy head with no search -- isolates what search contributes."""

    def __init__(
        self,
        evaluator: BatchEvaluator,
        temperature: float = 0.0,
        seed: int | None = None,
        name: str = "policy-only(no search)",
    ):
        self.evaluator = evaluator
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)
        self.name = name

    def select_move(self, board: HexBoard, last_move: int | None = None) -> int:
        priors, _ = self.evaluator.evaluate([board])
        p = priors[0]
        legal = board.legal_moves()
        canon = np.array([board.to_canonical_move(int(m)) for m in legal])
        vals = p[canon]
        if self.temperature <= 1e-3:
            return int(legal[int(np.argmax(vals))])
        w = np.power(vals.astype(np.float64), 1.0 / self.temperature)
        w /= w.sum()
        return int(self.rng.choice(legal, p=w))
