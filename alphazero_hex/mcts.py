"""PUCT Monte-Carlo tree search (AlphaZero style).

The tree is stored as flat per-node numpy arrays so that the selection step is
a handful of vector ops rather than a Python loop over children.

The search is written in a *coroutine* style: :meth:`Search.next_leaf` walks the
tree until it hits a position that needs a network evaluation and then returns,
handing control back to the caller.  The caller can therefore drive many
searches at once and evaluate all their leaves in a single batched forward pass
-- which is the difference between a useless and a usable amount of self-play
on a CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hex_game import HexBoard


@dataclass
class MCTSConfig:
    simulations: int = 128
    c_puct: float = 1.6
    dirichlet_alpha: float = 0.0  # 0 -> derived from board size
    dirichlet_eps: float = 0.25
    fpu_reduction: float = 0.25  # first-play-urgency: discount unvisited actions
    add_noise: bool = True

    def alpha_for(self, ncells: int) -> float:
        if self.dirichlet_alpha > 0:
            return self.dirichlet_alpha
        return max(0.03, 10.0 / ncells)


class Search:
    """One MCTS tree rooted at a position."""

    __slots__ = (
        "root_state", "cfg", "rng", "n_nodes",
        "moves", "P", "N", "W", "child", "term_value",
        "_pending_path", "_pending_state", "_pending_node",
        "sims_done", "root_noise_applied",
    )

    def __init__(self, root_state: HexBoard, cfg: MCTSConfig, rng: np.random.Generator):
        self.root_state = root_state
        self.cfg = cfg
        self.rng = rng
        self.n_nodes = 0
        self.moves: list[np.ndarray] = []
        self.P: list[np.ndarray] = []
        self.N: list[np.ndarray] = []
        self.W: list[np.ndarray] = []
        self.child: list[np.ndarray] = []
        self.term_value: list[float | None] = []
        self._pending_path: list[tuple[int, int]] = []
        self._pending_state: HexBoard | None = None
        self._pending_node: int = -1
        self.sims_done = 0
        self.root_noise_applied = False
        self._new_node()  # node 0 = root, unexpanded

    # ----------------------------------------------------------------- nodes
    def _new_node(self) -> int:
        idx = self.n_nodes
        self.n_nodes += 1
        self.moves.append(None)
        self.P.append(None)
        self.N.append(None)
        self.W.append(None)
        self.child.append(None)
        self.term_value.append(None)
        return idx

    def is_expanded(self, node: int) -> bool:
        return self.moves[node] is not None

    # ------------------------------------------------------------ traversal
    def next_leaf(self) -> tuple[HexBoard, int] | None:
        """Descend to a leaf needing evaluation.

        Returns ``(state, node)``; terminal leaves are backed up internally and
        the descent restarts.  Returns ``None`` once ``simulations`` sims are
        done.
        """
        while self.sims_done < self.cfg.simulations:
            state = self.root_state.copy()
            path: list[tuple[int, int]] = []
            node = 0
            while True:
                if self.term_value[node] is not None:
                    self._backup(path, self.term_value[node])
                    self.sims_done += 1
                    break
                if not self.is_expanded(node):
                    self._pending_path = path
                    self._pending_state = state
                    self._pending_node = node
                    return state, node
                a = self._select(node)
                move = int(self.moves[node][a])
                path.append((node, a))
                state.play(move)
                nxt = int(self.child[node][a])
                if nxt < 0:
                    nxt = self._new_node()
                    self.child[node][a] = nxt
                    if state.is_terminal():
                        # side to move at `state` has just lost
                        self.term_value[nxt] = -1.0
                node = nxt
        return None

    def _select(self, node: int) -> int:
        N = self.N[node]
        W = self.W[node]
        P = self.P[node]
        total = N.sum()
        sqrt_total = np.sqrt(total) if total > 0 else 1.0
        visited = N > 0
        # FPU: unvisited actions inherit the node's own value, reduced.
        parent_q = (W.sum() / total) if total > 0 else 0.0
        q = np.where(visited, W / np.maximum(N, 1.0), parent_q - self.cfg.fpu_reduction)
        u = self.cfg.c_puct * P * sqrt_total / (1.0 + N)
        return int(np.argmax(q + u))

    def expand(self, node: int, priors: np.ndarray, value: float) -> None:
        """Attach network output to the pending leaf and back the value up."""
        state = self._pending_state
        legal = state.legal_moves()
        if len(legal) == 0:
            # Unreachable in Hex (a full board always has a winner) but keeps
            # the search total rather than trusting the rules to be perfect.
            self.term_value[node] = 0.0
            self._backup(self._pending_path, 0.0)
            self.sims_done += 1
            self._pending_state = None
            return
        # `priors` are in the canonical frame; map them onto real board moves.
        canon_idx = np.array([state.to_canonical_move(int(m)) for m in legal], dtype=np.int64)
        p = priors[canon_idx].astype(np.float64)
        s = p.sum()
        p = p / s if s > 1e-12 else np.full(len(legal), 1.0 / max(len(legal), 1))

        self.moves[node] = legal
        self.P[node] = p.astype(np.float32)
        self.N[node] = np.zeros(len(legal), dtype=np.float32)
        self.W[node] = np.zeros(len(legal), dtype=np.float32)
        self.child[node] = np.full(len(legal), -1, dtype=np.int32)

        if node == 0 and self.cfg.add_noise and not self.root_noise_applied:
            self._apply_root_noise()

        self._backup(self._pending_path, value)
        self.sims_done += 1
        self._pending_state = None
        self._pending_node = -1

    def _apply_root_noise(self) -> None:
        k = len(self.P[0])
        if k <= 1:
            self.root_noise_applied = True
            return
        alpha = self.cfg.alpha_for(self.root_state.ncells)
        noise = self.rng.dirichlet(np.full(k, alpha)).astype(np.float32)
        eps = self.cfg.dirichlet_eps
        self.P[0] = (1 - eps) * self.P[0] + eps * noise
        self.root_noise_applied = True

    def _backup(self, path, value: float) -> None:
        # `value` is from the point of view of the side to move at the leaf.
        v = value
        for node, a in reversed(path):
            v = -v  # flip going up one ply
            self.N[node][a] += 1.0
            self.W[node][a] += v

    # -------------------------------------------------------------- readout
    def root_visit_distribution(self) -> tuple[np.ndarray, np.ndarray]:
        return self.moves[0], self.N[0]

    def root_value(self) -> float:
        N = self.N[0]
        total = N.sum()
        return float(self.W[0].sum() / total) if total > 0 else 0.0

    def policy_target(self, temperature: float) -> tuple[np.ndarray, np.ndarray]:
        """(moves, probabilities) from root visit counts."""
        moves, N = self.root_visit_distribution()
        if temperature <= 1e-3:
            probs = np.zeros_like(N)
            probs[int(np.argmax(N))] = 1.0
            return moves, probs
        counts = np.power(N, 1.0 / temperature)
        s = counts.sum()
        if s <= 0:
            return moves, np.full(len(moves), 1.0 / len(moves), dtype=np.float32)
        return moves, (counts / s).astype(np.float32)

    def child_search(self, move: int) -> "Search | None":
        """Reuse the subtree after playing ``move`` (tree reuse between plies)."""
        if not self.is_expanded(0):
            return None
        idx = np.flatnonzero(self.moves[0] == move)
        if len(idx) == 0:
            return None
        node = int(self.child[0][idx[0]])
        if node < 0:
            return None
        new_state = self.root_state.copy()
        new_state.play(move)
        sub = Search.__new__(Search)
        sub.root_state = new_state
        sub.cfg = self.cfg
        sub.rng = self.rng
        sub._pending_path = []
        sub._pending_state = None
        sub._pending_node = -1
        sub.sims_done = 0
        sub.root_noise_applied = False
        # Re-index the reachable subtree into fresh compact arrays.
        mapping = {node: 0}
        order = [node]
        sub.moves, sub.P, sub.N, sub.W, sub.child, sub.term_value = [], [], [], [], [], []
        i = 0
        while i < len(order):
            old = order[i]
            i += 1
            sub.moves.append(self.moves[old])
            sub.P.append(self.P[old])
            sub.N.append(self.N[old])
            sub.W.append(self.W[old])
            sub.term_value.append(self.term_value[old])
            if self.child[old] is None:
                sub.child.append(None)
                continue
            new_children = np.full(len(self.child[old]), -1, dtype=np.int32)
            for j, ch in enumerate(self.child[old]):
                ch = int(ch)
                if ch >= 0:
                    if ch not in mapping:
                        mapping[ch] = len(order)
                        order.append(ch)
                    new_children[j] = mapping[ch]
            sub.child.append(new_children)
        sub.n_nodes = len(sub.moves)
        if sub.is_expanded(0):
            # Inherited visits count towards this move's simulation budget.
            sub.sims_done = int(sub.N[0].sum())
            if sub.cfg.add_noise:
                sub.P[0] = sub.P[0].copy()
                sub._apply_root_noise()
        return sub
