"""Search tests: MCTS must see immediate wins and immediate threats."""

from __future__ import annotations

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from alphazero_hex.hex_game import BLACK, WHITE, HexBoard  # noqa: E402
from alphazero_hex.mcts import MCTSConfig, Search  # noqa: E402
from alphazero_hex.selfplay import Example, augment  # noqa: E402


class BlindEvaluator:
    """Uniform priors, value 0 -- all the strength must come from the tree."""

    def evaluate(self, states):
        n = states[0].n
        p = np.full((len(states), n * n), 1.0 / (n * n), dtype=np.float32)
        return p, np.zeros(len(states), dtype=np.float32)


def run_search(state, sims=800, seed=0):
    cfg = MCTSConfig(simulations=sims, add_noise=False)
    rng = np.random.default_rng(seed)
    s = Search(state, cfg, rng)
    ev = BlindEvaluator()
    while True:
        leaf = s.next_leaf()
        if leaf is None:
            break
        st, node = leaf
        pr, v = ev.evaluate([st])
        s.expand(node, pr[0], float(v[0]))
    moves, N = s.root_visit_distribution()
    return int(moves[int(np.argmax(N))]), s


def test_finds_mate_in_one():
    n = 5
    b = HexBoard(n)
    # BLACK holds column 2 in rows 0..3, so any row-4 cell adjacent to (3,2)
    # wins at once -- both (4,2) and (4,1), since (r+1,c-1) is a hex neighbour.
    for r in range(4):
        b.board[r * n + 2] = BLACK
    b = _rebuild(b, n)
    assert b.to_move == BLACK and not b.is_terminal()
    winning = set()
    for m in b.legal_moves():
        probe = b.copy()
        probe.play(int(m))
        if probe.winner == BLACK:
            winning.add(int(m))
    assert winning == {4 * n + 1, 4 * n + 2}, winning
    best, _ = run_search(b, sims=400)
    assert best in winning, f"expected an immediate win, got {divmod(best, n)}"
    print(f"ok finds mate in one (played {divmod(best, n)})")


def test_blocks_mate_in_one():
    n = 5
    b = HexBoard(n)
    # WHITE owns row 2 columns 0..3 and so threatens to reach the right edge.
    # Both (2,4) and (1,4) touch that chain, so pre-place a BLACK stone on
    # (1,4) to leave exactly one winning cell -- the one BLACK must occupy.
    for c in range(4):
        b.board[2 * n + c] = WHITE
    b.board[1 * n + 4] = BLACK
    b = _rebuild(b, n, to_move=BLACK)
    threats = set()
    for m in b.legal_moves():
        probe = b.copy()
        probe.to_move = WHITE
        probe.play(int(m))
        if probe.winner == WHITE:
            threats.add(int(m))
    assert threats == {2 * n + 4}, threats
    best, _ = run_search(b, sims=1500)
    assert best == 2 * n + 4, f"expected block at (2,4), got {divmod(best, n)}"
    print("ok blocks mate in one")


def _rebuild(b: HexBoard, n: int, to_move=BLACK) -> HexBoard:
    """Recompute the union-find from the raw stones (test helper)."""
    fresh = HexBoard(n)
    fresh.board = bytearray(b.board)
    for i in range(n * n):
        p = fresh.board[i]
        if p == 0:
            continue
        r, c = divmod(i, n)
        for nb in fresh._nei[i]:
            if fresh.board[nb] == p:
                fresh._union(i, nb)
        if p == BLACK:
            if r == 0:
                fresh._union(i, fresh.TOP)
            if r == n - 1:
                fresh._union(i, fresh.BOTTOM)
        else:
            if c == 0:
                fresh._union(i, fresh.LEFT)
            if c == n - 1:
                fresh._union(i, fresh.RIGHT)
    if fresh._find(fresh.TOP) == fresh._find(fresh.BOTTOM):
        fresh.winner = BLACK
    elif fresh._find(fresh.LEFT) == fresh._find(fresh.RIGHT):
        fresh.winner = WHITE
    fresh.to_move = to_move
    fresh.move_count = sum(1 for x in fresh.board if x)
    return fresh


def test_tree_reuse_preserves_stats():
    n = 5
    b = HexBoard(n)
    b.play(2 * n + 2)
    _, s = run_search(b, sims=300)
    move = int(s.moves[0][int(np.argmax(s.N[0]))])
    sub = s.child_search(move)
    assert sub is not None
    assert sub.root_state.board[move] != 0
    assert sub.sims_done > 0 and sub.sims_done <= 300
    print(f"ok tree reuse (inherited {sub.sims_done} sims)")


def test_augmentation_is_consistent():
    n = 5
    ex = Example(np.zeros((n, n), np.uint8), np.zeros(n * n, np.float32), 1.0, BLACK)
    ex.canon_board[0, 1] = 1
    ex.pi[0 * n + 1] = 0.7
    ex.pi[3 * n + 4] = 0.3
    _, rot = augment(ex)
    assert rot.canon_board[n - 1, n - 2] == 1
    assert abs(rot.pi[(n - 1 - 0) * n + (n - 1 - 1)] - 0.7) < 1e-6
    assert abs(rot.pi[(n - 1 - 3) * n + (n - 1 - 4)] - 0.3) < 1e-6
    assert abs(rot.pi.sum() - 1.0) < 1e-6 and rot.z == 1.0
    print("ok augmentation")


if __name__ == "__main__":
    test_finds_mate_in_one()
    test_blocks_mate_in_one()
    test_tree_reuse_preserves_stats()
    test_augmentation_is_consistent()
    print("\nall search tests passed")
