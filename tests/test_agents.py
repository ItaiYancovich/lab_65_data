"""Behavioural tests for the hand-written opponents."""

from __future__ import annotations

import sys
import pathlib
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from alphazero_hex.hex_game import BLACK, WHITE, HexBoard  # noqa: E402
from alphazero_hex import heuristics as H  # noqa: E402
from alphazero_hex.agents.base import RandomAgent  # noqa: E402
from alphazero_hex.agents.rule_based import RuleBasedAgent  # noqa: E402
from alphazero_hex.agents.minimax import MinimaxAgent  # noqa: E402
from alphazero_hex.agents.mcts_rollout import RolloutMCTSAgent, random_rollout  # noqa: E402


def slow_immediate_wins(board, player):
    out = []
    for m in board.legal_moves():
        probe = board.copy()
        probe.to_move = player
        probe.play(int(m))
        if probe.winner == player:
            out.append(int(m))
    return out


def test_fast_win_detection_matches_bruteforce():
    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(60):
        n = 7
        b = HexBoard(n)
        for m in rng.permutation(n * n):
            if b.is_terminal():
                break
            for player in (BLACK, WHITE):
                assert sorted(H.immediate_wins(b, player)) == sorted(
                    slow_immediate_wins(b, player)
                ), f"mismatch for player {player}\n{b}"
                checked += 1
            b.play(int(m))
    print(f"ok fast win detection matches brute force ({checked} positions)")


def test_bridge_table():
    n = 7
    bt = H.bridge_table(n)
    nei = H.neighbour_table(n)
    centre = 3 * n + 3
    assert len(bt[centre]) == 6
    for partner, ca, cb in bt[centre]:
        # Carriers must be the two cells adjacent to *both* endpoints.
        shared = set(nei[centre]) & set(nei[partner])
        assert shared == {ca, cb}, (centre, partner, shared, (ca, cb))
        assert partner not in nei[centre]  # a bridge is not adjacency
    print("ok bridge geometry")


def test_bridge_response():
    n = 7
    b = HexBoard(n)
    bt = H.bridge_table(n)
    a = 3 * n + 3
    partner, ca, cb = bt[a][0]
    b.board[a] = BLACK
    b.board[partner] = BLACK
    b.board[ca] = WHITE  # White intrudes
    b.to_move = BLACK
    assert H.broken_bridge_response(b, BLACK, ca) == cb
    assert H.broken_bridge_response(b, BLACK, cb) is None  # cb is empty, not White
    print("ok bridge intrusion response")


def test_two_distance_sanity():
    n = 7
    b = HexBoard(n)
    # Distance from the top edge grows one per row on an empty board.  Cells in
    # the last column have only one neighbour in the row above, so they cost an
    # extra stone, and that penalty spreads diagonally into the top-right
    # corner -- so r+1 is the cheapest cell in row r, not the value everywhere.
    d_top = H.two_distance(b.array().reshape(-1), n, BLACK, "top").reshape(n, n)
    for r in range(n):
        assert d_top[r].min() == r + 1, (r, d_top[r])
        assert d_top[r, 0] == r + 1, (r, d_top[r])
        assert (d_top[r] >= r + 1).all(), (r, d_top[r])
    assert (np.diff(d_top[:, 0]) == 1).all()
    # ...so connecting the two edges costs exactly n stones.
    pot, cost = H.connection_cost(b.array().reshape(-1), n, BLACK)
    assert pot == n, (pot, n)
    # Two-distance being flat, it is `move_scores` (centrality tie-break) that
    # must prefer the middle of the board.
    best = int(np.argmax(H.move_scores(b, BLACK)))
    r, c = divmod(best, n)
    assert 1 <= r <= n - 2 and 1 <= c <= n - 2, (r, c)
    # A wall of White across the middle must make Black's connection impossible.
    b2 = HexBoard(n)
    for c in range(n):
        b2.board[3 * n + c] = WHITE
    pot, _ = H.connection_cost(b2.array().reshape(-1), n, BLACK)
    assert pot >= H.INF, pot
    print(f"ok two-distance (empty-board best cell = {divmod(best, n)}, blocked = INF)")


def test_rollout_always_decides():
    rng = np.random.default_rng(0)
    b = HexBoard(9)
    for _ in range(200):
        v = random_rollout(b, rng)
        assert v in (1.0, -1.0)
    print("ok rollouts always produce a winner")


def play(a1, a2, n=7, seed=0, opening=None):
    """Return the winning agent; a1 is BLACK."""
    b = HexBoard(n)
    a1.reset(); a2.reset()
    last = None
    if opening is not None:
        b.play(opening); last = opening
    while not b.is_terminal():
        agent = a1 if b.to_move == BLACK else a2
        m = agent.select_move(b, last)
        assert b.is_legal(m), f"{agent.name} played illegal move {m}"
        b.play(m); last = m
    return a1 if b.winner == BLACK else a2


def test_rule_based_crushes_random():
    wins = 0
    games = 12
    for i in range(games):
        rb = RuleBasedAgent(seed=i, noise=0.01)
        rnd = RandomAgent(seed=100 + i)
        # Alternate colours so neither side gets the first-move advantage.
        winner = play(rb, rnd, n=7, seed=i) if i % 2 == 0 else play(rnd, rb, n=7, seed=i)
        wins += winner is rb
    assert wins >= games - 1, f"rule-based only won {wins}/{games} vs random"
    print(f"ok rule-based beats random {wins}/{games}")


def test_minimax_beats_random_and_is_legal():
    wins = 0
    games = 6
    for i in range(games):
        mm = MinimaxAgent(time_budget=0.3, beam=6, seed=i)
        rnd = RandomAgent(seed=200 + i)
        winner = play(mm, rnd, n=7) if i % 2 == 0 else play(rnd, mm, n=7)
        wins += winner is mm
    assert wins == games, f"minimax only won {wins}/{games} vs random"
    print(f"ok minimax beats random {wins}/{games}")


def test_rollout_mcts_beats_random():
    wins = 0
    games = 6
    for i in range(games):
        mc = RolloutMCTSAgent(simulations=600, seed=i)
        rnd = RandomAgent(seed=300 + i)
        winner = play(mc, rnd, n=7) if i % 2 == 0 else play(rnd, mc, n=7)
        wins += winner is mc
    assert wins == games, f"rollout MCTS only won {wins}/{games} vs random"
    print(f"ok rollout-MCTS beats random {wins}/{games}")


def test_agents_never_play_illegal_on_11x11():
    agents = [
        RandomAgent(seed=1),
        RuleBasedAgent(seed=2, noise=0.01),
        MinimaxAgent(time_budget=0.2, beam=5, seed=3),
        RolloutMCTSAgent(simulations=200, seed=4),
    ]
    for i in range(len(agents)):
        a, b = agents[i], agents[(i + 1) % len(agents)]
        t = time.time()
        play(a, b, n=11)
        print(f"    {a.name} vs {b.name}: legal game in {time.time() - t:.1f}s")
    print("ok all agents play legal 11x11 games")


if __name__ == "__main__":
    test_fast_win_detection_matches_bruteforce()
    test_bridge_table()
    test_bridge_response()
    test_two_distance_sanity()
    test_rollout_always_decides()
    test_rule_based_crushes_random()
    test_minimax_beats_random_and_is_legal()
    test_rollout_mcts_beats_random()
    test_agents_never_play_illegal_on_11x11()
    print("\nall agent tests passed")
