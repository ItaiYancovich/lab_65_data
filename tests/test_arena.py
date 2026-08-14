"""Tests for match balancing and the Elo fit."""

from __future__ import annotations

import sys
import pathlib
import math

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from alphazero_hex.arena import fit_elo, balanced_openings, play_match  # noqa: E402
from alphazero_hex.agents.base import RandomAgent  # noqa: E402
from alphazero_hex.agents.rule_based import RuleBasedAgent  # noqa: E402


def test_elo_recovers_known_ratings():
    """Simulate a field with known Elo and check the fit recovers the gaps."""
    true = {"A": 0.0, "B": 200.0, "C": 400.0, "D": 700.0}
    rng = np.random.default_rng(0)
    names = list(true)
    results = []
    n_games = 4000
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            p = 1.0 / (1.0 + 10 ** ((true[b] - true[a]) / 400.0))
            aw = int(rng.binomial(n_games, p))
            results.append((a, b, aw, n_games - aw))
    elo = fit_elo(results, anchor="A", anchor_rating=0.0)
    for name in names:
        err = abs(elo[name] - true[name])
        assert err < 25, f"{name}: fitted {elo[name]:.0f} vs true {true[name]:.0f}"
    print("ok elo recovers ratings:", {k: round(v) for k, v in elo.items()})


def test_elo_handles_perfect_scores():
    """A 100% score must give a large but finite rating, not an infinity."""
    results = [("weak", "strong", 0, 40), ("weak", "mid", 5, 35), ("mid", "strong", 8, 32)]
    elo = fit_elo(results, anchor="weak", anchor_rating=0.0)
    assert all(math.isfinite(v) for v in elo.values()), elo
    assert elo["strong"] > elo["mid"] > elo["weak"]
    print("ok elo finite under a perfect score:", {k: round(v) for k, v in elo.items()})


def test_openings_are_distinct_and_legal():
    ops = balanced_openings(11, 10, np.random.default_rng(1))
    assert len(ops) == 10 and len(set(ops)) == 10
    assert all(0 <= o < 121 for o in ops)
    print("ok balanced openings distinct")


def test_match_is_colour_balanced():
    a = RuleBasedAgent(seed=1, noise=0.3)
    b = RandomAgent(seed=2)
    res = play_match(a, b, board_size=7, games=8, rng=np.random.default_rng(3))
    assert res["games"] == 8
    assert res["a_wins"] + res["b_wins"] == 8
    # Each agent played Black in exactly half the games.
    assert res["a_win_rate"] == res["a_wins"] / 8
    print(f"ok balanced match (rule-based {res['a_wins']}/8 vs random, "
          f"black won {res['black_win_rate'] * 100:.0f}% of games)")


if __name__ == "__main__":
    test_elo_recovers_known_ratings()
    test_elo_handles_perfect_scores()
    test_openings_are_distinct_and_legal()
    test_match_is_colour_balanced()
    print("\nall arena tests passed")
