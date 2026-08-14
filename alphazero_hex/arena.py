"""Head-to-head evaluation: balanced matches, round-robin, and Elo.

Hex has a large first-player advantage on 11x11, so any honest comparison has
to control for colour.  Every match here plays each opening position twice --
once with each agent as Black -- so the advantage cancels exactly.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .hex_game import BLACK, WHITE, HexBoard
from .agents.base import Agent


@dataclass
class GameRecord:
    black: str
    white: str
    winner: str
    plies: int
    opening: int | None
    moves: list[int] = field(default_factory=list)
    seconds: float = 0.0


def play_game(
    black: Agent,
    white: Agent,
    board_size: int = 11,
    opening: int | None = None,
    max_plies: int | None = None,
    record_moves: bool = False,
) -> GameRecord:
    """Play one game. ``opening`` forces Black's first move if given."""
    board = HexBoard(board_size)
    black.reset()
    white.reset()
    moves: list[int] = []
    last: int | None = None
    t0 = time.time()
    if opening is not None:
        board.play(opening)
        moves.append(opening)
        last = opening
    limit = max_plies or board.ncells
    while not board.is_terminal() and board.move_count < limit:
        agent = black if board.to_move == BLACK else white
        move = agent.select_move(board, last)
        if not board.is_legal(move):
            raise ValueError(f"{agent.name} played illegal move {move}")
        board.play(move)
        moves.append(move)
        last = move
    winner = black.name if board.winner == BLACK else white.name
    return GameRecord(
        black=black.name,
        white=white.name,
        winner=winner,
        plies=board.move_count,
        opening=opening,
        moves=moves if record_moves else [],
        seconds=time.time() - t0,
    )


def balanced_openings(board_size: int, count: int, rng: np.random.Generator) -> list[int | None]:
    """A spread of forced first moves, to stop matches repeating one game.

    Deterministic agents would otherwise replay the same game every time, so
    the opening is what supplies variety; sampling across the board (rather
    than only strong cells) also probes positions the agents did not choose.
    """
    if count <= 1:
        return [None]
    n = board_size
    cells = []
    # Prefer central/short-diagonal cells, which are the sane Hex openings,
    # then fall back to anything for extra variety.
    ranked = sorted(
        range(n * n),
        key=lambda m: (abs(m // n - (n - 1) / 2) + abs(m % n - (n - 1) / 2)),
    )
    cells = ranked[: max(count, 1)]
    rng.shuffle(cells)
    return cells[:count]


def play_match(
    agent_a: Agent,
    agent_b: Agent,
    board_size: int = 11,
    games: int = 20,
    rng: np.random.Generator | None = None,
    openings: list[int | None] | None = None,
    verbose: bool = False,
) -> dict:
    """Play ``games`` games, colours alternating and openings mirrored."""
    rng = rng or np.random.default_rng(0)
    pairs = max(1, games // 2)
    if openings is None:
        openings = balanced_openings(board_size, pairs, rng)
    records: list[GameRecord] = []
    a_wins = 0
    a_wins_as_black = 0
    a_games_as_black = 0
    for i in range(pairs):
        op = openings[i % len(openings)]
        # Same opening played from both sides: the colour edge cancels out.
        r1 = play_game(agent_a, agent_b, board_size, opening=op)
        records.append(r1)
        a_games_as_black += 1
        if r1.winner == agent_a.name:
            a_wins += 1
            a_wins_as_black += 1
        r2 = play_game(agent_b, agent_a, board_size, opening=op)
        records.append(r2)
        if r2.winner == agent_a.name:
            a_wins += 1
        if verbose:
            print(f"    opening {op}: {r1.winner} (A black), {r2.winner} (B black)", flush=True)
    total = len(records)
    return {
        "a": agent_a.name,
        "b": agent_b.name,
        "games": total,
        "a_wins": a_wins,
        "b_wins": total - a_wins,
        "a_win_rate": a_wins / total,
        "a_win_rate_as_black": a_wins_as_black / max(a_games_as_black, 1),
        "black_win_rate": sum(1 for r in records if r.winner == r.black) / total,
        "avg_plies": float(np.mean([r.plies for r in records])),
        "avg_seconds": float(np.mean([r.seconds for r in records])),
        "records": records,
    }


# ------------------------------------------------------------------- Elo
def fit_elo(
    results: list[tuple[str, str, int, int]],
    anchor: str | None = None,
    anchor_rating: float = 0.0,
    iterations: int = 3000,
    prior_games: float = 1.0,
) -> dict[str, float]:
    """Maximum-likelihood Bradley-Terry ratings on the Elo scale.

    ``results`` is a list of ``(player_a, player_b, a_wins, b_wins)``.  A small
    symmetric prior (``prior_games`` virtual drawn games against a phantom
    average opponent) keeps ratings finite when someone wins or loses every
    game -- which happens a lot when the field spans random to AlphaZero.
    """
    players = sorted({p for r in results for p in (r[0], r[1])})
    index = {p: i for i, p in enumerate(players)}
    rating = np.zeros(len(players))  # natural (logistic) scale

    wins = np.zeros((len(players), len(players)))
    for a, b, aw, bw in results:
        ia, ib = index[a], index[b]
        wins[ia, ib] += aw
        wins[ib, ia] += bw

    games = wins + wins.T
    total_wins = wins.sum(axis=1) + prior_games * 0.5
    for _ in range(iterations):
        # Minorization-maximization update for Bradley-Terry.
        gamma = np.exp(rating)
        denom = np.zeros(len(players))
        for i in range(len(players)):
            mask = games[i] > 0
            if mask.any():
                denom[i] = (games[i][mask] / (gamma[i] + gamma[mask])).sum()
        denom += prior_games / (gamma + np.exp(np.mean(rating)))
        new_gamma = total_wins / np.maximum(denom, 1e-12)
        new_rating = np.log(np.maximum(new_gamma, 1e-12))
        new_rating -= new_rating.mean()
        if np.abs(new_rating - rating).max() < 1e-10:
            rating = new_rating
            break
        rating = new_rating

    scale = 400.0 / math.log(10.0)
    elo = {p: float(rating[index[p]] * scale) for p in players}
    if anchor is not None and anchor in elo:
        shift = anchor_rating - elo[anchor]
        elo = {p: v + shift for p, v in elo.items()}
    return elo


def elo_confidence(win_rate: float, games: int) -> float:
    """Rough +/- 1 sigma Elo uncertainty for a win rate over ``games`` games."""
    if games == 0:
        return float("inf")
    p = min(max(win_rate, 1e-6), 1 - 1e-6)
    se = math.sqrt(p * (1 - p) / games)
    # d(Elo)/dp at p, on the logistic scale
    return 400.0 / math.log(10.0) * se / (p * (1 - p))


def round_robin(
    agents: list[Agent],
    board_size: int = 11,
    games_per_pair: int = 20,
    rng: np.random.Generator | None = None,
    verbose: bool = True,
) -> dict:
    """Every agent against every other, colours balanced."""
    rng = rng or np.random.default_rng(0)
    matches = []
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            t0 = time.time()
            res = play_match(agents[i], agents[j], board_size, games_per_pair, rng)
            res.pop("records")
            res["seconds"] = time.time() - t0
            matches.append(res)
            if verbose:
                print(f"  {res['a']:<28} vs {res['b']:<28} "
                      f"{res['a_wins']:>3}-{res['b_wins']:<3} "
                      f"({res['a_win_rate'] * 100:5.1f}%)  [{res['seconds']:.0f}s]", flush=True)
    elo = fit_elo([(m["a"], m["b"], m["a_wins"], m["b_wins"]) for m in matches])
    return {"matches": matches, "elo": elo}
