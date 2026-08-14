"""Common agent interface."""

from __future__ import annotations

import numpy as np

from ..hex_game import HexBoard


class Agent:
    """Anything that can pick a Hex move.

    ``select_move`` receives the position and the opponent's last move (some
    agents use pattern responses that depend on it) and returns a legal cell.
    """

    name: str = "agent"

    def reset(self) -> None:
        """Called at the start of every game."""

    def select_move(self, board: HexBoard, last_move: int | None = None) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.name}>"


class RandomAgent(Agent):
    """Uniform random legal move -- the floor of the rating scale."""

    def __init__(self, seed: int | None = None, name: str = "random"):
        self.rng = np.random.default_rng(seed)
        self.name = name

    def select_move(self, board: HexBoard, last_move: int | None = None) -> int:
        return int(self.rng.choice(board.legal_moves()))
