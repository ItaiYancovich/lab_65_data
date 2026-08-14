#!/usr/bin/env python3
"""Play Hex against the trained agent, or watch two agents play.

    python3 play_hex.py --ckpt runs/az_hex/final.pt              # you are Black
    python3 play_hex.py --white human --black az                 # you are White
    python3 play_hex.py --black az --white rule --show           # watch a game

Moves are entered in the usual Hex notation: a column letter then a row
number, e.g. ``f6``.
"""

from __future__ import annotations

import argparse

import numpy as np

from alphazero_hex.agents.base import Agent
from alphazero_hex.hex_game import BLACK, WHITE, HexBoard, move_to_str, str_to_move
from alphazero_hex.registry import build_agent, spec


class HumanAgent(Agent):
    name = "human"

    def select_move(self, board: HexBoard, last_move: int | None = None) -> int:
        while True:
            try:
                text = input(f"your move ({'Black: top-bottom' if board.to_move == BLACK else 'White: left-right'}) > ")
            except EOFError:
                raise SystemExit("\nno input; exiting")
            try:
                move = str_to_move(text, board.n)
            except (ValueError, IndexError):
                print("  could not parse that; use e.g. f6")
                continue
            if not board.is_legal(move):
                print("  that cell is not available")
                continue
            return move


def make(kind: str, ckpt: str, sims: int, board_size: int, seed: int) -> Agent:
    if kind == "human":
        return HumanAgent()
    table = {
        "az": spec("az", f"alphazero({sims})", ckpt=ckpt, simulations=sims),
        "policy": spec("policy", "policy-only", ckpt=ckpt),
        "rule": spec("rule", "rule-based", noise=0.05),
        "minimax": spec("minimax", "alpha-beta", time_budget=1.0, beam=8),
        "rollout": spec("rollout", "mcts-rollout", simulations=3000),
        "random": spec("random", "random"),
    }
    if kind not in table:
        raise SystemExit(f"unknown agent {kind!r}; pick from {sorted(table) + ['human']}")
    return build_agent(table[kind], seed, board_size)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="runs/az_hex/final.pt")
    ap.add_argument("--black", default="human")
    ap.add_argument("--white", default="az")
    ap.add_argument("--board-size", type=int, default=11)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", action="store_true", help="print the board every ply")
    args = ap.parse_args()

    black = make(args.black, args.ckpt, args.sims, args.board_size, args.seed)
    white = make(args.white, args.ckpt, args.sims, args.board_size, args.seed + 1)
    board = HexBoard(args.board_size)
    human_playing = "human" in (args.black, args.white)
    last = None
    print(f"Black (B, top<->bottom): {black.name}\nWhite (W, left<->right): {white.name}\n")
    while not board.is_terminal():
        if human_playing or args.show:
            print(board)
        agent = black if board.to_move == BLACK else white
        move = agent.select_move(board, last)
        board.play(move)
        last = move
        print(f"{'Black' if board.to_move == WHITE else 'White'} "
              f"({agent.name}) plays {move_to_str(move, board.n)}")
    print(board)
    print(f"\n{'Black' if board.winner == BLACK else 'White'} wins "
          f"({(black if board.winner == BLACK else white).name}) in {board.move_count} moves")


if __name__ == "__main__":
    main()
