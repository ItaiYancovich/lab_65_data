"""Correctness tests for the Hex engine, encoding and search."""

from __future__ import annotations

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from alphazero_hex.hex_game import (  # noqa: E402
    BLACK, WHITE, HexBoard, neighbour_table, winner_of_full_board, rotate180_move,
)


def test_neighbours():
    n = 5
    nei = neighbour_table(n)
    assert set(nei[0]) == {1, n}  # corner (0,0): (0,1) and (1,0); (1,-1) off board
    centre = 2 * n + 2
    assert len(nei[centre]) == 6
    assert set(nei[centre]) == {centre - n, centre - n + 1, centre - 1,
                                centre + 1, centre + n - 1, centre + n}
    print("ok neighbours")


def test_black_vertical_win():
    n = 5
    b = HexBoard(n)
    # BLACK plays straight down column 2; WHITE plays harmlessly in column 0/4.
    black = [r * n + 2 for r in range(n)]
    white = [0, 4, n, n + 4, 2 * n]
    for i in range(n):
        b.play(black[i])
        if b.is_terminal():
            break
        b.play(white[i])
    assert b.winner == BLACK, b
    assert b.result_for(BLACK) == 1.0 and b.result_for(WHITE) == -1.0
    print("ok black vertical win")


def test_white_horizontal_win():
    n = 5
    b = HexBoard(n)
    white = [2 * n + c for c in range(n)]
    black = [0, 1, 2, 3, 4]
    for i in range(n):
        b.play(black[i])
        assert not b.is_terminal() or i == 4
        if b.is_terminal():
            break
        b.play(white[i])
    # Black filling the whole top row is not a win (it needs top->bottom).
    b2 = HexBoard(n)
    for i in range(n):
        b2.play(4 * n + i if i < 4 else 4 * n + 4)  # black across the bottom row
        if b2.is_terminal():
            break
        b2.play(white[i])
    assert b2.winner == WHITE, b2
    print("ok white horizontal win")


def test_diagonal_is_not_a_connection():
    """(r,c) and (r+1,c+1) are NOT adjacent on a hex rhombus."""
    n = 5
    nei = neighbour_table(n)
    assert (1 * n + 1) not in nei[0]
    print("ok diagonal not adjacent")


def test_no_draws_random_fill():
    rng = np.random.default_rng(0)
    for trial in range(200):
        n = 7
        b = HexBoard(n)
        order = rng.permutation(n * n)
        for m in order:
            b.play(int(m))
            if b.is_terminal():
                break
        assert b.winner in (BLACK, WHITE)
        # Cross-check the incremental union-find against an independent flood
        # fill on the same position with the remaining cells given to the loser.
        arr = b.array().reshape(-1).copy()
        empties = np.flatnonzero(arr == 0)
        arr[empties] = WHITE if b.winner == BLACK else BLACK
        assert winner_of_full_board(arr.reshape(n, n)) == b.winner
    print("ok no draws / union-find matches flood fill (200 random games)")


def test_canonical_encoding_roundtrip():
    rng = np.random.default_rng(1)
    n = 6
    b = HexBoard(n)
    for _ in range(9):
        legal = b.legal_moves()
        b.play(int(rng.choice(legal)))
    assert b.to_move == WHITE
    planes = b.canonical_planes()
    # Under the canonical view the side to move (White) occupies plane 0 and
    # the board is transposed, so plane 0 must equal the transpose of White's.
    assert np.array_equal(planes[0], (b.array() == WHITE).T.astype(np.float32))
    assert np.array_equal(planes[1], (b.array() == BLACK).T.astype(np.float32))
    # Move mapping must be an involution and land on the right cell.
    for m in b.legal_moves():
        m = int(m)
        cm = b.to_canonical_move(m)
        assert b.from_canonical_move(cm) == m
        assert planes[0].reshape(-1)[cm] == 0 and planes[1].reshape(-1)[cm] == 0
    print("ok canonical encoding")


def test_canonical_win_condition_consistency():
    """In the canonical frame the side to move always connects top<->bottom."""
    n = 5
    b = HexBoard(n)
    b.play(0)          # BLACK
    # Now WHITE to move. Give WHITE a horizontal chain in row 2 and check the
    # canonical view shows it as a *vertical* chain in column 2.
    b2 = b.copy()
    for c in range(3):
        b2.board[2 * n + c] = WHITE
    planes = b2.canonical_planes()
    own = planes[0]
    assert own[:, 2].sum() == 3 and own.sum() == 3
    print("ok canonical win orientation")


def test_rotation_symmetry():
    n = 11
    for m in range(n * n):
        assert rotate180_move(rotate180_move(m, n), n) == m
    # A rotated position has the same winner (the rotation maps top<->bottom
    # and left<->right, preserving both players' edge pairs).
    rng = np.random.default_rng(3)
    for _ in range(30):
        b = HexBoard(n)
        rb = HexBoard(n)
        for m in rng.permutation(n * n):
            b.play(int(m))
            rb.play(rotate180_move(int(m), n))
            if b.is_terminal():
                break
        assert b.winner == rb.winner
    print("ok 180-degree rotation symmetry")


if __name__ == "__main__":
    test_neighbours()
    test_black_vertical_win()
    test_white_horizontal_win()
    test_diagonal_is_not_a_connection()
    test_no_draws_random_fill()
    test_canonical_encoding_roundtrip()
    test_canonical_win_condition_consistency()
    test_rotation_symmetry()
    print("\nall engine tests passed")
