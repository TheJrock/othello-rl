"""Engine correctness (spec §4.2). Board strings: X = side to move, O = other side, . = empty."""

from __future__ import annotations

import numpy as np
import pytest

from othello_rl import engine as E


def bits(*squares: int) -> int:
    return sum(1 << i for i in squares)


def random_positions(n: int, count: int, rng: np.random.Generator) -> list[E.State]:
    """Positions met along random games (terminal ones excluded)."""
    out: list[E.State] = []
    while len(out) < count:
        s = E.initial(n)
        while not E.is_terminal(s) and len(out) < count:
            out.append(s)
            s, _, _ = E.step(s, int(rng.choice(np.flatnonzero(E.legal_mask(s)))))
    return out


# --- start position and move generation ---------------------------------------------------------


@pytest.mark.parametrize("n", E.SIZES)
def test_initial_position(n):
    s = E.initial(n)
    assert s.black_to_move
    assert s.me.bit_count() == 2 and s.opp.bit_count() == 2 and s.me & s.opp == 0
    mask = E.legal_mask(s)
    assert mask.shape == (E.action_size(n),) and mask.dtype == bool
    assert mask[:-1].sum() == 4 and not mask[E.pass_action(n)]


def test_string_round_trip():
    board = "....\n.XO.\n.OX.\n...."
    s = E.from_string(
        """
        . . . .
        . X O .
        . O X .
        . . . .
    """,
        black_to_move=False,
    )
    assert (s.n, s.black_to_move) == (4, False)
    assert s.me == bits(5, 10) and s.opp == bits(6, 9)
    assert E.to_string(s) == board


# Published 8×8 Othello perft counts. Confirm against a published table (spec §4.2) before relying on deeper values.
PERFT_8X8 = {1: 4, 2: 12, 3: 56, 4: 244, 5: 1396, 6: 8200}


@pytest.mark.parametrize("depth,expected", sorted(PERFT_8X8.items()))
def test_perft_8x8(depth, expected):
    assert E.perft(E.initial(8), depth) == expected


def test_flips_in_all_eight_directions():
    s = E.from_string("""
        ........
        .X.X.X..
        ..OOO...
        .XO.OX..
        ..OOO...
        .X.X.X..
        ........
        ........
    """)
    a = E.square(3, 3, 8)
    expected = bits(
        *(E.square(r, c, 8) for r, c in [(2, 2), (2, 3), (2, 4), (3, 2), (3, 4), (4, 2), (4, 3), (4, 4)])
    )
    assert E.legal_mask(s)[a]
    assert E.flips(s, a) == expected
    s2, r, done = E.step(s, a)
    assert s2.opp == s.me | expected | (1 << a)  # the mover's discs, now seen from the other side
    assert s2.me == s.opp & ~expected
    # Those were all of O's discs: a wipeout ends the game at once, and the mover wins.
    assert s2.me == 0
    assert (r, done) == (1.0, True)


def test_no_wraparound():
    # A buggy shift lets O at (1,0) "connect" to X at (0,5) across the row boundary in both directions
    # (X to (1,1) going west, O to (0,4) going east). Correctly, neither side has a move.
    s = E.from_string("""
        .....X
        O.....
        ......
        ......
        ......
        ......
    """)
    assert E.legal_moves(s) == 0
    assert E.legal_moves(E.State(s.opp, s.me, 6, not s.black_to_move)) == 0
    assert E.is_terminal(s)


# --- passes and the end of the game -------------------------------------------------------------


def test_forced_pass():
    s = E.from_string("""
        OOOO
        OOOO
        OOOX
        OOO.
    """)
    mask = E.legal_mask(s)
    assert mask[E.pass_action(4)] and mask.sum() == 1
    assert not E.is_terminal(s)
    s2, r, done = E.step(s, E.pass_action(4))
    assert (s2.me, s2.opp, s2.black_to_move) == (s.opp, s.me, not s.black_to_move)
    assert (r, done) == (0.0, False)
    assert E.legal_mask(s2)[E.square(3, 3, 4)]


def test_winning_move_ends_game_with_mover_reward():
    s = E.from_string("""
        XXXX
        XXXX
        XXXO
        XXX.
    """)
    s2, r, done = E.step(s, E.square(3, 3, 4))
    assert done and E.is_terminal(s2)
    assert s2.me == 0 and s2.opp == (1 << 16) - 1
    assert r == 1.0 and E.outcome(s2) == -1
    assert not E.legal_mask(s2).any()


@pytest.mark.parametrize("n", E.SIZES)
def test_random_game_invariants(n, rng):
    full = (1 << (n * n)) - 1
    for _ in range(100 if n < 8 else 20):
        s = E.initial(n)
        while not E.is_terminal(s):
            mask = E.legal_mask(s)
            assert mask.any() and E.legal_moves(s) & ~full == 0
            a = int(rng.choice(np.flatnonzero(mask)))
            s2, r, done = E.step(s, a)
            assert s2.me & s2.opp == 0 and (s2.me | s2.opp) & ~full == 0
            assert s2.black_to_move != s.black_to_move
            discs_before, discs_after = (s.me | s.opp).bit_count(), (s2.me | s2.opp).bit_count()
            if a == E.pass_action(n):
                assert (s2.me, s2.opp) == (s.opp, s.me)
            else:
                assert discs_after == discs_before + 1
            assert done == E.is_terminal(s2)
            assert r == (-E.outcome(s2) if done else 0.0)
            s = s2


# --- symmetries ---------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [4, 6])
@pytest.mark.parametrize("k", range(8))
def test_symmetry_round_trip(n, k, rng):
    x = rng.random((3, n, n)).astype(np.float32)
    np.testing.assert_array_equal(E.transform_planes(E.transform_planes(x, k), E.inverse_symmetry(k)), x)


@pytest.mark.parametrize("n", [4, 6])
@pytest.mark.parametrize("k", range(8))
def test_action_transform_matches_planes(n, k):
    for a in range(n * n):
        plane = np.zeros((n, n), np.float32)
        plane.flat[a] = 1
        assert E.transform_planes(plane, k).flat[E.transform_action(a, n, k)] == 1
    assert E.transform_action(E.pass_action(n), n, k) == E.pass_action(n)


@pytest.mark.parametrize("k", range(8))
def test_rules_are_symmetric(k, rng):
    for s in random_positions(6, 50, rng):
        t = E.transform_state(s, k)
        np.testing.assert_array_equal(E.legal_mask(t), E.transform_mask(E.legal_mask(s), 6, k))
        np.testing.assert_array_equal(E.to_planes(t), E.transform_planes(E.to_planes(s), k))
