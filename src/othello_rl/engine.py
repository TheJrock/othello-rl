"""Othello rules on bitboards for board sizes n in {4, 6, 8} (spec §4.2).

Conventions (web/src/engine.ts mirrors these exactly; the parity tests depend on it):
- Square index i = r * n + c (row r from the top, column c from the left) = bit i of a bitboard.
- Actions 0 .. n*n-1 place a disc on that square; action n*n is pass. Pass is legal only when no
  square is legal and the game is not over.
- A State is always seen from the side to move: `me` = side to move, `opp` = the other side.
  `black_to_move` is bookkeeping for display and colour-based stats; the rules never read it.
- Board strings (from_string / to_string, data files, fixtures): n rows of n characters joined by
  "\\n", with 'X' = side to move, 'O' = other side, '.' = empty. from_string ignores whitespace.
- D4 symmetry k in 0..7: k < 4 is np.rot90(x, k); k >= 4 is np.rot90(np.fliplr(x), k - 4), both
  applied to the last two axes.

Keep this module free of torch: the solvers and evaluation run it in tight loops and worker processes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIZES = (4, 6, 8)


def action_size(n: int) -> int:
    """n² squares plus pass."""
    return n * n + 1


def pass_action(n: int) -> int:
    return n * n


def square(r: int, c: int, n: int) -> int:
    return r * n + c


@dataclass(frozen=True, slots=True)
class State:
    me: int  # bitboard of the side to move
    opp: int  # bitboard of the other side
    n: int
    black_to_move: bool = True


def initial(n: int) -> State:
    """Standard start. With c = n // 2: White on (c-1, c-1) and (c, c), Black on (c-1, c) and (c, c-1).
    Black moves first, so `me` is Black's bitboard."""
    raise NotImplementedError("engine.initial")


def from_string(text: str, black_to_move: bool = True) -> State:
    """Parse a board string (see module docstring). n is inferred from the number of squares."""
    raise NotImplementedError("engine.from_string")


def to_string(s: State) -> str:
    """Inverse of from_string: n lines of n characters, no trailing newline."""
    raise NotImplementedError("engine.to_string")


def legal_moves(s: State) -> int:
    """Bitboard of squares where the side to move may play.

    Shift in all 8 directions with edge masks so discs never wrap from the end of one row to the
    start of the next (tests/test_engine.py::test_no_wraparound), and mask every result to the
    n*n board bits."""
    raise NotImplementedError("engine.legal_moves")


def flips(s: State, a: int) -> int:
    """Bitboard of opponent discs flipped by playing square `a` (0 if `a` is not a legal square)."""
    raise NotImplementedError("engine.flips")


def legal_mask(s: State) -> np.ndarray:
    """Bool array of shape (n*n + 1,). The pass entry is True only when no square is legal and the
    game is not over. All False for a terminal state."""
    raise NotImplementedError("engine.legal_mask")


def is_terminal(s: State) -> bool:
    """True when neither side has a legal square."""
    raise NotImplementedError("engine.is_terminal")


def outcome(s: State) -> int:
    """For a terminal state: +1 / 0 / −1 for the side to move, by disc count."""
    raise NotImplementedError("engine.outcome")


def step(s: State, a: int) -> tuple[State, float, bool]:
    """Apply action `a` (a legal square or pass) and hand the move to the other side.

    Returns (s2, r, done):
      s2    from the next mover's perspective (me/opp swapped, black_to_move flipped). It may be a
            forced-pass state, i.e. the next mover has no square but the game is not over.
      done  is_terminal(s2)
      r     for the player who just moved: −outcome(s2) if done, else 0.0
    """
    raise NotImplementedError("engine.step")


def to_planes(s: State) -> np.ndarray:
    """float32 array (3, n, n): [side-to-move discs, other side's discs, legal squares]."""
    raise NotImplementedError("engine.to_planes")


def transform_planes(x: np.ndarray, k: int) -> np.ndarray:
    """Apply D4 symmetry k to the last two axes of x (any leading shape)."""
    raise NotImplementedError("engine.transform_planes")


def transform_action(a: int, n: int, k: int) -> int:
    """Where square `a` lands under symmetry k. Pass maps to pass."""
    raise NotImplementedError("engine.transform_action")


def transform_mask(mask: np.ndarray, n: int, k: int) -> np.ndarray:
    """Apply symmetry k to a (…, n*n + 1) legal mask; the pass entry is unchanged."""
    raise NotImplementedError("engine.transform_mask")


def transform_state(s: State, k: int) -> State:
    """The same position under symmetry k (the rules are D4-invariant)."""
    raise NotImplementedError("engine.transform_state")


def inverse_symmetry(k: int) -> int:
    """The k' that undoes symmetry k. Rotations invert to (4 - k) % 4; reflections (k >= 4) are their own inverse."""
    raise NotImplementedError("engine.inverse_symmetry")


def perft(s: State, depth: int) -> int:
    """Number of leaf nodes `depth` plies below s, counting a forced pass as a ply. Terminal
    states count as leaves. Used to validate move generation against published 8×8 values."""
    raise NotImplementedError("engine.perft")
