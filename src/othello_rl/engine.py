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


# --- direction tables ---------------------------------------------------------------------------
# Moving one square is a bit shift: +1 east, +n south, +n+1 south-east, and so on (negative = right
# shift). A shift with an east component would carry a disc in column n-1 into column 0 of the next
# row, so the source is masked to drop that column first; likewise column 0 for west. Vertical
# overflow needs no pre-mask: bits shifted past n*n are removed when the result is ANDed with a
# bitboard that lives on the board (opp, me, or the empty squares).


def _full(n: int) -> int:
    return (1 << (n * n)) - 1


def _directions(n: int) -> tuple[tuple[int, int], ...]:
    """(shift, source pre-mask) for the 8 directions on an n×n board."""
    full = _full(n)
    col0 = sum(1 << square(r, 0, n) for r in range(n))
    not_first_col = full & ~col0  # sources allowed to move west
    not_last_col = full & ~(col0 << (n - 1))  # sources allowed to move east
    return (
        (+1, not_last_col),  # east
        (-1, not_first_col),  # west
        (+n, full),  # south
        (-n, full),  # north
        (+n + 1, not_last_col),  # south-east
        (+n - 1, not_first_col),  # south-west
        (-n + 1, not_last_col),  # north-east
        (-n - 1, not_first_col),  # north-west
    )


_FULL = {n: _full(n) for n in SIZES}
_DIRS = {n: _directions(n) for n in SIZES}


def _shift(x: int, d: int) -> int:
    return x << d if d > 0 else x >> -d


# --- construction and parsing -------------------------------------------------------------------


def initial(n: int) -> State:
    """Standard start. With c = n // 2: White on (c-1, c-1) and (c, c), Black on (c-1, c) and (c, c-1).
    Black moves first, so `me` is Black's bitboard."""
    if n not in SIZES:
        raise ValueError(f"board size must be one of {SIZES}, got {n}")
    c = n // 2
    black = (1 << square(c - 1, c, n)) | (1 << square(c, c - 1, n))
    white = (1 << square(c - 1, c - 1, n)) | (1 << square(c, c, n))
    return State(black, white, n, True)


def from_string(text: str, black_to_move: bool = True) -> State:
    """Parse a board string (see module docstring). n is inferred from the number of squares."""
    cells = "".join(text.split())
    n = int(round(len(cells) ** 0.5))
    if n * n != len(cells) or n not in SIZES:
        raise ValueError(f"expected 16, 36 or 64 squares, got {len(cells)}")
    me = opp = 0
    for i, ch in enumerate(cells):
        if ch == "X":
            me |= 1 << i
        elif ch == "O":
            opp |= 1 << i
        elif ch != ".":
            raise ValueError(f"unexpected character {ch!r} in board string")
    return State(me, opp, n, black_to_move)


def to_string(s: State) -> str:
    """Inverse of from_string: n lines of n characters, no trailing newline."""
    n = s.n
    cells = ["X" if s.me >> i & 1 else "O" if s.opp >> i & 1 else "." for i in range(n * n)]
    return "\n".join("".join(cells[r * n : (r + 1) * n]) for r in range(n))


# --- move generation ----------------------------------------------------------------------------


def legal_moves(s: State) -> int:
    """Bitboard of squares where the side to move may play.

    Shift in all 8 directions with edge masks so discs never wrap from the end of one row to the
    start of the next (tests/test_engine.py::test_no_wraparound), and mask every result to the
    n*n board bits."""
    n, me, opp = s.n, s.me, s.opp
    empty = _FULL[n] & ~(me | opp)
    moves = 0
    for d, pre in _DIRS[n]:
        # Squares reachable from one of my discs through a run of opponent discs. A run between my
        # disc and an empty square holds at most n-2 discs: one step here plus n-3 extensions.
        run = _shift(me & pre, d) & opp
        for _ in range(n - 3):
            run |= _shift(run & pre, d) & opp
        moves |= _shift(run & pre, d) & empty
    return moves


def flips(s: State, a: int) -> int:
    """Bitboard of opponent discs flipped by playing square `a` (0 if `a` is not a legal square)."""
    n, me, opp = s.n, s.me, s.opp
    if not 0 <= a < n * n or (me | opp) >> a & 1:
        return 0
    move = 1 << a
    flipped = 0
    for d, pre in _DIRS[n]:
        line = 0
        x = _shift(move & pre, d)
        while x & opp:
            line |= x
            x = _shift(x & pre, d)
        if x & me:
            flipped |= line
    return flipped


def _swap(s: State) -> State:
    """The same board with the other side to move (a pass)."""
    return State(s.opp, s.me, s.n, not s.black_to_move)


def legal_mask(s: State) -> np.ndarray:
    """Bool array of shape (n*n + 1,). The pass entry is True only when no square is legal and the
    game is not over. All False for a terminal state."""
    n = s.n
    mask = np.zeros(n * n + 1, dtype=bool)
    moves = legal_moves(s)
    if moves:
        while moves:
            low = moves & -moves  # lowest set bit
            mask[low.bit_length() - 1] = True
            moves ^= low
    elif legal_moves(_swap(s)):
        mask[n * n] = True  # forced pass
    return mask


def is_terminal(s: State) -> bool:
    """True when neither side has a legal square."""
    return legal_moves(s) == 0 and legal_moves(_swap(s)) == 0


def outcome(s: State) -> int:
    """For a terminal state: +1 / 0 / −1 for the side to move, by disc count."""
    diff = s.me.bit_count() - s.opp.bit_count()
    return (diff > 0) - (diff < 0)


def step(s: State, a: int) -> tuple[State, float, bool]:
    """Apply action `a` (a legal square or pass) and hand the move to the other side.

    Returns (s2, r, done):
      s2    from the next mover's perspective (me/opp swapped, black_to_move flipped). It may be a
            forced-pass state, i.e. the next mover has no square but the game is not over.
      done  is_terminal(s2)
      r     for the player who just moved: −outcome(s2) if done, else 0.0

    Raises ValueError for an illegal action, including a pass while a square is legal or once the
    game is over.
    """
    n = s.n
    if a == n * n:
        if legal_moves(s):
            raise ValueError("pass is only legal when no square is")
        s2 = _swap(s)
        if legal_moves(s2) == 0:
            raise ValueError("the game is over; no action is legal")
        return s2, 0.0, False  # the opponent has a square, so the game goes on
    f = flips(s, a)
    if f == 0:
        raise ValueError(f"square {a} is not a legal move")
    s2 = State(s.opp & ~f, s.me | f | (1 << a), n, not s.black_to_move)
    done = is_terminal(s2)
    return s2, (float(-outcome(s2)) if done else 0.0), done


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
    if depth == 0 or is_terminal(s):
        return 1
    return sum(perft(step(s, int(a))[0], depth - 1) for a in np.flatnonzero(legal_mask(s)))
