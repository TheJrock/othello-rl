"""Exact solvers used as ground truth (spec §4.4).

Values are win/draw/loss for the side to move: +1 / 0 / −1, matching the reward. A move is optimal
when it preserves the position's value.

Value of an action for the mover: if s2 = engine.step(s, a)[0], then
    value(s, a) = outcome-for-mover if s2 is terminal, else −V(s2)
because s2 is seen from the opponent's perspective.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .engine import State

Table = dict[tuple[int, int], int]  # (me, opp) bitboards → value for the side to move


# --- 4×4: full solve -------------------------------------------------------------------------


def solve_4x4() -> Table:
    """Memoized negamax over every position reachable from engine.initial(4).

    Check table[start] against the published 4×4 result before trusting anything downstream
    (spec §4.4)."""
    raise NotImplementedError("solver.solve_4x4")


def save_table(table: Table, path: str | Path) -> None:
    """Save as .npz with parallel arrays me, opp, value."""
    raise NotImplementedError("solver.save_table")


def load_table(path: str | Path) -> Table:
    raise NotImplementedError("solver.load_table")


def child_values(table: Table, s: State) -> np.ndarray:
    """float array (n*n + 1,): value(s, a) for each legal action, NaN for illegal actions."""
    raise NotImplementedError("solver.child_values")


def optimal_actions(table: Table, s: State) -> np.ndarray:
    """bool array (n*n + 1,): the legal actions that achieve V(s)."""
    raise NotImplementedError("solver.optimal_actions")


def eval_positions_4x4(table: Table) -> list[State]:
    """The 4×4 accuracy set: reachable non-terminal positions with ≥ 2 legal actions whose values
    are not all equal (otherwise every move is trivially optimal)."""
    raise NotImplementedError("solver.eval_positions_4x4")


def chance_accuracy(table: Table, positions: Sequence[State]) -> float:
    """Expected accuracy of a uniformly random mover: mean over positions of (#optimal / #legal).
    Drawn as the baseline line on F1."""
    raise NotImplementedError("solver.chance_accuracy")


@dataclass
class PerfectPlayer:
    """π* for 4×4: an optimal action, ties broken uniformly at random. Used for true regret (spec §4.7)."""

    table: Table
    name: str = "perfect"

    def act(self, states: Sequence[State], rng: np.random.Generator) -> list[int]:
        raise NotImplementedError("solver.PerfectPlayer.act")


# --- 6×6: endgame solver ----------------------------------------------------------------------


def solve_wld(s: State, alpha: int = -1, beta: int = 1, tt: dict | None = None) -> int:
    """Win/draw/loss value of s for the side to move, for any board size.

    Alpha-beta with a null window, move ordering (corners first, then fewest opponent replies), and
    a transposition table `tt` keyed on (me, opp). Practical up to ~12 empties on 6×6.
    tests/test_solver.py cross-checks it against the full 4×4 table."""
    raise NotImplementedError("solver.solve_wld")


def endgame_child_values(s: State) -> np.ndarray:
    """Like child_values, computed with solve_wld instead of a table."""
    raise NotImplementedError("solver.endgame_child_values")


def build_endgame_set(count: int, min_empties: int, max_empties: int, seed: int, workers: int) -> list[dict]:
    """The 6×6 endgame test set (spec §4.4): `count` distinct positions with min..max empties,
    sampled from games between randomized reference players, kept only if their legal moves don't
    all have the same value. Solve them in parallel (multiprocessing, `workers` processes).

    Returns records {"board", "black_to_move", "values"} (format in data/README.md)."""
    raise NotImplementedError("solver.build_endgame_set")
