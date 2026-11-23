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

from .engine import State, flips, initial, legal_mask, legal_moves, step

Table = dict[tuple[int, int], int]  # (me, opp) bitboards → value for the side to move


# --- 4×4: full solve -------------------------------------------------------------------------


def solve_4x4() -> Table:
    """Memoized negamax over every position reachable from engine.initial(4).

    Check table[start] against the published 4×4 result before trusting anything downstream
    (spec §4.4).

    Every reachable position is visited, with no alpha-beta cutoffs: child_values and the regret
    analysis look up children that optimal play never reaches, so the table must hold them all.
    Terminal positions are stored with their outcome and forced-pass positions with −V(swapped).
    Keys drop black_to_move because the rules never read it."""
    table: Table = {}

    def negamax(me: int, opp: int) -> int:
        key = (me, opp)
        v = table.get(key)
        if v is not None:
            return v
        moves = legal_moves(State(me, opp, 4))
        if moves == 0:
            if legal_moves(State(opp, me, 4)) == 0:  # neither side can move: game over
                diff = me.bit_count() - opp.bit_count()
                v = (diff > 0) - (diff < 0)
            else:  # forced pass: same board, other side to move
                v = -negamax(opp, me)
        else:
            v = -1
            s = State(me, opp, 4)
            while moves:
                bit = moves & -moves
                moves ^= bit
                f = flips(s, bit.bit_length() - 1)
                v = max(v, -negamax(opp & ~f, me | f | bit))  # no early exit at v == 1 (see above)
        table[key] = v
        return v

    start = initial(4)
    negamax(start.me, start.opp)
    return table


def save_table(table: Table, path: str | Path) -> None:
    """Save as .npz with parallel arrays me, opp, value."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = np.array(list(table), dtype=np.int64).reshape(-1, 2)
    with path.open("wb") as f:  # a file handle stops numpy from appending a second ".npz"
        np.savez_compressed(
            f, me=keys[:, 0], opp=keys[:, 1], value=np.fromiter(table.values(), np.int8, len(table))
        )


def load_table(path: str | Path) -> Table:
    with np.load(path) as data:
        return dict(zip(zip(data["me"].tolist(), data["opp"].tolist()), data["value"].tolist()))


def child_values(table: Table, s: State) -> np.ndarray:
    """float array (n*n + 1,): value(s, a) for each legal action, NaN for illegal actions."""
    values = np.full(s.n * s.n + 1, np.nan)
    for a in np.flatnonzero(legal_mask(s)):
        s2, r, done = step(s, int(a))
        values[a] = r if done else -table[(s2.me, s2.opp)]
    return values


def optimal_actions(table: Table, s: State) -> np.ndarray:
    """bool array (n*n + 1,): the legal actions that achieve V(s)."""
    values = child_values(table, s)
    legal = ~np.isnan(values)
    if not legal.any():  # terminal: nothing is legal, so nothing is optimal
        return legal
    return legal & (values == np.nanmax(values))


def eval_positions_4x4(table: Table) -> list[State]:
    """The 4×4 accuracy set: reachable non-terminal positions with ≥ 2 legal actions whose values
    are not all equal (otherwise every move is trivially optimal).

    Sorted by (me, opp) so the set is identical on every machine. black_to_move is left at its
    default (True); the table doesn't record colour and no metric reads it."""
    positions = []
    for me, opp in sorted(table):
        s = State(me, opp, 4)
        values = child_values(table, s)
        legal = values[~np.isnan(values)]
        if legal.size >= 2 and legal.min() != legal.max():
            positions.append(s)
    return positions


def chance_accuracy(table: Table, positions: Sequence[State]) -> float:
    """Expected accuracy of a uniformly random mover: mean over positions of (#optimal / #legal).
    Drawn as the baseline line on F1."""
    if not positions:
        raise ValueError("chance_accuracy needs at least one position")
    return float(np.mean([optimal_actions(table, s).sum() / legal_mask(s).sum() for s in positions]))


@dataclass
class PerfectPlayer:
    """π* for 4×4: an optimal action, ties broken uniformly at random. Used for true regret (spec §4.7)."""

    table: Table
    name: str = "perfect"

    def act(self, states: Sequence[State], rng: np.random.Generator) -> list[int]:
        # Imported here, not at the top: players imports torch, and this module also runs in the
        # endgame solver's worker processes, which shouldn't pay torch's start-up cost.
        from .players import legal_masks, pick_best

        if not states:
            return []
        masks = legal_masks(states)  # raises on terminal states, like every other player
        return pick_best(np.stack([child_values(self.table, s) for s in states]), masks, rng)


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
