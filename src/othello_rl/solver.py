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

from .engine import State, flips, initial, is_terminal, legal_mask, legal_moves, outcome, step

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


# --- 4×4: play against a uniformly random opponent (debugging ladder rung 1) ------------------
# Rung 1 is a stationary MDP whose environment is the random player, so its optimal policy is not
# π* but the one that maximises the expected outcome against random (expectimax). Both tables
# below hold expected outcomes (in [−1, 1], so expected score = (value + 1) / 2) for the side that
# is NOT random, keyed on (me, opp) from the side to move's perspective.


@dataclass
class VsRandom:
    """mover[(me, opp)]: the learner is to move and plays optimally against random from here on.
    random[(me, opp)]: the random player is to move (me = its discs); value for the learner."""

    mover: dict[tuple[int, int], float]
    random: dict[tuple[int, int], float]

    def child_values(self, s: State) -> np.ndarray:
        """Q*(s, a) of rung 1's MDP for the learner to move in s; NaN for illegal actions."""
        q = np.full(s.n * s.n + 1, np.nan)
        for a in np.flatnonzero(legal_mask(s)):
            s2, r, done = step(s, int(a))
            q[a] = r if done else self.random[(s2.me, s2.opp)]
        return q


def _expected_vs_random(choose, s: State, memo_mover: dict, memo_random: dict) -> tuple[float, float]:
    """(value if the learner is to move in s, value if random is to move in s) for the learner
    choosing uniformly among the actions in choose(s, values) (values = its actions' child values)."""

    def mover(s: State) -> float:
        key = (s.me, s.opp)
        if key not in memo_mover:
            if is_terminal(s):
                memo_mover[key] = float(outcome(s))
            else:
                values = np.full(s.n * s.n + 1, np.nan)
                for a in np.flatnonzero(legal_mask(s)):
                    s2, r, done = step(s, int(a))
                    values[a] = r if done else rand(s2)
                memo_mover[key] = float(np.mean(values[choose(s, values)]))
        return memo_mover[key]

    def rand(s: State) -> float:
        key = (s.me, s.opp)
        if key not in memo_random:
            if is_terminal(s):
                memo_random[key] = float(-outcome(s))
            else:
                children = [step(s, int(a)) for a in np.flatnonzero(legal_mask(s))]
                memo_random[key] = float(np.mean([-r if done else mover(s2) for s2, r, done in children]))
        return memo_random[key]

    return mover(s), rand(s)


def _best(s: State, values: np.ndarray) -> np.ndarray:
    return values == np.nanmax(values)


def solve_vs_random_4x4() -> VsRandom:
    """Expectimax over every position reachable from the 4×4 start (≈ 1 s): the best possible play
    against a uniformly random opponent, from either side. Every child of every position is
    visited, so the learner-to-move table covers every reachable position."""
    mover: dict = {}
    rand: dict = {}
    _expected_vs_random(_best, initial(4), mover, rand)  # learner as Black (mover) and as White (rand)
    return VsRandom(mover, rand)


def exact_score_vs_random(choose, starts: Sequence[State]) -> float:
    """Exact expected score against a uniformly random opponent, each start played once from each
    side (the learner to move, then random to move), like evaluate.play_match. `choose(s, values)`
    returns a bool mask of the learner's actions in s; it picks uniformly among them. No sampling
    noise, unlike a match: use it to check a 4×4 policy against the expectimax ceiling."""
    memo_mover: dict = {}
    memo_random: dict = {}
    scores = []
    for s in starts:
        as_mover, as_second = _expected_vs_random(choose, s, memo_mover, memo_random)
        scores += [(as_mover + 1) / 2, (as_second + 1) / 2]
    return float(np.mean(scores))


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
