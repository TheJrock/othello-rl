"""Solver consistency (spec §4.4). The published 4×4 start value is checked by hand, not here."""

from __future__ import annotations

import numpy as np
import pytest

from othello_rl import engine as E
from othello_rl import solver as S


@pytest.fixture(scope="module")
def table():
    return S.solve_4x4()


def sample_states(table, count, rng, max_empties=None):
    keys = list(table)
    states = []
    for i in rng.permutation(len(keys)):
        s = E.State(*keys[i], n=4)
        empties = 16 - (s.me | s.opp).bit_count()
        if not E.is_terminal(s) and (max_empties is None or empties <= max_empties):
            states.append(s)
        if len(states) == count:
            break
    return states


def test_table_covers_start(table):
    s = E.initial(4)
    assert (s.me, s.opp) in table
    assert table[(s.me, s.opp)] in (-1, 0, 1)


def test_negamax_consistency(table, rng):
    for s in sample_states(table, 500, rng):
        values = S.child_values(table, s)
        legal = E.legal_mask(s)
        assert np.isnan(values[~legal]).all() and not np.isnan(values[legal]).any()
        assert table[(s.me, s.opp)] == np.nanmax(values)


def test_endgame_solver_matches_full_solve(table, rng):
    for s in sample_states(table, 300, rng, max_empties=8):
        assert S.solve_wld(s) == table[(s.me, s.opp)]


def test_eval_positions_are_informative(table):
    positions = S.eval_positions_4x4(table)
    assert positions
    for s in positions[:500]:
        values = S.child_values(table, s)
        legal = values[~np.isnan(values)]
        assert len(legal) >= 2 and len(set(legal.tolist())) >= 2


def test_chance_accuracy_is_a_probability(table):
    acc = S.chance_accuracy(table, S.eval_positions_4x4(table))
    assert 0.0 < acc < 1.0
