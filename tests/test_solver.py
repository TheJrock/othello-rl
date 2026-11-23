"""Solver consistency (spec §4.4). The 4×4 start value is checked against the published result
(White wins 3–11; confirmed Nov 23, see spec §4.4) in test_start_value_matches_published."""

from __future__ import annotations

import numpy as np
import pytest

from othello_rl import engine as E
from othello_rl import solver as S
from othello_rl.evaluate import opening_bank
from othello_rl.players import RandomPlayer
from othello_rl.selfplay import play_games


@pytest.fixture(scope="module")
def table():
    return S.solve_4x4()


@pytest.fixture(scope="module")
def eval_positions(table):
    return S.eval_positions_4x4(table)


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


def test_eval_positions_are_informative(table, eval_positions):
    positions = eval_positions
    assert positions
    for s in positions[:500]:
        values = S.child_values(table, s)
        legal = values[~np.isnan(values)]
        assert len(legal) >= 2 and len(set(legal.tolist())) >= 2


def test_chance_accuracy_is_a_probability(table, eval_positions):
    acc = S.chance_accuracy(table, eval_positions)
    assert 0.0 < acc < 1.0


# --- the 4×4 table ------------------------------------------------------------------------------


def test_start_value_matches_published(table):
    """4×4 Othello is a second-player win (White wins 3–11 with perfect play)."""
    s = E.initial(4)
    assert table[(s.me, s.opp)] == -1  # for Black, the side to move
    assert len(table) == 58_533  # regression number (Nov 23): every position reachable from the start


def test_table_is_closed_under_moves(table, rng):
    """A full solve has no holes: every child of a stored position is stored too."""
    keys = list(table)
    for i in rng.choice(len(keys), 3000, replace=False):
        s = E.State(*keys[i], n=4)
        for a in np.flatnonzero(E.legal_mask(s)):
            s2 = E.step(s, int(a))[0]
            assert (s2.me, s2.opp) in table


def test_terminal_and_pass_entries(table):
    terminal = passes = 0
    for me, opp in table:
        s = E.State(me, opp, 4)
        if E.is_terminal(s):
            assert table[(me, opp)] == E.outcome(s)
            terminal += 1
        elif E.legal_mask(s)[E.pass_action(4)]:
            assert table[(me, opp)] == -table[(opp, me)]  # same board, other side to move
            passes += 1
    assert terminal > 0 and passes > 0


def test_table_round_trip(table, tmp_path):
    path = tmp_path / "solve4x4.npz"
    S.save_table(table, path)
    assert path.exists() and S.load_table(path) == table


def test_optimal_actions(table, rng):
    for s in sample_states(table, 500, rng):
        opt, values = S.optimal_actions(table, s), S.child_values(table, s)
        assert opt.any() and (E.legal_mask(s) | ~opt).all()  # non-empty and legal
        assert (values[opt] == table[(s.me, s.opp)]).all()
    over = next(E.State(me, opp, 4) for me, opp in table if E.is_terminal(E.State(me, opp, 4)))
    assert not S.optimal_actions(table, over).any()


def test_eval_set_size_and_chance_accuracy(table, eval_positions):
    assert len(eval_positions) == 7_912  # regression number (Nov 23)
    assert S.chance_accuracy(table, eval_positions) == pytest.approx(0.494, abs=0.001)


def test_random_player_accuracy_matches_chance(table, eval_positions, rng):
    """chance_accuracy is the expected accuracy of a uniform mover, so a real one should land on it."""
    actions = RandomPlayer().act(eval_positions, rng)
    hits = [S.optimal_actions(table, s)[a] for s, a in zip(eval_positions, actions)]
    assert np.mean(hits) == pytest.approx(S.chance_accuracy(table, eval_positions), abs=0.03)  # ≈ 5 SE


# --- the perfect player -------------------------------------------------------------------------


def test_perfect_play_achieves_the_table_value(table, rng):
    """Perfect vs perfect must end exactly at the solved value, from any starting position."""
    p = S.PerfectPlayer(table)
    starts = [E.initial(4), *opening_bank(4, plies=4, count=None, seed=0), *sample_states(table, 300, rng)]
    for s, traj in zip(starts, play_games([p] * len(starts), [p] * len(starts), starts, rng)):
        assert traj.outcome_for(s.black_to_move) == table[(s.me, s.opp)]


def test_perfect_player_never_does_worse_than_the_value(table, rng):
    perfect, rand = S.PerfectPlayer(table), RandomPlayer()
    starts = sample_states(table, 500, rng)
    black = [perfect if s.black_to_move else rand for s in starts]  # perfect plays the side to move
    white = [rand if s.black_to_move else perfect for s in starts]
    for s, traj in zip(starts, play_games(black, white, starts, rng)):
        assert traj.outcome_for(s.black_to_move) >= table[(s.me, s.opp)]
    start = E.initial(4)
    games = play_games([rand] * 200, [perfect] * 200, [start] * 200, rng)
    assert all(t.outcome_for(False) == 1 for t in games)  # as White from the start, it always wins


def test_perfect_player_passes_and_breaks_ties(table, rng):
    p = S.PerfectPlayer(table)
    forced = next(
        s
        for s in (E.State(me, opp, 4) for me, opp in table)
        if not E.is_terminal(s) and E.legal_mask(s)[E.pass_action(4)]
    )
    assert p.act([forced], rng) == [E.pass_action(4)]
    tied = next(s for s in sample_states(table, 5000, rng) if S.optimal_actions(table, s).sum() >= 2)
    chosen = set(p.act([tied] * 200, rng))
    assert chosen == set(np.flatnonzero(S.optimal_actions(table, tied)).tolist())
