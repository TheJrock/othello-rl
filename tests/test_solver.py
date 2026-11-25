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


def test_endgame_solver_windows_with_a_shared_table(table, rng):
    """Fail-soft bounds stay correct when null windows and one shared transposition table mix: a
    bound stored by one search must never be misread as an exact value by another."""
    tt: dict = {}
    for s in sample_states(table, 1500, rng):
        v = table[(s.me, s.opp)]
        for alpha, beta in [(-1, 0), (0, 1), (-1, 1)]:
            r = S.solve_wld(s, alpha, beta, tt=tt)
            if v <= alpha:
                assert v <= r <= alpha
            elif v >= beta:
                assert beta <= r <= v
            else:
                assert r == v


def test_endgame_solver_rejects_an_empty_window():
    with pytest.raises(ValueError):
        S.solve_wld(E.initial(4), 0, 0)


def test_endgame_child_values_match_the_table(table, rng):
    states = sample_states(table, 1000, rng)
    assert any(E.legal_mask(s)[16] for s in states)  # forced passes are covered
    for s in states:
        np.testing.assert_array_equal(S.endgame_child_values(s), S.child_values(table, s))


def _plain_negamax(s):
    """Reference for 6×6: no pruning, no table, no ordering."""
    legal = np.flatnonzero(E.legal_mask(s))
    if legal.size == 0:
        return E.outcome(s)
    return max(r if done else -_plain_negamax(s2) for s2, r, done in (E.step(s, int(a)) for a in legal))


def test_endgame_solver_matches_plain_negamax_on_6x6(rng):
    checked = 0
    while checked < 40:
        s = E.initial(6)
        while 36 - (s.me | s.opp).bit_count() > 7 and not E.is_terminal(s):
            s = E.step(s, int(rng.choice(np.flatnonzero(E.legal_mask(s)))))[0]
        if E.is_terminal(s):
            continue
        assert S.solve_wld(s) == _plain_negamax(s)
        checked += 1


@pytest.fixture(scope="module")
def endgame_4x4():
    return S.build_endgame_set(count=40, min_empties=5, max_empties=8, seed=0, workers=1, n=4)


def test_endgame_set_values_are_exact(table, endgame_4x4):
    for r in endgame_4x4:
        s = E.from_string(r["board"], r["black_to_move"])
        exact = S.child_values(table, s)
        assert r["values"] == [None if np.isnan(v) else int(v) for v in exact]


def test_endgame_set_positions_are_informative_distinct_and_in_range(endgame_4x4):
    assert len(endgame_4x4) == 40
    keys = set()
    empties = []
    for r in endgame_4x4:
        s = E.from_string(r["board"], r["black_to_move"])
        legal = [v for v in r["values"] if v is not None]
        assert len(legal) >= 2 and min(legal) != max(legal)  # the move choice matters
        empties.append(16 - (s.me | s.opp).bit_count())
        keys.add(S._canonical(s))
    assert len(keys) == 40  # distinct even up to symmetry
    assert set(empties) == {5, 6, 7, 8}  # every level appears


def test_endgame_set_is_seeded_and_independent_of_workers(endgame_4x4):
    assert S.build_endgame_set(count=40, min_empties=5, max_empties=8, seed=0, workers=2, n=4) == endgame_4x4
    assert S.build_endgame_set(count=40, min_empties=5, max_empties=8, seed=1, workers=1, n=4) != endgame_4x4


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


# --- 4×4 vs a uniformly random opponent (debugging ladder rung 1) -------------------------------


@pytest.fixture(scope="module")
def vs_random():
    return S.solve_vs_random_4x4()


@pytest.fixture(scope="module")
def bank():
    return opening_bank(4, 4, None, 0)  # the data/openings_4x4.json bank: all 120 positions


def every_legal(s, values):
    return ~np.isnan(values)


def test_random_vs_random_scores_exactly_half(bank):
    """Each opening from both sides, both players random: the two games mirror each other."""
    assert S.exact_score_vs_random(every_legal, bank) == pytest.approx(0.5)


def test_expectimax_values_are_consistent(vs_random, eval_positions):
    for s in eval_positions[::50]:
        q = vs_random.child_values(s)
        assert np.nanmax(q) == pytest.approx(vs_random.mover[(s.me, s.opp)])


def test_ceilings_against_random(table, vs_random, bank):
    """The numbers quoted in spec §4.6. From the start, best play beats random 95.9% as Black and
    always as White; on the bank the ceiling is 83.4%, and even π* only gets 77.8%, which is why a
    "≥ 95% vs random" bar can't apply to 4×4 bank matches."""
    start = E.initial(4)
    assert (vs_random.mover[(start.me, start.opp)] + 1) / 2 == pytest.approx(0.959, abs=5e-4)
    assert vs_random.random[(start.me, start.opp)] == 1.0
    best = S.exact_score_vs_random(lambda s, v: v == np.nanmax(v), bank)
    perfect = S.exact_score_vs_random(lambda s, v: S.optimal_actions(table, s), bank)
    assert best == pytest.approx(0.834, abs=5e-4)
    assert perfect == pytest.approx(0.778, abs=5e-4)
