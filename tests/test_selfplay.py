"""Transition cutting for both target modes (spec §3 D2–D3), checked on real random 4×4 games."""

from __future__ import annotations

import numpy as np
import pytest

from othello_rl import engine as E
from othello_rl.players import RandomPlayer
from othello_rl.selfplay import cut_transitions, play_games

N = 4
PASS = E.pass_action(N)


@pytest.fixture
def trajectories(rng):
    p = RandomPlayer()
    count = 200
    return play_games([p] * count, [p] * count, [E.initial(N)] * count, rng)


def learner_plies(traj, learner_is_black):
    return [i for i, ply in enumerate(traj.plies) if ply.state.black_to_move == learner_is_black]


def test_play_games_records_complete_games(trajectories):
    assert len(trajectories) == 200
    for traj in trajectories:
        assert E.is_terminal(traj.final)
        state = E.initial(N)
        for ply in traj.plies:  # replaying the recorded actions reproduces every recorded state
            assert ply.state == state
            state, _, _ = E.step(state, ply.action)
        assert state == traj.final


@pytest.mark.parametrize("learner_is_black", [True, False])
def test_negamax_cut(trajectories, learner_is_black):
    for traj in trajectories:
        idx = learner_plies(traj, learner_is_black)
        trs = cut_transitions(traj, learner_is_black, "negamax")
        assert len(trs) == len(idx)  # every learner ply, passes included
        for i, tr in zip(idx, trs):
            ply = traj.plies[i]
            np.testing.assert_array_equal(tr.s, E.to_planes(ply.state))
            assert tr.a == ply.action
            if tr.a == PASS:
                assert not tr.s[2].any()  # passes only when no square is legal
            nxt = traj.plies[i + 1].state if i + 1 < len(traj.plies) else traj.final
            np.testing.assert_array_equal(tr.s2, E.to_planes(nxt))
            last_ply = i == len(traj.plies) - 1
            assert tr.done == last_ply
            assert tr.r == (traj.outcome_for(learner_is_black) if last_ply else 0.0)
            assert tr.mask2.any() != tr.done


@pytest.mark.parametrize("learner_is_black", [True, False])
def test_env_cut(trajectories, learner_is_black):
    for traj in trajectories:
        idx = [i for i in learner_plies(traj, learner_is_black) if traj.plies[i].action != PASS]
        trs = cut_transitions(traj, learner_is_black, "env")
        assert len(trs) == len(idx)  # learner passes are absorbed into the environment
        for k, (i, tr) in enumerate(zip(idx, trs)):
            assert tr.a == traj.plies[i].action != PASS
            if k + 1 < len(idx):  # s2 is the learner's next real decision
                assert not tr.done and tr.r == 0.0
                np.testing.assert_array_equal(tr.s2, E.to_planes(traj.plies[idx[k + 1]].state))
                assert tr.mask2[:-1].any() and not tr.mask2[PASS]
            else:  # the learner's last move: the game always ends before its next decision
                assert tr.done and tr.r == traj.outcome_for(learner_is_black)
                assert not tr.mask2.any()


def test_unknown_mode_raises(trajectories):
    with pytest.raises(ValueError):
        cut_transitions(trajectories[0], True, "minimax")
