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


# --- the game runner ----------------------------------------------------------------------------


class Recorder:
    """Wraps a player and records every batch it is asked to act on."""

    def __init__(self, inner, name="recorder"):
        self.inner, self.name = inner, name
        self.batches: list[list[E.State]] = []

    def act(self, states, rng):
        self.batches.append(list(states))
        return self.inner.act(states, rng)


class Extreme:
    """Deterministic: always the lowest (or highest) legal square, so tests can tell who moved."""

    def __init__(self, highest: bool):
        self.highest, self.name = highest, "highest" if highest else "lowest"

    def act(self, states, rng):
        pick = max if self.highest else min
        return [pick(np.flatnonzero(E.legal_mask(s)[:-1]).tolist()) for s in states]


def test_outcome_for_matches_disc_count(trajectories):
    for traj in trajectories:
        f = traj.final
        black, white = (f.me, f.opp) if f.black_to_move else (f.opp, f.me)
        diff = black.bit_count() - white.bit_count()
        assert traj.outcome_for(True) == (diff > 0) - (diff < 0) == -traj.outcome_for(False)


def test_play_games_batches_one_call_per_player_per_ply(rng):
    p = Recorder(RandomPlayer())
    trajs = play_games([p] * 100, [p] * 100, [E.initial(N)] * 100, rng)
    assert len(p.batches[0]) == 100  # every game's first move in one call
    assert len(p.batches) <= max(len(t.plies) for t in trajs)  # at most one call per ply
    moves = [ply for t in trajs for ply in t.plies if ply.action != PASS]
    assert sum(len(b) for b in p.batches) == len(moves)  # each real move asked for exactly once
    assert any(ply.action == PASS for t in trajs for ply in t.plies)  # 4×4 random games do pass...
    assert all(E.legal_moves(s) for b in p.batches for s in b)  # ...but the runner plays them itself


def test_play_games_routes_each_ply_to_the_right_player(rng):
    n = 6
    low, high = Extreme(False), Extreme(True)
    white_first = E.step(E.initial(n), int(np.flatnonzero(E.legal_mask(E.initial(n)))[0]))[0]
    count = 40
    black = [low if i % 2 else high for i in range(count)]
    white = [high if i % 3 else low for i in range(count)]
    starts = [white_first if i % 4 == 0 else E.initial(n) for i in range(count)]
    trajs = play_games(black, white, starts, rng)
    for i, traj in enumerate(trajs):
        assert traj.plies[0].state == starts[i]
        for ply in traj.plies:
            if ply.action != E.pass_action(n):
                mover = black[i] if ply.state.black_to_move else white[i]
                assert ply.action == mover.act([ply.state], rng)[0]


def test_play_games_handles_terminal_starts(rng):
    over = E.from_string("X" + "." * (N * N - 1))
    p = RandomPlayer()
    trajs = play_games([p, p], [p, p], [over, E.initial(N)], rng)
    assert trajs[0].plies == [] and trajs[0].final == over
    assert E.is_terminal(trajs[1].final) and trajs[1].plies


def test_play_games_is_reproducible():
    p = RandomPlayer()
    a = play_games([p] * 20, [p] * 20, [E.initial(6)] * 20, np.random.default_rng(3))
    b = play_games([p] * 20, [p] * 20, [E.initial(6)] * 20, np.random.default_rng(3))
    assert [t.plies for t in a] == [t.plies for t in b]


def test_play_games_checks_lengths(rng):
    p = RandomPlayer()
    with pytest.raises(ValueError):
        play_games([p], [p, p], [E.initial(N)], rng)
