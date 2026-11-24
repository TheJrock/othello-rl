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


# --- spec §4.5 target tests, end to end: cut + targets against exact 4×4 values -----------------
# With the solver's exact values standing in for a perfect Q-function, the Bellman target of every
# transition must equal the exact value of the move. Any sign, pass or terminal mistake in
# cut_transitions or td_targets breaks the equality somewhere in a few hundred random games.


@pytest.fixture(scope="module")
def table():
    from othello_rl import solver

    return solver.solve_4x4()


def state_from_planes(planes: np.ndarray) -> E.State:
    bits = 1 << np.arange(N * N, dtype=object)
    return E.State(int(bits[planes[0].ravel() > 0].sum()), int(bits[planes[1].ravel() > 0].sum()), N)


def exact_targets(table, transitions, mode):
    """(td_targets with exact Q as both online and target net, exact Q(s, a)) for each transition."""
    import torch

    from othello_rl import solver
    from othello_rl.targets import td_targets

    q2 = np.stack([solver.child_values(table, state_from_planes(tr.s2)) for tr in transitions])
    q2 = torch.from_numpy(np.nan_to_num(q2, nan=0.0)).float()  # illegal entries are masked anyway
    y = td_targets(
        torch.tensor([tr.r for tr in transitions]),
        torch.tensor([tr.done for tr in transitions]),
        q2,
        q2,
        torch.from_numpy(np.stack([tr.mask2 for tr in transitions])),
        mode,
    )
    q_sa = [solver.child_values(table, state_from_planes(tr.s))[tr.a] for tr in transitions]
    return y.numpy(), np.array(q_sa, dtype=np.float32)


def games_with_learner(learner, opponent, count, rng):
    """`count` games from the start and from every 4×4 opening, the learner as Black in half."""
    from othello_rl.evaluate import opening_bank

    starts = [E.initial(N)] * count + opening_bank(N, plies=4, count=None, seed=0)
    as_black = [i % 2 == 0 for i in range(len(starts))]
    black = [learner if b else opponent for b in as_black]
    white = [opponent if b else learner for b in as_black]
    return list(zip(play_games(black, white, starts, rng), as_black))


def test_negamax_targets_match_exact_values(table, rng):
    p = RandomPlayer()
    games = games_with_learner(p, p, 200, rng)
    trs = [tr for traj, black in games for tr in cut_transitions(traj, black, "negamax")]
    y, q_sa = exact_targets(table, trs, "negamax")
    np.testing.assert_array_equal(y, q_sa)
    # The spec §4.5 cases all occur: (1) a winning last move, (2) s2 a forced-pass state, (3) done rows.
    assert any(tr.done and tr.r == 1.0 for tr in trs)
    assert any(tr.mask2[PASS] for tr in trs)
    # done only when the learner's own ply ended the game; otherwise its last transition bootstraps
    learner_ended = sum(traj.plies[-1].state.black_to_move == black for traj, black in games)
    assert sum(tr.done for tr in trs) == learner_ended < len(games)


def test_env_targets_match_exact_values_against_a_perfect_opponent(table, rng):
    """Against a perfect opponent, value-against-the-opponent is the minimax value, so the env-mode
    target (+Q of the learner's next decision) must also equal Q*(s, a)."""
    from othello_rl.solver import PerfectPlayer

    games = games_with_learner(RandomPlayer(), PerfectPlayer(table), 200, rng)
    trs = [tr for traj, black in games for tr in cut_transitions(traj, black, "env")]
    y, q_sa = exact_targets(table, trs, "env")
    np.testing.assert_array_equal(y, q_sa)
    # The spec §4.5 cases all occur: (4) the opponent's reply ends the game, (5) the opponent passes
    # so the learner moves again, (6) the learner's forced pass is absorbed.
    plies = [(ply, black) for traj, black in games for ply in traj.plies]
    assert any(ply.action == PASS and ply.state.black_to_move != black for ply, black in plies)
    assert any(ply.action == PASS and ply.state.black_to_move == black for ply, black in plies)
    assert any(
        traj.plies and traj.plies[-1].state.black_to_move != black and traj.plies[-1].action != PASS
        for traj, black in games
    )


def test_env_targets_depend_on_the_opponent(table, rng):
    """Spec D1 in one test: against a weak opponent, env-mode targets are not the minimax value.
    Q learns the value of a move against *this* opponent, which is why the opponent pool can matter."""
    p = RandomPlayer()
    games = games_with_learner(p, p, 200, rng)
    trs = [tr for traj, black in games for tr in cut_transitions(traj, black, "env")]
    y, q_sa = exact_targets(table, trs, "env")
    assert (y > q_sa).any()  # a random opponent often throws away its advantage


# --- collect: one training round, both sides' moves stored (spec D2) -----------------------------


def round_config(target):
    from othello_rl.config import Config

    return Config(board=N, target=target, parallel_games=16, pool_groups=4)


def test_collect_negamax_stores_every_ply_of_both_sides(rng):
    from othello_rl.selfplay import collect

    p = RandomPlayer()
    trs, stats = collect(p, [p] * 4, 16, round_config("negamax"), rng)
    # Negamax keeps every ply as a transition, passes included, so both sides give one per ply.
    assert len(trs) == round(stats["plies_per_game"] * 16) == stats["transitions_per_game"] * 16
    assert sum(tr.done for tr in trs) == 16  # exactly one ply ends each game, whoever made it


def test_collect_env_stores_every_real_move_of_both_sides(rng):
    from othello_rl.selfplay import collect

    learner = Recorder(RandomPlayer(), "learner")
    opponents = [Recorder(RandomPlayer(), f"opp{g}") for g in range(4)]
    trs, _ = collect(learner, opponents, 16, round_config("env"), rng)
    asked = sum(len(b) for p in (learner, *opponents) for b in p.batches)  # every non-pass ply
    assert len(trs) == asked  # env drops passes; every real move by either side is a transition
    assert sum(tr.done for tr in trs) == 32  # each game ends both sides' last transition


def test_collect_splits_groups_and_colours(rng):
    from othello_rl.selfplay import collect

    learner = Recorder(RandomPlayer(), "learner")
    opponents = [Recorder(RandomPlayer(), f"opp{g}") for g in range(4)]
    _, stats = collect(learner, opponents, 16, round_config("negamax"), rng)
    start = E.initial(N)
    assert learner.batches[0] == [start] * 8  # Black in half of all 16 games, one batched call
    for opp in opponents:
        assert opp.batches[0] == [start] * 2  # Black in half of its own group's 4 games
    assert 0.0 <= stats["learner_score"] <= 1.0


def test_collect_naive_batches_both_colours_together(rng):
    from othello_rl.selfplay import collect

    learner = Recorder(RandomPlayer(), "learner")
    collect(learner, [learner] * 4, 16, round_config("env"), rng)
    assert len(learner.batches[0]) == 16  # opponent is the learner itself: one call for every game


def test_collect_rejects_uneven_groups(rng):
    from othello_rl.selfplay import collect

    p = RandomPlayer()
    with pytest.raises(ValueError):
        collect(p, [p] * 4, 10, round_config("env"), rng)
