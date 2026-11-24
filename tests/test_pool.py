"""Opponent samplers (spec §4.6)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from othello_rl.config import Config
from othello_rl.nets import build_net
from othello_rl.players import NetPlayer, RandomPlayer
from othello_rl.pool import FixedSampler, NaiveSampler, make_sampler


@pytest.fixture
def learner():
    return NetPlayer(build_net(Config(board=4, channels=8, layers=1)), epsilon=0.3, name="learner")


def test_naive_plays_the_learner_object_itself(learner, rng):
    """The very same object, so play_games batches both sides into one forward pass."""
    opps = NaiveSampler(learner).opponents(8, rng)
    assert len(opps) == 8
    assert all(o is learner for o in opps)


def test_naive_stores_opponent_moves(learner):
    assert make_sampler(Config(sampling="naive"), learner).store_opponent_moves  # spec D2


def test_random_sampler_is_a_fixed_random_opponent_whose_moves_are_not_data(learner, rng):
    sampler = make_sampler(Config(sampling="random"), learner)
    assert isinstance(sampler, FixedSampler)
    opps = sampler.opponents(4, rng)
    assert len(opps) == 4 and all(isinstance(o, RandomPlayer) for o in opps)
    assert opps[0] is opps[3]  # one object, so its games batch together
    assert not sampler.store_opponent_moves  # rung 1 is a stationary single-agent MDP
    sampler.set_epsilon(0.0)  # no-ops: the opponent never changes
    sampler.on_checkpoint(None, 0)
    assert sampler.opponents(4, rng)[0] is opps[0]


# --- PoolSampler -------------------------------------------------------------------------------


def pool_setup(tmp_path, n_checkpoints, **cfg_overrides):
    """A PoolSampler whose learner has been checkpointed n_checkpoints times, the weights changing
    between checkpoints the way training changes them."""
    from othello_rl.nets import save_checkpoint
    from othello_rl.pool import PoolSampler

    cfg = Config(board=4, channels=8, layers=1, **cfg_overrides)
    torch.manual_seed(0)
    learner = NetPlayer(build_net(cfg), epsilon=0.3, name="learner")
    pool = PoolSampler(learner, cfg)
    for k in range(n_checkpoints):
        path = tmp_path / f"g{k}.pt"
        save_checkpoint(path, learner.net, games=k)
        pool.on_checkpoint(path, k)
        with torch.no_grad():  # "train" a little before the next checkpoint
            for p in learner.net.parameters():
                p.add_(torch.randn_like(p))
    return learner, pool


def test_pool_plays_the_learner_until_the_first_checkpoint(learner, rng):
    from othello_rl.pool import PoolSampler

    pool = PoolSampler(learner, Config(board=4))
    assert all(o is learner for o in pool.opponents(8, rng))


def test_snapshots_are_frozen_copies_of_the_saved_weights(tmp_path):
    """A snapshot must keep playing like the checkpoint it came from after the learner moves on.
    (Holding a reference to the live net instead would silently turn the pool into naive play.)"""
    from othello_rl.engine import initial

    learner, pool = pool_setup(tmp_path, 2)
    states = [initial(4)]
    first = pool.snapshots[0].q_values(states)
    with torch.no_grad():
        for p in learner.net.parameters():
            p.mul_(-3.0)
    np.testing.assert_array_equal(pool.snapshots[0].q_values(states), first)
    assert not np.allclose(pool.snapshots[0].q_values(states), pool.snapshots[1].q_values(states))
    assert not np.allclose(learner.q_values(states), first)
    net = pool.snapshots[0].net
    assert not net.training and not any(p.requires_grad for p in net.parameters())


@pytest.mark.parametrize("p_latest", [0.0, 0.5, 0.8, 1.0])
def test_mix_of_learner_and_snapshots(tmp_path, rng, p_latest):
    learner, pool = pool_setup(tmp_path, 3, pool_p_latest=p_latest)
    picks = [o for _ in range(2_500) for o in pool.opponents(8, rng)]  # 20,000 groups
    latest = np.mean([o is learner for o in picks])
    assert latest == pytest.approx(p_latest, abs=0.015)
    assert pool.stats() == {"pool_size": 3, "opp_past_frac": pytest.approx(1 - latest)}
    assert pool.stats() == {"pool_size": 3}  # counts reset after each call
    if p_latest < 1:  # uniform over all snapshots, game 0 included
        counts = np.array([sum(o is s for o in picks) for s in pool.snapshots])
        assert counts.min() > 0.9 * counts.mean()


def test_repeat_draws_are_one_object_so_their_games_batch_together(tmp_path, rng):
    _, pool = pool_setup(tmp_path, 2, pool_p_latest=0.0)
    picks = pool.opponents(64, rng)
    assert len({id(o) for o in picks}) == 2


def test_set_epsilon_reaches_old_and_new_snapshots(tmp_path):
    from othello_rl.nets import save_checkpoint

    learner, pool = pool_setup(tmp_path, 2)
    pool.set_epsilon(0.07)
    save_checkpoint(tmp_path / "late.pt", learner.net, games=99)
    pool.on_checkpoint(tmp_path / "late.pt", 99)
    assert [s.epsilon for s in pool.snapshots] == [0.07] * 3


def test_make_sampler_pool(learner):
    from othello_rl.pool import PoolSampler

    sampler = make_sampler(Config(sampling="pool"), learner)
    assert isinstance(sampler, PoolSampler) and sampler.store_opponent_moves  # spec D2
