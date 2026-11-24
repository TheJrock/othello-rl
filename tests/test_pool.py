"""Opponent samplers (spec §4.6)."""

from __future__ import annotations

import pytest

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
