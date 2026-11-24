"""Opponent sampling: the `sampling` factor of the 2×2 (spec §4.6), plus the debug-only fixed
random opponent for rung 1 of the debugging ladder.

This is the only module that branches on `sampling`. Snapshots are the run's own checkpoints, so
the pool needs no extra storage (spec D8).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from .config import Config
from .players import NetPlayer, Player, RandomPlayer


class OpponentSampler(Protocol):
    # Whether the opponent's moves are stored as training data. True for both 2×2 samplings (spec
    # D2); False only for the debugging ladder's fixed random opponent (rung 1, spec §4.6).
    store_opponent_moves: bool

    def opponents(self, n_groups: int, rng: np.random.Generator) -> list[Player]:
        """One opponent per game group for the next round."""
        ...

    def on_checkpoint(self, path: Path, games: int) -> None:
        """Called after every checkpoint is saved, including the one at games = 0."""
        ...

    def set_epsilon(self, eps: float) -> None:
        """Both players use the same ε in training games (spec §4.6)."""
        ...


class NaiveSampler:
    """Latest vs latest: every group plays the live learner network. Returning the learner object
    itself (not a copy) lets selfplay.play_games batch both sides into one forward pass per ply."""

    store_opponent_moves = True

    def __init__(self, learner: NetPlayer):
        self.learner = learner

    def opponents(self, n_groups: int, rng: np.random.Generator) -> list[Player]:
        return [self.learner] * n_groups

    def on_checkpoint(self, path: Path, games: int) -> None:
        return None  # naive self-play keeps no history

    def set_epsilon(self, eps: float) -> None:
        return None  # the opponent is the learner, whose ε the training loop already sets


class FixedSampler:
    """Debugging ladder rung 1 (spec §4.6): every group plays one fixed player that never changes.

    Only the learner's moves are stored (store_opponent_moves = False). With a fixed opponent and
    env targets, that is plain single-agent DQN on a stationary MDP: the environment is the
    opponent, and it never changes. Storing the fixed opponent's moves too would add transitions
    whose environment is the learner, which changes as it trains, so the MDP would no longer be
    stationary, and the shared network would also be fitting values against two different opponents."""

    store_opponent_moves = False

    def __init__(self, opponent: Player):
        self.opponent = opponent

    def opponents(self, n_groups: int, rng: np.random.Generator) -> list[Player]:
        return [self.opponent] * n_groups

    def on_checkpoint(self, path: Path, games: int) -> None:
        return None

    def set_epsilon(self, eps: float) -> None:
        return None  # the fixed opponent's play doesn't depend on ε


class PoolSampler:
    """Each group plays the live learner with probability cfg.pool_p_latest, otherwise a frozen
    snapshot drawn uniformly from all past checkpoints (game 0 included). Before the first checkpoint
    exists, every group plays the learner."""

    store_opponent_moves = True

    def __init__(self, learner: NetPlayer, cfg: Config):
        self.learner = learner
        self.cfg = cfg
        self.snapshots: list[NetPlayer] = []

    def opponents(self, n_groups: int, rng: np.random.Generator) -> list[Player]:
        raise NotImplementedError("pool.PoolSampler.opponents")

    def on_checkpoint(self, path: Path, games: int) -> None:
        """Load the checkpoint with nets.load_net and append it as a frozen NetPlayer."""
        raise NotImplementedError("pool.PoolSampler.on_checkpoint")

    def set_epsilon(self, eps: float) -> None:
        raise NotImplementedError("pool.PoolSampler.set_epsilon")


def make_sampler(cfg: Config, learner: NetPlayer) -> OpponentSampler:
    if cfg.sampling == "naive":
        return NaiveSampler(learner)
    if cfg.sampling == "pool":
        return PoolSampler(learner, cfg)
    if cfg.sampling == "random":  # debugging ladder rung 1 only, never an experiment condition
        return FixedSampler(RandomPlayer())
    raise ValueError(f"unknown sampling {cfg.sampling!r}")
