"""Opponent sampling: the `sampling` factor of the 2×2 (spec §4.6).

This is the only module that branches on `sampling`. Snapshots are the run's own checkpoints, so
the pool needs no extra storage (spec D8).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from .config import Config
from .players import NetPlayer, Player


class OpponentSampler(Protocol):
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
    """Latest vs latest: every group plays the live learner network."""

    def __init__(self, learner: NetPlayer):
        self.learner = learner

    def opponents(self, n_groups: int, rng: np.random.Generator) -> list[Player]:
        raise NotImplementedError("pool.NaiveSampler.opponents")

    def on_checkpoint(self, path: Path, games: int) -> None:
        return None  # naive self-play keeps no history

    def set_epsilon(self, eps: float) -> None:
        return None  # the opponent is the learner, whose ε the training loop already sets


class PoolSampler:
    """Each group plays the live learner with probability cfg.pool_p_latest, otherwise a frozen
    snapshot drawn uniformly from all past checkpoints (game 0 included). Before the first checkpoint
    exists, every group plays the learner."""

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
    raise ValueError(f"unknown sampling {cfg.sampling!r}")
