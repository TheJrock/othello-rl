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
from .nets import load_net
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

    def stats(self) -> dict[str, float]:
        """Fields for the next log row, covering the rounds since the last call; resets the counts."""
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

    def stats(self) -> dict[str, float]:
        return {}


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

    def stats(self) -> dict[str, float]:
        return {}


class PoolSampler:
    """Each group plays the live learner with probability cfg.pool_p_latest, otherwise a frozen
    snapshot drawn uniformly from all past checkpoints (game 0 included). Before the first checkpoint
    exists, every group plays the learner.

    Snapshots are loaded back from the checkpoint files, so the pool is exactly the saved checkpoints
    (spec D8) and a broken checkpoint fails here, not weeks later in the regret analysis. Each
    snapshot is one NetPlayer object, so groups that draw the same snapshot are batched into one
    forward pass by selfplay.play_games, just as groups that draw the learner are."""

    store_opponent_moves = True

    def __init__(self, learner: NetPlayer, cfg: Config):
        self.learner = learner
        self.cfg = cfg
        self.snapshots: list[NetPlayer] = []
        self.epsilon = cfg.eps_start
        self._past = 0  # groups that drew a past snapshot since the last stats() call
        self._total = 0  # groups drawn since the last stats() call

    def opponents(self, n_groups: int, rng: np.random.Generator) -> list[Player]:
        if not self.snapshots:
            return [self.learner] * n_groups
        latest = rng.random(n_groups) < self.cfg.pool_p_latest
        picks = rng.integers(0, len(self.snapshots), n_groups)
        self._past += int((~latest).sum())
        self._total += n_groups
        return [self.learner if lat else self.snapshots[i] for lat, i in zip(latest, picks)]

    def on_checkpoint(self, path: Path, games: int) -> None:
        """Load the checkpoint with nets.load_net and append it as a frozen NetPlayer."""
        net = load_net(path, self.cfg).to(self.cfg.device).requires_grad_(False)
        self.snapshots.append(NetPlayer(net, epsilon=self.epsilon, device=self.cfg.device, name=f"g{games}"))

    def set_epsilon(self, eps: float) -> None:
        self.epsilon = eps
        for snapshot in self.snapshots:
            snapshot.epsilon = eps

    def stats(self) -> dict[str, float]:
        """pool_size: snapshots available during these rounds. opp_past_frac: the fraction of game
        groups that played a past snapshot rather than the learner (≈ 1 − pool_p_latest)."""
        out: dict[str, float] = {"pool_size": len(self.snapshots)}
        if self._total:
            out["opp_past_frac"] = self._past / self._total
        self._past = self._total = 0
        return out


def make_sampler(cfg: Config, learner: NetPlayer) -> OpponentSampler:
    if cfg.sampling == "naive":
        return NaiveSampler(learner)
    if cfg.sampling == "pool":
        return PoolSampler(learner, cfg)
    if cfg.sampling == "random":  # debugging ladder rung 1 only, never an experiment condition
        return FixedSampler(RandomPlayer())
    raise ValueError(f"unknown sampling {cfg.sampling!r}")
