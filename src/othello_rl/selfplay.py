"""Batched games and transition cutting (spec §3 D2–D3, §4.6).

The work is split into two layers so the error-prone part can be unit-tested on its own:
1. play_games() plays many games at once and records full trajectories (both sides' plies, forced
   passes included). It knows nothing about learning.
2. cut_transitions() turns one trajectory into the learner's replay transitions for a target mode.

This file and targets.py are the only places that branch on `target`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .config import Config
from .engine import State
from .players import NetPlayer, Player
from .replay import Transition


@dataclass(frozen=True)
class Ply:
    state: State  # position before the move, from the mover's perspective
    action: int  # engine action index (pass = n*n)


@dataclass
class Trajectory:
    plies: list[Ply]
    final: State  # the terminal position

    def outcome_for(self, black: bool) -> int:
        """+1 / 0 / −1 for Black (black=True) or White, from engine.outcome(self.final)."""
        raise NotImplementedError("selfplay.Trajectory.outcome_for")


def play_games(
    black: Sequence[Player], white: Sequence[Player], starts: Sequence[State], rng: np.random.Generator
) -> list[Trajectory]:
    """Play len(starts) games. Game i uses black[i] and white[i] and starts from starts[i], whose
    `black_to_move` says who moves first.

    Batching: on every ply, group the unfinished games by the player to move (by object identity) and
    make one `act` call per player. Forced passes are recorded as plies with action = pass."""
    raise NotImplementedError("selfplay.play_games")


def cut_transitions(traj: Trajectory, learner_is_black: bool, mode: str) -> list[Transition]:
    """The learner's replay transitions from one game. The opponent's moves are never stored (spec D2).

    negamax  one transition per learner ply, passes included. s2 = the position right after the
             move (opponent to move; may be a forced-pass state). done and r are set only when the
             learner's ply ended the game.
    env      one transition per learner non-pass ply. s2 = the next position where the learner is to
             move and has a legal square, skipping the opponent's reply and any forced passes by
             either side. If the game ends first, done = True and r = the learner's outcome. Learner
             passes never appear because the environment absorbs them.

    In both modes r is from the learner's perspective (+1 / 0 / −1) and non-zero only when done;
    mask2 = engine.legal_mask(s2), which is all False when done. Raise ValueError for an unknown mode.
    """
    raise NotImplementedError("selfplay.cut_transitions")


def collect(
    learner: NetPlayer, opponents: Sequence[Player], n_games: int, cfg: Config, rng: np.random.Generator
) -> tuple[list[Transition], dict[str, float]]:
    """One round of training games from engine.initial(cfg.board).

    Split the n_games games evenly into len(opponents) groups; group g plays opponents[g], with the
    learner as Black in half of each group's games. Cut every trajectory with cfg.target. Return all
    learner transitions plus round stats for the log (e.g. {"plies_per_game", "learner_score"})."""
    raise NotImplementedError("selfplay.collect")
