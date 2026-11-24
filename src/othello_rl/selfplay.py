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
from .engine import State, is_terminal, legal_mask, legal_moves, outcome, pass_action, step, to_planes
from .players import NetPlayer, Player
from .replay import Transition
from .targets import TARGET_MODES


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
        o = outcome(self.final)  # for the side to move in the final position
        return o if self.final.black_to_move == black else -o


def play_games(
    black: Sequence[Player], white: Sequence[Player], starts: Sequence[State], rng: np.random.Generator
) -> list[Trajectory]:
    """Play len(starts) games. Game i uses black[i] and white[i] and starts from starts[i], whose
    `black_to_move` says who moves first.

    Batching: on every ply, group the unfinished games by the player to move (by object identity) and
    make one `act` call per player. Forced passes are recorded as plies with action = pass.

    The runner plays forced passes itself instead of asking the player: pass is the only legal action
    there, and asking would cost an `act` call (for a NetPlayer, a forward pass) to learn nothing.
    """
    if not len(black) == len(white) == len(starts):
        raise ValueError(f"got {len(black)} black, {len(white)} white, {len(starts)} starts")
    states = list(starts)
    plies: list[list[Ply]] = [[] for _ in states]
    active = [i for i, s in enumerate(states) if not is_terminal(s)]

    def record(i: int, a: int) -> bool:
        """Play action a in game i; True if the game goes on."""
        s = states[i]
        plies[i].append(Ply(s, a))
        states[i], _, done = step(s, a)
        return not done

    while active:
        still_active: list[int] = []
        groups: dict[int, tuple[Player, list[int]]] = {}
        for i in active:
            s = states[i]
            if legal_moves(s) == 0:  # not terminal (active), so this is a forced pass
                if record(i, pass_action(s.n)):
                    still_active.append(i)
                continue
            mover = black[i] if s.black_to_move else white[i]
            groups.setdefault(id(mover), (mover, []))[1].append(i)
        for mover, idx in groups.values():
            for i, a in zip(idx, mover.act([states[i] for i in idx], rng), strict=True):
                if record(i, a):
                    still_active.append(i)
        active = sorted(still_active)  # game order, so the grouping (and rng use) is reproducible
    return [Trajectory(p, s) for p, s in zip(plies, states)]


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
    if mode not in TARGET_MODES:
        raise ValueError(f"unknown target mode {mode!r}; expected one of {TARGET_MODES}")
    # states[i] is the position before ply i; states[-1] is the final (terminal) position.
    states = [ply.state for ply in traj.plies] + [traj.final]
    last = len(traj.plies)
    r_final = float(traj.outcome_for(learner_is_black))
    mine = [i for i, ply in enumerate(traj.plies) if ply.state.black_to_move == learner_is_black]
    if mode == "env":  # the environment absorbs the learner's forced passes
        mine = [i for i in mine if traj.plies[i].action != pass_action(traj.final.n)]
    # The index of each transition's s2 in `states`.
    if mode == "negamax":
        nexts = [i + 1 for i in mine]  # right after the move: the opponent to move
    else:
        nexts = mine[1:] + [last]  # the learner's next real decision, or the end of the game

    out = []
    for i, j in zip(mine, nexts):
        done = j == last
        out.append(
            Transition(
                s=to_planes(states[i]),
                a=traj.plies[i].action,
                r=r_final if done else 0.0,
                s2=to_planes(states[j]),  # when done, s2 is the final board; targets ignore it
                mask2=legal_mask(states[j]),
                done=done,
            )
        )
    return out


def collect(
    learner: NetPlayer, opponents: Sequence[Player], n_games: int, cfg: Config, rng: np.random.Generator
) -> tuple[list[Transition], dict[str, float]]:
    """One round of training games from engine.initial(cfg.board).

    Split the n_games games evenly into len(opponents) groups; group g plays opponents[g], with the
    learner as Black in half of each group's games. Cut every trajectory with cfg.target. Return all
    learner transitions plus round stats for the log (e.g. {"plies_per_game", "learner_score"})."""
    raise NotImplementedError("selfplay.collect")
