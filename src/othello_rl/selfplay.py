"""Batched games and transition cutting (spec §3 D2–D3, §4.6).

The work is split into two layers so the error-prone part can be unit-tested on its own:
1. play_games() plays many games at once and records full trajectories (both sides' plies, forced
   passes included). It knows nothing about learning.
2. cut_transitions() turns one trajectory into one side's replay transitions for a target mode.
   collect() calls it for both sides of every game: both sides' moves are stored (spec D2).

This file and targets.py are the only places that branch on `target`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .config import Config
from .engine import (
    State,
    initial,
    is_terminal,
    legal_mask,
    legal_moves,
    outcome,
    pass_action,
    step,
    to_planes,
)
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


def cut_transitions(traj: Trajectory, black: bool, mode: str) -> list[Transition]:
    """One side's replay transitions from one game: Black's if `black`, else White's. Call it for
    both sides to store every move of the game, as collect() does (spec D2). "The mover" below is
    that side; "the other side" may be the learner or its opponent.

    negamax  one transition per ply by the mover, passes included. s2 = the position right after the
             move (the other side to move; may be a forced-pass state). done and r are set only when
             the mover's ply ended the game.
    env      one transition per non-pass ply by the mover. s2 = the next position where the mover is
             to move and has a legal square, skipping the other side's reply and any forced passes by
             either side. If the game ends first, done = True and r = the mover's outcome. The
             mover's passes never appear because the environment absorbs them.

    In both modes r is from the mover's perspective (+1 / 0 / −1) and non-zero only when done;
    mask2 = engine.legal_mask(s2), which is all False when done. Raise ValueError for an unknown mode.
    """
    if mode not in TARGET_MODES:
        raise ValueError(f"unknown target mode {mode!r}; expected one of {TARGET_MODES}")
    # states[i] is the position before ply i; states[-1] is the final (terminal) position.
    states = [ply.state for ply in traj.plies] + [traj.final]
    last = len(traj.plies)
    r_final = float(traj.outcome_for(black))
    mine = [i for i, ply in enumerate(traj.plies) if ply.state.black_to_move == black]
    if mode == "env":  # the environment absorbs the mover's forced passes
        mine = [i for i in mine if traj.plies[i].action != pass_action(traj.final.n)]
    # The index of each transition's s2 in `states`.
    if mode == "negamax":
        nexts = [i + 1 for i in mine]  # right after the move: the other side to move
    else:
        nexts = mine[1:] + [last]  # the mover's next real decision, or the end of the game

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
    learner: NetPlayer,
    opponents: Sequence[Player],
    n_games: int,
    cfg: Config,
    rng: np.random.Generator,
    learner_only: bool = False,
) -> tuple[list[Transition], dict[str, float]]:
    """One round of training games from engine.initial(cfg.board).

    Split the n_games games evenly into len(opponents) groups; group g plays opponents[g], with the
    learner as Black in half of each group's games (one extra Black game when a group is odd). Cut
    every trajectory with cfg.target from BOTH sides (spec D2): the opponent's moves are training
    data too, whoever the opponent is. Return all transitions plus round stats for the log:
    plies_per_game, transitions_per_game, and learner_score (the learner's score in these games,
    draws counting half).

    learner_only=True cuts only the learner's side. Only the debugging ladder's rung 1 uses it
    (pool.FixedSampler.store_opponent_moves); every experiment condition stores both sides.

    Under naive sampling the opponent is the learner object itself, so play_games batches both
    sides into one forward pass per ply."""
    groups = len(opponents)
    if groups == 0 or n_games % groups:
        raise ValueError(f"n_games ({n_games}) must split evenly into {groups} opponent groups")
    per_group = n_games // groups
    black: list[Player] = []
    white: list[Player] = []
    learner_black: list[bool] = []
    for opponent in opponents:
        for j in range(per_group):
            lb = j % 2 == 0  # alternate colours inside each group
            black.append(learner if lb else opponent)
            white.append(opponent if lb else learner)
            learner_black.append(lb)
    trajs = play_games(black, white, [initial(cfg.board)] * n_games, rng)
    sides = [(lb,) if learner_only else (True, False) for lb in learner_black]
    transitions = [
        tr
        for t, s in zip(trajs, sides)
        for black_side in s
        for tr in cut_transitions(t, black_side, cfg.target)
    ]
    stats = {
        "plies_per_game": float(np.mean([len(t.plies) for t in trajs])),
        "transitions_per_game": len(transitions) / n_games,
        "learner_score": float(np.mean([(t.outcome_for(lb) + 1) / 2 for t, lb in zip(trajs, learner_black)])),
    }
    return transitions, stats
