"""Evaluation: opening banks, matches, scores, solver-based accuracy, cross-play (spec §4.7).

Agents play greedy (ε = 0) here. Every match starts from the fixed opening bank, with each opening
played once in each colour (spec D4).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import solver
from .config import Config
from .engine import State, from_string, initial, is_terminal, legal_mask, step, to_string
from .players import GreedyPlayer, NetPlayer, Player, PositionalPlayer, RandomPlayer
from .selfplay import play_games

# --- statistics ---------------------------------------------------------------------------------


def wilson_interval(score: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion. Draws count as half a win in `score`.

    Treating a draw as half a Bernoulli success slightly overstates the variance (a draw is less
    spread out than a coin flip between win and loss), so with draws the interval is a little
    conservative. Unlike the normal approximation, it stays inside [0, 1] and is sensible at 0% or 100%.
    """
    if n <= 0:
        raise ValueError(f"need at least one game, got n={n}")
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"score must be in [0, 1], got {score}")
    z2 = z * z
    denom = 1 + z2 / n
    center = (score + z2 / (2 * n)) / denom
    half = z * math.sqrt(score * (1 - score) / n + z2 / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


@dataclass
class MatchStats:
    """Counts from the first player's point of view."""

    wins: int
    draws: int
    losses: int

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games

    def ci(self, z: float = 1.96) -> tuple[float, float]:
        return wilson_interval(self.score, self.games, z)


# --- position files -----------------------------------------------------------------------------


def save_states(states: Sequence[State], path: str | Path) -> None:
    """Write positions in the data/README.md format."""
    records = [{"board": to_string(s), "black_to_move": s.black_to_move} for s in states]
    Path(path).write_text(json.dumps(records, indent=1) + "\n")


def load_states(path: str | Path) -> list[State]:
    return [from_string(r["board"], r["black_to_move"]) for r in json.loads(Path(path).read_text())]


def load_endgame_set(path: str | Path) -> list[dict[str, Any]]:
    """Records {"board", "black_to_move", "values"} from data/endgame6.jsonl."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def opening_bank(n: int, plies: int, count: int | None, seed: int) -> list[State]:
    """Distinct positions after `plies` uniformly random plies from engine.initial(n).

    count=None: every distinct position reachable in exactly `plies` plies (4×4 uses plies=4).
    Otherwise sample with a seeded rng until `count` distinct positions are found (6×6 and 8×8: plies=6,
    count=100). "Distinct" means the same discs and side to move; positions are not merged under symmetry.

    Terminal positions are never included (there would be nothing to play). Forced passes count as
    plies, as in engine.perft. Raises ValueError if `count` distinct positions can't be found."""
    if count is None:
        level = {initial(n)}
        for _ in range(plies):  # a terminal position has no legal action, so it drops out here
            level = {step(s, int(a))[0] for s in level for a in np.flatnonzero(legal_mask(s))}
        bank = [s for s in level if not is_terminal(s)]
        return sorted(bank, key=lambda s: (s.me, s.opp, s.black_to_move))  # stable order for the file

    rng = np.random.default_rng(seed)
    bank: list[State] = []
    seen: set[State] = set()
    max_tries = 100 * count  # generous: even 100 of the 236 6×6 positions at 4 plies takes only ~200
    for _ in range(max_tries):
        s = initial(n)
        for _ in range(plies):
            legal = np.flatnonzero(legal_mask(s))
            if legal.size == 0:  # the game ended before `plies` plies
                break
            s = step(s, int(rng.choice(legal)))[0]
        else:
            if not is_terminal(s) and s not in seen:
                seen.add(s)
                bank.append(s)
                if len(bank) == count:
                    return bank
    raise ValueError(
        f"found only {len(bank)} distinct {n}×{n} positions after {plies} plies in {max_tries} tries "
        f"(wanted {count}); use more plies or count=None to enumerate them all"
    )


# --- matches and metrics ------------------------------------------------------------------------


def reference_players(n: int) -> dict[str, Player]:
    return {"random": RandomPlayer(), "greedy": GreedyPlayer(), "positional": PositionalPlayer(n)}


def play_match(p1: Player, p2: Player, openings: Sequence[State], rng: np.random.Generator) -> MatchStats:
    """Each opening twice: once with p1 as the side to move, once as the other side. All games are
    batched through selfplay.play_games."""
    if not openings:
        raise ValueError("play_match needs at least one opening")
    black: list[Player] = []
    white: list[Player] = []
    p1_is_black: list[bool] = []
    for s in openings:
        for p1_moves_first in (True, False):
            p1_black = s.black_to_move == p1_moves_first  # p1 takes the colour whose turn it is, or not
            black.append(p1 if p1_black else p2)
            white.append(p2 if p1_black else p1)
            p1_is_black.append(p1_black)
    starts = [s for s in openings for _ in range(2)]
    results = [t.outcome_for(b) for t, b in zip(play_games(black, white, starts, rng), p1_is_black)]
    return MatchStats(wins=results.count(1), draws=results.count(0), losses=results.count(-1))


def optimal_move_accuracy(player: Player, positions: Sequence[State], table: solver.Table) -> float:
    """4×4: fraction of positions where the player's action is in solver.optimal_actions."""
    raise NotImplementedError("evaluate.optimal_move_accuracy")


def q_solver_mse(player: NetPlayer, positions: Sequence[State], table: solver.Table) -> float:
    """4×4: mean squared error between Q(s, a) and solver.child_values over legal actions. Logged for
    every run, but only meaningful for negamax runs, whose Q targets the minimax value (spec §4.5)."""
    raise NotImplementedError("evaluate.q_solver_mse")


def endgame_accuracy(player: Player, records: Sequence[dict[str, Any]]) -> float:
    """6×6: fraction of endgame positions where the player's action has the best recorded value."""
    raise NotImplementedError("evaluate.endgame_accuracy")


@dataclass
class EvalContext:
    """Everything evaluation needs, loaded once per run."""

    openings: list[State]
    references: dict[str, Player]
    table: solver.Table | None = None
    positions_4x4: list[State] | None = None
    endgame_set: list[dict[str, Any]] | None = None

    @classmethod
    def from_config(cls, cfg: Config) -> EvalContext:
        table = solver.load_table(cfg.solver_table) if cfg.solver_table else None
        return cls(
            openings=load_states(cfg.openings),
            references=reference_players(cfg.board),
            table=table,
            positions_4x4=solver.eval_positions_4x4(table) if table is not None else None,
            endgame_set=load_endgame_set(cfg.endgame_set) if cfg.endgame_set else None,
        )


def evaluate_checkpoint(
    net: torch.nn.Module,
    previous: torch.nn.Module | None,
    cfg: Config,
    ctx: EvalContext,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """The evaluation fields of one log row (spec §4.6–4.7):

    score_<ref>, ci_<ref>     for each reference player (random, greedy, positional)
    score_selfplay            vs the previous checkpoint (spec D5); omitted when previous is None
    accuracy_4x4, q_mse_4x4   when ctx.table is set
    endgame_accuracy          when ctx.endgame_set is set
    """
    raise NotImplementedError("evaluate.evaluate_checkpoint")


def crossplay_matrix(
    players: Sequence[Player], openings: Sequence[State], rng: np.random.Generator
) -> np.ndarray:
    """M[i, j] = score of players[i] against players[j] over the opening bank (spec §4.7, F4)."""
    raise NotImplementedError("evaluate.crossplay_matrix")


def summarize_run(rows: Sequence[dict[str, Any]], key: str = "score_positional") -> dict[str, float]:
    """Per-run summary numbers (spec §4.7): final (mean of the last 3 eval points), peak, max_drawdown
    (largest drop below the running max after the peak), forgetting_events (drops > 0.10 below the
    running max), spearman (score_selfplay vs `key`)."""
    raise NotImplementedError("evaluate.summarize_run")
