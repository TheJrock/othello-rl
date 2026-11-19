"""Evaluation: opening banks, matches, scores, solver-based accuracy, cross-play (spec §4.7).

Agents play greedy (ε = 0) here. Every match starts from the fixed opening bank, with each opening
played once in each colour (spec D4).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import solver
from .config import Config
from .engine import State, from_string, to_string
from .players import GreedyPlayer, NetPlayer, Player, PositionalPlayer, RandomPlayer

# --- statistics ---------------------------------------------------------------------------------


def wilson_interval(score: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion. Draws count as half a win in `score`."""
    raise NotImplementedError("evaluate.wilson_interval")


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

    count=None: every distinct position reachable in exactly `plies` plies (4×4 uses plies=2).
    Otherwise sample with a seeded rng until `count` distinct positions are found (6×6: plies=4,
    count=100). "Distinct" means the same discs and side to move; positions are not merged under symmetry."""
    raise NotImplementedError("evaluate.opening_bank")


# --- matches and metrics ------------------------------------------------------------------------


def reference_players(n: int) -> dict[str, Player]:
    return {"random": RandomPlayer(), "greedy": GreedyPlayer(), "positional": PositionalPlayer(n)}


def play_match(p1: Player, p2: Player, openings: Sequence[State], rng: np.random.Generator) -> MatchStats:
    """Each opening twice: once with p1 as the side to move, once as the other side. All games are
    batched through selfplay.play_games."""
    raise NotImplementedError("evaluate.play_match")


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
