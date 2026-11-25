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


def solver_values(table: solver.Table, positions: Sequence[State]) -> np.ndarray:
    """(P, n*n + 1) matrix of solver.child_values: the exact value of every legal move, NaN for
    illegal ones. Computed once per run (EvalContext) instead of at every eval point."""
    return np.stack([solver.child_values(table, s) for s in positions])


def optimal_move_accuracy(
    player: Player, positions: Sequence[State], values: np.ndarray, rng: np.random.Generator
) -> float:
    """Fraction of positions where the player's action achieves the position's value. `values`
    holds exact action values with NaN for illegal actions: solver_values(table, positions) on 4×4,
    endgame_values(records) for the 6×6 endgame set."""
    actions = np.asarray(player.act(positions, rng))
    chosen = values[np.arange(len(positions)), actions]  # NaN (never optimal) if the move is illegal
    return float(np.mean(chosen == np.nanmax(values, axis=1)))


def _q_vs_solver(
    player: NetPlayer, positions: Sequence[State], values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Q(s, a) and the exact value, flattened over every legal (position, action) pair."""
    legal = ~np.isnan(values)
    return player.q_values(positions)[legal], values[legal]


def q_solver_mse(player: NetPlayer, positions: Sequence[State], values: np.ndarray) -> float:
    """4×4: mean squared error between Q(s, a) and the exact value over legal actions. Logged for
    every run, but only meaningful for negamax runs, whose Q targets the minimax value (spec §4.5)."""
    q, v = _q_vs_solver(player, positions, values)
    return float(np.mean((q - v) ** 2))


def q_solver_corr(player: NetPlayer, positions: Sequence[State], values: np.ndarray) -> float:
    """4×4: Pearson correlation between Q(s, a) and the exact value over legal actions (the "Q
    correlates with solver values" check for negamax, spec §6). Unlike the MSE it ignores scale and
    offset, so it shows whether Q ranks moves correctly even before its magnitudes are right.
    NaN if Q is constant."""
    q, v = _q_vs_solver(player, positions, values)
    if q.std() == 0:
        return float("nan")
    return float(np.corrcoef(q, v)[0, 1])


def endgame_values(records: Sequence[dict[str, Any]]) -> tuple[list[State], np.ndarray]:
    """(positions, values) from endgame-set records, with values shaped like solver_values: one row
    per position, NaN for illegal actions."""
    positions = [from_string(r["board"], r["black_to_move"]) for r in records]
    values = np.array([[np.nan if v is None else v for v in r["values"]] for r in records], dtype=float)
    return positions, values


def endgame_accuracy(player: Player, records: Sequence[dict[str, Any]], rng: np.random.Generator) -> float:
    """6×6: fraction of endgame positions where the player's action has the best recorded value
    (optimal_move_accuracy on the endgame set). The set's positions are all won or drawn for the
    side to move, since in a lost one every move is equally bad (solver.build_endgame_set)."""
    positions, values = endgame_values(records)
    return optimal_move_accuracy(player, positions, values, rng)


@dataclass
class EvalContext:
    """Everything evaluation needs, loaded once per run."""

    openings: list[State]
    references: dict[str, Player]
    table: solver.Table | None = None
    positions_4x4: list[State] | None = None
    values_4x4: np.ndarray | None = None  # solver_values(table, positions_4x4)
    endgame_set: list[dict[str, Any]] | None = None

    @classmethod
    def from_config(cls, cfg: Config) -> EvalContext:
        table = solver.load_table(cfg.solver_table) if cfg.solver_table else None
        positions = solver.eval_positions_4x4(table) if table is not None else None
        return cls(
            openings=load_states(cfg.openings),
            references=reference_players(cfg.board),
            table=table,
            positions_4x4=positions,
            values_4x4=solver_values(table, positions) if table is not None else None,
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

    score_<ref>, ci_<ref>                 for each reference player (random, greedy, positional)
    score_selfplay, ci_selfplay           vs the previous checkpoint (spec D5); omitted when previous is None
    accuracy_4x4, q_mse_4x4, q_corr_4x4   when ctx.table is set
    endgame_accuracy                      when ctx.endgame_set is set

    Both networks play greedy (ε = 0) through the opening bank, each opening from both sides.
    """
    player = NetPlayer(net, epsilon=0.0, device=cfg.device, name="eval")
    row: dict[str, Any] = {}

    def match(name: str, opponent: Player) -> None:
        stats = play_match(player, opponent, ctx.openings, rng)
        row[f"score_{name}"] = stats.score
        row[f"ci_{name}"] = list(stats.ci())

    for name, ref in ctx.references.items():
        match(name, ref)
    if previous is not None:
        match("selfplay", NetPlayer(previous, epsilon=0.0, device=cfg.device, name="previous"))
    if ctx.values_4x4 is not None:
        row["accuracy_4x4"] = optimal_move_accuracy(player, ctx.positions_4x4, ctx.values_4x4, rng)
        row["q_mse_4x4"] = q_solver_mse(player, ctx.positions_4x4, ctx.values_4x4)
        row["q_corr_4x4"] = q_solver_corr(player, ctx.positions_4x4, ctx.values_4x4)
    if ctx.endgame_set is not None:
        row["endgame_accuracy"] = endgame_accuracy(player, ctx.endgame_set, rng)
    return row


def crossplay_matrix(
    players: Sequence[Player], openings: Sequence[State], rng: np.random.Generator
) -> np.ndarray:
    """M[i, j] = score of players[i] against players[j] over the opening bank (spec §4.7, F4)."""
    raise NotImplementedError("evaluate.crossplay_matrix")


def summarize_run(
    rows: Sequence[dict[str, Any]], key: str = "score_positional", threshold: float = 0.10
) -> dict[str, float]:
    """Per-run summary numbers (spec §4.7) for the metric `key` over a run's log rows:

    final              mean of the last 3 eval points
    peak, peak_games   the best value and where it came (the first time, if tied)
    max_drawdown       peak minus the lowest value at or after the peak (≥ 0)
    forgetting_events  times the value falls more than `threshold` below its running max (each
                       entry into that state counts once; recovering and falling again counts again)
    spearman           Spearman ρ between score_selfplay and `key` over the eval points that have
                       both, except the first of them; NaN if either is constant or fewer than 3
                       points are left. Game 0 has no self-play score, and the first self-play
                       score is the first checkpoint vs the untrained net: always ~0.97–0.99,
                       at the metric's lowest point, so it would pull ρ negative on its own
                       (spec §4.7, decided Nov 25)

    Rows without `key` are skipped, so this also works on a run that is still going."""
    from scipy.stats import spearmanr  # here, not at the top: training imports this module

    points = [(r["games"], r[key]) for r in rows if key in r]
    if not points:
        raise ValueError(f"no rows with {key!r}")
    games = np.array([g for g, _ in points])
    v = np.array([x for _, x in points], dtype=float)
    i_peak = int(np.argmax(v))
    running = np.maximum.accumulate(v)
    below = v < running - threshold
    entries = int(below[0]) + int(np.sum(below[1:] & ~below[:-1]))

    pairs = np.array([(r["score_selfplay"], r[key]) for r in rows if key in r and "score_selfplay" in r])
    pairs = pairs[1:]  # drop the first checkpoint vs the untrained net (see the docstring)
    rho = float("nan")
    if len(pairs) >= 3 and pairs[:, 0].std() > 0 and pairs[:, 1].std() > 0:
        rho = float(spearmanr(pairs[:, 0], pairs[:, 1]).statistic)
    return {
        "final": float(v[-3:].mean()),
        "peak": float(v[i_peak]),
        "peak_games": int(games[i_peak]),
        "max_drawdown": float(v[i_peak] - v[i_peak:].min()),
        "forgetting_events": entries,
        "spearman": rho,
    }
