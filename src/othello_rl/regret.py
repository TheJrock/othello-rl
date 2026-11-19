"""Position-level checkpoint regret, adapted from MADRID (Samvelyan et al., AAMAS 2024; spec §4.7).

    reg_R(θ) = V^θ(R, T) − V^θ(T, T)

θ is a position with a side to move (MADRID's "level"), T is the target agent (a final checkpoint),
and R is a reference policy. V^θ(X, T) is the final outcome (+1 / 0 / −1) for the side to move at θ
when X plays that side and T plays the other. With greedy agents each V is one deterministic game,
so no repeats are needed (MADRID used 4 because football is stochastic).

Reference sets (spec D8):
    self-history  every checkpoint of T's own run. Measures forgetting, but is biased toward pool
                  agents, which trained against exactly these checkpoints.
    held-out      the heuristic players + every 5th checkpoint of every *other* run. The same set
                  for every T; this is the fair cross-condition comparison.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import solver
from .engine import State
from .players import Player


def position_values(x: Player, t: Player, positions: Sequence[State], rng: np.random.Generator) -> np.ndarray:
    """V^θ(X, T) for every θ: X plays the side to move at θ, T the other side. Batched through
    selfplay.play_games."""
    raise NotImplementedError("regret.position_values")


def regret_matrix(
    target: Player, references: dict[str, Player], positions: Sequence[State], rng: np.random.Generator
) -> tuple[list[str], np.ndarray]:
    """(names, R) with R[i, j] = reg_{names[i]}(θ_j). Compute V^θ(T, T) once and reuse it."""
    raise NotImplementedError("regret.regret_matrix")


def self_history_references(run_dir: str | Path, every: int = 1) -> dict[str, Player]:
    """Greedy NetPlayers for every `every`-th checkpoint of this run, named "self@g<games>"."""
    raise NotImplementedError("regret.self_history_references")


def heldout_references(
    target_run: str | Path, all_runs: Sequence[Path], n: int, every: int = 5
) -> dict[str, Player]:
    """The 3 heuristic players + every `every`-th checkpoint of every run in `all_runs` except
    `target_run`, named "<target>-<sampling>/seed<k>@g<games>"."""
    raise NotImplementedError("regret.heldout_references")


def true_regret_4x4(
    target: Player, positions: Sequence[State], table: solver.Table, rng: np.random.Generator
) -> np.ndarray:
    """V^θ(π*, T) − V^θ(T, T) with π* = solver.PerfectPlayer(table). Something MADRID can't compute
    in football: compare it with the max over checkpoint references to see how much true regret the
    checkpoints recover (F8)."""
    raise NotImplementedError("regret.true_regret_4x4")


def sample_positions(n: int, count: int, min_empties: int, max_empties: int, seed: int) -> list[State]:
    """Distinct non-terminal positions spread evenly over min..max empties, taken from games among
    the reference players with random moves mixed in for variety. Used once to build
    data/regret_positions_6x6.json."""
    raise NotImplementedError("regret.sample_positions")


def summarize(R: np.ndarray, names: Sequence[str]) -> dict[str, Any]:
    """frac_positive: share of positions where any reference has regret > 0.
    mean_max: mean over positions of max_i R[i, θ].
    by_reference: {name: mean regret} (MADRID Fig. 4b analog)."""
    raise NotImplementedError("regret.summarize")


def regret_grid(
    positions: Sequence[State], max_regret: np.ndarray, empties_bins: np.ndarray, diff_bins: np.ndarray
) -> np.ndarray:
    """Mean max-regret per (empties × disc-differential) cell (MADRID Fig. 6 analog); NaN for empty cells."""
    raise NotImplementedError("regret.regret_grid")
