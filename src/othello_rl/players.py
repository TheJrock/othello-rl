"""Players: anything that picks moves for a batch of positions (spec §4.3).

Every player implements `act(states, rng) -> actions` on a batch, so the game runner makes one call
per player per ply and NetPlayer does one forward pass for all of them (spec §4.6). Every player must
return the pass action when pass is the only legal action.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from .engine import State


class Player(Protocol):
    name: str

    def act(self, states: Sequence[State], rng: np.random.Generator) -> list[int]:
        """One legal action (engine action index, pass included) per state."""
        ...


# Square weights for the positional player (spec §4.3). Corners high; X-squares (diagonal to a
# corner) and C-squares (edge squares next to a corner) negative.
POSITIONAL_WEIGHTS: dict[int, np.ndarray] = {
    4: np.array(
        [
            [100, -20, -20, 100],
            [-20, 0, 0, -20],
            [-20, 0, 0, -20],
            [100, -20, -20, 100],
        ]
    ),
    6: np.array(
        [
            [100, -20, 10, 10, -20, 100],
            [-20, -50, -2, -2, -50, -20],
            [10, -2, 1, 1, -2, 10],
            [10, -2, 1, 1, -2, 10],
            [-20, -50, -2, -2, -50, -20],
            [100, -20, 10, 10, -20, 100],
        ]
    ),
    8: np.array(
        [
            [100, -20, 10, 5, 5, 10, -20, 100],
            [-20, -50, -2, -2, -2, -2, -50, -20],
            [10, -2, -1, -1, -1, -1, -2, 10],
            [5, -2, -1, -1, -1, -1, -2, 5],
            [5, -2, -1, -1, -1, -1, -2, 5],
            [10, -2, -1, -1, -1, -1, -2, 10],
            [-20, -50, -2, -2, -2, -2, -50, -20],
            [100, -20, 10, 5, 5, 10, -20, 100],
        ]
    ),
}


@dataclass
class RandomPlayer:
    """Uniform over legal moves."""

    name: str = "random"

    def act(self, states: Sequence[State], rng: np.random.Generator) -> list[int]:
        raise NotImplementedError("players.RandomPlayer.act")


@dataclass
class GreedyPlayer:
    """Maximise discs flipped (engine.flips); ties broken uniformly at random."""

    name: str = "greedy"

    def act(self, states: Sequence[State], rng: np.random.Generator) -> list[int]:
        raise NotImplementedError("players.GreedyPlayer.act")


@dataclass
class PositionalPlayer:
    """Maximise POSITIONAL_WEIGHTS[n] of the square played; ties → most flips → uniformly at random."""

    n: int
    name: str = "positional"

    def act(self, states: Sequence[State], rng: np.random.Generator) -> list[int]:
        raise NotImplementedError("players.PositionalPlayer.act")


class NetPlayer:
    """ε-greedy over masked Q-values from a QNet.

    `epsilon` is mutable: the training loop sets it every round for the learner, and the pool sets it
    for its snapshots (spec §4.6). Evaluation uses epsilon = 0.
    """

    def __init__(self, net: torch.nn.Module, epsilon: float = 0.0, device: str = "cpu", name: str = "net"):
        self.net = net
        self.epsilon = epsilon
        self.device = device
        self.name = name

    def q_values(self, states: Sequence[State]) -> np.ndarray:
        """Raw Q-values, shape (B, n*n + 1), from one forward pass over the stacked engine.to_planes.
        No masking. Runs under torch.no_grad()."""
        raise NotImplementedError("players.NetPlayer.q_values")

    def act(self, states: Sequence[State], rng: np.random.Generator) -> list[int]:
        """Per state: with probability epsilon a uniformly random legal action, otherwise the masked
        argmax (exact ties broken at random)."""
        raise NotImplementedError("players.NetPlayer.act")
