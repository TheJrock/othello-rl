"""Players: anything that picks moves for a batch of positions (spec §4.3).

Every player implements `act(states, rng) -> actions` on a batch, so the game runner makes one call
per player per ply and NetPlayer does one forward pass for all of them (spec §4.6). Every player must
return the pass action when pass is the only legal action.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch

from .engine import State, flips, legal_mask, to_planes


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


# --- shared helpers -----------------------------------------------------------------------------
# Every player is "score the legal actions, take the best, break exact ties uniformly at random".
# Pass needs no special case: engine.legal_mask marks it legal only when nothing else is, so a
# forced-pass row has exactly one legal action and pick_best returns it.


def legal_masks(states: Sequence[State]) -> np.ndarray:
    """(B, n*n + 1) bool legal masks. Raises ValueError for a terminal state: no action is legal
    there, so a caller that asks for one has a bug."""
    masks = np.stack([legal_mask(s) for s in states])
    if not masks.any(axis=1).all():
        raise ValueError("cannot act in a terminal state")
    return masks


def pick_best(scores: np.ndarray, masks: np.ndarray, rng: np.random.Generator) -> list[int]:
    """Per row, an action drawn uniformly from the legal actions with the highest score.

    Each tied action gets an independent uniform key in [0, 1) and the largest key wins, which is a
    uniform choice among the ties in one vectorized step. Everything else gets key −1, so an illegal
    action can never be returned. scores: (B, A) numeric; masks: (B, A) bool with ≥ 1 True per row.
    """
    legal_scores = np.where(masks, scores, -np.inf)
    best = masks & (legal_scores == legal_scores.max(axis=1, keepdims=True))
    keys = np.where(best, rng.random(best.shape), -1.0)
    return keys.argmax(axis=1).tolist()


def _flip_counts(states: Sequence[State], candidates: np.ndarray) -> np.ndarray:
    """(B, n*n + 1) discs flipped by each candidate square; 0 elsewhere (pass included)."""
    counts = np.zeros(candidates.shape, dtype=np.int64)
    for i, s in enumerate(states):
        for a in np.flatnonzero(candidates[i, :-1]):
            counts[i, a] = flips(s, int(a)).bit_count()
    return counts


# --- reference players --------------------------------------------------------------------------


@dataclass
class RandomPlayer:
    """Uniform over legal moves."""

    name: str = "random"

    def act(self, states: Sequence[State], rng: np.random.Generator) -> list[int]:
        if not states:
            return []
        masks = legal_masks(states)
        return pick_best(np.zeros(masks.shape), masks, rng)  # every legal action ties


@dataclass
class GreedyPlayer:
    """Maximise discs flipped (engine.flips); ties broken uniformly at random."""

    name: str = "greedy"

    def act(self, states: Sequence[State], rng: np.random.Generator) -> list[int]:
        if not states:
            return []
        masks = legal_masks(states)
        return pick_best(_flip_counts(states, masks), masks, rng)


@dataclass
class PositionalPlayer:
    """Maximise POSITIONAL_WEIGHTS[n] of the square played; ties → most flips → uniformly at random."""

    n: int
    name: str = "positional"
    _weights: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n not in POSITIONAL_WEIGHTS:
            raise ValueError(f"no positional weights for board size {self.n}")
        # One weight per action. Pass gets 0, which never matters: it is legal only when nothing else is.
        self._weights = np.append(POSITIONAL_WEIGHTS[self.n].ravel(), 0)

    def act(self, states: Sequence[State], rng: np.random.Generator) -> list[int]:
        if not states:
            return []
        masks = legal_masks(states)
        if masks.shape[1] != self._weights.size:
            raise ValueError(f"PositionalPlayer({self.n}) got {states[0].n}×{states[0].n} positions")
        weights = np.where(masks, self._weights, -np.inf)
        top = masks & (weights == weights.max(axis=1, keepdims=True))  # legal squares of the best weight
        return pick_best(_flip_counts(states, top), top, rng)


# --- network player -----------------------------------------------------------------------------


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
        planes = torch.from_numpy(np.stack([to_planes(s) for s in states])).to(self.device)
        with torch.no_grad():
            q = self.net(planes)
        return q.float().cpu().numpy()

    def act(self, states: Sequence[State], rng: np.random.Generator) -> list[int]:
        """Per state: with probability epsilon a uniformly random legal action, otherwise the masked
        argmax (exact ties broken at random)."""
        if not states:
            return []
        masks = legal_masks(states)
        q = self.q_values(states)
        explore = rng.random(len(states)) < self.epsilon
        # An exploring row scores every action 0, so pick_best makes it uniform over the legal actions.
        return pick_best(np.where(explore[:, None], 0.0, q), masks, rng)
