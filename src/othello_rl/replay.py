"""Transitions, the replay buffer, and symmetry augmentation (spec §4.5)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Transition:
    """One learner transition. How s2 is chosen depends on the target mode; see selfplay.cut_transitions."""

    s: np.ndarray  # (3, n, n) planes, learner to move
    a: int  # action taken (pass = n*n)
    r: float  # learner's perspective; non-zero only when done
    s2: np.ndarray  # (3, n, n) planes of the next state
    mask2: np.ndarray  # (n*n + 1,) bool legal mask of s2; all False when done
    done: bool


@dataclass
class Batch:
    s: torch.Tensor  # (B, 3, n, n) float32
    a: torch.Tensor  # (B,) int64
    r: torch.Tensor  # (B,) float32
    s2: torch.Tensor  # (B, 3, n, n) float32
    mask2: torch.Tensor  # (B, n*n + 1) bool
    done: torch.Tensor  # (B,) bool


class ReplayBuffer:
    """Fixed-capacity ring buffer over preallocated numpy arrays.

    Store planes as uint8 (they are 0/1), which keeps a 300k-transition 6×6 buffer around 65 MB.
    """

    def __init__(self, capacity: int, n: int, seed: int = 0):
        self.capacity = capacity
        self.n = n
        raise NotImplementedError("replay.ReplayBuffer.__init__")

    def __len__(self) -> int:
        raise NotImplementedError("replay.ReplayBuffer.__len__")

    def extend(self, transitions: Iterable[Transition]) -> None:
        """Append, overwriting the oldest entries once full."""
        raise NotImplementedError("replay.ReplayBuffer.extend")

    def sample(self, batch_size: int, augment: bool = True, device: str = "cpu") -> Batch:
        """Uniform sample with replacement. With `augment`, apply one random D4 symmetry per sampled
        transition, consistently to s, a, s2 and mask2 (engine.transform_planes / transform_action /
        transform_mask)."""
        raise NotImplementedError("replay.ReplayBuffer.sample")
