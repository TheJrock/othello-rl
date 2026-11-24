"""Transitions, the replay buffer, and symmetry augmentation (spec §4.5)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch

from .engine import transform_mask, transform_planes


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
        self.rng = np.random.default_rng(seed)
        self.s = np.zeros((capacity, 3, n, n), dtype=np.uint8)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, 3, n, n), dtype=np.uint8)
        self.mask2 = np.zeros((capacity, n * n + 1), dtype=bool)
        self.done = np.zeros(capacity, dtype=bool)
        self.size = 0  # filled entries
        self.ptr = 0  # next slot to write

    def __len__(self) -> int:
        return self.size

    def extend(self, transitions: Iterable[Transition]) -> None:
        """Append, overwriting the oldest entries once full."""
        batch = list(transitions)[-self.capacity :]  # more than capacity: only the newest survive
        if not batch:
            return
        idx = (self.ptr + np.arange(len(batch))) % self.capacity
        self.s[idx] = np.stack([t.s for t in batch])
        self.a[idx] = [t.a for t in batch]
        self.r[idx] = [t.r for t in batch]
        self.s2[idx] = np.stack([t.s2 for t in batch])
        self.mask2[idx] = np.stack([t.mask2 for t in batch])
        self.done[idx] = [t.done for t in batch]
        self.ptr = int((self.ptr + len(batch)) % self.capacity)
        self.size = min(self.size + len(batch), self.capacity)

    def sample(self, batch_size: int, augment: bool = True, device: str = "cpu") -> Batch:
        """Uniform sample with replacement. With `augment`, apply one random D4 symmetry per sampled
        transition, consistently to s, a, s2 and mask2 (engine.transform_planes / transform_action /
        transform_mask)."""
        if self.size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        idx = self.rng.integers(0, self.size, batch_size)
        s, a, s2, mask2 = self.s[idx], self.a[idx], self.s2[idx], self.mask2[idx]
        if augment:
            ks = self.rng.integers(0, 8, batch_size)
            for k in range(1, 8):  # k = 0 is the identity
                rows = np.flatnonzero(ks == k)
                if rows.size == 0:
                    continue
                s[rows] = transform_planes(s[rows], k)
                s2[rows] = transform_planes(s2[rows], k)
                mask2[rows] = transform_mask(mask2[rows], self.n, k)
                # Move the action through the very table that moves the mask, as a one-hot row, so
                # the action and the mask can never disagree about where a square went.
                onehot = np.zeros((rows.size, self.n * self.n + 1), dtype=bool)
                onehot[np.arange(rows.size), a[rows]] = True
                a[rows] = transform_mask(onehot, self.n, k).argmax(axis=1)

        def put(x: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
            return torch.from_numpy(x).to(device=device, dtype=dtype)

        return Batch(
            s=put(s, torch.float32),
            a=put(a, torch.int64),
            r=put(self.r[idx], torch.float32),
            s2=put(s2, torch.float32),
            mask2=put(mask2, torch.bool),
            done=put(self.done[idx], torch.bool),
        )
