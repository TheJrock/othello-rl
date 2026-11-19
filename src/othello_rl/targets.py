"""Double-DQN targets for both target formulations (spec §3 D1, §4.5).

This file and selfplay.py are the only places that branch on `target`:

    env      y = r if done else  Q_target(s2, argmax_a' Q_online(s2, a'))    s2 = learner to move again
    negamax  y = r if done else −Q_target(s2, argmax_a' Q_online(s2, a'))    s2 = opponent to move

γ = 1 in both. The argmax is over legal actions only. Illegal actions are filled with MASK_FILL (not
−inf) and terminal rows are selected with torch.where, so rows whose mask is all False (terminal
s2) never produce NaNs.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from .replay import Batch

TargetMode = Literal["env", "negamax"]
TARGET_MODES = ("env", "negamax")
MASK_FILL = -1e9


def masked_argmax(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per row, the index of the largest q among legal actions. q: (B, A) float; mask: (B, A) bool.
    Rows with no legal action return an arbitrary index; callers must ignore those rows."""
    raise NotImplementedError("targets.masked_argmax")


def td_targets(
    r: torch.Tensor,
    done: torch.Tensor,
    q_next_online: torch.Tensor,
    q_next_target: torch.Tensor,
    mask_next: torch.Tensor,
    mode: TargetMode,
) -> torch.Tensor:
    """Targets y, shape (B,). The online network picks the action and the target network values it.
    Raise ValueError for an unknown mode. A pure function of tensors (tests/test_targets.py)."""
    raise NotImplementedError("targets.td_targets")


def td_loss(
    online: nn.Module, target_net: nn.Module, batch: Batch, mode: TargetMode
) -> tuple[torch.Tensor, dict[str, float]]:
    """Huber loss between online(batch.s)[batch.a] and td_targets(...), with the targets computed
    under torch.no_grad(). Returns (loss, info) where info = {"loss", "mean_q", "mean_target"}."""
    raise NotImplementedError("targets.td_loss")
