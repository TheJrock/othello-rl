"""Q-network and checkpoint I/O (spec §4.5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import Config


class QNet(nn.Module):
    """Small conv Q-network. ONNX-friendly: no batch norm, only conv / ReLU / linear / tanh.

    trunk  `layers` × [Conv2d(3×3, `channels`, padding=1) → ReLU]   (the first takes 3 input planes)
    head   Conv2d(1×1, 2) → flatten → Linear(2·n², n² + 1) → tanh

    Input (B, 3, n, n) float32 planes from engine.to_planes. Output (B, n² + 1) Q-values in [−1, 1].
    About 115k parameters at n = 6, 64 channels, 4 layers.

    No ReLU after the 1×1 conv (spec §4.5). With only 2 channels, a ReLU there starts out dead on
    every input in ~15–25% of initialisations (both channels negative everywhere): the output is then
    constant and no gradient reaches the trunk, so that seed never learns. Without it, the 1×1 conv
    and the linear layer are a linear read-out of the trunk, which cannot die.
    """

    def __init__(self, n: int, channels: int = 64, layers: int = 3):
        super().__init__()
        self.n = n
        trunk: list[nn.Module] = []
        for i in range(layers):
            trunk += [nn.Conv2d(3 if i == 0 else channels, channels, 3, padding=1), nn.ReLU()]
        self.trunk = nn.Sequential(*trunk)
        self.head = nn.Sequential(
            nn.Conv2d(channels, 2, 1),  # no ReLU here: see the class docstring
            nn.Flatten(),
            nn.Linear(2 * n * n, n * n + 1),
            nn.Tanh(),  # Q in [−1, 1], the range of the ±1 / 0 outcome
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))


def build_net(cfg: Config) -> QNet:
    return QNet(cfg.board, channels=cfg.channels, layers=cfg.layers)


def save_checkpoint(path: str | Path, net: nn.Module, **meta: Any) -> None:
    """Save the state dict plus plain metadata (games, grad_steps, ...)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), **meta}, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=True)


def load_net(path: str | Path, cfg: Config) -> QNet:
    """A QNet in eval mode with weights from a checkpoint written by save_checkpoint."""
    net = build_net(cfg)
    net.load_state_dict(load_checkpoint(path)["state_dict"])
    return net.eval()
