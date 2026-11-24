"""Q-network shape, size and checkpoint I/O (spec §4.5)."""

from __future__ import annotations

import pytest
import torch

from othello_rl.config import Config
from othello_rl.nets import QNet, build_net, load_net, save_checkpoint


@pytest.mark.parametrize(
    ("n", "layers", "params"),
    [(4, 3, 76_339), (6, 4, 115_407), (8, 5, 158_019)],  # spec §4.5: "about 120k; about 160k on 8×8"
)
def test_parameter_count(n, layers, params):
    """Computed by hand: conv 3→64 (1,792) + (L−1) × conv 64→64 (36,928) + 1×1 conv (130) + linear."""
    net = QNet(n, channels=64, layers=layers)
    assert sum(p.numel() for p in net.parameters()) == params


@pytest.mark.parametrize("n", [4, 6, 8])
def test_forward_shape_and_range(n):
    q = QNet(n, channels=16, layers=2)(torch.randn(5, 3, n, n) * 10)
    assert q.shape == (5, n * n + 1)
    assert (q.abs() <= 1).all()


@pytest.mark.parametrize("n", [4, 6, 8])
def test_gradients_reach_every_parameter_for_every_seed(n):
    """Guards against a dead-at-birth network. With a ReLU after the 2-channel head conv, ~1 in 5
    initialisations had both channels negative on every input, so no gradient reached the trunk;
    50 seeds would then fail with probability > 0.9999."""
    x = torch.rand(16, 3, n, n).round()  # 0/1 planes, like engine.to_planes
    for seed in range(50):
        torch.manual_seed(seed)
        net = QNet(n, channels=64, layers=3)
        net(x).sum().backward()
        dead = [name for name, p in net.named_parameters() if p.grad is None or not p.grad.abs().sum()]
        assert not dead, f"seed {seed}: no gradient for {dead}"


def test_checkpoint_round_trip(tmp_path):
    cfg = Config(board=6, layers=2, channels=16)
    torch.manual_seed(0)
    net = build_net(cfg)
    path = tmp_path / "ckpt" / "g0000000.pt"
    save_checkpoint(path, net, games=0, grad_steps=0)
    loaded = load_net(path, cfg)
    x = torch.rand(3, 3, 6, 6)
    assert not loaded.training
    torch.testing.assert_close(loaded(x), net(x))
