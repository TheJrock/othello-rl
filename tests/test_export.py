"""ONNX export matches PyTorch (spec §5.4). The web side repeats this check in web/tests/model.parity.test.ts."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import torch

from othello_rl.export import export_onnx
from othello_rl.nets import QNet


def test_onnx_matches_torch(tmp_path, rng):
    torch.manual_seed(0)
    net = QNet(6, channels=16, layers=2).eval()
    path = tmp_path / "net.onnx"
    export_onnx(net, 6, path)

    x = rng.random((8, 3, 6, 6), dtype=np.float32)
    with torch.no_grad():
        want = net(torch.from_numpy(x)).numpy()
    got = ort.InferenceSession(str(path)).run(["q"], {"planes": x})[0]
    assert got.shape == (8, 37)
    np.testing.assert_allclose(got, want, atol=1e-5)


def test_qnet_output_range():
    net = QNet(4, channels=8, layers=2).eval()
    with torch.no_grad():
        q = net(torch.randn(5, 3, 4, 4) * 10)
    assert q.shape == (5, 17)
    assert (q.abs() <= 1).all()
