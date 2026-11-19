"""ONNX export, web demo data, and Python→TypeScript parity fixtures (spec §5).

Fixture formats (mirrored by web/src/types.ts):
    web/tests/fixtures/engine.json   [{"n": 6, "plies": [{"board", "black_to_move", "legal": [int], "action",
                                       "next_board", "next_black_to_move", "reward", "done"}, ...]}, ...]
    web/tests/fixtures/model.json    {"n": 6, "model": "models/final.onnx",
                                      "cases": [{"board", "black_to_move", "planes": [float], "q": [float]}, ...]}
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from .config import Config

SNAPSHOT_LABELS = {"untrained": "Untrained", "early": "Early", "mid": "Midway", "final": "Final"}


def export_onnx(net: torch.nn.Module, n: int, path: str | Path) -> None:
    """torch.onnx.export of the net in eval mode: input "planes" (B, 3, n, n) float32, output "q"
    (B, n² + 1), dynamic batch axis. tests/test_export.py checks onnxruntime against torch."""
    raise NotImplementedError("export.export_onnx")


def select_snapshots(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Demo snapshots chosen by skill milestone (spec §5.3), as {id: games}:
    untrained = 0; early = first row with score_random ≥ 0.60; mid = first row with
    score_positional ≥ 0.50; final = last row."""
    raise NotImplementedError("export.select_snapshots")


def write_engine_fixtures(path: str | Path, n: int = 6, games: int = 500, seed: int = 0) -> None:
    """Random games with every ply recorded (format above) for web/tests/engine.parity.test.ts."""
    raise NotImplementedError("export.write_engine_fixtures")


def write_model_fixtures(
    net: torch.nn.Module, path: str | Path, n: int = 6, count: int = 100, seed: int = 0
) -> None:
    """Positions with their planes and Q-vectors (format above) for web/tests/model.parity.test.ts."""
    raise NotImplementedError("export.write_model_fixtures")


def load_teaching_positions(path: str | Path) -> list[dict[str, Any]]:
    """Hand-picked demo positions (data/README.md format)."""
    return json.loads(Path(path).read_text())


def write_manifest(
    path: str | Path,
    cfg: Config,
    run_dir: str | Path,
    snapshots: dict[str, int],
    rows: Sequence[dict[str, Any]],
    teaching_ids: Sequence[str],
) -> None:
    """web/public/data/manifest.json in the spec §5.2 format (types.ts: Manifest)."""
    raise NotImplementedError("export.write_manifest")


def write_curves(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    """Learning curves for the demo: games vs score_random / score_greedy / score_positional /
    score_selfplay (the flat line of F3)."""
    raise NotImplementedError("export.write_curves")


def write_heatmaps(
    path: str | Path, run_dir: str | Path, cfg: Config, teaching: Sequence[dict[str, Any]]
) -> None:
    """Q-values of the legal moves for every checkpoint × teaching position, precomputed so the demo's
    checkpoint slider needs no inference."""
    raise NotImplementedError("export.write_heatmaps")
