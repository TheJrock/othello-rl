"""Backfilled metrics: rundir's backfill file and scripts/backfill.py (6×6 endgame accuracy)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from othello_rl import rundir
from othello_rl.config import Config
from othello_rl.evaluate import load_endgame_set, opening_bank, save_states
from othello_rl.train import train

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("backfill", ROOT / "scripts" / "backfill.py")
backfill = importlib.util.module_from_spec(_spec)
sys.modules["backfill"] = backfill
_spec.loader.exec_module(backfill)

METRIC = "endgame_accuracy"


def _write_log(run: Path, rows: list[dict]) -> None:
    (run / rundir.LOG_FILE).write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_read_log_merges_backfill_without_overriding_training(tmp_path):
    run = rundir.prepare_run_dir(Config(runs_root=str(tmp_path), exp="t"))
    _write_log(run, [{"games": 0, "a": 1}, {"games": 10, "a": 2, "b": 5}])
    rundir.write_backfill(run, [{"games": 0, "b": 7}, {"games": 10, "b": 9}, {"games": 99, "b": 0}])
    rundir.write_backfill(run, [{"games": 10, "c": 3}])  # a second metric merges in; b survives

    assert rundir.read_log(run) == [
        {"games": 0, "a": 1, "b": 7},  # backfill adds what training didn't log
        {"games": 10, "a": 2, "b": 5, "c": 3},  # training's own b wins
    ]  # and the backfill row for games=99 has no log row, so it's ignored
    assert rundir.read_log(run, backfill=False) == [{"games": 0, "a": 1}, {"games": 10, "a": 2, "b": 5}]
    assert not (run / "backfill.jsonl.tmp").exists()


def test_overwriting_a_run_deletes_its_backfill(tmp_path):
    cfg = Config(runs_root=str(tmp_path), exp="t")
    run = rundir.prepare_run_dir(cfg)
    _write_log(run, [{"games": 0}])
    rundir.write_backfill(run, [{"games": 0, "b": 1}])
    rundir.prepare_run_dir(cfg, overwrite=True)
    assert rundir.read_backfill(run) == {}


@pytest.fixture(scope="module")
def endgame_run(tmp_path_factory):
    """A tiny 6×6 run trained *with* the endgame set, so its log has the live values."""
    d = tmp_path_factory.mktemp("bf")
    save_states(opening_bank(6, 2, 3, 0), d / "openings.json")
    records = load_endgame_set(ROOT / "data" / "endgame6.jsonl")[:40]
    (d / "endgame.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    cfg = Config(
        exp="t", board=6, channels=8, layers=1, games=32, parallel_games=16, eval_every=16,
        batch_size=32, replay_capacity=1_000, target_update_every=5, solver_table=None,
        openings=str(d / "openings.json"), endgame_set=str(d / "endgame.jsonl"), runs_root=str(d / "runs"),
    )  # fmt: skip
    return train(cfg), records


def test_backfill_reproduces_what_training_logs(endgame_run, tmp_path):
    run, records = endgame_run
    live = [r[METRIC] for r in rundir.read_log(run)]
    assert len(live) == 3 and len(set(live)) > 1  # three eval points, and the metric moves

    # The same run as if it had trained without the set: drop the metric from a copy of its log.
    copy = tmp_path / "t" / run.parent.name / run.name
    copy.mkdir(parents=True)
    for name in ("config.yaml", "meta.json"):
        (copy / name).write_text((run / name).read_text())
    (copy / rundir.CKPT_DIR).symlink_to(run / rundir.CKPT_DIR)
    _write_log(copy, [{k: v for k, v in r.items() if k != METRIC} for r in rundir.read_log(run)])

    summary = backfill.backfill_run(copy, records)
    assert "3 checkpoints" in summary
    assert [r[METRIC] for r in rundir.read_log(copy)] == live  # exactly training's values
    assert METRIC not in (copy / rundir.LOG_FILE).read_text()  # the log itself is untouched


def test_backfill_skips_what_is_done_and_fills_what_is_missing(endgame_run, tmp_path):
    run, records = endgame_run
    assert "skip" in backfill.backfill_run(run, records)  # training logged it everywhere
    assert backfill.endgame_rows(run, records, force=True) == []  # force never overrides training

    copy = tmp_path / "t" / run.parent.name / run.name
    copy.mkdir(parents=True)
    (copy / "config.yaml").write_text((run / "config.yaml").read_text())
    (copy / rundir.CKPT_DIR).symlink_to(run / rundir.CKPT_DIR)
    _write_log(copy, [{"games": g} for g, _ in rundir.list_checkpoints(run)])

    rundir.write_backfill(copy, [{"games": 0, METRIC: 0.5}])  # as if a first pass stopped early
    assert "2 checkpoints" in backfill.backfill_run(copy, records)  # only the missing ones
    assert rundir.read_backfill(copy)[0][METRIC] == 0.5
    assert "skip" in backfill.backfill_run(copy, records)
    assert "3 checkpoints" in backfill.backfill_run(copy, records, force=True)
    assert rundir.read_backfill(copy)[0][METRIC] != 0.5


def test_backfill_skips_other_boards(tmp_path):
    run = rundir.prepare_run_dir(Config(runs_root=str(tmp_path), exp="t", board=4))
    assert "skip" in backfill.backfill_run(run, [])
