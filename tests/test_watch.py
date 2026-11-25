"""scripts/watch.py on synthetic run directories: statuses, ETA, and the bars."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from othello_rl.config import Config

_spec = importlib.util.spec_from_file_location(
    "watch", Path(__file__).resolve().parents[1] / "scripts" / "watch.py"
)
watch = importlib.util.module_from_spec(_spec)
sys.modules["watch"] = watch  # dataclasses look their module up here
_spec.loader.exec_module(watch)


def make_run(root, exp, cond, rows, board=6, games=400_000, eval_every=10_000, stdout="", partial=""):
    run = root / exp / cond / "seed0"
    run.mkdir(parents=True)
    Config(board=board, games=games, eval_every=eval_every).save(run / "config.yaml")
    if rows is not None:
        (run / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows) + partial)
    (run / "stdout.log").write_text(stdout)
    return run


def rows_to(games, step=10_000, secs_per_step=100.0):
    return [
        {
            "games": g,
            "wall_time": i * secs_per_step,
            "score_positional": 0.5 + g / 1e6,
            "accuracy_4x4": 0.7,
            **({"score_selfplay": 0.52} if g else {}),
        }
        for i, g in enumerate(range(0, games + 1, step))
    ]


def test_running_run_progress_metrics_and_eta(tmp_path):
    run = make_run(tmp_path, "p", "env-naive", rows_to(100_000))
    st = watch.run_status(run, now=os.path.getmtime(run / "log.jsonl") + 5)
    assert (st.state, st.games, st.budget) == ("running", 100_000, 400_000)
    assert st.fraction == 0.25
    assert st.metric == pytest.approx(0.6) and st.selfplay == 0.52
    assert st.rate == pytest.approx(100.0)  # 10k games per 100 s
    assert watch._status_text(st) == "ETA 50m"  # 300k games left at 100 games/s


def test_rate_follows_recent_pace(tmp_path):
    rows = rows_to(50_000) + [
        {"games": 60_000 + 10_000 * i, "wall_time": 500 + 200 * (i + 1)} for i in range(3)
    ]
    assert watch.recent_rate(rows) == pytest.approx(50.0)  # the last 3 intervals took 200 s each


def test_partial_last_line_is_ignored(tmp_path):
    run = make_run(tmp_path, "p", "env-pool", rows_to(30_000), partial='{"games": 40000, "wall')
    assert watch.run_status(run, now=0).games == 30_000


def test_done_failed_starting_and_stalled(tmp_path):
    done = make_run(tmp_path, "p", "negamax-naive", rows_to(400_000))
    assert watch.run_status(done, now=1e12).state == "done"  # never "stalled" once finished
    failed = make_run(tmp_path, "p", "negamax-pool", None, stdout="Traceback (most recent call last):\n  ...")
    assert watch.run_status(failed, now=0).state == "failed"
    starting = make_run(tmp_path, "q", "env-naive", None)
    assert watch.run_status(starting, now=0).state == "starting"
    stalled = make_run(tmp_path, "q", "env-pool", rows_to(100_000))
    mtime = os.path.getmtime(stalled / "log.jsonl")
    assert watch.run_status(stalled, now=mtime + 60).state == "running"
    assert watch.run_status(stalled, now=mtime + 2 * 3600).state == "stalled"  # usual gap is 100 s


def test_bar_widths_and_eighths():
    assert watch.bar(0.0, 10) == "·" * 10
    assert watch.bar(1.0, 10) == "█" * 10
    assert watch.bar(0.25, 8) == "██" + "·" * 6  # no blank partial cell
    assert watch.bar(0.3, 10) == "███" + "·" * 7
    assert watch.bar(0.0625, 10) == "▋" + "·" * 9  # 0.625 of a cell = 5 eighths
    assert watch.bar(0.05, 10) == "▌" + "·" * 9  # half a cell = 4 eighths
    assert all(len(watch.bar(f / 100, 23)) == 23 for f in range(101))


def test_default_shows_only_experiments_still_going(tmp_path):
    make_run(tmp_path, "finished", "env-naive", rows_to(30_000), board=4, games=30_000, eval_every=1000)
    make_run(tmp_path, "going", "env-naive", rows_to(100_000))
    assert {s.exp for s in watch.collect(tmp_path, None, now=0)} == {"going"}
    assert {s.exp for s in watch.collect(tmp_path, ["finished"], now=0)} == {"finished"}


def test_render_labels_4x4_accuracy(tmp_path):
    make_run(tmp_path, "e1", "env-naive", rows_to(30_000, step=1000), board=4, games=30_000, eval_every=1000)
    text = watch.render(watch.collect(tmp_path, ["e1"], now=0), width=120, now=0)
    assert "acc" in text and "30k / 30k" in text and "done in" in text
