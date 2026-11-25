"""scripts/analyze.py on synthetic logs: curves, figures, and the T1 table (spec §4.9)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "analyze", Path(__file__).resolve().parents[1] / "scripts" / "analyze.py"
)
analyze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze)


def _log(points, level, seed, endgame=True):
    """A run whose metrics rise to `level`, with a little seed-specific noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(points):
        x = level * (1 - np.exp(-i / 3)) + 0.01 * rng.standard_normal()
        row = {
            "games": 1000 * i,
            "accuracy_4x4": x,
            "score_positional": x,
            "score_random": x,
            "score_greedy": x,
        }
        if endgame:
            row["endgame_accuracy"] = x
        if i:
            row["score_selfplay"] = 0.5 + 0.01 * rng.standard_normal()
        rows.append(row)
    return rows


@pytest.fixture
def logs():
    """Pool better than naive by 0.10 under env and by 0.02 under negamax; one env-pool seed is
    still running (5 of 8 eval points)."""
    level = {
        ("env", "naive"): 0.6,
        ("env", "pool"): 0.7,
        ("negamax", "naive"): 0.7,
        ("negamax", "pool"): 0.72,
    }
    out = {cond: [_log(8, lv, seed) for seed in range(3)] for cond, lv in level.items()}
    out[("env", "pool")][2] = out[("env", "pool")][2][:5]
    return out


def test_curve_uses_points_every_seed_has(logs):
    games, y = analyze.curve(logs[("env", "pool")], "score_positional")
    assert list(games) == [0, 1000, 2000, 3000, 4000]  # the running seed shortens the curve
    assert y.shape == (3, 5)
    games, _ = analyze.curve(logs[("env", "pool")], "score_selfplay")
    assert games[0] == 1000  # no self-play score at game 0


def test_figures_are_written(logs, tmp_path):
    analyze.fig_accuracy_4x4(logs, tmp_path)
    analyze.fig_scores_6x6(logs, tmp_path)
    analyze.fig_selfplay_vs_external(logs, tmp_path, 6)
    analyze.fig_endgame_6x6(logs, tmp_path)
    for name in ("f1_accuracy", "f2_scores", "f2_scores_appendix", "f3_selfplay_vs_external", "f5_endgame"):
        assert (tmp_path / f"{name}.pdf").stat().st_size > 1000


def test_redrawing_the_same_data_gives_the_same_bytes(logs, tmp_path):
    # Committed figures must only show up in `git status` when their content changes (no timestamp).
    first = tmp_path / "f5_endgame.pdf"
    analyze.fig_endgame_6x6(logs, tmp_path)
    before = first.read_bytes()
    analyze.fig_endgame_6x6(logs, tmp_path)
    assert first.read_bytes() == before
    assert b"/CreationDate" not in before  # the timestamp would differ on any later redraw


def test_missing_metric_is_a_skip_not_a_crash(tmp_path):
    no_endgame = {("env", "naive"): [_log(4, 0.5, 0, endgame=False)]}
    with pytest.raises(analyze.Skip):
        analyze.fig_endgame_6x6(no_endgame, tmp_path)


def test_t1_gaps_and_h4(logs, tmp_path):
    rows = {label: cells for _, label, cells in analyze.t1_rows(analyze.summaries(logs, "score_positional"))}
    final = [c for c, _, _ in analyze.T1_COLUMNS].index("final")
    env_gap, se = rows["pool - naive (env)"][final]
    neg_gap, _ = rows["pool - naive (negamax)"][final]
    # Every run is cut at 4,000 games (the running seed), so "final" is eval points 2-4 everywhere.
    rise = np.mean([1 - np.exp(-i / 3) for i in (2, 3, 4)])
    assert env_gap == pytest.approx(0.10 * rise, abs=0.015) and se > 0
    assert neg_gap == pytest.approx(0.02 * rise, abs=0.015)
    assert rows["H4: env gap - negamax gap"][final][0] == pytest.approx(env_gap - neg_gap)


def test_t1_files(logs, tmp_path):
    analyze.table_summary(logs, tmp_path / "t1", "score_positional")
    tex = (tmp_path / "t1" / "t1_summary.tex").read_text()
    assert tex.startswith(r"\begin{tabular}") and tex.rstrip().endswith(r"\end{tabular}")
    assert r"\envT{}-\poolS{}" in tex and r"H4: \envT{} gap" in tex
    assert tex.count(r"\\") == 1 + 4 + 3  # header, 4 conditions, 2 gaps + H4
    assert "cut at 4,000 games" in tex  # says so when a running seed shortens every run
    assert (tmp_path / "t1" / "t1_summary.txt").exists()


def test_one_seed_has_no_spread(tmp_path):
    one = {("env", "naive"): [_log(6, 0.5, 0)], ("env", "pool"): [_log(6, 0.6, 1)]}
    rows = analyze.t1_rows(analyze.summaries(one, "score_positional"))
    for kind, _, cells in rows:
        assert all(np.isnan(spread) for _, spread in cells)  # no std or SE from a single seed
