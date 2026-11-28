"""scripts/analyze.py on synthetic logs: curves, figures, and the T1 table (spec §4.9)."""

from __future__ import annotations

import importlib.util
import json
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
    # Every run is cut at 4,000 games (the running seed), so "final" is eval points 0-4 everywhere
    # (all of them, since that's fewer than 10). Without the cut, the finished seeds would use 0-7.
    rise = np.mean([1 - np.exp(-i / 3) for i in range(5)])
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


# --- F4: cross-play (scripts/crossplay.py's files) ------------------------------------------------


def _crossplay_file(path, players, score, kind="within"):
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / "config.yaml").touch()  # rundir.find_runs only needs it to exist
    path.write_text(json.dumps({"kind": kind, "players": players, "score": score}))


def _antisymmetric(k, rng):
    m = rng.uniform(0.3, 0.7, (k, k))
    m = np.triu(m, 1) + np.tril(1 - m.T, -1)
    np.fill_diagonal(m, np.nan)
    return m


def _json(m):
    return [[None if np.isnan(x) else float(x) for x in row] for row in m]


@pytest.fixture
def crossplay_runs(tmp_path):
    """Two seeds of each condition, 4 checkpoints each, plus an across-run file of all 8 finals."""
    rng = np.random.default_rng(0)
    root = tmp_path / "runs"
    games = [1000, 2000, 3000, 4000]
    within = {}
    for target in ("env", "negamax"):
        for sampling in ("naive", "pool"):
            for seed in (0, 1):
                m = _antisymmetric(4, rng)
                run = f"{target}-{sampling}/seed{seed}"
                players = [{"run": run, "games": g} for g in games]
                _crossplay_file(root / "t" / run / "crossplay.json", players, _json(m))
                within[(target, sampling, seed)] = m
    runs = [f"{t}-{s}/seed{k}" for t in ("env", "negamax") for s in ("naive", "pool") for k in (0, 1)]
    finals = _antisymmetric(len(runs), rng)
    _crossplay_file(
        root / "t" / "crossplay_finals.json",
        [{"run": r, "games": 4000} for r in runs],
        _json(finals),
        "across",
    )
    return root, games, within, finals


def test_crossplay_mean_and_final_rows(crossplay_runs):
    root, games, within, _ = crossplay_runs
    results = analyze.load_crossplay("t", root)
    assert sorted(results) == [("env", "naive"), ("env", "pool"), ("negamax", "naive"), ("negamax", "pool")]
    g, stack = analyze.crossplay_mean(results[("env", "pool")])
    assert list(g) == games and stack.shape == (2, 4, 4)
    np.testing.assert_array_equal(
        stack.mean(axis=0), (within[("env", "pool", 0)] + within[("env", "pool", 1)]) / 2
    )
    rows = analyze.final_vs_earlier(g, stack)  # the final checkpoint's score, not the earlier one's
    assert [r["games"] for r in rows[1]] == games[:-1]
    assert [r["final_vs"] for r in rows[1]] == list(within[("env", "pool", 1)][-1, :-1])


def test_crossplay_mean_refuses_mismatched_checkpoints():
    a = {"players": [{"games": 1}, {"games": 2}], "score": [[None, 0.5], [0.5, None]]}
    b = {"players": [{"games": 1}, {"games": 3}], "score": [[None, 0.5], [0.5, None]]}
    with pytest.raises(ValueError, match="different checkpoints"):
        analyze.crossplay_mean([a, b])


def test_crossplay_figures_are_written(crossplay_runs, tmp_path):
    root, *_ = crossplay_runs
    analyze.fig_crossplay("t", root, tmp_path)
    analyze.fig_crossplay_finals("t", root, tmp_path)
    for name in ("f4_crossplay", "f4_final_vs_earlier", "f4_finals"):
        pdf = (tmp_path / f"{name}.pdf").read_bytes()
        assert len(pdf) > 1000 and b"/CreationDate" not in pdf


def test_crossplay_figures_skip_without_files(tmp_path):
    with pytest.raises(analyze.Skip, match="make crossplay"):
        analyze.fig_crossplay("t", tmp_path, tmp_path)
    with pytest.raises(analyze.Skip, match="make crossplay"):
        analyze.fig_crossplay_finals("t", tmp_path, tmp_path)


def test_colour_scale_is_symmetric_and_rounded_up():
    assert analyze._vlim(np.array([[np.nan, 0.81], [0.19, np.nan]])) == pytest.approx(0.35)  # 0.31 up
    assert analyze._vlim(np.array([[0.5, 0.5]])) == pytest.approx(0.05)  # never narrower than ±0.05
    assert analyze._vlim(np.array([[0.2, 0.6]])) == pytest.approx(0.30)  # exactly on a step stays there
    assert analyze._cond_name(("negamax", "pool")) == analyze._cond_name("negamax-pool") == "N-pool"
