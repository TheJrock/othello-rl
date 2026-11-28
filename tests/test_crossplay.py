"""scripts/crossplay.py: checkpoint choice, both kinds of round robin, skipping and recomputing
(spec §4.7, E3). The runs are fake: randomly initialised 4×4 nets saved as checkpoints."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from othello_rl import evaluate, nets, rundir
from othello_rl.config import Config
from othello_rl.engine import initial
from othello_rl.evaluate import opening_bank, save_states
from othello_rl.players import NetPlayer

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("crossplay", ROOT / "scripts" / "crossplay.py")
crossplay = importlib.util.module_from_spec(_spec)
sys.modules["crossplay"] = crossplay
_spec.loader.exec_module(crossplay)

GAMES = 5  # each fake run trains to 5 "games", with a checkpoint at every game


def _ckpts(games):
    return [(g, Path(f"g{g:07d}.pt")) for g in games]


def test_evenly_spaced_matches_the_spec_examples():
    pick = crossplay.evenly_spaced
    e1 = [g for g, _ in pick(_ckpts(range(0, 30_001, 1_000)), 15)]
    assert e1 == [g * 1000 for g in (1, 3, 5, 7, 9, 11, 13, 16, 18, 20, 22, 24, 26, 28, 30)]
    e2 = [g for g, _ in pick(_ckpts(range(0, 400_001, 10_000)), 15)]
    assert e2 == [g * 10_000 for g in (1, 4, 7, 9, 12, 15, 18, 21, 23, 26, 29, 32, 34, 37, 40)]
    e6 = [g for g, _ in pick(_ckpts(range(0, 1_000_001, 20_000)), 15)]
    assert e6[:4] == [20_000, 100_000, 160_000, 240_000] and e6[-1] == 1_000_000 and len(set(e6)) == 15
    assert [g for g, _ in pick(_ckpts([0, 10, 20]), 15)] == [10, 20]  # all the trained ones, never 0


def _net(cfg, seed):
    torch.manual_seed(seed)
    return nets.build_net(cfg)


def _make_run(root, openings, target, sampling, salt, last=GAMES):
    cfg = Config(
        exp="t", board=4, channels=4, layers=1, games=GAMES, eval_every=1, pool_groups=1, target=target,
        sampling=sampling, openings=str(openings), solver_table=None, runs_root=str(root),
    )  # fmt: skip
    run = rundir.prepare_run_dir(cfg)
    for g in range(last + 1):
        nets.save_checkpoint(rundir.checkpoint_path(run, g), _net(cfg, 100 * salt + g))
    return run


@pytest.fixture
def exp(tmp_path):
    """Three finished runs and one that stopped at game 3."""
    openings = tmp_path / "openings.json"
    save_states(opening_bank(4, plies=2, count=None, seed=0), openings)  # 12 openings -> 24 games a pair
    root = tmp_path / "runs"
    conds = [("env", "naive"), ("env", "pool"), ("negamax", "naive")]
    runs = [_make_run(root, openings, t, s, salt) for salt, (t, s) in enumerate(conds)]
    unfinished = _make_run(root, openings, "negamax", "pool", salt=3, last=3)
    return root, runs, unfinished


def _main(root, *extra):
    crossplay.main(["--exp", "t", "--runs-root", str(root), "--checkpoints", "3", "--jobs", "1", *extra])


def test_within_run_round_robin(exp, capsys):
    root, runs, unfinished = exp
    _main(root)
    out = capsys.readouterr().out
    assert f"skip {unfinished}: unfinished" in out
    assert "crossplay_finals.json: 1 of 4 runs unfinished" in out
    assert not (unfinished / rundir.CROSSPLAY_FILE).exists()
    assert not (root / "t" / rundir.CROSSPLAY_FINALS_FILE).exists()

    result = json.loads((runs[0] / rundir.CROSSPLAY_FILE).read_text())
    assert [p["games"] for p in result["players"]] == [1, 3, 5]  # 3 evenly spaced, untrained left out
    assert result["kind"] == "within" and result["games_per_pair"] == 24
    m = crossplay.score_matrix(result)
    assert np.isnan(np.diag(m)).all() and (m + m.T)[~np.eye(3, dtype=bool)] == pytest.approx(1)

    # Exactly evaluate.crossplay_matrix on the same checkpoints.
    cfg = rundir.load_run_config(runs[0])
    players = [NetPlayer(nets.load_net(rundir.checkpoint_path(runs[0], g), cfg)) for g in (1, 3, 5)]
    openings = evaluate.load_states(cfg.openings)
    direct = evaluate.crossplay_matrix(players, openings, np.random.default_rng(crossplay.RNG_SEED))
    np.testing.assert_array_equal(m, direct)

    # The start-position matrix: the same checkpoints, one game a colour from engine.initial.
    start = result["start"]
    assert start["players"] == [result["players"]] and start["games_per_pair"] == 2
    from_start = evaluate.crossplay_matrix(players, [initial(4)], np.random.default_rng(crossplay.RNG_SEED))
    np.testing.assert_array_equal(crossplay.score_matrix(result, start=True), from_start)
    assert set(from_start[~np.eye(3, dtype=bool)]) <= {0, 0.25, 0.5, 0.75, 1}


def _finish(run, last=GAMES, salt=5):
    cfg = rundir.load_run_config(run)
    for g in range(len(rundir.list_checkpoints(run)), last + 1):
        nets.save_checkpoint(rundir.checkpoint_path(run, g), _net(cfg, 100 * salt + g))


def test_across_start_matrix_averages_each_runs_last_checkpoints(exp, capsys):
    root, runs, unfinished = exp
    _finish(unfinished)
    runs = [*runs, unfinished]
    _main(root, "--kind", "across", "--start-checkpoints", "3")
    result = json.loads((root / "t" / rundir.CROSSPLAY_FINALS_FILE).read_text())
    start = result["start"]
    assert [[p["games"] for p in level] for level in start["players"]] == [[3] * 4, [4] * 4, [5] * 4]
    assert start["games_per_pair"] == 6 and result["format"] == crossplay.FORMAT

    def matrix(games):
        players = [
            NetPlayer(nets.load_net(rundir.checkpoint_path(r, games), rundir.load_run_config(r)))
            for r in runs
        ]
        return evaluate.crossplay_matrix(players, [initial(4)], np.random.default_rng(crossplay.RNG_SEED))

    expected = np.mean([matrix(g) for g in (3, 4, 5)], axis=0)
    np.testing.assert_allclose(crossplay.score_matrix(result, start=True), expected)
    assert "from the starting position, each run's last 3 checkpoints" in capsys.readouterr().out

    # Another --start-checkpoints changes the across file only; the within files don't use it.
    _main(root)
    capsys.readouterr()
    _main(root, "--start-checkpoints", "2")
    out = capsys.readouterr().out
    assert "1 round robins" in out and out.count("up to date") == 4

    # More than a run has: every trained checkpoint, never the untrained one.
    _main(root, "--kind", "across", "--start-checkpoints", "10")
    result = json.loads((root / "t" / rundir.CROSSPLAY_FINALS_FILE).read_text())
    assert [level[0]["games"] for level in result["start"]["players"]] == [1, 2, 3, 4, 5]


def test_files_of_an_older_format_are_recomputed(exp, capsys):
    root, runs, _ = exp
    _main(root, "--kind", "within")
    old = json.loads((runs[0] / rundir.CROSSPLAY_FILE).read_text())
    del old["format"], old["start"]  # as the first version of the script wrote it
    (runs[0] / rundir.CROSSPLAY_FILE).write_text(json.dumps(old))
    capsys.readouterr()
    _main(root, "--kind", "within")
    out = capsys.readouterr().out
    assert "1 round robins" in out and out.count("up to date") == 2
    assert "start" in json.loads((runs[0] / rundir.CROSSPLAY_FILE).read_text())


def test_skips_what_is_up_to_date_and_recomputes_what_changed(exp, capsys):
    root, runs, unfinished = exp
    _main(root)
    before = {r: (r / rundir.CROSSPLAY_FILE).read_text() for r in runs}
    capsys.readouterr()

    _main(root)  # nothing changed
    assert capsys.readouterr().out.count("up to date") == 3
    assert {r: (r / rundir.CROSSPLAY_FILE).read_text() for r in runs} == before

    # The fourth run finishes: its round robin and the across one are new, the rest stay.
    cfg = rundir.load_run_config(unfinished)
    for g in (4, 5):
        nets.save_checkpoint(rundir.checkpoint_path(unfinished, g), _net(cfg, 500 + g))
    _main(root)
    out = capsys.readouterr().out
    assert out.count("up to date") == 3 and "2 round robins" in out
    finals = json.loads((root / "t" / rundir.CROSSPLAY_FINALS_FILE).read_text())
    assert [p["run"] for p in finals["players"]] == [
        "env-naive/seed0", "env-pool/seed0", "negamax-naive/seed0", "negamax-pool/seed0",
    ]  # fmt: skip
    assert {p["games"] for p in finals["players"]} == {GAMES}

    # A retrained run (same games, different weights) is recognised by its checkpoint hash.
    nets.save_checkpoint(rundir.checkpoint_path(runs[1], GAMES), _net(cfg, 999))
    _main(root)
    out = capsys.readouterr().out
    assert "2 round robins" in out and f"{runs[1] / rundir.CROSSPLAY_FILE}:" in out
    assert out.count("up to date") == 3

    _main(root, "--force")
    assert "5 round robins" in capsys.readouterr().out


def test_kind_and_checkpoints_options(exp, capsys):
    root, runs, _ = exp
    _main(root, "--kind", "across")  # within skipped entirely, across refused (one run unfinished)
    out = capsys.readouterr().out
    assert "unfinished" in out and not any((r / rundir.CROSSPLAY_FILE).exists() for r in runs)
    _main(root, "--kind", "within")
    crossplay.main(["--exp", "t", "--runs-root", str(root), "--checkpoints", "5", "--jobs", "1"])
    assert len(json.loads((runs[0] / rundir.CROSSPLAY_FILE).read_text())["players"]) == 5


def test_overwriting_a_run_deletes_its_crossplay(exp):
    root, runs, _ = exp
    _main(root)
    cfg = rundir.load_run_config(runs[0])
    (runs[0] / rundir.LOG_FILE).write_text("")  # prepare_run_dir only refuses runs with a log
    rundir.prepare_run_dir(cfg, overwrite=True)
    assert not (runs[0] / rundir.CROSSPLAY_FILE).exists()


def test_worker_processes(exp):
    """--jobs > 1 runs each round robin in its own process, from the command line as `make` does."""
    root, runs, _ = exp
    cmd = [sys.executable, str(ROOT / "scripts" / "crossplay.py"), "--exp", "t", "--runs-root", str(root)]
    done = subprocess.run(
        [*cmd, "--checkpoints", "3", "--jobs", "2"], check=False, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert all((r / rundir.CROSSPLAY_FILE).exists() for r in runs)


def _result(kind, players, score, start_score=None, levels=1):
    start = {"players": [players] * levels, "score": score if start_score is None else start_score}
    return {"kind": kind, "players": players, "score": score, "start": start}


def test_describe_within():
    players = [{"run": "env-naive/seed0", "games": g} for g in (1000, 2000, 3000)]
    start = [[None, 0.5, 0.25], [0.5, None, 0.0], [0.75, 1.0, None]]
    result = _result("within", players, [[None, 0.5, 0.2], [0.5, None, 0.6], [0.8, 0.4, None]], start)
    assert crossplay.describe(result) == (
        "3 checkpoints 1k-3k; against the final one the best earlier checkpoint (2k) scores 0.600, "
        "the first (1k) 0.200; from the start the final one scores 0.875 against the earlier ones"
    )
    one = _result("within", players[:1], [[None]])
    assert crossplay.describe(one) == "one checkpoint, nothing to play"


def test_condition_blocks():
    players = [{"run": r, "games": 9} for r in ("a/seed0", "a/seed1", "b/seed0", "b/seed1")]
    score = [
        [None, 0.5, 0.9, 0.7],
        [0.5, None, 0.6, 0.8],
        [0.1, 0.4, None, 0.5],
        [0.3, 0.2, 0.5, None],
    ]
    start = [[None, 0.5, 0.0, 0.5], [0.5, None, 0.25, 0.25], [1.0, 0.75, None, 0.5], [0.5, 0.75, 0.5, None]]
    result = _result("across", players, score, start, levels=5)
    names, blocks = crossplay.condition_blocks(result)
    assert names == ["a", "b"]
    np.testing.assert_allclose(blocks, [[np.nan, 0.75], [0.25, np.nan]])
    _, from_start = crossplay.condition_blocks(result, start=True)
    np.testing.assert_allclose(from_start, [[np.nan, 0.25], [0.75, np.nan]])
    text = crossplay.describe(result)
    assert "0.750" in text and "each run's last 5 checkpoints" in text.splitlines()[4]
