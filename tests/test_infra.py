"""Config and run-directory plumbing (implemented; these should pass from day one)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from othello_rl import evaluate, nets, rundir
from othello_rl.config import SAMPLINGS, Config, load_config, parse_cli
from othello_rl.players import NetPlayer

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("name,board", [("4x4.yaml", 4), ("6x6.yaml", 6), ("8x8.yaml", 8)])
def test_shipped_configs_load(name, board):
    cfg = load_config(CONFIGS / name)
    assert cfg.board == board
    assert cfg.action_size == board * board + 1


@pytest.mark.parametrize("name", ["4x4.yaml", "6x6.yaml", "8x8.yaml"])
def test_shipped_configs_can_start_a_run(name, monkeypatch):
    """Everything a run loads before its first eval point: the data files the config names exist, and
    every metric it turns on is implemented. (The 6×6 pilots once died at start-up on a missing
    endgame set.)"""
    monkeypatch.chdir(CONFIGS.parent)  # config paths are relative to the repo root
    cfg = load_config(CONFIGS / name)
    if cfg.solver_table and not Path(cfg.solver_table).exists():
        pytest.skip(f"{cfg.solver_table} is built by `make data` (not in git)")
    ctx = evaluate.EvalContext.from_config(cfg)
    assert ctx.openings
    if ctx.endgame_set is not None:
        evaluate.endgame_accuracy(NetPlayer(nets.build_net(cfg)), ctx.endgame_set[:1])


def test_cli_overrides_yaml():
    cfg = parse_cli(
        [
            "--config",
            str(CONFIGS / "6x6.yaml"),
            "--exp",
            "e2-6x6",
            "--target",
            "negamax",
            "--sampling",
            "pool",
            "--seed",
            "2",
            "--lr",
            "1e-4",
            "--endgame-set",
            "none",
        ]
    )
    assert (cfg.board, cfg.target, cfg.sampling, cfg.seed) == (6, "negamax", "pool", 2)
    assert cfg.lr == pytest.approx(1e-4)
    assert cfg.endgame_set is None
    assert cfg.replay_capacity == 1_280_000  # untouched YAML value survives
    assert cfg.run_dir == Path("runs/e2-6x6/negamax-pool/seed2")


def test_dash_and_underscore_flags():
    assert parse_cli(["--parallel_games", "64"]).parallel_games == 64
    assert parse_cli(["--parallel-games", "64"]).parallel_games == 64


def test_unknown_yaml_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("board: 4\nlearning_rate: 0.1\n")
    with pytest.raises(KeyError, match="learning_rate"):
        load_config(path)


def test_yaml_scientific_notation_is_coerced(tmp_path):
    path = tmp_path / "sci.yaml"
    path.write_text("lr: 3e-4\naugment: 'false'\n")  # YAML reads both of these as strings
    cfg = load_config(path)
    assert cfg.lr == pytest.approx(3e-4)
    assert cfg.augment is False


@pytest.mark.parametrize(
    "override",
    [
        {"target": "minimax"},
        {"sampling": "league"},
        {"board": 5},
        {"pool_p_latest": 1.5},
        {"parallel_games": 100, "pool_groups": 8},
        {"eval_every": 1_004, "pool_groups": 8},  # a cut-short round must still split into groups
        {"games": 30_004, "pool_groups": 8},
        {"eps_start": 0.01, "eps_end": 0.05},
    ],
)
def test_invalid_values_are_rejected(override):
    with pytest.raises(ValueError):
        Config(**override)


def test_checkpoints_sort_by_games(tmp_path):
    for games in (10_000, 0, 5_000):
        path = rundir.checkpoint_path(tmp_path, games)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert rundir.checkpoint_path(tmp_path, 5_000).name == "g0005000.pt"
    assert [g for g, _ in rundir.list_checkpoints(tmp_path)] == [0, 5_000, 10_000]


def test_run_dir_lifecycle(tmp_path):
    cfg = Config(runs_root=str(tmp_path), exp="t", target="negamax", sampling="pool", seed=3)
    run = rundir.prepare_run_dir(cfg)
    assert run == tmp_path / "t" / "negamax-pool" / "seed3"
    assert (run / "meta.json").exists()
    assert rundir.load_run_config(run) == cfg
    assert rundir.run_condition(run) == ("negamax", "pool", 3)

    with rundir.JsonlLogger(run / rundir.LOG_FILE) as log:
        log.log({"games": 0, "score": np.float32(0.5), "ci": np.array([0.4, 0.6])})
    assert rundir.read_log(run) == [
        {"games": 0, "score": 0.5, "ci": [pytest.approx(0.4), pytest.approx(0.6)]}
    ]
    assert rundir.find_runs(tmp_path) == [run]

    with pytest.raises(FileExistsError):
        rundir.prepare_run_dir(cfg)
    rundir.checkpoint_path(run, 0).touch()
    rundir.prepare_run_dir(cfg, overwrite=True)
    assert not (run / rundir.LOG_FILE).exists()
    assert rundir.list_checkpoints(run) == []


def test_debug_sampling_is_accepted_but_never_swept():
    """sampling=random is debugging ladder rung 1 only (spec §4.6): valid for a single run, but not
    one of the 2×2 values that scripts/sweep.py iterates over."""
    assert Config(sampling="random").sampling == "random"
    assert SAMPLINGS == ("naive", "pool")
