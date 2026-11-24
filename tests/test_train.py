"""Training loop: ε schedule, eval points, and tiny end-to-end runs (spec §4.5–4.6)."""

from __future__ import annotations

import pytest

from othello_rl import rundir, solver
from othello_rl.config import Config
from othello_rl.evaluate import opening_bank, save_states
from othello_rl.train import epsilon_at, eval_points, train


def test_epsilon_schedule():
    cfg = Config(games=1_000, eps_start=1.0, eps_end=0.05, eps_decay_frac=0.2)  # decays over 200 games
    assert epsilon_at(0, cfg) == 1.0
    assert epsilon_at(100, cfg) == pytest.approx(0.525)
    assert epsilon_at(200, cfg) == pytest.approx(0.05)
    assert epsilon_at(999, cfg) == pytest.approx(0.05)


def test_eval_points_land_on_multiples_and_the_end():
    assert eval_points(Config(games=3_000, eval_every=1_000)) == [0, 1_000, 2_000, 3_000]
    assert eval_points(Config(games=2_504, eval_every=1_000)) == [0, 1_000, 2_000, 2_504]


@pytest.fixture(scope="module")
def data_files(tmp_path_factory, table):
    d = tmp_path_factory.mktemp("data")
    save_states(opening_bank(4, 4, None, 0)[:3], d / "openings.json")
    solver.save_table(table, d / "solve4x4.npz")
    return d


def tiny_config(data_files, tmp_path, **overrides) -> Config:
    """Seconds, not minutes: a 1-layer, 8-channel net, 64 games in rounds of 16."""
    base = {
        "exp": "t",
        "board": 4,
        "channels": 8,
        "layers": 1,
        "games": 64,
        "parallel_games": 16,
        "pool_groups": 4,
        "eval_every": 32,
        "batch_size": 32,
        "replay_capacity": 1_000,
        "target_update_every": 5,
        "openings": str(data_files / "openings.json"),
        "solver_table": None,
        "runs_root": str(tmp_path),
    }
    return Config(**{**base, **overrides})


@pytest.mark.parametrize(("target", "sampling"), [("negamax", "naive"), ("env", "random")])
def test_tiny_run_logs_checkpoints_and_paces_gradient_steps(data_files, tmp_path, target, sampling):
    # Rounds of 24 games: the round before each eval point must be cut short to 8.
    cfg = tiny_config(
        data_files, tmp_path, target=target, sampling=sampling, parallel_games=24, eps_decay_frac=0.25
    )
    run = train(cfg)
    rows = rundir.read_log(run)
    assert [r["games"] for r in rows] == [0, 32, 64]
    assert [g for g, _ in rundir.list_checkpoints(run)] == [0, 32, 64]
    assert [r["epsilon"] for r in rows] == [1.0, cfg.eps_end, cfg.eps_end]  # what the learner used
    assert "loss" not in rows[0] and "score_selfplay" not in rows[0]  # untrained snapshot
    assert {"loss", "grad_norm", "learner_score", "score_selfplay", "score_positional"} <= rows[1].keys()
    # One gradient step per 16 new transitions (the buffer passes batch_size in the first round).
    transitions = sum(r["transitions_per_game"] * 32 for r in rows[1:])  # per-game means × 32 games
    assert rows[-1]["grad_steps"] == round(transitions) // cfg.transitions_per_grad_step
    assert rows[-1]["buffer"] == min(round(transitions), cfg.replay_capacity)
    for r in rows[1:]:
        if sampling == "naive":  # negamax, both sides stored (D2): one transition per ply
            assert r["transitions_per_game"] == r["plies_per_game"]
        else:  # rung 1: only the learner's real moves, about half the plies
            assert r["transitions_per_game"] < 0.6 * r["plies_per_game"]


def test_tiny_run_logs_4x4_solver_metrics(data_files, tmp_path):
    cfg = tiny_config(data_files, tmp_path, solver_table=str(data_files / "solve4x4.npz"), games=32)
    rows = rundir.read_log(train(cfg))
    for r in rows:
        assert 0.0 <= r["accuracy_4x4"] <= 1.0
        assert -1.0 <= r["q_corr_4x4"] <= 1.0
        assert r["q_mse_4x4"] >= 0.0


def test_same_seed_same_log(data_files, tmp_path):
    def log(exp):
        rows = rundir.read_log(train(tiny_config(data_files, tmp_path, exp=exp, games=32)))
        return [{k: v for k, v in r.items() if k != "wall_time"} for r in rows]

    assert log("a") == log("b")


def test_refuses_to_overwrite_a_run(data_files, tmp_path):
    cfg = tiny_config(data_files, tmp_path, games=32)
    train(cfg)
    with pytest.raises(FileExistsError):
        train(cfg)
