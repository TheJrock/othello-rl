"""Train one run: python -m othello_rl.train --config configs/4x4.yaml --target env --sampling naive --seed 0

Writes runs/<exp>/<target>-<sampling>/seed<k>/ (see rundir.py). Never plots (spec §4.1).
"""

from __future__ import annotations

import argparse
import copy
import random
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from . import evaluate, nets, pool, rundir, selfplay, targets
from .config import Config, add_config_args, config_from_args
from .players import NetPlayer
from .replay import ReplayBuffer


def seed_everything(seed: int) -> np.random.Generator:
    """Seed `random`, numpy and torch; return the numpy Generator the run passes everywhere."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def epsilon_at(games_played: int, cfg: Config) -> float:
    """Linear from cfg.eps_start to cfg.eps_end over the first cfg.eps_decay_frac · cfg.games games,
    then constant at cfg.eps_end."""
    progress = min(1.0, games_played / (cfg.eps_decay_frac * cfg.games))
    return (1.0 - progress) * cfg.eps_start + progress * cfg.eps_end  # exactly eps_end once decayed


def eval_points(cfg: Config) -> list[int]:
    """Games counts at which to checkpoint and evaluate: 0 (the demo's "Untrained" snapshot), every
    multiple of cfg.eval_every, and cfg.games itself."""
    points = list(range(0, cfg.games, cfg.eval_every))
    return points + [cfg.games]


def _format_row(row: dict[str, Any]) -> str:
    """One progress line for stdout (scripts/sweep.py sends it to <run_dir>/stdout.log)."""
    keys = [
        ("games", "g", "{:d}"),
        ("grad_steps", "steps", "{:d}"),
        ("epsilon", "eps", "{:.2f}"),
        ("loss", "loss", "{:.4f}"),
        ("score_random", "rand", "{:.2f}"),
        ("score_greedy", "greedy", "{:.2f}"),
        ("score_positional", "pos", "{:.2f}"),
        ("score_selfplay", "self", "{:.2f}"),
        ("accuracy_4x4", "acc", "{:.3f}"),
        ("q_corr_4x4", "qcorr", "{:.3f}"),
        ("wall_time", "t", "{:.0f}s"),
    ]
    return "  ".join(f"{short}={fmt.format(row[k])}" for k, short, fmt in keys if k in row)


def train(cfg: Config, overwrite: bool = False) -> Path:
    """Run one training job and return its run directory (spec §4.5–4.6).

    Each round plays up to cfg.parallel_games games against the sampler's opponents, adds their
    transitions to the replay buffer, and then takes one gradient step per
    cfg.transitions_per_grad_step new transitions. The last round before an eval point is cut short
    so that eval points land exactly on multiples of cfg.eval_every (checkpoint g0001000.pt, not
    g0001024.pt). At every eval point (game 0 included): save a checkpoint, evaluate it, log one
    JSONL row, and hand the checkpoint to the sampler (the pool's snapshots).

    Training stats in a row are means since the previous eval point: per game for plies_per_game,
    transitions_per_game and learner_score, per gradient step for loss, mean_q, mean_target and
    grad_norm. The game-0 row has none of them. `epsilon` is the ε the learner actually played the
    last round with (eps_start at game 0).
    """
    run_dir = rundir.prepare_run_dir(cfg, overwrite)
    rng = seed_everything(cfg.seed)
    if cfg.torch_threads:
        torch.set_num_threads(cfg.torch_threads)

    online = nets.build_net(cfg).to(cfg.device)
    target_net = copy.deepcopy(online).requires_grad_(False)
    opt = torch.optim.Adam(online.parameters(), lr=cfg.lr)
    learner = NetPlayer(online, epsilon=cfg.eps_start, device=cfg.device, name="learner")
    sampler = pool.make_sampler(cfg, learner)
    buffer = ReplayBuffer(cfg.replay_capacity, cfg.board, seed=cfg.seed)
    ctx = evaluate.EvalContext.from_config(cfg)

    games = grad_steps = pending = 0
    previous: nn.Module | None = None  # the last checkpoint's network, for the self-play score (D5)
    # Training stats since the last eval point: per-game sums (weighted by round size, since the
    # round before an eval point can be short) and per-gradient-step values.
    game_sums: defaultdict[str, float] = defaultdict(float)
    window_games = 0
    step_info: defaultdict[str, list[float]] = defaultdict(list)
    start = time.perf_counter()

    with rundir.JsonlLogger(run_dir / rundir.LOG_FILE) as logger:

        def checkpoint() -> None:
            nonlocal previous, window_games
            path = rundir.checkpoint_path(run_dir, games)
            nets.save_checkpoint(path, online, games=games, grad_steps=grad_steps)
            row: dict[str, Any] = {
                "games": games,
                "grad_steps": grad_steps,
                "epsilon": learner.epsilon,
                "buffer": len(buffer),
                **{k: v / window_games for k, v in game_sums.items()},
                **{k: float(np.mean(v)) for k, v in step_info.items()},
                **evaluate.evaluate_checkpoint(online, previous, cfg, ctx, rng),
            }
            row["wall_time"] = time.perf_counter() - start
            logger.log(row)
            print(_format_row(row), flush=True)
            sampler.on_checkpoint(path, games)
            previous = copy.deepcopy(online).requires_grad_(False)
            game_sums.clear()
            step_info.clear()
            window_games = 0

        checkpoint()
        for point in eval_points(cfg)[1:]:
            while games < point:
                n = min(cfg.parallel_games, point - games)
                eps = epsilon_at(games, cfg)
                learner.epsilon = eps
                sampler.set_epsilon(eps)
                transitions, stats = selfplay.collect(
                    learner,
                    sampler.opponents(cfg.pool_groups, rng),
                    n,
                    cfg,
                    rng,
                    learner_only=not sampler.store_opponent_moves,
                )
                buffer.extend(transitions)
                games += n
                pending += len(transitions)
                window_games += n
                for k, v in stats.items():
                    game_sums[k] += v * n

                while pending >= cfg.transitions_per_grad_step and len(buffer) >= cfg.batch_size:
                    pending -= cfg.transitions_per_grad_step
                    batch = buffer.sample(cfg.batch_size, cfg.augment, cfg.device)
                    loss, info = targets.td_loss(online, target_net, batch, cfg.target)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(online.parameters(), cfg.grad_clip)
                    opt.step()
                    grad_steps += 1
                    if grad_steps % cfg.target_update_every == 0:
                        target_net.load_state_dict(online.state_dict())
                    for k, v in info.items():
                        step_info[k].append(v)
                    step_info["grad_norm"].append(float(grad_norm))  # before clipping
            checkpoint()
    return run_dir


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train one self-play DQN run (spec §4.6).")
    add_config_args(parser)
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing run's log and checkpoints"
    )
    args = parser.parse_args(argv)
    print(train(config_from_args(args), overwrite=args.overwrite))


if __name__ == "__main__":
    main()
