"""Measure training throughput on this machine and project per-run times (spec §4.8).

    python scripts/throughput.py --config configs/6x6.yaml --device mps
    python scripts/throughput.py --config configs/8x8.yaml --device cpu --torch-threads 1
    python scripts/throughput.py --config configs/6x6.yaml --device cuda --json   # one JSON line, for the spec

Times the three parts of a run with the real training code: self-play rounds (selfplay.collect, both
naive and pool-style with 8 distinct opponent networks), gradient steps (replay sampling with D4
augmentation, the Double-DQN loss, backward, clipping, Adam), and one evaluate_checkpoint. Then it
projects a full run of `games` from the config for transitions_per_grad_step 16 and 32, and how many
games fit in a time limit. Evaluation doesn't include 6×6 endgame accuracy yet (one batched forward
pass over 500 positions: negligible).

Run nothing else heavy at the same time; to measure runs that share a machine, start several copies.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import time

import numpy as np
import torch
from torch import nn

from othello_rl import evaluate, nets, selfplay, targets
from othello_rl.config import load_config
from othello_rl.players import NetPlayer
from othello_rl.replay import ReplayBuffer


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--torch-threads", type=int, default=0, help="0 = torch default")
    p.add_argument("--rounds", type=int, default=3, help="timed self-play rounds (after one warm-up)")
    p.add_argument(
        "--steps", type=int, default=0, help="timed gradient steps (default: 30 on cpu, 200 on a GPU)"
    )
    p.add_argument(
        "--limit-hours", type=float, default=0.0, help="per-run limit (default: 6 for 6x6, 24 for 8x8)"
    )
    p.add_argument("--json", action="store_true", help="also print one JSON line with every number")
    args = p.parse_args()

    cfg = load_config(args.config, device=args.device, torch_threads=args.torch_threads, endgame_set=None)
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    steps_to_time = args.steps or (30 if args.device == "cpu" else 200)
    limit_h = args.limit_hours or {4: 1.0, 6: 6.0, 8: 24.0}[cfg.board]
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    online = nets.build_net(cfg).to(cfg.device)
    target_net = copy.deepcopy(online).requires_grad_(False)
    opt = torch.optim.Adam(online.parameters(), lr=cfg.lr)
    learner = NetPlayer(online, epsilon=cfg.eps_end, device=cfg.device, name="learner")
    snapshots = [
        NetPlayer(copy.deepcopy(online), epsilon=cfg.eps_end, device=cfg.device, name=f"snap{i}")
        for i in range(cfg.pool_groups)
    ]
    buffer = ReplayBuffer(cfg.replay_capacity, cfg.board, seed=0)

    def time_rounds(opponents) -> tuple[float, float]:
        """(seconds per game, transitions per game) over args.rounds rounds, after one warm-up."""
        selfplay.collect(learner, opponents, cfg.parallel_games, cfg, rng)  # warm-up (kernel compiles)
        seconds = transitions = 0.0
        for _ in range(args.rounds):
            t = time.perf_counter()
            trs, _ = selfplay.collect(learner, opponents, cfg.parallel_games, cfg, rng)
            seconds += time.perf_counter() - t
            transitions += len(trs)
            buffer.extend(trs)
        games = args.rounds * cfg.parallel_games
        return seconds / games, transitions / games

    naive_s, tpg = time_rounds([learner] * cfg.pool_groups)
    pool_s, _ = time_rounds(snapshots)  # worst case: every group plays a different past network

    def grad_step() -> None:
        batch = buffer.sample(cfg.batch_size, cfg.augment, cfg.device)
        loss, _ = targets.td_loss(online, target_net, batch, cfg.target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(online.parameters(), cfg.grad_clip)
        opt.step()

    for _ in range(10):  # warm-up
        grad_step()
    sync(cfg.device)
    t = time.perf_counter()
    for _ in range(steps_to_time):
        grad_step()
    sync(cfg.device)
    step_s = (time.perf_counter() - t) / steps_to_time

    ctx = evaluate.EvalContext.from_config(cfg)
    previous = copy.deepcopy(online)
    evaluate.evaluate_checkpoint(online, previous, cfg, ctx, rng)  # warm-up
    t = time.perf_counter()
    evaluate.evaluate_checkpoint(online, previous, cfg, ctx, rng)
    eval_s = time.perf_counter() - t

    eval_points = cfg.games // cfg.eval_every + 1
    threads = args.torch_threads or torch.get_num_threads()
    print(
        f"{cfg.board}x{cfg.board}  device={cfg.device}  torch threads={threads}  ({platform.machine()}, torch {torch.__version__})"
    )
    print(
        f"  self-play   naive {1000 * naive_s:6.2f} ms/game   pool-style {1000 * pool_s:6.2f} ms/game   "
        f"{tpg:.1f} transitions/game (both sides)"
    )
    print(f"  grad step   {1000 * step_s:6.1f} ms        eval point {eval_s:5.1f} s")
    out = {
        "board": cfg.board,
        "device": cfg.device,
        "threads": threads,
        "naive_ms_per_game": 1000 * naive_s,
        "pool_ms_per_game": 1000 * pool_s,
        "transitions_per_game": tpg,
        "step_ms": 1000 * step_s,
        "eval_s": eval_s,
    }
    for tpgs in (16, 32):
        per_game = {"naive": naive_s + tpg / tpgs * step_s, "pool": pool_s + tpg / tpgs * step_s}
        eval_h = eval_points * eval_s / 3600
        parts = {k: cfg.games * v / 3600 + eval_h for k, v in per_game.items()}
        fit = {k: int((limit_h - eval_h) * 3600 / v) for k, v in per_game.items()}
        print(
            f"  tpgs={tpgs:2d}     {cfg.games:,} games: naive {parts['naive']:5.1f} h, pool {parts['pool']:5.1f} h "
            f"(self-play {cfg.games * naive_s / 3600:.1f}–{cfg.games * pool_s / 3600:.1f} h, steps "
            f"{cfg.games * tpg / tpgs * step_s / 3600:.1f} h, eval {eval_h:.1f} h)   "
            f"games in {limit_h:g} h: {fit['pool']:,}–{fit['naive']:,}"
        )
        out[f"run_h_tpgs{tpgs}"] = parts
        out[f"games_in_limit_tpgs{tpgs}"] = fit
    if args.json:
        print(json.dumps(out))


if __name__ == "__main__":
    main()
