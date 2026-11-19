"""Train one run: python -m othello_rl.train --config configs/4x4.yaml --target env --sampling naive --seed 0

Writes runs/<exp>/<target>-<sampling>/seed<k>/ (see rundir.py). Never plots (spec §4.1).
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .config import Config, add_config_args, config_from_args


def seed_everything(seed: int) -> np.random.Generator:
    """Seed `random`, numpy and torch; return the numpy Generator the run passes everywhere."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def epsilon_at(games_played: int, cfg: Config) -> float:
    """Linear from cfg.eps_start to cfg.eps_end over the first cfg.eps_decay_frac · cfg.games games,
    then constant at cfg.eps_end."""
    raise NotImplementedError("train.epsilon_at")


def train(cfg: Config, overwrite: bool = False) -> Path:
    """Run one training job and return its run directory.

    Outline (spec §4.5–4.6):

        run_dir = rundir.prepare_run_dir(cfg, overwrite)
        rng = seed_everything(cfg.seed); if cfg.torch_threads: torch.set_num_threads(cfg.torch_threads)
        online = nets.build_net(cfg); target_net = copy.deepcopy(online); opt = Adam(online, cfg.lr)
        learner = NetPlayer(online, epsilon=cfg.eps_start, device=cfg.device, name="learner")
        sampler = pool.make_sampler(cfg, learner)
        buffer = ReplayBuffer(cfg.replay_capacity, cfg.board, seed=cfg.seed)
        ctx = evaluate.EvalContext.from_config(cfg)

        checkpoint + evaluate + sampler.on_checkpoint at games = 0     # the demo's "Untrained" snapshot
        while games < cfg.games:
            eps = epsilon_at(games, cfg); learner.epsilon = eps; sampler.set_epsilon(eps)
            transitions, stats = selfplay.collect(learner, sampler.opponents(cfg.pool_groups, rng),
                                                  cfg.parallel_games, cfg, rng)
            buffer.extend(transitions); games += cfg.parallel_games; pending += len(transitions)
            while pending >= cfg.transitions_per_grad_step and len(buffer) >= cfg.batch_size:
                pending -= cfg.transitions_per_grad_step
                loss, info = targets.td_loss(online, target_net, buffer.sample(cfg.batch_size, cfg.augment,
                                                                               cfg.device), cfg.target)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), cfg.grad_clip); opt.step(); grad_steps += 1
                if grad_steps % cfg.target_update_every == 0: target_net.load_state_dict(online.state_dict())
            if games crossed a multiple of cfg.eval_every, or games >= cfg.games:
                path = rundir.checkpoint_path(run_dir, games)
                nets.save_checkpoint(path, online, games=games, grad_steps=grad_steps)
                row = evaluate.evaluate_checkpoint(online, previous_net, cfg, ctx, rng)
                logger.log({"games", "grad_steps", "wall_time", "epsilon", mean of info, **stats, **row})
                sampler.on_checkpoint(path, games)
    """
    raise NotImplementedError("train.train")


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
