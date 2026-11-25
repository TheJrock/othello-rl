"""Backfill 6×6 endgame accuracy from a finished run's checkpoints (spec §4.4, §4.8 row P).

    python scripts/backfill.py --exp pilot-6x6
    python scripts/backfill.py --exp pilot-6x6 --force      # recompute values already backfilled
    make backfill EXPS=pilot-6x6

The 6×6 pilots trained before the endgame set existed, so their logs have no `endgame_accuracy`.
Every eval point saved a checkpoint, so the metric can be computed afterwards exactly as training
would have: the checkpoint's greedy move on each position of data/endgame6.jsonl. Results go to
backfill.jsonl in the run directory. log.jsonl is never touched; rundir.read_log merges the two by
`games`, so analyze.py sees backfilled values like logged ones. Rerunning only computes checkpoints
that have no value yet, so it's safe on a run still in progress.

Runs on the CPU by default (`--device`), so it doesn't compete with training on the GPU. Training
evaluated on the GPU instead, and the two can disagree in the last float digit, which flips a move
only when two Q-values are within that margin. Exact ties are broken by a seeded rng.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from othello_rl import evaluate, nets, rundir
from othello_rl.players import NetPlayer

ROOT = Path(__file__).resolve().parents[1]
METRIC = "endgame_accuracy"


def endgame_rows(run: Path, records: list[dict], device: str = "cpu", force: bool = False) -> list[dict]:
    """{"games", "endgame_accuracy"} for every checkpoint that has no value yet: not logged by
    training, and not backfilled before (unless `force`)."""
    cfg = rundir.load_run_config(run)
    done = {r["games"] for r in rundir.read_log(run, backfill=False) if METRIC in r}
    if not force:
        done |= {g for g, r in rundir.read_backfill(run).items() if METRIC in r}
    rows = []
    for games, path in rundir.list_checkpoints(run):
        if games in done:
            continue
        player = NetPlayer(nets.load_net(path, cfg).to(device), epsilon=0.0, device=device)
        rng = np.random.default_rng([cfg.seed, games])  # only breaks exact ties
        rows.append({"games": games, METRIC: evaluate.endgame_accuracy(player, records, rng)})
    return rows


def backfill_run(run: Path, records: list[dict], device: str = "cpu", force: bool = False) -> str:
    """Backfill one run; returns a one-line summary of what happened."""
    cfg = rundir.load_run_config(run)
    if cfg.board != 6:
        return f"skip {run}: {cfg.board}×{cfg.board} (the endgame set is 6×6)"
    rows = endgame_rows(run, records, device, force)
    if not rows:
        return f"skip {run}: every checkpoint already has {METRIC} (--force to recompute)"
    rundir.write_backfill(run, rows)
    accs = [r[METRIC] for r in rows]
    return f"{run}: {len(rows)} checkpoints, {METRIC} {accs[0]:.3f} -> {accs[-1]:.3f} (max {max(accs):.3f})"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--exp", nargs="+", required=True, help="experiments to backfill, e.g. pilot-6x6")
    p.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    p.add_argument("--endgame-set", type=Path, default=ROOT / "data" / "endgame6.jsonl")
    p.add_argument("--device", default="cpu")
    p.add_argument("--force", action="store_true", help="recompute values already backfilled")
    args = p.parse_args(argv)

    torch.set_num_threads(1)  # training's setting; leaves cores for runs in progress
    records = evaluate.load_endgame_set(args.endgame_set)
    runs = [run for exp in args.exp for run in rundir.find_runs(args.runs_root, exp)]
    if not runs:
        sys.exit(f"no runs found for {' '.join(args.exp)} under {args.runs_root}")
    for run in runs:
        start = time.perf_counter()
        summary = backfill_run(run, records, args.device, args.force)
        print(f"{summary}  [{time.perf_counter() - start:.1f}s]")


if __name__ == "__main__":
    main()
