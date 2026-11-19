"""Launch the 2×2×seeds grid as parallel training processes (spec §4.8).

    python scripts/sweep.py --config configs/4x4.yaml --exp e1-4x4 --seeds 0 1 2 3 4 --jobs 8
    python scripts/sweep.py ... --dry-run              # print the commands only
    python scripts/sweep.py ... -- --games 2000        # flags after -- are passed to every run

Each run's stdout/stderr goes to <run_dir>/stdout.log. Runs that already have a log are skipped
unless --overwrite. Exits non-zero if any run fails.
"""

from __future__ import annotations

import argparse
import itertools
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from othello_rl.config import SAMPLINGS, TARGETS, parse_cli
from othello_rl.rundir import LOG_FILE


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1 :]

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--exp", required=True, help="experiment name, e.g. e1-4x4, pilot-6x6, e2-6x6")
    p.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    p.add_argument("--samplings", nargs="+", choices=SAMPLINGS, default=list(SAMPLINGS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="runs at once")
    p.add_argument(
        "--torch-threads", type=int, default=1, help="per run, so parallel runs don't oversubscribe cores"
    )
    p.add_argument("--overwrite", action="store_true", help="re-run conditions that already have a log")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    jobs: list[tuple[Path, list[str]]] = []
    for target, sampling, seed in itertools.product(args.targets, args.samplings, args.seeds):
        cfg_argv = [
            "--config",
            str(args.config),
            "--exp",
            args.exp,
            "--target",
            target,
            "--sampling",
            sampling,
            "--seed",
            str(seed),
            "--torch-threads",
            str(args.torch_threads),
            *extra,
        ]
        cfg = parse_cli(cfg_argv)  # validates every config before anything launches
        if (cfg.run_dir / LOG_FILE).exists() and not args.overwrite:
            print(f"[skip] {cfg.run_dir} already has a log (use --overwrite to re-run)")
            continue
        cmd = [
            sys.executable,
            "-m",
            "othello_rl.train",
            *cfg_argv,
            *(["--overwrite"] if args.overwrite else []),
        ]
        jobs.append((cfg.run_dir, cmd))

    if args.dry_run:
        for _, cmd in jobs:
            print(shlex.join(cmd))
        return 0

    pending = list(jobs)
    running: list[tuple[Path, subprocess.Popen, object, float]] = []
    failed: list[Path] = []
    start = time.monotonic()
    print(f"{len(jobs)} runs, {args.jobs} at a time")
    try:
        while pending or running:
            while pending and len(running) < args.jobs:
                run_dir, cmd = pending.pop(0)
                run_dir.mkdir(parents=True, exist_ok=True)
                log = (run_dir / "stdout.log").open("w")
                running.append(
                    (
                        run_dir,
                        subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT),
                        log,
                        time.monotonic(),
                    )
                )
                print(f"[start] {run_dir}")
            time.sleep(2)
            for item in list(running):
                run_dir, proc, log, t0 = item
                if proc.poll() is None:
                    continue
                log.close()
                running.remove(item)
                status = (
                    "done" if proc.returncode == 0 else f"FAILED (exit {proc.returncode}, see stdout.log)"
                )
                print(f"[{status}] {run_dir} in {_fmt(time.monotonic() - t0)}")
                if proc.returncode:
                    failed.append(run_dir)
    except KeyboardInterrupt:
        for _, proc, log, _ in running:
            proc.terminate()
        for _, proc, log, _ in running:
            proc.wait()
            log.close()
        print("interrupted: terminated running jobs")
        return 130

    print(f"finished {len(jobs)} runs in {_fmt(time.monotonic() - start)}; {len(failed)} failed")
    for run_dir in failed:
        print(f"  failed: {run_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
