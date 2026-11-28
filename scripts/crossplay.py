"""Cross-play round robins (spec §4.7, E3): a run's checkpoints against each other, and every run's
final checkpoint against every other run's.

    python scripts/crossplay.py --exp e1-4x4 e2-6x6            # both kinds, 4 processes
    python scripts/crossplay.py --exp e2-6x6 --kind across --jobs 1
    make crossplay EXPS="e1-4x4 e2-6x6"

within  15 evenly spaced checkpoints of each run, from the first trained one to the final one
        (the untrained net is left out: every checkpoint beats it, so it would waste a slot)
        -> runs/<exp>/<target>-<sampling>/seed<k>/crossplay.json
across  every run's final checkpoint -> runs/<exp>/crossplay_finals.json

Every pair plays evaluate.play_match through the runs' opening bank: each opening once with each
player moving first, both greedy. The results sit next to the runs, like backfill.jsonl, and
analyze.py draws F4 from them. A file is recomputed only when its players change (a new seed,
another --checkpoints, or a retrained run, since checkpoints are matched by a hash of their
contents), or with --force. Unfinished runs are skipped because their checkpoint set would still
change, and so is an experiment's across matrix while any of its runs is unfinished.

CPU only, one thread per process, so it can run beside training on the GPU. Each process takes
one round robin at a time; --jobs sets how many run at once. Cost: ~0.5 s per 6×6 match (105 per
run, ~1 min), ~2 s per 8×8 match; 4×4 takes seconds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from othello_rl import evaluate, nets, rundir
from othello_rl.players import NetPlayer

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = 15  # per run, for the within-run round robin (spec §4.7)
KINDS = ("within", "across")
RNG_SEED = 0  # the nets are greedy, so the generator only breaks exact Q ties


def evenly_spaced(checkpoints: Sequence[tuple[int, Path]], k: int) -> list[tuple[int, Path]]:
    """k of a run's (games, path) checkpoints, evenly spaced from the first trained one to the final
    one, or all the trained ones if there are no more than k. Index i * (n - 1) / (k - 1), rounded
    half up in integers (np.round of a linspace would depend on float error at exact halves).
    E1 (30 trained, every 1k): 1k, 3k, ..., 13k, 16k, 18k, ..., 30k. E2 (40, every 10k): 10k, 40k,
    70k, 90k, ..., 400k. E6 (50, every 20k): 20k, 100k, 160k, 240k, ..., 1M."""
    trained = [c for c in checkpoints if c[0] > 0]
    n = len(trained)
    if n <= k:
        return list(trained)
    return [trained[(2 * i * (n - 1) + k - 1) // (2 * (k - 1))] for i in range(k)]


def is_finished(run: Path) -> bool:
    """True once the run has saved its final checkpoint (at cfg.games)."""
    ckpts = rundir.list_checkpoints(run)
    return bool(ckpts) and ckpts[-1][0] == rundir.load_run_config(run).games


def _player(run: Path, games: int, path: Path) -> dict[str, Any]:
    """How a checkpoint is recorded in the results, and recognised on the next pass."""
    sha1 = hashlib.sha1(path.read_bytes()).hexdigest()[:16]
    return {"run": f"{run.parent.name}/{run.name}", "games": games, "sha1": sha1}


@dataclass
class Job:
    """One round robin: its players (checkpoint paths and the runs they come from) and its file."""

    kind: str
    out: Path
    runs: list[Path]  # the run each player comes from (its config builds the net)
    checkpoints: list[tuple[int, Path]]  # (games, path), as rundir.list_checkpoints gives them

    @property
    def players(self) -> list[dict[str, Any]]:
        return [_player(r, g, p) for r, (g, p) in zip(self.runs, self.checkpoints, strict=True)]

    def openings(self) -> str:
        """The players' opening bank (from their configs), which must be the same for all of them."""
        banks = {rundir.load_run_config(r).openings for r in self.runs}
        if len(banks) != 1:
            raise ValueError(f"{self.out}: the players' runs use different opening banks {sorted(banks)}")
        return banks.pop()

    def up_to_date(self) -> bool:
        if not self.out.exists():
            return False
        old = json.loads(self.out.read_text())
        return old.get("players") == self.players and old.get("openings") == self.openings()


def plan(runs_root: Path, exp: str, kinds: Sequence[str], k: int, force: bool) -> tuple[list[Job], list[str]]:
    """The round robins to compute for one experiment, plus a line for each one skipped."""
    runs = rundir.find_runs(runs_root, exp)
    finished = [r for r in runs if is_finished(r)]
    unfinished = [r for r in runs if r not in finished]
    jobs: list[Job] = []
    notes = [f"skip {r}: unfinished" for r in unfinished] if "within" in kinds else []
    if "within" in kinds:
        for run in finished:
            picked = evenly_spaced(rundir.list_checkpoints(run), k)
            jobs.append(Job("within", run / rundir.CROSSPLAY_FILE, [run] * len(picked), picked))
    if "across" in kinds:
        out = runs_root / exp / rundir.CROSSPLAY_FINALS_FILE
        if unfinished:
            notes.append(f"skip {out}: {len(unfinished)} of {len(runs)} runs unfinished")
        elif len(runs) < 2:
            notes.append(f"skip {out}: needs at least 2 runs, found {len(runs)}")
        else:
            jobs.append(Job("across", out, runs, [rundir.list_checkpoints(r)[-1] for r in runs]))
    if not force:
        current = [j for j in jobs if j.up_to_date()]
        notes += [f"skip {j.out}: up to date (--force to recompute)" for j in current]
        jobs = [j for j in jobs if j not in current]
    return jobs, notes


def _resolve(path: str) -> Path:
    """Config paths are relative to the repo root, where training runs."""
    return Path(path) if Path(path).is_absolute() else ROOT / path


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    tmp.replace(path)  # atomic, so a reader never sees half a file


def compute(job: Job) -> str:
    """Play the round robin, write its file, and return a summary line. Runs in a worker process."""
    torch.set_num_threads(1)
    start = time.perf_counter()
    bank = job.openings()
    openings = evaluate.load_states(_resolve(bank))
    players = [
        NetPlayer(nets.load_net(p, rundir.load_run_config(r)), epsilon=0.0, device="cpu", name=str(p))
        for r, (_, p) in zip(job.runs, job.checkpoints, strict=True)
    ]
    m = evaluate.crossplay_matrix(players, openings, np.random.default_rng(RNG_SEED))
    result = {
        "kind": job.kind,
        "players": job.players,
        "score": [[None if math.isnan(x) else float(x) for x in row] for row in m],  # row vs column
        "games_per_pair": 2 * len(openings),
        "openings": bank,
        "rng_seed": RNG_SEED,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        **rundir.git_info(),
    }
    _write_json(job.out, result)
    return f"{job.out}: {describe(result)}  [{time.perf_counter() - start:.0f}s]"


# --- summaries (printed; F4 and the report's numbers come from analyze.py) -------------------------


def _k(games: int) -> str:
    return f"{games // 1000}k" if games % 1000 == 0 else str(games)


def score_matrix(result: dict[str, Any]) -> np.ndarray:
    return np.array(result["score"], dtype=float)  # None (the diagonal) -> NaN


def describe(result: dict[str, Any]) -> str:
    return describe_within(result) if result["kind"] == "within" else describe_across(result)


def describe_within(result: dict[str, Any]) -> str:
    """How well the best earlier checkpoint does against the final one (spec §4.7, H2), with the
    first checkpoint's score for scale. A size, not a count of scores above 0.5: on 4×4 every
    checkpoint from ~7k on is within a few games of every other, so a count would mostly count
    one-game margins."""
    games = [p["games"] for p in result["players"]]
    vs_final = score_matrix(result)[:-1, -1]  # each earlier checkpoint's score against the final one
    if vs_final.size == 0:
        return "one checkpoint, nothing to play"
    j = int(np.argmax(vs_final))
    return (
        f"{len(games)} checkpoints {_k(games[0])}-{_k(games[-1])}; against the final one the best "
        f"earlier checkpoint ({_k(games[j])}) scores {vs_final[j]:.3f}, the first ({_k(games[0])}) "
        f"{vs_final[0]:.3f}"
    )


def condition_blocks(result: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    """evaluate.block_means by condition: B[a, b] = mean score of condition a's runs against
    condition b's (NaN when a == b)."""
    return evaluate.block_means(score_matrix(result), [p["run"].split("/")[0] for p in result["players"]])


def describe_across(result: dict[str, Any]) -> str:
    names, blocks = condition_blocks(result)
    width = max(len(n) for n in names)
    head = f"{len(result['players'])} final checkpoints; mean score of row vs column:"
    lines = [head, " " * (width + 2) + "  ".join(f"{n:>{width}}" for n in names)]
    for a, row in zip(names, blocks, strict=True):
        cells = ["-" if np.isnan(x) else f"{x:.3f}" for x in row]
        lines.append(f"  {a:<{width}}" + "".join(f"  {c:>{width}}" for c in cells))
    return "\n".join(lines)


# --- entry point --------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--exp", nargs="+", required=True, help="experiments, e.g. e1-4x4 e2-6x6")
    p.add_argument("--kind", nargs="+", choices=KINDS, default=list(KINDS))
    p.add_argument("--checkpoints", type=int, default=CHECKPOINTS, help="per run, for --kind within")
    p.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    p.add_argument("--jobs", type=int, default=4, help="round robins at once, one process each")
    p.add_argument("--force", action="store_true", help="recompute files that are up to date")
    args = p.parse_args(argv)

    jobs: list[Job] = []
    for exp in args.exp:
        if not rundir.find_runs(args.runs_root, exp):
            sys.exit(f"no runs found for {exp} under {args.runs_root}")
        exp_jobs, notes = plan(args.runs_root, exp, args.kind, args.checkpoints, args.force)
        for note in notes:
            print(note)
        jobs += sorted(exp_jobs, key=lambda j: j.kind != "across")  # the biggest one first
    if not jobs:
        return
    print(f"{len(jobs)} round robins, {min(args.jobs, len(jobs))} at a time")
    if args.jobs <= 1:
        for job in jobs:
            print(compute(job), flush=True)
        return
    torch.set_num_threads(1)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for future in as_completed([pool.submit(compute, job) for job in jobs]):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
