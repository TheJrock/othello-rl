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
contents), when it was written by an older version of this script (FORMAT), or with --force.
Unfinished runs are skipped because their checkpoint set would still change, and so is an
experiment's across matrix while any of its runs is unfinished.

Each file also holds a second matrix, "start": the same round robin played from the standard
starting position alone. The bank starts every game 4 (4×4) or 6 (6×6, 8×8) random plies in, so
it never tests a net's own opening. From the start, greedy nets play one game per colour
(the games are deterministic), and a single game flips with tiny weight changes. So:
  within  the start matrix uses the same 15 checkpoints, 2 games a pair (noisy cell by cell);
  across  it averages over each run's last --start-checkpoints checkpoints (default 5): the
          round robin between every run's k-th-from-last checkpoint, for each k, then the mean,
          so 10 games a pair. On 4×4 it carries almost no information: 4×4 is a White win and
          late nets convert it, so nearly every game goes to White (E1: 0.496-0.504 everywhere).

CPU only, one thread per process, so it can run beside training on the GPU. Each process takes
one round robin at a time; --jobs sets how many run at once. Cost: ~0.5 s per 6×6 match (105 per
run, ~1 min), ~2 s per 8×8 match; 4×4 takes seconds. The start games add next to nothing.
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
from othello_rl.engine import initial
from othello_rl.players import NetPlayer

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = 15  # per run, for the within-run round robin (spec §4.7)
START_CHECKPOINTS = 5  # per run, averaged in the across-run start-position matrix
KINDS = ("within", "across")
RNG_SEED = 0  # the nets are greedy, so the generator only breaks exact Q ties (it never matters)
FORMAT = 2  # 2: adds "start" (Nov 27). Files of another format are recomputed


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
    """One round robin: its players (checkpoint paths and the runs they come from) and its file.
    start[k][i] is player i's checkpoint in the k-th start-position round robin; "start" in the
    file is the mean of those round robins."""

    kind: str
    out: Path
    runs: list[Path]  # the run each player comes from (its config builds the net)
    checkpoints: list[tuple[int, Path]]  # (games, path), as rundir.list_checkpoints gives them
    start: list[list[tuple[int, Path]]]

    @property
    def players(self) -> list[dict[str, Any]]:
        return [_player(r, g, p) for r, (g, p) in zip(self.runs, self.checkpoints, strict=True)]

    @property
    def start_players(self) -> list[list[dict[str, Any]]]:
        return [
            [_player(r, g, p) for r, (g, p) in zip(self.runs, level, strict=True)] for level in self.start
        ]

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
        return (
            old.get("format") == FORMAT
            and old.get("players") == self.players
            and old.get("openings") == self.openings()
            and old["start"]["players"] == self.start_players
        )


def last_trained(run: Path, k: int) -> list[tuple[int, Path]]:
    """The run's last k checkpoints, leaving out the untrained one."""
    return [c for c in rundir.list_checkpoints(run) if c[0] > 0][-k:]


def plan(
    runs_root: Path, exp: str, kinds: Sequence[str], k: int, force: bool, start_k: int = START_CHECKPOINTS
) -> tuple[list[Job], list[str]]:
    """The round robins to compute for one experiment, plus a line for each one skipped."""
    runs = rundir.find_runs(runs_root, exp)
    finished = [r for r in runs if is_finished(r)]
    unfinished = [r for r in runs if r not in finished]
    jobs: list[Job] = []
    notes = [f"skip {r}: unfinished" for r in unfinished] if "within" in kinds else []
    if "within" in kinds:
        for run in finished:
            picked = evenly_spaced(rundir.list_checkpoints(run), k)
            jobs.append(Job("within", run / rundir.CROSSPLAY_FILE, [run] * len(picked), picked, [picked]))
    if "across" in kinds:
        out = runs_root / exp / rundir.CROSSPLAY_FINALS_FILE
        if unfinished:
            notes.append(f"skip {out}: {len(unfinished)} of {len(runs)} runs unfinished")
        elif len(runs) < 2:
            notes.append(f"skip {out}: needs at least 2 runs, found {len(runs)}")
        else:
            lasts = [last_trained(r, start_k) for r in runs]
            levels = min(len(c) for c in lasts)  # the same number from every run
            start = [[c[len(c) - levels + i] for c in lasts] for i in range(levels)]
            jobs.append(Job("across", out, runs, [rundir.list_checkpoints(r)[-1] for r in runs], start))
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
    t0 = time.perf_counter()
    bank = job.openings()
    openings = evaluate.load_states(_resolve(bank))
    loaded: dict[Path, NetPlayer] = {}

    def player(run: Path, path: Path) -> NetPlayer:
        if path not in loaded:
            net = nets.load_net(path, rundir.load_run_config(run))
            loaded[path] = NetPlayer(net, epsilon=0.0, device="cpu", name=str(path))
        return loaded[path]

    def matrix(checkpoints: list[tuple[int, Path]], starts: list) -> np.ndarray:
        players = [player(r, p) for r, (_, p) in zip(job.runs, checkpoints, strict=True)]
        return evaluate.crossplay_matrix(players, starts, np.random.default_rng(RNG_SEED))

    m = matrix(job.checkpoints, openings)
    origin = [initial(rundir.load_run_config(job.runs[0]).board)]
    m_start = np.mean([matrix(level, origin) for level in job.start], axis=0)  # NaN diagonal stays NaN
    result = {
        "format": FORMAT,
        "kind": job.kind,
        "players": job.players,
        "score": _cells(m),  # row's score vs column
        "games_per_pair": 2 * len(openings),
        "openings": bank,
        "start": {
            "players": job.start_players,  # [k][i]: player i's checkpoint in the k-th round robin
            "score": _cells(m_start),  # the mean over k
            "games_per_pair": 2 * len(job.start),
        },
        "rng_seed": RNG_SEED,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        **rundir.git_info(),
    }
    _write_json(job.out, result)
    return f"{job.out}: {describe(result)}  [{time.perf_counter() - t0:.0f}s]"


def _cells(m: np.ndarray) -> list[list[float | None]]:
    return [[None if math.isnan(x) else float(x) for x in row] for row in m]


# --- summaries (printed; F4 and the report's numbers come from analyze.py) -------------------------


def _k(games: int) -> str:
    return f"{games // 1000}k" if games % 1000 == 0 else str(games)


def score_matrix(result: dict[str, Any], start: bool = False) -> np.ndarray:
    """The bank matrix, or with start=True the start-position one. None (the diagonal) -> NaN."""
    return np.array(result["start"]["score"] if start else result["score"], dtype=float)


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
    final_from_start = score_matrix(result, start=True)[-1, :-1].mean()
    return (
        f"{len(games)} checkpoints {_k(games[0])}-{_k(games[-1])}; against the final one the best "
        f"earlier checkpoint ({_k(games[j])}) scores {vs_final[j]:.3f}, the first ({_k(games[0])}) "
        f"{vs_final[0]:.3f}; from the start the final one scores {final_from_start:.3f} against "
        f"the earlier ones"
    )


def condition_blocks(result: dict[str, Any], start: bool = False) -> tuple[list[str], np.ndarray]:
    """evaluate.block_means by condition: B[a, b] = mean score of condition a's runs against
    condition b's (NaN when a == b), from the bank or (start=True) from the starting position."""
    groups = [p["run"].split("/")[0] for p in result["players"]]
    return evaluate.block_means(score_matrix(result, start), groups)


def _block_table(head: str, names: list[str], blocks: np.ndarray) -> list[str]:
    width = max(len(n) for n in names)
    lines = [head, " " * (width + 2) + "  ".join(f"{n:>{width}}" for n in names)]
    for a, row in zip(names, blocks, strict=True):
        cells = ["-" if np.isnan(x) else f"{x:.3f}" for x in row]
        lines.append(f"  {a:<{width}}" + "".join(f"  {c:>{width}}" for c in cells))
    return lines


def describe_across(result: dict[str, Any]) -> str:
    n = len(result["players"])
    levels = len(result["start"]["players"])
    bank = _block_table(f"{n} final checkpoints; mean score of row vs column:", *condition_blocks(result))
    head = (
        f"from the starting position, each run's last {levels} checkpoints (mean of {levels} round robins):"
    )
    return "\n".join(bank + _block_table(head, *condition_blocks(result, start=True)))


# --- entry point --------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--exp", nargs="+", required=True, help="experiments, e.g. e1-4x4 e2-6x6")
    p.add_argument("--kind", nargs="+", choices=KINDS, default=list(KINDS))
    p.add_argument("--checkpoints", type=int, default=CHECKPOINTS, help="per run, for --kind within")
    p.add_argument(
        "--start-checkpoints", type=int, default=START_CHECKPOINTS,
        help="per run, averaged in the across-run start-position matrix",
    )  # fmt: skip
    p.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    p.add_argument("--jobs", type=int, default=4, help="round robins at once, one process each")
    p.add_argument("--force", action="store_true", help="recompute files that are up to date")
    args = p.parse_args(argv)

    jobs: list[Job] = []
    for exp in args.exp:
        if not rundir.find_runs(args.runs_root, exp):
            sys.exit(f"no runs found for {exp} under {args.runs_root}")
        exp_jobs, notes = plan(
            args.runs_root, exp, args.kind, args.checkpoints, args.force, args.start_checkpoints
        )
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
