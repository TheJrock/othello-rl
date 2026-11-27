"""Run directories, metadata, checkpoint paths and JSONL logs (spec §4.1).

    runs/<exp>/<target>-<sampling>/seed<k>/
        config.yaml          resolved Config
        meta.json            git commit + dirty flag, host, start time, versions
        log.jsonl            one row per eval point (spec §4.6), append-only, written by training
        backfill.jsonl       optional: metrics computed later from the checkpoints (scripts/backfill.py),
                             one row per eval point keyed by `games`; read_log merges it in
        crossplay.json       optional: round robin of the run's checkpoints (scripts/crossplay.py)
        stdout.log           written by scripts/sweep.py
        ckpt/g0005000.pt     one checkpoint per eval point; these are also the opponent pool
                             and the regret references (spec D8)
    runs/<exp>/crossplay_finals.json   optional: round robin of every run's final checkpoint

Torch-free on purpose, so analysis scripts can import it cheaply.
"""

from __future__ import annotations

import json
import platform
import shutil
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from .config import Config, load_config

CKPT_DIR = "ckpt"
LOG_FILE = "log.jsonl"
BACKFILL_FILE = "backfill.jsonl"
CROSSPLAY_FILE = "crossplay.json"  # in a run directory
CROSSPLAY_FINALS_FILE = "crossplay_finals.json"  # in an experiment directory
REPO_ROOT = Path(__file__).resolve().parents[2]


def git_info(cwd: Path = REPO_ROOT) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _torch_version() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    return torch.__version__


def prepare_run_dir(cfg: Config, overwrite: bool = False) -> Path:
    """Create the run directory and write config.yaml + meta.json.

    Refuses to touch a run that already has a log unless `overwrite`, in which case the old log,
    backfill, cross-play results and checkpoints are deleted so results from two different runs
    can never mix.
    """
    run_dir = cfg.run_dir
    log = run_dir / LOG_FILE
    if log.exists():
        if not overwrite:
            raise FileExistsError(f"{run_dir} already has a {LOG_FILE}; pass --overwrite to replace that run")
        log.unlink()
        (run_dir / BACKFILL_FILE).unlink(missing_ok=True)
        (run_dir / CROSSPLAY_FILE).unlink(missing_ok=True)
        shutil.rmtree(run_dir / CKPT_DIR, ignore_errors=True)
    (run_dir / CKPT_DIR).mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.yaml")
    meta = {
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _torch_version(),
        **git_info(),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return run_dir


def load_run_config(run_dir: str | Path) -> Config:
    return load_config(Path(run_dir) / "config.yaml")


def checkpoint_path(run_dir: str | Path, games: int) -> Path:
    return Path(run_dir) / CKPT_DIR / f"g{games:07d}.pt"


def list_checkpoints(run_dir: str | Path) -> list[tuple[int, Path]]:
    """(games, path) for every checkpoint in the run, oldest first."""
    return sorted((int(p.stem[1:]), p) for p in (Path(run_dir) / CKPT_DIR).glob("g*.pt"))


def _jsonable(obj: Any) -> Any:
    if hasattr(obj, "tolist"):  # numpy scalars/arrays, torch tensors
        return obj.tolist()
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


class JsonlLogger:
    """Append-only JSONL writer that flushes every row, so a crashed run keeps its log."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = self.path.open("a")

    def log(self, row: dict[str, Any]) -> None:
        self._file.write(json.dumps(row, default=_jsonable) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_backfill(run_dir: str | Path) -> dict[int, dict[str, Any]]:
    """Backfilled metrics by `games` ({} if the run has none)."""
    path = Path(run_dir) / BACKFILL_FILE
    return {row["games"]: row for row in _read_jsonl(path)} if path.exists() else {}


def write_backfill(run_dir: str | Path, rows: list[dict[str, Any]]) -> None:
    """Merge rows ({"games": g, metric: value, ...}) into the run's backfill file, replacing any
    earlier value of the same metric at the same eval point. Written atomically (temp file, then
    rename), so a reader never sees half a file."""
    merged = read_backfill(run_dir)
    for row in rows:
        merged.setdefault(row["games"], {}).update(row)
    path = Path(run_dir) / BACKFILL_FILE
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(merged[g], default=_jsonable) + "\n" for g in sorted(merged)))
    tmp.replace(path)


def read_log(run_dir: str | Path, backfill: bool = True) -> list[dict[str, Any]]:
    """The run's log rows. With `backfill`, each row also gets the backfilled metrics for its eval
    point, unless training already logged that metric (training's own value always wins)."""
    rows = _read_jsonl(Path(run_dir) / LOG_FILE)
    if backfill and (extra := read_backfill(run_dir)):
        rows = [{**extra.get(r["games"], {}), **r} for r in rows]
    return rows


def find_runs(root: str | Path = "runs", exp: str | None = None) -> list[Path]:
    """Every run directory under root (optionally one experiment), sorted."""
    return sorted(p.parent for p in Path(root).glob(f"{exp or '*'}/*/seed*/config.yaml"))


def run_condition(run_dir: str | Path) -> tuple[str, str, int]:
    """(target, sampling, seed) parsed from a run directory path."""
    run_dir = Path(run_dir)
    target, sampling = run_dir.parent.name.split("-", 1)
    return target, sampling, int(run_dir.name.removeprefix("seed"))
