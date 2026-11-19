"""Run directories, metadata, checkpoint paths and JSONL logs (spec §4.1).

    runs/<exp>/<target>-<sampling>/seed<k>/
        config.yaml          resolved Config
        meta.json            git commit + dirty flag, host, start time, versions
        log.jsonl            one row per eval point (spec §4.6)
        stdout.log           written by scripts/sweep.py
        ckpt/g0005000.pt     one checkpoint per eval point; these are also the opponent pool
                             and the regret references (spec D8)

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

    Refuses to touch a run that already has a log unless `overwrite`, in which case the old log and
    checkpoints are deleted so results from two different runs can never mix.
    """
    run_dir = cfg.run_dir
    log = run_dir / LOG_FILE
    if log.exists():
        if not overwrite:
            raise FileExistsError(f"{run_dir} already has a {LOG_FILE}; pass --overwrite to replace that run")
        log.unlink()
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


def read_log(run_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / LOG_FILE
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def find_runs(root: str | Path = "runs", exp: str | None = None) -> list[Path]:
    """Every run directory under root (optionally one experiment), sorted."""
    return sorted(p.parent for p in Path(root).glob(f"{exp or '*'}/*/seed*/config.yaml"))


def run_condition(run_dir: str | Path) -> tuple[str, str, int]:
    """(target, sampling, seed) parsed from a run directory path."""
    run_dir = Path(run_dir)
    target, sampling = run_dir.parent.name.split("-", 1)
    return target, sampling, int(run_dir.name.removeprefix("seed"))
