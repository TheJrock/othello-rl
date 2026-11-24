"""Experiment configuration: dataclass defaults < YAML file < command-line flags.

    python -m othello_rl.train --config configs/6x6.yaml --exp e2-6x6 --target env --sampling pool --seed 0

`target` and `sampling` are the two factors of the 2×2 design (spec §1). Only selfplay.py and
targets.py may branch on `target`, and only pool.py may branch on `sampling` (spec §4.1), so the
four conditions provably differ nowhere else.

Every field is also a CLI flag: `--parallel-games 64` or `--parallel_games 64`. Optional fields
accept `none`. Values from YAML or the CLI are coerced to the field's type, so `lr: 3e-4` (which
YAML reads as a string) still works.
"""

from __future__ import annotations

import argparse
import dataclasses
import types
import typing
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

TARGETS = ("env", "negamax")
SAMPLINGS = ("naive", "pool")
BOARDS = (4, 6, 8)
CHOICES: dict[str, tuple] = {"target": TARGETS, "sampling": SAMPLINGS, "board": BOARDS}


@dataclass
class Config:
    # --- identity: runs/<exp>/<target>-<sampling>/seed<seed>/
    exp: str = "dev"
    board: int = 4
    target: str = "env"  # env (opponent-as-environment) | negamax
    sampling: str = "naive"  # naive (latest vs latest) | pool (past checkpoints)
    seed: int = 0

    # --- network (spec §4.5)
    channels: int = 64
    layers: int = 3

    # --- self-play and replay
    games: int = 30_000
    parallel_games: int = 128
    replay_capacity: int = 100_000  # both sides' transitions are stored (spec D2)
    augment: bool = True  # random D4 transform per sampled transition

    # --- optimisation
    batch_size: int = 256
    lr: float = 3e-4
    transitions_per_grad_step: int = 16
    target_update_every: int = 1_000  # gradient steps between hard target-net updates
    grad_clip: float = 10.0

    # --- exploration (both players play ε-greedy with the same ε)
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.2  # fraction of `games` over which ε decays linearly

    # --- opponent pool (used only when sampling == "pool")
    pool_groups: int = 8  # opponents per round; parallel games are split evenly between them
    pool_p_latest: float = 0.5  # chance a group plays the latest network instead of a past snapshot

    # --- evaluation and checkpoints (spec §4.7)
    eval_every: int = 1_000  # games between eval + checkpoint
    openings: str = "data/openings_4x4.json"
    solver_table: str | None = "data/cache/solve4x4.npz"  # 4×4 only
    endgame_set: str | None = None  # 6×6 only

    # --- infrastructure
    device: str = "cpu"
    torch_threads: int = 0  # 0 = torch default; use 1–2 when running many seeds at once
    runs_root: str = "runs"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name, allowed in CHOICES.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name}={getattr(self, name)!r}; expected one of {allowed}")
        positive = (
            "channels",
            "layers",
            "games",
            "parallel_games",
            "replay_capacity",
            "batch_size",
            "transitions_per_grad_step",
            "target_update_every",
            "eval_every",
            "pool_groups",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if not 0.0 <= self.pool_p_latest <= 1.0:
            raise ValueError(f"pool_p_latest must be in [0, 1], got {self.pool_p_latest}")
        if not 0.0 < self.eps_decay_frac <= 1.0:
            raise ValueError(f"eps_decay_frac must be in (0, 1], got {self.eps_decay_frac}")
        if not 0.0 <= self.eps_end <= self.eps_start <= 1.0:
            raise ValueError("need 0 <= eps_end <= eps_start <= 1")
        if self.parallel_games % self.pool_groups:
            raise ValueError(
                f"parallel_games ({self.parallel_games}) must be divisible by pool_groups ({self.pool_groups})"
            )

    @property
    def run_dir(self) -> Path:
        return Path(self.runs_root) / self.exp / f"{self.target}-{self.sampling}" / f"seed{self.seed}"

    @property
    def action_size(self) -> int:
        return self.board * self.board + 1

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise KeyError(f"unknown config keys: {unknown}")
        return cls(**{name: _coerce(name, value) for name, value in data.items()})


_HINTS = typing.get_type_hints(Config)


def _field_type(name: str) -> tuple[type, bool]:
    """(base type, is_optional) for a Config field."""
    hint = _HINTS[name]
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        return args[0], True
    return hint, False


def _coerce(name: str, value: Any) -> Any:
    typ, optional = _field_type(name)
    if value is None or (optional and isinstance(value, str) and value.lower() in ("none", "null", "")):
        if optional:
            return None
        raise ValueError(f"{name} cannot be None")
    if typ is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "yes", "1"):
            return True
        if isinstance(value, str) and value.lower() in ("false", "no", "0"):
            return False
        raise ValueError(f"{name}: expected a boolean, got {value!r}")
    if typ is int:
        if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
            raise ValueError(f"{name}: expected an integer, got {value!r}")
        return int(value)
    if typ is float:
        if isinstance(value, bool):
            raise ValueError(f"{name}: expected a number, got {value!r}")
        return float(value)
    return str(value)


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Defaults < YAML at `path` < keyword overrides."""
    data: dict[str, Any] = {}
    if path is not None:
        data = yaml.safe_load(Path(path).read_text()) or {}
    data.update(overrides)
    return Config.from_dict(data)


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Add --config plus one flag per Config field. Unset flags don't override the YAML."""
    parser.add_argument("--config", type=Path, default=None, help="YAML file with defaults for a board size")
    for f in fields(Config):
        flags = [f"--{f.name.replace('_', '-')}"]
        if "_" in f.name:
            flags.append(f"--{f.name}")
        kwargs: dict[str, Any] = {
            "dest": f.name,
            "default": argparse.SUPPRESS,
            "help": f"default: {f.default!r}",
        }
        if f.name in CHOICES:
            kwargs["choices"] = CHOICES[f.name]
            kwargs["type"] = type(CHOICES[f.name][0])
        parser.add_argument(*flags, **kwargs)


def config_from_args(args: argparse.Namespace) -> Config:
    values = vars(args)
    known = {f.name for f in fields(Config)}
    return load_config(values.get("config"), **{k: v for k, v in values.items() if k in known})


def parse_cli(argv: Sequence[str] | None = None, description: str | None = None) -> Config:
    parser = argparse.ArgumentParser(description=description)
    add_config_args(parser)
    return config_from_args(parser.parse_args(argv))
