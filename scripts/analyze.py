"""runs/ → figures/ and tables/ (spec §4.9). Never trains, so it is safe to re-run after every rsync.

    python scripts/analyze.py --exp e1-4x4 e2-6x6

Figures that aren't implemented yet are reported as [todo] and skipped, so this is usable from day one.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from othello_rl import rundir

ROOT = Path(__file__).resolve().parents[1]
Condition = tuple[str, str]  # (target, sampling)
Logs = dict[Condition, list[list[dict[str, Any]]]]  # condition -> one log (list of rows) per seed


def load_experiment(exp: str, root: Path) -> tuple[int | None, Logs]:
    """(board size, logs grouped by condition) for every finished-or-running run of one experiment."""
    logs: Logs = defaultdict(list)
    board = None
    for run in rundir.find_runs(root, exp):
        if not (run / rundir.LOG_FILE).exists():
            continue
        target, sampling, _ = rundir.run_condition(run)
        logs[(target, sampling)].append(rundir.read_log(run))
        board = rundir.load_run_config(run).board
    return board, dict(logs)


# --- figures (spec §4.9). Each writes one file into `out`. ----------------------------------------


def fig_accuracy_4x4(logs: Logs, out: Path) -> None:
    """F1: optimal-move accuracy vs games. Two panels (env, negamax), naive vs pool, seed bands, plus a
    chance-accuracy line."""
    raise NotImplementedError("analyze.fig_accuracy_4x4")


def fig_scores_6x6(logs: Logs, out: Path) -> None:
    """F2: score vs positional (random/greedy in the appendix version), same two-panel layout as F1."""
    raise NotImplementedError("analyze.fig_scores_6x6")


def fig_selfplay_vs_external(logs: Logs, out: Path) -> None:
    """F3 (headline figure): self-play score and external score on the same axes."""
    raise NotImplementedError("analyze.fig_selfplay_vs_external")


def fig_crossplay(exp: str, root: Path, out: Path) -> None:
    """F4: cross-play heatmaps, naive vs pool (evaluate.crossplay_matrix over 15 checkpoints per run)."""
    raise NotImplementedError("analyze.fig_crossplay")


def fig_endgame_6x6(logs: Logs, out: Path) -> None:
    """F5: 6×6 endgame accuracy vs games."""
    raise NotImplementedError("analyze.fig_endgame_6x6")


def fig_regret(exp: str, root: Path, out: Path) -> None:
    """F7: self-history regret by reference checkpoint for final naive vs pool agents (MADRID Fig. 4b
    analog), plus the regret grid over (empties × disc differential) (Fig. 6 analog)."""
    raise NotImplementedError("analyze.fig_regret")


def fig_true_vs_checkpoint_regret(exp: str, root: Path, out: Path) -> None:
    """F8 (4×4): checkpoint-estimated vs true regret."""
    raise NotImplementedError("analyze.fig_true_vs_checkpoint_regret")


def table_summary(logs: Logs, out: Path) -> None:
    """T1: evaluate.summarize_run numbers, mean ± std across seeds, plus the (pool − naive) gap per
    target type (H4)."""
    raise NotImplementedError("analyze.table_summary")


def _try(name: str, fn: Callable[..., None], *args: Any) -> None:
    try:
        fn(*args)
        print(f"[ok]   {name}")
    except NotImplementedError as e:
        print(f"[todo] {name}: {e}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--exp", nargs="+", default=["e1-4x4", "e2-6x6"])
    p.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    p.add_argument("--figures", type=Path, default=ROOT / "figures")
    p.add_argument("--tables", type=Path, default=ROOT / "tables")
    args = p.parse_args(argv)
    args.figures.mkdir(parents=True, exist_ok=True)
    args.tables.mkdir(parents=True, exist_ok=True)

    for exp in args.exp:
        board, logs = load_experiment(exp, args.runs_root)
        if not logs:
            print(f"[skip] {exp}: no runs with logs under {args.runs_root / exp}")
            continue
        n_runs = sum(len(v) for v in logs.values())
        print(f"{exp}: {board}×{board}, {n_runs} runs across {len(logs)} conditions")
        fig_dir = args.figures / exp
        fig_dir.mkdir(parents=True, exist_ok=True)
        _try("F3 self-play vs external", fig_selfplay_vs_external, logs, fig_dir)
        _try("F4 cross-play", fig_crossplay, exp, args.runs_root, fig_dir)
        _try("F7 checkpoint regret", fig_regret, exp, args.runs_root, fig_dir)
        _try("T1 summary table", table_summary, logs, args.tables / exp)
        if board == 4:
            _try("F1 4×4 accuracy", fig_accuracy_4x4, logs, fig_dir)
            _try("F8 true vs checkpoint regret", fig_true_vs_checkpoint_regret, exp, args.runs_root, fig_dir)
        if board == 6:
            _try("F2 6×6 scores", fig_scores_6x6, logs, fig_dir)
            _try("F5 endgame accuracy", fig_endgame_6x6, logs, fig_dir)


if __name__ == "__main__":
    main()
