"""runs/ → figures/ and tables/ (spec §4.9). Never trains, so it is safe to re-run after every rsync.

    python scripts/analyze.py --exp e1-4x4 e2-6x6
    python scripts/analyze.py --exp pilot-6x6        # works on runs that are still going, and on 1 seed

Figures that aren't implemented yet are reported as [todo], and ones whose data isn't there yet as
[skip], so this is usable from day one.

Output names are the contract with report/main.tex, which includes them by path (figures/<exp>/f*.pdf,
tables/<exp>/t1_summary.tex). Save figures as PDF, sized for the report's 5.5 in text width
and at most 1.6-1.7 in tall (figures count toward the page limit; see report/main.tex).

Curves are the mean over seeds with a band from the lowest to the highest seed, over the eval points
every seed has reached (so a run that is still going shortens its condition's curve, never distorts it).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from othello_rl import rundir
from othello_rl.evaluate import FINAL_POINTS, summarize_run

ROOT = Path(__file__).resolve().parents[1]
Condition = tuple[str, str]  # (target, sampling)
Logs = dict[Condition, list[list[dict[str, Any]]]]  # condition -> one log (list of rows) per seed

TARGETS = ("env", "negamax")
SAMPLINGS = ("naive", "pool")
TARGET_TITLES = {"env": "env target (A)", "negamax": "negamax target (N)"}
LATEX_NAMES = {"env": r"\envT{}", "negamax": r"\negT{}", "naive": r"\naiveS{}", "pool": r"\poolS{}"}

# Style (dataviz skill, reference palette, light mode on white paper): categorical slots 1-2 for the
# two samplers (validated: CVD ΔE 24.7, normal-vision ΔE 33.6, both ≥ 3:1 on white), text in ink
# tokens, recessive hairline grid, thin lines, bands as light fills of the series colour.
COLORS = {"naive": "#2a78d6", "pool": "#eb6834"}
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
WIDTH = 5.5  # in: the report's text width
STYLE = {
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "lines.linewidth": 1.4,
    "legend.frameon": False,
    "pdf.fonttype": 42,  # embed TrueType, so text stays text in the PDF
}
EXTERNAL = {4: ("accuracy_4x4", "optimal-move accuracy"), 6: ("score_positional", "score vs positional")}


def _save(fig: plt.Figure, path: Path) -> None:
    """Save without a creation date, so redrawing unchanged data leaves the committed PDF unchanged."""
    fig.savefig(path, metadata={"CreationDate": None})


class Skip(Exception):
    """The data a figure needs isn't there yet (reported as [skip], not as an error)."""


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


def external_metric(board: int) -> tuple[str, str]:
    """(log key, axis label) of the metric H1 is judged on: exact accuracy on 4×4, score vs
    positional on 6×6 and 8×8."""
    return EXTERNAL.get(board, EXTERNAL[6])


# --- curves -----------------------------------------------------------------------------------------


def curve(runs: Sequence[Sequence[dict[str, Any]]], key: str) -> tuple[np.ndarray, np.ndarray]:
    """(games, values[seed, point]) over the eval points where every seed has `key`."""
    per_seed = [{r["games"]: r[key] for r in rows if key in r} for rows in runs]
    common = sorted(set.intersection(*(set(d) for d in per_seed))) if per_seed else []
    return np.array(common), np.array([[d[g] for g in common] for d in per_seed])


def _thousands(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "0" if x == 0 else f"{x / 1000:g}k"))


def _plot_condition(ax: plt.Axes, runs, key: str, color: str, **line) -> int:
    """Mean line plus a min–max band over seeds; returns the number of seeds."""
    games, y = curve(runs, key)
    if not len(games):
        return 0
    ax.plot(games, y.mean(axis=0), color=color, **line)
    if len(y) > 1 and line.get("linestyle", "-") == "-":
        ax.fill_between(games, y.min(axis=0), y.max(axis=0), color=color, alpha=0.18, linewidth=0)
    return len(y)


def _reference_line(ax: plt.Axes, y: float, label: str) -> None:
    """A labelled horizontal reference (chance, or an even score) in muted ink: dotted, so it never
    reads as the dashed self-play series, and labelled just under its right end, where the curves
    (which rise above chance and hover above an even score) leave room."""
    ax.axhline(y, color=MUTED, linewidth=1.0, linestyle=(0, (1, 1.5)), zorder=1)
    ax.annotate(
        label,
        (1.0, y),
        xycoords=("axes fraction", "data"),
        xytext=(-2, -2),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=6.5,
        color=INK_2,
    )


def _legend(fig: plt.Figure, axes, handles: list | None = None, in_title_row: bool = False) -> None:
    """One legend for the whole figure, placed where it can't cover data: in the empty right end
    of the top-right panel's title row (in_title_row, for a short legend; the titles are left-aligned),
    or else in its own row above the panels."""
    if handles is None:
        seen: dict[str, Any] = {}
        for ax in np.ravel(axes):
            for h, label in zip(*ax.get_legend_handles_labels()):
                seen.setdefault(label, h)
        handles = [seen[s] for s in SAMPLINGS if s in seen]
    if in_title_row:
        np.atleast_2d(axes)[0, -1].legend(  # top row, rightmost panel
            handles=handles,
            loc="lower right",
            bbox_to_anchor=(1.0, 1.0),
            ncols=len(handles),
            handlelength=1.8,
            columnspacing=1.2,
            borderaxespad=0.1,
        )
        return
    fig.legend(
        handles=handles,
        loc="outside upper right",
        ncols=len(handles),
        handlelength=1.8,
        columnspacing=1.2,
        borderaxespad=0.2,
    )


def two_panel(
    logs: Logs,
    key: str,
    ylabel: str,
    out: Path,
    height: float = 1.6,
    reference: tuple[float, str] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """The F1/F2/F5 layout: env | negamax panels, naive vs pool, seed bands, shared y axis."""
    if not any(curve(runs, key)[0].size for runs in logs.values()):
        raise Skip(f"no runs have {key!r} yet")
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(WIDTH, height), sharey=True, layout="constrained")
        for ax, target in zip(axes, TARGETS):
            for sampling in SAMPLINGS:
                _plot_condition(ax, logs.get((target, sampling), []), key, COLORS[sampling], label=sampling)
            if reference:
                _reference_line(ax, *reference)
            ax.set_title(TARGET_TITLES[target], loc="left", color=INK)
            ax.set_xlabel("training games")
            _thousands(ax)
        axes[0].set_ylabel(ylabel)
        if ylim:
            axes[0].set_ylim(*ylim)
        _legend(fig, axes, in_title_row=True)
        _save(fig, out)
        plt.close(fig)


def chance_4x4() -> float | None:
    """Expected accuracy of a uniformly random mover on the 4×4 eval set (None without the table)."""
    from othello_rl import solver

    path = ROOT / "data" / "cache" / "solve4x4.npz"
    if not path.exists():
        return None
    table = solver.load_table(path)
    return solver.chance_accuracy(table, solver.eval_positions_4x4(table))


def chance_endgame() -> float | None:
    """The same for the 6×6 endgame set."""
    path = ROOT / "data" / "endgame6.jsonl"
    if not path.exists():
        return None
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    legal = [[v for v in r["values"] if v is not None] for r in records]
    return float(np.mean([v.count(max(v)) / len(v) for v in legal]))


# --- figures (spec §4.9). Each writes one file into `out`. ----------------------------------------


def fig_accuracy_4x4(logs: Logs, out: Path) -> None:
    """F1: optimal-move accuracy vs games. Two panels (env, negamax), naive vs pool, seed bands, plus a
    chance-accuracy line. Writes f1_accuracy.pdf."""
    chance = chance_4x4()
    two_panel(
        logs,
        "accuracy_4x4",
        "optimal-move accuracy",
        out / "f1_accuracy.pdf",
        reference=(chance, f"chance {chance:.1%}") if chance else None,
    )


def fig_scores_6x6(logs: Logs, out: Path) -> None:
    """F2: score vs positional, same two-panel layout as F1, with an even-score line. Also writes
    the appendix version with random and greedy (f2_scores_appendix.pdf). Writes f2_scores.pdf."""
    two_panel(
        logs,
        "score_positional",
        "score vs positional",
        out / "f2_scores.pdf",
        reference=(0.5, "even"),
        ylim=(0, 1),
    )
    with plt.rc_context(STYLE):
        refs = ("random", "greedy", "positional")
        fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 4.2), sharex=True, sharey=True, layout="constrained")
        for row, ref in zip(axes, refs):
            for ax, target in zip(row, TARGETS):
                for sampling in SAMPLINGS:
                    _plot_condition(
                        ax, logs.get((target, sampling), []), f"score_{ref}", COLORS[sampling], label=sampling
                    )
                _reference_line(ax, 0.5, "even")
                if ref == refs[0]:
                    ax.set_title(TARGET_TITLES[target], loc="left", color=INK)
                _thousands(ax)
            row[0].set_ylabel(f"score vs {ref}")
        for ax in axes[-1]:
            ax.set_xlabel("training games")
        axes[0, 0].set_ylim(0, 1)
        _legend(fig, axes, in_title_row=True)
        _save(fig, out / "f2_scores_appendix.pdf")
        plt.close(fig)


def fig_selfplay_vs_external(logs: Logs, out: Path, board: int = 6) -> None:
    """F3 (headline figure): self-play score and external score on the same axes.
    Writes f3_selfplay_vs_external.pdf.

    Solid: the external metric (score vs positional; exact accuracy on 4×4), mean with seed bands.
    Dashed: the self-play score (vs the previous checkpoint, D5), mean only. Colour = sampler."""
    key, label = external_metric(board)
    if not any(curve(runs, "score_selfplay")[0].size for runs in logs.values()):
        raise Skip("no runs have score_selfplay yet")
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 1.7), sharey=True, layout="constrained")
        for ax, target in zip(axes, TARGETS):
            for sampling in SAMPLINGS:
                runs = logs.get((target, sampling), [])
                _plot_condition(ax, runs, key, COLORS[sampling])
                _plot_condition(
                    ax, runs, "score_selfplay", COLORS[sampling], linestyle=(0, (3, 2)), linewidth=1.1
                )
            _reference_line(ax, 0.5, "even")
            ax.set_title(TARGET_TITLES[target], loc="left", color=INK)
            ax.set_xlabel("training games")
            _thousands(ax)
        axes[0].set_ylabel("score" if board != 4 else "score / accuracy")
        axes[0].set_ylim(0, 1)
        handles = [Line2D([], [], color=COLORS[s], label=s) for s in SAMPLINGS] + [
            Line2D([], [], color=INK_2, label=label),
            Line2D([], [], color=INK_2, linestyle=(0, (3, 2)), linewidth=1.1, label="vs previous checkpoint"),
        ]
        _legend(fig, axes, handles)
        _save(fig, out / "f3_selfplay_vs_external.pdf")
        plt.close(fig)


def fig_crossplay(exp: str, root: Path, out: Path) -> None:
    """F4: cross-play heatmaps, naive vs pool (evaluate.crossplay_matrix over 15 checkpoints per run).
    Writes f4_crossplay.pdf."""
    raise NotImplementedError("analyze.fig_crossplay")


def fig_endgame_6x6(logs: Logs, out: Path) -> None:
    """F5: 6×6 endgame accuracy vs games, F1's layout with the endgame set's chance line.
    Writes f5_endgame.pdf."""
    chance = chance_endgame()
    two_panel(
        logs,
        "endgame_accuracy",
        "endgame accuracy",
        out / "f5_endgame.pdf",
        reference=(chance, f"chance {chance:.1%}") if chance else None,
    )


def fig_regret(exp: str, root: Path, out: Path) -> None:
    """F7: self-history regret by reference checkpoint for final naive vs pool agents (MADRID Fig. 4b
    analog), plus the regret grid over (empties × disc differential) (Fig. 6 analog).
    Writes f7_regret.pdf."""
    raise NotImplementedError("analyze.fig_regret")


def fig_true_vs_checkpoint_regret(exp: str, root: Path, out: Path) -> None:
    """F8 (4×4): checkpoint-estimated vs true regret. Writes f8_true_vs_checkpoint_regret.pdf."""
    raise NotImplementedError("analyze.fig_true_vs_checkpoint_regret")


# --- T1 -----------------------------------------------------------------------------------------------

T1_COLUMNS = [  # (summarize_run key, header, decimals)
    ("final", "Final", 3),
    ("peak", "Peak", 3),
    ("max_drawdown", "Max drawdown", 3),
    ("forgetting_events", "Forgetting events", 1),
    ("spearman", r"Spearman $\rho$", 2),
]


def common_horizon(logs: Logs) -> int:
    """The last eval point every run in the experiment has reached."""
    return min(rows[-1]["games"] for runs in logs.values() for rows in runs)


def summaries(logs: Logs, key: str) -> dict[Condition, dict[str, np.ndarray]]:
    """summarize_run per seed, as arrays over seeds: {condition: {column: values}}. Every run is cut
    at common_horizon(logs) first, so runs that are still going are compared with the others at the
    same point in training ("final" = the last FINAL_POINTS eval points up to there)."""
    horizon = common_horizon(logs)
    out = {}
    for cond, runs in logs.items():
        per_seed = [summarize_run([r for r in rows if r["games"] <= horizon], key) for rows in runs]
        out[cond] = {c: np.array([s[c] for s in per_seed], dtype=float) for c, _, _ in T1_COLUMNS}
    return out


def _mean_sd(x: np.ndarray) -> tuple[float, float]:
    x = x[~np.isnan(x)]
    if not len(x):
        return math.nan, math.nan
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else math.nan


def _gap(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """mean(a) − mean(b) and its standard error (Welch); SE is NaN with fewer than 2 seeds a side."""
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if not len(a) or not len(b):
        return math.nan, math.nan
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) if min(len(a), len(b)) > 1 else math.nan
    return float(a.mean() - b.mean()), se


def t1_rows(
    stats: dict[Condition, dict[str, np.ndarray]],
) -> list[tuple[str, str, list[tuple[float, float]]]]:
    """[(kind, label, [(value, spread) per column])]. kind is "cond" (mean ± sd over seeds) or "gap"
    (difference of means ± SE)."""
    rows = []
    for target in TARGETS:
        for sampling in SAMPLINGS:
            if (target, sampling) in stats:
                s = stats[(target, sampling)]
                rows.append(("cond", f"{target}-{sampling}", [_mean_sd(s[c]) for c, _, _ in T1_COLUMNS]))
    gaps = {}
    for target in TARGETS:
        if (target, "naive") in stats and (target, "pool") in stats:
            pool, naive = stats[(target, "pool")], stats[(target, "naive")]
            gaps[target] = {c: (pool[c], naive[c]) for c, _, _ in T1_COLUMNS}
            rows.append(
                ("gap", f"pool - naive ({target})", [_gap(*gaps[target][c]) for c, _, _ in T1_COLUMNS])
            )
    if len(gaps) == 2:  # H4: is the pool's effect bigger under env than under negamax?
        h4 = []
        for c, _, _ in T1_COLUMNS:
            (ge, se_e), (gn, se_n) = _gap(*gaps["env"][c]), _gap(*gaps["negamax"][c])
            h4.append((ge - gn, math.hypot(se_e, se_n)))
        rows.append(("gap", "H4: env gap - negamax gap", h4))
    return rows


def _fmt(value: float, spread: float, decimals: int, signed: bool, tex: bool) -> str:
    if math.isnan(value):
        return "--" if tex else "-"
    v = f"{value:+.{decimals}f}" if signed else f"{value:.{decimals}f}"
    if math.isnan(spread):
        return f"${v}$" if tex else v
    pm = r" \pm " if tex else " ± "
    return f"${v}{pm}{spread:.{decimals}f}$" if tex else f"{v}{pm}{spread:.{decimals}f}"


def _tex_label(label: str) -> str:
    if label.startswith("pool - naive"):
        target = label[label.index("(") + 1 : -1]
        return rf"\poolS{{}}$-$\naiveS{{}}, {LATEX_NAMES[target]}"
    if label.startswith("H4"):
        return r"H4: \envT{} gap $-$ \negT{} gap"
    target, sampling = label.split("-")
    return f"{LATEX_NAMES[target]}-{LATEX_NAMES[sampling]}"


def table_summary(logs: Logs, out: Path, key: str = "score_positional") -> None:
    """T1: evaluate.summarize_run numbers, mean ± std across seeds, plus the (pool − naive) gap per
    target type (H4). Writes t1_summary.tex: a bare booktabs `tabular` (no `table` wrapper; the report
    supplies the caption). `out` is a directory that may not exist yet.

    Also writes t1_summary.txt (the same numbers as plain text) and prints it."""
    stats = summaries(logs, key)
    rows = t1_rows(stats)
    seeds = sorted({len(runs) for runs in logs.values()})
    horizon = common_horizon(logs)
    cut = horizon < max(rows[-1]["games"] for runs in logs.values() for rows in runs)
    cut_note = f"All runs cut at {horizon:,} games (the shortest run). " if cut else ""
    out.mkdir(parents=True, exist_ok=True)

    key_tex = key.replace("_", r"\_")
    tex = [
        r"\begin{tabular}{l" + "c" * len(T1_COLUMNS) + "}",
        r"\toprule",
        " & ".join(["", *(h for _, h, _ in T1_COLUMNS)]) + r" \\",
        r"\midrule",
    ]
    previous = "cond"
    for kind, label, cells in rows:
        if kind != previous:
            tex.append(r"\midrule")
            previous = kind
        tex.append(
            " & ".join(
                [_tex_label(label)]
                + [_fmt(v, s, d, kind == "gap", tex=True) for (v, s), (_, _, d) in zip(cells, T1_COLUMNS)]
            )
            + r" \\"
        )
    tex += [
        r"\bottomrule",
        (
            rf"\multicolumn{{{len(T1_COLUMNS) + 1}}}{{l}}{{\footnotesize Metric: \texttt{{{key_tex}}}; "
            rf"{'/'.join(map(str, seeds))} seed(s) per condition. {cut_note}Final: mean of the last "
            rf"{FINAL_POINTS} eval points. Condition rows: mean $\pm$ std over seeds. "
            r"Gap rows: difference of means $\pm$ standard error.}"
        ),
        r"\end{tabular}",
    ]
    (out / "t1_summary.tex").write_text("\n".join(tex) + "\n")

    width = max(len(label) for _, label, _ in rows) + 2
    lines = [
        f"T1 ({key}; final = mean of the last {FINAL_POINTS} eval points; condition rows mean ± sd "
        f"over seeds, gap rows difference ± SE). {cut_note}".rstrip(),
        f"{'':{width}}"
        + "".join(f"{h.replace('$', '').replace(chr(92) + 'rho', 'rho'):>20}" for _, h, _ in T1_COLUMNS),
    ]
    for kind, label, cells in rows:
        lines.append(
            f"{label:{width}}"
            + "".join(
                f"{_fmt(v, s, d, kind == 'gap', tex=False):>20}"
                for (v, s), (_, _, d) in zip(cells, T1_COLUMNS)
            )
        )
    (out / "t1_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join("       " + line for line in lines))


def _try(name: str, fn: Callable[..., None], *args: Any) -> None:
    try:
        fn(*args)
        print(f"[ok]   {name}")
    except NotImplementedError as e:
        print(f"[todo] {name}: {e}")
    except Skip as e:
        print(f"[skip] {name}: {e}")


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
        key, _ = external_metric(board)
        _try("F3 self-play vs external", fig_selfplay_vs_external, logs, fig_dir, board)
        _try("F4 cross-play", fig_crossplay, exp, args.runs_root, fig_dir)
        _try("F7 checkpoint regret", fig_regret, exp, args.runs_root, fig_dir)
        _try("T1 summary table", table_summary, logs, args.tables / exp, key)
        if board == 4:
            _try("F1 4×4 accuracy", fig_accuracy_4x4, logs, fig_dir)
            _try("F8 true vs checkpoint regret", fig_true_vs_checkpoint_regret, exp, args.runs_root, fig_dir)
        if board in (6, 8):
            _try("F2 scores", fig_scores_6x6, logs, fig_dir)
        if board == 6:
            _try("F5 endgame accuracy", fig_endgame_6x6, logs, fig_dir)


if __name__ == "__main__":
    main()
