"""Live progress bars for training runs, redrawn in place (read-only: it never touches a run).

    python scripts/watch.py                       # every experiment with a run still going
    python scripts/watch.py --exp pilot-6x6 e1-4x4
    python scripts/watch.py --once                # print once (no redraw), e.g. to paste somewhere
    make watch

One bar per run: games played out of the run's fixed budget (`games` in its config.yaml; runs never
stop early), then the latest eval point's external metric (score vs positional; exact accuracy on
4×4) and score vs the previous checkpoint, then an ETA from the run's recent pace.

Progress comes from each run's log.jsonl, which gains a row only at eval points, so bars move in
steps (every 10k games on 6×6, 20k on 8×8). Smooth bars need training to write a small progress
line every round (a to-do before the Dec 1 code freeze, spec §6). A run appears once it has started;
runs a sweep hasn't launched yet aren't shown.

Status: ETA while running; "done" with its total time; "stalled?" when the log hasn't changed for
much longer than the run's usual gap between eval points; "FAILED" when stdout.log shows a
traceback (e.g. a crash at start-up, before any log row).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from othello_rl import rundir

ROOT = Path(__file__).resolve().parents[1]
BLOCKS = " ▏▎▍▌▋▊▉█"  # eighths, for sub-character precision
STALL_FACTOR = 3.0  # "stalled?" after this many times the usual gap between eval points…
STALL_MIN_S = 15 * 60  # …and never sooner than this


@dataclass
class RunStatus:
    exp: str
    name: str  # "<target>-<sampling> s<seed>"
    budget: int
    board: int = 6
    games: int = 0
    metric: float | None = None  # score vs positional, or accuracy on 4×4
    selfplay: float | None = None  # score vs the previous checkpoint
    elapsed: float = 0.0  # wall seconds at the last log row
    rate: float | None = None  # games per second over recent eval intervals
    since_update: float = 0.0  # seconds since the log last changed
    state: str = "running"  # running | done | stalled | failed | starting

    @property
    def fraction(self) -> float:
        return min(1.0, self.games / self.budget) if self.budget else 0.0


def read_rows(path: Path) -> list[dict]:
    """Log rows, ignoring a last line that's still being written."""
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def recent_rate(rows: list[dict], intervals: int = 3) -> float | None:
    """Games per second over the last few eval intervals (pace changes as runs share the GPU)."""
    timed = [r for r in rows if "wall_time" in r]
    if len(timed) < 2:
        return None
    a, b = timed[max(0, len(timed) - 1 - intervals)], timed[-1]
    dt = b["wall_time"] - a["wall_time"]
    return (b["games"] - a["games"]) / dt if dt > 0 else None


def run_status(run: Path, now: float) -> RunStatus:
    cfg = rundir.load_run_config(run)
    target, sampling, seed = rundir.run_condition(run)
    st = RunStatus(
        exp=run.parents[1].name, name=f"{target}-{sampling} s{seed}", budget=cfg.games, board=cfg.board
    )
    log, out = run / rundir.LOG_FILE, run / "stdout.log"
    if out.exists() and "Traceback (most recent call last)" in out.read_text(errors="replace"):
        st.state = "failed"
    if not log.exists():
        if st.state != "failed":
            st.state = "starting"
        return st
    rows = read_rows(log)
    if not rows:
        return st
    last = rows[-1]
    metric_key = "accuracy_4x4" if cfg.board == 4 else "score_positional"
    st.games, st.elapsed = last["games"], last.get("wall_time", 0.0)
    st.metric, st.selfplay = last.get(metric_key), last.get("score_selfplay")
    st.rate = recent_rate(rows)
    st.since_update = now - log.stat().st_mtime
    if st.state == "failed":
        return st
    if st.games >= st.budget:
        st.state = "done"
    elif st.rate and st.since_update > max(STALL_MIN_S, STALL_FACTOR * cfg.eval_every / st.rate):
        st.state = "stalled"
    return st


def bar(fraction: float, width: int) -> str:
    cells = fraction * width
    full = int(cells)
    eighths = int((cells - full) * 8)
    part = BLOCKS[eighths] if full < width and eighths else ""
    return ("█" * full + part).ljust(width, "·")


def _duration(seconds: float) -> str:
    m = int(seconds) // 60
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def _games(n: int) -> str:
    return f"{n / 1_000_000:g}M" if n >= 1_000_000 and n % 100_000 == 0 else f"{n / 1000:g}k"


def _status_text(st: RunStatus) -> str:
    if st.state == "done":
        return f"done in {_duration(st.elapsed)}"
    if st.state == "failed":
        return "FAILED (see stdout.log)"
    if st.state == "starting":
        return "starting"
    if st.state == "stalled":
        return f"stalled? no update for {_duration(st.since_update)}"
    if st.rate:
        return f"ETA {_duration((st.budget - st.games) / st.rate)}"
    return "ETA after the 2nd eval point"


def render(statuses: list[RunStatus], width: int, now: float) -> str:
    """The whole screen: one block per experiment, one line per run."""
    name_w = max((len(s.name) for s in statuses), default=10) + 2
    count_w = max((len(f"{_games(s.games)} / {_games(s.budget)}") for s in statuses), default=12) + 2
    fixed = name_w + count_w + 7 + 9 + 2 + 30  # name, counts, metric, self, gaps, status
    bar_w = max(10, min(50, width - fixed))
    stamp = time.strftime("%a %H:%M:%S", time.localtime(now))
    lines = [f"training progress  {stamp}   (bars move at eval points; Ctrl-C to quit)"]
    for exp in dict.fromkeys(s.exp for s in statuses):
        runs = [s for s in statuses if s.exp == exp]
        label = "acc" if runs[0].board == 4 else "pos"
        lines += ["", f"{exp:{name_w + bar_w + 1 + count_w}}{label:>7}{'prev':>9}"]
        for s in runs:
            counts = f"{_games(s.games)} / {_games(s.budget)}"
            metric = f"{s.metric:7.2f}" if s.metric is not None else f"{'-':>7}"
            prev = f"{s.selfplay:9.2f}" if s.selfplay is not None else f"{'-':>9}"
            lines.append(
                f"{s.name:{name_w}}{bar(s.fraction, bar_w)} {counts:{count_w}}{metric}{prev}  {_status_text(s)}"
            )
    return "\n".join(lines)


def collect(root: Path, exps: list[str] | None, now: float) -> list[RunStatus]:
    """Statuses for the chosen experiments, or (exps=None) for every experiment with a run that
    isn't done yet."""
    statuses = []
    for run in rundir.find_runs(root):
        exp = run.parents[1].name
        if exps is not None and exp not in exps:
            continue
        statuses.append(run_status(run, now))
    if exps is None:
        active = {s.exp for s in statuses if s.state != "done"}
        statuses = [s for s in statuses if s.exp in active]
    return statuses


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--exp", nargs="+", help="experiments to show (default: any with a run still going)")
    p.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    p.add_argument("--every", type=float, default=30.0, help="seconds between redraws")
    p.add_argument("--once", action="store_true", help="print once and exit")
    args = p.parse_args(argv)

    def screen() -> str:
        now = time.time()
        statuses = collect(args.runs_root, args.exp, now)
        if not statuses:
            return "no runs in progress" + ("" if args.exp else " (pass --exp to show finished ones)")
        return render(statuses, shutil.get_terminal_size((100, 24)).columns, now)

    if args.once:
        print(screen())
        return
    try:
        sys.stdout.write("\033[?25l")  # hide the cursor while redrawing
        while True:
            sys.stdout.write("\033[H\033[2J" + screen() + "\n")  # home, clear, draw
            sys.stdout.flush()
            time.sleep(args.every)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h\n")  # show the cursor again


if __name__ == "__main__":
    main()
