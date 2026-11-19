"""Build the fixed benchmark files in data/ (spec §4.4, §4.7; formats in data/README.md).

    python scripts/build_data.py solve4x4
    python scripts/build_data.py openings --board 4
    python scripts/build_data.py openings --board 6
    python scripts/build_data.py endgame6 --workers 8
    python scripts/build_data.py regret-positions

Refuses to overwrite an existing file without --force: once experiments use a benchmark, it must
not change.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from othello_rl import evaluate, regret, solver

DATA = Path(__file__).resolve().parents[1] / "data"
OPENING_DEFAULTS = {4: (2, None), 6: (4, 100)}  # board -> (plies, count); None = every distinct position


def _guard(path: Path, force: bool) -> Path:
    if path.exists() and not force:
        sys.exit(
            f"{path} already exists. Benchmarks must stay fixed across runs; pass --force only if no "
            "experiment depends on it yet."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def cmd_solve4x4(args: argparse.Namespace) -> None:
    out = _guard(args.out, args.force)
    table = solver.solve_4x4()
    solver.save_table(table, out)
    print(f"solved {len(table):,} positions -> {out}")


def cmd_openings(args: argparse.Namespace) -> None:
    plies, count = OPENING_DEFAULTS[args.board]
    plies = args.plies if args.plies is not None else plies
    count = args.count if args.count is not None else count
    out = _guard(args.out or DATA / f"openings_{args.board}x{args.board}.json", args.force)
    bank = evaluate.opening_bank(args.board, plies=plies, count=count, seed=args.seed)
    evaluate.save_states(bank, out)
    print(f"{len(bank)} openings ({plies} plies) -> {out}")


def cmd_endgame6(args: argparse.Namespace) -> None:
    out = _guard(args.out, args.force)
    records = solver.build_endgame_set(
        count=args.count,
        min_empties=args.min_empties,
        max_empties=args.max_empties,
        seed=args.seed,
        workers=args.workers,
    )
    with out.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"{len(records)} solved positions -> {out}")


def cmd_regret_positions(args: argparse.Namespace) -> None:
    out = _guard(args.out, args.force)
    positions = regret.sample_positions(
        n=6, count=args.count, min_empties=args.min_empties, max_empties=args.max_empties, seed=args.seed
    )
    evaluate.save_states(positions, out)
    print(f"{len(positions)} positions -> {out}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--force", action="store_true", help="overwrite an existing file")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("solve4x4", help="full 4×4 solution (gitignored cache)")
    s.add_argument("--out", type=Path, default=DATA / "cache" / "solve4x4.npz")
    s.set_defaults(func=cmd_solve4x4)

    s = sub.add_parser("openings", help="opening bank for one board size")
    s.add_argument("--board", type=int, choices=sorted(OPENING_DEFAULTS), required=True)
    s.add_argument("--plies", type=int)
    s.add_argument("--count", type=int)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", type=Path)
    s.set_defaults(func=cmd_openings)

    s = sub.add_parser("endgame6", help="solved 6×6 endgame test set")
    s.add_argument("--count", type=int, default=500)
    s.add_argument("--min-empties", type=int, default=10)
    s.add_argument("--max-empties", type=int, default=12)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    s.add_argument("--out", type=Path, default=DATA / "endgame6.jsonl")
    s.set_defaults(func=cmd_endgame6)

    s = sub.add_parser("regret-positions", help="6×6 positions for the checkpoint-regret analysis")
    s.add_argument("--count", type=int, default=2000)
    s.add_argument("--min-empties", type=int, default=8)
    s.add_argument("--max-empties", type=int, default=28)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", type=Path, default=DATA / "regret_positions_6x6.json")
    s.set_defaults(func=cmd_regret_positions)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
