"""Export one run for the web demo (spec §5): ONNX snapshots, manifest/curves/heatmaps JSON, and the
Python→TypeScript parity fixtures.

    python scripts/export_demo.py --run runs/e2-6x6/env-pool/seed0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from othello_rl import export, nets, rundir

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--web", type=Path, default=ROOT / "web")
    p.add_argument("--teaching", type=Path, default=ROOT / "data" / "teaching_positions.json")
    args = p.parse_args(argv)

    cfg = rundir.load_run_config(args.run)
    rows = rundir.read_log(args.run)
    models = args.web / "public" / "models"
    data = args.web / "public" / "data"
    fixtures = args.web / "tests" / "fixtures"
    for d in (models, data, fixtures):
        d.mkdir(parents=True, exist_ok=True)

    snapshots = export.select_snapshots(rows)
    for snap_id, games in snapshots.items():
        net = nets.load_net(rundir.checkpoint_path(args.run, games), cfg)
        export.export_onnx(net, cfg.board, models / f"{snap_id}.onnx")
        print(f"{snap_id:>9}: games {games:,} -> {models / f'{snap_id}.onnx'}")

    teaching = export.load_teaching_positions(args.teaching) if args.teaching.exists() else []
    export.write_manifest(data / "manifest.json", cfg, args.run, snapshots, rows, [t["id"] for t in teaching])
    export.write_curves(data / "curves.json", rows)
    export.write_heatmaps(data / "heatmaps.json", args.run, cfg, teaching)

    export.write_engine_fixtures(fixtures / "engine.json", n=cfg.board)
    final = nets.load_net(rundir.checkpoint_path(args.run, snapshots["final"]), cfg)
    export.write_model_fixtures(final, fixtures / "model.json", n=cfg.board)
    print(f"demo data -> {data}; fixtures -> {fixtures}")


if __name__ == "__main__":
    main()
