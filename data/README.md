# data/

Fixed evaluation benchmarks. Build them once with `make data` (seeded), commit them, and don't regenerate them in the middle of an experiment: every run must be scored on identical positions (spec §4.1, D4). `scripts/build_data.py` refuses to overwrite an existing file unless you pass `--force`.

| File | Built by | Contents |
|---|---|---|
| `openings_4x4.json` | `build_data.py openings --board 4` | Every distinct position after 4 plies (120) |
| `openings_6x6.json` | `build_data.py openings --board 6` | 100 distinct positions after 6 random plies (seed 0) |
| `openings_8x8.json` | `build_data.py openings --board 8` | Same as 6×6, for the gated 8×8 add-on (spec §4.8) |
| `endgame6.jsonl` | `build_data.py endgame6` | 500 solved 6×6 positions with 10–12 empties, all won or drawn for the side to move: a lost position has no wrong move, so the filter drops it (spec §4.4) |
| `regret_positions_6x6.json` | `build_data.py regret-positions` | 2,000 positions for the checkpoint-regret analysis (spec §4.7) |
| `teaching_positions.json` | by hand | 2–3 demo positions with ≤ 12 empties (spec §5.3) |
| `cache/solve4x4.npz` | `build_data.py solve4x4` | Full 4×4 solution. Gitignored because it is cheap to rebuild |

## Formats

Boards are strings from the side-to-move's perspective: `X` = side to move, `O` = other side, `.` = empty, one row per line (see `engine.from_string`).

- Position files (`openings_*.json`, `regret_positions_6x6.json`): `[{"board": "...", "black_to_move": true}, ...]`
- `endgame6.jsonl`: one object per line, `{"board": "...", "black_to_move": true, "values": [...]}`. `values[a]` is the win/draw/loss value (+1/0/−1) of action `a` for the side to move, or `null` if `a` is illegal.
- `teaching_positions.json`: `[{"id": "corner_vs_flips", "board": "...", "black_to_move": true, "note": "..."}, ...]`
