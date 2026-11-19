// Data contracts between the Python exporter (src/othello_rl/export.py) and the demo (spec §5.2).

export interface Snapshot {
  id: string; // "untrained" | "early" | "mid" | "final"
  label: string;
  games: number;
  model: string; // path relative to the site base, e.g. "models/final.onnx"
  score_vs_positional: number;
}

export interface Manifest {
  board_size: number;
  condition: string; // e.g. "env-pool"
  snapshots: Snapshot[];
  teaching_positions: string[];
}

// web/tests/fixtures/engine.json
export interface EngineFixturePly {
  board: string;
  black_to_move: boolean;
  legal: number[];
  action: number;
  next_board: string;
  next_black_to_move: boolean;
  reward: number;
  done: boolean;
}

export interface EngineFixtureGame {
  n: number;
  plies: EngineFixturePly[];
}

// web/tests/fixtures/model.json
export interface ModelFixture {
  n: number;
  model: string;
  cases: { board: string; black_to_move: boolean; planes: number[]; q: number[] }[];
}
