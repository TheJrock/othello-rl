import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import * as E from "../src/engine";
import type { EngineFixtureGame } from "../src/types";

// Written by scripts/export_demo.py (export.write_engine_fixtures). Skipped until it exists.
const path = fileURLToPath(new URL("./fixtures/engine.json", import.meta.url));
const games: EngineFixtureGame[] | null = existsSync(path) ? JSON.parse(readFileSync(path, "utf8")) : null;

describe.skipIf(!games)("engine parity with engine.py (spec §5.4)", () => {
  it("matches legal moves, next states and rewards on every ply", () => {
    for (const game of games!) {
      for (const ply of game.plies) {
        const s = E.fromString(ply.board, ply.black_to_move);
        const legal = E.legalMask(s).flatMap((ok, i) => (ok ? [i] : []));
        expect(legal).toEqual(ply.legal);
        const { next, reward, done } = E.step(s, ply.action);
        expect(E.toString(next)).toBe(ply.next_board);
        expect(next.blackToMove).toBe(ply.next_black_to_move);
        expect([reward, done]).toEqual([ply.reward, ply.done]);
      }
    }
  });
});
