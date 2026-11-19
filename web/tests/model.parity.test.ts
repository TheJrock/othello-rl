import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import * as E from "../src/engine";
import type { ModelFixture } from "../src/types";

// Written by scripts/export_demo.py (export.write_model_fixtures). Skipped until it exists.
const path = fileURLToPath(new URL("./fixtures/model.json", import.meta.url));
const fixture: ModelFixture | null = existsSync(path) ? JSON.parse(readFileSync(path, "utf8")) : null;

describe.skipIf(!fixture)("model parity with PyTorch (spec §5.4)", () => {
  it("builds the same input planes as engine.to_planes", () => {
    for (const c of fixture!.cases) {
      expect(Array.from(E.toPlanes(E.fromString(c.board, c.black_to_move)))).toEqual(c.planes);
    }
  });

  // Load the bytes of web/public/<fixture.model> into an onnxruntime-web session and compare its Q-values
  // with c.q (max abs diff < 1e-4). If onnxruntime-web won't initialise under Node, use onnxruntime-node
  // for this test; it runs the same .onnx file.
  it.todo("onnxruntime-web Q-values match PyTorch within 1e-4");
});
