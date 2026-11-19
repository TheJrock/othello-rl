import { describe, expect, it } from "vitest";
import { actionSize, passAction } from "../src/engine";

describe("action conventions (same as engine.py)", () => {
  it("has n*n squares plus pass", () => {
    expect(actionSize(6)).toBe(37);
    expect(passAction(6)).toBe(36);
    expect(actionSize(4)).toBe(17);
  });
});
