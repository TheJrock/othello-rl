/**
 * Port of src/othello_rl/engine.py. Keep the function names and conventions identical; the parity
 * tests in tests/engine.parity.test.ts replay Python-generated games (spec §5.4).
 *
 * Representation differs on purpose: an Int8Array of cells (+1 = side to move, -1 = other side,
 * 0 = empty), row-major with index = r * n + c. JS bitwise operators are 32-bit, so a 6×6 bitboard
 * would need BigInt; a cell array is simpler and plenty fast for one game in a browser.
 *
 * Board strings: n rows of n characters joined by "\n"; X = side to move, O = other side, . = empty.
 */

export interface State {
  n: number;
  cells: Int8Array;
  blackToMove: boolean;
}

export interface StepResult {
  next: State;
  reward: number; // for the player who just moved; non-zero only when done
  done: boolean;
}

export const actionSize = (n: number): number => n * n + 1;
export const passAction = (n: number): number => n * n;

const todo = (name: string): Error => new Error(`engine.${name} not implemented`);

/** Same start as engine.initial: Black to move. */
export function initial(n: number): State {
  throw todo("initial");
}

export function fromString(text: string, blackToMove = true): State {
  throw todo("fromString");
}

export function toString(s: State): string {
  throw todo("toString");
}

/** Length n*n + 1; the pass entry is true only when no square is legal and the game is not over. */
export function legalMask(s: State): boolean[] {
  throw todo("legalMask");
}

export function isTerminal(s: State): boolean {
  throw todo("isTerminal");
}

/** For a terminal state: +1 / 0 / -1 for the side to move. */
export function outcome(s: State): number {
  throw todo("outcome");
}

export function step(s: State, action: number): StepResult {
  throw todo("step");
}

/** 3 * n * n floats, channel-major: [side to move, other side, legal squares]. Model input. */
export function toPlanes(s: State): Float32Array {
  throw todo("toPlanes");
}
