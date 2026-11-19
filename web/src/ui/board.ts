import type { State } from "../engine";

export interface BoardOptions {
  /** Squares the human may tap, usually legalMask(s) on the human's turn. */
  clickable: boolean[];
  /** Optional per-square overlay, e.g. win probability (Q + 1) / 2 on the agent's legal moves. */
  overlay?: (number | null)[];
  lastMove?: number;
  onMove?: (action: number) => void;
}

/** Draw the board (SVG or canvas) into `container`. Must fit a 360 px wide phone (spec §5.5). */
export function renderBoard(container: HTMLElement, s: State, opts: BoardOptions): void {
  throw new Error("ui/board.renderBoard not implemented");
}
