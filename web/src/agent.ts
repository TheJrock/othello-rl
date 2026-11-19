import type { State } from "./engine";

/** A trained snapshot running in the browser with onnxruntime-web (spec §5.2). */
export interface Agent {
  /** Raw Q-values, length n*n + 1, in [-1, 1]. Masking happens in chooseMove, not in the model. */
  qValues(s: State): Promise<Float32Array>;
  /** Greedy masked argmax; exact ties broken at random. */
  chooseMove(s: State): Promise<number>;
}

/**
 * Load one snapshot lazily, only when the visitor selects it:
 *
 *   const ort = await import("onnxruntime-web");
 *   ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@<installed version>/dist/";
 *   const session = await ort.InferenceSession.create(modelUrl);
 *   const { q } = await session.run({ planes: new ort.Tensor("float32", toPlanes(s), [1, 3, n, n]) });
 */
export async function loadAgent(modelUrl: string): Promise<Agent> {
  throw new Error("agent.loadAgent not implemented");
}
