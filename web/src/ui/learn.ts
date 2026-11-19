import type { Manifest } from "../types";

/**
 * "Watch it learn" (spec §5.1 item 3): a checkpoint slider driving the precomputed Q heatmaps on the
 * teaching positions (data/heatmaps.json) and the learning curve with the flat self-play line
 * (data/curves.json). Everything is precomputed, so no inference runs here.
 */
export async function mountLearnView(container: HTMLElement, manifest: Manifest, baseUrl: string): Promise<void> {
  throw new Error("ui/learn.mountLearnView not implemented");
}
