import { defineConfig } from "vitest/config";

// Served from the root of its own subdomain, https://othello.johnnygoodlad.dev/ (spec §5.5), so the
// base path is "/". It would only be "/othello-rl/" if the site were served from a repo subpath.
export default defineConfig({
  base: "/",
  test: {
    include: ["tests/**/*.test.ts"],
  },
});
