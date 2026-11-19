import { defineConfig } from "vitest/config";

// If the portfolio is the <username>.github.io repo on a custom domain, this repo's Pages site is
// served at https://<your-domain>/othello-rl/ (spec §5.5). Change `base` if the repo is renamed or
// the build is copied into a folder of the portfolio repo instead.
export default defineConfig({
  base: "/othello-rl/",
  test: {
    include: ["tests/**/*.test.ts"],
  },
});
