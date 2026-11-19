import "./style.css";

// Placeholder until the play view lands (spec §5.1; timeline Thu Dec 3).
// Views to mount here: play (ui/board.ts + agent.ts), watch-it-learn (ui/learn.ts), about.
// Load demo data from `${import.meta.env.BASE_URL}data/manifest.json` so paths work under /othello-rl/.
const app = document.querySelector<HTMLElement>("#app");
if (app) {
  app.innerHTML = `
    <h1>Othello RL</h1>
    <p>A self-play DQN agent for 6×6 Othello. Play it, and watch it learn.</p>
    <p class="muted">Demo under construction.</p>
  `;
}
