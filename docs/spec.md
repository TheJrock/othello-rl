# Othello Self-Play — Project Spec & Timeline

Companion to `docs/proposal.md` · Written Wed Nov 18, 2026 · Rev. 4 (2×2 design, MADRID-style checkpoint regret, repo scaffold; Nov 19: gated 8×8 add-on, D9 and §4.8; Nov 23: deeper opening banks, §4.7; a demo pass rule, §5.1; both sides' transitions stored, D2; Nov 24: debugging ladder results and 4×4 vs-random bars, §4.6; `eps_end` 0.1 and `target_update_every` 250 on all boards, §4.5, with E7 to compare; throughput on laptop and server, pilot budgets and gate 1, §4.8 and §6)

| Date | Milestone |
|---|---|
| Tue Nov 24 | 8×8 gate 1: throughput (§4.8) |
| **Wed Nov 25** | **M1** — 4×4 DQN learning under both target types and checked against the exact solver; 4×4 sweep and 6×6 pilot runs launched to run over Thanksgiving, plus the 8×8 pilot (gate 1 passed Nov 24), all on the laptop GPU |
| Thu Nov 26 – Sat Nov 28 | Thanksgiving: light days, no new code on the critical path |
| Mon Nov 30 | 8×8 gate 2: pilot review (§4.8) |
| **Tue Dec 1** | **M2** — Experiment code frozen; final 6×6 runs launched (plus the 8×8 pair if gate 2 passed) |
| **Sat Dec 5** | **M3** — All figures final; demo's play view live on your domain |
| **Thu Dec 10** | Presentation |
| **Thu Dec 17** | Report due |

Sundays are left unscheduled as rest/buffer.

**Done by Dec 10 means:** (1) 4×4 and 6×6 results for the 2×2 design (target type × opponent sampling) across seeds, (2) the self-play-win-rate-vs-external-metric figure and the checkpoint-regret figure, (3) a rehearsed deck, (4) a playable demo on your portfolio domain. 8×8 is not part of this list. It is a gated Could (D9, §4.8).

---

## 1. Research question, design, hypotheses

**Question (from proposal):** Does training against a pool of past checkpoints produce stronger, more stable Othello agents than naive latest-vs-latest self-play, and does self-play win rate reveal the difference?

**Design: 2×2.** Two target formulations × two opponent-sampling schemes. Everything else is shared.

| | Naive (latest vs latest) | Pool (past checkpoints) |
|---|---|---|
| **Opponent-as-environment** targets | A-naive | A-pool |
| **Negamax** targets | N-naive | N-pool |

| | Hypothesis | Evidence | What would change my mind |
|---|---|---|---|
| H1 | Pool agents end stronger | Final score vs reference opponents (6×6); optimal-move accuracy (4×4) | Naive ≥ pool on most seeds under both target types |
| H2 | Pool agents are more stable | Smaller drawdowns after peak; fewer forgetting events; lower self-history regret (§4.7) | Naive curves as smooth as pool; earlier checkpoints rarely beat the final naive agent |
| H3 | Self-play win rate hides the difference | Self-play score stays roughly 45–65% in all conditions; weak correlation with external metrics; positions where earlier checkpoints beat the final agent are invisible to it | Self-play score tracks external score closely (Spearman ρ > 0.7) |
| **H4** | **The pool matters more under opponent-as-environment targets than under negamax** | Interaction: (pool − naive) gap is larger in the A row than the N row | Similar gaps in both rows. That would mean the pool helps through data diversity, which both rows share, rather than by fixing the opponent assumption |

A null result on H1/H2 is still a valid finding ("at this scale, naive self-play doesn't forget"). H3 and H4 make the presentation work either way.

---

## 2. Scope

| Tier | Items |
|---|---|
| **Must** (Dec 10) | Engine + tests · random/greedy/positional opponents · 4×4 exact solver · DQN with both target types · 2×2 on 4×4 (5 seeds) and 6×6 (3 seeds; fallback in §4.8) · external-metric and self-play curves · demo play view · deck |
| **Should** | MADRID-style checkpoint regret analysis (§4.7) · 6×6 endgame-accuracy metric · cross-play matrices · "Watch it learn" view (can slip into Dec 11–16) |
| **Could** | **8×8 A pair (A-naive vs A-pool, 1 seed each), only if both gates pass (D9, §4.8)** · snapshot-vs-snapshot autoplay in the demo · let demo visitors pick an A or N agent · 4×4 "perfect play" coach mode · replay-buffer-size ablation for A-naive (§3 D1) · pool-mix ablation (80/20) |
| **Won't** | Anything on 8×8 beyond the gated A pair: the N row, more seeds, cross-play, regret, endgame accuracy (future work) · MAP-Elites/CMA-ME search. Small Othello position sets can be enumerated or sampled directly (§4.7). |

---

## 3. Design decisions (read before coding)

**D1. Run both target formulations. Your proposal already contains both.**
The proposal's MDP row ("opponent's policy is part of the environment") describes formulation A. Its Method section ("negamax targets") describes formulation N. They make different assumptions about the opponent, and that difference is what decides whether a pool should matter.

- **A — Opponent-as-environment.** The learner moves, then the opponent replies, and `s'` is the learner's next turn. The target is `r + max_a' Q(s', a')` with no sign flip. Q estimates value *against the opponents it trained with*, so the agent learns a **best response to its opponent distribution**.
  - Naive: the opponent is its latest self, so it chases a moving best response. That is the classic setup for cycling and forgetting.
  - Pool: the MDP becomes a slowly changing mixture over past checkpoints.
  - The replayed transitions were generated under *older opponents*, i.e. a different transition function (the proposal's Lectures 14–15 connection). Transitions are stochastic whenever the opponent is (ε-greedy or sampled from a pool).
- **N — Negamax.** `s'` is the opponent's turn, and the target is `−max_a' Q(s', a')`. Your own network stands in for the opponent's best reply, so the target is the game's minimax value whatever the real opponent does. The pool only changes *which positions reach the replay buffer*. The non-stationarity is the shifting state distribution plus bootstrapping from a moving network.
- **Prediction (H4):** the pool helps A more than N. The literature leans this way: AlphaGo's policy-gradient stage sampled past opponents, while AlphaZero's minimax-style value targets used only the latest network. On small Othello boards (fairly transitive games) the effect may be small. Either outcome is reportable.
- **Note for the analysis:** under A-naive, the replay buffer already holds transitions from recent past selves, so the buffer acts as an *implicit short-window pool*. With 600k transitions on 6×6 (both sides' moves, D2), that is roughly the last ~10% of training. Report this; never tune buffer size per condition.
- **If you could only run one,** run A. It is the formulation where the pool's mechanism actually operates, and it matches the proposal's MDP and course connection.

**D2. Store both sides' moves, in all four conditions.** *(Revised Nov 23 at the instructor's direction. It was "learner's moves only".)* In each game, one side is "the learner" (the current network: Black in half the games, White in the other half). The other side is the opponent the sampler chose. Every ply by either side becomes a transition from that side's own perspective (`selfplay.cut_transitions`, called once per color), and all of them enter replay. Q-learning is off-policy, so a move is valid training data whoever played it. Data volume stays identical across the 2×2 because every condition stores both sides, so the *only* differences are still the target and the opponent sampling.
- **What the opponent's transitions mean in each cell:**
  - **N:** the target doesn't depend on who moved, so the opponent's moves are plain extra data, as in AlphaZero-style training on both players' positions. Under the pool, the snapshots' moves are more off-policy data.
  - **A-naive:** the opponent is the learner itself, so its transitions come from the same MDP.
  - **A-pool:** the opponent's transitions have the *current learner* as their environment, not the pool.
- **Consequence for H1/H4 (report it):** in A-pool, past snapshots are now the environment for about **25%** of all transitions: the learner's side, in the half of games played against a past snapshot. With learner-only data that share was 50%. The other 75% have the current network as their environment, and in A-naive that's 100%. So the A-row contrast between pool and naive is weaker than the 50/50 mix suggests, which works against finding H1/H4 effects. `pool_p_latest` stays at 0.5 (decided Nov 23). It's fixed before E1 and never changed after seeing results.
- **Cost:** twice the transitions per game, so twice the gradient steps per game at a fixed `transitions_per_grad_step` (§4.8). Replay capacities are doubled so the buffer still spans ~10% of training (§4.5).

**D3. Passes.**
- **N:** action space = N² squares + 1 pass. Pass is legal **only** when no square is legal. A forced pass is a real transition, so the sign flips consistently.
- **A:** the environment step applies the learner's move, then lets the opponent play (including any forced passes on either side) until the learner has a legal square or the game ends. The learner never sees a pass state, so passes are absorbed into the dynamics. This contrast is good material for the "two views" slide.
- **Both:** terminal `r` = outcome for the learner (+1/0/−1). Mask illegal actions with `−1e9`, not `−inf`. Select the terminal target with `torch.where` so NaNs can't leak in.

**D4. Evaluation needs opening randomization.** Two deterministic greedy policies play exactly two distinct games (one per color). All evaluation games start from a fixed **opening bank**, and each opening is played with both colors.

**D5. "Self-play win rate" = current network vs. the previous checkpoint**, from the opening bank, greedy play. This is the number a practitioner would watch. (Latest vs itself is trivially 50%.)

**D6. Fairness.** All four conditions use identical hyperparameters, game budgets, and gradient-step budgets. Tune shared knobs only far enough that the **naive pilots learn**, and never tune per condition.

**D7. The demo is fully static: inference runs in the browser** (ONNX + `onnxruntime-web`, rules ported to TypeScript, hosted on GitHub Pages). **Your server is for training compute, not hosting.**

**D8. Checkpoints have two jobs: training opponents (pool) and evaluation references (MADRID).** Keep every checkpoint from every run; they're ~0.5 MB each. Caveat: a pool agent was *trained against* its own checkpoints, so regret measured against its own history is biased in its favor. Always also report regret against a **common held-out reference set** (§4.7).

**D9. 8×8 is an add-on, not a third board in the design.** It asks one question: does the 6×6 A-row result (pool vs naive) still hold on the full board? It runs only the A pair, because A is where the pool's mechanism operates (D1). With one seed per arm it is illustrative and never evidence for H1–H4 on its own. It runs only if it costs nothing on the critical path:
- It runs on hardware E2 doesn't use, such as the laptop GPU (`--device mps`).
- It needs no new code after M2, only `configs/8x8.yaml`.
- It never takes time from Dec 7–9.

Two gates decide whether it runs at all (§4.8). Code costs almost nothing extra because the engine, players, network, and web engine are all size-generic. Compute is the limit.

---

## 4. System spec

### 4.1 Repo layout (`othello-rl`)

```
othello-rl/
├── README.md
├── pyproject.toml             # package + deps; `pip install -e ".[dev]"`
├── Makefile                   # test / data / sweep-4x4 / sweep-6x6 / sweep-8x8 / figures / demo
├── .gitignore                 # runs/ *.pt .venv/ __pycache__/ node_modules/ web/dist/ docs/private/
├── docs/
│   ├── proposal.md
│   ├── spec.md                # this file
│   └── private/               # gitignored: course directions PDF, papers
├── src/othello_rl/
│   ├── engine.py              # rules only (no torch): bitboards, legal_mask, step, to_planes, D4 transforms
│   ├── players.py             # Player protocol; Random, Greedy, Positional, NetPlayer
│   ├── solver.py              # 4×4 full solve + π*; 6×6 endgame WLD solver
│   ├── nets.py                # conv Q-network
│   ├── replay.py              # buffer; random D4 transform at sample time
│   ├── targets.py             # Double-DQN target; `target` = env | negamax (the sign)
│   ├── selfplay.py            # batched games; cuts transitions per `target` (A: through opponent reply)
│   ├── pool.py                # opponent sampling: `sampling` = naive | pool; loads from the run's ckpt/
│   ├── train.py               # main loop: `python -m othello_rl.train --config ... --target ... --sampling ... --seed ...`
│   ├── evaluate.py            # opening banks, matches, score + Wilson CI, cross-play
│   ├── regret.py              # MADRID-style position regret (§4.7)
│   ├── export.py              # ONNX, parity fixtures, demo JSON
│   ├── config.py              # dataclass config; YAML + CLI overrides
│   └── rundir.py              # run dirs, meta.json, checkpoint paths, JSONL logs
├── configs/
│   ├── 4x4.yaml               # board-size defaults; target/sampling/seed come from the CLI
│   ├── 6x6.yaml
│   └── 8x8.yaml               # gated add-on (D9)
├── data/                      # fixed benchmarks: generated once by a seeded script, committed
│   ├── openings_4x4.json  openings_6x6.json  openings_8x8.json
│   ├── endgame6.jsonl
│   ├── regret_positions_6x6.json
│   ├── teaching_positions.json
│   └── cache/                 # gitignored: solve4x4.npz (cheap to rebuild)
├── scripts/                   # thin CLIs; logic lives in the package
│   ├── build_data.py          # solve 4×4; build openings / endgame / regret / teaching sets
│   ├── sweep.py               # launch the 2×2×seeds grid as parallel processes
│   ├── analyze.py             # runs/ → figures/ + tables/
│   └── export_demo.py         # chosen run → web/public/
├── tests/
│   ├── conftest.py            # NotImplementedError → "skipped: TODO" until Dec 1, so CI stays green
│   ├── test_infra.py          # config + run-dir plumbing
│   ├── test_engine.py         # perft, flips, passes, terminal, symmetry round-trip
│   ├── test_solver.py
│   ├── test_targets.py        # both modes: sign, forced pass, terminal, no NaNs
│   ├── test_selfplay.py       # A-mode step: opponent reply ends game, auto-pass
│   └── test_export.py         # torch vs ONNX outputs
├── runs/                      # gitignored; rsync from the server
│   └── e2-6x6/env-pool/seed0/{config.yaml, meta.json, log.jsonl, stdout.log, ckpt/g0005000.pt …}
├── figures/  tables/          # written by analyze.py; committed (the report uses them)
├── web/
│   ├── package.json  vite.config.ts  index.html
│   ├── src/{main.ts, engine.ts, agent.ts, ui/…}     # engine.ts mirrors engine.py's API names
│   ├── public/{models/*.onnx, data/*.json}         # written by export_demo.py, committed
│   └── tests/{engine.parity.test.ts, model.parity.test.ts, fixtures/}
├── report/                    # main.tex (NeurIPS 2025 style) + references.bib; `make report`
└── .github/workflows/
    ├── tests.yml              # pytest + vitest on push
    └── pages.yml              # build web/ and deploy to GitHub Pages
```

**Rules that keep this honest:**
- **Data flows one way.** package → `runs/` → `analyze.py` → `figures/`, `tables/`; and `runs/` → `export_demo.py` → `web/public/`. Training never plots and analysis never trains, so you can re-plot on the laptop after an `rsync` without touching the server.
- **The 2×2 lives in config, not in code forks.** `target` is read only in `selfplay.py` and `targets.py`; `sampling` is read only in `pool.py`. A grep then shows the conditions differ only there (useful in the report).
- **`data/` is the benchmark.** Every run evaluates on identical, committed positions. Never regenerate it mid-experiment.
- **Checkpoints are the pool.** `pool.py` loads snapshots from the run's own `ckpt/`, and `regret.py` uses the same files (D8). There is no separate storage.
- **`meta.json`** per run: git commit hash, host, start time, torch version. Run directories are named `<exp>/<target>-<sampling>/seed<k>` (exp = `dev`, `e1-4x4`, `pilot-6x6`, `e2-6x6`, `pilot-8x8`, `e6-8x8`) so `analyze.py` can glob them.

### 4.2 Engine (`engine.py`)

- Board size N ∈ {4, 6, 8}. State = `(me_bits, opp_bits)` as Python ints, from the side-to-move perspective, plus a color flag (display only).
- Move generation by directional shifts with edge masks (±1, ±N, ±(N±1)).
- API: `initial(n)`, `legal_mask(s)`, `step(s, a) -> (s', r, done)`, `score(s)`, `to_planes(s) -> float32[3,N,N]` (mine, theirs, legal), `transform(planes, action, k)` for the 8 D4 symmetries (pass maps to pass).
- **Tests:** 8×8 perft from the start position (depths 1–6: 4, 12, 56, 244, 1396, 8200; confirm against a published table) · flips in all 8 directions · forced pass · double-pass terminal · symmetry round-trip · 10k random games without exceptions or illegal states.

### 4.3 Reference opponents (`players.py`)

| Player | Rule (ties broken uniformly at random) |
|---|---|
| Random | Uniform over legal moves |
| Greedy | Maximize discs flipped |
| Positional | Maximize square weight; ties → most flips |

```
6×6 weights                      4×4 weights
 100 -20  10  10 -20 100          100 -20 -20 100
 -20 -50  -2  -2 -50 -20          -20   0   0 -20
  10  -2   1   1  -2  10          -20   0   0 -20
  10  -2   1   1  -2  10          100 -20 -20 100
 -20 -50  -2  -2 -50 -20
 100 -20  10  10 -20 100

8×8 weights (the standard table; gated add-on, D9)
 100 -20  10   5   5  10 -20 100
 -20 -50  -2  -2  -2  -2 -50 -20
  10  -2  -1  -1  -1  -1  -2  10
   5  -2  -1  -1  -1  -1  -2   5
   5  -2  -1  -1  -1  -1  -2   5
  10  -2  -1  -1  -1  -1  -2  10
 -20 -50  -2  -2  -2  -2 -50 -20
 100 -20  10   5   5  10 -20 100
```

Sanity check: on 6×6 (and 8×8), positional ≫ random and greedy > random.

Measured Nov 23 on the `data/` opening banks (§4.7), each opening played from both sides; score of the first player with its 95% Wilson interval:

| | Games | greedy vs random | positional vs random | positional vs greedy |
|---|---|---|---|---|
| 4×4 | 240 | 48% [42–54] | 68% [61–73] | 70% [64–75] |
| 6×6 | 200 | 63% [56–69] | 82% [76–87] | 79% [73–84] |
| 8×8 | 200 | 66% [59–72] | 81% [75–86] | 90% [85–93] |

Both checks pass on 6×6 and 8×8. On 4×4, greedy is no better than random: grabbing discs mostly hands over edges and corners on a board that small. That's fine, because 4×4 is judged against the exact solver.

### 4.4 Solvers (`solver.py`)

- **4×4 full solve:** memoized negamax over win/loss/draw values for every reachable position, giving the value of every (position, move) pair and an optimal policy π\*. The start position should match the published result (4×4 and 6×6 are both reported as second-player wins; confirm the source).
  - *Confirmed Nov 23.* The solve covers all 58,533 reachable positions (0.45 s). It has no alpha-beta cutoffs, so every position the regret analysis might look up is in the table. V(start) = −1 for Black: White wins. A separate disc-margin solve gives exactly **3–11** under perfect play, with 2 squares left empty. This matches the published 4×4 result (White wins 3–11; Wikipedia, *Computer Othello*), and `tests/test_solver.py` now checks it.
- **4×4 eval set:** all reachable non-terminal positions with ≥2 legal moves whose moves don't all have the same value. Also compute **chance accuracy** (expected accuracy of a uniform-random mover) for a baseline line on plots.
  - *Measured Nov 23:* 7,912 positions; chance accuracy **49.4%**. A random player's measured accuracy matches it.
- **6×6 endgame solver:** alpha-beta WLD search with a null window, move ordering (corners first, then fewest opponent replies), and a transposition table.
- **6×6 endgame test set:** 500 positions with 10–12 empties, sampled from games between randomized reference players, filtered as above. Solve once with `multiprocessing` and cache to `data/endgame6.jsonl`.

### 4.5 DQN (`nets.py`, `replay.py`, `targets.py`)

**Network** (small, ONNX-friendly, no batch norm): trunk of L conv layers (3×3, 64 channels, ReLU, same padding) → head: 1×1 conv to 2 channels → flatten → linear → N²+1 outputs → `tanh`. L = 3 for 4×4, L = 4 for 6×6, and L = 5 for 8×8. About 120k parameters (~0.5 MB); about 160k on 8×8.
- *Exact counts (Nov 23):* 76,339 (4×4), 115,407 (6×6), 158,019 (8×8).
- **No ReLU after the 1×1 head conv. This is deliberate.** With one there (as an early `nets.py` docstring had it), both head channels start out negative on every input in 15–25% of initializations. The output is then constant, no gradient reaches the trunk, and that seed never learns, which would look like an RL failure. Without the ReLU, the head is a linear read-out of the trunk and can't die. `tests/test_nets.py` checks 50 initialization seeds per board size.

**Loss (Double DQN, γ = 1).** The only line that differs between the two target types is the sign:
```
q_sa   = online(s)[a]
a_star = argmax_{a' legal in s'} online(s')[a']      # select with online net
q_next = target(s')[a_star]                          # evaluate with target net

# N (negamax):                 s' = opponent to move
y = where(done, r, -q_next)
# A (opponent-as-environment): s' = learner to move again, after the opponent's reply
y = where(done, r, +q_next)

loss = huber(q_sa, y)
```
Replay tuple: `(planes_s, a, r, planes_s', legal_mask_s', done)`. At sample time, apply one random D4 transform consistently to `s`, `a`, `s'`, and the mask.

**Target unit tests.**
- N: (1) a terminal winning move gives y = +1 for the mover; (2) across a forced-pass state, the sign flips twice correctly; (3) no NaNs when `s'` is terminal.
- A: (4) the opponent's reply ends the game → `r` is from the learner's perspective; (5) the opponent has no move → the learner moves again and `s'` is correct; (6) the learner has no move → the environment auto-passes.
- *Implemented Nov 23.* Tensor-level cases are in `tests/test_targets.py`. Cases (1)–(6) are checked end to end in `tests/test_selfplay.py` with the 4×4 solver's exact values standing in for a perfect Q:
  - **N:** every negamax target from random games equals the exact value of the move.
  - **A against a perfect opponent:** every env target also equals the exact value.
  - **A against a random opponent:** some env targets differ from the exact value. This is D1 in one test: A learns value against *its* opponents.
  - Each test also checks that its games contain the pass and terminal cases.

**4×4 Q-vs-solver MSE applies to N only.** A's Q estimates value against its training opponents, not the minimax value. Optimal-move accuracy applies to both.

**Starting hyperparameters** (shared by all four conditions):

| | 4×4 | 6×6 | 8×8 (gated, D9) |
|---|---|---|---|
| Training games | 30k | 400k for the pilots (Nov 24). E2: 200k–400k, set Nov 30 from the pilot curves | 1M for P8 (gate 1, Nov 24). E6: ≤ 1M, set at gate 2 from P8's curve |
| Parallel games | 128 | 256 | 256 |
| Replay capacity | 100k transitions (~13 MB) | 1.28M transitions (≈ last 10% of training; ~340 MB) | 6M transitions (≈ last 10% of training; ~2.8 GB) |
| Batch / optimizer | 256 / Adam 3e-4 | same | same |
| Gradient steps | 1 per 16 new transitions, both sides (each sampled ≈ 16×) | same | same |
| Target net | Hard update every 250 gradient steps (1,000 before Nov 24) | same | same |
| Exploration | ε linear 1.0 → 0.1 over first 20% of games, then 0.1 (0.05 before Nov 24) | same | same |
| Loss / clipping | Huber, grad-norm clip 10 | same | same |
| Eval + checkpoint every | 1k games (31 points) | 10k games (41 points) | 20k games (51 points) |
| Seeds | 5 | 3 | 1 per arm |
| Device | GPU (`mps`) where available: ~2–5 min/run; a CPU gradient step is ~50 ms even here (§4.8) | GPU (CPU-only exceeds 6 h/run, §4.8) | GPU only (`--device mps` or `cuda`) |

Replay capacities were doubled on Nov 23, when D2 began storing both sides' moves. That keeps each buffer at ≈ 10% of training's transitions.

**Budget rule (Nov 24).** When a 6×6 or 8×8 budget changes, rescale the other two settings with it. Keep `replay_capacity` ≈ 10% of training's transitions (games × 32 on 6×6, × 60 on 8×8) and `eval_every` ≈ games / 40 on 6×6 or / 50 on 8×8. That keeps the buffer's reach and the pool's snapshot count the same at any budget. For example, E2 at 200k games means `replay_capacity: 640000` and `eval_every: 5000`.

**Settings change (Nov 24, before any experiment run).** `eps_end` went from 0.05 to 0.1 and `target_update_every` from 1,000 to 250, on all three boards and for all four conditions. The debugging ladder (§4.6) showed slow value propagation: with γ = 1, a value moves back one decision per target update, and a 4×4 run got only ~11–23 updates. It also showed narrow coverage under naive self-play. A one-seed rerun with both changes sped up early learning (N's 4×4 accuracy at 2k games went from 57% to 71%), raised late accuracy 2–3 points on the self-play rungs, and showed no instability. Neither change costs measurable time: a target update is a ~0.2 ms weight copy, which adds ≈ 0.3 s per 6×6 run and ≈ 1.6 s per 8×8 run, and ε doesn't change the number of forward passes or transitions. Target updates per run are now ≈ 92 on 4×4, ≈ 1,600 on 6×6 and ≈ 7,600 on 8×8. The evidence is from 4×4 only. 6×6 and 8×8 already got ~400 and ~1,900 updates per run at 1,000, so the change was applied there for consistency and is untested on those boards. **E7** (§4.8) reruns the old settings on the same code to measure the difference.

### 4.6 Training loop and opponent pool (`selfplay.py`, `pool.py`, `train.py`)

- Run B games in parallel. Each ply, batch every position needing a given network into **one forward pass**.
- The learner plays Black in half the games and White in the other half. Both sides play ε-greedy with the current ε. **Both sides' transitions are stored, each from its own perspective (D2).** For "mover", read "whichever side the transition belongs to":
  - **N transition:** one ply by the mover, `s → s'` (the other side to move, possibly a forced-pass state).
  - **A transition:** the mover's ply + the other side's reply + any forced passes, `s → s'` (the mover to move again with a legal square, or terminal).
- **Naive:** opponent = the current online network.
- **Pool:** add a frozen copy every checkpoint interval (keep all). Each round, split the B games into 8 groups. Each group's opponent is the latest network with p = 0.5, otherwise one drawn uniformly from past snapshots. (MADRID uses checkpoints only *after* training, so it doesn't prescribe a training-time sampling rule. This 50/50 mix is a common default; if you want a cited rule, check how the TiZero paper, its ref. [29], samples its self-play opponents.)
  - *Implemented Nov 24 (`pool.PoolSampler`).* Snapshots are loaded back from the checkpoint files (D8), so the pool is exactly the saved checkpoints, and they're frozen (eval mode, no gradients). They play with the learner's current ε. Two groups that draw the same snapshot get the same object, so their games share one forward pass. Each log row adds `pool_size` (snapshots available since the last eval point) and `opp_past_frac` (the fraction of groups that played a past snapshot, ≈ 1 − `pool_p_latest`). Tests cover the mix for p = 0, 0.5, 0.8 and 1, uniform draws, frozen weights, ε, and batching. Seven planted bugs were all caught, including "snapshot = the live network", which would silently turn the pool into naive play.
- Log one JSONL line per eval point: games, grad steps, wall time, ε, mean loss, mean max-Q, score vs each reference opponent (+ Wilson CI), self-play score (D5), 4×4 accuracy (+ Q-vs-solver MSE for N), 6×6 endgame accuracy.
  - *Implemented Nov 24* (`train.train`, `evaluate.evaluate_checkpoint`). Each row also has `buffer` (replay size), `pool_size` and `opp_past_frac` (pool runs), `grad_norm` (before clipping), `plies_per_game`, `transitions_per_game`, `learner_score` (training games), `mean_target`, and `q_corr_4x4`, the Pearson correlation of Q with the exact values. Unlike the MSE, it ignores scale, so it shows whether Q ranks moves correctly before its magnitudes are right. Training stats are means since the previous eval point. `epsilon` is the ε the learner actually used. Eval points land exactly on multiples of `eval_every` (the round before one is cut short), so `games`, `eval_every` and `parallel_games` must all divide by `pool_groups`.

**Debugging ladder.** Get each rung working before the next. Rung 1 is also a useful "stationary MDP" reference point for the talk.
1. A vs a **fixed random opponent**: plain single-agent DQN on a stationary MDP. It should beat random ≥ 95% quickly *from the start position* (see the bars below).
2. A-naive self-play.
3. N-naive self-play (check Q against solver values).
4. Pool sampling for both.

*Rung 1 setup (Nov 24).* `--sampling random` is a debug-only value (`config.DEBUG_SAMPLINGS`), never swept. Only the learner's moves are stored (`pool.FixedSampler.store_opponent_moves = False`). The fixed opponent's moves would be transitions whose environment is the learner, which keeps changing, so the MDP would no longer be stationary. The one network would also be fitting values against two different opponents. Rung 1 is A only: N needs both sides' moves, since its bootstrap states are the opponent's positions.

*Bars for 4×4, from an exact solve against random (Nov 24, `solver.solve_vs_random_4x4`, tests in `tests/test_solver.py`).* Against a random opponent the best policy is expectimax, not π\*. Exact expected scores, each start played from both sides:

| Player vs random | 4×4 bank | Start position |
|---|---|---|
| Random | 50.0% | 50.0% |
| π\* (minimax-perfect) | 77.8% | 88.1% |
| Best possible (expectimax) | **83.4%** | **98.0%** |

So "≥ 95% vs random" can only apply from the start position. On the 4×4 bank even the best possible player scores 83.4%: four random plies leave about four decisions per side, and many openings are lost however you play. The Tue Nov 24 bar "≥ 90% vs random" fails for π\* itself in both columns, so for rungs 2–3 it becomes "close to π\*". On 6×6 and 8×8 the ≥ 95% intuition is fine; random is far weaker there (§4.3).

*Results (Nov 24, 4×4, seed 0, 30k games, `mps`).* Logs are in `runs/ladder/`; `scripts/ladder_check.py` recomputes the exact columns from the final checkpoints. "Self-play positions" are the ~200 distinct positions with ≥ 2 legal moves that the final net reaches playing itself (ε = 0.05) from the start, i.e. where its training data comes from.

| | Rung 1: A vs random | Rung 2: A-naive | Rung 3: N-naive |
|---|---|---|---|
| Exact score vs random: start / bank | 96.7% / 74.2% | 95.2% / 71.8% | 91.8% / 70.3% |
| Minimax accuracy: self-play positions / all 7,912 (chance 49.4%) | 92.0% / 72.9% | **99.0%** / 72.6% | **97.9%** / 74.8% |
| Q corr with minimax values: self-play positions / all | 0.67 / 0.38 | **0.87** / 0.40 | **0.82** / 0.41 |
| Final TD loss | 0.022 | 0.015 | 0.003 |
| Wall time (rungs 2–3 ran at the same time) | 2¼ min | 5½ min | 5½ min |

- **Rung 1 passes.** From the start it scores 96.7% (best possible 98.0%). On the positions it reaches against random, it picks a best-vs-random move 94% of the time, and its Q matches the exact values of its MDP (corr 0.80).
- **Rungs 2–3 pass on the machinery.** Where their data comes from, both nets play the minimax move ~98–99% of the time, and Q correlates 0.82–0.87 with the exact values. The negamax target is right, and the D2 cutting is too.
- **The whole-set numbers are a coverage finding, not a bug.** Naive self-play at ε = 0.05 keeps replaying a few lines from the start: ~200 distinct decision positions out of 7,912. Off those lines Q is barely informative (corr ~0.4, MSE ~0.96, about what Q = 0 would score). So 4×4 accuracy plateaus around 73–75% by ~10k games. That's above chance, but far from the ~98% the nets reach on their own lines. This is H1–H2's narrow-distribution story in miniature. Pool sampling (rung 4) should widen coverage if anything does.
- **Value propagation is slow.** Mean Q in rung 1 rose from 0.25 at 11k games to 0.79 at 26k. With γ = 1, a value moves back one decision per target-net update, and a 4×4 run gets only ~11–23 updates (`target_update_every: 1000`). Changed to 250 on Nov 24, below and §4.5.
- A's loss is ~5× N's because A's targets include the opponent's ε-greedy reply, which is noise to the learner. N's target depends only on the position.

*Rerun with `eps_end: 0.1`, `target_update_every: 250` (Nov 24, `runs/ladder-tuned/`, seed 0).* Means over the last 10 eval points, baseline → rerun:

| | Accuracy, all 7,912 | Q corr, all | Score vs positional | Score vs random |
|---|---|---|---|---|
| Rung 1: A vs random | 74.3% → 73.9% | 0.39 → 0.39 | 58.5% → 57.3% | 72.6% → 72.6% |
| Rung 2: A-naive | 73.8% → 75.5% | 0.41 → 0.43 | 55.6% → 60.5% | 71.6% → 73.8% |
| Rung 3: N-naive | 75.8% → **78.5%** | 0.43 → 0.47 | 56.3% → 59.1% | 70.8% → 72.8% |

- **Faster early learning comes from `target_update_every`.** ε is nearly identical in both runs for the first 6k games (it decays to `eps_end` over the first 20%), so early gaps are the target update's doing. N's accuracy at 2k games went from 57% to 71%, and at 5k from 72% to 79%. Rung 1's mean Q at 5k went from 0.03 to 0.39, the faster value propagation predicted above. No instability: losses are similar, and nothing collapsed.
- **Late gains in the self-play rungs are modest (+2–3 points accuracy, +3–5 vs positional) and can't be split between the two knobs.** This is one seed, and within-run jitter is ~±1 point.
- **ε = 0.1 widens coverage only a little.** In 8,000 self-play games from the final nets, about one buffer's worth, the eval-set positions reached go from ~150 (1.9%) at ε = 0.05 to ~220–270 (2.8–3.4%) at ε = 0.1, before D4 augmentation. The narrowness is structural: one start position and a nearly deterministic policy. The knobs don't fix it; it stays a property of naive self-play for the pool comparison to address.
- **Decision (Nov 24): adopt both, on every board** (§4.5). `runs/ladder/` (old settings) vs `runs/ladder-tuned/` (new) is a one-seed preview of E7 on 4×4. Keep both directories: `runs/` is not in git. `make ladder` now writes to `runs/ladder-tuned/` and refuses to replace an existing run, so the baseline can't be overwritten by accident.

*Rung 4: pool sampling (Nov 24, current settings, `runs/ladder-tuned/`, seed 0), against rungs 2–3 rerun with the same settings.* Training-style coverage counts the eval-set positions reached in 8,000 games (about one buffer) from the final net at ε = 0.1. For naive that's the net against itself; for pool, half against itself and half against a uniformly drawn checkpoint of its own run.

| Last 10 eval points | A-naive | A-pool | N-naive | N-pool |
|---|---|---|---|---|
| Accuracy, all 7,912 | 75.5% | 75.4% | 78.5% | 77.9% |
| Q corr with minimax, all | 0.43 | 0.44 | 0.47 | 0.47 |
| Score vs positional | 60.5% | 59.3% | 59.1% | 60.7% |
| Minimax accuracy on own self-play positions (final net) | 97.7% | 96.6% | 96.4% | 97.6% |
| Training-style coverage of the eval set | 3.4% | **5.1%** | 2.8% | **4.0%** |
| `opp_past_frac` (target 0.5) | — | 0.501 | — | 0.491 |
| Wall time (two runs sharing the GPU) | 4¾ min | 5½ min | 5½ min | 5¾ min |

- **Rung 4 passes.** The pool behaves as designed: ~50% of groups play past snapshots, and the pool reached 30 snapshots by game 30k. Both pool runs learn their own positions as well as the naive runs do.
- **On one seed, pool ≈ naive on every 4×4 metric**, although pool training reaches ~1.4–1.5× more eval positions. The extra positions mostly come from games against weak early snapshots, and on 4×4 they don't buy accuracy. One seed can't separate the conditions (within-run jitter is ~±1 point). E1's 5 seeds will, and "more coverage without more accuracy" is worth watching for.
- `opp_past_frac` ≈ 0.5 confirms D2's dilution arithmetic. Half of A-pool's games are against past snapshots, and in each of those, only the learner's side has a snapshot as its environment, so ~25% of transitions do.
- The self-play score sits at 0.50 in all four conditions: F3's flat line, already visible on 4×4.


### 4.7 Evaluation protocol (`evaluate.py`, `regret.py`)

**Standard metrics**
- **Opening banks:** 4×4: all 120 distinct positions after 4 plies. 6×6 and 8×8: 100 distinct positions after 6 random plies (seed 0, saved in `data/`). Each is played with both colors; agents play greedy (ε = 0).
  - *Revised Nov 23 (was 2 plies on 4×4, 4 plies on 6×6/8×8).* Shallower banks have too few genuinely different openings. Mirror images and rotations of an opening are the same opening to a symmetric player, and a net trained with D4 augmentation is close to one. At 2 plies, 4×4 has 12 positions but only 3 up to symmetry, and its 24-game matches gave ±20-point intervals. At 4 plies, 6×6 and 8×8 have 236 positions (60 up to symmetry), and a 100-position sample held only 49 different ones. The revised banks hold 31 (4×4) and 97–99 (6×6/8×8) different openings. The cost is more random moves before the agents take over, which is harmless: every opening is played from both sides, so a lopsided opening favors each player once.
- **Score** = (wins + 0.5·draws) / games, with a 95% Wilson interval.
- **Per-run summary:** final score (mean of last 3 eval points) · peak · max drawdown after the peak · forgetting events (drops > 10 points below the running max) · Spearman ρ between self-play score and score vs positional.
- **Cross-play matrix:** 15 evenly spaced checkpoints per run, round robin over the opening bank.

**Position-level checkpoint regret (adapted from MADRID, Samvelyan et al. 2024)**
MADRID uses 48 checkpoints saved during TiZero's training, plus 3 heuristic bots, as reference policies. It finds starting states ("levels") where swapping a reference policy in for the target does better against the target than the target does against itself. Their regret, `V(π_R, π_T) − V(π_T, π_T)`, is **cross-play minus self-play**, which is the same idea as H3. The Othello adaptation:

- **Level θ** = a board position with a side to move. `V^θ(X, T)` = final outcome (+1/0/−1) for the side to move when X plays that side and the target T plays the other. With greedy agents each V is a single deterministic game, so no repeats are needed (MADRID needed 4 because football is stochastic).
- **Regret** `reg_R(θ) = V^θ(R, T) − V^θ(T, T)` ∈ {−2,…,2}. It is positive when a reference R handles θ better than T does.
- **Reference sets** (D8):
  - **Self-history:** every checkpoint from T's own run. Positive regret means *T forgot how to handle θ*. This is the direct forgetting measure, but it is biased toward pool agents.
  - **Common held-out set:** the 3 heuristic players + every 5th checkpoint from all runs *except T's own*. This is the same set for every T and is the fair comparison across conditions.
- **Positions:**
  - 4×4: the full accuracy eval set (enumerable).
  - 6×6: 2,000 positions sampled from games among reference players and checkpoints, spread across game phases (8–28 empties).
  - No MAP-Elites search is needed at this scale. Keep MADRID's *grid* only for visualization: bin positions by (empties × disc differential).
- **4×4 bonus (something MADRID can't do):** you have the true π\*, so compute **true regret** `V^θ(π*, T) − V^θ(T, T)` and check how much of it the checkpoint references recover. This answers the question "are checkpoints a good proxy for the optimal policy?"
- **Summary numbers per final agent:** fraction of positions with any positive-regret reference · mean max-regret · regret by reference index.

### 4.8 Experiments

| ID | What | Runs | When |
|---|---|---|---|
| E0 | 4×4 supervised sanity check: fit the net to exact solver values (separates RL bugs from capacity issues) | 1 | Nov 24, only if stuck |
| E1 | 4×4 2×2 | 4 × 5 seeds = 20 (minutes each) | Launch Nov 25 |
| P | 6×6 pilots, 400k games (their curves set E2's budget) | 4 × 1 seed | Launch Nov 25 on the laptop GPU (~6–7 h sharing it with P8) |
| E2 | 6×6 2×2 | 4 × 3 seeds = 12 | Launch Dec 1 on the laptop GPU, 4 at a time: ~8½ h at 200k games, ~16½ h at 400k |
| E3 | Cross-play matrices | — | Dec 1 (4×4) / Dec 4 (6×6) |
| E5 | Checkpoint regret (self-history + held-out; + true regret on 4×4) | — | Dec 2 (4×4) / Dec 4 (6×6) |
| E4 | *(Could)* A-naive replay-buffer ablation (¼ size), or pool mix 80/20 | 3 seeds | Only with spare compute |
| P8 | *(Could)* 8×8 pilot: A-naive, 1M games (`make pilot-8x8`) | 1 | Launch Nov 25 alongside P (gate 1 passed Nov 24); ~1 day |
| E6 | *(Could)* 8×8 A pair: A-naive vs A-pool on the frozen code (`make sweep-8x8`) | 2 × 1 seed | Launch Dec 1 (Dec 2 at the latest), only if gate 2 passed |
| E7 | *(Could)* Settings comparison: rerun with the pre-Nov-24 settings (`eps_end` 0.05, `target_update_every` 1,000) on the same code, compared with E1 / P. 4×4: `make sweep-4x4-old` (all 4 conditions × 5 seeds, ~5 min per run on `mps`). 6×6: `make pilot-6x6-old` (1 seed per condition, GPU hours) | 20 (4×4) + 4 (6×6) | 4×4 any time after E1 launches; 6×6 only with spare compute, never ahead of E2 |

**Compute:** On Tue Nov 24, measure games/s and gradient steps/s on your laptop and server, then fix the 6×6 game budget so one run takes ≤ 6 h. ~~Run as many seeds in parallel on the server as it has cores for (`tmux`), and `rsync` `runs/` back.~~ *Revised Nov 24: the server is too small (below), so every 6×6/8×8 run goes on the laptop GPU, four at a time (`JOBS=4`; `DEVICE` defaults to `mps` on macOS). Keep the laptop plugged in and awake: run sweeps under `caffeinate -i`, inside `tmux` so closing the terminal doesn't kill them, with the lid open.* **Fallback:** if the 12 E2 runs can't finish by Thu Dec 3, run 3 seeds for the A pair and 2 seeds for the N pair (10 runs).

Measured Nov 19 on the laptop (M1 Pro): one Double-DQN gradient step at batch 256, with the §4.5 network. Steps per game and per-run times were recomputed Nov 23 for both sides' transitions (D2), which doubles both.

| | CPU (8 threads) | GPU (`mps`) | Steps per game | Gradient steps alone, per run |
|---|---|---|---|---|
| 6×6, L = 4 | 104 ms | 7.6 ms | ≈ 2 | 200k games: ~11.6 h CPU / ~50 min GPU |
| 8×8, L = 5 | 198 ms | 11.2 ms | ≈ 3.8 | 500k games: ~104 h CPU / ~6 h GPU |
| 4×4, L = 3 *(Nov 24)* | 50 ms (1 thread; 2–8 threads are slower) | 3.8 ms | ≈ 0.75 (measured) | 30k games: ~19 min CPU / ~1.5 min GPU |

The 4×4 row is from the debugging ladder: a whole 30k-game run took 5½ min on `mps` with two runs sharing the GPU, and 2¼ min for rung 1, which stores only the learner's moves (≈ 0.38 steps per game).

**Whole-loop measurement (Nov 24, laptop M1 Pro, `scripts/throughput.py`).** The script times the real training code: self-play rounds (naive, and pool-style with 8 distinct opponent networks as the worst case), full gradient steps (replay sampling with D4 augmentation, three forward passes, backward, clipping, Adam), and one eval point. It then projects a full run. On the GPU the real step is slower than the bare network step above (11.5 vs 7.6 ms on 6×6) because of the CPU-side sampling and three forward passes. Run hours are for the current budgets (6×6 200k games, 8×8 500k placeholder) at `transitions_per_grad_step` 16 / 32, pool-style:

| | Self-play (naive / pool) | Gradient step | Eval point | Run hours, tpgs 16 / 32 | Games in the limit, tpgs 16 |
|---|---|---|---|---|---|
| 6×6 `mps` | 2.5–2.8 / 3.6 ms per game | 11.5 ms | 1.0 s | **1.5 / 0.8 h** | ~820k in 6 h |
| 6×6 CPU, 1 thread | 3.7 / 4.7 ms | 89.7 ms | 1.3 s | 10.2 / 5.3 h | ~118k |
| 6×6 CPU, 4 threads (fastest) | 2.6 / 3.6 ms | 75.6 ms | 1.0 s | 8.6 / 4.4 h | ~140k |
| 6×6 CPU, 8 threads | 2.6 / 4.4 ms | 103.7 ms | 1.2 s | 11.8 / 6.0 h | ~102k |
| 8×8 `mps` | 5.9 / 7.7 ms | 15.5 ms | 2.6 s | **9.2 / 5.1 h** | ~1.31M in 24 h |
| 8×8 CPU, 1 thread | 21.5 / 23.7 ms | 324 ms | 8.1 s | 172 / 88 h | ~69k |
| 8×8 CPU, 4 threads | 11.1 / 16.6 ms | 172 ms | 4.8 s | 92 / 47 h | ~130k |

Several 6×6 runs sharing the laptop GPU (2 torch threads each): 2 at once take 15 ms per step and 1.8–2.0 h per run; 4 at once take 20–23 ms and 2.4–2.8 h per run. Four together finish ~1.5 runs per hour, about twice one alone. At that rate the laptop GPU alone could do all 12 E2 runs in three batches of four, ~8½ h at 200k games. Gradient steps dominate everywhere. Self-play is ≤ 15% of a GPU run, and evaluation is negligible: under 3 min per run on every board.

- **6×6:** 200k games fits the 6 h limit easily on the laptop GPU at `transitions_per_grad_step: 16`, so no fallback is needed there. On a CPU, 16 doesn't fit (8.6 h even at the best thread count), and 32 does (4.4–5.3 h).
- **8×8:** 500k games takes ~9 h on the laptop GPU alone, so a 24 h budget fits ~1.3M games. CPU is out (≥ 92 h).
- **Budgets (decided Nov 24).** The holiday leaves spare GPU time, so the pilots get generous budgets, and the experiments' budgets are set on Nov 30 from the pilots' curves instead of guessed now. 6×6 pilots: 400k games, ~3 h alone or ~5.5 h with four sharing the GPU. E2 is then 200k–400k: ~8½–16½ h for all 12 runs, four at a time, either way done before the Dec 3 fallback deadline. P8: 1M games, ~18.4 h alone on the GPU. E6 is ≤ 1M, set at gate 2. Replay and eval spacing follow the budget rule in §4.5.
- **Server (measured Nov 24):** 2 ARM cores (Neoverse-N1, aarch64), ~11 GB RAM, no GPU. With 1 torch thread: 6×6 self-play 7.7 / 9.1 ms per game, gradient step 445 ms, eval 3.0 s, so a 200k-game run takes **50 h** (25 h at tpgs 32). 8×8: 26 / 29 ms per game, 783 ms per step, 11.3 s per eval point, so 500k games take **412 h**. That's ~5× slower per core than the laptop CPU and ~39× slower than the laptop GPU. **Decision: E1, P, E2, P8 and E6 all run on the laptop GPU.** The server is only useful for low-priority 4×4 work such as E7's 4×4 half, one run at a time so the user's sites keep a core.
- **`transitions_per_grad_step` stays at 16** for every condition. The 32 fallback was only for CPU-only training, which no experiment uses now.
- **8×8 shares the laptop GPU with E2 by taking turns.** E2 has priority: E6 starts only after E2 finishes. The P8 pilot runs over the holiday next to the 6×6 pilots, which aren't E2.

Self-play, inference, and evaluation come on top. **On CPU only, gradient steps alone take ~11.6 h per 6×6 run, about twice the 6 h limit, so 6×6 needs a GPU.** If one isn't available for all 12 runs, set `transitions_per_grad_step: 32` for every condition. That restores ≈ 1 step per game, with each transition sampled ≈ 8× instead of 16×. Decide this at the Nov 24 throughput measurement, before E1. Runs that share cores with `--torch-threads 1` are slower still. Train on a GPU wherever one is available: the laptop's (`DEVICE=mps`) is 14–18× faster than its CPU on this network.

**8×8 gates (D9).** 8×8 runs only if both gates pass. If either fails, 8×8 returns to Won't and nothing else in the plan changes.
- **Gate 1 (Tue Nov 24, throughput):** measure 8×8 alongside 6×6, then set `games` in `configs/8x8.yaml` (and rescale `replay_capacity` to ≈ 10% of all transitions, both sides, D2). It passes if one 8×8 run of that budget fits in ≤ 24 h on hardware E2 doesn't need. A CPU-only setup fails.
- **Gate 2 (Mon Nov 30, pilot review):** it passes if (a) the 6×6 pilots learned (≥ 50% vs positional, the §7 bar) and (b) P8 reached ≥ 50% vs positional on 8×8.
- **Cutoffs:** results by Sun Dec 6 earn a slide; results by Mon Dec 14 go in the report; later results are dropped. 8×8 never takes compute from E2, and it never justifies a code change after M2.
- **Scope:** score vs random/greedy/positional, self-play score (D5), and F9. No cross-play, regret, or endgame metrics on 8×8.

### 4.9 Figures

| ID | Figure | Used in |
|---|---|---|
| F1 | 4×4 optimal-move accuracy vs games: two panels (A, N), naive vs pool, seed bands + chance line | slides, report |
| F2 | 6×6 score vs positional (and random/greedy in the appendix), same two-panel layout | slides, report |
| **F3** | **Self-play score and external score on the same axes, flat line vs moving line** (headline figure) | slides, report, demo |
| F4 | Cross-play heatmaps, naive vs pool | report (slides if striking) |
| F5 | 6×6 endgame accuracy vs games | report |
| F6 | Q-value heatmaps on 2–3 teaching positions across checkpoints | slides, demo |
| **F7** | **Self-history regret by reference checkpoint index for final naive vs pool agents** (MADRID Fig. 4b analog), plus a regret grid over (empties × disc differential) (Fig. 6 analog) | slides, report |
| F8 | 4×4: checkpoint-estimated vs true regret | report (slides if time) |
| T1 | Summary table: §4.7 numbers, mean ± std across seeds, with the (pool − naive) gap per target type (H4) | slides, report |
| F9 | *(Could, E6)* 8×8: score vs positional and self-play score, A-naive vs A-pool, one seed each (F2/F3 layout, same code) | report; slides if done by Dec 6 |
| F10 | *(Could, E7)* 4×4 accuracy vs games (F1 layout), old vs new settings, per condition with seed bands. Does faster value propagation help every condition equally, or does it change the naive-vs-pool gap? | report (discussion or appendix) |

---

## 5. Web demo spec

### 5.1 What visitors can do

1. **Play the agent** on a 6×6 board, or on 8×8 if the 8×8 agent qualifies (§5.3). Pick a color and pick an opponent snapshot: *Untrained → Early → Midway → Final*.
   - **Passes are never automatic for the visitor.** With no legal square, the board says so ("No legal moves: you must pass") and the visitor presses a Pass button themselves, so it's clear the pass was forced. When the agent has to pass, the board says so too ("The agent has no legal moves and passes"). A turn with a single legal square is an ordinary turn: the visitor taps it. (The Python game runner auto-plays forced passes, but it only runs bot-vs-bot games for training and evaluation; this rule is for the play view.)
2. **See what it's thinking.** A toggle overlays the agent's value estimate on each legal square as win probability `(Q+1)/2`.
3. **Watch it learn.** A slider over training checkpoints updates (a) Q heatmaps on teaching positions and (b) the learning curve, with the flat self-play line (F3) and the current checkpoint marked.
4. *(Could)* **Watch two snapshots play** each other, or play an A agent vs an N agent.
5. **About:** a three-sentence summary, the MDP, and links to the code and report.

### 5.2 Architecture

```
train.py ──► runs/.../ckpt/*.pt
                │
export.py ──────┼──► web/public/models/<snapshot>.onnx       (play)
                ├──► web/public/data/manifest.json           (snapshot list + stats)
                ├──► web/public/data/curves.json             (learning curves)
                ├──► web/public/data/heatmaps.json           (Q for teaching positions × all checkpoints)
                └──► web/tests/fixtures/*.json               (parity tests)

web/ (Vite + TypeScript, no framework)
  src/engine.ts   port of engine.py
  src/agent.ts    onnxruntime-web session per snapshot, masked argmax
  src/ui/*        board (SVG or canvas), controls, chart
```

- **Export:** `torch.onnx.export(model, dummy[1,3,n,n], ..., input_names=["planes"], output_names=["q"], dynamic_axes={"planes": {0: "b"}, "q": {0: "b"}})`. Masking happens in TypeScript.
- **Runtime:** `onnxruntime-web` with `ort.env.wasm.wasmPaths` pointed at a pinned jsDelivr version. Load each snapshot lazily, only when selected.
- Heatmaps and curves are precomputed JSON, so the slider needs no inference.
- The agent plays greedy, breaking exact ties randomly, with a ~400 ms delay.

`manifest.json` sketch (numbers are placeholders):
```json
{
  "board_size": 6,
  "condition": "A-pool",
  "snapshots": [
    {"id": "untrained", "label": "Untrained", "games": 0,      "model": "models/untrained.onnx", "score_vs_positional": 0.08},
    {"id": "early",     "label": "Early",     "games": 15000,  "model": "models/early.onnx",     "score_vs_positional": 0.31},
    {"id": "mid",       "label": "Midway",    "games": 60000,  "model": "models/mid.onnx",       "score_vs_positional": 0.55},
    {"id": "final",     "label": "Final",     "games": 200000, "model": "models/final.onnx",     "score_vs_positional": 0.80}
  ],
  "teaching_positions": ["corner_vs_flips", "x_square_trap", "endgame_parity"]
}
```

### 5.3 Snapshot and teaching-position selection

- **Board size (decide Dec 4):** use 8×8 if E6 has finished and its better final agent scores ≥ 70% vs positional. Otherwise use 6×6. The web engine is size-generic and `manifest.json` carries `board_size`. If the play view takes its size from there (Dec 3), this is an export choice, not a code change.
- Use one seed from the condition with the highest final score on the chosen board, and state in the About text which condition and that it's one seed.
- **Untrained** = game 0. **Early** = first checkpoint with ≥ 60% vs random. **Midway** = first checkpoint with ≥ 50% vs positional. **Final** = last checkpoint.
- Teaching positions should have ≤ 12 empties so the endgame solver can confirm the correct move. Search cost depends mostly on the number of empties, so this should also work on 8×8. Check it on Dec 4 if 8×8 is chosen.

### 5.4 Correctness

- **Engine parity:** Python writes 500 random games (legal masks + resulting boards at every ply). The TS engine must match exactly.
- **Model parity:** Python writes 100 positions with their Q-vectors. ORT-web output must match within 1e-4.
- Run both with `vitest` in GitHub Actions before every deploy.

### 5.5 Hosting

- Deploy `web/dist` from the `othello-rl` repo with the official GitHub Pages Action, on the subdomain **`othello.johnnygoodlad.dev`**. The portfolio (`johnnygoodlad.dev` repo) is a project site with its own custom domain rather than the `<username>.github.io` user site, so other repos are *not* served under `johnnygoodlad.dev/<repo>/` automatically. A subdomain keeps the demo independent of the portfolio repo. Vite `base` is `/`.
- Setup order (custom domain in GitHub *before* DNS, which prevents subdomain takeover): repo Settings → Pages → Custom domain `othello.johnnygoodlad.dev` → Namecheap Advanced DNS: `CNAME` host `othello` → `thejrock.github.io.` → wait for the certificate → tick Enforce HTTPS. No `CNAME` file is needed with Actions deploys.
- The server's wildcard DNS (`*.johnnygoodlad.dev` points at the server) doesn't conflict: a specific `othello` CNAME record takes precedence over a wildcard. Never add an `othello.johnnygoodlad.dev` block to the server's Caddyfile.
- `.dev` is HSTS-preloaded, so browsers will only load the subdomain over HTTPS. Until GitHub issues its certificate, it shows a certificate error, so don't link it from the portfolio until HTTPS works.
- *Fallback:* copy `web/dist` into a folder of your portfolio repo.
- Link it from the portfolio home page. Test on desktop Chrome/Safari and on a phone (tap-to-move, fits at 360 px).
- Record a 60–90 s screen capture as a presentation backup.

---

## 6. Timeline

Hours are estimates: ≈ 65 h through Dec 10 and ≈ 15–20 h for Dec 11–17. The gated 8×8 add-on adds ≈ 4–6 h if both gates pass: measuring, launching, F9, and the demo export. Every day M1 lands early is a day of 8×8 compute; finishing off-critical-path work early does not buy any.

### Week 1 — Foundations and 4×4 (Nov 18–25)

**Wed Nov 18** (~3 h)
- [x] Ask the instructor about presentation length/format and report length/format (§11).
- [x] Create repo and environment (`torch numpy pytest pyyaml matplotlib scipy`), set up the §4.1 layout.
- [x] Engine: bitboards, move generation, flips. *(done Nov 19)*

**Thu Nov 19** (~3 h)
- [x] Engine: passes, terminal, scoring, `to_planes`, D4 transforms.
- [x] Engine tests (§4.2), all passing. 8×8 perft matches Egaroucid's published table through depth 9.
- [x] 8×8 groundwork (D9): `configs/8x8.yaml`, 8×8 positional weights, 8×8 opening-bank defaults, `make pilot-8x8` / `sweep-8x8`, `DEVICE=` for the sweep targets.

**Fri Nov 20** (~3–4 h)
- [x] Random / greedy / positional players + network-player wrapper. *(done Nov 23)*
- [x] Batched game runner (any two players, many games at once). *(done Nov 23)*
- [x] Eval harness: opening banks, both colors, score + Wilson CI. Sanity-check opponent strengths. *(done Nov 23; banks revised, §4.7; results in §4.3)*

**Sat Nov 21** (~3 h)
- [x] 4×4 full solver, π\*, eval set, and chance accuracy; check the start-position value. *(done Nov 23; White wins 3–11, matches published, §4.4)*

**Mon Nov 23** (~4–5 h)
- [x] DQN core: network, replay buffer, Double-DQN loss with a `target: negamax | env` switch, masking, symmetry at sample time. Use `cfg.device` throughout so runs can use `cuda` or `mps`. *(done Nov 23. Head ReLU removed, §4.5. Smoke test: offline negamax on 5,000 random 4×4 games reached 79% optimal-move accuracy (chance 49%) after 8,000 steps, ~1 min on `mps`.)*
- [x] A-mode environment step (opponent replies + auto-passes). *(done Nov 23: `selfplay.cut_transitions`, both modes)*
- [x] Target unit tests for both modes (§4.5). *(done Nov 23; exact-value checks, §4.5)*

**Tue Nov 24** (~4–5 h)
- [x] Debugging ladder rungs 1–3 on 4×4 (§4.6), with JSONL logging. *(done Nov 24; results in §4.6)*
- [x] N must show Q correlating with solver values. Both must show accuracy clearly above chance and ≥ 90% vs random. Run E0 if stuck. *(Nov 24: passed on the positions the nets actually play, 98–99% accuracy and Q corr 0.82–0.87. Whole-set accuracy is 73–75% vs 49.4% chance. "≥ 90% vs random" was revised: π\* itself scores 77.8% on the 4×4 bank and 88.1% from the start, §4.6. E0 not needed: the nets fit what they visit, and the shortfall is coverage.)*
- [x] Measure throughput on laptop (CPU and `mps`) and server, for 6×6 and 8×8; fix the 6×6 budget and whether 12 runs fit (§4.8 fallback). *(Done Nov 24, §4.8. The laptop GPU runs a 6×6 run in 1.5 h, and 4 runs share it at ~2× the throughput of one. The server, 2 ARM cores with no GPU, would take 50 h per 6×6 run, so every experiment runs on the laptop GPU at `transitions_per_grad_step: 16`, with no fallback. 6×6 pilot budget: 400k games; E2's budget is set Nov 30 from the pilot curves.)*
- [x] **8×8 gate 1** (§4.8): set the 8×8 game budget, or drop 8×8. *(Passed Nov 24. `games: 1000000` (~18.4 h alone on the laptop GPU), `replay_capacity: 6000000`, `eval_every: 20000`. The laptop GPU is hardware E2 doesn't need outside E2's own window: P8 shares it with the 6×6 pilots, and E6 starts after E2 finishes.)*

**Wed Nov 25** (~4 h) — **M1**
- [x] Opponent pool, checkpointing, YAML configs, run directories with commit hash. *(Done Nov 24, a day early: checkpointing, configs and run directories with `train.py`, then `pool.PoolSampler`. Ladder rung 4 passed, §4.6. E1 and the pilots are ready to launch.)*
- [ ] Launch **E1** (4×4 2×2, 20 runs): `caffeinate -i make sweep-4x4` in `tmux`, four at a time on the laptop GPU, ~30 min.
- [ ] Launch **6×6 pilots** (4 runs, 400k games) ~~on the server before leaving for the holiday~~ on the laptop GPU after E1: `caffeinate -i make pilot-6x6` (*Nov 24: the server is too slow, §4.8. The user is home for the holiday, and the laptop can run day and night*). ~6–7 h sharing the GPU with P8.
- [ ] If gate 1 passed (it did, Nov 24): launch the **8×8 pilot** P8 (1M games) ~~on hardware the 6×6 pilots don't use~~ on the laptop GPU alongside the 6×6 pilots: `caffeinate -i make pilot-8x8` in a second `tmux` window (*Nov 24: there is no second GPU, and the pilots aren't E2, so sharing is fine*). ~1 day.
- [ ] *(optional)* On the server: E7's 4×4 half. Do it after `git pull` of the code E1 ran, so both use the same code. Run `.venv/bin/python scripts/build_data.py solve4x4`, then `nice -n 10 make sweep-4x4-old JOBS=1` in `tmux`. That's one run at a time so the sites keep a core, roughly 1½ days (estimate; check with `scripts/throughput.py --config configs/4x4.yaml` first).

**Thu Nov 26 – Sat Nov 28** — Thanksgiving (optional ~2 h Sat)
- [ ] *(optional)* Start the 6×6 endgame solver or the TS engine port. Both are off the training critical path.
- [ ] Glance at pilot logs (6×6 and P8); don't change training code.

### Week 2 — 6×6 experiments and demo (Nov 30 – Dec 5)

**Mon Nov 30** (~4 h)
- [ ] Review E1 and the pilots. If 6×6 isn't learning, change at most 2 shared knobs (lr, budget, ε schedule), the same for all four conditions.
- [ ] Set E2's budget from the pilot curves: 200k if they've plateaued by then, up to 400k if still rising. Rescale `replay_capacity` and `eval_every` with it (§4.5 budget rule).
- [ ] **8×8 gate 2** (§4.8): go/no-go on E6. If go, set E6's budget (≤ 1M) from P8's curve and rescale likewise.
- [ ] 6×6 endgame solver + 500-position test set, cached.

**Tue Dec 1** (~4 h) — **M2: experiment code freeze**
- [ ] Launch **E2** (6×6 2×2, 12 runs, endgame accuracy in eval): `caffeinate -i make sweep-6x6` on the laptop GPU, four at a time in three batches, ~8½ h at 200k or ~16½ h at 400k.
- [ ] If gate 2 passed: launch **E6** (8×8 A pair) ~~on hardware E2 doesn't use~~ on the laptop GPU once E2 has finished (`make sweep-8x8`), on Dec 2 at the latest.
- [ ] Analysis scripts: seed-band curves, §4.7 summary numbers, Spearman ρ, H4 gap.
- [ ] Cross-play script; run E3 on 4×4.

**Wed Dec 2** (~4–5 h)
- [ ] `regret.py`: self-history + held-out references; true regret with π\* on 4×4. Run E5 on 4×4.
- [ ] Finalize the 4×4 versions of F1, F3, F7, F8, and T1.

**Thu Dec 3** (~4–5 h)
- [ ] Web: scaffold Vite + TS, port the engine, engine parity test passing.
- [ ] `export.py`: ONNX + Q fixtures; model parity passing (use pilot checkpoints for now).
- [ ] Play view: board, color choice, snapshot picker, pass handling (the visitor presses Pass themselves; agent passes are announced, §5.1), game over, Q overlay. Take the board size from `manifest.json` so 8×8 needs no UI changes.
- [ ] **Cut rule:** if play isn't working by tonight, "Watch it learn" moves to Dec 11–16 and the talk uses a static F3.

**Fri Dec 4** (~4–5 h)
- [ ] E2 done: 6×6 F2, F4, F5, T1; run E5 on 6×6 → F7.
- [ ] If E6 finished: F9 (the F2/F3 code pointed at `e6-8x8`).
- [ ] Pick the demo board size (6×6 or 8×8), snapshots, and teaching positions (§5.3); export final ONNX + `manifest/curves/heatmaps.json`.

**Sat Dec 5** (~3–4 h) — **M3**
- [ ] Deploy the play view to GitHub Pages under your domain; link it from the portfolio.
- [ ] "Watch it learn" view if time allows (otherwise Dec 11–16).
- [ ] Test desktop + phone; record the backup video.

### Week 3 — Presentation (Dec 7–10)

**Mon Dec 7** (~4 h)
- [ ] Draft the deck from §8 with all figures in. The 8×8 slide goes in only if F9 exists today. Otherwise 8×8 is report-only.

**Tue Dec 8** (~3 h)
- [ ] Finish the deck; full timed rehearsal; cut to time.

**Wed Dec 9** (~2 h) — buffer day
- [ ] Second rehearsal; QR code to the demo; backup video embedded; finish anything that slipped.

**Thu Dec 10**
- [ ] Present. Note the questions you get for the report's discussion section.

### Week 4 — Report + demo polish (Dec 11–17)

- [ ] **Fri Dec 11:** Outline; Method and Setup sections (adapt §3–4 of this spec).
- [ ] **Sat Dec 12:** Results with figures and captions. Finish "Watch it learn" if it slipped.
- [ ] **Mon Dec 14:** Intro, related work (MADRID, TiZero, AlphaGo/AlphaZero opponent handling), discussion, limitations.
- [ ] **Tue Dec 15:** Full draft; appendix (hyperparameters, engine/solver validation, demo link, F9 if E6 ran).
- [ ] **Wed Dec 16:** Revise using presentation Q&A; proofread; repo README; link the report from the demo's About page.
- [ ] **Thu Dec 17:** Submit.

No new experiments in report week. Extra seeds may run in the background if they finish by Dec 14. The same cutoff applies to E6.

---

## 7. Risks and cut rules

| Risk | Early signal | Response |
|---|---|---|
| Sign/pass bug in N targets | 4×4 Q doesn't correlate with solver values | Target unit tests; debugging ladder; E0. If N still isn't right by Nov 25, launch the A runs anyway and finish N on Nov 30 (it's the contrast arm). *Ruled out Nov 24: rung 3 plays the minimax move on 98% of its own positions, Q corr 0.82 (§4.6)* |
| DQN instability | Loss spikes, Q saturates at ±1 | Lower lr, more frequent target updates, confirm grad clipping; never per condition |
| 6×6 too slow / 12 runs don't fit | Measured budget > 6 h/run or too few server cores (on CPU, gradient steps alone are ~11.6 h with both sides stored; §4.8) | Train on a GPU (`DEVICE=cuda`, or `mps` on the laptop), shrink the net to 32 channels, cut the game budget, use the §4.8 seed fallback |
| 8×8 too slow or not learning | Gate 1: > 24 h/run on spare hardware. Gate 2: P8 < 50% vs positional | Drop 8×8 to future work. Never delay E2, change code after M2, or borrow from Dec 7–9 for it |
| 6×6 not learning by Dec 1 | Pilot score vs positional < 50% | 4×4 becomes the main result; present 6×6 as preliminary |
| Pool ≈ naive everywhere | Overlapping seed bands in both rows | That's a result: lean on F3, F7, and the D1 framing (the replay buffer is already an implicit pool). In the A row, also note the D2 dilution: only ~25% of A-pool transitions face a past snapshot |
| Demo eats presentation time | Not playable by Dec 3 | Play view only for the talk; the rest moves to Dec 11–16 |
| Server unavailable | — | Run on the laptop GPU (`mps`) overnight with a reduced 6×6 budget and the seed fallback. 8×8 is dropped, because E2 needs the laptop. *This happened (Nov 24): the server is 2 ARM cores with no GPU. The laptop GPU carries everything. A 6×6 run is 1.5 h alone or ~2.7 h with four sharing, so no budget cut was needed. 8×8 takes turns with E2 instead of being dropped (§4.8)* |

**Slip rule:** if any milestone slips by more than one day, apply the matching cut immediately rather than borrowing from the Dec 7–9 presentation days.

---

## 8. Presentation outline (~15 minutes)

1. **Hook:** F3. "This line says the agent is stuck. This one says it's improving. Which do you believe?" Put the demo QR code up.
2. **Self-play as non-stationary RL:** the opponent changes with every update; replayed data comes from older players.
3. **Two views of self-play (D1):** opponent-as-environment (learn a best response; passes are just dynamics) vs negamax (learn the minimax value; passes need explicit sign flips). Use a small diagram. Prediction: the pool should matter more for the first view.
4. **Setup:** the 2×2, boards, reference opponents, solvers as ground truth.
5. **4×4 results:** F1 (accuracy against exact play) + T1.
6. **6×6 results:** F2, H4 gap. *(If F9 exists by Dec 6)* one 8×8 slide: does the A-row result hold on the full board? One seed per arm.
7. **What self-play hides:** F3 + F7. Earlier checkpoints beat the final agent from specific positions, and self-play score never shows it (MADRID's cross-play-minus-self-play idea).
8. **Takeaways + demo:** "Watch it learn" (F6 or live demo).

## 9. Report outline

4–6 pages, single-column conference template (NeurIPS or ICLR style). Figures count toward the limit; references and the appendix don't. The limit is soft, but no padding: every paragraph has a purpose and figures stay modest (full width, ≤ 1.7 in tall). Working budget: ~2,600 words + ~1.3 pages of figures/tables ≈ 5.5 pages.

Abstract · 1 Introduction and question · 2 Background and related work (self-play, DQN, opponent pools, MADRID checkpoint regret) · 3 Formulation (MDP, both views, targets, passes) · 4 Method (network, Double DQN, symmetries, pool) · 5 Experimental setup (2×2, boards, seeds, budgets, opponents, solvers, metrics, regret) · 6 Results · 7 Discussion (what self-play metrics hide; when pools matter) · 8 Limitations and future work (a full 8×8 study with the N row and more seeds, prioritized pool sampling, search, MAP-Elites over positions) · 9 Conclusion · Appendix (hyperparameters and the Nov 24 settings change with E7/F10 if run, validation, demo link, 8×8 A pair if E6 ran)

**Citation to settle before writing the validation appendix:** our 4×4 solve reproduces the published perfect-play result exactly, *White wins 3–11* (§4.4). So far that figure has only been seen in Wikipedia's *Computer Othello* article (via a mirror), which isn't a citable primary source. Check Takizawa (2023) below for a primary citation of the 4×4 (and 6×6) result. If it doesn't give one, follow that article's references. The 6×6 result (White wins 16–20) is credited to Joel Feinstein (1993, *British Othello Federation* newsletter).

## 10. References

- Samvelyan, Paglieri, Jiang, Parker-Holder, Rocktäschel. *Multi-Agent Diagnostics for Robustness via Illuminated Diversity* (MADRID). AAMAS 2024. arXiv:2401.13460. (Local PDF kept out of the repo.)
- Lin, Huang, Pearce, Chen, Tu. *TiZero: Mastering Multi-Agent Football with Curriculum Learning and Self-Play.* AAMAS 2023. (MADRID's ref. [29]; source of the checkpoints MADRID evaluates.)
- Takizawa. *Othello is Solved.* arXiv:2310.19387, 2023. https://arxiv.org/abs/2310.19387. *To check:* does it report the 4×4 (White wins 3–11) and 6×6 (White wins 16–20) results, and whom does it credit? (§9 note.)
- *Computer Othello* (Wikipedia; mirror: https://www.wikiwand.com/en/articles/Computer_Othello). Where the exact 4×4 score was found. A lead to primary sources only; don't cite it in the report.

## 11. Open items

- [x] Presentation: ~15 minutes. Report: 4–6 pages in a single-column conference template (IEEE, NeurIPS, ICLR, or equivalent). Figures count toward the pages, references and the appendix don't; the limit is soft but no filler or oversized figures (Nov 19).
- [x] Server specs: CPU cores, GPU? These decide whether all 12 E2 runs fit and whether 8×8 can run beside them (Nov 24). *Run on the server: `nproc`, `nvidia-smi`, and `scripts/throughput.py` for 6×6 and 8×8 (CPU with `--torch-threads 1`, plus `--device cuda` if it has a GPU), with `--json`.* *Setup (Nov 24): the server is compute-only for this project. Clone it to `/srv/othello-rl` with a read-only deploy key (step 3 of the server's own project guide). Use a Python venv and `tmux`, with no compose stack, Caddy block, or database. Bring results back with `make sync SERVER=… SERVER_DIR=/srv/othello-rl`. Other sites share the machine, so leave cores free when setting `JOBS`.* *Specs (Nov 24): 2 ARM cores (Neoverse-N1), ~11 GB RAM, no NVIDIA GPU, Ubuntu 24.04, tests pass (205 passed, 2 skipped). A 6×6 run takes 50 h there, so everything heavy goes on the laptop GPU (§4.8).* A GPU matters for 6×6 too: on CPU, gradient steps alone take ~11.6 h per 6×6 run with both sides stored, about twice the 6 h/run limit (§4.8). Without enough GPU, use the `transitions_per_grad_step: 32` fallback.
- [x] Demo URL: `othello.johnnygoodlad.dev` (portfolio is a project site with a custom domain; §5.5).
- [x] Opponent transitions: the instructor asked for the opponent's moves to be included in training. D2 was revised on Nov 23 to store both sides' moves in all four conditions. `pool_p_latest` stays at 0.5, and the resulting A-pool dilution (~25% past-snapshot environment) goes in the report (D2).
- [x] Shared knobs before E1 (decide by Wed Nov 25; the same for all four conditions). The ladder (§4.6) found two things. (a) Naive self-play at ε = 0.05 covers only ~200 of 7,912 positions, so 4×4 accuracy plateaus at ~75%. (b) Value propagation is slow, with only ~11–23 target updates per 4×4 run. Options: keep the §4.5 settings, so coverage is part of the result, or change `eps_end` (e.g. 0.1) and/or `target_update_every` (e.g. 250), checked with one quick ladder rerun first. *Rerun done Nov 24 (§4.6): `target_update_every: 250` clearly speeds early learning. Both knobs together add +2–3 points of late accuracy on the self-play rungs. ε = 0.1 widens coverage only from ~2% to ~3%. Decided Nov 24: both adopted on all three boards (§4.5). The old settings stay available as E7 (§4.8) for a comparison in the report.*
