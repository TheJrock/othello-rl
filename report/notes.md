# Report notes

A running list of things worth mentioning in the report or coming back to. No format; add freely.
The full details and numbers live in `docs/spec.md` (section refs in brackets).

## Things for the report

### Setup / formulation
- The network's input is just mover discs, opponent discs and legal squares. Nothing says who moved first or which colour is to move. One network plays both colours, and a position looks the same to it whichever colour is moving.
- Both sides' moves are stored as training data (D2). This was the instructor's call on Nov 23; before that it was learner-only. Side effect: in A-pool, past snapshots are the "environment" for only ~25% of transitions, not 50%, so the A-row pool-vs-naive contrast is weaker than the 50/50 mix suggests. That works against finding a pool effect. [§3 D2]
- **D1 in one test.** With a perfect opponent, every A target equals the exact minimax value of the move. Against a random opponent, some don't. A learns values against *its* opponents; N learns the game's value whatever the opponent does. A clean way to explain the A/N difference in the Formulation section. [§4.5 target tests]
- Training opponents explore (ε-greedy) like the learner; evaluation is greedy for every network. Three reasons [§4.6]:
  - the naive opponent *is* the learner
  - it keeps naive vs pool a comparison of *who*, not of how noisy
  - coverage is already the bottleneck
- The cost of opponents exploring: A learns values against an opponent that plays a random move 10% of the time, not against its best play. N's max over replies avoids this. This explains A's ~5× higher loss.
- The untrained game-0 snapshot in the pool is not a random player: random weights but fixed, so a fixed arbitrary policy.
- Training games always start from the initial position, while evaluation uses the opening bank (on 4×4, all 120 positions after 4 plies). That's part of why training coverage is so narrow. Random-opening starts would be the obvious fix: limitation / future work.
- **Pre-registration, for honesty.** `pool_p_latest` = 0.5 was fixed before E1 and never changed after seeing results. The Nov 24 settings change was made before any experiment, on one ladder seed. E1 and everything after ran with fixed settings.

### Validation (appendix?)
- 4×4 solver: all 58,533 reachable positions in 0.45 s. White wins 3–11 under perfect play, matching the published result. Eval set: 7,912 positions with chance accuracy 49.4%, and a random player measures exactly that. [§4.4]
- Debugging ladder on 4×4. Rung 1: env target vs a fixed random opponent, a stationary MDP. Rung 2: A-naive. Rung 3: N-naive. Rung 4: pool. All four passed. Rung 1 stores only the learner's moves, to keep its MDP stationary. [§4.6]
- **Minimax-optimal ≠ best vs a weak opponent.** Against a uniformly random opponent:

  | | on the 4×4 opening bank | from the start |
  |---|---|---|
  | Best possible (expectimax) | 83.4% | 98.0% |
  | π\* (perfect minimax play) | 77.8% | 88.1% |

  So the "≥ 95% vs random" bar I'd planned was impossible on the bank, and even perfect play scores only ~78% there. Score vs random saturates well below 100% and isn't a measure of optimality. Nice small point for the report.
- **A bug that would have looked like an RL failure.** With a ReLU after the network's 1×1 head conv, 15–25% of initializations start with a dead head: constant output, no gradient, and that seed never learns. Removed; a test checks 50 initialization seeds per board. A good "lessons learned" sentence. [§4.5]
- Reference players: on 4×4, greedy is no better than random (48%), because grabbing discs hands over edges and corners on a tiny board. Positional beats both. On 6×6 and 8×8 the expected order random < greedy < positional holds. That's one reason 4×4 is judged against the solver instead. [§4.3]
- Opening banks needed to be deeper than planned. Mirror images and rotations are the same opening to a symmetric net. At 2 plies, 4×4 has 12 openings but only 3 distinct up to symmetry, which gave ±20-point intervals. Revised to 4 plies (4×4) and 6 plies (6×6, 8×8). [§4.7]
- Planted-bug tests for the pool: 7 of 7 caught. The scary one: a snapshot that's secretly the live network silently turns pool into naive, and nothing crashes.
- `q_corr_4x4` (correlation of Q with exact values) shows whether Q ranks moves right before its scale is right. A useful companion to the MSE.

### Results: 4×4
- **Coverage.** Naive self-play reaches only ~200 of 7,912 eval positions (~2–3%). Nets are 97–99% optimal on positions they actually visit (Q corr 0.82–0.87 there), but only ~75–78% on the whole set; off those lines Q is barely informative (corr ~0.4). This is the narrow-distribution story (H1–H2) in miniature. [§4.6]
- **Pool widens coverage but doesn't buy accuracy on 4×4.** Pool training reaches ~1.4–1.5× more eval positions (rung 4), yet E1 shows no accuracy gain.
- **E1 (5 seeds per condition):**
  - N beats A by +2.7 points of accuracy; every N seed beats every A seed (p ≈ 10⁻⁹).
  - Pool ≈ naive: +0.2 points (p = 0.78), so H1 is null on 4×4.
  - Seeds are very consistent (sd 0.2–0.6 points). [§4.8]
- **Accuracy peaks around 10k games, then declines.**
  - −2.3 points by 30k in all 10 N runs; −1.3 in all 5 A-naive runs; A-pool roughly flat, but from a lower peak.
  - Pool doesn't prevent the drop under N, so on 4×4, "drawdown after the peak" happens under both samplers. H2-relevant.
  - **Cause confirmed (Nov 24): the replay buffer.** With a buffer that keeps every transition (400k vs 100k), the decline disappears in all 4 runs (A-naive and N-naive, seeds 0–1), and late accuracy is ~3 points higher (A 75.5 → 78.3%, N 78.4 → 81.5%). Q corr rises too (N 0.47 → 0.53). Older, more varied data is harder to fit (loss ~2×), but fitting it helps.
- **Headline-worthy: on 4×4, buffer size matters more than the opponent pool.** Keeping all the data adds ~3 points, as much as choosing N over A (+2.7) and far more than pool vs naive (+0.2). Spec D1 already called the buffer an *implicit pool* of recent past selves. Lengthening history on the data side helps; adding history on the opponent side doesn't. A nice framing for the discussion.
- **H4 on 4×4: no pool gap in either row.** But the buffer helps both rows about equally (+2.8 A, +3.1 N). H4's "what would change my mind" column says equal gaps in both rows would mean the help comes through *data diversity*, which both rows share, rather than through fixing A's opponent assumption. The buffer result is exactly a data-diversity effect. Worth connecting in the discussion. [§2 H4]
- **H3 on 4×4: the self-play score can't see the decline.**
  - The score vs the previous checkpoint is above 0.5 only in the first ~4k games. After that it sits at 0.47–0.54 in every run, including while accuracy falls 2.3 points.
  - Its Spearman correlation with accuracy is −0.2 to −0.3, far from H3's ρ > 0.7 threshold for "self-play tracks external strength".
  - Why: comparing consecutive checkpoints can't see a slow decline (~0.1 point per eval interval). Cross-play against older checkpoints (E3) or regret (E5) might.
- **E3 on 4×4 (Nov 27): the accuracy decline costs no games.** In each run's 15-checkpoint round robin, everything from ~7k on is within ~3 points of everything else. The ~10k checkpoints (around the accuracy peak) score 0.49–0.50 against the final one, and no earlier checkpoint gets more than 127 of 240 points against it. So the late nets make fewer exactly-optimal moves but don't lose more games. Framing: a decline in a metric, not in strength, and on 4×4 the flat self-play score was *right* about strength. [§4.8 E3]
- **E3 across runs on 4×4:** pool beats naive head-to-head by +1.2 ± 0.6 points, and N beats A by +0.8 ± 0.6, both small. Pool's edge is one the accuracy metric didn't show (+0.2 points). N-pool vs A-naive is +2.1 ± 0.7.
- **Match scores are much coarser than exact accuracy.** N vs A differ by 2.7 points of accuracy (p ≈ 10⁻⁹), but their scores vs random are identical. A 240-game match has a ~±6-point interval. On 6×6, with no exact metric, small effects will need the seeds, or bigger matches, to show up.
- The big-buffer curves match E1's exactly until game 8k, when E1's buffer fills: a free determinism check (same seed → same run, even on the GPU).
- My budget rule was off for 4×4: its 100k buffer holds ~27% of a run's transitions, not the 10% I'd assumed (games average ~12.4 plies). 6×6 and 8×8 really are ~10%.
- Pool learners win 53–55% of their training games, since they beat older snapshots. A-pool's loss is higher than A-naive's (0.032 vs 0.024): more varied opponents mean noisier A targets.

### 6×6 endgame metric (Nov 24)
- The endgame test set (500 positions, 10–12 empties, exact win/draw/loss values) contains only positions that are won (93%) or drawn (7%) for the side to move. A lost position can't test anything, because every move loses equally. So "endgame accuracy" means *keeping* a won or drawn result, never saving a lost one. Say this when defining the metric. Chance accuracy is 48.6%. [§4.4]
- First readings on the 6×6 pilot checkpoints: untrained 46%, 72–77% at 10k games, ~78–79% at 20k. Plenty of headroom.
- The exact solver turned out cheap (~0.1 s per 12-empty position, in plain Python). Deeper endgames (14–16 empties) would be feasible if a harder set is ever wanted.

### 6×6 pilots: endgame accuracy (backfilled Nov 25, 1 seed per condition)
- **Fast rise, then flat:** 46% untrained, 72–77% at 10k games, 80–85% by ~80–100k, then flat to 400k. In every run the mean over 200k–400k equals the mean over 100k–200k (within 0.004). **No peak-then-decline on 6×6.** Drawdowns of 0.024–0.042 are within eval noise (one eval's SE ≈ 0.017 at 500 positions).
- **Ordering over 200k–400k:** negamax-pool 0.850 > negamax-naive 0.844 > env-pool 0.833 > env-naive 0.815.
  - N beats A by ~2–3 points, the same direction and size as E1 on 4×4 (+2.7).
  - Pool beats naive by 0.6–1.8 points.
  - **Only one seed per condition:** a hint, not a result. E2's 3 seeds decide.
- **Endgame accuracy separates the conditions better than score vs positional,** which sits at ~0.89–0.90 for all four (near its ceiling). Lead with endgame accuracy for 6×6 in the report. At 10k games it already shows N ahead (0.75–0.77 vs 0.72).
- **The self-play score vs endgame accuracy:** Spearman ρ −0.18 to −0.44 per run (first point dropped, as in T1). The H3 pattern holds on 6×6 as well.
- Backfill checks: a test shows it reproduces training's logged values exactly, and CPU vs GPU evaluation agree on 163 of 164 pilot checkpoints (the other differs by one position of 500).

### E2: 6×6 2×2, 3 seeds (finished Nov 26; numbers in spec §4.8)
- **Headline: the target matters, the sampler doesn't.** N beats A on endgame accuracy by 1.4 points, with every N run above every A run, which replicates E1 on a bigger board. Pool vs naive is +0.8 points of endgame accuracy under both targets, not significant (p = 0.14), and there's no H4 interaction (bounded to about ±2 points).
- **H2 framing:** the curves are flat after ~100k games, and the drawdowns are what eval noise alone produces (plateau SD 0.020, the same as a 200-game match's binomial SD). Say so explicitly, and don't read T1's drawdown column as forgetting. On 6×6, any forgetting evidence has to come from cross-play and regret.
- **H3 nuance, worth a sentence:** ρ ≈ 0 on the plateau. The overall negative ρ comes from the learning phase, where the self-play score is high *because* the net is improving fast. It's a rate-of-improvement signal, not a strength signal.
- Scores vs greedy (0.97) and random (0.99) are at the ceiling on 6×6. On 8×8 the score vs greedy fell to ~0.68 (P8).
- T1's "final" is now the **last 10 eval points** (decided Nov 27, spec §4.7). The old last-3 version was noise-dominated on 6×6 (SE ≈ 0.012, against effects of ≤ 0.007) and gave pool − naive the opposite sign from the 200k–400k means. With 10 points: pool − naive is +0.4 (A) and 0.0 (N) points vs positional. In the report, define final as the last 10 points and say it was changed after E2 for noise reasons.
- **E3 on 6×6 (Nov 27):**
  - **No forgetting in whole games.** The best earlier checkpoint scores 0.51–0.58 against its final one, exactly what the best of 14 equal-strength checkpoints gives by chance (simulated 90% range 0.525–0.578).
  - **But the nets keep improving in play long after score vs positional flattens.** The final checkpoint scores ~0.55 against its own 120k checkpoint, and each checkpoint's mean score against its run's other checkpoints rises until ~320k. Worth a sentence: the fixed references saturate on 6×6, and the round robin is the sensitive strength measure there.
  - **Head-to-head across runs:** N beats A by +1.1 ± 0.7 points, which is E2's target effect showing up in real games, but small. Pool vs naive: nothing (−0.6 under A, +0.6 under N). H1 is null on 6×6 by its most direct test too.

### 8×8 pilot P8 (finished Nov 25, 1M games in 22h09m)
- **Final numbers:**
  - Score vs positional averages 0.908 over 200k–500k, 0.892 over 500k–800k and 0.884 over 800k–1M. Final (last 3 points) 0.879, peak 0.938 at 840k, max drawdown 0.098. That's just under the 10-point threshold, so 0 forgetting events. The ~2-point drift down is suggestive at most with one seed.
  - Score vs greedy **kept falling**: 0.776 → 0.704 → 0.678 over the same segments, with lows of 0.61 at 920k and 980k (CI 0.54–0.67).
  - Score vs the previous checkpoint ≈ 0.51 throughout. Its Spearman ρ with score vs positional is −0.10 (first point dropped).
  - `mean_q` flat at ~0.42–0.44 from 400k on.
- The midpoint notes below still hold. The fall vs greedy is the main 8×8 story, and it's the dynamic E6's pool run should be compared on.

#### Midpoint notes (500k of 1M games, Nov 25)
- **Gate 2's 8×8 bar is already cleared:** ~0.90 vs positional since 200k games (95% CI at 500k: 0.81–0.91), far above the 50% bar.
- **Plateau vs positional since ~200k**, which is exactly where ε reaches its floor (0.2 × 1M). Since then it has stayed between 0.87 and 0.93, all within eval noise. No decline yet, though the last two points (0.89, 0.87) are the lowest since 180k. The budget caveat below applies: a shorter E6 would hit the ε floor sooner.
- **Score vs greedy fell while score vs positional rose:** 0.92 at 20k–60k (CI 0.87–0.95 at 40k), down to 0.70–0.83 from 280k on (CI 0.64–0.77 at 320k). The CIs don't overlap, so it's real. 6×6 shows nothing like it: all four pilots stay at ~0.97 vs greedy throughout. So on 8×8 the agent got stronger against one fixed opponent and weaker against another. That's non-transitivity, and a good argument for using several external references rather than one (H3-adjacent). A guess, untested: once ε is low, self-play lines stop covering the greedy player's flip-maximising openings.
- **`mean_q` fell from 0.75 to ~0.44 and flattened after ~300k.** Q is the value of greedy play against an ε-greedy opponent, so it should fall as the opponent's ε falls and stop once ε stops (plus a lag, since the 6M buffer holds ~100k games). It's a consistency check on the A target, not a strength measure.
- Pace: 6.4 games/s while sharing the GPU with the four 6×6 pilots, 15.9 alone. The checkpoint-vs-previous score averages ~0.5 after 240k; the 480k checkpoint lost to the 460k one (0.38, CI 0.31–0.45).

### E1's summary table (T1, `tables/e1-4x4/`)
- **H4 on 4×4 is null:** env gap − negamax gap is +0.001 ± 0.004 in final accuracy. Pool − naive is +0.002 (A) and +0.001 (N). *(Nov 27: with final now the last 10 points. With the last 3 it was +0.000 ± 0.005 and +0.004 under both targets.)* The final column now equals E1's §4.8 results table exactly.
- **No forgetting events at the 10-point threshold in any of the 20 runs.** The post-peak decline is real but only ~3 points (max drawdown 0.026–0.038). If forgetting matters on 4×4, a smaller threshold would be needed, but don't change the definition after seeing the data. Just report that the threshold doesn't fire on 4×4.
- **Spearman ρ between the self-play score and accuracy is negative in every condition** (−0.16 to −0.31). Clear H3 support: the self-play score doesn't track strength.
  - *Updated Nov 25:* the first point is always the 1k net vs the untrained net (self ≈ 0.98, accuracy at its lowest), and it dragged ρ down. **T1 now drops it** (spec §4.7). New range: −0.07 to −0.25 per condition, and a few single seeds are slightly positive (up to +0.12). Still nowhere near H3's ρ > 0.7, so the conclusion stands, but say "≈ 0 to weakly negative", not "negative". When defining ρ in the report, say the first point is dropped and why.
- **Caveat:** "peak" and "max drawdown" are biased upward by eval noise, since the peak is the max of noisy points. The seed bands in F1 show how big that noise is.

### Settings change (Nov 24)
- Changed `eps_end` 0.05 → 0.1 and `target_update_every` 1000 → 250 on all boards, after the ladder rerun.
  - Why the target update mattered: with γ = 1, a value moves back about one decision per target-net update, and a 4×4 run had only ~11–23 updates at 1,000. It now has ~92.
  - Faster early learning: N's accuracy at 2k games went from 57% to 71%.
  - +2–3 points late.
  - ε = 0.1 widens coverage only from ~2% to ~3%.
  - The evidence is from 4×4 only. It was applied to 6×6 and 8×8 for consistency, untested there.
- If there's time, E7 reruns the old settings for a comparison. [§4.5, §4.8]

### Compute (appendix / reproducibility)
- **E6's A-naive seed 0 is the P8 pilot**, copied in rather than rerun: same config, seed and training code, and runs are deterministic per seed. Its `meta.json` shows P8's commit (4d458d1) and its config says `exp: pilot-8x8`. Say so in the appendix.
- **E2 and E6 launched early (Nov 25)**, as one chain over Thanksgiving, once the pilots had answered the Nov 30 questions. Training code is frozen from that launch.
- E2 took 18 h 55 min for 12 runs (6.0–6.4 h each, 4 at a time), against the ~16½ h estimate.
- Everything trains on the laptop's Apple GPU (M1 Pro, `mps`). Even 4×4 was too slow on CPU: 50 ms per backward pass vs 3.8 ms on the GPU, and more CPU threads made it slower for this small net.
- Four runs sharing the GPU get ~2× the total throughput of one run alone.
- The server (2 ARM cores, no GPU) is ~39× slower than the laptop GPU, so it isn't used for the main runs.
- Budget rule of thumb: replay ≈ 10% of a run's transitions; ~40–50 eval points per run. [§4.8]

## Come back to
- [x] **Buffer test** (Nov 24, `make replay-4x4`): confirmed, see above. E4 is now "buffer size", with a bigger buffer rather than ¼ size.
- [x] Do the 6×6 pilots show the same peak-then-decline? **No** (Nov 25): endgame accuracy and score vs positional both plateau from ~100k to 400k, with no decline (see "6×6 pilots" above). On 6×6, ε reaches its floor at 80k games and the buffer holds ~40k games.
- [x] ~~If they do: raise E2's buffer for all conditions?~~ Not needed on this evidence. Keep the buffer setting for E2 unless E2's own curves say otherwise.
- [x] Backfill the 6×6 pilots' endgame accuracy from their checkpoints (Nov 25, `make backfill EXPS=pilot-6x6`).
- [x] T1's Spearman ρ: drop the first self-play point (always the 1k/10k net vs the untrained one)? **Yes** (Nov 25). E1's T1 regenerated.
- [ ] Optional: run E1's full 2×2 with the big buffer on 4×4 (~26 min). Does the pool matter even less once the buffer keeps everything? D1's implicit-pool idea predicts yes.
- [x] **E3 on 4×4:** does an E1 run's ~10k-game checkpoint beat its final checkpoint in real games? **No** (Nov 27): 0.49–0.50 against it. The accuracy decline costs no games (see Results: 4×4).
- [ ] Caveat when picking E2's budget from the pilots: the ε schedule and the buffer scale with `games`. A 200k run reaches ε's floor at 40k (the pilot at 80k), so the pilot's curve at 200k only approximates a 200k run.
- [ ] P8's fall vs greedy: when P8 finishes, play its 60k checkpoint against its final one (crossplay). If the early checkpoint wins, the self-play score missed a regression. Also check whether the fall vs greedy continues past 500k. *(Nov 27: E6's round robin covers it, since its 15 checkpoints include 20k, which scored 0.915 vs greedy like 40k–80k. Run `make crossplay EXPS=e6-8x8` once E6 finishes.)*
- [ ] Does pool's extra coverage matter on 6×6, where the state space is far larger than 4×4's? That's the real test of H1. *E2 (Nov 27): not on endgame accuracy or score vs positional (+0.8 and +0.4 points, not significant). Head-to-head cross-play between pool and naive agents is the last direct test.* **E3 (Nov 27): nothing there either** (−0.6 ± 1.1 under A, +0.6 ± 1.9 under N). On 6×6, H1 is null on every measure.
- [x] **T1's "final" window:** last 3 eval points or the last 10? **The last 10, on every board** (decided Nov 27; spec §4.7). E1's and E2's T1 regenerated.
- [ ] **E2+ (in the plan Nov 27, if time):** E2 seeds 3–4 after E6, `make sweep-6x6-extra` (spec §4.8 row E2+). Rules fixed in advance: both seeds, one analysis at 5 seeds, no further extension. In the report, say it was added after 3 seeds (optional stopping) and keep the 3-seed numbers.
- [x] E3 addition: an across-run round robin of every final checkpoint (pool vs naive, A vs N, head-to-head), alongside the within-run ones. **In the plan for all three boards** (Nov 27, spec §4.7).
- [ ] A-pool's flat curve: is pool slowing A's early learning slightly (lower peak), or protecting it from the decline? *F4 (Nov 27) leans towards slower learning: A-pool's final checkpoint still scores 0.52–0.54 against its own 5k–13k checkpoints, where A-naive's scores 0.50–0.53 (`figures/e1-4x4/f4_final_vs_earlier.pdf`, the blue block in `f4_crossplay.pdf`). So A-pool keeps improving in play for longer. Small (~2 points) and 5 seeds; check against the accuracy curves before using it.*
- [ ] If citing a training-time pool sampling rule, check how TiZero (MADRID's ref. [29]) samples opponents. [§4.6]
- [ ] E7 (old settings) if there's spare compute: is the settings change itself a finding?
- [ ] `learner_score` in the logs comes from training games with ε on. Don't plot it as a strength measure.
- [ ] Wording: `score_selfplay` (`self=` in stdout) is the score vs the *previous checkpoint* (D5), not vs itself. Its first value is always ~0.97–0.99, because the previous checkpoint is the untrained game-0 net. In the report, call it "score vs the previous checkpoint", or define "self-play score" once, clearly.
- [ ] E1's `meta.json` files say `dirty: true`. Only the spec was uncommitted then; the code is commit 091c38e. Mention in the reproducibility appendix if relevant.
