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
- **Match scores are much coarser than exact accuracy.** N vs A differ by 2.7 points of accuracy (p ≈ 10⁻⁹), but their scores vs random are identical. A 240-game match has a ~±6-point interval. On 6×6, with no exact metric, small effects will need the seeds, or bigger matches, to show up.
- The big-buffer curves match E1's exactly until game 8k, when E1's buffer fills: a free determinism check (same seed → same run, even on the GPU).
- My budget rule was off for 4×4: its 100k buffer holds ~27% of a run's transitions, not the 10% I'd assumed (games average ~12.4 plies). 6×6 and 8×8 really are ~10%.
- Pool learners win 53–55% of their training games, since they beat older snapshots. A-pool's loss is higher than A-naive's (0.032 vs 0.024): more varied opponents mean noisier A targets.

### Settings change (Nov 24)
- Changed `eps_end` 0.05 → 0.1 and `target_update_every` 1000 → 250 on all boards, after the ladder rerun.
  - Why the target update mattered: with γ = 1, a value moves back about one decision per target-net update, and a 4×4 run had only ~11–23 updates at 1,000. It now has ~92.
  - Faster early learning: N's accuracy at 2k games went from 57% to 71%.
  - +2–3 points late.
  - ε = 0.1 widens coverage only from ~2% to ~3%.
  - The evidence is from 4×4 only. It was applied to 6×6 and 8×8 for consistency, untested there.
- If there's time, E7 reruns the old settings for a comparison. [§4.5, §4.8]

### Compute (appendix / reproducibility)
- Everything trains on the laptop's Apple GPU (M1 Pro, `mps`). Even 4×4 was too slow on CPU: 50 ms per backward pass vs 3.8 ms on the GPU, and more CPU threads made it slower for this small net.
- Four runs sharing the GPU get ~2× the total throughput of one run alone.
- The server (2 ARM cores, no GPU) is ~39× slower than the laptop GPU, so it isn't used for the main runs.
- Budget rule of thumb: replay ≈ 10% of a run's transitions; ~40–50 eval points per run. [§4.8]

## Come back to
- [x] **Buffer test** (Nov 24, `make replay-4x4`): confirmed, see above. E4 is now "buffer size", with a bigger buffer rather than ¼ size.
- [ ] Do the 6×6 pilots show the same peak-then-decline? On 6×6, ε reaches its floor at 80k games and the buffer holds ~40k games, a smaller share of training than on 4×4. Read the curves for a peak, not just a plateau, when setting E2's budget (Nov 30).
- [ ] If they do: raise E2's buffer for all conditions (one of the ≤ 2 shared knobs)? A full 6×6 buffer is ~3.4 GB per run, fine on 32 GB. Could also run a 6×6 big-buffer pilot to check first.
- [ ] Optional: run E1's full 2×2 with the big buffer on 4×4 (~26 min). Does the pool matter even less once the buffer keeps everything? D1's implicit-pool idea predicts yes.
- [ ] **E3 on 4×4:** does an E1 run's ~10k-game checkpoint beat its final checkpoint in real games? If so, that's a concrete H2/H3 example: a decline the self-play score never flagged. Cheap once `crossplay_matrix` exists.
- [ ] Caveat when picking E2's budget from the pilots: the ε schedule and the buffer scale with `games`. A 200k run reaches ε's floor at 40k (the pilot at 80k), so the pilot's curve at 200k only approximates a 200k run.
- [ ] Does pool's extra coverage matter on 6×6, where the state space is far larger than 4×4's? That's the real test of H1.
- [ ] A-pool's flat curve: is pool slowing A's early learning slightly (lower peak), or protecting it from the decline?
- [ ] If citing a training-time pool sampling rule, check how TiZero (MADRID's ref. [29]) samples opponents. [§4.6]
- [ ] E7 (old settings) if there's spare compute: is the settings change itself a finding?
- [ ] `learner_score` in the logs comes from training games with ε on. Don't plot it as a strength measure.
- [ ] E1's `meta.json` files say `dirty: true`. Only the spec was uncommitted then; the code is commit 091c38e. Mention in the reproducibility appendix if relevant.
