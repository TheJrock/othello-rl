"""Exact checks for 4×4 debugging-ladder runs (spec §4.6), with no sampling noise.

    python scripts/ladder_check.py runs/ladder/env-random/seed0 runs/ladder/env-naive/seed0 ...

For each run's final checkpoint (greedy, ε = 0), prints:
  vs random   exact expected score against a uniformly random opponent, on the opening bank and from
              the start position, next to the best possible (expectimax) and π*'s score
  minimax     optimal-move accuracy and Q vs the exact minimax values, over the whole 7,912-position
              eval set and over the distinct positions the net reaches in self-play from the start
              (ε = 0.05, where self-play training data comes from)
  vs-random   Q vs the exact values of rung 1's MDP (learner vs random), on the distinct positions the
              net reaches against random from the start (where rung 1's training data comes from)

Coverage is the point of the split: a net can fit the positions its own games produce and still be
wrong on the many eval positions those games never reach.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from othello_rl import rundir, solver
from othello_rl.engine import State, initial, legal_mask, to_planes
from othello_rl.evaluate import load_states, solver_values
from othello_rl.nets import load_net
from othello_rl.players import NetPlayer, RandomPlayer
from othello_rl.selfplay import play_games


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--table", default="data/cache/solve4x4.npz")
    p.add_argument("--openings", default="data/openings_4x4.json")
    args = p.parse_args()

    table = solver.load_table(args.table)
    bank = load_states(args.openings)
    start = [initial(4)]
    vr = solver.solve_vs_random_4x4()
    keys = sorted(table)
    positions = solver.eval_positions_4x4(table)
    minimax = solver_values(table, positions)

    def best(s, v):
        return v == np.nanmax(v)

    def perfect(s, v):
        return solver.optimal_actions(table, s)

    print(f"{'':28s} {'bank':>6s} {'start':>6s}")
    for name, choose in (("best possible vs random", best), ("π* (minimax) vs random", perfect)):
        print(
            f"{name:28s} {solver.exact_score_vs_random(choose, bank):6.3f} {solver.exact_score_vs_random(choose, start):6.3f}"
        )

    for run in args.runs:
        cfg = rundir.load_run_config(run)
        games, ckpt = rundir.list_checkpoints(run)[-1]
        net = load_net(ckpt, cfg)
        with torch.no_grad():
            q_all = net(torch.from_numpy(np.stack([to_planes(State(m, o, 4)) for m, o in keys]))).numpy()
        q = dict(zip(keys, q_all))

        def greedy(s, v, q=q):  # bind this run's Q
            legal = legal_mask(s)
            row = np.where(legal, q[(s.me, s.opp)], -np.inf)
            return row == row.max()

        print(f"\n{run} (g={games})")
        print(
            f"  vs random    bank {solver.exact_score_vs_random(greedy, bank):.3f}   start "
            f"{solver.exact_score_vs_random(greedy, start):.3f}"
        )

        def fit(states, exact, q=q):
            """corr, mse and best-move rate of Q against exact values (NaN = illegal) on states."""
            qn = np.stack([q[(s.me, s.opp)] for s in states])
            ok = ~np.isnan(exact)
            picks = np.where(ok, qn, -np.inf).argmax(axis=1)
            best_rate = np.mean(exact[np.arange(len(states)), picks] == np.nanmax(exact, axis=1))
            return np.corrcoef(qn[ok], exact[ok])[0, 1], np.mean((qn[ok] - exact[ok]) ** 2), best_rate

        def decisions(black, white, mine):
            """Distinct positions with ≥ 2 legal moves where mine(game, state) says the net is to move."""
            trajs = play_games(black, white, start * len(black), np.random.default_rng(0))
            seen = {
                (ply.state.me, ply.state.opp): ply.state
                for i, t in enumerate(trajs)
                for ply in t.plies
                if mine(i, ply.state) and legal_mask(ply.state).sum() >= 2
            }
            return list(seen.values())

        corr, mse, rate = fit(positions, minimax)
        print(
            f"  minimax, all 7,912 eval positions   accuracy {rate:.3f} (chance 0.494)   Q corr {corr:.3f}   mse {mse:.4f}"
        )

        # Where self-play training data comes from: the net (ε = 0.05) against itself from the start.
        me, rnd, n = NetPlayer(net, epsilon=0.05), RandomPlayer(), 2000
        sp = decisions([me] * n, [me] * n, lambda i, s: True)
        corr, mse, rate = fit(sp, np.stack([solver.child_values(table, s) for s in sp]))
        print(
            f"  minimax, {len(sp):5d} self-play positions   accuracy {rate:.3f}   Q corr {corr:.3f}   mse {mse:.4f}"
        )

        # Where rung 1's data comes from: the net against random from the start, its own decisions.
        half = n // 2
        vsr = decisions(
            [me] * half + [rnd] * half,
            [rnd] * half + [me] * half,
            lambda i, s, half=half: s.black_to_move == (i < half),
        )
        corr, mse, rate = fit(vsr, np.stack([vr.child_values(s) for s in vsr]))
        print(
            f"  vs-random, {len(vsr):5d} positions vs random   best-vs-random move {rate:.3f}   Q corr {corr:.3f}   mse {mse:.4f}"
        )


if __name__ == "__main__":
    main()
