"""Eval harness: Wilson intervals, opening banks, matches, 4×4 solver metrics (spec §4.7, D4)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from othello_rl import engine as E
from othello_rl import solver
from othello_rl.evaluate import (
    EvalContext,
    MatchStats,
    endgame_accuracy,
    endgame_values,
    evaluate_checkpoint,
    load_endgame_set,
    load_states,
    opening_bank,
    optimal_move_accuracy,
    play_match,
    q_solver_corr,
    q_solver_mse,
    reference_players,
    save_states,
    solver_values,
    wilson_interval,
)
from othello_rl.players import NetPlayer, RandomPlayer


class Lowest:
    """Deterministic: always the lowest legal action (pass included when forced)."""

    def __init__(self, name="lowest"):
        self.name = name
        self.batches: list[list[E.State]] = []

    def act(self, states, rng):
        self.batches.append(list(states))
        return [int(np.flatnonzero(E.legal_mask(s))[0]) for s in states]


class Highest(Lowest):
    def act(self, states, rng):
        return [int(np.flatnonzero(E.legal_mask(s))[-1]) for s in states]


# --- Wilson interval ----------------------------------------------------------------------------


def test_wilson_known_values():
    lo, hi = wilson_interval(0.5, 100)
    assert lo == pytest.approx(0.4038, abs=1e-4) and hi == pytest.approx(0.5962, abs=1e-4)
    assert wilson_interval(0.0, 10) == pytest.approx((0.0, 0.2775), abs=1e-4)
    assert wilson_interval(1.0, 10) == pytest.approx((0.7225, 1.0), abs=1e-4)


@pytest.mark.parametrize("score", [0.0, 0.1, 0.37, 0.5, 0.9, 1.0])
def test_wilson_properties(score):
    lo, hi = wilson_interval(score, 50)
    assert 0.0 <= lo <= score <= hi <= 1.0
    lo2, hi2 = wilson_interval(1 - score, 50)
    assert (lo2, hi2) == pytest.approx((1 - hi, 1 - lo))  # symmetric under swapping the players
    lo3, hi3 = wilson_interval(score, 500)
    assert hi3 - lo3 < hi - lo  # more games, narrower interval


def test_wilson_rejects_bad_input():
    for score, n in [(0.5, 0), (1.2, 10), (-0.1, 10)]:
        with pytest.raises(ValueError):
            wilson_interval(score, n)


def test_match_stats_ci_uses_draws_as_half():
    m = MatchStats(wins=3, draws=2, losses=5)
    assert m.games == 10 and m.score == 0.4
    assert m.ci() == wilson_interval(0.4, 10)


# --- opening banks ------------------------------------------------------------------------------


def test_enumerated_4x4_bank():
    bank = opening_bank(4, plies=2, count=None, seed=0)
    assert len(bank) == 12  # 4 first moves × 3 replies; two different plies can't transpose this early
    assert len(set(bank)) == 12
    for s in bank:
        assert s.black_to_move and (s.me | s.opp).bit_count() == 6 and not E.is_terminal(s)
    assert bank == opening_bank(4, plies=2, count=None, seed=99)  # enumeration ignores the seed


def test_sampled_bank_is_distinct_reproducible_and_reachable():
    bank = opening_bank(6, plies=4, count=100, seed=0)
    assert len(bank) == len(set(bank)) == 100
    for s in bank:
        assert s.black_to_move and (s.me | s.opp).bit_count() == 8 and not E.is_terminal(s)
    assert bank == opening_bank(6, plies=4, count=100, seed=0)
    assert bank != opening_bank(6, plies=4, count=100, seed=1)
    every = opening_bank(6, plies=4, count=None, seed=0)
    assert len(every) == 236  # regression number (Nov 23); 60 up to symmetry
    assert set(bank) <= set(every)


def test_bank_refuses_impossible_counts():
    with pytest.raises(ValueError, match="found only 12"):
        opening_bank(4, plies=2, count=13, seed=0)


def test_bank_file_round_trip(tmp_path):
    bank = opening_bank(6, plies=4, count=20, seed=0)
    save_states(bank, tmp_path / "openings.json")
    assert load_states(tmp_path / "openings.json") == bank


# --- matches ------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", E.SIZES)
def test_player_vs_clone_scores_exactly_half(n, rng):
    """Each opening is played once with each side moving first, so against an identical
    deterministic player the two games mirror each other and wins cancel losses exactly."""
    openings = opening_bank(n, plies=2, count=None, seed=0)
    for p1, p2 in [(Lowest(), Lowest()), (Highest(), Highest())]:
        m = play_match(p1, p2, openings, rng)
        assert m.games == 2 * len(openings)
        assert m.wins == m.losses and m.score == 0.5
    same = Lowest()
    assert play_match(same, same, openings, rng).score == 0.5  # one object on both sides


def test_p1_moves_first_in_exactly_one_game_per_opening(rng):
    openings = opening_bank(6, plies=4, count=30, seed=0)
    p1 = Lowest("p1")
    play_match(p1, Highest("p2"), openings, rng)
    assert p1.batches[0] == openings  # one batched call, every opening once, in order


def test_swapping_players_mirrors_the_result(rng):
    openings = opening_bank(6, plies=4, count=30, seed=0)
    a = play_match(Lowest(), Highest(), openings, rng)
    b = play_match(Highest(), Lowest(), openings, rng)
    assert (b.wins, b.draws, b.losses) == (a.losses, a.draws, a.wins)


def test_play_match_needs_openings(rng):
    with pytest.raises(ValueError):
        play_match(Lowest(), Highest(), [], rng)


def test_reference_strengths_on_6x6_bank(rng):
    """Spec §4.3 sanity check: positional ≫ random and greedy > random on 6×6 (the data/ bank's recipe)."""
    refs = reference_players(6)
    openings = opening_bank(6, plies=6, count=100, seed=0)
    assert play_match(refs["positional"], refs["random"], openings, rng).score > 0.75
    assert play_match(refs["greedy"], refs["random"], openings, rng).score > 0.5
    assert play_match(refs["positional"], refs["greedy"], openings, rng).score > 0.5


# --- 4×4 solver metrics -------------------------------------------------------------------------


class SolverQNet(torch.nn.Module):
    """A perfect Q for 4×4: decodes the input planes back into a position and returns the solver's
    exact value for every legal move (−1 for illegal ones, which play is masked from anyway)."""

    def __init__(self, table, scale=1.0, offset=0.0):
        super().__init__()
        self.table, self.scale, self.offset = table, scale, offset

    def forward(self, x):
        rows = []
        for planes in x.cpu().numpy().round().astype(np.uint8):
            me, opp = (int(np.packbits(p.ravel(), bitorder="little")[:2].view("<u2")[0]) for p in planes[:2])
            rows.append(np.nan_to_num(solver.child_values(self.table, E.State(me, opp, 4)), nan=-1.0))
        return torch.tensor(np.stack(rows) * self.scale + self.offset, dtype=torch.float32)


@pytest.fixture(scope="module")
def positions(table):
    return solver.eval_positions_4x4(table)[::10]  # every 10th: ~790 positions, enough and quick


@pytest.fixture(scope="module")
def values(table, positions):
    return solver_values(table, positions)


def test_solver_qnet_decodes_planes(table, positions):
    """The test double itself: Q from the planes equals child_values of the original position."""
    q = NetPlayer(SolverQNet(table)).q_values(positions[:20])
    for s, row in zip(positions[:20], q):
        legal = E.legal_mask(s)
        np.testing.assert_array_equal(row[legal], solver.child_values(table, s)[legal])


def test_perfect_q_scores_perfectly(table, positions, values, rng):
    player = NetPlayer(SolverQNet(table))
    assert optimal_move_accuracy(player, positions, values, rng) == 1.0
    assert q_solver_mse(player, positions, values) == 0.0
    assert q_solver_corr(player, positions, values) == pytest.approx(1.0)


def test_corr_ignores_scale_and_offset_but_mse_does_not(table, positions, values, rng):
    """A Q that ranks every move correctly but is shrunk toward 0.2 still plays perfectly: the
    correlation stays 1 while the MSE shows the magnitudes are off."""
    player = NetPlayer(SolverQNet(table, scale=0.3, offset=0.2))
    assert optimal_move_accuracy(player, positions, values, rng) == 1.0
    assert q_solver_corr(player, positions, values) == pytest.approx(1.0)
    assert q_solver_mse(player, positions, values) > 0.1


def test_random_accuracy_matches_chance(table, positions, values, rng):
    chance = solver.chance_accuracy(table, positions)
    acc = np.mean([optimal_move_accuracy(RandomPlayer(), positions, values, rng) for _ in range(20)])
    assert acc == pytest.approx(chance, abs=0.02)


def test_illegal_moves_are_never_optimal(positions, values, rng):
    class PlaysPass:
        name = "pass"

        def act(self, states, rng):
            return [16] * len(states)  # pass is illegal in every eval position (≥ 2 legal moves)

    assert optimal_move_accuracy(PlaysPass(), positions, values, rng) == 0.0


def test_evaluate_checkpoint_fields(table, positions, values, rng):
    from othello_rl.config import Config
    from othello_rl.nets import build_net

    cfg = Config(board=4, channels=8, layers=1)
    ctx = EvalContext(
        openings=opening_bank(4, 4, None, 0)[:5],
        references=reference_players(4),
        table=table,
        positions_4x4=positions,
        values_4x4=values,
    )
    net = build_net(cfg)
    first = evaluate_checkpoint(net, None, cfg, ctx, rng)
    assert "score_selfplay" not in first  # no previous checkpoint at game 0
    row = evaluate_checkpoint(net, net, cfg, ctx, rng)
    for name in ("random", "greedy", "positional", "selfplay"):
        assert 0.0 <= row[f"score_{name}"] <= 1.0
        lo, hi = row[f"ci_{name}"]
        assert lo <= row[f"score_{name}"] <= hi
    assert row["score_selfplay"] == 0.5  # a greedy net against itself, each opening from both sides
    assert {"accuracy_4x4", "q_mse_4x4", "q_corr_4x4"} <= row.keys()


# --- endgame accuracy (the 6×6 metric, checked on 4×4 records against the full table) -------------


@pytest.fixture(scope="module")
def endgame_records(table, tmp_path_factory):
    """A 4×4 endgame set written to and read back from the data/ file format (null ↔ NaN)."""
    import json

    records = solver.build_endgame_set(count=60, min_empties=5, max_empties=9, seed=0, workers=1, n=4)
    path = tmp_path_factory.mktemp("endgame") / "endgame4.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return load_endgame_set(path)


def test_endgame_values_match_solver_values(table, endgame_records):
    positions, values = endgame_values(endgame_records)
    np.testing.assert_array_equal(values, solver_values(table, positions))


def test_endgame_accuracy_perfect_random_and_illegal(table, endgame_records, rng):
    assert endgame_accuracy(solver.PerfectPlayer(table), endgame_records, rng) == 1.0
    positions, _ = endgame_values(endgame_records)
    chance = solver.chance_accuracy(table, positions)
    acc = np.mean([endgame_accuracy(RandomPlayer(), endgame_records, rng) for _ in range(200)])
    assert acc == pytest.approx(chance, abs=0.02)

    class PlaysPass:
        name = "pass"

        def act(self, states, rng):
            return [16] * len(states)

    assert endgame_accuracy(PlaysPass(), endgame_records, rng) == 0.0


def test_evaluate_checkpoint_logs_endgame_accuracy(endgame_records, rng):
    from othello_rl.config import Config
    from othello_rl.nets import build_net

    cfg = Config(board=4, channels=8, layers=1)
    ctx = EvalContext(
        openings=opening_bank(4, 4, None, 0)[:5], references=reference_players(4), endgame_set=endgame_records
    )
    row = evaluate_checkpoint(build_net(cfg), None, cfg, ctx, rng)
    assert 0.0 <= row["endgame_accuracy"] <= 1.0
    assert "accuracy_4x4" not in row
