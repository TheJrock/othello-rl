"""Eval harness: Wilson intervals, opening banks, matches (spec §4.7, D4)."""

from __future__ import annotations

import numpy as np
import pytest

from othello_rl import engine as E
from othello_rl.evaluate import (
    MatchStats,
    load_states,
    opening_bank,
    play_match,
    reference_players,
    save_states,
    wilson_interval,
)


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
