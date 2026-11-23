"""Reference players and the network wrapper (spec §4.3). Board strings: X = side to move, O = other side."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch
from torch import nn

from othello_rl import engine as E
from othello_rl.nets import QNet
from othello_rl.players import POSITIONAL_WEIGHTS, GreedyPlayer, NetPlayer, PositionalPlayer, RandomPlayer


def random_positions(n: int, count: int, rng: np.random.Generator) -> list[E.State]:
    """Positions met along random games (terminal ones excluded; forced passes included)."""
    out: list[E.State] = []
    while len(out) < count:
        s = E.initial(n)
        while not E.is_terminal(s) and len(out) < count:
            out.append(s)
            s, _, _ = E.step(s, int(rng.choice(np.flatnonzero(E.legal_mask(s)))))
    return out


class FixedQ(nn.Module):
    """Returns the same Q-vector for every position and records each input batch."""

    def __init__(self, q):
        super().__init__()
        self.register_buffer("q", torch.as_tensor(q, dtype=torch.float32))
        self.inputs: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.inputs.append(x)
        return self.q.repeat(len(x), 1)


class NoiseQ(nn.Module):
    """Independent random Q-values in [−1, 1] for every position and action, illegal ones included."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        return torch.rand(len(x), n * n + 1) * 2 - 1


def all_players(n: int) -> list:
    return [
        RandomPlayer(),
        GreedyPlayer(),
        PositionalPlayer(n),
        NetPlayer(NoiseQ(), epsilon=0.0),
        NetPlayer(NoiseQ(), epsilon=0.5),
    ]


def forced_pass_state(n: int) -> E.State:
    """X to move with no square: its only disc can't flank the O in the corner, but O can flank X."""
    s = E.from_string("OX" + "." * (n * n - 2))
    assert E.legal_mask(s)[E.pass_action(n)]
    return s


# --- contract shared by every player ------------------------------------------------------------


@pytest.mark.parametrize("n", E.SIZES)
def test_players_return_one_legal_action_per_state(n, rng):
    states = random_positions(n, 300, rng)
    for p in all_players(n):
        actions = p.act(states, rng)
        assert len(actions) == len(states)
        assert all(type(a) is int for a in actions)
        for s, a in zip(states, actions):
            assert E.legal_mask(s)[a], f"{p.name} played illegal {a} in\n{E.to_string(s)}"


@pytest.mark.parametrize("n", E.SIZES)
def test_players_pass_when_forced(n, rng):
    s = forced_pass_state(n)
    for p in all_players(n):
        assert p.act([s, s], rng) == [E.pass_action(n)] * 2, p.name


@pytest.mark.parametrize("n", E.SIZES)
def test_players_reject_terminal_states(n, rng):
    s = E.from_string("X" + "." * (n * n - 1))  # Black has no opponent discs to flank: game over
    assert E.is_terminal(s)
    for p in all_players(n):
        with pytest.raises(ValueError):
            p.act([E.initial(n), s], rng)


def test_empty_batch(rng):
    for p in all_players(4):
        assert p.act([], rng) == []


def test_same_seed_same_actions():
    states = random_positions(6, 200, np.random.default_rng(1))
    for p in all_players(6)[:3]:  # NoiseQ draws from torch's global RNG, so skip the net players
        a1 = p.act(states, np.random.default_rng(7))
        a2 = p.act(states, np.random.default_rng(7))
        assert a1 == a2, p.name


@pytest.mark.parametrize(
    "player", [RandomPlayer(), GreedyPlayer(), PositionalPlayer(4)], ids=lambda p: p.name
)
def test_ties_are_broken_at_random(player, rng):
    """From the start, all 4 moves flip one disc and (on 4×4) have weight 0: every player must spread out."""
    s = E.initial(4)
    counts = Counter(player.act([s] * 400, rng))
    assert set(counts) == set(np.flatnonzero(E.legal_mask(s)).tolist())
    assert min(counts.values()) > 60  # expected 100 each; this fails with probability < 1e-8


# --- greedy and positional ----------------------------------------------------------------------


def test_greedy_maximises_flips(rng):
    s = E.from_string("XOO. O... .... ....")
    # (0,3) flips the two O's on row 0; (2,0) flips the one O on column 0. Nothing else is legal.
    assert np.flatnonzero(E.legal_mask(s)).tolist() == [3, 8]
    assert E.flips(s, 3).bit_count() == 2 and E.flips(s, 8).bit_count() == 1
    assert GreedyPlayer().act([s] * 50, rng) == [3] * 50


# 6×6 positions found by a seeded search and checked by hand. X to move.
WEIGHT_BEATS_FLIPS = """
......
....X.
..OXO.
..OO..
.XOX..
..O...
"""
TIE_BROKEN_BY_FLIPS = """
...O..
...O..
.OOO..
..OOX.
.....X
......
"""


def test_positional_prefers_weight_over_flips(rng):
    s = E.from_string(WEIGHT_BEATS_FLIPS)
    w = POSITIONAL_WEIGHTS[6].ravel()
    legal = np.flatnonzero(E.legal_mask(s)).tolist()
    assert legal == [13, 17, 22]
    assert [E.flips(s, a).bit_count() for a in legal] == [2, 1, 1]
    assert [w[a] for a in legal] == [-2, 10, -2]  # 17 = (2,5) is an edge square worth 10
    assert PositionalPlayer(6).act([s] * 50, rng) == [17] * 50
    assert GreedyPlayer().act([s] * 50, rng) == [13] * 50  # the two heuristics really do disagree


def test_positional_breaks_weight_ties_by_flips(rng):
    s = E.from_string(TIE_BROKEN_BY_FLIPS)
    w = POSITIONAL_WEIGHTS[6].ravel()
    legal = np.flatnonzero(E.legal_mask(s)).tolist()
    assert legal == [8, 19]
    assert w[8] == w[19] == -2
    assert E.flips(s, 8).bit_count() == 1 and E.flips(s, 19).bit_count() == 2
    assert PositionalPlayer(6).act([s] * 50, rng) == [19] * 50


def test_positional_checks_board_size(rng):
    with pytest.raises(ValueError):
        PositionalPlayer(5)
    with pytest.raises(ValueError):
        PositionalPlayer(6).act([E.initial(4)], rng)


@pytest.mark.parametrize("n", E.SIZES)
def test_positional_weights_are_symmetric(n):
    """The rules are D4-invariant, so the weights must be too; this catches a mistyped entry."""
    w = POSITIONAL_WEIGHTS[n]
    assert w.shape == (n, n)
    for k in range(8):
        np.testing.assert_array_equal(E.transform_planes(w, k), w)


# --- network player -----------------------------------------------------------------------------


def test_q_values_is_one_forward_pass_over_stacked_planes(rng):
    states = random_positions(4, 20, rng)
    net = FixedQ(np.linspace(-1, 1, E.action_size(4)))
    q = NetPlayer(net).q_values(states)
    assert q.shape == (20, E.action_size(4)) and q.dtype == np.float32
    assert len(net.inputs) == 1
    np.testing.assert_array_equal(net.inputs[0].numpy(), np.stack([E.to_planes(s) for s in states]))
    NetPlayer(net).act(states, rng)
    assert len(net.inputs) == 2  # act also batches everything into one forward pass


@pytest.mark.parametrize("n", E.SIZES)
def test_greedy_net_plays_best_legal_action(n, rng):
    """Q rises with the action index and pass is highest, so the right answer is the highest legal
    index. Illegal squares and an unforced pass must be skipped even though their Q is larger."""
    states = random_positions(n, 200, rng)
    p = NetPlayer(FixedQ(np.linspace(-1, 1, E.action_size(n))), epsilon=0.0)
    expected = [int(np.flatnonzero(E.legal_mask(s)).max()) for s in states]
    assert p.act(states, rng) == expected


def test_net_breaks_exact_ties_at_random(rng):
    """A saturated tanh gives exact ties at ±1.0; argmax alone would always pick the lowest index."""
    q = np.ones(E.action_size(4))
    s = E.initial(4)
    counts = Counter(NetPlayer(FixedQ(q)).act([s] * 400, rng))
    assert set(counts) == set(np.flatnonzero(E.legal_mask(s)).tolist())


def test_net_epsilon_explores_uniformly_and_is_mutable(rng):
    s = E.initial(4)
    legal = np.flatnonzero(E.legal_mask(s)).tolist()
    p = NetPlayer(FixedQ(np.linspace(-1, 1, E.action_size(4))), epsilon=1.0)
    counts = Counter(p.act([s] * 400, rng))
    assert set(counts) == set(legal) and min(counts.values()) > 60
    p.epsilon = 0.0  # the training loop and the pool set it between rounds
    assert p.act([s] * 50, rng) == [max(legal)] * 50


def test_net_player_with_qnet(rng):
    states = random_positions(4, 50, rng)
    p = NetPlayer(QNet(4, layers=3).eval())
    assert p.q_values(states).shape == (50, E.action_size(4))
    assert all(E.legal_mask(s)[a] for s, a in zip(states, p.act(states, rng)))
