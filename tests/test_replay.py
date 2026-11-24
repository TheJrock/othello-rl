"""Replay buffer: ring semantics and symmetry augmentation (spec §4.5)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from othello_rl import engine as E
from othello_rl.replay import ReplayBuffer, Transition


def state_from_planes(planes: np.ndarray, n: int) -> E.State:
    bits = 1 << np.arange(n * n, dtype=object)
    return E.State(int(bits[planes[0].ravel() > 0].sum()), int(bits[planes[1].ravel() > 0].sum()), n)


def game_transitions(n: int, count: int, rng: np.random.Generator) -> list[Transition]:
    """One transition per ply of random games, with s2 = the position right after the move."""
    out: list[Transition] = []
    while len(out) < count:
        s = E.initial(n)
        while not E.is_terminal(s) and len(out) < count:
            a = int(rng.choice(np.flatnonzero(E.legal_mask(s))))
            s2, r, done = E.step(s, a)
            out.append(Transition(E.to_planes(s), a, r, E.to_planes(s2), E.legal_mask(s2), done))
            s = s2
    return out


def numbered(count: int, n: int = 4) -> list[Transition]:
    """Dummy transitions whose action field is their index, to track what the ring keeps."""
    z = np.zeros((3, n, n), dtype=np.float32)
    return [
        Transition(z, i % (n * n + 1), float(i), z, np.zeros(n * n + 1, bool), False) for i in range(count)
    ]


def test_ring_keeps_the_newest(rng):
    buf = ReplayBuffer(capacity=10, n=4)
    assert len(buf) == 0
    buf.extend(numbered(7))
    assert len(buf) == 7
    buf.extend(numbered(20)[7:12])  # 12 written in total: the two oldest are overwritten
    assert len(buf) == 10
    assert sorted(buf.r.tolist()) == list(map(float, range(2, 12)))
    buf.extend(numbered(100)[50:75])  # more than capacity in one call: only the last 10 survive
    assert sorted(buf.r.tolist()) == list(map(float, range(65, 75)))
    buf.extend([])
    assert len(buf) == 10


def test_sample_without_augmentation_returns_stored_values(rng):
    trs = game_transitions(6, 50, rng)
    buf = ReplayBuffer(capacity=100, n=6, seed=1)
    buf.extend(trs)
    assert buf.s.dtype == np.uint8 and buf.s2.dtype == np.uint8  # planes are 0/1: store a byte each
    b = buf.sample(200, augment=False)
    assert b.s.shape == (200, 3, 6, 6) and b.s.dtype == torch.float32
    assert b.a.dtype == torch.int64 and b.mask2.dtype == torch.bool and b.done.dtype == torch.bool
    by_planes = {(tr.s.tobytes(), tr.a): tr for tr in trs}
    for i in range(200):
        tr = by_planes[(b.s[i].numpy().tobytes(), int(b.a[i]))]
        np.testing.assert_array_equal(b.s2[i].numpy(), tr.s2)
        np.testing.assert_array_equal(b.mask2[i].numpy(), tr.mask2)
        assert b.r[i].item() == tr.r and b.done[i].item() == tr.done


@pytest.mark.parametrize("n", E.SIZES)
def test_augmentation_keeps_s_a_s2_and_mask_consistent(n, rng):
    """After a random symmetry, playing the transformed action in the transformed position must give
    exactly the transformed next position, and the transformed mask must be its legal mask."""
    trs = game_transitions(n, 300, rng)
    buf = ReplayBuffer(capacity=1000, n=n, seed=2)
    buf.extend(trs)
    b = buf.sample(2000, augment=True)
    originals = {tr.s.tobytes() for tr in trs}
    moved = 0
    for i in range(len(b.a)):
        s = state_from_planes(b.s[i].numpy(), n)
        a = int(b.a[i])
        np.testing.assert_array_equal(b.s[i].numpy(), E.to_planes(s))  # legal plane moved with the discs
        s2, r, done = E.step(s, a)  # raises if the action no longer fits the position
        np.testing.assert_array_equal(b.s2[i].numpy()[:2], E.to_planes(s2)[:2])
        np.testing.assert_array_equal(b.mask2[i].numpy(), E.legal_mask(s2))
        assert b.r[i].item() == r and b.done[i].item() == done
        moved += b.s[i].numpy().tobytes() not in originals
    assert moved > 1000  # 7 of the 8 symmetries move the board (a few positions are symmetric)


def test_every_symmetry_is_used():
    """Track one asymmetric position through 400 augmented samples: all 8 images must appear."""
    s = E.from_string("XO.. .XO. .OO. ....")
    images = {E.transform_state(s, k) for k in range(8)}
    assert len(images) == 8
    tr = Transition(E.to_planes(s), 0, 0.0, E.to_planes(s), E.legal_mask(s), False)
    buf = ReplayBuffer(capacity=1, n=4, seed=3)
    buf.extend([tr])
    b = buf.sample(400, augment=True)
    seen = {state_from_planes(x.numpy(), 4) for x in b.s}
    assert seen == images


def test_pass_stays_pass(rng):
    n = 4
    z = E.to_planes(E.from_string("OX" + "." * 14))
    tr = Transition(z, E.pass_action(n), 0.0, z, np.zeros(n * n + 1, bool), False)
    buf = ReplayBuffer(capacity=4, n=n)
    buf.extend([tr])
    assert (buf.sample(64, augment=True).a == E.pass_action(n)).all()


def test_sampling_an_empty_buffer_raises():
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=4, n=4).sample(8)
