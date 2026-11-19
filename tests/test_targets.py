"""Target computation for both modes (spec §4.5). Pure tensor tests, no network or engine needed."""

from __future__ import annotations

import pytest
import torch

from othello_rl.targets import masked_argmax, td_targets


def t(x, dtype=torch.float32):
    return torch.tensor(x, dtype=dtype)


ALL_LEGAL = t([[True, True, True]], torch.bool)


def test_masked_argmax_ignores_illegal_actions():
    q = t([[0.9, 0.1, 0.5], [0.2, 0.8, 0.3]])
    mask = t([[False, True, True], [True, False, True]], torch.bool)
    assert masked_argmax(q, mask).tolist() == [2, 2]


@pytest.mark.parametrize("mode", ["env", "negamax"])
def test_terminal_rows_use_reward_only(mode):
    r = t([1.0, -1.0, 0.0])
    done = t([True, True, True], torch.bool)
    q = t([[5.0, -5.0, 2.0]] * 3)
    no_legal = torch.zeros(3, 3, dtype=torch.bool)  # terminal s2 has no legal actions
    y = td_targets(r, done, q, q, no_legal, mode)
    assert torch.isfinite(y).all()
    assert torch.equal(y, r)


def test_sign_depends_on_mode():
    r, done = t([0.0]), t([False], torch.bool)
    q_online = t([[0.1, 0.9, 0.5]])  # online net picks action 1 ...
    q_target = t([[0.2, 0.6, 0.3]])  # ... and the target net values it at 0.6 (Double DQN)
    assert td_targets(r, done, q_online, q_target, ALL_LEGAL, "env").item() == pytest.approx(0.6)
    assert td_targets(r, done, q_online, q_target, ALL_LEGAL, "negamax").item() == pytest.approx(-0.6)


def test_bootstrap_respects_next_state_mask():
    r, done = t([0.0]), t([False], torch.bool)
    q_online = t([[0.9, 0.1, 0.5]])
    q_target = t([[1.0, 0.2, 0.4]])
    mask = t([[False, True, True]], torch.bool)  # action 0 is illegal in s2
    assert td_targets(r, done, q_online, q_target, mask, "env").item() == pytest.approx(0.4)


def test_mixed_batch():
    r = t([0.0, 1.0])
    done = t([False, True], torch.bool)
    q_online = t([[0.1, 0.9, 0.5], [0.0, 0.0, 0.0]])
    q_target = t([[0.2, 0.6, 0.3], [9.0, 9.0, 9.0]])
    mask = t([[True, True, True], [False, False, False]], torch.bool)
    assert td_targets(r, done, q_online, q_target, mask, "negamax").tolist() == pytest.approx([-0.6, 1.0])


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        td_targets(
            t([0.0]), t([False], torch.bool), t([[0.0, 0.0, 0.0]]), t([[0.0, 0.0, 0.0]]), ALL_LEGAL, "minimax"
        )
