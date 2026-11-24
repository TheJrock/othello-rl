"""Target computation for both modes (spec §4.5). Pure tensor tests, no network or engine needed."""

from __future__ import annotations

import pytest
import torch

from othello_rl.targets import masked_argmax, td_loss, td_targets


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


# --- the loss (small real networks; still no engine) --------------------------------------------


@pytest.mark.parametrize("mode", ["env", "negamax"])
def test_td_loss_is_a_semi_gradient_huber_loss(mode):
    import copy

    from othello_rl.nets import QNet
    from othello_rl.replay import Batch

    torch.manual_seed(0)
    online = QNet(4, channels=8, layers=2)
    target_net = copy.deepcopy(online)
    with torch.no_grad():  # make the target net differ, so Double DQN's two roles are distinguishable
        for p in target_net.parameters():
            p.add_(0.1 * torch.randn_like(p))
    b = 6
    batch = Batch(
        s=torch.rand(b, 3, 4, 4),
        a=torch.randint(0, 17, (b,)),
        r=t([0.0, 0.0, 0.0, 1.0, -1.0, 0.0]),
        s2=torch.rand(b, 3, 4, 4),
        mask2=torch.rand(b, 17) < 0.5,
        done=t([False, False, False, True, True, False], torch.bool),
    )
    batch.mask2[batch.done] = False  # terminal s2 has no legal actions
    batch.mask2[~batch.done, 0] = True  # every live s2 has at least one

    loss, info = td_loss(online, target_net, batch, mode)
    loss.backward()

    with torch.no_grad():
        y = td_targets(batch.r, batch.done, online(batch.s2), target_net(batch.s2), batch.mask2, mode)
    q_sa = online(batch.s)[torch.arange(b), batch.a]
    assert loss.item() == pytest.approx(torch.nn.functional.smooth_l1_loss(q_sa, y).item())
    assert set(info) == {"loss", "mean_q", "mean_target"}
    assert info["mean_target"] == pytest.approx(y.mean().item())
    assert all(p.grad is None for p in target_net.parameters())  # the target net is never trained

    # Gradients flow only through Q(s, a): identical to differentiating against a constant target.
    got = [p.grad.clone() for p in online.parameters()]
    online.zero_grad()
    torch.nn.functional.smooth_l1_loss(online(batch.s)[torch.arange(b), batch.a], y).backward()
    for g, p in zip(got, online.parameters()):
        torch.testing.assert_close(g, p.grad)
