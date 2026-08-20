import math
import torch
from training.ppo import compute_gae, ppo_policy_loss, value_loss, entropy_bonus


def test_gae_matches_hand_computed_example():
    # T=2, terminal at t=1 (done=[False, True]); gamma=0.9, lam=0.95.
    # Hand-derived (see plan Task 9 design notes): advantages = [2.2325, 1.5],
    # returns = [2.7325, 2.0].
    rewards = torch.tensor([1.0, 2.0])
    values = torch.tensor([0.5, 0.5, 0.5])  # length T+1 (last is the bootstrap value)
    dones = torch.tensor([0.0, 1.0])
    adv, ret = compute_gae(rewards, values, dones, gamma=0.9, lam=0.95)
    assert torch.allclose(adv, torch.tensor([2.2325, 1.5]), atol=1e-4)
    assert torch.allclose(ret, torch.tensor([2.7325, 2.0]), atol=1e-4)


def test_ppo_policy_loss_clips_large_positive_ratio():
    # Hand-derived: 2-action categorical, old_log_prob=log(0.5), new logits=[2,0]
    # for the taken action (action 0) -> new_log_prob=log(0.8808...), ratio~1.7616,
    # clip_eps=0.2 -> clipped ratio=1.2. advantage=1.0 (positive) -> min(1.7616,
    # 1.2)=1.2 -> loss = -1.2.
    old_log_probs = torch.tensor([math.log(0.5)])
    logits = torch.tensor([[2.0, 0.0]])
    new_log_probs = torch.log_softmax(logits, dim=-1)[:, 0]
    advantages = torch.tensor([1.0])
    loss = ppo_policy_loss(new_log_probs, old_log_probs, advantages, clip_eps=0.2)
    assert torch.allclose(loss, torch.tensor(-1.2), atol=1e-3)


def test_value_loss_is_mse():
    values = torch.tensor([1.0, 2.0, 3.0])
    returns = torch.tensor([1.5, 2.0, 2.5])
    loss = value_loss(values, returns)
    expected = ((torch.tensor([0.5, 0.0, 0.5])) ** 2).mean()
    assert torch.allclose(loss, expected)


def test_entropy_bonus_is_nonnegative_and_zero_for_deterministic_logits():
    uniform_logits = torch.zeros(1, 4)
    deterministic_logits = torch.tensor([[100.0, -100.0, -100.0, -100.0]])
    assert entropy_bonus(uniform_logits).item() > 0.0
    assert entropy_bonus(deterministic_logits).item() < 1e-3
