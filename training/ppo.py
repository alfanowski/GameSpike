import torch
import torch.nn.functional as F


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                 gamma: float = 0.99, lam: float = 0.95):
    """rewards, dones: (T,). values: (T+1,) -- includes the bootstrap value after
    the last step (0.0 for a truly terminal end, or the critic's own V(s_T) for a
    truncated/non-terminal end). Returns (advantages, returns), each (T,)."""
    T = rewards.shape[0]
    advantages = torch.zeros(T, dtype=rewards.dtype)
    last_adv = 0.0
    for t in reversed(range(T)):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
        last_adv = delta + gamma * lam * not_done * last_adv
        advantages[t] = last_adv
    returns = advantages + values[:T]
    return advantages, returns


def ppo_policy_loss(new_log_probs: torch.Tensor, old_log_probs: torch.Tensor,
                     advantages: torch.Tensor, clip_eps: float = 0.2) -> torch.Tensor:
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -torch.min(unclipped, clipped).mean()


def value_loss(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(values, returns)


def entropy_bonus(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1).mean()
