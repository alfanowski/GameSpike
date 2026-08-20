from collections import deque
import torch


class NoveltyGate:
    """Trajectory-novelty write-gate (design doc §4, adapted from the EMG vertical's
    abnormal-activation detector): a k-nearest-neighbor novelty score over a
    sliding-window buffer of recent state-summary vectors. No trained parameters --
    the curiosity signal is a byproduct of the buffer, not a learned model, per the
    design doc's "zero extra trained-parameter cost" claim."""

    def __init__(self, dim: int, capacity: int = 512, k: int = 8):
        self.dim = dim
        self.capacity = capacity
        self.k = k
        self.buffer = deque(maxlen=capacity)

    def score(self, state_vec: torch.Tensor) -> float:
        if len(self.buffer) == 0:
            return 1.0  # defined maximal novelty when nothing has been seen yet
        stacked = torch.stack(list(self.buffer))          # (n, dim)
        dists = torch.linalg.norm(stacked - state_vec.unsqueeze(0), dim=1)
        k = min(self.k, dists.shape[0])
        topk = torch.topk(dists, k, largest=False).values
        return topk.mean().item()

    def push(self, state_vec: torch.Tensor):
        self.buffer.append(state_vec.detach().clone())
