import torch
import torch.nn as nn
from models.baseline_transformer import Block


class ActorCriticReadout(nn.Module):
    """Windowed causal-attention readout over the reservoir's recent spike-feature
    history, emitting an action distribution and a value estimate from the LAST
    position in the window -- deliberately NOT the same interface as
    AttentionReadout (spiking-reservoir-lm), which emits per-position next-byte
    logits for teacher-forced generation. RL needs "the action given the window
    ending now", not a prediction at every past position. This module reuses
    AttentionReadout's proven internal shape (in_proj + positional embedding +
    causal Blocks + final LayerNorm) and its `Block` dependency directly, adapted
    to that different interface -- see design doc §4 and plan Task 7.
    """

    def __init__(self, reservoir_size, n_actions, d_model=64, n_layers=2, n_heads=4, context_len=64):
        super().__init__()
        self.context_len = context_len
        self.in_proj = nn.Linear(reservoir_size, d_model)
        self.pos_emb = nn.Embedding(context_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, context_len) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.actor_head = nn.Linear(d_model, n_actions)
        self.critic_head = nn.Linear(d_model, 1)

    def forward(self, spike_window: torch.Tensor):
        B, T, _ = spike_window.shape
        assert T <= self.context_len, f"window length {T} exceeds context_len {self.context_len}"
        pos = torch.arange(T, device=spike_window.device)
        x = self.in_proj(spike_window) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        last = x[:, -1, :]  # decision is made from the most recent position only
        return self.actor_head(last), self.critic_head(last).squeeze(-1)
