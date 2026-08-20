import torch
import torch.nn as nn

from models.embedding_init import EMBED_INIT_MODES, init_embedding_bias_


class PolicyValueGRU(nn.Module):
    """Mandatory-control baseline (spec §5): a fully-trained recurrent feature
    extractor at a parameter budget matched to PolicyValueReservoir, so any
    difference in results is attributable to the frozen reservoir specifically,
    not to "having a recurrent memory" in general."""

    def __init__(self, obs_dim=12, embed_dim=32, hidden_dim=192, n_actions=10,
                 embed_init_mode="legacy", embed_scale=1.0, obs_mean=None):
        super().__init__()
        if embed_init_mode not in EMBED_INIT_MODES:
            raise ValueError(
                f"unknown embed_init_mode: {embed_init_mode!r}; expected one of "
                f"{EMBED_INIT_MODES}"
            )
        self.hidden_dim = hidden_dim
        self.embed_init_mode = embed_init_mode
        self.embed_scale = float(embed_scale)
        self.embedding = nn.Linear(obs_dim, embed_dim)
        # The SAME treatment the reservoir arm gets, deliberately -- input centring is
        # a generic init correction and a control only one arm receives is not a
        # control. The full argument (including why this arm is expected to benefit
        # LESS, since its embedding feeds a trainable GRU that can learn to absorb a
        # DC offset while the reservoir's feeds a frozen nonlinearity that cannot)
        # lives in models/embedding_init.py, written once so the two arms cannot drift.
        #
        # This arm's LEGACY weight init is nn.Linear's own default -- unlike the
        # reservoir arm it was never overridden -- so `embed_scale` rescales that draw
        # in place instead of re-drawing from a scaled normal. At embed_scale=1.0 the
        # branch is not taken at all, so the historical init is bit-identical and the
        # global RNG stream is untouched.
        if self.embed_scale != 1.0:
            with torch.no_grad():
                self.embedding.weight.mul_(self.embed_scale)
        # legacy bias for THIS arm is nn.Linear's default uniform draw, i.e. leave it
        # exactly as constructed (legacy_bias_init=None) -- NOT zeros, which is what
        # the reservoir arm happens to use.
        init_embedding_bias_(self.embedding, embed_init_mode, obs_mean,
                             legacy_bias_init=None)
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.actor_head = nn.Linear(hidden_dim, n_actions)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)

    def forward(self, obs: torch.Tensor, h: torch.Tensor):
        emb = torch.tanh(self.embedding(obs)).unsqueeze(1)  # (B, 1, embed_dim)
        out, h_next = self.gru(emb, h)                       # out: (B, 1, hidden_dim)
        pooled = out[:, -1, :]                                # (B, hidden_dim)
        logits = self.actor_head(pooled)
        value = self.critic_head(pooled).squeeze(-1)
        return logits, value, h_next

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
