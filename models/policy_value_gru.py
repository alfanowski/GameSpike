import torch
import torch.nn as nn


class PolicyValueGRU(nn.Module):
    """Mandatory-control baseline (spec §5): a fully-trained recurrent feature
    extractor at a parameter budget matched to PolicyValueReservoir, so any
    difference in results is attributable to the frozen reservoir specifically,
    not to "having a recurrent memory" in general."""

    def __init__(self, obs_dim=12, embed_dim=32, hidden_dim=192, n_actions=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Linear(obs_dim, embed_dim)
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
