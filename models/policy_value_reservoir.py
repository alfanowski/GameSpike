import math
import torch
import torch.nn as nn
from models.spiking_reservoir import SpikingReservoir
from models.actor_critic_readout import ActorCriticReadout


class PolicyValueReservoir(nn.Module):
    """embedding (trainable) -> FROZEN spiking reservoir (stepped incrementally,
    one env step at a time) -> windowed attention actor/critic readout. Mirrors
    SpikingBackpropLM's wiring (spiking-reservoir-lm/models/spiking_backprop_lm.py)
    but stateful/incremental instead of whole-sequence, and continuous-observation
    instead of byte-embedding, per design doc §4."""

    def __init__(self, obs_dim=12, embed_dim=32, reservoir_size=8192, n_actions=10,
                 use_tensor_train=True, tt_rank=8, tt_n_cores=4, context_len=64, seed=0,
                 d_model=16, n_layers=2, n_heads=4):
        # d_model=16 (not ActorCriticReadout's own 64 default) is REQUIRED by the
        # matched-parameter-budget rule (spec §5): this arm's trainable count must
        # land within 10% of the GRU baseline's. The readout's in_proj maps the full
        # reservoir_size (8192) into d_model, so it alone costs ~8.2k params per unit
        # of d_model and dominates the budget -- d_model=64 overshoots the baseline by
        # 4.7x (629,163 vs 132,715). d_model is therefore the coarse knob (only 16
        # lands in band; 12 -> 0.78x, 20 -> 1.32x) and n_layers the fine one. At
        # d_model=16/n_layers=2 this arm is 139,179 params, ratio 1.049. Enforced by
        # tests/test_parameter_parity.py -- retune these, never the tolerance.
        super().__init__()
        self.reservoir_size = reservoir_size
        self.context_len = context_len
        self.embedding = nn.Linear(obs_dim, embed_dim)
        # Same input-current calibration rationale as spiking_backprop_lm.py: scale
        # the embedding's init so the induced reservoir input current lands in the
        # ~0.3-std band W_in was tuned for, instead of assuming it transfers from a
        # discrete byte-embedding to a continuous observation vector unchanged.
        #
        # The fan-in term is obs_dim*embed_dim, NOT embed_dim: spiking_backprop_lm.py
        # uses nn.Embedding (a row lookup, effective fan-in 1, so 1/sqrt(embed_dim) is
        # right there), but this arm takes a CONTINUOUS observation through
        # nn.Linear(obs_dim, embed_dim), whose fan-in is obs_dim. Carrying the
        # nn.Embedding formula over unchanged overshoots by sqrt(obs_dim) (~3.6x
        # measured), which is why the extra obs_dim factor is here.
        #
        # MEASURED (synthetic obs ~ N(0,1), obs_dim=12, embed_dim=32, 4096 samples):
        # induced input-current std = 0.3163 (1.05x the 0.3 target; it was 1.0918,
        # i.e. 3.65x, under the old 1/sqrt(embed_dim) init), giving a mean spike rate
        # of 2.4% -- inside the ~2% band spiking_reservoir.py documents as healthy.
        #
        # KNOWN GAP: that measurement assumes standard-normal observations. The real
        # obs vector's scale comes from the env's normalization (Task 5), so a residual
        # calibration gap may remain until real observation statistics are available;
        # this init is correct STRUCTURALLY (right fan-in) but its scalar has only been
        # validated against synthetic input. Re-measure against real obs before
        # trusting the spike rate in the actual experiment.
        nn.init.normal_(self.embedding.weight, std=1.0 / math.sqrt(obs_dim * embed_dim))
        nn.init.zeros_(self.embedding.bias)
        self.reservoir = SpikingReservoir(
            reservoir_size=reservoir_size, input_dim=embed_dim, seed=seed,
            use_tensor_train=use_tensor_train, tt_rank=tt_rank, tt_n_cores=tt_n_cores,
        )
        self.readout = ActorCriticReadout(
            reservoir_size=reservoir_size, n_actions=n_actions, d_model=d_model,
            n_layers=n_layers, n_heads=n_heads, context_len=context_len,
        )

    def init_state(self, batch_size: int, device: torch.device):
        mem = torch.zeros(batch_size, self.reservoir_size, device=device)
        spk = torch.zeros(batch_size, self.reservoir_size, device=device)
        window = torch.zeros(batch_size, 0, self.reservoir_size, device=device)
        return mem, spk, window

    def forward(self, obs: torch.Tensor, mem, spk, window):
        emb = self.embedding(obs)                       # (B, embed_dim), trainable
        spk_next, mem_next = self.reservoir.step(emb, mem, spk)  # frozen, surrogate grad to emb
        feat = self.reservoir.readout_feature(spk_next, mem_next).unsqueeze(1)  # (B, 1, N)
        window_next = torch.cat([window, feat], dim=1)
        if window_next.shape[1] > self.context_len:
            window_next = window_next[:, -self.context_len:, :]
        action_logits, value = self.readout(window_next)
        return action_logits, value, mem_next, spk_next, window_next

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def assert_reservoir_frozen(self):
        assert list(self.reservoir.parameters()) == [], (
            "reservoir must have zero nn.Parameters -- frozen-reservoir invariant violated"
        )
