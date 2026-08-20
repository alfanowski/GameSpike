import math
import torch
import torch.nn as nn
from models.spiking_reservoir import SpikingReservoir
from models.actor_critic_readout import ActorCriticReadout
from models.embedding_init import EMBED_INIT_MODES, init_embedding_bias_

# Reservoir buffers that are transient STATE rather than frozen WEIGHTS, and so are
# excluded from the bit-identity snapshot below. snntorch's Leaky keeps its membrane
# slot in a buffer named `lif.mem`: it is mutated by any forward pass (including the
# pure-inference ones during rollout collection) and even changes shape, from `(0,)`
# at construction to `(B, N)` after the first call. This codebase never reads it --
# `mem` is threaded explicitly through `SpikingReservoir.step` -- so including it
# would make the tripwire fire on every normal forward pass.
TRANSIENT_RESERVOIR_BUFFERS = frozenset({"lif.mem"})


class PolicyValueReservoir(nn.Module):
    """embedding (trainable) -> FROZEN spiking reservoir (stepped incrementally,
    one env step at a time) -> windowed attention actor/critic readout. Mirrors
    SpikingBackpropLM's wiring (spiking-reservoir-lm/models/spiking_backprop_lm.py)
    but stateful/incremental instead of whole-sequence, and continuous-observation
    instead of byte-embedding, per design doc §4."""

    def __init__(self, obs_dim=12, embed_dim=32, reservoir_size=8192, n_actions=10,
                 use_tensor_train=True, tt_rank=8, tt_n_cores=4, context_len=64, seed=0,
                 d_model=16, n_layers=2, n_heads=4,
                 embed_init_mode="legacy", embed_scale=1.0, obs_mean=None):
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
        if embed_init_mode not in EMBED_INIT_MODES:
            raise ValueError(
                f"unknown embed_init_mode: {embed_init_mode!r}; expected one of "
                f"{EMBED_INIT_MODES}"
            )
        self.reservoir_size = reservoir_size
        self.context_len = context_len
        self.embed_init_mode = embed_init_mode
        self.embed_scale = float(embed_scale)
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
        # KNOWN GAP -- NOW MEASURED, and the measurement did not agree. That 0.3163
        # figure was taken against synthetic N(0,1) observations. Against 6,000 REAL
        # rollout steps the induced input-current std is 0.128683, not ~0.3163, and
        # the two calibration targets this comment states -- "input-current std ~0.3"
        # AND "spike rate ~2%" -- turn out to be MUTUALLY INCOMPATIBLE under real
        # observations. The spike-rate target is the one that corresponds to healthy
        # dynamics, so that is the one to calibrate against; the 0.3 figure is kept
        # above only as the historical record of how this scalar was chosen.
        #
        # The real problem is not the scalar at all, it is the DC component:
        #   * real observations are dominated by their own mean -- 77.70% of the
        #     observation energy is DC (||E[obs]||^2 = 1.331336 of E||obs||^2 =
        #     1.713384), so 76.11% of the reservoir's input-current variance is DC;
        #   * the LIF neuron integrates DC with gain 1/(1-beta) = 10.0 but AC with
        #     gain 1/sqrt(1-beta^2) = 2.2942, a 4.3589x amplification favouring DC;
        #   * so every unit acquires a FROZEN membrane offset, std 0.943583 across
        #     units, range [-3.5080, +3.4847], against a threshold of 1.0. Measured:
        #     14.93% of units sit permanently below -threshold (silent forever) and
        #     14.50% permanently above (saturated).
        #   * `embed_scale` alone CANNOT fix this -- it multiplies DC and AC together,
        #     so a scale sweep floors at ~20% silent even at 32x the default.
        # `embed_init_mode="centered"` removes the DC term instead; see
        # models/embedding_init.py for the algebra and for why BOTH arms get it.
        #
        # STRICTLY ADDITIVE AND OFF BY DEFAULT: embed_init_mode="legacy" with
        # embed_scale=1.0 is BIT-IDENTICAL to the two lines that were here before
        # (same RNG draws in the same order, same std expression, same zeroed bias),
        # because 200 existing checkpoints have to stay loadable and reproducible.
        nn.init.normal_(self.embedding.weight,
                        std=embed_scale / math.sqrt(obs_dim * embed_dim))
        # legacy bias for THIS arm is an explicit zero (it always was); the centered
        # branch is shared with the baseline arm verbatim.
        init_embedding_bias_(self.embedding, embed_init_mode, obs_mean,
                             legacy_bias_init=nn.init.zeros_)
        self.reservoir = SpikingReservoir(
            reservoir_size=reservoir_size, input_dim=embed_dim, seed=seed,
            use_tensor_train=use_tensor_train, tt_rank=tt_rank, tt_n_cores=tt_n_cores,
        )
        self.readout = ActorCriticReadout(
            reservoir_size=reservoir_size, n_actions=n_actions, d_model=d_model,
            n_layers=n_layers, n_heads=n_heads, context_len=context_len,
        )
        # Reference copy taken at construction: this is literally "the weights as
        # initialized" that spec §3's runtime tripwire compares against.
        self.snapshot_frozen_weights()

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

    def snapshot_frozen_weights(self):
        """(Re)take the reference copy `assert_reservoir_frozen` compares against.

        Called once at construction, and again by `training.train.load_checkpoint`
        after a resume: a resumed run's frozen weights are the ones that came off
        disk, not the ones this process happened to construct before overwriting
        them, so the reference point has to move with them or every subsequent
        checkpoint would trip the wire spuriously.
        """
        self._frozen_snapshot = {
            name: buf.detach().clone()
            for name, buf in self.reservoir.named_buffers()
            if name not in TRANSIENT_RESERVOIR_BUFFERS
        }

    def assert_reservoir_frozen(self):
        """Runtime tripwire for the frozen-reservoir invariant (spec §3).

        Two independent halves, because either one alone can be satisfied while the
        invariant is broken:
          * zero `nn.Parameter`s under the reservoir -- no optimizer can even be
            handed them;
          * every frozen buffer still BIT-IDENTICAL to its initialization -- catches
            an in-place write that never went through an optimizer at all
            (`W_in.mul_(...)`, a stray `load_state_dict`, a fine-tuning experiment
            someone forgot to revert).
        Called in production by `training.train.save_checkpoint` before every write
        to disk, so a corrupted reservoir can never be silently persisted and later
        evaluated as if it were frozen.
        """
        assert list(self.reservoir.parameters()) == [], (
            "reservoir must have zero nn.Parameters -- frozen-reservoir invariant violated"
        )
        live = dict(self.reservoir.named_buffers())
        for name, expected in self._frozen_snapshot.items():
            actual = live.get(name)
            assert actual is not None, (
                f"frozen reservoir buffer {name} disappeared -- frozen-reservoir "
                "invariant violated"
            )
            assert torch.equal(expected, actual), (
                f"frozen reservoir buffer {name} is no longer bit-identical to its "
                "initialization -- the reservoir was trained/mutated, so this run is "
                "no longer the frozen-reservoir experiment spec §3 describes"
            )
