"""Shared observation-embedding initialisation for BOTH experimental arms.

This module exists so the two arms cannot drift apart. `PolicyValueReservoir` and
`PolicyValueGRU` both start with an `nn.Linear(obs_dim, embed_dim)` over the same
12-dim observation, and the initialisation of that one layer is now a scientific
knob (`--embed-init-mode`, `--embed-scale`). Writing the knob out twice would let
one arm's copy be "fixed" and the other's forgotten, which is exactly the failure
the §5 control is supposed to make impossible -- so the rule is written down once,
here, and both arms call it.

WHY CENTRING IS APPLIED TO BOTH ARMS (this is a control requirement, not a
courtesy). Input centring is a GENERIC initialisation correction: it removes the
observation's DC component from the embedding's output. It is not a
reservoir-specific advantage and must not be handed to one arm only.

  We fully expect it to help the RESERVOIR arm more, and the asymmetry has a
  mechanism: on the baseline arm the embedding feeds a TRAINABLE GRU, which can
  learn to absorb a constant DC offset in its own input weights and biases within
  a few updates. On the reservoir arm the embedding feeds a FROZEN nonlinearity --
  a fixed W_in into fixed LIF neurons -- which cannot adapt at all: a DC offset
  there becomes a permanent per-unit membrane bias that silences or saturates
  units for the entire run.

  That expected asymmetry is a RESULT, not a licence to apply the treatment
  asymmetrically. Giving centring to the reservoir arm alone would mean the two
  arms differ in initialisation as well as architecture, and the measured gap
  would no longer be attributable to the architecture -- the comparison would be
  confounded in exactly the way `apply_grad_clipping` already documents for
  gradient clipping ("a control that only one arm receives is not a control").

MEASURED EFFECT on the reservoir arm (mean over 8 seeds, real observations):
silent-unit fraction 44.7403% -> 1.7532%, mean spike rate 0.024351 -> 0.020912
(into the ~2% band `spiking_reservoir.py` documents as healthy), and zero
saturated units. Scaling the embedding alone cannot do this: scale multiplies DC
and AC together, so an embedding-scale sweep floors at ~20% silent even at 32x.
"""
import torch
import torch.nn as nn

# "legacy"   -- the initialisation every existing checkpoint was produced under.
#               THE DEFAULT, and it stays the default: 20 completed runs and 200
#               checkpoints on disk must remain bit-reproducible.
# "centered" -- the embedding bias is set so the layer's output is the embedding
#               of the CENTRED observation. See `centred_bias` for why this costs
#               zero new parameters.
EMBED_INIT_MODES = ("legacy", "centered")


def centred_bias(weight: torch.Tensor, obs_mean) -> torch.Tensor:
    """The bias that makes `Linear(obs)` compute `W @ (obs - mu)`.

    The embedding is LINEAR, so
        W @ (obs - mu) == W @ obs + (-W @ mu)
    exactly. Centring the input is therefore ALGEBRAICALLY IDENTICAL to a
    particular BIAS INITIALISATION -- verified numerically equivalent to a max
    absolute difference of 5.96e-08 (float32 rounding). That identity is the whole
    reason this fix is cheap and safe:

      * it costs ZERO new parameters (`nn.Linear` already has a bias);
      * it changes no shapes, so every existing `state_dict` still loads;
      * the bias stays TRAINABLE, so the run can move away from this point if the
        data says so -- this is an initialisation, not a hard constraint.
    """
    mu = torch.as_tensor(obs_mean, dtype=weight.dtype, device=weight.device)
    if mu.shape != (weight.shape[1],):
        raise ValueError(
            f"obs_mean has shape {tuple(mu.shape)} but the embedding expects one "
            f"entry per input dimension, i.e. shape ({weight.shape[1]},)"
        )
    return -(weight @ mu)


def init_embedding_bias_(embedding: nn.Linear, embed_init_mode: str, obs_mean,
                         legacy_bias_init=None) -> None:
    """Initialise `embedding.bias` in place according to `embed_init_mode`.

    `legacy_bias_init` is the arm's OWN historical bias initialisation, passed in
    as a callable rather than assumed, because the two arms genuinely differ:
    `PolicyValueReservoir` explicitly zeroes its bias while `PolicyValueGRU` keeps
    `nn.Linear`'s default uniform draw. Reproducing "whatever this arm did before"
    is the point of `legacy`, so each arm states its own. None means "leave the
    bias exactly as `nn.Linear.__init__` left it", which also consumes no RNG.

    The `centered` branch is IDENTICAL for both arms -- that is the control.
    """
    if embed_init_mode not in EMBED_INIT_MODES:
        raise ValueError(
            f"unknown embed_init_mode: {embed_init_mode!r}; expected one of {EMBED_INIT_MODES}"
        )
    if embed_init_mode == "legacy":
        if legacy_bias_init is not None:
            legacy_bias_init(embedding.bias)
        return
    if obs_mean is None:
        raise ValueError(
            "embed_init_mode='centered' needs obs_mean (the per-dimension mean of the "
            "real observation). Pass envs.mario_land_env.OBS_MEAN, or re-measure it if "
            "the observation construction has changed."
        )
    with torch.no_grad():
        embedding.bias.copy_(centred_bias(embedding.weight, obs_mean))
