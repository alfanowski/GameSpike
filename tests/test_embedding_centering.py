"""The embedding-init knob: `legacy` must be bit-identical, `centered` must work.

Four things are being defended here, in descending order of how expensive they
would be to get wrong:

  1. BACKWARD COMPATIBILITY. 20 completed training runs and 200 checkpoints exist
     on disk and a results write-up depends on them. `embed_init_mode="legacy"`
     with `embed_scale=1.0` -- the defaults -- must reproduce the historical
     initialisation BIT-for-bit on both arms, and old checkpoints (which carry
     neither new key) must still load.
  2. THE FIX ITSELF. Centring must actually remove the DC-driven silent units,
     measured on real observations, not merely be "a plausible-looking change".
  3. THE ALGEBRA. `centered` is exactly "embed the centred observation", which is
     what makes it free (no new parameters) and reversible (the bias is trainable).
  4. THE INVARIANTS the rest of the experiment rests on: matched parameter budget
     (spec §5) and the frozen reservoir (spec §3) hold under BOTH modes.

WHICH OBSERVATION DATA THE REGRESSION TEST USES, and why it matters. It uses
STORED REAL OBSERVATIONS: `tests/data/real_obs_6000.npy`, 6,000 steps collected on
the real ROM (3,000 under the trained reservoir-arm policy
`checkpoints/reservoir_seed0/step_1000064.pt`, 3,000 under a uniform-random
policy, pooled) -- the same protocol as the diagnostic that motivated this change.

An i.i.d. Gaussian surrogate matching the measured per-dimension means and stds
was tried FIRST and does NOT reproduce the effect. Same seed, same 6,000-step
window, `centered` + scale 3.0: 24.1211% silent on the surrogate against 1.6602%
on the real data -- and the surrogate's spike rate collapses to 0.002064 against
0.017001. It is not merely noisier, it points the wrong way. The reason is that
real observations are strongly
TEMPORALLY CORRELATED (the level timer drifts monotonically, lives and powerup
state are constant for long stretches, the on-ground flag comes in runs), and a
LIF membrane with beta=0.9 integrates that low-frequency energy with a gain of up
to 1/(1-beta)=10. Drawing each step independently destroys exactly the structure
that drives the neurons. So the fixture is REQUIRED, not a convenience, and a
synthetic fallback would have made this file pass while testing nothing.

Regenerating the fixture (only needed if the observation construction changes --
in which case `envs.mario_land_env.OBS_MEAN` has to be re-measured too): roll out
the env for 3,000 steps under each policy, `np.stack` the observations in
collection order and save as float32.
"""
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from envs.mario_land_env import OBS_DIM, OBS_MEAN
from models.actor_critic_readout import ActorCriticReadout
from models.embedding_init import EMBED_INIT_MODES, centred_bias
from models.policy_value_gru import PolicyValueGRU
from models.policy_value_reservoir import PolicyValueReservoir
from models.spiking_reservoir import SpikingReservoir
from training.train import build_model, load_checkpoint, save_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_OBS_PATH = Path(__file__).resolve().parent / "data" / "real_obs_6000.npy"

# The production geometry, spelled out once. These mirror `training.train.build_model`
# so the identity tests below are about the arms the experiment actually runs.
RES_KWARGS = dict(obs_dim=OBS_DIM, embed_dim=32, reservoir_size=8192, n_actions=10,
                  use_tensor_train=True, tt_rank=8, tt_n_cores=4, context_len=64, seed=0)
GRU_KWARGS = dict(obs_dim=OBS_DIM, embed_dim=32, hidden_dim=192, n_actions=10)

# spec §5's tolerance, mirrored from tests/test_parameter_parity.py.
PARITY_TOLERANCE = 0.10


def _assert_state_dicts_bit_identical(a, b, what):
    assert a.keys() == b.keys(), f"{what}: state_dict keys differ"
    for key in a:
        assert torch.equal(a[key], b[key]), (
            f"{what}: tensor {key!r} is not bit-identical -- the default path changed, "
            "which would make the 200 existing checkpoints irreproducible"
        )


# --------------------------------------------------------------------------- #
# 1. legacy + scale 1.0 == the historical initialisation, bit for bit.
# --------------------------------------------------------------------------- #

def test_reservoir_legacy_reproduces_the_historical_init_bit_for_bit():
    """Reconstruct the PRE-CHANGE code path by hand and diff against the new one.

    Comparing against the new defaults alone would only prove the defaults are
    self-consistent. This replays the historical construction sequence explicitly
    -- same submodules, same order, therefore the same draws off the global RNG --
    so a reordering, an extra draw, or a changed std shows up as a mismatch.
    """
    seed = 4242
    torch.manual_seed(seed)
    historical_embedding = nn.Linear(RES_KWARGS["obs_dim"], RES_KWARGS["embed_dim"])
    nn.init.normal_(historical_embedding.weight,
                    std=1.0 / math.sqrt(RES_KWARGS["obs_dim"] * RES_KWARGS["embed_dim"]))
    nn.init.zeros_(historical_embedding.bias)
    historical_reservoir = SpikingReservoir(
        reservoir_size=RES_KWARGS["reservoir_size"], input_dim=RES_KWARGS["embed_dim"],
        seed=RES_KWARGS["seed"], use_tensor_train=True, tt_rank=RES_KWARGS["tt_rank"],
        tt_n_cores=RES_KWARGS["tt_n_cores"])
    historical_readout = ActorCriticReadout(
        reservoir_size=RES_KWARGS["reservoir_size"], n_actions=RES_KWARGS["n_actions"],
        d_model=16, n_layers=2, n_heads=4, context_len=RES_KWARGS["context_len"])

    torch.manual_seed(seed)
    model = PolicyValueReservoir(**RES_KWARGS, embed_init_mode="legacy", embed_scale=1.0)

    _assert_state_dicts_bit_identical(historical_embedding.state_dict(),
                                      model.embedding.state_dict(), "reservoir embedding")
    _assert_state_dicts_bit_identical(historical_reservoir.state_dict(),
                                      model.reservoir.state_dict(), "frozen reservoir")
    _assert_state_dicts_bit_identical(historical_readout.state_dict(),
                                      model.readout.state_dict(), "readout")


def test_gru_legacy_reproduces_the_historical_init_bit_for_bit():
    """Same replay for the baseline arm. Note its legacy bias is nn.Linear's DEFAULT
    uniform draw, not zeros -- the two arms genuinely differed and `legacy` has to
    reproduce each arm's own history, not a tidied-up common one."""
    seed = 4242
    torch.manual_seed(seed)
    historical_embedding = nn.Linear(GRU_KWARGS["obs_dim"], GRU_KWARGS["embed_dim"])
    historical_gru = nn.GRU(input_size=GRU_KWARGS["embed_dim"],
                            hidden_size=GRU_KWARGS["hidden_dim"], batch_first=True)
    historical_actor = nn.Linear(GRU_KWARGS["hidden_dim"], GRU_KWARGS["n_actions"])
    historical_critic = nn.Linear(GRU_KWARGS["hidden_dim"], 1)

    torch.manual_seed(seed)
    model = PolicyValueGRU(**GRU_KWARGS, embed_init_mode="legacy", embed_scale=1.0)

    _assert_state_dicts_bit_identical(historical_embedding.state_dict(),
                                      model.embedding.state_dict(), "gru embedding")
    _assert_state_dicts_bit_identical(historical_gru.state_dict(), model.gru.state_dict(), "gru")
    _assert_state_dicts_bit_identical(historical_actor.state_dict(),
                                      model.actor_head.state_dict(), "actor head")
    _assert_state_dicts_bit_identical(historical_critic.state_dict(),
                                      model.critic_head.state_dict(), "critic head")
    # The bias is the default uniform draw, NOT zeros. Stated as an assertion so a
    # future "cleanup" that zeroes it fails here instead of silently changing the
    # baseline arm's history.
    assert not torch.all(model.embedding.bias == 0)


@pytest.mark.parametrize("arm", ["baseline", "reservoir"])
def test_passing_the_new_kwargs_explicitly_does_not_perturb_the_rng_stream(arm):
    """`build_model(arm)` and `build_model(arm, embed_init_mode="legacy",
    embed_scale=1.0)` must be the same run. If the new code drew a single extra
    random number, every weight downstream of it would shift."""
    torch.manual_seed(7)
    implicit, _ = build_model(arm, seed=3)
    torch.manual_seed(7)
    explicit, _ = build_model(arm, seed=3, embed_init_mode="legacy", embed_scale=1.0)
    _assert_state_dicts_bit_identical(implicit.state_dict(), explicit.state_dict(),
                                      f"{arm} defaults vs explicit legacy")


# --------------------------------------------------------------------------- #
# 2. centered == the exact bias initialisation it claims to be.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("arm", ["baseline", "reservoir"])
def test_centered_sets_bias_to_minus_w_at_obs_mean(arm):
    model, _ = build_model(arm, seed=0, embed_init_mode="centered")
    mu = torch.tensor(OBS_MEAN, dtype=model.embedding.weight.dtype)
    expected = -(model.embedding.weight @ mu)
    assert torch.allclose(model.embedding.bias, expected, rtol=0.0, atol=1e-7), (
        f"{arm}: centered bias is not -(W @ obs_mean); max abs diff "
        f"{(model.embedding.bias - expected).abs().max().item():.3e}"
    )


@pytest.mark.parametrize("arm", ["baseline", "reservoir"])
def test_centered_embedding_is_algebraically_the_centred_observation(arm):
    """The identity that makes this fix free: because the embedding is LINEAR,
        W @ (x - mu) == W @ x + (-W @ mu)
    exactly. Centring the input is therefore a BIAS INITIALISATION and nothing
    else -- no new parameters, no shape change, and the bias stays trainable so the
    run can move off this point."""
    model, _ = build_model(arm, seed=0, embed_init_mode="centered")
    mu = torch.tensor(OBS_MEAN, dtype=torch.float32)
    torch.manual_seed(11)
    x = torch.rand(64, OBS_DIM) * 2.0 - 1.0          # inside the env's [-1, 1] box
    with torch.no_grad():
        centred_input = x - mu
        lhs = centred_input @ model.embedding.weight.T   # W @ (x - mu), bias-free
        rhs = model.embedding(x)                          # W @ x + b, the real layer
    max_diff = (lhs - rhs).abs().max().item()
    assert max_diff < 1e-6, (
        f"{arm}: W@(x-mu) and the centred-init layer disagree by {max_diff:.3e}"
    )


def test_centred_bias_rejects_a_wrongly_shaped_obs_mean():
    """A stale OBS_MEAN after an observation-shape change must fail loudly at
    construction, not silently centre on the wrong dimensions."""
    weight = torch.zeros(8, OBS_DIM)
    with pytest.raises(ValueError, match="one entry per input dimension"):
        centred_bias(weight, OBS_MEAN[:-1])


def test_centered_without_obs_mean_is_an_error_not_a_silent_no_op():
    with pytest.raises(ValueError, match="obs_mean"):
        PolicyValueGRU(**GRU_KWARGS, embed_init_mode="centered", obs_mean=None)


@pytest.mark.parametrize("arm", ["baseline", "reservoir"])
def test_embed_scale_multiplies_the_weight_init_std(arm):
    torch.manual_seed(5)
    unscaled, _ = build_model(arm, seed=0)
    torch.manual_seed(5)
    scaled, _ = build_model(arm, seed=0, embed_scale=3.0)
    ratio = scaled.embedding.weight.std().item() / unscaled.embedding.weight.std().item()
    assert abs(ratio - 3.0) < 1e-4, f"{arm}: embed_scale=3.0 gave a std ratio of {ratio}"


def test_unknown_embed_init_mode_is_rejected():
    for factory in (lambda m: PolicyValueGRU(**GRU_KWARGS, embed_init_mode=m),
                    lambda m: PolicyValueReservoir(**RES_KWARGS, embed_init_mode=m),
                    lambda m: build_model("baseline", embed_init_mode=m)):
        with pytest.raises(ValueError, match="embed_init_mode"):
            factory("centred")   # the British spelling is NOT one of the modes


# --------------------------------------------------------------------------- #
# 3. THE REGRESSION TEST THAT MATTERS: silent units under real observations.
# --------------------------------------------------------------------------- #

def _load_real_observations():
    if not REAL_OBS_PATH.exists():
        pytest.skip(
            f"missing {REAL_OBS_PATH}; this test needs REAL observations -- an i.i.d. "
            "synthetic surrogate does not reproduce the effect (see module docstring)"
        )
    obs = torch.as_tensor(np.load(REAL_OBS_PATH), dtype=torch.float32)
    assert obs.ndim == 2 and obs.shape[1] == OBS_DIM
    return obs


def _silent_fraction(model, obs):
    """Fraction of reservoir units that never spike once over the whole window,
    plus the mean spike rate and the fraction that spike on EVERY step.

    "Never fires over N steps" is a lower bound on "permanently silent", not the
    same thing: a unit firing at rate 1e-4 would read as silent over 6,000 steps.
    That limitation is real and is recorded in docs/EXPERIMENT_LOG.md next to the
    numbers rather than hidden behind this helper.
    """
    mem, imem, spk, _window = model.init_state(1, torch.device("cpu"))
    ever = torch.zeros(model.reservoir_size, dtype=torch.bool)
    always = torch.ones(model.reservoir_size, dtype=torch.bool)
    total_rate = 0.0
    with torch.no_grad():
        for t in range(obs.shape[0]):
            emb = model.embedding(obs[t:t + 1])
            spk, mem, imem = model.reservoir.step(emb, mem, spk, imem)
            fired = spk[0] > 0
            ever |= fired
            always &= fired
            total_rate += float(fired.float().mean())
    return (1.0 - ever.float().mean().item(),
            total_rate / obs.shape[0],
            always.float().mean().item())


def test_centering_rescues_the_silent_half_of_the_reservoir():
    """The whole point of the change, measured end to end on real observations.

    `torch.manual_seed` is called before each `build_model`, exactly as
    `run_training` does, and it is NOT optional here: the embedding's weight is
    drawn from the GLOBAL RNG, so without it this test's numbers would drift with
    whatever ran before it in the session.

    Numbers from the run that introduced this test (seed 0, all 6,000 stored
    steps, reservoir_size=8192):

        legacy   scale=1.0   silent 46.1060%   spike rate 0.018749   saturated 0%
        centered scale=3.0   silent  1.6602%   spike rate 0.017001   saturated 0%

    and over 8 seeds (seed s = both the global seed and the frozen-reservoir seed,
    i.e. what `--seed s` actually produces):

        legacy   scale=1.0   silent 45.5917% mean (43.3594-47.8149), rate 0.022551
        centered scale=3.0   silent  2.0523% mean ( 1.2939- 2.6489), rate 0.018482

    with zero saturated units in every cell -- reproducing the 8-seed diagnostic
    that motivated the fix (44.7403% -> 1.7532%, rate 0.024351 -> 0.020912).

    Scale alone does not do this: `legacy` at the same 3.0 scale still leaves
    30.3329% silent (8-seed mean) and does it by pushing the spike rate to
    0.117302, five times the healthy band -- it multiplies the DC and AC components
    together, so it buys firing by brute force rather than by fixing the offset.
    Nor does centring alone: `centered` at scale 1.0 removes the DC drive without
    replacing it and leaves 65.9454% silent at a spike rate of 0.000474. The two
    knobs are only a fix together.
    """
    obs = _load_real_observations()
    torch.manual_seed(0)
    legacy, _ = build_model("reservoir", seed=0)
    torch.manual_seed(0)
    centered, _ = build_model("reservoir", seed=0, embed_init_mode="centered", embed_scale=3.0)

    legacy_silent, legacy_rate, legacy_sat = _silent_fraction(legacy, obs)
    centered_silent, centered_rate, centered_sat = _silent_fraction(centered, obs)
    report = (f"legacy: silent={legacy_silent:.4%} rate={legacy_rate:.6f} "
              f"saturated={legacy_sat:.4%} | centered+3.0: silent={centered_silent:.4%} "
              f"rate={centered_rate:.6f} saturated={centered_sat:.4%}")

    assert legacy_silent > 0.30, f"legacy init no longer shows the pathology -- {report}"
    assert centered_silent < 0.10, f"centring failed to rescue the silent units -- {report}"
    assert centered_silent < legacy_silent / 3.0, (
        f"centring must be a dramatic improvement, not a marginal one -- {report}"
    )
    # The spike rate must land in the ~2% band spiking_reservoir.py documents as
    # healthy: rescuing units by driving the whole reservoir into saturation would
    # satisfy the silent-fraction assertion above while being strictly worse.
    assert 0.005 < centered_rate < 0.10, f"centred spike rate out of band -- {report}"
    assert centered_sat == 0.0, f"centring must not saturate units -- {report}"


# --------------------------------------------------------------------------- #
# 4. The invariants the rest of the experiment rests on, under BOTH modes.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("embed_init_mode", list(EMBED_INIT_MODES))
def test_parameter_parity_holds_under_both_init_modes(embed_init_mode):
    """Spec §5's matched-budget rule. Centring is a BIAS INITIALISATION -- the bias
    already existed -- so it cannot change either count; this test is what makes
    that claim checkable rather than asserted."""
    gru, _ = build_model("baseline", embed_init_mode=embed_init_mode)
    res, _ = build_model("reservoir", embed_init_mode=embed_init_mode)
    gru_count = gru.trainable_parameter_count()
    res_count = res.trainable_parameter_count()
    ratio = res_count / gru_count
    assert (1 - PARITY_TOLERANCE) <= ratio <= (1 + PARITY_TOLERANCE), (
        f"embed_init_mode={embed_init_mode!r} broke spec §5 parity: GRU={gru_count}, "
        f"reservoir-arm={res_count}, ratio={ratio:.3f}"
    )


def test_parameter_counts_are_identical_across_init_modes_and_scales():
    counts = {
        (mode, scale): (build_model("baseline", embed_init_mode=mode,
                                    embed_scale=scale)[0].trainable_parameter_count(),
                        build_model("reservoir", embed_init_mode=mode,
                                    embed_scale=scale)[0].trainable_parameter_count())
        for mode in EMBED_INIT_MODES for scale in (1.0, 3.0)
    }
    assert len(set(counts.values())) == 1, (
        f"the embedding-init knob changed the parameter budget: {counts}"
    )


@pytest.mark.parametrize("embed_init_mode", list(EMBED_INIT_MODES))
def test_frozen_reservoir_invariant_holds_under_both_init_modes(embed_init_mode):
    """Spec §3. The embedding init touches only the TRAINABLE embedding, so the
    frozen buffers must be untouched -- and, since `assert_reservoir_frozen`
    compares against a snapshot taken at construction, this also proves the new
    code does not perturb the reservoir's own RNG stream."""
    model, _ = build_model("reservoir", seed=2, embed_init_mode=embed_init_mode,
                           embed_scale=3.0)
    model.assert_reservoir_frozen()
    obs = torch.zeros(1, OBS_DIM)
    mem, imem, spk, window = model.init_state(1, torch.device("cpu"))
    for _ in range(3):
        _logits, _value, mem, imem, spk, window = model(obs, mem, imem, spk, window)
    model.assert_reservoir_frozen()   # a forward pass must not have moved anything


# --------------------------------------------------------------------------- #
# 5. Backward compatibility with checkpoints that predate both new keys.
# --------------------------------------------------------------------------- #

def test_a_checkpoint_without_the_new_keys_still_loads(tmp_path):
    """Synthesised pre-change checkpoint: written, then stripped of the new keys to
    match exactly what the 200 files on disk contain. This runs everywhere, unlike
    the on-disk test below, which needs a real completed run present."""
    model, optimizer = build_model("reservoir", seed=1)
    path = tmp_path / "step_128.pt"
    save_checkpoint(model, optimizer, 128, str(path))
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    assert ckpt["embed_init_mode"] == "legacy" and ckpt["embed_scale"] == 1.0
    del ckpt["embed_init_mode"], ckpt["embed_scale"]
    torch.save(ckpt, path)

    fresh, fresh_opt = build_model("reservoir", seed=1)
    step = load_checkpoint(fresh, fresh_opt, str(path), expected_arm="reservoir",
                           expected_seed=1)
    assert step == 128
    reloaded = torch.load(path, map_location="cpu", weights_only=True)
    assert reloaded.get("embed_init_mode", "legacy") == "legacy"
    assert reloaded.get("embed_scale", 1.0) == 1.0


def test_an_existing_on_disk_checkpoint_still_loads():
    """The real thing: one of the 200 checkpoints from the completed runs. Skipped
    cleanly when no run is present (a fresh clone, or a git worktree -- `*.pt` is
    gitignored), because a skip is honest and a synthesised stand-in dressed up as
    the real file would not be."""
    candidates = sorted(REPO_ROOT.glob("checkpoints/*/step_*.pt"))
    if not candidates:
        pytest.skip("no existing checkpoints on disk to test backward compatibility against")
    path = candidates[0]
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    arm, seed = ckpt.get("arm"), ckpt.get("seed")
    if arm is None or seed is None:
        pytest.skip(f"{path} carries no arm/seed label")
    # `.get`, never `ckpt[...]`: these files predate both keys and indexing them
    # would turn 20 completed runs into unloadable files.
    assert ckpt.get("embed_init_mode", "legacy") in EMBED_INIT_MODES
    assert float(ckpt.get("embed_scale", 1.0)) > 0.0

    model, optimizer = build_model(arm, seed=seed)
    step = load_checkpoint(model, optimizer, str(path), expected_arm=arm, expected_seed=seed)
    assert step == int(path.stem.split("_")[-1])
    if arm == "reservoir":
        model.assert_reservoir_frozen()
