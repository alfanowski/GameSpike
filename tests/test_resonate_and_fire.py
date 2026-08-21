"""The resonate-and-fire neuron model (docs/EXPERIMENT_LOG.md §23), and the two
gates that make the pilot's comparison legitimate at all.

§23.5's G0e is non-negotiable and has two halves, both pinned here:

  * **G0e-ii** -- the resonate-and-fire cell evaluated at omega == 0 must reproduce
    `snn.Leaky(beta=0.9)` BIT-EXACTLY, establishing that LIF is the omega == 0 point
    of the new family and that the swap is a strict generalisation rather than a
    different model that happens to look similar.
  * **G0e-i (construction half)** -- `neuron_model="lif"` must be bit-identical to the
    code that produced the published v2 runs. Those runs are this pilot's experimental
    CONTROL; a perturbed LIF path silently invalidates the whole comparison, and no
    downstream number would look wrong. The commit that published them is `708b32d`,
    so that is what the LIF path is compared against -- the real historical file, read
    out of git, not a paraphrase of it written here.

`torch.equal`, never `allclose`, in both: the claim is that the arithmetic is the
same arithmetic, and an `allclose` version of this test would pass for a cell that
merely approximates LIF -- which is exactly the failure mode that would make the v2
control unusable while every test stayed green.
"""
import importlib.util
import math
import pathlib
import subprocess

import pytest
import torch
import snntorch as snn

from models.policy_value_reservoir import PolicyValueReservoir
from models.spiking_reservoir import ResonateFireCell, SpikingReservoir

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# The commit that published the v2 LIF runs this pilot uses as its control:
# "Phase 1 v2: the corrected comparison -- the frozen reservoir still loses (#2)".
V2_COMMIT = "708b32d"

# The production geometry (training.train.build_model's reservoir arm), and a small
# DENSE one -- the two W_res construction branches consume the generator differently,
# so an RNG perturbation could hide in either.
PROD = dict(reservoir_size=8192, input_dim=32, use_tensor_train=True,
            tt_rank=8, tt_n_cores=4)
DENSE = dict(reservoir_size=64, input_dim=8, use_tensor_train=False)


# --------------------------------------------------------------------------- #
# G0e-ii: omega == 0 IS snn.Leaky, bit for bit.
# --------------------------------------------------------------------------- #

def _spiking_drive(T, B, N, seed=0):
    """A random input sequence hot enough that the reset path is exercised hard.
    A drive that never crosses threshold would make the reset arithmetic -- the one
    part of the update where snntorch's `reset_delay=True` ordering could disagree
    with a naive transcription of §23.2's pseudocode -- untested."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(T, B, N, generator=g) * 0.8 + 0.15


def test_rf_cell_at_zero_frequency_is_snn_leaky_bit_for_bit():
    B, N, T = 4, 64, 256
    cell = ResonateFireCell(omega=torch.zeros(N), beta=0.9)
    lif = snn.Leaky(beta=0.9, learn_beta=False, learn_threshold=False)

    drive = _spiking_drive(T, B, N)
    u = torch.zeros(B, N)
    v = torch.zeros(B, N)
    mem = torch.zeros(B, N)
    spikes = 0.0
    for t in range(T):
        spk_rf, u, v = cell(drive[t], u, v)
        spk_lif, mem = lif(drive[t], mem)
        assert torch.equal(spk_rf, spk_lif), (
            f"step {t}: spike trains diverged -- the omega == 0 cell is not snn.Leaky"
        )
        assert torch.equal(u, mem), (
            f"step {t}: membrane traces diverged -- the omega == 0 cell is not snn.Leaky"
        )
        spikes += float(spk_rf.sum())

    # The reset path has to have been exercised for the equality above to mean
    # anything; a silent reservoir would satisfy it trivially.
    rate = spikes / (T * B * N)
    assert rate > 0.05, f"drive too weak to exercise the reset path (rate={rate:.4f})"
    assert torch.equal(v, torch.zeros_like(v)), (
        "the quadrature companion must stay identically zero at omega == 0 -- if it "
        "drifts, the decoupling argument in §23.2 is false"
    )


def test_rf_cell_uses_snntorchs_default_surrogate_gradient():
    """§23.2 fixes the surrogate as snntorch's default -- the one `snn.Leaky()` gets
    with no `spike_grad=`. Asserted on the GRADIENT, not on the object identity: two
    `surrogate.atan()` closures are different objects with identical numerics, and it
    is the numerics the pre-registration fixes."""
    x = torch.linspace(-2.0, 2.0, 41)
    lif = snn.Leaky(beta=0.9)
    cell = ResonateFireCell(omega=torch.zeros(41), beta=0.9)

    a = x.clone().requires_grad_(True)
    lif.spike_grad(a - lif.threshold).sum().backward()
    b = x.clone().requires_grad_(True)
    cell.spike_grad(b - cell.threshold).sum().backward()
    assert torch.equal(a.grad, b.grad), (
        "the resonate-and-fire cell's surrogate gradient is not snn.Leaky's default"
    )


# --------------------------------------------------------------------------- #
# G0e-i (construction half): the LIF path is still the v2 LIF path.
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def v2_reservoir_module(tmp_path_factory):
    """The REAL `models/spiking_reservoir.py` as of the v2 publication commit,
    extracted from git and imported under its own module name.

    Extracted rather than re-derived: the claim under test is "identical to the
    code that produced the published numbers", and a hand-copied reference would
    only prove the copy matches, which is the bug this is meant to catch."""
    src = subprocess.run(
        ["git", "show", f"{V2_COMMIT}:models/spiking_reservoir.py"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    path = tmp_path_factory.mktemp("v2_reservoir") / "v2_spiking_reservoir.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("v2_spiking_reservoir", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("kwargs", [PROD, DENSE], ids=["prod_tt", "small_dense"])
def test_lif_construction_is_bit_identical_to_the_v2_commit(v2_reservoir_module, kwargs):
    """W_in and the TT cores (or the dense W_res) must come off the SAME generator
    values they came off before the neuron-model switch existed. This is the
    RNG-stream invariance §23.5 G0e-i rests on: the resonate-and-fire `omega` draw
    is allowed to exist only where it cannot shift the stream the LIF path reads."""
    old = v2_reservoir_module.SpikingReservoir(seed=0, **kwargs)
    new = SpikingReservoir(seed=0, **kwargs)          # neuron_model defaults to "lif"
    a, b = old.state_dict(), new.state_dict()
    assert a.keys() == b.keys(), (
        f"the LIF path's buffer set changed: {sorted(set(b) ^ set(a))}"
    )
    for key in a:
        assert torch.equal(a[key], b[key]), (
            f"{key!r} is no longer bit-identical to {V2_COMMIT} -- the published v2 "
            "LIF runs are this pilot's control, so a perturbed LIF path invalidates "
            "the entire comparison"
        )


@pytest.mark.parametrize("kwargs", [PROD, DENSE], ids=["prod_tt", "small_dense"])
def test_lif_dynamics_are_bit_identical_to_the_v2_commit(v2_reservoir_module, kwargs):
    """Construction parity is necessary but not sufficient: the per-step arithmetic
    could have been reordered while every buffer stayed identical. Drive both with
    the same input sequence and compare the spike train AND the membrane trace at
    every step."""
    old = v2_reservoir_module.SpikingReservoir(seed=0, **kwargs)
    new = SpikingReservoir(seed=0, **kwargs)
    B, N, T = 2, kwargs["reservoir_size"], 32
    g = torch.Generator().manual_seed(11)
    drive = torch.randn(T, B, kwargs["input_dim"], generator=g)

    mem_o = torch.zeros(B, N); spk_o = torch.zeros(B, N)
    mem_n = torch.zeros(B, N); spk_n = torch.zeros(B, N); imem_n = None
    for t in range(T):
        spk_o, mem_o = old.step(drive[t], mem_o, spk_o)
        spk_n, mem_n, imem_n = new.step(drive[t], mem_n, spk_n, imem_n)
        assert torch.equal(spk_o, spk_n), f"step {t}: LIF spike train changed"
        assert torch.equal(mem_o, mem_n), f"step {t}: LIF membrane trace changed"
    assert float(spk_o.sum()) > 0, "the reservoir never fired -- the test is vacuous"


def test_lif_forward_parallel_is_bit_identical_to_the_v2_commit(v2_reservoir_module):
    """The LIF-only guard added to `forward_parallel` must be a pure no-op on the
    LIF path -- including on this research artifact, which no other test covers."""
    old = v2_reservoir_module.SpikingReservoir(seed=5, **DENSE)
    new = SpikingReservoir(seed=5, **DENSE)
    g = torch.Generator().manual_seed(13)
    seq = torch.randn(2, 20, DENSE["input_dim"], generator=g)
    assert torch.equal(old.forward_parallel(seq), new.forward_parallel(seq))


def test_lif_forward_is_bit_identical_to_the_v2_commit(v2_reservoir_module):
    """`forward()` gained an extra threaded state; pin its output too, since it is
    the entry point `SpikingReservoir` is used through outside the RL loop."""
    old = v2_reservoir_module.SpikingReservoir(seed=5, **DENSE)
    new = SpikingReservoir(seed=5, **DENSE)
    g = torch.Generator().manual_seed(12)
    seq = torch.randn(3, 24, DENSE["input_dim"], generator=g)
    assert torch.equal(old(seq), new(seq)), "LIF forward() output changed"


# --------------------------------------------------------------------------- #
# The frequency distribution, exactly as pre-registered (§23.2).
# --------------------------------------------------------------------------- #

def _rf(reservoir_size=8192, seed=0, **over):
    kwargs = dict(PROD, neuron_model="rf")
    kwargs["reservoir_size"] = reservoir_size
    kwargs.update(over)
    return SpikingReservoir(seed=seed, **kwargs)


def test_omega_lies_inside_the_preregistered_period_band():
    res = _rf()
    w = res.omega
    lo = 2.0 * math.pi / res.rf_period_max
    hi = 2.0 * math.pi / res.rf_period_min
    assert w.shape == (8192,)
    assert float(w.min()) >= lo and float(w.max()) <= hi, (
        f"omega range [{float(w.min()):.6f}, {float(w.max()):.6f}] escapes "
        f"[{lo:.6f}, {hi:.6f}] = 2*pi/[T_max, T_min]"
    )


def test_periods_are_log_uniform_over_the_preregistered_support():
    """§23.2 fixes LOG-uniform T on [2, 32] -- equal density per octave, the standard
    spacing for a filter bank spanning a wide frequency range. A uniform-in-T draw
    would satisfy the range check above while putting ~94% of the units in the top
    three octaves and starving the fast end, so the SHAPE is tested, not just the
    support.

    Two independent readings: a Kolmogorov-Smirnov statistic against U[0,1] on the
    normalised log-period, and an octave-bucket count. Both at generous tolerance --
    this is a distribution check on one seeded draw, not a test of torch.rand."""
    res = _rf(reservoir_size=8192, seed=0)
    T = 2.0 * math.pi / res.omega.double()
    lo, hi = math.log(res.rf_period_min), math.log(res.rf_period_max)
    z = ((T.log() - lo) / (hi - lo)).sort().values          # -> U[0,1] if log-uniform
    n = z.numel()

    grid = torch.arange(1, n + 1, dtype=torch.float64) / n
    ks = float(torch.maximum(grid - z, z - (grid - 1.0 / n)).max())
    # KS 1% critical value at n=8192 is ~1.63/sqrt(n) = 0.018; 0.05 is ~3x that.
    assert ks < 0.05, f"log-period is not uniform: KS statistic {ks:.4f}"

    # Five octaves span [2, 32]; log-uniform puts an equal share in each.
    counts = torch.histc(z.float(), bins=5, min=0.0, max=1.0)
    share = counts / n
    assert float(share.min()) > 0.15 and float(share.max()) < 0.25, (
        f"per-octave shares {share.tolist()} are not equal -- the draw is not "
        "log-uniform (a uniform-in-T draw gives roughly [0.03, 0.06, 0.13, 0.25, 0.52])"
    )


def test_the_frequency_band_follows_the_period_arguments():
    """rf_period_min/max are real knobs, not decoration: the pilot's own §23.2
    reasoning (Nyquist at one end, the BPTT window at the other) is what fixes them
    at [2, 32], and a follow-up that changes the window has to be able to move them."""
    res = _rf(reservoir_size=1024, rf_period_min=4.0, rf_period_max=8.0)
    T = 2.0 * math.pi / res.omega
    assert float(T.min()) >= 4.0 and float(T.max()) <= 8.0


def test_a_period_below_nyquist_is_refused():
    with pytest.raises(ValueError, match="Nyquist"):
        _rf(reservoir_size=64, rf_period_min=1.5)


# --------------------------------------------------------------------------- #
# G0a: the analytic gain property that is the whole point of the swap (§23.3/§23.5).
# --------------------------------------------------------------------------- #

def test_dc_gain_of_a_lif_reservoir_is_one_over_one_minus_beta_everywhere():
    res = SpikingReservoir(seed=0, reservoir_size=512, input_dim=8,
                           use_tensor_train=True, tt_rank=4, tt_n_cores=2)
    gains = res.dc_gain()
    assert gains.shape == (512,)
    assert torch.allclose(gains, torch.full((512,), 10.0)), (
        "a LIF unit integrates a constant input with gain 1/(1-0.9) = 10.0; "
        f"got min {float(gains.min()):.4f}, max {float(gains.max()):.4f}"
    )


def test_g0a_the_resonate_and_fire_draw_collapses_the_dc_gain():
    """§23.5's G0a, verbatim: mean analytic DC gain < 3.0 (LIF: 10.0) and DC/AC
    ratio < 2.0 (LIF: 4.3589). This test IS the gate -- if it fails, the frequencies
    were not drawn as §23.2 specifies and the pilot does not launch."""
    res = _rf(reservoir_size=8192, seed=0)
    mean_dc = float(res.dc_gain().mean())
    ac = res.ac_gain()
    assert mean_dc < 3.0, f"G0a: mean DC gain {mean_dc:.4f} is not < 3.0"
    assert ac == pytest.approx(1.0 / math.sqrt(1.0 - 0.9 ** 2), rel=1e-6)
    assert (mean_dc / ac) < 2.0, f"G0a: DC/AC ratio {mean_dc / ac:.4f} is not < 2.0"


def test_the_ac_gain_is_identical_in_both_neuron_models():
    """§23.3's load-bearing claim: rotating the pole changes the DC gain and leaves
    the accumulation gain for a zero-mean input EXACTLY alone, because that depends
    only on |lambda| = beta. If this ever stops holding, the mechanism is no longer
    the one H10 is about."""
    lif = SpikingReservoir(seed=0, reservoir_size=512, input_dim=8,
                           use_tensor_train=True, tt_rank=4, tt_n_cores=2)
    rf = _rf(reservoir_size=512, seed=0, input_dim=8, tt_rank=4, tt_n_cores=2)
    assert lif.ac_gain() == rf.ac_gain()


# --------------------------------------------------------------------------- #
# The policy/value wrapper: state threading, parameter parity, frozen-ness.
# --------------------------------------------------------------------------- #

def _pv(neuron_model="rf", **over):
    kwargs = dict(obs_dim=12, embed_dim=16, reservoir_size=256, n_actions=10,
                  use_tensor_train=True, tt_rank=4, tt_n_cores=2, context_len=8,
                  seed=0, neuron_model=neuron_model)
    kwargs.update(over)
    return PolicyValueReservoir(**kwargs)


@pytest.mark.parametrize("neuron_model", ["lif", "rf"])
def test_init_state_has_the_same_arity_in_both_neuron_models(neuron_model):
    """Four elements ALWAYS -- (mem, imem, spk, window) -- so the shape of the
    recurrent state never depends on a flag. A state tuple whose arity varies is
    the kind of thing that unpacks fine on one arm and silently drops a component
    on the other; in LIF mode `imem` is simply a zeros tensor nothing reads."""
    model = _pv(neuron_model)
    state = model.init_state(3, device=torch.device("cpu"))
    assert len(state) == 4
    mem, imem, spk, window = state
    assert mem.shape == (3, 256) and imem.shape == (3, 256) and spk.shape == (3, 256)
    assert window.shape == (3, 0, 256)


@pytest.mark.parametrize("neuron_model", ["lif", "rf"])
def test_forward_threads_the_quadrature_state_and_shapes_are_right(neuron_model):
    B = 3
    model = _pv(neuron_model)
    mem, imem, spk, window = model.init_state(B, device=torch.device("cpu"))
    obs = torch.randn(B, 12)
    logits, value, mem2, imem2, spk2, window2 = model(obs, mem, imem, spk, window)
    assert logits.shape == (B, 10)
    assert value.shape == (B,)
    assert mem2.shape == (B, 256) and imem2.shape == (B, 256) and spk2.shape == (B, 256)
    assert window2.shape == (B, 1, 256)


def test_the_quadrature_state_is_inert_in_lif_mode():
    """In LIF mode `imem` must be pure pass-through: the arithmetic never reads it,
    so injecting garbage into it cannot move a single output bit. This is what makes
    "always 4 elements" free rather than a behavioural change to the control arm."""
    model = _pv("lif")
    mem, imem, spk, window = model.init_state(2, device=torch.device("cpu"))
    obs = torch.randn(2, 12)
    clean = model(obs, mem, imem, spk, window)
    dirty = model(obs, mem, torch.randn_like(imem) * 100.0, spk, window)
    for a, b in zip(clean[:3] + clean[4:], dirty[:3] + dirty[4:]):
        assert torch.equal(a, b), "LIF mode read the quadrature state"


def test_the_quadrature_state_is_load_bearing_in_rf_mode():
    """The mirror image, so the test above cannot pass for the trivial reason that
    `imem` is dead everywhere: in rf mode `v` genuinely feeds back into `u`."""
    model = _pv("rf")
    mem, imem, spk, window = model.init_state(2, device=torch.device("cpu"))
    obs = torch.randn(2, 12)
    clean = model(obs, mem, imem, spk, window)
    dirty = model(obs, mem, torch.randn_like(imem) * 5.0, spk, window)
    assert not torch.equal(clean[2], dirty[2]), (
        "the quadrature companion did not influence the membrane -- the rf cell is "
        "not actually rotating"
    )


def test_a_multi_step_rf_rollout_slides_the_window_at_context_len():
    model = _pv("rf")
    state = model.init_state(1, device=torch.device("cpu"))
    obs = torch.randn(1, 12)
    for step in range(model.context_len + 5):
        logits, value, *state = model(obs, *state)
        assert state[3].shape[1] == min(step + 1, model.context_len)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()


def test_rf_adds_zero_trainable_parameters():
    """§23.2's admissibility condition. The state doubles in width, but state is
    ACTIVATION memory, not parameters: `omega` is a frozen buffer and the readout
    still sees an N-wide binary spike vector. If this ever stops holding, the arm is
    no longer admissible in the matched-parameter comparison spec §5 requires."""
    lif = _pv("lif")
    rf = _pv("rf")
    assert rf.trainable_parameter_count() == lif.trainable_parameter_count()
    named_lif = {n: p.shape for n, p in lif.named_parameters() if p.requires_grad}
    named_rf = {n: p.shape for n, p in rf.named_parameters() if p.requires_grad}
    assert named_lif == named_rf, "the rf arm has different trainable tensors"


def test_the_rf_reservoir_holds_zero_nn_parameters_and_a_frozen_omega():
    model = _pv("rf")
    assert list(model.reservoir.parameters()) == []
    buffers = dict(model.reservoir.named_buffers())
    assert "rf.omega" in buffers, sorted(buffers)
    assert model.reservoir.omega is buffers["rf.omega"]
    for name in ("rf.omega", "rf.cos_omega", "rf.sin_omega"):
        assert buffers[name].requires_grad is False, f"{name} is not frozen"
    model.assert_reservoir_frozen()


def test_the_frozen_tripwire_catches_a_mutated_omega():
    """`omega` is a frozen weight in every sense that matters -- it defines the
    reservoir as surely as W_in does -- so the runtime tripwire has to cover it. A
    mutation test, not an inspection: an exclusion list that quietly grew an `omega`
    entry would still pass a "is it in the snapshot?" check."""
    model = _pv("rf")
    model.assert_reservoir_frozen()
    with torch.no_grad():
        model.reservoir.omega[0] += 1e-3
    with pytest.raises(AssertionError, match="rf.omega"):
        model.assert_reservoir_frozen()


def test_gradients_flow_through_the_rf_reservoir():
    """The surrogate-gradient path has to survive the extra state. Checked on the
    two tensors on opposite sides of the frozen reservoir: the embedding (upstream,
    reachable only THROUGH the reservoir) and the readout's in_proj (downstream)."""
    model = _pv("rf")
    state = model.init_state(2, device=torch.device("cpu"))
    for _ in range(5):
        logits, value, *state = model(torch.randn(2, 12), *state)
    (logits.sum() + value.sum()).backward()

    for name, param in (("embedding.weight", model.embedding.weight),
                        ("readout.in_proj.weight", model.readout.in_proj.weight)):
        assert param.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(param.grad).all(), f"{name} gradient has NaN/Inf"
        assert float(param.grad.abs().sum()) > 0.0, f"{name} gradient is identically zero"


def test_forward_parallel_refuses_to_run_on_a_resonate_and_fire_reservoir():
    """`forward_parallel` composes a SCALAR affine map; a resonate-and-fire unit's
    between-reset map is a 2x2 rotation. Silently dropping the quadrature component
    would produce plausible, wrong numbers, so it must refuse."""
    res = SpikingReservoir(seed=0, reservoir_size=64, input_dim=8,
                           use_tensor_train=False, neuron_model="rf")
    with pytest.raises(NotImplementedError, match="LIF-only"):
        res.forward_parallel(torch.randn(1, 4, 8))


def test_the_frequency_draw_leaves_the_frozen_weights_untouched():
    """The rf and lif reservoirs at the same seed must share W_in and the TT cores
    BIT FOR BIT -- the omega draw comes last, so it cannot shift the stream the
    pre-existing draws read. Stated against the lif construction directly (as well
    as against 708b32d above) because this is what makes an lif-vs-rf comparison a
    controlled one: two reservoirs differing only in the neuron model, not two
    unrelated reservoirs."""
    lif = SpikingReservoir(seed=4, **PROD)
    rf = SpikingReservoir(seed=4, neuron_model="rf", **PROD)
    a, b = lif.state_dict(), rf.state_dict()
    assert set(a) < set(b), "the rf reservoir dropped a frozen weight"
    for key in a:
        assert torch.equal(a[key], b[key]), (
            f"{key!r} differs between the lif and rf constructions at the same seed "
            "-- the omega draw perturbed the generator, so the two arms are not a "
            "controlled comparison"
        )
    assert sorted(set(b) - set(a)) == [
        "rf.beta", "rf.cos_omega", "rf.omega", "rf.sin_omega", "rf.threshold"]


def test_omega_is_reproducible_from_the_seed_alone():
    """Like every other frozen weight here: same seed, same frequencies; different
    seed, different frequencies. Without the first half the pilot's three seeds are
    not reproducible; without the second they are not three seeds."""
    assert torch.equal(_rf(reservoir_size=512, seed=7).omega,
                       _rf(reservoir_size=512, seed=7).omega)
    assert not torch.equal(_rf(reservoir_size=512, seed=7).omega,
                           _rf(reservoir_size=512, seed=8).omega)


def test_omega_is_not_reachable_in_lif_mode():
    """A LIF reservoir has no frequencies, and asking for them is a bug in the
    caller rather than a request for a tensor of zeros -- zeros are a VALID omega
    (they are the LIF point of the family), so returning them would let rf-specific
    analysis run silently against the control arm and report plausible numbers.

    Matched on nn.Module's own message, not on the property's: torch's
    `__getattr__` catches the AttributeError a property raises and re-raises its
    generic one. The exception type is the contract."""
    res = SpikingReservoir(seed=0, reservoir_size=64, input_dim=8)
    with pytest.raises(AttributeError, match="no attribute 'omega'"):
        res.omega


def test_an_unknown_neuron_model_is_refused():
    with pytest.raises(ValueError, match="unknown neuron_model"):
        SpikingReservoir(seed=0, reservoir_size=64, input_dim=8, neuron_model="resonate")
