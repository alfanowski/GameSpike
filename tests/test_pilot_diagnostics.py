"""Regression tests for `analysis/pilot_diagnostics.py`'s reservoir primitives.

WHY THIS FILE EXISTS, stated bluntly because the failure it guards against had
already happened once. `silent_fraction` is the instrument BOTH shipped v2 fixes
were accepted on (§12), the instrument A9 reports its whole trajectory table with
(§15.6, `analysis/reservoir_health.py`), and the instrument §23.4's `--embed-scale`
calibration selects the resonate-and-fire pilot's operating point with. It is
imported by three modules and was covered by none of them: when the reservoir
state tuple gained its resonate-and-fire quadrature component (§23.2 -- `(mem,
spk, window)` became `(mem, imem, spk, window)`, and `SpikingReservoir.step`
started returning a 3-tuple), `silent_fraction` kept unpacking the old arities and
every caller of A9 broke at once. Nothing in the suite noticed, because nothing in
the suite called it.

So these tests are deliberately shallow on VALUE and strict on ARITY. They assert
that the function runs end to end against a real (small) `PolicyValueReservoir` in
BOTH neuron models and returns three finite fractions -- which is exactly the
property a state-tuple change breaks, and exactly the property no amount of
band-verdict unit testing on synthetic fixtures can cover. The measured numbers
themselves are not pinned here: they are pinned, on the real 6,000-step fixture at
the production geometry, by `tests/test_embedding_centering.py`, which is where
that comparison belongs.

Small geometry throughout (64 units, 2 TT cores) so the whole file costs a
fraction of a second: the property under test is the plumbing, and the plumbing is
identical at 64 units and at 8192.
"""
import math

import numpy as np
import pytest
import torch

from analysis.pilot_diagnostics import reservoir_at, silent_fraction
from models.policy_value_reservoir import PolicyValueReservoir

SMALL_RESERVOIR = 64
OBS_DIM = 12
N_STEPS = 24


def _small_model(neuron_model):
    """A production-shaped `PolicyValueReservoir` at toy size.

    64 units factor into two TT modes of 8, which is the minimum
    `SpikingReservoir._build_tt_cores` accepts (it needs a composite
    `reservoir_size`); every other argument is scaled down proportionally. The TT
    path is kept ON rather than switched to the dense one, because that is what
    `build_model` constructs and therefore what `silent_fraction` is ever handed.
    """
    torch.manual_seed(0)
    model = PolicyValueReservoir(
        obs_dim=OBS_DIM, embed_dim=8, reservoir_size=SMALL_RESERVOIR, n_actions=10,
        use_tensor_train=True, tt_rank=4, tt_n_cores=2, context_len=8, seed=0,
        d_model=8, n_layers=1, n_heads=2, neuron_model=neuron_model,
    )
    model.eval()
    return model


def _obs():
    """A short, temporally-correlated observation window.

    A cumulative sum rather than i.i.d. noise on purpose: `silent_fraction`'s own
    docstring (and `tests/test_embedding_centering.py`'s at length) records that
    an i.i.d. surrogate measures something qualitatively different from real
    observations, because a beta=0.9 membrane integrates low-frequency energy. The
    values here are not compared against anything, but a fixture that is wrong in
    kind invites someone to start comparing them.
    """
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.normal(0.0, 0.2, size=(N_STEPS, OBS_DIM)), axis=0)
    return torch.as_tensor(walk, dtype=torch.float32)


class TestSilentFractionRunsInBothNeuronModels:
    """The arity guard: §23.2's state change must not be able to break this again."""

    @pytest.mark.parametrize("neuron_model", ["lif", "rf"])
    def test_returns_three_fractions_in_unit_interval(self, neuron_model):
        silent, rate, saturated = silent_fraction(_small_model(neuron_model), _obs())
        for name, value in (("silent", silent), ("spike rate", rate),
                            ("saturated", saturated)):
            assert isinstance(value, float), f"{name} is {type(value)}, not a float"
            assert 0.0 <= value <= 1.0, f"{name} = {value} is not a fraction"

    @pytest.mark.parametrize("neuron_model", ["lif", "rf"])
    def test_silent_and_saturated_cannot_overlap(self, neuron_model):
        # A unit that never fired and a unit that fired on every step are disjoint
        # sets over a non-empty window, so the two fractions sum to at most 1.
        silent, _rate, saturated = silent_fraction(_small_model(neuron_model), _obs())
        assert silent + saturated <= 1.0

    @pytest.mark.parametrize("neuron_model", ["lif", "rf"])
    def test_spike_rate_lies_between_the_saturated_and_never_silent_bounds(
            self, neuron_model):
        # Every always-firing unit contributes 1.0 to the mean rate at every step
        # and every never-firing unit contributes 0.0, so the mean rate is bounded
        # below by the saturated fraction and above by 1 - silent.
        silent, rate, saturated = silent_fraction(_small_model(neuron_model), _obs())
        assert saturated <= rate <= 1.0 - silent

    def test_the_two_neuron_models_do_not_measure_the_same_thing(self):
        # If `rf` silently fell back to the LIF arithmetic -- or if the quadrature
        # component were dropped on the floor by a mis-threaded state tuple -- the
        # two modes would agree exactly. They must not: the rotation changes the
        # response at every unit whose drawn omega is nonzero, which is all of them
        # over T in [2, 32].
        lif = silent_fraction(_small_model("lif"), _obs())
        rf = silent_fraction(_small_model("rf"), _obs())
        assert lif != rf


class TestReservoirAtSelectsTheNeuronModel:
    """`reservoir_at` is the ONLY constructor A9 and §23.4's calibration go
    through, so it is the only place that can select the neuron model for them.

    These build a full production-geometry model (8192 units), which is ~0.6 s
    each -- the cheapest honest test of "did the argument actually reach
    `build_model`", since the thing being checked is which cell the reservoir got.
    """

    def test_default_is_lif_and_has_no_frequencies(self):
        model, drift = reservoir_at(0, "legacy", 1.0)
        assert model.reservoir.neuron_model == "lif"
        assert drift is None
        # §23's `omega` property REFUSES a LIF reservoir rather than returning
        # zeros, so rf-specific analysis cannot run silently against the control.
        with pytest.raises(AttributeError):
            _ = model.reservoir.omega

    def test_rf_builds_the_resonate_and_fire_cell(self):
        model, _ = reservoir_at(0, "centered", 3.0, neuron_model="rf")
        assert model.reservoir.neuron_model == "rf"
        assert model.reservoir.omega.shape == (model.reservoir_size,)

    def test_period_bounds_reach_the_frequency_draw(self):
        # w = 2*pi/T with T log-uniform on [T_min, T_max], so a NON-DEFAULT pair
        # must bracket every drawn omega in [2*pi/T_max, 2*pi/T_min]. Defaults
        # would put omega in [2*pi/32, pi] and fail the lower bound here.
        t_min, t_max = 4.0, 8.0
        model, _ = reservoir_at(0, "centered", 3.0, neuron_model="rf",
                                rf_period_min=t_min, rf_period_max=t_max)
        omega = model.reservoir.omega
        assert float(omega.min()) >= 2.0 * math.pi / t_max - 1e-6
        assert float(omega.max()) <= 2.0 * math.pi / t_min + 1e-6

    def test_the_default_lif_path_is_untouched_by_the_new_arguments(self):
        # Passing the defaults EXPLICITLY must construct the same frozen reservoir
        # as omitting them: the arguments are additive, and 400 committed
        # checkpoints depend on the default path staying bit-identical (§23.5,
        # G0e-i).
        implicit, _ = reservoir_at(0, "legacy", 1.0)
        explicit, _ = reservoir_at(0, "legacy", 1.0, neuron_model="lif",
                                   rf_period_min=2.0, rf_period_max=32.0)
        assert torch.equal(implicit.reservoir.W_in, explicit.reservoir.W_in)
        assert torch.equal(implicit.embedding.weight, explicit.embedding.weight)
