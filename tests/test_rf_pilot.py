"""Tests for the resonate-and-fire pilot's PRE-REGISTERED BANDS, as code.

Scope, deliberately narrow, the same way `tests/test_reservoir_health.py` is: every
test below runs on a synthetic fixture or a closed-form identity. None loads a
checkpoint, reads a result JSON, or builds a production-geometry reservoir. The
report sections that do those things are exercised by running the module
(`analysis/rf_pilot.py --stage preflight`), not from here.

WHAT THIS FILE IS ACTUALLY GUARDING. `docs/EXPERIMENT_LOG.md` §23 fixes every band
in this pilot BEFORE any resonate-and-fire number existed, and §23.6 requires the
verdicts be "computed in code, against the bands below [...] never by eyeballing a
number afterwards". A band is only pre-registered in any meaningful sense if the
code implements the band that was written down -- including which end of each
interval is closed. So the tests here are overwhelmingly BOUNDARY tests: for every
threshold, one value just inside and one just outside, plus the boundary value
itself, checked against the exact wording §23.6 uses (`>=`, `<=`, `<`, `in
[a, b]`). A silent off-by-one-epsilon in a comparison operator is precisely the
class of error a pre-registration exists to prevent, and it is invisible in a
report that prints only the verdict.
"""
import math

import numpy as np
import pytest
import torch

from analysis.reservoir_health import (AMBIGUOUS_PHRASE, band_verdict,
                                       dc_offset_factor, induced_membrane_offset)
from analysis.rf_pilot import (
    DECISION_AMBIGUOUS,
    DECISION_INFORMATIVE_NEGATIVE,
    DECISION_NOT_CONFIRMED,
    DECISION_SCALE_UP,
    EMBED_SCALE_GRID,
    EMBED_SCALE_GRID_COARSE,
    EMBED_SCALE_GRID_REFINEMENT,
    GA2_CONFIRMED_BELOW,
    GA2_FALSIFIED_AT_OR_ABOVE,
    GA_CONFIRMED_HIGH,
    GA_CONFIRMED_LOW,
    GA_FALSIFIED_AT_OR_ABOVE,
    GA_FALSIFIED_AT_OR_BELOW,
    GB_NOT_PROMISING_AT_OR_BELOW,
    GB_PROMISING_AT_OR_ABOVE,
    GB_THRESHOLD_AGREEMENT_TOL,
    PERIOD_BAND_EDGES,
    PERIOD_BAND_LABELS,
    RF_PERIOD_BIN_EDGES,
    WELCH_SEGMENT_LEN,
    ac_gain,
    band_bin_counts,
    band_power_density,
    band_power_fraction_in,
    band_power_fractions,
    bin_periods,
    dc_gain_at_period,
    dc_power_fraction,
    decision_rule,
    ga2_verdict,
    ga_verdict,
    gb_threshold_agrees,
    gb_threshold_from_data,
    gb_verdict,
    mean_dc_gain_log_uniform,
    period_bin_index,
    resonant_u_response_var,
    select_embed_scale,
    unresolved_slow_fraction,
    welch_psd,
)

BETA = 0.9  # §23.2: the resonate-and-fire path holds the LIF path's own beta


# --------------------------------------------------------------------------- #
# §23.10(b) -- the induced-offset factor is the REAL PART of the complex DC
# gain, not its magnitude, and it must reduce to the LIF formula at omega = 0
# --------------------------------------------------------------------------- #

def _magnitude_dc_gain(beta, omega):
    """`|1/(1 - beta*e^{i w})|` -- the quantity §23.3 tabulates and G0a gates on,
    written here independently of `SpikingReservoir.dc_gain` so the comparison
    below is against the definition rather than against the same code twice."""
    re = 1.0 - beta * torch.cos(omega)
    im = beta * torch.sin(omega)
    return 1.0 / torch.sqrt(re * re + im * im)


class TestDcOffsetFactorReducesToLifAtZeroFrequency:
    """The property that makes the LIF path PROVABLY unchanged, per §23.10(b):
    the real-part factor `(1 - b cos w)/(1 - 2b cos w + b^2)` is exactly
    `1/(1-b)` at `w = 0`, which is the formula `analysis/reservoir_health.py`
    used before resonate-and-fire existed."""

    def test_none_omega_returns_the_scalar_lif_factor(self):
        # `omega=None` IS the LIF arm -- SpikingReservoir.omega refuses to hand
        # back a tensor of zeros for a LIF reservoir, so the caller passes None.
        assert dc_offset_factor(BETA, None) == 1.0 / (1.0 - BETA)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_all_zero_omega_equals_one_over_one_minus_beta_exactly(self, dtype):
        # EXACTLY, not approximately: `torch.equal`, not `allclose`. Floating-point
        # associativity means the algebraically-equivalent forms of this expression
        # do NOT all round to 1/(1-beta) at cos(w) = 1, so "the lif path is
        # unchanged" has to be a tested property of this implementation rather
        # than an appeal to the algebra.
        omega = torch.zeros(7, dtype=dtype)
        expected = torch.full((7,), 1.0 / (1.0 - BETA), dtype=dtype)
        assert torch.equal(dc_offset_factor(BETA, omega), expected)

    def test_a_mixed_omega_keeps_the_zero_entries_exact(self):
        # A single zero-frequency unit inside an otherwise-rotating population
        # must still read exactly 1/(1-beta): the LIF point of the family is a
        # per-unit property, not a whole-reservoir mode. Compared IN the tensor's
        # own dtype -- at float32, 1/(1-beta) is 10.0f, and widening it to a
        # Python float first would compare against float64's 10.000000000000002
        # and could never hold.
        omega = torch.tensor([0.0, 0.5, 0.0, math.pi / 4], dtype=torch.float64)
        factor = dc_offset_factor(BETA, omega)
        assert float(factor[0]) == 1.0 / (1.0 - BETA)
        assert float(factor[2]) == 1.0 / (1.0 - BETA)

    @pytest.mark.parametrize("beta", [0.5, 0.8, 0.9, 0.95])
    def test_holds_at_other_betas_too(self, beta):
        omega = torch.zeros(3, dtype=torch.float64)
        expected = torch.full((3,), 1.0 / (1.0 - beta), dtype=torch.float64)
        assert torch.equal(dc_offset_factor(beta, omega), expected)


class TestDcOffsetFactorAgainstTheMagnitude:
    """§23.10(b): "The real part is smaller than the magnitude everywhere except
    in the limit w -> 0", i.e. the operationally relevant attenuation of the
    standing offset is LARGER than §23.3's table claims."""

    def test_strictly_smaller_than_the_magnitude_on_the_open_interval(self):
        # Excludes both endpoints: 0 and pi are the two frequencies at which the
        # pole is REAL, so the DC gain has no imaginary part to lose and the two
        # quantities coincide. Everywhere strictly between, it must be smaller.
        omega = torch.linspace(0.01, math.pi - 0.01, 64, dtype=torch.float64)
        real = dc_offset_factor(BETA, omega)
        magnitude = _magnitude_dc_gain(BETA, omega)
        assert bool((real < magnitude).all())

    def test_equal_to_the_magnitude_at_omega_pi(self):
        # The Nyquist period T = 2 (§23.2's T_min) maps to w = pi exactly, where
        # the pole is real and negative: DC gain 1/(1+beta) = 0.5263 at beta=0.9,
        # which is the T=2 entry in §23.3's own per-octave table. Recorded as a
        # test rather than a footnote because "strictly smaller for w > 0" is the
        # natural-but-wrong generalisation of §23.10(b)'s sentence.
        omega = torch.tensor([math.pi], dtype=torch.float64)
        real = float(dc_offset_factor(BETA, omega)[0])
        assert real == pytest.approx(1.0 / (1.0 + BETA), abs=1e-12)
        assert real == pytest.approx(float(_magnitude_dc_gain(BETA, omega)[0]), abs=1e-12)

    def test_matches_the_closed_form_on_the_preregistered_octaves(self):
        # §23.3's per-octave DC gains are stated as MAGNITUDES; the real part at
        # the same frequencies is what the offset column needs. Checked against
        # the formula §23.10(b) writes out, term for term.
        periods = torch.tensor([2.0, 4.0, 8.0, 16.0, 32.0], dtype=torch.float64)
        omega = 2.0 * math.pi / periods
        cos_w = torch.cos(omega)
        expected = (1.0 - BETA * cos_w) / (1.0 - 2.0 * BETA * cos_w + BETA ** 2)
        assert torch.allclose(dc_offset_factor(BETA, omega), expected, atol=1e-12)


class TestInducedMembraneOffset:
    """The offset column itself: `drive * factor`, with the LIF branch written as
    the historical DIVISION so A9's committed output reproduces byte for byte."""

    def test_lif_branch_is_bit_identical_to_the_historical_division(self):
        # `drive / (1 - beta)` and `drive * (1/(1 - beta))` differ by up to one
        # ULP, because 1/(1-beta) is itself a rounded value. A9's table is
        # committed at results_v2_health.txt to four decimal places and has to
        # reproduce byte for byte, so the LIF branch must be the DIVISION.
        drive = torch.randn(2048, generator=torch.Generator().manual_seed(0))
        assert torch.equal(induced_membrane_offset(drive, BETA, None),
                           drive / (1.0 - BETA))

    def test_rf_branch_scales_the_drive_per_unit(self):
        drive = torch.tensor([1.0, -2.0, 4.0], dtype=torch.float64)
        omega = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        got = induced_membrane_offset(drive, BETA, omega)
        assert torch.allclose(got, drive * dc_offset_factor(BETA, omega), atol=1e-12)

    def test_a_zero_frequency_rf_unit_matches_the_lif_offset_exactly(self):
        # The single-variable-swap property, at the level A9 actually reports:
        # an rf reservoir whose omega happened to be all zeros must produce the
        # SAME offset column as the LIF arm, not merely a close one.
        drive = torch.randn(512, generator=torch.Generator().manual_seed(1),
                            dtype=torch.float64)
        assert torch.equal(induced_membrane_offset(drive, BETA, torch.zeros(512,
                                                                            dtype=torch.float64)),
                           induced_membrane_offset(drive, BETA, None))


# --------------------------------------------------------------------------- #
# §23.6 GA -- the primary gate. Mean spike rate at step_1000064 on the fixture.
#   CONFIRMED  : mean in [0.005, 0.050]      (CLOSED at both ends)
#   FALSIFIED  : mean >= 0.100 OR mean <= 0.002
#   AMBIGUOUS  : otherwise
# --------------------------------------------------------------------------- #

class TestGaBandsMatchTheWordingOfSection236:
    def test_the_four_thresholds_are_the_preregistered_numbers(self):
        # Pinned literally: these came off §23.6 before any rf number existed, and
        # a "harmless" rounding of one of them is the exact failure a
        # pre-registration is for.
        assert (GA_CONFIRMED_LOW, GA_CONFIRMED_HIGH) == (0.005, 0.050)
        assert GA_FALSIFIED_AT_OR_ABOVE == 0.100
        assert GA_FALSIFIED_AT_OR_BELOW == 0.002

    def test_the_bands_are_disjoint_and_ordered(self):
        # If these ever overlapped, the verdict would depend on the order the
        # branches happen to be written in rather than on §23.6.
        assert (GA_FALSIFIED_AT_OR_BELOW < GA_CONFIRMED_LOW
                <= GA_CONFIRMED_HIGH < GA_FALSIFIED_AT_OR_ABOVE)

    # --- the starved edge: FALSIFIED "at or below" 0.002 -------------------- #
    def test_exactly_the_starved_threshold_falsifies(self):
        assert ga_verdict(0.002) == "FALSIFIED"

    def test_just_above_the_starved_threshold_is_ambiguous(self):
        assert ga_verdict(0.002001) == "AMBIGUOUS"

    def test_well_below_the_starved_threshold_falsifies(self):
        assert ga_verdict(0.0) == "FALSIFIED"

    # --- the lower confirmed edge: "mean in [0.005, ...]" ------------------- #
    def test_exactly_the_lower_confirmed_bound_confirms(self):
        # A CLOSED interval: 0.005 is IN [0.005, 0.050].
        assert ga_verdict(0.005) == "CONFIRMED"

    def test_just_below_the_lower_confirmed_bound_is_ambiguous(self):
        assert ga_verdict(0.004999) == "AMBIGUOUS"

    # --- the upper confirmed edge: "... 0.050]" ----------------------------- #
    def test_exactly_the_upper_confirmed_bound_confirms(self):
        assert ga_verdict(0.050) == "CONFIRMED"

    def test_just_above_the_upper_confirmed_bound_is_ambiguous(self):
        assert ga_verdict(0.050001) == "AMBIGUOUS"

    # --- the runaway edge: FALSIFIED "at or above" 0.100 -------------------- #
    def test_exactly_the_runaway_threshold_falsifies(self):
        assert ga_verdict(0.100) == "FALSIFIED"

    def test_just_below_the_runaway_threshold_is_ambiguous(self):
        assert ga_verdict(0.099999) == "AMBIGUOUS"

    def test_the_lif_v2_reference_falsifies(self):
        # §23.6's own LIF reference, 0.148469, is what H10 says resonate-and-fire
        # should move away from. It sits above the runaway threshold, so the band
        # would falsify the LIF arm itself -- which is the point of the band.
        assert ga_verdict(0.148469) == "FALSIFIED"

    def test_the_documented_healthy_band_confirms(self):
        # §23.6: "Documented healthy band: ~2%."
        assert ga_verdict(0.02) == "CONFIRMED"


# --------------------------------------------------------------------------- #
# §23.6 GA2 -- co-primary. Mean final silent fraction.
#   CONFIRMED < 0.15, FALSIFIED >= 0.25, AMBIGUOUS between.
# --------------------------------------------------------------------------- #

class TestGa2BandsMatchTheWordingOfSection236:
    def test_the_two_thresholds_are_the_preregistered_numbers(self):
        assert GA2_CONFIRMED_BELOW == 0.15
        assert GA2_FALSIFIED_AT_OR_ABOVE == 0.25

    def test_just_below_the_confirmed_bound_confirms(self):
        assert ga2_verdict(0.1499) == "CONFIRMED"

    def test_exactly_the_confirmed_bound_is_not_confirmed(self):
        # "CONFIRMED: < 15%", strictly -- 15% itself is ambiguous.
        assert ga2_verdict(0.15) == "AMBIGUOUS"

    def test_midband_is_ambiguous(self):
        assert ga2_verdict(0.20) == "AMBIGUOUS"

    def test_just_below_the_falsified_bound_is_ambiguous(self):
        assert ga2_verdict(0.2499) == "AMBIGUOUS"

    def test_exactly_the_falsified_bound_falsifies(self):
        # "FALSIFIED: >= 25%" -- the boundary itself falsifies.
        assert ga2_verdict(0.25) == "FALSIFIED"

    def test_the_lif_v2_reference_falsifies(self):
        # §23.6's LIF reference is 30.9570% silent.
        assert ga2_verdict(0.309570) == "FALSIFIED"

    def test_it_is_the_shared_band_verdict_helper_not_a_second_copy(self):
        # §23.6 says GA2 has the same three-way shape A9 already implements, and
        # the task is explicit that it be REUSED. If a second copy were ever
        # introduced, these would drift apart at a boundary and nothing else
        # would notice.
        for value in (0.0, 0.1499, 0.15, 0.2, 0.2499, 0.25, 0.9):
            assert ga2_verdict(value) == band_verdict(
                value, GA2_CONFIRMED_BELOW, GA2_FALSIFIED_AT_OR_ABOVE)


# --------------------------------------------------------------------------- #
# §23.6 GB -- secondary. mean_extrinsic_return, final/continuous, seeds 0-2.
#   PROMISING >= 36.9268, NOT PROMISING <= 35.4972, AMBIGUOUS between.
# --------------------------------------------------------------------------- #

class TestGbBandsMatchTheWordingOfSection236:
    def test_the_two_thresholds_are_the_preregistered_numbers(self):
        assert GB_PROMISING_AT_OR_ABOVE == 36.9268
        assert GB_NOT_PROMISING_AT_OR_BELOW == 35.4972

    def test_exactly_the_promising_threshold_is_promising(self):
        # "PROMISING: R&F mean >= 36.9268" -- closed at the boundary.
        assert gb_verdict(36.9268) == "PROMISING"

    def test_just_below_the_promising_threshold_is_ambiguous(self):
        assert gb_verdict(36.9267) == "AMBIGUOUS"

    def test_exactly_the_not_promising_threshold_is_not_promising(self):
        # "NOT PROMISING: R&F mean <= 35.4972" -- also closed at the boundary.
        assert gb_verdict(35.4972) == "NOT PROMISING"

    def test_just_above_the_not_promising_threshold_is_ambiguous(self):
        assert gb_verdict(35.4973) == "AMBIGUOUS"

    def test_the_gru_baseline_would_be_promising(self):
        # 39.7861 -- if the pilot matched the GRU it would clear the bar, which is
        # a sanity check that the band points the way §23.6 intends.
        assert gb_verdict(39.7861) == "PROMISING"

    def test_matching_the_lif_arm_exactly_is_not_promising(self):
        assert gb_verdict(35.4972) == "NOT PROMISING"


# --------------------------------------------------------------------------- #
# §23.6 / §23.11 -- the derived-threshold guard
# --------------------------------------------------------------------------- #

class TestGbThresholdAgreement:
    def test_the_threshold_closes_one_third_of_the_seed_matched_gap(self):
        # §23.6: "PROMISING: R&F mean >= 36.9268 (closes at least one third of the
        # seed-matched gap)". Derived from the two arm means, per the task.
        assert gb_threshold_from_data(30.0, 60.0) == pytest.approx(40.0)

    def test_the_preregistered_constant_agrees_with_its_own_stated_inputs(self):
        # §23.6's per-seed figures: LIF 33.806 / 34.842 / 37.844, GRU 40.904 /
        # 44.842 / 33.612. §23.11 records that the constant carries a
        # fourth-decimal rounding slip against them; the guard's job is to pass
        # anyway, at 1e-3, and to fail on anything larger.
        lif = (33.806 + 34.842 + 37.844) / 3.0
        gru = (40.904 + 44.842 + 33.612) / 3.0
        assert gb_threshold_agrees(gb_threshold_from_data(lif, gru))

    def test_a_rounding_slip_inside_the_tolerance_still_agrees(self):
        assert gb_threshold_agrees(GB_PROMISING_AT_OR_ABOVE + 0.9 * GB_THRESHOLD_AGREEMENT_TOL)

    def test_a_data_handling_error_fires_the_guard(self):
        # The case this exists for: the wrong seeds, the wrong regime, or the
        # wrong selection read out of results_v2/ would move the derived threshold
        # by far more than a fourth decimal.
        assert not gb_threshold_agrees(gb_threshold_from_data(35.4972, 50.0))

    def test_the_guard_is_symmetric_about_the_preregistered_constant(self):
        assert not gb_threshold_agrees(GB_PROMISING_AT_OR_ABOVE + 10 * GB_THRESHOLD_AGREEMENT_TOL)
        assert not gb_threshold_agrees(GB_PROMISING_AT_OR_ABOVE - 10 * GB_THRESHOLD_AGREEMENT_TOL)

    def test_the_tolerance_is_the_one_the_task_fixed(self):
        assert GB_THRESHOLD_AGREEMENT_TOL == 1e-3


# --------------------------------------------------------------------------- #
# §23.7 -- the decision rule, all nine (GA verdict x GB verdict) combinations
# --------------------------------------------------------------------------- #

class TestDecisionRule:
    """§23.7, clause by clause:
      SCALE-UP RECOMMENDED  <=>  GA CONFIRMED and GB PROMISING   (a BICONDITIONAL)
      GA FALSIFIED          =>   stop, whatever GB says
      GA CONFIRMED + GB NOT PROMISING => the informative negative
      any other combination =>   stop and report
    """

    def test_confirmed_and_promising_is_the_only_scale_up(self):
        assert decision_rule("CONFIRMED", "PROMISING") == DECISION_SCALE_UP

    def test_confirmed_and_not_promising_is_the_informative_negative(self):
        assert decision_rule("CONFIRMED", "NOT PROMISING") == DECISION_INFORMATIVE_NEGATIVE

    def test_confirmed_and_ambiguous_stops_as_ambiguous(self):
        assert decision_rule("CONFIRMED", "AMBIGUOUS") == DECISION_AMBIGUOUS

    @pytest.mark.parametrize("gb", ["PROMISING", "NOT PROMISING", "AMBIGUOUS"])
    def test_falsified_stops_whatever_gb_says(self, gb):
        assert decision_rule("FALSIFIED", gb) == DECISION_NOT_CONFIRMED

    @pytest.mark.parametrize("gb", ["PROMISING", "NOT PROMISING", "AMBIGUOUS"])
    def test_ambiguous_ga_stops_as_ambiguous_whatever_gb_says(self, gb):
        assert decision_rule("AMBIGUOUS", gb) == DECISION_AMBIGUOUS

    def test_every_combination_is_covered_and_returns_a_preregistered_string(self):
        # The exhaustive sweep, so a fourth GA or GB verdict string could not be
        # introduced without this failing.
        allowed = {DECISION_SCALE_UP, DECISION_INFORMATIVE_NEGATIVE,
                   DECISION_NOT_CONFIRMED, DECISION_AMBIGUOUS}
        seen = set()
        for ga in ("CONFIRMED", "FALSIFIED", "AMBIGUOUS"):
            for gb in ("PROMISING", "NOT PROMISING", "AMBIGUOUS"):
                out = decision_rule(ga, gb)
                assert out in allowed
                seen.add((ga, gb))
        assert len(seen) == 9

    def test_scale_up_is_reachable_from_no_other_combination(self):
        # The biconditional, tested as one: exactly one of the nine cells may
        # recommend scaling up.
        recommending = [(ga, gb)
                        for ga in ("CONFIRMED", "FALSIFIED", "AMBIGUOUS")
                        for gb in ("PROMISING", "NOT PROMISING", "AMBIGUOUS")
                        if decision_rule(ga, gb) == DECISION_SCALE_UP]
        assert recommending == [("CONFIRMED", "PROMISING")]

    def test_an_unknown_verdict_string_raises(self):
        # Better a crash than a silent fall-through to "ambiguous" that reads as
        # a real verdict.
        with pytest.raises(ValueError):
            decision_rule("MAYBE", "PROMISING")
        with pytest.raises(ValueError):
            decision_rule("CONFIRMED", "GOOD")


# --------------------------------------------------------------------------- #
# §23.4 -- the --embed-scale calibration selector
# --------------------------------------------------------------------------- #

class TestSelectEmbedScale:
    """§23.4: "the value selected is the one whose INITIAL mean spike rate [...]
    minimises |log(rate_RF / rate_LIF_v2_init)|". A ratio criterion, so it is
    symmetric in log space -- half the reference rate and twice it are equally
    far -- and it is undefined at rate 0."""

    def test_the_coarse_grid_is_the_preregistered_one(self):
        assert EMBED_SCALE_GRID_COARSE == (3.0, 4.5, 6.0, 9.0, 12.0, 18.0)

    def test_the_refinement_is_the_one_declared_in_23_12(self):
        """§23.12 fixes the refinement as nine log-spaced points on [3.0, 4.5] --
        the two adjacent coarse-grid values that bracket the transition. The
        endpoints are already in the coarse grid, so the refinement itself is the
        seven interior points, and the union is what the criterion sees."""
        assert EMBED_SCALE_GRID_REFINEMENT == (3.156, 3.32, 3.493, 3.674,
                                               3.865, 4.066, 4.278)
        assert EMBED_SCALE_GRID == (3.0, 3.156, 3.32, 3.493, 3.674, 3.865,
                                    4.066, 4.278, 4.5, 6.0, 9.0, 12.0, 18.0)
        # Log-spaced, not linearly spaced: consecutive ratios are constant.
        ratios = [b / a for a, b in zip(EMBED_SCALE_GRID[:8], EMBED_SCALE_GRID[1:9])]
        assert max(ratios) - min(ratios) < 1e-3
        # The refinement stays strictly inside the bracket it was derived from,
        # so no value outside the measured interval can be selected by it.
        assert all(3.0 < s < 4.5 for s in EMBED_SCALE_GRID_REFINEMENT)

    def test_picks_the_exact_match(self):
        rates = [(3.0, 0.001), (4.5, 0.010), (6.0, 0.100)]
        assert select_embed_scale(rates, 0.010)[0] == 4.5

    def test_picks_on_the_log_ratio_not_the_absolute_difference(self):
        # Reference 0.010. Candidate 0.005 is 0.005 away in absolute terms and a
        # factor 2 away in log terms; candidate 0.021 is 0.011 away absolutely
        # but a factor 2.1 away. An absolute-difference selector would pick 0.005
        # here too, so make the discriminating case the other way round:
        # candidate 0.019 (factor 1.9) vs candidate 0.0045 (factor 2.22). The
        # absolute differences are 0.009 and 0.0055 -- so absolute difference
        # would pick 0.0045 and the log ratio must pick 0.019.
        rates = [(3.0, 0.0045), (4.5, 0.019)]
        assert select_embed_scale(rates, 0.010)[0] == 4.5

    def test_a_tie_resolves_to_the_earlier_grid_value(self):
        # Half the reference and twice the reference are exactly equidistant in
        # log space. The grid is enumerated in ascending order in §23.4, and the
        # FIRST minimiser wins -- a rule the written grid order fixes, not one
        # chosen after seeing a tie.
        rates = [(3.0, 0.005), (4.5, 0.020)]
        assert select_embed_scale(rates, 0.010)[0] == 3.0

    def test_an_empty_table_selects_nothing(self):
        assert select_embed_scale([], 0.010)[0] is None

    def test_an_all_silent_table_selects_nothing(self):
        # A reservoir that never fires has no log ratio at all. G0b would fail on
        # such a grid anyway; the selector must not invent a winner first.
        rates = [(3.0, 0.0), (4.5, 0.0), (6.0, 0.0)]
        assert select_embed_scale(rates, 0.010)[0] is None

    def test_a_zero_reference_rate_selects_nothing(self):
        rates = [(3.0, 0.01), (4.5, 0.02)]
        assert select_embed_scale(rates, 0.0)[0] is None

    def test_silent_candidates_are_skipped_not_ranked(self):
        rates = [(3.0, 0.0), (4.5, 0.009), (6.0, 0.0)]
        assert select_embed_scale(rates, 0.010)[0] == 4.5

    def test_all_candidates_out_of_the_g0b_band_still_yields_a_selection(self):
        # §23.4's selection criterion and §23.5's G0b band are SEPARATE steps: the
        # selector minimises the log ratio whatever the resulting rate is, and G0b
        # then decides whether that rate is healthy. Collapsing the two would make
        # "no grid value lands in band" unreportable, and §23.5 says that outcome
        # "is itself a reportable finding about the mechanism".
        rates = [(3.0, 0.30), (4.5, 0.40), (6.0, 0.50)]
        selected, criterion = select_embed_scale(rates, 0.010)
        assert selected == 3.0
        assert criterion == pytest.approx(math.log(30.0))

    def test_the_returned_criterion_is_the_absolute_log_ratio(self):
        rates = [(9.0, 0.02)]
        _selected, criterion = select_embed_scale(rates, 0.010)
        assert criterion == pytest.approx(math.log(2.0))


# --------------------------------------------------------------------------- #
# --stage spectrum -- the spectral estimator and the band arithmetic.
#
# Same scope rule as the rest of this file: synthetic signals and closed forms
# only, no fixture, no checkpoint, no production-geometry reservoir. A hand-rolled
# Welch has exactly two places it goes quietly wrong -- the window normalisation
# and the one-sided doubling -- and both produce a spectrum that LOOKS right and
# integrates to the wrong number, so the tests below check it against signals
# whose answer is known in closed form rather than against its own plausibility.
# --------------------------------------------------------------------------- #

SINE_BIN = 48                  # 512/48 = 10.667 steps, mid-band in `8 <= T < 16`
BAND_8_16 = PERIOD_BAND_LABELS.index("8-16")


def _sinusoid(n_steps=6000, bin_index=SINE_BIN, segment_len=WELCH_SEGMENT_LEN):
    """A sinusoid landing exactly on a Welch bin CENTRE, well inside its band.

    Both properties are deliberate. On a bin centre the periodic Hann leaks into
    the two neighbours and nowhere else; a frequency between bins would smear
    across the whole spectrum and the test would be measuring leakage rather than
    the estimator. Mid-band, because bin 48's neighbours (T = 10.9 and 10.4) are
    still inside `8 <= T < 16` -- a sinusoid parked ON a band edge would put its
    leakage in the next band over and the test would have to tolerate it.
    """
    t = np.arange(n_steps, dtype=np.float64)
    return np.sin(2.0 * np.pi * bin_index * t / segment_len)


class TestWelchOnSignalsWithAKnownAnswer:
    def test_a_pure_sinusoid_puts_essentially_all_power_in_its_own_band(self):
        freqs, psd, _n = welch_psd(_sinusoid())
        fractions = band_power_fractions(freqs, psd[:, 0])
        assert fractions[BAND_8_16] > 0.999
        for band, fraction in enumerate(fractions):
            if band != BAND_8_16:
                assert fraction < 0.001

    def test_the_sinusoid_also_reads_through_the_closed_interval_helper(self):
        # `band_power_fraction_in` is the arbitrary-band version §4's tradeoff
        # table uses, and it has to agree with the fixed-band decomposition on a
        # signal where both are unambiguous.
        freqs, psd, _n = welch_psd(_sinusoid())
        assert band_power_fraction_in(freqs, psd[:, 0], 8.0, 16.0) > 0.999
        assert band_power_fraction_in(freqs, psd[:, 0], 2.0, 8.0) < 0.001

    def test_a_sinusoid_integrates_back_to_its_own_variance(self):
        # Parseval, which is what pins the window normalisation AND the one-sided
        # doubling at once: a unit sinusoid has variance exactly 0.5, and an
        # estimator that forgot either factor would still produce a spectrum with
        # a clean peak in the right place.
        freqs, psd, _n = welch_psd(_sinusoid())
        df = float(freqs[1] - freqs[0])
        assert float(psd.sum() * df) == pytest.approx(0.5, rel=1e-3)

    def test_white_noise_integrates_back_to_its_own_variance(self):
        noise = np.random.default_rng(0).standard_normal(32768)
        freqs, psd, _n = welch_psd(noise)
        df = float(freqs[1] - freqs[0])
        assert float(psd.sum() * df) == pytest.approx(float(noise.var()), rel=0.02)

    def test_white_noise_spreads_power_evenly_across_the_spectrum(self):
        """"Evenly" means FLAT IN FREQUENCY, which is flat DENSITY and NOT equal
        band fractions -- the bands are octaves in period and the bins are uniform
        in frequency, so `T<4` holds 128 bins where `32-64` holds 8. Writing this
        test against equal fractions would fail on a genuinely white signal, and
        that trap is precisely why the report carries a density row at all."""
        noise = np.random.default_rng(0).standard_normal(32768)
        freqs, psd, _n = welch_psd(noise)
        fractions = band_power_fractions(freqs, psd[:, 0])
        counts = band_bin_counts(freqs)
        density = band_power_density(fractions, counts)
        assert np.allclose(density, 1.0, rtol=0.2)
        # And the fractions themselves track the bin counts, which is the same
        # statement read the other way round.
        assert np.allclose(fractions, counts / counts.sum(), rtol=0.2)

    def test_the_segment_mean_bin_is_reported_and_not_folded_into_a_band(self):
        # A pure ramp is almost entirely slower than one segment. Its resolved
        # fractions still sum to 1 -- they are fractions OF the resolved power --
        # and the unresolved share has to be large, or bin 0 is being smuggled
        # into the slowest band where it would be indistinguishable from measured
        # low-frequency power.
        ramp = np.linspace(-1.0, 1.0, 6000)
        freqs, psd, _n = welch_psd(ramp - ramp.mean())
        assert unresolved_slow_fraction(psd[:, 0]) > 0.5
        assert float(band_power_fractions(freqs, psd[:, 0]).sum()) == pytest.approx(1.0)

    def test_a_record_shorter_than_one_segment_raises(self):
        with pytest.raises(ValueError):
            welch_psd(np.zeros(WELCH_SEGMENT_LEN - 1))

    def test_channels_are_independent(self):
        # (T, D) must be D separate spectra, not one spectrum of the sum: the
        # observation table prints one row per slot and pools by SUMMING the
        # per-channel spectra afterwards.
        # Bin 24 is 512/24 = 21.33 steps, mid-band in `16 <= T < 32` -- chosen the
        # same way SINE_BIN was, so its leakage neighbours stay in its own band.
        stacked = np.stack([_sinusoid(), _sinusoid(bin_index=24)], axis=1)
        freqs, psd, _n = welch_psd(stacked)
        assert psd.shape[1] == 2
        assert band_power_fractions(freqs, psd[:, 0])[BAND_8_16] > 0.999
        assert band_power_fractions(freqs, psd[:, 1])[
            PERIOD_BAND_LABELS.index("16-32")] > 0.999


class TestBandPowerFractionsAreADecomposition:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_the_fractions_sum_to_one(self, seed):
        # The property that makes them readable as a decomposition at all.
        # PERIOD_BAND_EDGES spans [0, inf), so no resolved bin can fall outside
        # it and nothing may be silently dropped.
        signal = np.random.default_rng(seed).standard_normal((6000, 3))
        freqs, psd, _n = welch_psd(signal - signal.mean(axis=0))
        for channel in range(psd.shape[1]):
            assert float(band_power_fractions(freqs, psd[:, channel]).sum()) \
                == pytest.approx(1.0)
        assert float(band_power_fractions(freqs, psd.sum(axis=1)).sum()) \
            == pytest.approx(1.0)

    def test_every_resolved_bin_is_counted_exactly_once(self):
        freqs, _psd, _n = welch_psd(np.zeros(6000))
        assert int(band_bin_counts(freqs).sum()) == len(freqs) - 1

    def test_a_channel_with_no_power_reads_nan_rather_than_zero(self):
        # A dimension that is identically zero has NO spectrum; reporting 0.0 for
        # every band would put it in the table as a measurement.
        freqs, psd, _n = welch_psd(np.zeros(6000))
        assert bool(np.isnan(band_power_fractions(freqs, psd[:, 0])).all())
        assert math.isnan(band_power_fraction_in(freqs, psd[:, 0], 2.0, 32.0))

    def test_band_power_fraction_in_is_closed_at_both_ends(self):
        # §4's rows are candidate SUPPORTS written `T in [T_min, T_max]` with both
        # endpoints meant, so a bin sitting exactly on an endpoint is inside. Two
        # rows sharing an endpoint therefore share that bin, which is why the
        # column does not sum to 1 and why the report says so.
        freqs, psd, _n = welch_psd(_sinusoid())
        exact_period = WELCH_SEGMENT_LEN / SINE_BIN
        assert band_power_fraction_in(freqs, psd[:, 0], exact_period, 64.0) > 0.3
        assert band_power_fraction_in(freqs, psd[:, 0], 2.0, exact_period) > 0.3

    def test_the_widest_band_holds_everything(self):
        freqs, psd, _n = welch_psd(
            np.random.default_rng(3).standard_normal(6000))
        assert band_power_fraction_in(freqs, psd[:, 0], 2.0, float(WELCH_SEGMENT_LEN)) \
            == pytest.approx(1.0)


class TestPeriodBucketing:
    """§3 buckets 8,192 frozen `T_i` into octaves and the counts have to sum back
    to the reservoir size, so what happens exactly ON an edge is load-bearing
    rather than a detail: half-open at the bottom, and the LAST bin closed at the
    top because §23.2's support ends at T = 32 and a draw landing there is real."""

    @pytest.mark.parametrize("period,expected", [
        (2.0, 0),           # the support's lower endpoint, inside the first bin
        (2.0001, 0),
        (3.9999, 0),
        (4.0, 1),           # ON an internal edge -> the SLOWER bin
        (7.9999, 1),
        (8.0, 2),
        (15.9999, 2),
        (16.0, 3),
        (31.9999, 3),
        (32.0, 3),          # the support's upper endpoint -> the LAST bin
    ])
    def test_octave_bins_including_every_boundary(self, period, expected):
        assert period_bin_index(period, RF_PERIOD_BIN_EDGES) == expected

    @pytest.mark.parametrize("period", [1.9999, 32.0001, 0.5, 1000.0])
    def test_outside_the_support_is_none_not_a_clamped_bin(self, period):
        # None rather than the nearest bin: a period outside §23.2's support means
        # the draw is not what was pre-registered, and quietly filing it in the
        # end bin would hide that in a table whose counts still added up.
        assert period_bin_index(period, RF_PERIOD_BIN_EDGES) is None

    @pytest.mark.parametrize("period,expected", [
        (0.5, 0), (3.9999, 0), (4.0, 1), (8.0, 2), (16.0, 3), (32.0, 4),
        (64.0, 5), (10_000.0, 5), (math.inf, 5),
    ])
    def test_the_report_bands_cover_everything_including_infinity(self, period, expected):
        # PERIOD_BAND_EDGES ends at inf so the DC bin's period has somewhere to
        # go; `band_power_fractions` drops that bin, but the bucketing must not
        # be the thing that decides so.
        assert period_bin_index(period, PERIOD_BAND_EDGES) == expected

    def test_bin_periods_maps_the_dc_bin_to_infinity(self):
        freqs, _psd, _n = welch_psd(np.zeros(6000))
        periods = bin_periods(freqs)
        assert math.isinf(periods[0])
        assert periods[1] == pytest.approx(float(WELCH_SEGMENT_LEN))
        assert periods[-1] == pytest.approx(2.0)     # Nyquist


class TestDcPowerFraction:
    def test_a_hand_computed_value(self):
        # x = [1, 3]: mean 2 so mean^2 = 4; mean(x^2) = (1 + 9)/2 = 5; 4/5 = 0.8.
        per_channel, pooled = dc_power_fraction(np.array([[1.0], [3.0]]))
        assert per_channel[0] == pytest.approx(0.8)
        assert pooled == pytest.approx(0.8)

    def test_a_hand_computed_value_pooled_over_two_channels(self):
        # Channel A as above (mean^2 = 4, mean square = 5). Channel B = [1, -1]:
        # mean 0, mean square 1. Pooled = (4 + 0)/(5 + 1) = 2/3, and NOT the mean
        # of the two per-channel shares (0.8 and 0.0), which would be 0.4 --
        # RESULTS.md §7.1's figure is ||E x||^2 / E||x||^2, an energy ratio.
        per_channel, pooled = dc_power_fraction(np.array([[1.0, 1.0], [3.0, -1.0]]))
        assert per_channel[0] == pytest.approx(0.8)
        assert per_channel[1] == pytest.approx(0.0)
        assert pooled == pytest.approx(2.0 / 3.0)

    def test_a_constant_channel_is_all_dc(self):
        per_channel, pooled = dc_power_fraction(np.full((16, 1), 2.5))
        assert per_channel[0] == pytest.approx(1.0)
        assert pooled == pytest.approx(1.0)

    def test_an_identically_zero_channel_is_nan_not_zero_and_not_one(self):
        # The three reserved observation slots (RESULTS.md v1 §9). 0/0 is not 0
        # ("no DC") and not 1 ("all DC"); it is a channel with no power, and the
        # report has to say so rather than average a fabricated number in.
        per_channel, pooled = dc_power_fraction(np.array([[1.0, 0.0], [3.0, 0.0]]))
        assert math.isnan(per_channel[1])
        # A zero channel contributes nothing to EITHER side of the pooled ratio,
        # so pooling over 1 or over 2 channels is the same number.
        assert pooled == pytest.approx(0.8)

    def test_the_chunked_accumulation_matches_the_one_shot_one(self):
        # The chunking exists so a 6,000 x 8,192 float32 input current never gets
        # a float64 copy; it must not change the answer.
        x = np.random.default_rng(7).standard_normal((1000, 4)) + 0.3
        one_shot = dc_power_fraction(x, chunk=10_000)
        chunked = dc_power_fraction(x, chunk=37)
        assert np.allclose(one_shot[0], chunked[0])
        assert one_shot[1] == pytest.approx(chunked[1])

    def test_it_accepts_a_one_dimensional_signal(self):
        per_channel, pooled = dc_power_fraction(np.array([1.0, 3.0]))
        assert per_channel[0] == pytest.approx(0.8)
        assert pooled == pytest.approx(0.8)


class TestAnalyticGainOverABand:
    """§23.3's construction number, as a function of the band rather than as the
    single row §23.3 tabulates. §4's tradeoff table is entirely this function."""

    def test_a_degenerate_band_at_omega_zero_is_exactly_the_lif_gain(self):
        # omega = 0 is T = inf, and the answer is §23.3's 10.0 -- reported as
        # 10.0000 and equal to it to within a rounding of the fourth decimal.
        #
        # IT IS PINNED AS `1.0/(1.0 - BETA)` AND NOT AS THE LITERAL 10.0, and the
        # difference is real rather than pedantic: in float64, `1 - 0.9` is
        # 0.09999999999999998 and its reciprocal is 10.000000000000002, so the
        # LIF gain in this codebase's own arithmetic is NOT the literal 10.0.
        # `analysis/reservoir_health.dc_offset_factor` uses that same expression,
        # and its tests pin it the same way (`test_none_omega_returns_the_scalar_lif_factor`).
        # Asserting the literal here would make this function the one place that
        # disagrees with the module it has to line up with, and would force
        # `dc_gain_at_period` to special-case beta = 0.9 to pass.
        assert mean_dc_gain_log_uniform(math.inf, math.inf, 0.9) == 1.0 / (1.0 - 0.9)
        assert dc_gain_at_period(math.inf, 0.9) == 1.0 / (1.0 - 0.9)
        assert mean_dc_gain_log_uniform(math.inf, math.inf, 0.9) == pytest.approx(10.0)
        assert f"{dc_gain_at_period(math.inf, 0.9):.4f}" == "10.0000"

    @pytest.mark.parametrize("beta", [0.5, 0.8, 0.9, 0.95])
    def test_the_degenerate_lif_case_holds_at_other_betas(self, beta):
        assert mean_dc_gain_log_uniform(math.inf, math.inf, beta) == 1.0 / (1.0 - beta)

    def test_a_degenerate_band_anywhere_is_the_pointwise_gain(self):
        for period in (2.0, 4.0, 8.0, 16.0, 32.0):
            assert mean_dc_gain_log_uniform(period, period, 0.9) \
                == pytest.approx(dc_gain_at_period(period, 0.9), abs=1e-12)

    def test_the_preregistered_octaves_match_section_23_3s_table(self):
        # §23.3: "T=2 -> 0.5263, T=4 -> 0.7433, T=8 -> 1.3644, T=16 -> 2.6081,
        # T=32 -> 4.7359."
        for period, quoted in ((2, 0.5263), (4, 0.7433), (8, 1.3644),
                               (16, 2.6081), (32, 4.7359)):
            assert dc_gain_at_period(period, 0.9) == pytest.approx(quoted, abs=1e-4)

    def test_the_preregistered_band_reproduces_section_23_3s_headline(self):
        # §23.3's own analytic figures for T ~ logU[2, 32]: mean DC gain 1.7846,
        # DC/AC 0.7779. If this function did not reproduce them, §4's table would
        # be a different quantity wearing §23.3's name.
        gain = mean_dc_gain_log_uniform(2.0, 32.0, 0.9)
        assert gain == pytest.approx(1.7846, abs=1e-3)
        assert gain / ac_gain(0.9) == pytest.approx(0.7779, abs=1e-3)

    def test_the_ac_gain_is_the_preregistered_constant(self):
        assert ac_gain(0.9) == pytest.approx(2.2942, abs=1e-4)

    def test_the_gain_increases_with_the_slowest_period_in_the_band(self):
        # The whole content of §4's tradeoff: admitting slower units raises the DC
        # gain the construction exists to suppress.
        gains = [mean_dc_gain_log_uniform(2.0, t, 0.9) for t in (8, 16, 32, 64, 128)]
        assert gains == sorted(gains)
        assert all(g < 10.0 for g in gains)

    def test_quadrature_is_converged_at_the_default_sample_count(self):
        coarse = mean_dc_gain_log_uniform(2.0, 128.0, 0.9, n_samples=2000)
        fine = mean_dc_gain_log_uniform(2.0, 128.0, 0.9, n_samples=200_000)
        assert coarse == pytest.approx(fine, abs=1e-6)

    def test_an_unbounded_band_raises_rather_than_returning_the_limit(self):
        # log-uniform on [2, inf) is not a distribution. Returning 1/(1-beta)
        # would report a mean over a band nobody could draw from.
        with pytest.raises(ValueError):
            mean_dc_gain_log_uniform(2.0, math.inf, 0.9)

    @pytest.mark.parametrize("bad", [(0.0, 32.0), (-2.0, 32.0), (32.0, 2.0)])
    def test_a_malformed_band_raises(self, bad):
        with pytest.raises(ValueError):
            mean_dc_gain_log_uniform(bad[0], bad[1], 0.9)


class TestResonantResponse:
    """§3b's linearised prediction. Checked against the one case with a closed
    form: at omega = 0 the resonate-and-fire unit IS the LIF membrane (§23.2), and
    a LIF membrane driven by white noise of variance s^2 has variance
    s^2/(1-beta^2) -- i.e. exactly `ac_gain(beta)^2`, the constant §23.3 states
    the AC accumulation gain in."""

    def test_a_zero_frequency_unit_reproduces_the_lif_accumulation_gain(self):
        noise = np.random.default_rng(11).standard_normal((32768, 1))
        freqs, psd, _n = welch_psd(noise - noise.mean(axis=0))
        variance = resonant_u_response_var(freqs, psd, np.zeros(1), BETA)
        assert float(variance[0]) == pytest.approx(
            float(noise.var()) * ac_gain(BETA) ** 2, rel=0.05)

    def test_a_resonant_unit_amplifies_its_own_frequency_most(self):
        # A sinusoid at one period, read by a bank tuned across periods: the unit
        # tuned to that period must respond most. This is the property the whole
        # construction rests on and it is worth one test.
        signal = _sinusoid()[:, None]
        freqs, psd, _n = welch_psd(signal)
        periods = np.array([2.5, 5.0, WELCH_SEGMENT_LEN / SINE_BIN, 24.0])
        psd_per_unit = np.repeat(psd, len(periods), axis=1)
        variance = resonant_u_response_var(freqs, psd_per_unit,
                                           2.0 * np.pi / periods, BETA)
        assert int(np.argmax(variance)) == 2

    def test_a_mismatched_psd_shape_raises(self):
        freqs, psd, _n = welch_psd(np.zeros((6000, 2)))
        with pytest.raises(ValueError):
            resonant_u_response_var(freqs, psd, np.zeros(3), BETA)


# --------------------------------------------------------------------------- #
# the ambiguous phrase §17 requires be reported "in exactly those words"
# --------------------------------------------------------------------------- #

def test_the_ambiguous_phrase_is_reused_not_retyped():
    # §23.6's GA band says AMBIGUOUS is "reported with §17's own required phrase".
    # Imported from reservoir_health rather than duplicated, so there is one copy
    # of a string two pre-registrations require verbatim.
    assert AMBIGUOUS_PHRASE == "confirms the direction while falsifying the magnitude"
