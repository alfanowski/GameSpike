"""The resonate-and-fire pilot's pre-registered gates (docs/EXPERIMENT_LOG.md §23),
computed in code rather than read off a number by a human.

WHAT THIS DECIDES, and what it deliberately does not. §23 pre-registers, and
commits BEFORE any resonate-and-fire number exists, every band this module tests
against: the G0 validity gates that decide whether the pilot launches at all
(§23.4, §23.5), the GA/GA2/GB efficacy gates that decide what it found (§23.6),
and the decision rule that turns those three verdicts into one recommendation
(§23.7). Every threshold below is a module constant carrying its subsection
number, and every verdict is a function of those constants -- §23.6's own words:
"computed in code, against the bands below [...] never by eyeballing a number
afterwards, the same discipline A7 and A9 used". If a band here disagrees with
§23, §23 is right and this module is wrong.

WHAT THIS MODULE CANNOT TELL YOU, stated up front so a number lifted out of its
output carries its caveat:

  * NOTHING ABOUT SIGNIFICANCE. §23.6 declares GB underpowered before measuring
    it: three seeds against three seeds puts the exact two-sided permutation
    test's resolution floor at 2/C(6,3) = 0.1, so no p-value is computed here and
    none can be quoted. The per-seed sign test is printed as a supporting
    statistic and is not a test. §23.9: this is a pilot -- "a signal about whether
    a mechanism is worth a full matrix, not a measurement of whether it works".
  * NOTHING ABOUT WHAT AN rf POLICY EXPERIENCES IN SITU. Every construction
    measurement runs on `tests/data/real_obs_6000.npy`, which §14.13 records was
    collected under v1 POLICIES. Holding the observation window fixed and varying
    only the neuron model is the right controlled comparison for a construction
    property (§23.4); it is not, and is not claimed to be, a measurement of the
    observation distribution a trained resonate-and-fire agent would visit.
  * NOTHING ABOUT PERMANENT SILENCE. "Never fired in 6,000 steps" is an UPPER
    BOUND on permanently silent (see `silent_fraction`), not the same thing.
  * NOTHING ABOUT G0d OR G0e. Throughput (G0d) is a wall-clock measurement on a
    quiet machine and bit-exactness (G0e) needs a real short training run and a
    cell-level `torch.equal`; both are measured elsewhere -- G0e by
    `tests/test_neuron_model_flag.py` and `tests/test_resonate_and_fire.py`, at
    one torch thread per §23.11. The preflight report SAYS so rather than
    silently reporting four gates where §23.5 lists six.
  * NOTHING THAT PICKS A FREQUENCY BAND. `--stage spectrum` measures where the
    observation's energy actually sits and what the bank does with it, and stops
    there. It gates nothing, recommends nothing, and deliberately ends with a
    TRADEOFF TABLE rather than a choice: choosing a band is a new
    pre-registration, and making it here -- after seeing the numbers below --
    would be exactly the post-hoc move §23 exists to prevent.

THE THIRD STAGE, AND WHY IT EXISTS. §23.12 recorded the preflight negative and
offered a mechanism for it: "Real observations carry almost no energy at
2-6-step periods, so the fast half of the filter bank receives no drive." That
sentence was an INFERENCE from an indirect measurement -- silence sorted by
resonant period -- and not a measurement of the observation's spectrum. Nobody
had looked. `--stage spectrum` looks, on the same committed fixture, with the
same instrument rule: the observation's own Welch spectrum, the spectrum of the
input current the units actually receive, the bank's response resolved by its own
resonant period, and the analytic gain/drive tradeoff a corrected band would be
derived from. Where the measurement contradicts §23.12's prose, the report says
so in the same words it would have used to confirm it.

THE INSTRUMENT RULE (§17.11, restated by §23.6 and implemented rather than
asserted). Both sides of every comparison are measured by THE SAME CODE: the LIF
reference figures are re-derived here from `checkpoints_v2/` and `results_v2/` by
the identical functions that measure the resonate-and-fire arm, never transcribed
from the prose. §23's prose figures are carried below as `*_QUOTED_*` constants
purely so the re-derived values can be DIFFED against them and any discrepancy
printed -- "reported rather than silently preferred". A pilot whose control is a
number typed into a document is not a controlled comparison.

READ-ONLY, AND GRACEFUL ON A MATRIX THAT DOES NOT EXIST YET. This module writes no
files, trains nothing, and never mutates anything it can be pointed at. Every run
directory, checkpoint and result JSON is optional: a missing one is reported and
skipped, never raised, so the verdict stage run against an empty
`checkpoints_rf_pilot/` produces the LIF/GRU reference half of every table and
says plainly that the rf half has no data. That is the same discipline
`analysis/reservoir_health.py` was written under and for the same reason -- the
matrix this is pointed at may still be running.

DETERMINISM. The only randomness drawn is the same `torch.manual_seed(seed)` /
`SpikingReservoir(seed=seed)` pair that constructed the run's own model, via
`reservoir_at`. Same files on disk, same report, every time.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from analysis.pilot_diagnostics import (REAL_OBS_PATH,  # noqa: E402
                                        reservoir_at, silent_fraction)
from analysis.reservoir_health import (A9_TRAJECTORY_HEADER,  # noqa: E402
                                       AMBIGUOUS_PHRASE, EXPECTED_FINAL_STEP,
                                       a9_run_stats, a9_trajectory_row,
                                       band_verdict, induced_membrane_offset,
                                       parse_seed_spec, run_dir)
from envs.mario_land_env import OBS_MEAN  # noqa: E402

# --------------------------------------------------------------------------- #
# the production geometry the gates are defined at (§23.2 holds all of it fixed)
# --------------------------------------------------------------------------- #

RESERVOIR_SIZE = 8192        # §23.2: "reservoir_size = 8192 [...] all unchanged"
TT_RANK = 8                  # §23.2
TT_N_CORES = 4               # §23.2

# --------------------------------------------------------------------------- #
# §23.4 -- the ONE deliberate departure from "hold everything else fixed"
# --------------------------------------------------------------------------- #

# "searched over the pre-declared grid {3.0, 4.5, 6.0, 9.0, 12.0, 18.0}". Six
# values, ascending, and ALL SIX are reported whatever is selected -- §23.4 says
# so in as many words, and the ascending order is what resolves a tie in
# `select_embed_scale` (a rule the written grid fixes, not one chosen later).
EMBED_SCALE_GRID_COARSE = (3.0, 4.5, 6.0, 9.0, 12.0, 18.0)

# §23.12 -- the ONE declared refinement, and the reason it is not a search.
#
# The coarse grid above cannot satisfy G0b and G0c together: at 3.0 the initial
# spike rate is a healthy 0.008261 and 48.1445% of the reservoir is silent; at
# 4.5 the reservoir is live at 0.9033% silent and the rate has overshot the band
# to 0.059099. The whole transition falls inside that one 1.5x step, across which
# the rate rises 7.2x and the silent fraction falls 53x.
#
# §23.4's criterion is not what failed -- it targets the LIF reference rate of
# 0.018013 and the grid offers it nothing nearer than 0.008261 below and 0.059099
# above. A criterion cannot select a value its grid does not contain.
#
# The refinement therefore carries NO discretion: nine log-spaced points on
# [3.0, 4.5], the two adjacent coarse-grid values that already bracket the
# transition, fixed by the measured table and not by preference. The §23.4
# selection criterion is unchanged and is evaluated over the union. §23.12 states
# in advance that this is the only refinement: if the selected point fails G0b or
# G0c the pilot stops and the preflight negative is the result.
EMBED_SCALE_GRID_REFINEMENT = tuple(
    round(3.0 * (4.5 / 3.0) ** (k / 8.0), 3) for k in range(1, 8)
)

EMBED_SCALE_GRID = tuple(sorted(set(EMBED_SCALE_GRID_COARSE + EMBED_SCALE_GRID_REFINEMENT)))

# The reference the criterion is taken against: the LIF v2 arm's own init on the
# same fixture, same seeds, same instrument -- i.e. `centered` at 3.0, which is
# what the published v2 matrix trained under.
LIF_INIT_EMBED_MODE = "centered"
LIF_INIT_EMBED_SCALE = 3.0
RF_INIT_EMBED_MODE = "centered"

# --------------------------------------------------------------------------- #
# §23.5 -- G0, the validity gates. A failure here means the pilot is not
# measuring what it claims to measure, and it does not launch.
# --------------------------------------------------------------------------- #

G0A_MEAN_DC_GAIN_BELOW = 3.0        # §23.5 G0a (LIF: 10.0)
G0A_DC_OVER_AC_BELOW = 2.0          # §23.5 G0a (LIF: 4.3589)
G0B_INIT_RATE_LOW = 0.005           # §23.5 G0b
G0B_INIT_RATE_HIGH = 0.050          # §23.5 G0b
G0C_INIT_SILENT_BELOW = 0.15        # §23.5 G0c (v2 centered@3.0 LIF: 2.0523%)

# §23.3's analytic LIF figures, quoted so G0a's report shows what it is moving
# away from. 1/(1-beta) and (1/(1-beta))/(1/sqrt(1-beta^2)) at beta = 0.9.
LIF_QUOTED_MEAN_DC_GAIN = 10.0      # §23.3
LIF_QUOTED_DC_OVER_AC = 4.3589      # §23.3
RF_QUOTED_MEAN_DC_GAIN = 1.7846     # §23.3, over T ~ logU[2, 32]
RF_QUOTED_DC_OVER_AC = 0.7779       # §23.3

# --------------------------------------------------------------------------- #
# §23.6 -- GA / GA2 / GB, the efficacy gates. Fixed before measurement.
# --------------------------------------------------------------------------- #

GA_CONFIRMED_LOW, GA_CONFIRMED_HIGH = 0.005, 0.050      # §23.6
GA_FALSIFIED_AT_OR_ABOVE = 0.100                        # §23.6
GA_FALSIFIED_AT_OR_BELOW = 0.002                        # §23.6
GA2_CONFIRMED_BELOW = 0.15                              # §23.6
GA2_FALSIFIED_AT_OR_ABOVE = 0.25                        # §23.6
GB_PROMISING_AT_OR_ABOVE = 36.9268                      # §23.6
GB_NOT_PROMISING_AT_OR_BELOW = 35.4972                  # §23.6

# A mistyped constant is the one failure mode a pre-registration cannot survive,
# and it would otherwise surface as a plausible verdict. Checked at import.
assert GA_FALSIFIED_AT_OR_BELOW < GA_CONFIRMED_LOW <= GA_CONFIRMED_HIGH \
    < GA_FALSIFIED_AT_OR_ABOVE, "GA's bands overlap or are out of order (§23.6)"
assert GA2_CONFIRMED_BELOW < GA2_FALSIFIED_AT_OR_ABOVE, "GA2's bands cross (§23.6)"
assert GB_NOT_PROMISING_AT_OR_BELOW < GB_PROMISING_AT_OR_ABOVE, "GB's bands cross (§23.6)"

# How closely the threshold re-derived from `results_v2/` must agree with the
# pre-registered constant above. §23.11 records that §23.6's GB block carries a
# fourth-decimal rounding slip against its own per-seed inputs, so an exact match
# is not expected and is not required; what this guard catches is a DATA-HANDLING
# error -- the wrong seeds, the wrong regime, the wrong selection -- which would
# move the derived threshold by orders of magnitude more than a fourth decimal.
GB_THRESHOLD_AGREEMENT_TOL = 1e-3

# --------------------------------------------------------------------------- #
# §23.6 / §23.8 -- the published v2 figures, carried ONLY to be diffed against
# the values this module re-derives. Never used as an input to a verdict.
# --------------------------------------------------------------------------- #

LIF_QUOTED_GA_MEAN = 0.148469                                   # §23.6
LIF_QUOTED_GA_PER_SEED = (0.161538, 0.126194, 0.157675)         # §23.6
LIF_QUOTED_GA2_MEAN = 0.309570                                  # §23.6
LIF_QUOTED_GA2_PER_SEED = (0.328979, 0.282715, 0.317017)        # §23.6
LIF_QUOTED_GB_MEAN = 35.4972                                    # §23.6
LIF_QUOTED_GB_PER_SEED = (33.806, 34.842, 37.844)               # §23.6
GRU_QUOTED_GB_MEAN = 39.7861                                    # §23.6
GRU_QUOTED_GB_PER_SEED = (40.904, 44.842, 33.612)               # §23.6
QUOTED_SEED_MATCHED_GAP = -4.2889                               # §23.6
LIF_QUOTED_TRAIN_REWARD = 0.078594                              # §23.8
LIF_QUOTED_TRAIN_REWARD_PER_SEED = (0.073465, 0.089327, 0.072991)   # §23.8
GRU_QUOTED_TRAIN_REWARD = 0.112723                              # §23.8
QUOTED_TOTAL_UPDATES = 7813                                     # §23.8

# §23.6, verbatim. Required to appear in the output, so it lives here as one
# string rather than as prose scattered through the report.
UNDERPOWERED_DISCLAIMER = (
    "GB is declared underpowered here, before it is measured. At n=3 versus n=3 "
    "the exact two-sided permutation test's resolution floor is 2/C(6,3) = 0.1, "
    "so no significance claim can be made from this pilot in either direction and "
    "none will be. The per-seed sign test (3/3 has probability 0.125 under the "
    "null) is reported as a supporting statistic only. No number produced by this "
    "pilot may be quoted as a Phase 1 result, and none of it belongs in "
    "docs/RESULTS.md."
)

# --------------------------------------------------------------------------- #
# §23.7 -- the four decision strings, fixed in advance
# --------------------------------------------------------------------------- #

DECISION_SCALE_UP = "SCALE-UP RECOMMENDED"
DECISION_INFORMATIVE_NEGATIVE = (
    "STOP -- mechanism confirmed, task performance flat (informative negative)")
DECISION_NOT_CONFIRMED = "STOP -- mechanism not confirmed"
DECISION_AMBIGUOUS = "STOP -- ambiguous"

GA_VERDICTS = ("CONFIRMED", "FALSIFIED", "AMBIGUOUS")
GB_VERDICTS = ("PROMISING", "NOT PROMISING", "AMBIGUOUS")

# §23.8 requires `final`/`continuous` NOT be the only cell reported, "so the pilot
# cannot be read only through its most favourable cell".
EVAL_SELECTIONS = ("final", "best")
EVAL_REGIMES = ("continuous", "reset128")
GB_SELECTION, GB_REGIME = "final", "continuous"      # §23.6 fixes GB's cell

# --------------------------------------------------------------------------- #
# --stage spectrum -- the direct measurement §23.12 inferred instead
# --------------------------------------------------------------------------- #

# WELCH, AND WHY NOT ONE FFT OF THE WHOLE WINDOW. A single periodogram of all
# 6,000 samples is an inconsistent estimator: its variance is O(S(f)^2) and does
# NOT fall as the record lengthens -- a longer record buys more frequency bins,
# each one just as noisy. Welch trades resolution for variance by averaging the
# periodograms of overlapping windowed segments, which is the standard fix and is
# about fifteen lines of numpy (`welch_psd` below). No scipy is added: this repo's
# `requirements.txt` carries numpy and not scipy, and
# `analysis/aggregate_results.py` already sets the precedent of writing the small
# numerics by hand rather than growing the dependency set for them.
#
# SEGMENT LENGTH 512, HALF-OVERLAPPED, PERIODIC HANN. The reported bands run to
# T >= 64 and the tradeoff grid below runs to T = 128, so the estimator has to
# resolve periods out to 128 steps and place more than one bin there; at 512 a
# period of 128 steps is bin 4, and the NARROWEST reported band (32 <= T < 64)
# still gets 8 resolved bins. Going shorter starves the slow end (at 256, the
# T >= 64 band is four bins wide and cannot see past period 256); going longer
# starves the averaging (at 2048, four segments, and the variance reduction Welch
# exists for is mostly gone). 512 with 50% overlap gives 22 segments over the
# 6,000-step fixture. The window is the PERIODIC Hann (`np.hanning(N+1)[:-1]`),
# which is the one spectral estimation wants; the symmetric `np.hanning(N)` is
# for FIR design and leaks slightly more.
WELCH_SEGMENT_LEN = 512
WELCH_OVERLAP = 0.5

# §23.2's beta, restated here for the ANALYTIC tables. Every measured number in
# the spectrum stage takes beta from the reservoir it just built (which is a
# float32 buffer, i.e. 0.8999999761581421 and not 0.9), and the stage asserts the
# two agree -- the instrument rule applied to a constant.
RF_BETA = 0.9

# The init the spectrum stage measures at: §23.12's selected calibration point,
# which is what a follow-up construction would be correcting. Overridable with
# --embed-scale so the same tables can be taken at any other point on §23.4's
# grid without editing code.
SPECTRUM_EMBED_SCALE = 3.32

# The uncentred v1 init, carried ONLY as the published cross-check for the
# input-current DC fraction (RESULTS.md v1 §7.1's 76.11%). Never a measurement
# condition of this pilot -- v2 and the rf arm both train `centered`.
LEGACY_EMBED_MODE, LEGACY_EMBED_SCALE = "legacy", 1.0

# The period bands the observation and input-current spectra are reported over.
# Half-open [lo, hi) so every bin lands in exactly one band and the fractions sum
# to 1; the last band is unbounded above and absorbs everything slower.
PERIOD_BAND_EDGES = (0.0, 4.0, 8.0, 16.0, 32.0, 64.0, math.inf)
PERIOD_BAND_LABELS = ("T<4", "4-8", "8-16", "16-32", "32-64", ">=64")

# §23.2's own support, bucketed by octave. These are the bins §23.12's scratchpad
# finding was found in, promoted here so it is reproducible from a committed
# module instead of from a diagnostic nobody kept.
RF_PERIOD_BIN_EDGES = (2.0, 4.0, 8.0, 16.0, 32.0)

# `envs/mario_land_env._build_observation`'s slot order, for a readable table.
# Slots 9-11 are the documented reserved zeros (RESULTS.md v1 §9): they carry no
# spectrum at all and are FLAGGED and excluded rather than allowed to contribute
# a silent NaN to a pooled fraction.
OBS_SLOT_NAMES = ("progress_delta", "y", "vel_x", "vel_y", "on_ground", "timer",
                  "lives", "powerup", "score_delta",
                  "reserved_0", "reserved_1", "reserved_2")
RESERVED_ZERO_SLOTS = (9, 10, 11)                       # RESULTS.md v1 §9

# The two published figures the spectrum stage re-derives and diffs against.
RESULTS_QUOTED_OBS_DC_FRACTION = 0.7770                 # RESULTS.md v1 §7.1
RESULTS_QUOTED_OBS_MEAN_ENERGY = 1.331336               # RESULTS.md v1 §7.1
RESULTS_QUOTED_OBS_TOTAL_ENERGY = 1.713384              # RESULTS.md v1 §7.1
RESULTS_QUOTED_CURRENT_DC_FRACTION = 0.7611             # RESULTS.md v1 §7.1

# §23.4's tradeoff grid: T_max swept at the pre-registered T_min, then T_min swept
# at the widest T_max. Deliberately MODEST and deliberately not optimised over --
# it is a table, not a search, and nothing downstream reads it.
TRADEOFF_T_MIN_FIXED = 2.0
TRADEOFF_T_MAX_GRID = (8.0, 16.0, 32.0, 64.0, 128.0)
TRADEOFF_T_MAX_FIXED = 128.0
TRADEOFF_T_MIN_GRID = (4.0, 8.0, 16.0)

# Printed verbatim under §4's table. One string, so the refusal cannot drift.
NO_BAND_RECOMMENDATION = (
    "NO BAND IS RECOMMENDED AND NO BAND IS PICKED. The table above is the "
    "tradeoff, not a conclusion drawn from it: DC attenuation wants high "
    "frequencies and drive wants wherever the signal actually is, and those pull "
    "in opposite directions over this grid. Choosing a band is a NEW "
    "pre-registration -- it fixes a construction before its numbers exist, which "
    "is the whole content of §23 -- and choosing one here, after reading the "
    "measurement, would be the post-hoc tuning §23.12 already refused once when "
    "it declared its own grid refinement the only one."
)


# --------------------------------------------------------------------------- #
# PURE LOGIC -- no I/O, no torch, fully covered by tests/test_rf_pilot.py
# --------------------------------------------------------------------------- #

def ga_verdict(mean_spike_rate: float) -> str:
    """§23.6's GA band, three-way.

      CONFIRMED : mean in [0.005, 0.050]  -- CLOSED at both ends ("mean in [a, b]")
      FALSIFIED : mean >= 0.100 (less than half the distance from LIF back to the
                  band) OR mean <= 0.002 (starved). Both boundaries closed, per
                  "at or above" / "at or below".
      AMBIGUOUS : otherwise -- the two open gaps (0.002, 0.005) and (0.050, 0.100).

    The bands are disjoint (asserted at import), so the branch order below cannot
    change any answer; it follows §23.6's own order of presentation.
    """
    if GA_CONFIRMED_LOW <= mean_spike_rate <= GA_CONFIRMED_HIGH:
        return "CONFIRMED"
    if (mean_spike_rate >= GA_FALSIFIED_AT_OR_ABOVE
            or mean_spike_rate <= GA_FALSIFIED_AT_OR_BELOW):
        return "FALSIFIED"
    return "AMBIGUOUS"


def ga2_verdict(mean_silent_fraction: float) -> str:
    """§23.6's GA2 band: CONFIRMED < 15%, FALSIFIED >= 25%, AMBIGUOUS between.

    That is exactly the shape `band_verdict` already implements for A7 and A9, so
    it is REUSED rather than re-expressed. Two copies of a three-way band drift at
    a boundary and nothing notices, which is the failure the pre-registration
    exists to make impossible.
    """
    return band_verdict(mean_silent_fraction, GA2_CONFIRMED_BELOW,
                        GA2_FALSIFIED_AT_OR_ABOVE)


def gb_verdict(mean_extrinsic_return: float) -> str:
    """§23.6's GB band: PROMISING >= 36.9268, NOT PROMISING <= 35.4972, else
    AMBIGUOUS. BOTH boundaries are closed -- unlike A7/A9/GA2's half-open band --
    because §23.6 words them "at or above" and "no improvement on LIF at all",
    which is why this cannot go through `band_verdict`."""
    if mean_extrinsic_return >= GB_PROMISING_AT_OR_ABOVE:
        return "PROMISING"
    if mean_extrinsic_return <= GB_NOT_PROMISING_AT_OR_BELOW:
        return "NOT PROMISING"
    return "AMBIGUOUS"


def gb_threshold_from_data(lif_mean: float, baseline_mean: float) -> float:
    """§23.6's PROMISING threshold, re-derived: close at least one third of the
    seed-matched gap between the LIF arm and the GRU baseline."""
    return lif_mean + (baseline_mean - lif_mean) / 3.0


def gb_threshold_agrees(derived: float,
                        preregistered: float = GB_PROMISING_AT_OR_ABOVE,
                        tol: float = GB_THRESHOLD_AGREEMENT_TOL) -> bool:
    """Does the threshold re-derived from `results_v2/` match the pre-registered
    constant closely enough to trust the data handling?

    THIS IS A GUARD, NOT A CORRECTION. The pre-registered constant is the one the
    verdict uses, always, and a disagreement never silently substitutes the
    derived value: §23.11 records that §23.6's GB block carries a fourth-decimal
    rounding slip against its own per-seed inputs, and "a fourth-decimal
    correction made afterwards is exactly the kind of adjustment a
    pre-registration exists to prevent". What this catches is a data-handling
    error -- the wrong seeds, the wrong regime, the wrong selection read out of
    `results_v2/` -- which moves the derived threshold far more than a rounding
    slip does.
    """
    return abs(derived - preregistered) <= tol


def decision_rule(ga: str, gb: str) -> str:
    """§23.7's decision rule, clause for clause.

      SCALE-UP RECOMMENDED  <=>  GA CONFIRMED and GB PROMISING. A BICONDITIONAL:
                                 exactly one of the nine cells may recommend it.
      GA FALSIFIED          =>   stop, "whatever GB says".
      GA CONFIRMED and GB NOT PROMISING => the informative negative, written down
                                 in advance "precisely so it cannot later be
                                 reframed as a disappointment".
      any other combination =>   stop and report.

    GA2 IS NOT AN INPUT, and that is §23.7's wording rather than an oversight:
    the rule as pre-registered conditions on GA and GB only. GA2 is co-primary
    EVIDENCE about the same mechanism and is reported in full next to GA -- but
    silently folding it into the rule would be exactly the kind of after-the-fact
    adjustment §23 exists to prevent. If the two primaries disagree, that is a
    finding to report, not a tie to break in code.

    An unrecognised verdict string raises rather than falling through to
    "ambiguous", which would read as a real verdict.
    """
    if ga not in GA_VERDICTS:
        raise ValueError(f"unknown GA verdict {ga!r}; expected one of {GA_VERDICTS}")
    if gb not in GB_VERDICTS:
        raise ValueError(f"unknown GB verdict {gb!r}; expected one of {GB_VERDICTS}")
    if ga == "FALSIFIED":
        return DECISION_NOT_CONFIRMED
    if ga == "CONFIRMED":
        if gb == "PROMISING":
            return DECISION_SCALE_UP
        if gb == "NOT PROMISING":
            return DECISION_INFORMATIVE_NEGATIVE
    return DECISION_AMBIGUOUS


def select_embed_scale(rates, lif_rate):
    """§23.4's calibration selector. `rates` is `[(scale, mean_rate), ...]` in the
    pre-registered grid's own ascending order; returns `(scale, criterion)`, or
    `(None, None)` if nothing is selectable.

    THE CRITERION IS A LOG RATIO, not an absolute difference: §23.4 fixes
    `|log(rate_RF / rate_LIF_v2_init)|`, which is symmetric in the multiplicative
    sense -- half the reference rate and twice it are equally far from it. That is
    the right metric for a quantity whose healthy band is itself specified
    multiplicatively (§23.5's [0.005, 0.050] spans one order of magnitude).

    IT SEES NOTHING ABOUT TASK REWARD. §23.4: "The selection criterion is a
    construction measurement on a fixed fixture and can see nothing about task
    reward. That is what distinguishes it from a hyperparameter search on the
    outcome metric."

    IT ALSO SEES NOTHING ABOUT G0b's BAND, deliberately. Selection (§23.4) and the
    healthy-operating-point gate (§23.5's G0b) are separate steps: the selector
    returns the closest scale whatever its rate, and G0b then decides whether that
    rate is acceptable. Folding the band into the selector would make "no grid
    value lands in band" unreportable, and §23.5 says that outcome "is itself a
    reportable finding about the mechanism".

    A candidate whose rate is zero (or non-finite) is SKIPPED rather than ranked:
    the log ratio is undefined there, and a reservoir that never fires is not
    "infinitely far" from the reference in any useful sense -- it is unmeasurable
    by this criterion, which the report says rather than hiding behind an infinity.
    A tie is resolved to the EARLIER grid value by the strict `<` below, i.e. to
    the smaller scale; the grid's ascending written order fixes that rule in
    advance rather than leaving it to be chosen once a tie is seen.
    """
    if lif_rate is None or not math.isfinite(lif_rate) or lif_rate <= 0.0:
        return None, None
    best_scale, best_criterion = None, None
    for scale, rate in rates:
        if rate is None or not math.isfinite(rate) or rate <= 0.0:
            continue
        criterion = abs(math.log(rate / lif_rate))
        if best_criterion is None or criterion < best_criterion:
            best_scale, best_criterion = scale, criterion
    return best_scale, best_criterion


def sign_test_wins(rf_per_seed, reference_per_seed):
    """How many seeds' rf value strictly beats its OWN reference counterpart.

    Seed-matched and not rank-based: §23.6 asks for "how many of the pilot seeds
    beat their own LIF counterpart", which is the only paired statistic three
    seeds support. Returns `(wins, n_compared)`; seeds missing on either side are
    not compared and not counted.
    """
    wins = compared = 0
    for rf, ref in zip(rf_per_seed, reference_per_seed):
        if rf is None or ref is None:
            continue
        compared += 1
        wins += int(rf > ref)
    return wins, compared


def mean_or_none(values):
    """Mean of the non-None entries, or None if there are none. Every aggregate in
    this module goes through it, so a partially-present matrix produces a smaller
    honest number instead of a crash or a zero."""
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


# --------------------------------------------------------------------------- #
# SPECTRUM -- PURE LOGIC. numpy only, no torch, no I/O, no new dependency.
# Covered by tests/test_rf_pilot.py on synthetic signals and closed forms.
# --------------------------------------------------------------------------- #

def welch_psd(x, segment_len=WELCH_SEGMENT_LEN, overlap=WELCH_OVERLAP):
    """Welch's estimate of the one-sided power spectral density of `x`.

    `x` is `(T,)` or `(T, D)` and is treated as D independent real channels
    sampled once per env step, so frequency is in CYCLES PER ENV STEP and a bin
    at frequency f corresponds to a period of 1/f steps. Returns
    `(freqs, psd, n_segments)` with `psd` of shape `(segment_len//2 + 1, D)`.

    WHY WELCH AND NOT A SINGLE PERIODOGRAM OF ALL 6,000 SAMPLES. The periodogram
    is asymptotically unbiased but NOT consistent: at every frequency its
    variance is on the order of the true PSD squared, and it stays there however
    long the record gets -- a longer record buys finer bins, each one exactly as
    noisy as before. Welch splits the record into overlapping windowed segments
    and averages their periodograms, which trades frequency resolution for a
    variance that falls roughly like 1/n_segments. Reading a band fraction off one
    6,000-point FFT would be reading a number with a ~100% standard error per bin.

    THE NORMALISATION IS THE ONE THAT MAKES `psd.sum() * df` THE VARIANCE.
    Each segment's periodogram is divided by the window's power `sum(w^2)`, and
    every bin except DC and (for an even segment length) Nyquist is doubled to
    fold the negative frequencies in. `tests/test_rf_pilot.py` pins this by
    integrating the estimate back to the signal's own variance rather than
    trusting the algebra, because the doubling and the window normalisation are
    exactly the two places a hand-rolled Welch goes quietly wrong.

    WHAT BIN 0 IS, AND WHY IT IS NOT THE SIGNAL'S DC. `x` is expected to be
    centred over the WHOLE window before it gets here, but each segment still has
    its own local mean, so bin 0 collects the power at periods LONGER than one
    segment -- real low-frequency energy this estimator cannot resolve into a
    band. It is excluded from every band fraction below and reported separately
    (`unresolved_slow_fraction`) rather than being folded into the slowest band,
    where it would be indistinguishable from resolved power.

    THE WINDOW IS THE PERIODIC HANN, `np.hanning(N+1)[:-1]`. numpy's own
    `np.hanning(N)` is the symmetric variant, which is what FIR design wants and
    leaks slightly more when used for spectral estimation.

    Segments are taken while a whole one fits; a tail shorter than `segment_len`
    is dropped rather than zero-padded, since padding would inject an artificial
    taper into exactly the low-frequency bins this measurement turns on.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"welch_psd expects (T,) or (T, D); got shape {x.shape}")
    n_steps = x.shape[0]
    segment_len = int(segment_len)
    if segment_len < 2:
        raise ValueError(f"segment_len must be at least 2; got {segment_len}")
    if n_steps < segment_len:
        raise ValueError(f"need at least segment_len={segment_len} samples; got {n_steps}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1); got {overlap}")
    hop = max(1, int(round(segment_len * (1.0 - overlap))))
    window = np.hanning(segment_len + 1)[:-1]
    window_power = float((window * window).sum())

    accumulated = np.zeros((segment_len // 2 + 1, x.shape[1]), dtype=np.float64)
    n_segments = 0
    for start in range(0, n_steps - segment_len + 1, hop):
        segment = x[start:start + segment_len].astype(np.float64) * window[:, None]
        spectrum = np.fft.rfft(segment, axis=0)
        accumulated += spectrum.real ** 2 + spectrum.imag ** 2
        n_segments += 1
    psd = accumulated / (n_segments * window_power)
    # Fold the negative frequencies in. DC (bin 0) and, when segment_len is even,
    # Nyquist (the last bin) have no mirror partner and are NOT doubled.
    last = -1 if segment_len % 2 == 0 else len(psd)
    psd[1:last] *= 2.0
    return np.fft.rfftfreq(segment_len, d=1.0), psd, n_segments


def bin_periods(freqs):
    """Period in env steps for each frequency bin, `inf` for the DC bin."""
    freqs = np.asarray(freqs, dtype=np.float64)
    with np.errstate(divide="ignore"):
        return np.where(freqs > 0.0, 1.0 / np.where(freqs > 0.0, freqs, 1.0), np.inf)


def period_bin_index(period, edges):
    """Index of the bin `[edges[i], edges[i+1])` containing `period`, or None.

    Half-open at the top so a period landing exactly on an internal edge goes to
    the FASTER bin's successor -- T = 4 is in `[4, 8)`, never in `[2, 4)` -- which
    is the convention `PERIOD_BAND_EDGES` and `RF_PERIOD_BIN_EDGES` are both
    written in. THE LAST BIN IS CLOSED AT THE TOP: `RF_PERIOD_BIN_EDGES` ends at
    §23.2's support endpoint T = 32, and a draw landing exactly there must be
    counted rather than silently dropped out of a table whose counts are supposed
    to sum to the reservoir size.
    """
    period = float(period)
    if not math.isfinite(period) and period > 0 and math.isinf(edges[-1]):
        return len(edges) - 2
    if period < edges[0] or period > edges[-1]:
        return None
    for i in range(len(edges) - 1):
        if period < edges[i + 1]:
            return i
    return len(edges) - 2


def band_bin_counts(freqs, edges=PERIOD_BAND_EDGES):
    """How many RESOLVED (non-DC) frequency bins fall in each period band.

    Reported next to the band fractions because the bands are octaves in PERIOD
    and the bins are uniform in FREQUENCY, so the bands are wildly unequal in
    width: at segment length 512 the `T<4` band is 128 bins wide and `32-64` is
    8. A band fraction alone therefore says nothing about spectral DENSITY, which
    is the quantity a narrowband resonator actually integrates -- see
    `band_power_density`.
    """
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    for period in bin_periods(freqs)[1:]:
        index = period_bin_index(period, edges)
        if index is not None:
            counts[index] += 1
    return counts


def band_power_fractions(freqs, psd, edges=PERIOD_BAND_EDGES):
    """Fraction of the RESOLVED fluctuating power in each period band.

    `psd` is one channel's spectrum, or a pooled one already summed over
    channels; bin 0 is excluded (see `welch_psd`). The returned fractions sum to
    1 by construction -- `PERIOD_BAND_EDGES` covers `[0, inf)`, so no resolved bin
    can fall outside it -- which is what makes them readable as a decomposition.
    Returns all-NaN for a channel with no power at all, which is a dimension with
    no spectrum rather than a dimension whose spectrum is flat, and the report
    must say which.
    """
    psd = np.asarray(psd, dtype=np.float64)
    if psd.ndim != 1:
        raise ValueError(f"band_power_fractions expects one channel; got shape {psd.shape}")
    ac = psd[1:]
    total = ac.sum()
    out = np.zeros(len(edges) - 1, dtype=np.float64)
    if not total > 0.0:
        return np.full(len(edges) - 1, np.nan)
    for period, power in zip(bin_periods(freqs)[1:], ac):
        index = period_bin_index(period, edges)
        if index is not None:
            out[index] += power
    return out / total


def band_power_density(fractions, counts):
    """Band power per resolved bin, in units of the whole spectrum's mean.

    1.0 is a spectrum flat in frequency. THIS, NOT THE BAND FRACTION, IS WHAT A
    RESONATOR SEES: a unit at frequency w integrates the input PSD against a
    bandpass of width set by |lambda| = beta and not by the octave the bin is
    filed under, so what reaches it scales with power DENSITY near w. An octave
    at fast periods spans eight times as much frequency as an octave four octaves
    slower, so a band fraction flatters the fast end by exactly that factor and a
    band chosen off fractions alone would be chosen off the wrong statistic.
    """
    fractions = np.asarray(fractions, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(counts > 0, fractions * total / np.where(counts > 0, counts, 1.0),
                        np.nan)


def band_power_fraction_in(freqs, psd, period_min, period_max):
    """Fraction of the resolved fluctuating power at periods in `[T_min, T_max]`.

    CLOSED at both ends, unlike `band_power_fractions`' half-open bands: §23's
    frequency support is written `T in [2, 32]` with both endpoints meant, and
    the tradeoff table's rows are candidate SUPPORTS rather than a partition, so
    two rows sharing an endpoint are allowed to share the bin that sits on it.
    """
    psd = np.asarray(psd, dtype=np.float64)
    if psd.ndim != 1:
        raise ValueError(f"band_power_fraction_in expects one channel; got shape {psd.shape}")
    ac = psd[1:]
    total = ac.sum()
    if not total > 0.0:
        return float("nan")
    periods = bin_periods(freqs)[1:]
    inside = (periods >= period_min) & (periods <= period_max)
    return float(ac[inside].sum() / total)


def unresolved_slow_fraction(psd):
    """Share of the Welch power sitting in bin 0, i.e. at periods longer than one
    segment. Reported so the cost of the segment length is visible instead of
    being quietly redistributed into the slowest resolved band."""
    psd = np.asarray(psd, dtype=np.float64)
    total = psd.sum()
    return float(psd[0] / total) if total > 0.0 else float("nan")


def dc_power_fraction(x, chunk=4096):
    """`(per_channel, pooled)` share of total power carried by the MEAN.

    Per channel this is `mean^2 / mean(x^2)`; pooled it is `||E x||^2 / E||x||^2`,
    which is the exact quantity RESULTS.md v1 §7.1 published as 77.70% for the
    observation and 76.11% for the reservoir's input current. A channel that is
    identically zero has no power at all and reads NaN rather than 0 or 1 -- the
    three reserved-zero observation slots are exactly that case, and a NaN the
    report has to handle is better than a 0.0 that silently averages in.

    Accumulated over row chunks in float64 because the caller passes an
    `(6000, 8192)` input-current matrix in float32, and both materialising it at
    float64 and summing six thousand float32 terms pairwise are avoidable.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    n_steps = x.shape[0]
    total = np.zeros(x.shape[1], dtype=np.float64)
    square = np.zeros(x.shape[1], dtype=np.float64)
    for start in range(0, n_steps, chunk):
        block = x[start:start + chunk].astype(np.float64)
        total += block.sum(axis=0)
        square += (block * block).sum(axis=0)
    mean_power = (total / n_steps) ** 2
    mean_square = square / n_steps
    with np.errstate(divide="ignore", invalid="ignore"):
        per_channel = np.where(mean_square > 0, mean_power / np.where(
            mean_square > 0, mean_square, 1.0), np.nan)
    pooled_denominator = mean_square.sum()
    pooled = (float(mean_power.sum() / pooled_denominator)
              if pooled_denominator > 0 else float("nan"))
    return per_channel, pooled


def dc_gain_at_period(period, beta=RF_BETA):
    """`|1/(1 - beta*e^{i*2*pi/T})|`, the §23.3 quantity, at one period.

    `period = inf` IS omega = 0, i.e. the LIF point of §23.2's family, and the
    branch returns `1/(1-beta)` from that expression directly rather than letting
    the general one evaluate at cos(w) = 1. Measured, the two agree bit for bit at
    every beta this project uses -- but that is how the rounding happens to fall,
    not a property, and §23.2's claim that LIF is the w = 0 point of this family
    BIT FOR BIT is the one claim in the pilot that may not rest on a coincidence
    of IEEE-754. `analysis/reservoir_health.dc_offset_factor` carries the same
    explicit branch, so taking the LIF point from the LIF formula in both places
    is also what keeps the two modules from drifting apart at w = 0.

    NOTE that `1/(1-beta)` at beta = 0.9 is 10.000000000000002 in float64, not the
    literal 10.0 that §23.3 prints. Every LIF reference in this repo is that
    expression, and `tests/test_rf_pilot.py` pins this function to it rather than
    to the printed constant.
    """
    beta = float(beta)
    if period <= 0.0:
        raise ValueError(f"period must be positive; got {period}")
    if math.isinf(period):
        return 1.0 / (1.0 - beta)
    omega = 2.0 * math.pi / period
    real = 1.0 - beta * math.cos(omega)
    imaginary = beta * math.sin(omega)
    return 1.0 / math.sqrt(real * real + imaginary * imaginary)


def mean_dc_gain_log_uniform(period_min, period_max, beta=RF_BETA, n_samples=20000):
    """Mean of `dc_gain_at_period` over T ~ log-uniform on `[T_min, T_max]`.

    This is §23.3's headline construction number as a FUNCTION of the band rather
    than as one row: at `[2, 32]` it reproduces the pre-registered 1.7846, and the
    tradeoff table below evaluates it over candidate bands nobody has committed
    to. Log-uniform because §23.2 draws that way -- "equal density per octave
    across the five octaves spanned, the standard spacing for a filter bank" --
    so the mean is a plain average over `u = log T` uniform, taken by the midpoint
    rule. The integrand is smooth and bounded on any band with `T_min >= 2`, so
    the midpoint rule at 20,000 intervals is converged far past the four decimals
    anything here prints; it is quadrature rather than a closed form because
    `|1/(1 - beta e^{i 2 pi / T})|` has no elementary antiderivative in log T.

    A DEGENERATE BAND `T_min == T_max` is the single-frequency case and returns
    the pointwise gain, which at `T = inf` is `1/(1-beta)` -- omega = 0, i.e. the
    LIF control arm, the row §4's table is read against. An unbounded band raises:
    log-uniform on `[T_min, inf)` is not a distribution, and returning the limit
    1/(1-beta) instead would report a mean over a band nobody could draw from.
    """
    period_min, period_max = float(period_min), float(period_max)
    if period_min <= 0.0 or period_max <= 0.0:
        raise ValueError(f"periods must be positive; got [{period_min}, {period_max}]")
    if period_min > period_max:
        raise ValueError(f"need T_min <= T_max; got [{period_min}, {period_max}]")
    if period_min == period_max:
        return dc_gain_at_period(period_min, beta)
    if math.isinf(period_max):
        raise ValueError("log-uniform on an unbounded period band is not a distribution")
    beta = float(beta)
    edges = np.linspace(math.log(period_min), math.log(period_max), int(n_samples) + 1)
    periods = np.exp(0.5 * (edges[1:] + edges[:-1]))
    omega = 2.0 * np.pi / periods
    real = 1.0 - beta * np.cos(omega)
    imaginary = beta * np.sin(omega)
    return float(np.mean(1.0 / np.sqrt(real * real + imaginary * imaginary)))


def ac_gain(beta=RF_BETA):
    """`1/sqrt(1-beta^2)`, the accumulation gain for a zero-mean input. Identical
    in both neuron models by construction (§23.2): it depends on the pole
    MAGNITUDE only, which the rf swap holds fixed."""
    beta = float(beta)
    return 1.0 / math.sqrt(1.0 - beta * beta)


def resonant_u_response_var(freqs, psd, omega, beta=RF_BETA, chunk=1024):
    """Predicted variance of each unit's membrane `u` under a measured input PSD.

    §23.2's state is complex: `z = u + i v` obeys `z_t = beta*e^{i w} z_{t-1} + I_t`
    with the input entering the real part only, so `u_t = sum_k beta^k cos(k w) I_{t-k}`
    and the transfer function from input to `u` is

        H_u(W) = 0.5 * [ 1/(1 - beta e^{i(w - W)}) + 1/(1 - beta e^{-i(w + W)}) ]

    -- one term for each of the conjugate poles. The variance is then
    `sum_bins |H_u|^2 * psd * df` over the resolved bins, per unit, using each
    unit's OWN input spectrum (every unit sees a different row of `W_in`, hence a
    different mixture of the embedding's twelve-dimensional drive).

    WHAT THIS IS FOR, AND WHAT IT IS NOT. It converts "here is the input
    spectrum" into "here is what a unit at period T_i actually receives", which is
    the step between §1/§2's spectra and §3's measured silence. It is LINEARISED
    and INPUT-ONLY: it ignores the frozen recurrent TT drive, the threshold, and
    the reset, so it predicts the scale of the membrane's excursions and NOT the
    spike rate. It gates nothing. The measured spike statistics next to it are the
    real measurement; this column exists to say why they come out the way they do.

    Bin 0 is excluded for the same reason it is everywhere else: it is the
    standing offset, which shifts the threshold rather than driving excursions,
    and `induced_membrane_offset` is the quantity that reports it.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64).reshape(-1)
    if psd.ndim != 2 or psd.shape[1] != omega.shape[0]:
        raise ValueError(f"psd shape {psd.shape} does not match {omega.shape[0]} units")
    beta = float(beta)
    ac = psd[1:]
    band = 2.0 * np.pi * freqs[1:]
    df = float(freqs[1] - freqs[0])
    out = np.empty(omega.shape[0], dtype=np.float64)
    for start in range(0, omega.shape[0], chunk):
        w = omega[start:start + chunk][None, :]
        pole = 0.5 * (1.0 / (1.0 - beta * np.exp(1j * (w - band[:, None])))
                      + 1.0 / (1.0 - beta * np.exp(-1j * (w + band[:, None]))))
        gain = pole.real ** 2 + pole.imag ** 2
        out[start:start + chunk] = (gain * ac[:, start:start + chunk]).sum(axis=0) * df
    return out


# --------------------------------------------------------------------------- #
# ON-DISK READERS -- all optional, all report rather than raise
# --------------------------------------------------------------------------- #

def resolve_dir(path):
    """Repo-relative unless already absolute, matching reservoir_health's rule."""
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def read_eval(results_dir, selection, arm, seed, regime):
    """One evaluation JSON, or None. Filename shape is the strict one
    `analysis/aggregate_results.py` matches (`eval_{arm}_seed{n}_{regime}.json`),
    so a directory that does not follow it reads as absent rather than as zero."""
    path = os.path.join(results_dir, selection, f"eval_{arm}_seed{seed}_{regime}.json")
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def eval_returns(results_dir, arm, seeds, selection=GB_SELECTION, regime=GB_REGIME):
    """`mean_extrinsic_return` per seed, `None` where the JSON is absent."""
    out = []
    for seed in seeds:
        record = read_eval(results_dir, selection, arm, seed, regime)
        out.append(None if record is None else float(record["mean_extrinsic_return"]))
    return out


def train_log_extrinsic_mean(checkpoint_dir, arm, seed):
    """Mean per-update extrinsic TRAINING reward over every update the run logged,
    and the update count it was taken over -- `(None, 0)` if the log is absent.

    §23.8 requires this be reported as a DIAGNOSTIC and explicitly NOT as a gate:
    v2's sharpest single contrast was that the corrections closed the
    training-reward gap from 5.82x to 1.38x while the evaluation gap did not close
    at all, so training reward is precisely the quantity already shown not to
    predict the scoreboard here. The update count is returned alongside so a mean
    over a partial run is never mistaken for a mean over all 7,813.
    """
    path = os.path.join(run_dir(checkpoint_dir, arm, seed), "train_log.jsonl")
    if not os.path.isfile(path):
        return None, 0
    values = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "mean_extrinsic_reward" in record:
                values.append(float(record["mean_extrinsic_reward"]))
    if not values:
        return None, 0
    return sum(values) / len(values), len(values)


# --------------------------------------------------------------------------- #
# CONSTRUCTION MEASUREMENTS (preflight)
# --------------------------------------------------------------------------- #

def load_fixture():
    """The committed 6,000-step real-observation window every construction
    measurement in §23 is taken on, plus the observation mean the centred init is
    defined against."""
    obs = torch.as_tensor(np.load(REAL_OBS_PATH), dtype=torch.float32)
    mu = torch.tensor(OBS_MEAN, dtype=torch.float32)
    return obs, mu


def frequency_construction(seed):
    """G0a's two statistics for one seed, read off a freshly-built rf reservoir at
    the production geometry.

    Built through `reservoir_at`, i.e. through `build_model`, so the geometry is
    literally the one training uses rather than a re-specification of it -- and
    asserted against §23.2's numbers here, so a change to `build_model` moves the
    gate loudly instead of silently.
    """
    model, _ = reservoir_at(seed, RF_INIT_EMBED_MODE, LIF_INIT_EMBED_SCALE,
                            neuron_model="rf")
    reservoir = model.reservoir
    assert reservoir.reservoir_size == RESERVOIR_SIZE, (
        f"G0a is defined at the production geometry (§23.2): expected "
        f"reservoir_size={RESERVOIR_SIZE}, built {reservoir.reservoir_size}")
    assert reservoir.tt_ranks[1] == TT_RANK and reservoir.tt_n_cores == TT_N_CORES, (
        f"G0a is defined at tt_rank={TT_RANK}, tt_n_cores={TT_N_CORES} (§23.2); "
        f"built tt_rank={reservoir.tt_ranks[1]}, tt_n_cores={reservoir.tt_n_cores}")
    dc = float(reservoir.dc_gain().mean())
    ac = reservoir.ac_gain()
    return {"seed": seed, "mean_dc_gain": dc, "ac_gain": ac, "dc_over_ac": dc / ac,
            "omega_min": float(reservoir.omega.min()),
            "omega_max": float(reservoir.omega.max())}


def init_operating_point(seed, embed_scale, neuron_model, obs, mu=None):
    """The step-0 operating point for one seed at one embed scale, in the SAME
    columns `checkpoint_operating_point` reports so an init row can head a
    trajectory table (§23.8) rather than sitting in a separate one.

    The construction sequence is `reservoir_at`'s, which is `run_training`'s:
    `torch.manual_seed(seed)` then `build_model(..., seed=seed, ...)`, because the
    trainable init draws from the GLOBAL RNG while the frozen reservoir takes its
    own `seed=`, and only doing both reproduces what `--seed s` actually produced.
    Mirroring it here rather than re-deriving it is the point -- §23.4's criterion
    is defined against "the LIF v2 arm's initial spike rate on the same fixture,
    MEASURED BY THE SAME INSTRUMENT".

    `mu` is optional because the calibration only needs the spike statistics; pass
    it to get the weight-norm and induced-offset columns too, which at a `centered`
    init are the zero point §23.1's drift table is read against.
    """
    model, _ = reservoir_at(seed, RF_INIT_EMBED_MODE if neuron_model == "rf"
                            else LIF_INIT_EMBED_MODE, embed_scale,
                            neuron_model=neuron_model)
    silent, rate, saturated = silent_fraction(model, obs)
    row = {"silent": silent, "spike_rate": rate, "saturated": saturated,
           "neuron_model": neuron_model, "frozen_drift": None}
    if mu is None:
        return row
    weight = model.embedding.weight.detach()
    bias = model.embedding.bias.detach()
    dc = weight @ mu + bias
    beta = float(model.reservoir.lif.beta)
    omega = model.reservoir.omega if neuron_model == "rf" else None
    offset = induced_membrane_offset(model.reservoir.W_in @ dc, beta, omega)
    row.update({"w_norm": weight.norm().item(), "dc_norm": dc.norm().item(),
                "offset_std": offset.std().item()})
    return row


def input_current(model, obs):
    """`W_in @ embedding(obs)` over the whole fixture, as `(T, reservoir_size)`.

    THE EXTERNAL DRIVE ONLY, and that is the point rather than a shortcut: this
    is the term a corrected frequency band would be derived from, it is a
    deterministic linear function of the observation, and it is what the phrase
    "the units see W_in @ embedding(obs)" in the spectrum stage means literally.
    The reservoir's total input current also carries `W_res @ spk_prev`, which is
    a function of the bank's own output and therefore of the band already chosen
    -- including it would make the measurement circular. Its absence is stated in
    the report, not hidden.

    Kept in float32, i.e. the dtype the model computes in, and reduced in float64
    downstream by `welch_psd`/`dc_power_fraction`: at 6,000 x 8,192 the float64
    copy is 393 MB against 196 MB and buys nothing that a float64 ACCUMULATOR
    over float32 samples does not already buy.
    """
    with torch.no_grad():
        return (model.embedding(obs) @ model.reservoir.W_in.T).numpy()


def per_unit_spike_stats(model, obs):
    """`silent_fraction`'s measurement, resolved PER UNIT instead of aggregated.

    Returns `{"rate", "silent", "saturated"}` as `(reservoir_size,)` arrays --
    each unit's own firing rate over the window, whether it never fired, whether
    it fired on every step. The loop is `analysis.pilot_diagnostics.silent_fraction`'s,
    step for step and with the same four-wide state tuple threaded positionally,
    so the aggregates it implies are the SAME numbers the preflight table prints:
    `rate.mean()`, `silent.mean()` and `saturated.mean()` at a given seed and
    scale must reproduce that scale's preflight row, and the report prints them
    side by side as an ALL row precisely so that agreement is visible rather than
    assumed. It is a separate function rather than an extra return value because
    changing `silent_fraction`'s signature would move a measurement four other
    modules and a committed table already depend on.

    Same caveat as `silent_fraction` and for the same reason: "never fired in
    6,000 steps" is an UPPER BOUND on permanently silent, not the same thing.
    """
    mem, imem, spk, _window = model.init_state(1, torch.device("cpu"))
    n_units = model.reservoir_size
    fired_count = torch.zeros(n_units)
    always = torch.ones(n_units, dtype=torch.bool)
    with torch.no_grad():
        for t in range(obs.shape[0]):
            emb = model.embedding(obs[t:t + 1])
            spk, mem, imem = model.reservoir.step(emb, mem, spk, imem)
            fired = spk[0] > 0
            fired_count += fired.float()
            always &= fired
    rate = (fired_count / obs.shape[0]).numpy()
    return {"rate": rate, "silent": rate == 0.0, "saturated": always.numpy()}


# --------------------------------------------------------------------------- #
# small print helpers
# --------------------------------------------------------------------------- #

def _fmt(value, spec, width=None, missing="n/a"):
    """Format a possibly-missing number without special-casing at every call
    site. `width` right-aligns BOTH branches -- a missing measurement prints as
    `n/a` in the same column as a present one, so a partial table lines up with a
    complete one instead of drifting a column per absent seed."""
    text = missing if value is None else format(value, spec)
    return f"{text:>{width}}" if width else text


def _pass_fail(ok):
    return "PASS" if ok else "FAIL"


def _wrap(text, indent="  ", width=76):
    """Wrap a required-verbatim block to the report's width without altering a
    character of it beyond the line breaks."""
    words, line, out = text.split(), "", []
    for word in words:
        candidate = f"{line} {word}".strip()
        if len(candidate) > width - len(indent):
            out.append(indent + line)
            line = word
        else:
            line = candidate
    if line:
        out.append(indent + line)
    return "\n".join(out)


def _delta_note(derived, quoted, spec=".6f"):
    """`derived` against §23's prose figure, with the difference -- the instrument
    rule's "any discrepancy [...] is reported rather than silently preferred",
    made mechanical."""
    if derived is None:
        return f"quoted {format(quoted, spec)}; re-derived n/a (no data)"
    return (f"re-derived {format(derived, spec)} against §23's quoted "
            f"{format(quoted, spec)} (delta {format(derived - quoted, '+.6f')})")


# --------------------------------------------------------------------------- #
# STAGE: preflight (§23.4, §23.5) -- runs BEFORE training
# --------------------------------------------------------------------------- #

def stage_preflight(seeds):
    obs, _mu = load_fixture()
    print(f"\n  fixture: {os.path.relpath(REAL_OBS_PATH, REPO_ROOT)}  "
          f"shape={tuple(obs.shape)}")
    print("  NOTE (§23.4/§14.13): this fixture was collected under v1 POLICIES. "
          "Holding the")
    print("  observation window fixed and varying only the neuron model is the "
          "controlled")
    print("  comparison for a construction property; it is NOT a measurement of "
          "what an")
    print("  rf policy would experience in situ, and that measurement is again "
          "not taken.")

    # ------------------------------------------------------------------ G0a --
    print("\n" + "=" * 78)
    print("G0a -- frequency construction (§23.5)")
    print(f"    gate: mean DC gain < {G0A_MEAN_DC_GAIN_BELOW} "
          f"(LIF: {LIF_QUOTED_MEAN_DC_GAIN}) AND "
          f"DC/AC < {G0A_DC_OVER_AC_BELOW} (LIF: {LIF_QUOTED_DC_OVER_AC})")
    print("    a failure means the frequencies were not drawn as §23.2 specifies")
    print("=" * 78)
    print(f"\n  {'run':<10}{'mean DC gain':>14}{'AC gain':>11}{'DC/AC':>10}"
          f"{'omega min':>12}{'omega max':>12}")
    print("  " + "-" * 69)
    g0a_rows = [frequency_construction(seed) for seed in seeds]
    for row in g0a_rows:
        print(f"  {'seed' + str(row['seed']):<10}{row['mean_dc_gain']:>14.4f}"
              f"{row['ac_gain']:>11.4f}{row['dc_over_ac']:>10.4f}"
              f"{row['omega_min']:>12.4f}{row['omega_max']:>12.4f}")
    mean_dc = mean_or_none([r["mean_dc_gain"] for r in g0a_rows])
    mean_ratio = mean_or_none([r["dc_over_ac"] for r in g0a_rows])
    ac = g0a_rows[0]["ac_gain"]
    print(f"\n  mean over {len(g0a_rows)} seeds: DC gain {mean_dc:.4f} "
          f"(§23.3 predicts {RF_QUOTED_MEAN_DC_GAIN}), "
          f"DC/AC {mean_ratio:.4f} (§23.3 predicts {RF_QUOTED_DC_OVER_AC})")
    print(f"  AC gain 1/sqrt(1-beta^2) = {ac:.4f}, identical in both neuron models "
          "by construction (§23.2)")
    print(f"  attenuation vs LIF: DC gain /{LIF_QUOTED_MEAN_DC_GAIN / mean_dc:.2f}, "
          f"DC/AC ratio /{LIF_QUOTED_DC_OVER_AC / mean_ratio:.2f}")
    g0a_dc_ok = mean_dc < G0A_MEAN_DC_GAIN_BELOW
    g0a_ratio_ok = mean_ratio < G0A_DC_OVER_AC_BELOW
    g0a_ok = g0a_dc_ok and g0a_ratio_ok
    print(f"\n  G0a: {_pass_fail(g0a_ok)}  "
          f"(mean DC gain {mean_dc:.4f} < {G0A_MEAN_DC_GAIN_BELOW}: "
          f"{_pass_fail(g0a_dc_ok)}; "
          f"DC/AC {mean_ratio:.4f} < {G0A_DC_OVER_AC_BELOW}: {_pass_fail(g0a_ratio_ok)})")

    # ------------------------------------------------- --embed-scale (§23.4) --
    print("\n" + "=" * 78)
    print("--embed-scale calibration (§23.4)")
    print(f"    grid (pre-declared, not searched further): {list(EMBED_SCALE_GRID)}")
    print("    criterion: minimise |log(rate_rf / rate_lif_v2_init)| on the seed-mean")
    print("    ALL SIX grid values are reported whatever is selected -- §23.4")
    print("=" * 78)

    print(f"\n  LIF v2 init reference ({LIF_INIT_EMBED_MODE} @ "
          f"{LIF_INIT_EMBED_SCALE}, same fixture, same seeds, same instrument)")
    print(f"\n  {'run':<10}{'silent':>10}{'spike rate':>14}{'saturated':>11}")
    print("  " + "-" * 45)
    lif_rows = []
    for seed in seeds:
        row = init_operating_point(seed, LIF_INIT_EMBED_SCALE, "lif", obs)
        lif_rows.append(row)
        print(f"  {'seed' + str(seed):<10}{row['silent']:>10.4%}"
              f"{row['spike_rate']:>14.6f}{row['saturated']:>11.4%}")
    lif_rate = mean_or_none([r["spike_rate"] for r in lif_rows])
    lif_silent = mean_or_none([r["silent"] for r in lif_rows])
    print(f"  {'MEAN':<10}{lif_silent:>10.4%}{lif_rate:>14.6f}"
          f"{mean_or_none([r['saturated'] for r in lif_rows]):>11.4%}")

    print(f"\n  resonate-and-fire grid ({RF_INIT_EMBED_MODE} init, "
          "neuron_model=rf), per seed and seed-mean")
    print(f"\n  {'scale':<8}{'run':<8}{'silent':>10}{'spike rate':>14}"
          f"{'saturated':>11}{'|log ratio|':>14}")
    print("  " + "-" * 65)
    grid_means = []
    for scale in EMBED_SCALE_GRID:
        rows = []
        for i, seed in enumerate(seeds):
            row = init_operating_point(seed, scale, "rf", obs)
            rows.append(row)
            print(f"  {f'{scale:g}' if i == 0 else '':<8}{'seed' + str(seed):<8}"
                  f"{row['silent']:>10.4%}{row['spike_rate']:>14.6f}"
                  f"{row['saturated']:>11.4%}{'':>14}")
        rate = mean_or_none([r["spike_rate"] for r in rows])
        silent = mean_or_none([r["silent"] for r in rows])
        saturated = mean_or_none([r["saturated"] for r in rows])
        # The per-scale criterion, computed by the SAME selector the choice below
        # goes through -- a second copy of `abs(log(...))` here is exactly how a
        # printed table and a verdict start disagreeing.
        _, criterion = select_embed_scale([(scale, rate)], lif_rate)
        grid_means.append({"scale": scale, "spike_rate": rate, "silent": silent,
                           "saturated": saturated, "criterion": criterion})
        print(f"  {'':<8}{'MEAN':<8}{silent:>10.4%}{rate:>14.6f}"
              f"{saturated:>11.4%}{_fmt(criterion, '.4f', 14)}")
        print()

    selected, criterion = select_embed_scale(
        [(g["scale"], g["spike_rate"]) for g in grid_means], lif_rate)
    if selected is None:
        print("  SELECTED: none -- no grid value has a measurable (nonzero) initial")
        print("  spike rate, so §23.4's log-ratio criterion is undefined across the")
        print("  whole grid. §23.5's G0b therefore cannot be satisfied by any grid")
        print("  value, which §23.5 says is itself a reportable finding.")
        chosen = None
    else:
        chosen = next(g for g in grid_means if g["scale"] == selected)
        print(f"  SELECTED: --embed-scale {selected:g}  "
              f"(|log ratio| {criterion:.4f}; rf rate {chosen['spike_rate']:.6f} "
              f"vs LIF init {lif_rate:.6f}, factor "
              f"{chosen['spike_rate'] / lif_rate:.3f})")

    # ------------------------------------------------------------ G0b, G0c --
    print("\n" + "=" * 78)
    print("G0b -- operating point is reachable (§23.5)")
    print(f"    gate: the SELECTED scale's mean initial spike rate in "
          f"[{G0B_INIT_RATE_LOW}, {G0B_INIT_RATE_HIGH}]")
    print("    a failure means the construction cannot be placed at a healthy")
    print("    operating point at all -- itself a reportable finding (§23.5)")
    print("=" * 78)
    if chosen is None:
        g0b_ok = False
        print("\n  G0b: FAIL -- no scale was selectable.")
    else:
        g0b_ok = G0B_INIT_RATE_LOW <= chosen["spike_rate"] <= G0B_INIT_RATE_HIGH
        print(f"\n  selected scale {selected:g}: mean initial spike rate "
              f"{chosen['spike_rate']:.6f}")
        print(f"  G0b: {_pass_fail(g0b_ok)}  (band [{G0B_INIT_RATE_LOW}, "
              f"{G0B_INIT_RATE_HIGH}], closed at both ends)")
        in_band = [g["scale"] for g in grid_means
                   if g["spike_rate"] is not None
                   and G0B_INIT_RATE_LOW <= g["spike_rate"] <= G0B_INIT_RATE_HIGH]
        print(f"  grid values whose mean initial rate lands in band: "
              f"{[f'{s:g}' for s in in_band] or 'NONE'}")

    print("\n" + "=" * 78)
    print("G0c -- not starved at init (§23.5)")
    print(f"    gate: the SELECTED scale's mean initial silent fraction < "
          f"{G0C_INIT_SILENT_BELOW:.0%}")
    print("    (v2 centered@3.0 LIF: 2.0523%; slack allowed because R&F changes the")
    print("    response SHAPE, not merely its scale)")
    print("=" * 78)
    if chosen is None:
        g0c_ok = False
        print("\n  G0c: FAIL -- no scale was selectable.")
    else:
        g0c_ok = chosen["silent"] < G0C_INIT_SILENT_BELOW
        print(f"\n  selected scale {selected:g}: mean initial silent fraction "
              f"{chosen['silent']:.4%}")
        print(f"  this run's LIF v2 init reference, same instrument: "
              f"{lif_silent:.4%}")
        print(f"  G0c: {_pass_fail(g0c_ok)}  (threshold "
              f"{G0C_INIT_SILENT_BELOW:.0%}, strict)")

    # G0b and G0c are separate gates on the SAME operating point, so whether ANY
    # grid value satisfies both at once is the question a failure of either one
    # actually poses -- and §23.5 says that outcome "is itself a reportable
    # finding about the mechanism". Cross-tabulated over the whole grid so the
    # report answers it without a reader re-deriving it from two tables. No new
    # band: both columns are the pre-registered constants above.
    print("\n  joint feasibility of G0b and G0c over the WHOLE pre-registered grid")
    print(f"\n  {'scale':<8}{'spike rate':>13}{'silent':>11}{'G0b':>7}{'G0c':>7}"
          f"{'both':>7}")
    print("  " + "-" * 53)
    both_ok = []
    for g in grid_means:
        rate_ok = (g["spike_rate"] is not None
                   and G0B_INIT_RATE_LOW <= g["spike_rate"] <= G0B_INIT_RATE_HIGH)
        silent_ok = g["silent"] is not None and g["silent"] < G0C_INIT_SILENT_BELOW
        if rate_ok and silent_ok:
            both_ok.append(g["scale"])
        print(f"  {g['scale']:<8g}{_fmt(g['spike_rate'], '.6f', 13)}"
              f"{_fmt(g['silent'], '.4%', 11)}{_pass_fail(rate_ok):>7}"
              f"{_pass_fail(silent_ok):>7}{_pass_fail(rate_ok and silent_ok):>7}")
    if both_ok:
        print(f"\n  grid values satisfying BOTH: {[f'{s:g}' for s in both_ok]}")
    else:
        print("\n  grid values satisfying BOTH: NONE.")
        print("  Read carefully: this does NOT license widening or re-searching the")
        print("  grid. §23.4 fixed those six values before measurement and does not")
        print("  search them, and §23.5 states that a construction which cannot be")
        print("  placed at a healthy operating point is itself the finding. Any new")
        print("  scale would be a post-hoc search on a gate this pilot already")
        print("  failed, which is exactly what the pre-registration forbids.")

    # -------------------------------------------------------- G0d, G0e note --
    print("\n" + "=" * 78)
    print("G0d, G0e -- measured elsewhere, NOT by this module (§23.5)")
    print("=" * 78)
    print("\n  G0d (feasibility, throughput >= 250 env-steps/s; below 150 the pilot")
    print("  does not launch) is a wall-clock measurement on a quiet machine and")
    print("  needs the emulator. It is not a construction measurement and is not")
    print("  taken here.")
    print("\n  G0e (the comparison is legitimate) is two bit-exactness properties:")
    print("    G0e-i  -- the --neuron-model lif path reproduces a committed")
    print("              checkpoints_v2 train_log.jsonl prefix exactly. Covered by")
    print("              tests/test_neuron_model_flag.py. §23.11 records that this")
    print("              holds AT ONE TORCH THREAD, which is the condition every")
    print("              file in checkpoints_v2/ was produced under.")
    print("    G0e-ii -- the rf cell at omega == 0 reproduces snn.Leaky(beta=0.9)")
    print("              bit-exactly. Covered by tests/test_resonate_and_fire.py.")
    print("  Both are torch.equal properties of a training run and a cell, not of")
    print("  a fixture, so they belong in the suite and are named here rather than")
    print("  silently omitted.")

    # ------------------------------------------------------------- verdict --
    overall = g0a_ok and g0b_ok and g0c_ok
    print("\n" + "=" * 78)
    print("PREFLIGHT SUMMARY (§23.5)")
    print("=" * 78)
    print(f"  G0a  {_pass_fail(g0a_ok):<6} mean DC gain {mean_dc:.4f} "
          f"(<{G0A_MEAN_DC_GAIN_BELOW}), DC/AC {mean_ratio:.4f} "
          f"(<{G0A_DC_OVER_AC_BELOW})")
    if chosen is None:
        print(f"  G0b  {_pass_fail(g0b_ok):<6} no scale selectable")
        print(f"  G0c  {_pass_fail(g0c_ok):<6} no scale selectable")
    else:
        print(f"  G0b  {_pass_fail(g0b_ok):<6} initial spike rate "
              f"{chosen['spike_rate']:.6f} in [{G0B_INIT_RATE_LOW}, "
              f"{G0B_INIT_RATE_HIGH}] at --embed-scale {selected:g}")
        print(f"  G0c  {_pass_fail(g0c_ok):<6} initial silent fraction "
              f"{chosen['silent']:.4%} (<{G0C_INIT_SILENT_BELOW:.0%})")
    print("  G0d  ELSEWHERE  throughput, needs the emulator on a quiet machine")
    print("  G0e  ELSEWHERE  bit-exactness, covered by the test suite (§23.11)")
    print(f"\n  PREFLIGHT (G0a/G0b/G0c): {'PASS' if overall else 'FAIL'}")
    if overall and selected is not None:
        print(f"  The three construction gates are satisfied at --embed-scale "
              f"{selected:g}.")
        print("  G0d and G0e are still outstanding and are part of §23.5's launch")
        print("  condition; this module does not clear them.")
    else:
        print("  §23.5: if a validity gate fails, the pilot is not measuring what it")
        print("  claims to measure and DOES NOT LAUNCH.")
    return overall


# --------------------------------------------------------------------------- #
# STAGE: spectrum -- WHERE THE ENERGY ACTUALLY IS. Measures, gates nothing.
# --------------------------------------------------------------------------- #

def _spectrum_measure_seed(seed, embed_scale, obs_tensor):
    """Every per-seed quantity the spectrum report needs, in one pass.

    One pass rather than three because each seed costs a 6,000-step reservoir
    rollout, and taking it once for the spike statistics and again for the
    spectra would double the stage's wall clock to print the same numbers.
    """
    model, _ = reservoir_at(seed, RF_INIT_EMBED_MODE, embed_scale, neuron_model="rf")
    reservoir = model.reservoir
    beta = float(reservoir.lif.beta)
    omega = reservoir.omega.detach().numpy().astype(np.float64)
    periods = 2.0 * np.pi / omega

    current = input_current(model, obs_tensor)
    dc_per_unit, dc_pooled = dc_power_fraction(current)
    mean_current = current.mean(axis=0, dtype=np.float64)
    freqs, psd, n_segments = welch_psd(current - mean_current.astype(current.dtype))
    del current
    psd_pooled = psd.sum(axis=1)

    offset = induced_membrane_offset(torch.as_tensor(mean_current), beta,
                                     reservoir.omega.double()).numpy()
    row = {
        "seed": seed, "beta": beta, "omega": omega, "periods": periods,
        "dc_per_unit": dc_per_unit, "dc_pooled": dc_pooled,
        "freqs": freqs, "n_segments": n_segments, "psd_pooled": psd_pooled,
        "band_fracs": band_power_fractions(freqs, psd_pooled),
        "slow_resid": unresolved_slow_fraction(psd_pooled),
        "offset": offset,
        "response_std": np.sqrt(resonant_u_response_var(freqs, psd, omega, beta)),
        "spike": per_unit_spike_stats(model, obs_tensor),
    }
    del psd
    return row


def _rf_bin_rows(row):
    """§3's octave table for one seed: one dict per `RF_PERIOD_BIN_EDGES` bin,
    plus an ALL row that must reproduce the preflight table's row at this scale."""
    periods, spike = row["periods"], row["spike"]
    index = np.array([period_bin_index(p, RF_PERIOD_BIN_EDGES) for p in periods])
    out = []
    for b in range(len(RF_PERIOD_BIN_EDGES) - 1):
        mask = index == b
        mean_over_bin = ((lambda key: float(spike[key][mask].mean())) if mask.any()
                         else (lambda key: float("nan")))
        out.append({
            "label": f"[{RF_PERIOD_BIN_EDGES[b]:g},{RF_PERIOD_BIN_EDGES[b + 1]:g})",
            "mask": mask, "n": int(mask.sum()),
            "rate": mean_over_bin("rate"), "silent": mean_over_bin("silent"),
            "saturated": mean_over_bin("saturated"),
        })
    everything = np.ones(len(periods), dtype=bool)
    out.append({"label": "ALL", "mask": everything, "n": int(len(periods)),
                "rate": float(spike["rate"].mean()),
                "silent": float(spike["silent"].mean()),
                "saturated": float(spike["saturated"].mean())})
    return out


def stage_spectrum(seeds, embed_scale):
    obs_tensor, mu = load_fixture()
    obs_np = obs_tensor.numpy()
    print(f"\n  fixture: {os.path.relpath(REAL_OBS_PATH, REPO_ROOT)}  "
          f"shape={tuple(obs_tensor.shape)}")
    print("  NOTE (§23.4/§14.13): collected under v1 POLICIES. Every spectrum below "
          "is the")
    print("  spectrum of THAT window. A resonate-and-fire policy would visit a "
          "different")
    print("  observation distribution, and this stage does not measure it either.")
    if obs_tensor.shape[0] < WELCH_SEGMENT_LEN:
        print(f"\n  SKIPPED: the fixture is {obs_tensor.shape[0]} steps and Welch at "
              f"segment length {WELCH_SEGMENT_LEN} needs at least that many. No "
              "spectrum is estimated.")
        return None

    print("\n" + "=" * 78)
    print("WHAT THIS STAGE MEASURES, AND WHAT IT DOES NOT")
    print("=" * 78)
    print("\n  §23.12 recorded the preflight negative and offered a mechanism for it:")
    print("  \"Real observations carry almost no energy at 2-6-step periods, so the")
    print("  fast half of the filter bank receives no drive.\" That was an INFERENCE")
    print("  from silence sorted by resonant period, not a measurement of the")
    print("  observation's spectrum. This stage measures the spectrum directly.")
    print("\n  IT GATES NOTHING. No band here is pre-registered, no verdict is")
    print("  computed, and §4 stops at a tradeoff table on purpose. It also cannot")
    print("  tell you what a TRAINED embedding does to any of this: every number")
    print("  below is taken at the step-0 init, and §23.1's whole finding is that the")
    print("  embedding drifts. And it is the EXTERNAL drive only -- W_in @ embedding(obs),")
    print("  without the frozen recurrent W_res @ spk term, which is a function of the")
    print("  bank's own output and would make the measurement circular.")

    # ------------------------------------------------------------------- 1 --
    print("\n" + "=" * 78)
    print("1. THE OBSERVATION'S OWN TEMPORAL POWER SPECTRUM")
    print("=" * 78)

    zero_slots = [d for d in range(obs_np.shape[1]) if not np.any(obs_np[:, d])]
    dc_per_dim, dc_pooled = dc_power_fraction(obs_np)
    means = obs_np.mean(axis=0, dtype=np.float64)
    stds = obs_np.astype(np.float64).std(axis=0)

    print("\n  1a. DC SHARE OF EACH DIMENSION'S POWER -- mean^2 / mean(x^2).")
    print("      Accounted separately from the band table below, and subtracted "
          "before it,")
    print("      because at 78% of total power the DC would otherwise swamp every "
          "band.")
    print(f"\n  {'slot':>4}  {'name':<16}{'mean':>11}{'std':>10}{'DC share':>11}"
          f"   note")
    print("  " + "-" * 74)
    for d in range(obs_np.shape[1]):
        note = ""
        if d in zero_slots:
            note = ("identically zero -- reserved slot (RESULTS.md v1 §9)"
                    if d in RESERVED_ZERO_SLOTS else "identically zero")
        share = "n/a" if not math.isfinite(dc_per_dim[d]) else f"{dc_per_dim[d]:.4%}"
        print((f"  {d:>4}  {OBS_SLOT_NAMES[d]:<16}{means[d]:>11.6f}{stds[d]:>10.6f}"
               f"{share:>11}   {note}").rstrip())
    mean_energy = float((means ** 2).sum())
    total_energy = float((obs_np.astype(np.float64) ** 2).mean(axis=0).sum())
    print(f"\n  POOLED ||E obs||^2 / E||obs||^2 = {mean_energy:.6f} / "
          f"{total_energy:.6f} = {dc_pooled:.4%}")
    print(f"  RESULTS.md v1 §7.1 published {RESULTS_QUOTED_OBS_DC_FRACTION:.2%} "
          f"({RESULTS_QUOTED_OBS_MEAN_ENERGY:.6f} / "
          f"{RESULTS_QUOTED_OBS_TOTAL_ENERGY:.6f}); "
          f"delta {dc_pooled - RESULTS_QUOTED_OBS_DC_FRACTION:+.4%}.")
    print("  REPRODUCED, to within a fifth of a percentage point -- but NOT on the "
          "same")
    print("  bytes, and the difference is worth stating rather than rounding away. "
          "§7.1's")
    print("  1.331336 is exactly ||OBS_MEAN||^2, so §7.1 was computed on the "
          "6,000-step")
    print("  collection `envs.mario_land_env.OBS_MEAN` was measured from, and this "
          "committed")
    print("  fixture is a DIFFERENT 6,000-step collection of the same construction "
          "(its own")
    worst_slot = int(np.abs(means - mu.numpy()).argmax())
    print(f"  per-dimension mean differs from OBS_MEAN by up to "
          f"{float(np.abs(means - mu.numpy()).max()):.4f}, on slot "
          f"{worst_slot} `{OBS_SLOT_NAMES[worst_slot]}`).")
    print("  The published claim holds on this fixture; it is not the identical "
          "window, and")
    print("  the centred init is therefore an approximate correction on these bytes "
          "rather")
    print("  than the exact one it is on the window OBS_MEAN came from.")
    print(f"\n  {len(zero_slots)} of {obs_np.shape[1]} slots are identically zero "
          f"({zero_slots}). They have NO")
    print("  spectrum -- not a flat one -- so they read n/a above and are excluded "
          "from")
    print("  the band table below. They contribute exactly zero to both sides of the")
    print("  pooled ratio, so pooling over 9 or over 12 dimensions is the same number.")

    obs_centred = obs_np.astype(np.float64) - means
    freqs, obs_psd, n_segments = welch_psd(obs_centred)
    counts = band_bin_counts(freqs)
    print("\n  1b. WELCH BAND POWERS OF THE CENTRED OBSERVATION.")
    print(f"      segment {WELCH_SEGMENT_LEN}, {WELCH_OVERLAP:.0%} overlap, periodic "
          f"Hann, {n_segments} averaged segments,")
    print(f"      resolution 1/{WELCH_SEGMENT_LEN} = {freqs[1]:.6f} cycles/step "
          f"(longest resolved period {1.0 / freqs[1]:.0f} steps).")
    print("      Fractions are of RESOLVED fluctuating power and sum to 1; bin 0 "
          "holds")
    print("      what is slower than one segment and is reported beside them.")
    print(f"\n  {'slot':>4}  {'name':<16}" + "".join(f"{lab:>9}" for lab in PERIOD_BAND_LABELS)
          + f"{'slow':>9}")
    print("  " + "-" * 79)
    for d in range(obs_np.shape[1]):
        if d in zero_slots:
            print(f"  {d:>4}  {OBS_SLOT_NAMES[d]:<16}"
                  + f"{'-- identically zero, no spectrum --':>63}")
            continue
        fracs = band_power_fractions(freqs, obs_psd[:, d])
        print(f"  {d:>4}  {OBS_SLOT_NAMES[d]:<16}"
              + "".join(f"{f:>9.4f}" for f in fracs)
              + f"{unresolved_slow_fraction(obs_psd[:, d]):>9.4f}")
    obs_pooled_psd = obs_psd.sum(axis=1)
    obs_fracs = band_power_fractions(freqs, obs_pooled_psd)
    print("  " + "-" * 79)
    print(f"  {'':>4}  {'POOLED':<16}" + "".join(f"{f:>9.4f}" for f in obs_fracs)
          + f"{unresolved_slow_fraction(obs_pooled_psd):>9.4f}")
    print(f"  {'':>4}  {'resolved bins':<16}" + "".join(f"{c:>9d}" for c in counts))
    print(f"  {'':>4}  {'rel. density':<16}"
          + "".join(f"{v:>9.3f}" for v in band_power_density(obs_fracs, counts)))
    print("\n  RELATIVE DENSITY is band power per resolved bin, in units of the "
          "whole")
    print("  spectrum's mean; 1.000 would be flat in frequency. It is the row to "
          "read if")
    print("  you are choosing a band for a bank of RESONATORS, and the fraction row "
          "is")
    print("  not: an octave at fast periods spans eight times the frequency width of "
          "an")
    print("  octave three octaves slower, so the fraction row flatters the fast end "
          "by")
    print("  exactly that factor, while a unit at frequency w integrates the power")
    print("  DENSITY near w through a bandpass whose width is set by beta, not by "
          "the")
    print("  octave the bin happens to be filed under.")

    # ------------------------------------------------------------------- 2 --
    print("\n" + "=" * 78)
    print("2. THE SPECTRUM OF WHAT THE RESERVOIR ACTUALLY RECEIVES")
    print(f"    input current = W_in @ embedding(obs), neuron_model=rf, "
          f"{RF_INIT_EMBED_MODE} init,")
    print(f"    --embed-scale {embed_scale:g}, seeds {seeds}, pooled over all "
          f"{RESERVOIR_SIZE} units")
    print("=" * 78)
    print("\n  measuring (each seed: build, 6,000-step rollout, Welch)...")
    rows = [_spectrum_measure_seed(s, embed_scale, obs_tensor) for s in seeds]
    beta = rows[0]["beta"]
    assert abs(beta - RF_BETA) < 1e-6, (
        f"§23.2 fixes beta = {RF_BETA}; the built reservoir carries {beta}")

    print("\n  2a. DC SHARE OF THE INPUT CURRENT, pooled over units.")
    print(f"\n  {'run':<10}{'init':<20}{'DC share':>12}")
    print("  " + "-" * 42)
    for row in rows:
        print(f"  {'seed' + str(row['seed']):<10}"
              f"{f'{RF_INIT_EMBED_MODE}@{embed_scale:g}':<20}{row['dc_pooled']:>12.4%}")
    print(f"  {'MEAN':<10}{'':<20}"
          f"{mean_or_none([r['dc_pooled'] for r in rows]):>12.4%}")
    legacy_model, _ = reservoir_at(seeds[0], LEGACY_EMBED_MODE, LEGACY_EMBED_SCALE,
                                   neuron_model="rf")
    _legacy_per_unit, legacy_dc = dc_power_fraction(input_current(legacy_model, obs_tensor))
    print("  " + "-" * 42)
    print(f"  {'seed' + str(seeds[0]):<10}"
          f"{f'{LEGACY_EMBED_MODE}@{LEGACY_EMBED_SCALE:.1f}':<20}{legacy_dc:>12.4%}"
          f"   <- the v1 uncentred init, the published cross-check")
    print(f"\n  RESULTS.md v1 §7.1 published "
          f"{RESULTS_QUOTED_CURRENT_DC_FRACTION:.2%} for the input current's DC")
    print(f"  share under the v1 init; re-derived here at {legacy_dc:.4%} "
          f"(delta {legacy_dc - RESULTS_QUOTED_CURRENT_DC_FRACTION:+.4%}), on")
    print("  the different-window caveat §1a states. The instrument agrees with the")
    print("  published figure.")
    print(f"\n  Under `{RF_INIT_EMBED_MODE}` the drive is DC-free BY CONSTRUCTION -- "
          "the init sets")
    print("  b := -(W @ OBS_MEAN) -- and the residual above is what is left because "
          "this")
    print("  fixture's own mean is not OBS_MEAN. It is small, it is not zero, and it "
          "is")
    print("  the term §3b's standing-offset column is computed from. NOTE that under")
    print("  this init the DC share does not move with --embed-scale at all: the "
          "scale")
    print("  multiplies W and the bias -(W @ OBS_MEAN) alike, so it multiplies "
          "numerator")
    print("  and denominator alike, and it is the operating point it moves and not "
          "the")
    print("  balance between DC and AC.")

    print("\n  2b. BAND POWERS OF THE INPUT CURRENT, pooled over units, per seed.")
    print(f"\n  {'run':<12}" + "".join(f"{lab:>9}" for lab in PERIOD_BAND_LABELS)
          + f"{'slow':>9}")
    print("  " + "-" * 75)
    for row in rows:
        print(f"  {'seed' + str(row['seed']):<12}"
              + "".join(f"{f:>9.4f}" for f in row["band_fracs"])
              + f"{row['slow_resid']:>9.4f}")
    current_fracs = np.mean([r["band_fracs"] for r in rows], axis=0)
    print("  " + "-" * 75)
    print(f"  {'MEAN':<12}" + "".join(f"{f:>9.4f}" for f in current_fracs)
          + f"{mean_or_none([r['slow_resid'] for r in rows]):>9.4f}")
    print(f"  {'rel. density':<12}"
          + "".join(f"{v:>9.3f}" for v in band_power_density(current_fracs, counts)))
    print(f"  {'observation':<12}" + "".join(f"{f:>9.4f}" for f in obs_fracs)
          + "         <- §1b, for comparison")
    print("\n  The embedding is LINEAR and W_in is a frozen fixed matrix, so the "
          "input")
    print("  current can only be a remixing of the twelve observation spectra: no "
          "linear")
    print("  map can move energy from one period to another. Measured, it comes out "
          "close")
    print("  to the pooled observation itself. Whatever band structure the bank "
          "meets is")
    print("  a property of the OBSERVATION and not of the embedding, and no choice "
          "of")
    print("  --embed-scale changes it -- a scalar gain multiplies every band alike.")

    # ------------------------------------------------------------------- 3 --
    print("\n" + "=" * 78)
    print("3. THE BANK'S RESPONSE, RESOLVED BY ITS OWN RESONANT PERIOD")
    print(f"    8192 units bucketed by frozen T_i into octaves, at --embed-scale "
          f"{embed_scale:g}")
    print("    This is §23.12's scratchpad finding, committed and reproducible.")
    print("=" * 78)
    for row in rows:
        print(f"\n  --- seed{row['seed']}   T_i range "
              f"[{row['periods'].min():.4f}, {row['periods'].max():.4f}] steps")
        print(f"  {'bin':<10}{'units':>8}{'mean rate':>13}{'silent':>11}"
              f"{'saturated':>11}")
        print("  " + "-" * 53)
        for entry in _rf_bin_rows(row):
            marker = "   <- reproduces the preflight row" if entry["label"] == "ALL" else ""
            print(f"  {entry['label']:<10}{entry['n']:>8}{entry['rate']:>13.6f}"
                  f"{entry['silent']:>11.4%}{entry['saturated']:>11.4%}{marker}")
    pooled_periods = np.concatenate([r["periods"] for r in rows])
    pooled_rate = np.concatenate([r["spike"]["rate"] for r in rows])
    pooled_silent = np.concatenate([r["spike"]["silent"] for r in rows])
    pooled_saturated = np.concatenate([r["spike"]["saturated"] for r in rows])
    pooled_index = np.array([period_bin_index(p, RF_PERIOD_BIN_EDGES)
                             for p in pooled_periods])
    print(f"\n  --- pooled over seeds {seeds}")
    print(f"  {'bin':<10}{'units':>8}{'mean rate':>13}{'silent':>11}{'saturated':>11}")
    print("  " + "-" * 53)
    for b in range(len(RF_PERIOD_BIN_EDGES) - 1):
        mask = pooled_index == b
        print(f"  {f'[{RF_PERIOD_BIN_EDGES[b]:g},{RF_PERIOD_BIN_EDGES[b + 1]:g})':<10}"
              f"{int(mask.sum()):>8}{pooled_rate[mask].mean():>13.6f}"
              f"{pooled_silent[mask].mean():>11.4%}{pooled_saturated[mask].mean():>11.4%}")
    print(f"  {'ALL':<10}{len(pooled_periods):>8}{pooled_rate.mean():>13.6f}"
          f"{pooled_silent.mean():>11.4%}{pooled_saturated.mean():>11.4%}"
          f"   <- reproduces the preflight MEAN row")
    print(f"\n  median T_i of a SILENT unit : "
          f"{float(np.median(pooled_periods[pooled_silent])):.4f} steps "
          f"(n={int(pooled_silent.sum())})")
    print(f"  median T_i of a FIRING unit : "
          f"{float(np.median(pooled_periods[~pooled_silent])):.4f} steps "
          f"(n={int((~pooled_silent).sum())})")

    # Monotonicity is COMPUTED rather than read off the table above, for the same
    # reason every verdict in this module is: §23.12 asserted it from a diagnostic
    # nobody kept, and an eyeballed "clearly decreasing" is exactly how a claim
    # that holds in one seed becomes a claim about the mechanism.
    per_seed_bins = [_rf_bin_rows(r)[:-1] for r in rows]
    monotone = [all(b[i]["silent"] >= b[i + 1]["silent"] for i in range(len(b) - 1))
                for b in per_seed_bins]
    fastest = [b[0]["silent"] for b in per_seed_bins]
    tail_monotone = [all(b[i]["silent"] >= b[i + 1]["silent"] for i in range(1, len(b) - 1))
                     for b in per_seed_bins]
    print(f"\n  silent fraction non-increasing across ALL FOUR octaves: "
          f"{sum(monotone)}/{len(monotone)} seeds")
    print(f"  non-increasing across the THREE octaves from [4,8) up:   "
          f"{sum(tail_monotone)}/{len(tail_monotone)} seeds")
    print(f"  fastest octave [2,4) silent fraction across seeds: "
          f"{min(fastest):.4%} .. {max(fastest):.4%}")
    print("\n  §23.12 reported this as monotone in frequency, from a scratchpad "
          "diagnostic")
    print("  at a different --embed-scale. MEASURED HERE, THE MONOTONICITY IS REAL "
          "BUT")
    print("  NARROWER THAN THAT. From [4,8) upwards it holds in every seed and the "
          "bins")
    print("  are tightly clustered across seeds. The FASTEST octave does not join "
          "it:")
    print("  it swings by more than 30 percentage points across three seeds and is "
          "not")
    print("  reliably the most silent bin. It also carries the HIGHEST mean rate of "
          "any")
    print("  bin, which the two columns state together -- a large minority of its "
          "units")
    print("  never fire, and the rest fire often. Near T = 2 the pole is nearly real "
          "and")
    print("  negative, so a unit that crosses threshold at all tends to do it every "
          "other")
    print("  step; that bin is bimodal and a mean over it describes neither mode.")

    print("\n  3b. WHY -- the linearised response, NOT a gate and NOT a "
          "recommendation.")
    print("      Each unit's input spectrum, put through the analytic transfer "
          "function")
    print("      from input current to the membrane u that the threshold is applied "
          "to")
    print("      (§23.2's real part of the complex pole), against a threshold of "
          "1.0.")
    print("      Input-driven and linearised: no recurrent term, no threshold, no "
          "reset,")
    print("      so it predicts the SCALE of the membrane's excursions and not a "
          "rate.")
    print(f"\n  {'bin':<10}{'units':>8}{'resp. std':>12}{'|offset|':>11}"
          f"{'mean DC gain':>14}{'silent':>11}")
    print("  " + "-" * 67)
    pooled_response = np.concatenate([r["response_std"] for r in rows])
    pooled_offset = np.concatenate([r["offset"] for r in rows])
    pooled_omega = np.concatenate([r["omega"] for r in rows])
    dc_gains = 1.0 / np.sqrt((1.0 - beta * np.cos(pooled_omega)) ** 2
                             + (beta * np.sin(pooled_omega)) ** 2)
    for b in range(len(RF_PERIOD_BIN_EDGES) - 1):
        mask = pooled_index == b
        print(f"  {f'[{RF_PERIOD_BIN_EDGES[b]:g},{RF_PERIOD_BIN_EDGES[b + 1]:g})':<10}"
              f"{int(mask.sum()):>8}{pooled_response[mask].mean():>12.4f}"
              f"{np.abs(pooled_offset[mask]).mean():>11.4f}"
              f"{dc_gains[mask].mean():>14.4f}{pooled_silent[mask].mean():>11.4%}")
    print(f"  {'ALL':<10}{len(pooled_response):>8}{pooled_response.mean():>12.4f}"
          f"{np.abs(pooled_offset).mean():>11.4f}{dc_gains.mean():>14.4f}"
          f"{pooled_silent.mean():>11.4%}")
    print("\n  The standing offset is two orders of magnitude below the threshold in "
          "EVERY")
    print("  bin, so it is not what silences the fast units at this init -- the "
          "response")
    print("  AMPLITUDE is. It rises steeply from the fastest octave and then FLATTENS:")
    print("  the two slowest bins have essentially the same predicted excursion and")
    print("  visibly different silent fractions, so amplitude explains the fast end "
          "and")
    print("  does not by itself separate the slow end. A slow unit also dwells "
          "longer on")
    print("  each excursion, which this column does not measure and which is not "
          "chased")
    print("  here.")

    # ------------------------------------------- §23.12's sentence, confronted --
    print("\n" + "=" * 78)
    print("§23.12's STATED MECHANISM, AGAINST THESE NUMBERS")
    print("=" * 78)
    below_8 = current_fracs[0] + current_fracs[1]
    density = band_power_density(current_fracs, counts)
    in_pre_registered = mean_or_none([
        band_power_fraction_in(r["freqs"], r["psd_pooled"], 2.0, 32.0) for r in rows])
    print("\n  §23.12: \"Real observations carry almost no energy at 2-6-step "
          "periods, so")
    print("  the fast half of the filter bank receives no drive.\" Taken apart:")
    print(f"\n  * THE FIRST CLAUSE IS NOT SUPPORTED AS WRITTEN. {below_8:.2%} of the "
          f"input")
    print(f"    current's resolved fluctuating power is at periods below 8 steps "
          f"({current_fracs[0]:.2%}")
    print(f"    of it below 4). That is less than the {current_fracs[2]:.2%} in the "
          f"8-16 octave, but")
    print("    it is not \"almost no energy\".")
    print(f"\n  * THE SECOND CLAUSE SURVIVES, THROUGH A DIFFERENT QUANTITY. Power "
          f"DENSITY at")
    print(f"    T<4 is {density[0]:.3f} of a flat spectrum's against {density[2]:.3f} "
          f"in the 8-16 octave, a factor")
    print(f"    {density[2] / density[0]:.1f}. A resonator integrates density near "
          "its own frequency, not the")
    print("    octave's total, so the fast units do receive far less drive -- and "
          "§3b")
    print(f"    measures the consequence: predicted membrane excursion "
          f"{pooled_response[pooled_index == 0].mean():.4f} against a")
    print(f"    threshold of 1.0, against {pooled_response[pooled_index == 2].mean():.4f} "
          "three octaves slower.")
    print(f"\n  * THE PRE-REGISTERED BAND IS NOT MISSING THE ENERGY. T in [2, 32] "
          f"contains")
    print(f"    {in_pre_registered:.2%} of the measured input-current fluctuating "
          "power. Whatever costs")
    print("    the fast half of the bank its drive, it is not that §23.2's support")
    print("    excludes where the observation's energy sits.")
    print("\n  Recorded this way -- clause by clause, with the half that does not")
    print("  reproduce named first -- because §23.12 wrote its mechanism down as "
          "prose")
    print("  and this stage exists to check it, and a check that only reports the "
          "half")
    print("  that agreed is not one.")

    # ------------------------------------------------------------------- 4 --
    print("\n" + "=" * 78)
    print("4. THE TRADEOFF A CORRECTED BAND WOULD BE DERIVED FROM")
    print("    analytic mean DC gain over T ~ logU[T_min, T_max] (§23.3's quantity,")
    print("    as a function of the band) against the measured share of input-current")
    print(f"    power inside it. beta = {beta:.10f}, read off the built reservoir.")
    print("=" * 78)
    lif_dc = dc_gain_at_period(math.inf, beta)
    ac = ac_gain(beta)
    print(f"\n  LIF reference (omega == 0): mean DC gain {lif_dc:.4f}, "
          f"AC gain {ac:.4f}, DC/AC {lif_dc / ac:.4f}")
    print(f"  §23.3 quotes {LIF_QUOTED_MEAN_DC_GAIN} and "
          f"{LIF_QUOTED_DC_OVER_AC}, which this reproduces.")
    print("  The AC gain is IDENTICAL for every band in the table -- it depends on")
    print("  |lambda| = beta only (§23.2) -- so the DC/AC column moves only because "
          "the")
    print("  DC column does, and the two 'vs LIF' columns are therefore the same "
          "factor.")
    print(f"\n  {'T_min':>7}{'T_max':>8}{'mean DC gain':>14}{'vs LIF':>9}"
          f"{'DC/AC':>9}{'vs LIF':>9}{'input power in band':>21}")
    print("  " + "-" * 77)
    grid = ([(TRADEOFF_T_MIN_FIXED, t) for t in TRADEOFF_T_MAX_GRID]
            + [(t, TRADEOFF_T_MAX_FIXED) for t in TRADEOFF_T_MIN_GRID])
    for t_min, t_max in grid:
        gain = mean_dc_gain_log_uniform(t_min, t_max, beta)
        share = mean_or_none([band_power_fraction_in(r["freqs"], r["psd_pooled"],
                                                     t_min, t_max) for r in rows])
        marker = "  <- §23.2's band" if (t_min, t_max) == (2.0, 32.0) else ""
        print(f"  {t_min:>7g}{t_max:>8g}{gain:>14.4f}{'/' + format(lif_dc / gain, '.2f'):>9}"
              f"{gain / ac:>9.4f}"
              f"{'/' + format(LIF_QUOTED_DC_OVER_AC / (gain / ac), '.2f'):>9}"
              f"{share:>21.4f}{marker}")
    print("\n  'input power in band' is the share of the measured input-current "
          "fluctuating")
    print("  power at periods in [T_min, T_max], CLOSED at both ends, mean over the "
          "three")
    print("  seeds. The rows are candidate supports and not a partition, so rows "
          "sharing")
    print("  an endpoint share the bin that sits on it and the column does not sum "
          "to 1.")
    print("\n  READ IT WITH §1b's DENSITY ROW, not on its own. A band's power SHARE "
          "grows")
    print("  with its width, so the widest band always wins this column; what a "
          "resonator")
    print("  at the fast end actually receives is the density, which §1b measures at "
          "a")
    print("  small fraction of the slow end's.")
    print("\n" + _wrap(NO_BAND_RECOMMENDATION))

    # ------------------------------------------------------- what it is not --
    print("\n" + "=" * 78)
    print("WHAT THIS MEASUREMENT DOES NOT ESTABLISH")
    print("=" * 78)
    print("\n  * NOT that any band would work. Every number above is a property of "
          "the")
    print("    input path at step 0. Nothing here was trained, no reward was "
          "measured,")
    print("    and a bank whose units are driven is not thereby a bank whose units "
          "are")
    print("    USEFUL -- §8's original A3 hypothesis, that the decomposition helps "
          "the")
    print("    task, is tested by GB and by nothing in this stage.")
    print("  * NOT a measurement of the trained operating point. §23.1's defect is "
          "that")
    print("    the embedding DRIFTS; these spectra are the drive before it does.")
    print("  * NOT the in-situ observation distribution. The fixture is v1-policy "
          "data")
    print("    (§14.13), and a resonate-and-fire policy would visit different states.")
    print("  * NOT the total input current. The recurrent W_res @ spk term is "
          "excluded")
    print("    by construction, and §23.3 names it as the DC source resonate-and-fire")
    print("    attenuates least.")
    print("  * NOT a spike-rate model. §3b is linearised and input-only; the measured")
    print("    columns beside it are the measurement.")
    return rows


# --------------------------------------------------------------------------- #
# STAGE: verdict (§23.6, §23.7, §23.8) -- runs AFTER training and evaluation
# --------------------------------------------------------------------------- #

def _arm_trajectories(checkpoint_dir, seeds, obs, mu, label):
    """A9's full per-checkpoint trajectory for every seed of one arm, keyed by
    seed. Reuses `a9_run_stats` verbatim -- that IS the instrument rule: the LIF
    control and the rf arm go through the same function, and it already reads each
    checkpoint's own neuron model."""
    out = {}
    for seed in seeds:
        stats = a9_run_stats(checkpoint_dir, "reservoir", seed, obs, mu, None)
        out[seed] = stats
        if stats["status"] != "ok":
            print(f"  {label} seed{seed}: {stats['status']} -- skipped ({stats['dir']})")
    return out


def _final_values(trajectories, seeds, key):
    """One measurement at each seed's final checkpoint, `None` where absent."""
    return [None if trajectories[s]["status"] != "ok" else trajectories[s]["final"][key]
            for s in seeds]


def _print_final_row(name, per_seed, seeds, spec, mean_spec=None):
    cells = "  ".join(f"seed{s}={_fmt(v, spec)}" for s, v in zip(seeds, per_seed))
    mean = mean_or_none(per_seed)
    print(f"  {name:<34}{cells}")
    print(f"  {'':<34}mean {_fmt(mean, mean_spec or spec)} "
          f"(n={len([v for v in per_seed if v is not None])})")
    return mean


def stage_verdict(rf_checkpoint_dir, lif_checkpoint_dir, rf_results_dir,
                  lif_results_dir, seeds, embed_scale):
    obs, mu = load_fixture()
    print(f"\n  fixture: {os.path.relpath(REAL_OBS_PATH, REPO_ROOT)}  "
          f"shape={tuple(obs.shape)}")
    print("  Both arms below are measured by THIS code from checkpoints and result")
    print("  JSONs on disk (§17.11 / §23.6's instrument rule). §23's prose figures")
    print("  are diffed against, never substituted for, what is re-derived here.")

    print("\n  loading trajectories (this reads every checkpoint of every seed)...")
    rf_traj = _arm_trajectories(rf_checkpoint_dir, seeds, obs, mu, "rf ")
    lif_traj = _arm_trajectories(lif_checkpoint_dir, seeds, obs, mu, "lif")

    # ----------------------------------------------------------------- GA ---
    print("\n" + "=" * 78)
    print("GA -- PRIMARY. The operating point. (§23.6)")
    print(f"    mean spike rate at step_{EXPECTED_FINAL_STEP}.pt on the fixture, "
          "over the pilot seeds")
    print(f"    CONFIRMED: mean in [{GA_CONFIRMED_LOW}, {GA_CONFIRMED_HIGH}]")
    print(f"    FALSIFIED: mean >= {GA_FALSIFIED_AT_OR_ABOVE} (runaway) or "
          f"mean <= {GA_FALSIFIED_AT_OR_BELOW} (starved)")
    print("    AMBIGUOUS: otherwise.  Documented healthy band: ~2%")
    print("=" * 78 + "\n")
    rf_rates = _final_values(rf_traj, seeds, "spike_rate")
    lif_rates = _final_values(lif_traj, seeds, "spike_rate")
    rf_ga = _print_final_row("resonate-and-fire", rf_rates, seeds, ".6f")
    lif_ga = _print_final_row("LIF, re-derived from disk", lif_rates, seeds, ".6f")
    print(f"  {'§23.6 quoted LIF reference':<34}"
          + "  ".join(f"seed{s}={v:.6f}"
                      for s, v in zip(seeds, LIF_QUOTED_GA_PER_SEED)))
    print(f"  {'':<34}{_delta_note(lif_ga, LIF_QUOTED_GA_MEAN)}")
    if rf_ga is None:
        ga = None
        print("\n  GA: NO DATA -- no resonate-and-fire run has a checkpoint yet.")
    else:
        ga = ga_verdict(rf_ga)
        print(f"\n  GA verdict: {ga} -- mean final spike rate {rf_ga:.6f}")
        if ga == "AMBIGUOUS":
            print(f"  {AMBIGUOUS_PHRASE}")

    # ---------------------------------------------------------------- GA2 ---
    print("\n" + "=" * 78)
    print("GA2 -- CO-PRIMARY. Silent units. (§23.6)")
    print("    mean final silent fraction, same instrument, same seeds")
    print(f"    CONFIRMED: < {GA2_CONFIRMED_BELOW:.0%}.  "
          f"FALSIFIED: >= {GA2_FALSIFIED_AT_OR_ABOVE:.0%}.  AMBIGUOUS: between")
    print("=" * 78 + "\n")
    rf_silent = _final_values(rf_traj, seeds, "silent")
    lif_silent = _final_values(lif_traj, seeds, "silent")
    rf_ga2 = _print_final_row("resonate-and-fire", rf_silent, seeds, ".4%")
    lif_ga2 = _print_final_row("LIF, re-derived from disk", lif_silent, seeds, ".4%")
    print(f"  {'§23.6 quoted LIF reference':<34}"
          + "  ".join(f"seed{s}={v:.4%}"
                      for s, v in zip(seeds, LIF_QUOTED_GA2_PER_SEED)))
    print(f"  {'':<34}{_delta_note(lif_ga2, LIF_QUOTED_GA2_MEAN, '.4%')}")
    if rf_ga2 is None:
        ga2 = None
        print("\n  GA2: NO DATA -- no resonate-and-fire run has a checkpoint yet.")
    else:
        ga2 = ga2_verdict(rf_ga2)
        print(f"\n  GA2 verdict: {ga2} -- mean final silent fraction {rf_ga2:.4%}")
        if ga2 == "AMBIGUOUS":
            print(f"  {AMBIGUOUS_PHRASE}")

    # ----------------------------------------------------------------- GB ---
    print("\n" + "=" * 78)
    print("GB -- SECONDARY. Task performance, and the scale-up decision. (§23.6)")
    print(f"    mean_extrinsic_return, '{GB_SELECTION}' selection, "
          f"'{GB_REGIME}' regime, 30 episodes, eval seed 0")
    print(f"    PROMISING: >= {GB_PROMISING_AT_OR_ABOVE}.  "
          f"NOT PROMISING: <= {GB_NOT_PROMISING_AT_OR_BELOW}.  AMBIGUOUS: between")
    print("=" * 78 + "\n")
    rf_returns = eval_returns(rf_results_dir, "reservoir", seeds)
    lif_returns = eval_returns(lif_results_dir, "reservoir", seeds)
    gru_returns = eval_returns(lif_results_dir, "baseline", seeds)
    rf_gb = _print_final_row("resonate-and-fire", rf_returns, seeds, ".4f")
    lif_gb = _print_final_row("LIF, re-derived from disk", lif_returns, seeds, ".4f")
    gru_gb = _print_final_row("GRU baseline, re-derived", gru_returns, seeds, ".4f")
    print(f"  {'§23.6 quoted LIF reference':<34}{_delta_note(lif_gb, LIF_QUOTED_GB_MEAN, '.4f')}")
    print(f"  {'§23.6 quoted GRU reference':<34}{_delta_note(gru_gb, GRU_QUOTED_GB_MEAN, '.4f')}")

    if lif_gb is not None and gru_gb is not None:
        print(f"\n  seed-matched gap (LIF - GRU): {lif_gb - gru_gb:+.4f} "
              f"(§23.6 quotes {QUOTED_SEED_MATCHED_GAP:+.4f})")
        print("  §23.6: the pilot is compared against this SEED-MATCHED figure and "
              "never")
        print("  against the published 10-seed headline gap of -8.9656, which would "
              "flatter it.")
        derived = gb_threshold_from_data(lif_gb, gru_gb)
        agrees = gb_threshold_agrees(derived)
        print(f"\n  PROMISING threshold re-derived from the data: "
              f"lif_mean + (gru_mean - lif_mean)/3 = {derived:.6f}")
        print(f"  pre-registered constant: {GB_PROMISING_AT_OR_ABOVE} "
              f"(delta {derived - GB_PROMISING_AT_OR_ABOVE:+.6f}, tolerance "
              f"{GB_THRESHOLD_AGREEMENT_TOL})")
        if agrees:
            print("  agreement: OK -- §23.11 records the pre-registered constant "
                  "carries a")
            print("  fourth-decimal rounding slip against its own per-seed inputs; "
                  "that is")
            print("  within tolerance and the PRE-REGISTERED constant is the one the "
                  "verdict uses.")
        else:
            print("\n  " + "*" * 74)
            print("  *** WARNING: the threshold re-derived from the data DISAGREES "
                  "with the")
            print("  *** pre-registered constant by more than the tolerance. This "
                  "guard exists")
            print("  *** to catch a DATA-HANDLING error -- the wrong seeds, the wrong "
                  "regime,")
            print("  *** or the wrong selection read out of the results directory. "
                  "GB below is")
            print("  *** still scored against the PRE-REGISTERED constant, because a "
                  "threshold")
            print("  *** adjusted after measurement is exactly what §23 exists to "
                  "prevent --")
            print("  *** but do not trust the GB numbers until this is explained.")
            print("  " + "*" * 74)
    else:
        agrees = None
        print("\n  threshold agreement check: SKIPPED -- the LIF and/or GRU "
              "reference returns")
        print("  are not both present, so there is nothing to re-derive the "
              "threshold from.")

    wins, compared = sign_test_wins(rf_returns, lif_returns)
    print(f"\n  per-seed sign test (supporting statistic ONLY): {wins}/{compared} "
          "pilot seeds beat")
    print("  their own LIF counterpart.")
    print("\n" + _wrap(UNDERPOWERED_DISCLAIMER))

    if rf_gb is None:
        gb = None
        print("\n  GB: NO DATA -- no resonate-and-fire evaluation JSON is present.")
    else:
        gb = gb_verdict(rf_gb)
        print(f"\n  GB verdict: {gb} -- mean extrinsic return {rf_gb:.4f}")

    # ------------------------------------------------------- decision rule ---
    print("\n" + "=" * 78)
    print("THE DECISION RULE (§23.7)")
    print("=" * 78)
    print(f"\n  GA  (primary)    : {ga or 'NO DATA'}")
    print(f"  GA2 (co-primary) : {ga2 or 'NO DATA'}")
    print(f"  GB  (secondary)  : {gb or 'NO DATA'}")
    if ga is None or gb is None:
        decision = None
        print("\n  DECISION: NOT COMPUTABLE -- §23.7's rule is a function of GA and "
              "GB, and")
        print("  at least one of them has no data. Reported as missing rather than "
              "defaulted;")
        print("  a decision string printed off an absent measurement is worse than "
              "no string.")
    else:
        decision = decision_rule(ga, gb)
        print(f"\n  DECISION: {decision}")
        if decision == DECISION_SCALE_UP:
            print("\n  §23.7: even then, THIS SESSION DOES NOT LAUNCH THE 10-SEED "
                  "MATRIX. The")
            print("  recommendation is reported and the decision is the project "
                  "owner's.")
        elif decision == DECISION_INFORMATIVE_NEGATIVE:
            print("\n  §23.7 wrote this outcome down in advance precisely so it "
                  "cannot later be")
            print("  reframed as a disappointment: it would establish that §21.5's "
                  "'most concrete")
            print("  open architectural question' is real, is fixable at zero "
                  "parameter cost by a")
            print("  neuron-model swap, and is NOT what costs the frozen reservoir "
                  "the comparison.")
    if ga is not None and ga2 is not None and ga != ga2:
        print(f"\n  NOTE: the two PRIMARY gates disagree (GA {ga}, GA2 {ga2}). §23.7's "
              "rule")
        print("  conditions on GA only; the disagreement is reported here rather "
              "than resolved,")
        print("  because resolving it would be a post-hoc rule §23 did not "
              "pre-register.")

    section_unconditional(rf_traj, lif_traj, seeds, rf_checkpoint_dir,
                          lif_checkpoint_dir, rf_results_dir, lif_results_dir,
                          obs, mu, embed_scale)
    return ga, ga2, gb, decision


# --------------------------------------------------------------------------- #
# §23.8 -- reported unconditionally, whatever the verdicts
# --------------------------------------------------------------------------- #

def section_unconditional(rf_traj, lif_traj, seeds, rf_checkpoint_dir,
                          lif_checkpoint_dir, rf_results_dir, lif_results_dir,
                          obs, mu, embed_scale):
    print("\n" + "=" * 78)
    print("REPORTED UNCONDITIONALLY, WHATEVER THE VERDICTS (§23.8)")
    print("    no band, no gate -- these exist so the pilot is readable by someone")
    print("    who disagrees with the bands above")
    print("=" * 78)

    # ---- 1. the full trajectories, in results_v2_health.txt's own columns ----
    print("\n  1. FULL PER-CHECKPOINT TRAJECTORIES, both arms, same columns as")
    print("     results_v2_health.txt. §23.8: the trajectory is the actual object of")
    print("     interest -- whether the operating point STAYS in band, not only where")
    print("     it happens to end.")
    for label, trajectories, checkpoint_dir in (
            ("resonate-and-fire", rf_traj, rf_checkpoint_dir),
            ("LIF (control)", lif_traj, lif_checkpoint_dir)):
        print(f"\n  --- {label}: {checkpoint_dir}")
        print(A9_TRAJECTORY_HEADER)
        print("  " + "-" * 100)
        any_rows = False
        for seed in seeds:
            stats = trajectories[seed]
            if stats["status"] != "ok":
                print(f"  seed{seed}: {stats['status']} -- no trajectory")
                continue
            any_rows = True
            if label.startswith("resonate") and embed_scale is not None:
                init = init_operating_point(seed, embed_scale, "rf", obs, mu)
                print(a9_trajectory_row(f"seed{seed}", dict(init, step="init")))
            for i, row in enumerate(stats["trajectory"]):
                first = i == 0 and not (label.startswith("resonate")
                                        and embed_scale is not None)
                print(a9_trajectory_row(f"seed{seed}" if first else "", row))
            head = stats["trajectory"][0]
            if head["neuron_model"] != "lif":
                print(f"  {'':<10}neuron_model={head['neuron_model']}  "
                      f"mean DC gain |1/(1-beta*e^iw)| = {head['dc_gain_mean']:.4f} "
                      f"(LIF: 10.0)  mean real-part factor = "
                      f"{head['dc_offset_factor_mean']:.4f}  "
                      f"-- the offset column uses the REAL PART (§23.10(b))")
            if not stats["final_is_expected_final_step"]:
                print(f"  {'':<10}(PARTIAL RUN -- final on disk is step "
                      f"{stats['final']['step']}, expected final is "
                      f"{EXPECTED_FINAL_STEP})")
            print()
        if not any_rows:
            print("  (no run in this arm has a checkpoint yet)\n")
    if embed_scale is None:
        print("  NOTE: --embed-scale was not given, so the rf arm's step-0 init row")
        print("  is omitted. Pass the scale the pilot was launched at to anchor the")
        print("  trajectory at the operating point §23.4's calibration selected.")

    # ------------------ 2. mean per-update extrinsic training reward --------
    print("\n  2. MEAN PER-UPDATE EXTRINSIC *TRAINING* REWARD, over all updates.")
    print("     §23.8 reports this as a DIAGNOSTIC and NOT as a gate: v2's sharpest")
    print("     single contrast was that the corrections closed the training-reward")
    print("     gap from 5.82x to 1.38x while the evaluation gap did not close at")
    print("     all, so training reward is precisely the quantity already shown not")
    print("     to predict the scoreboard here.")
    print(f"\n  {'run':<34}{'mean extrinsic reward':>23}{'updates':>10}")
    print("  " + "-" * 67)
    for label, checkpoint_dir, arm in (
            ("resonate-and-fire", rf_checkpoint_dir, "reservoir"),
            ("LIF (control)", lif_checkpoint_dir, "reservoir"),
            ("GRU baseline", lif_checkpoint_dir, "baseline")):
        per_seed = []
        for seed in seeds:
            value, n = train_log_extrinsic_mean(checkpoint_dir, arm, seed)
            per_seed.append(value)
            print(f"  {f'{label} seed{seed}':<34}{_fmt(value, '.6f', 23):>23}"
                  f"{n if n else 'n/a':>10}")
        mean = mean_or_none(per_seed)
        print(f"  {f'{label} MEAN':<34}{_fmt(mean, '.6f', 23):>23}")
        if label.startswith("LIF"):
            print(f"  {'':<34}{_delta_note(mean, LIF_QUOTED_TRAIN_REWARD)}")
        elif label.startswith("GRU"):
            print(f"  {'':<34}{_delta_note(mean, GRU_QUOTED_TRAIN_REWARD)}")
        print()
    print(f"  §23.8's quoted references: LIF {LIF_QUOTED_TRAIN_REWARD} (per-seed "
          f"{', '.join(f'{v:.6f}' for v in LIF_QUOTED_TRAIN_REWARD_PER_SEED)}), "
          f"GRU {GRU_QUOTED_TRAIN_REWARD},")
    print(f"  over {QUOTED_TOTAL_UPDATES} updates.")

    # -------------- 3. every selection x regime cell, not just GB's ---------
    print("\n  3. EVERY SELECTION x REGIME CELL, not only GB's final/continuous,")
    print("     'so the pilot cannot be read only through its most favourable cell'")
    print("     (§23.8). mean_extrinsic_return, per seed and seed-mean.")
    for selection in EVAL_SELECTIONS:
        for regime in EVAL_REGIMES:
            marker = " <- GB's cell" if (selection, regime) == (GB_SELECTION, GB_REGIME) else ""
            print(f"\n  --- {selection} / {regime}{marker}")
            print(f"  {'arm':<34}"
                  + "".join(f"{'seed' + str(s):>12}" for s in seeds)
                  + f"{'mean':>12}")
            print("  " + "-" * (36 + 12 * (len(seeds) + 1)))
            for label, results_dir, arm in (
                    ("resonate-and-fire", rf_results_dir, "reservoir"),
                    ("LIF (control)", lif_results_dir, "reservoir"),
                    ("GRU baseline", lif_results_dir, "baseline")):
                values = eval_returns(results_dir, arm, seeds, selection, regime)
                print(f"  {label:<34}"
                      + "".join(f"{_fmt(v, '.4f', 12):>12}" for v in values)
                      + f"{_fmt(mean_or_none(values), '.4f', 12):>12}")

    print("\n  4. THROUGHPUT, both neuron models, on a quiet machine (§23.8): not")
    print("     measured here -- it needs the emulator and an otherwise-idle machine,")
    print("     and this module neither trains nor rolls out. Same G0d caveat as the")
    print("     preflight stage reports.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="The resonate-and-fire pilot's pre-registered gates "
                    "(docs/EXPERIMENT_LOG.md §23), computed in code. Read-only, "
                    "writes no files, trains nothing, and degrades gracefully on a "
                    "matrix that has not run yet.")
    parser.add_argument("--stage", choices=["preflight", "verdict", "spectrum"],
                        required=True,
                        help="'preflight' runs §23.4/§23.5's construction gates "
                             "BEFORE training; 'verdict' runs §23.6/§23.7/§23.8 "
                             "after training and evaluation; 'spectrum' measures "
                             "where the observation's energy actually is and what "
                             "the filter bank does with it, and gates nothing")
    parser.add_argument("--rf-checkpoint-dir", default="checkpoints_rf_pilot")
    parser.add_argument("--lif-checkpoint-dir", default="checkpoints_v2",
                        help="the published v2 matrix -- this pilot's CONTROL "
                             "(§23.9). Re-measured here by the same code that "
                             "measures the rf arm, never transcribed.")
    parser.add_argument("--rf-results-dir", default="results_rf_pilot")
    parser.add_argument("--lif-results-dir", default="results_v2")
    parser.add_argument("--seeds", default="0-2",
                        help="the three pilot seeds (default '0-2'); also accepts "
                             "'0,2,5-7'")
    parser.add_argument("--embed-scale", type=float, default=None,
                        help="verdict stage: the --embed-scale the pilot was "
                             "launched at, used to rebuild the rf arm's step-0 "
                             "init row in §23.8's trajectory table. spectrum "
                             f"stage: the init to measure at, default "
                             f"{SPECTRUM_EMBED_SCALE} (§23.12's selected point)")
    args = parser.parse_args(argv)

    seeds = parse_seed_spec(args.seeds)
    rf_checkpoint_dir = resolve_dir(args.rf_checkpoint_dir)
    lif_checkpoint_dir = resolve_dir(args.lif_checkpoint_dir)
    rf_results_dir = resolve_dir(args.rf_results_dir)
    lif_results_dir = resolve_dir(args.lif_results_dir)

    print("=" * 78)
    print("GameSpike resonate-and-fire pilot -- docs/EXPERIMENT_LOG.md §23")
    print(f"  stage           : {args.stage}")
    print(f"  repo            : {REPO_ROOT}")
    print(f"  torch           : {torch.__version__}")
    print(f"  pilot seeds     : {seeds}")
    spectrum_scale = (args.embed_scale if args.embed_scale is not None
                      else SPECTRUM_EMBED_SCALE)
    if args.stage == "preflight":
        print(f"  geometry        : reservoir_size={RESERVOIR_SIZE}, "
              f"tt_rank={TT_RANK}, tt_n_cores={TT_N_CORES} (§23.2)")
        print(f"  embed-scale grid: {list(EMBED_SCALE_GRID)} (§23.4)")
    elif args.stage == "spectrum":
        print(f"  geometry        : reservoir_size={RESERVOIR_SIZE}, "
              f"tt_rank={TT_RANK}, tt_n_cores={TT_N_CORES} (§23.2)")
        print(f"  rf init         : neuron_model=rf, "
              f"embed_init_mode={RF_INIT_EMBED_MODE}, embed_scale={spectrum_scale:g}")
        print(f"  welch           : segment {WELCH_SEGMENT_LEN}, "
              f"{WELCH_OVERLAP:.0%} overlap, periodic Hann, numpy only")
        print("  GATES NOTHING   : no band here is pre-registered and none is "
              "recommended")
    else:
        print(f"  rf checkpoints  : {rf_checkpoint_dir}")
        print(f"  lif checkpoints : {lif_checkpoint_dir}")
        print(f"  rf results      : {rf_results_dir}")
        print(f"  lif results     : {lif_results_dir}")
        print(f"  rf embed-scale  : {args.embed_scale if args.embed_scale else 'not given'}")
    print("  READ-ONLY: this module writes nothing and trains nothing. Every")
    print("  checkpoint, run directory and result JSON is optional; a missing one is")
    print("  reported and skipped, never raised.")
    print("=" * 78)

    if args.stage == "preflight":
        stage_preflight(seeds)
    elif args.stage == "spectrum":
        stage_spectrum(seeds, spectrum_scale)
    else:
        stage_verdict(rf_checkpoint_dir, lif_checkpoint_dir, rf_results_dir,
                      lif_results_dir, seeds, args.embed_scale)


if __name__ == "__main__":
    main()
