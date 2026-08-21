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
EMBED_SCALE_GRID = (3.0, 4.5, 6.0, 9.0, 12.0, 18.0)

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
    parser.add_argument("--stage", choices=["preflight", "verdict"], required=True,
                        help="'preflight' runs §23.4/§23.5's construction gates "
                             "BEFORE training; 'verdict' runs §23.6/§23.7/§23.8 "
                             "after training and evaluation")
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
                             "init row in §23.8's trajectory table")
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
    if args.stage == "preflight":
        print(f"  geometry        : reservoir_size={RESERVOIR_SIZE}, "
              f"tt_rank={TT_RANK}, tt_n_cores={TT_N_CORES} (§23.2)")
        print(f"  embed-scale grid: {list(EMBED_SCALE_GRID)} (§23.4)")
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
    else:
        stage_verdict(rf_checkpoint_dir, lif_checkpoint_dir, rf_results_dir,
                      lif_results_dir, seeds, args.embed_scale)


if __name__ == "__main__":
    main()
