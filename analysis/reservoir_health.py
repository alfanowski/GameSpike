"""Re-runnable measurement script for the two pre-registered ablations A7 and A9.

WHAT THIS DECIDES, and for whom. `docs/EXPERIMENT_LOG.md` pre-registers two
hypotheses about the v2 corrected-input reservoir matrix, BEFORE the matrix
finished running (§14.5 for A7, §15.6 for A9), specifically so the verdicts below
cannot be tuned to a result after the fact. This module computes both, from
whatever checkpoints happen to be on disk when it is invoked, and prints a
report. It is READ-ONLY with respect to every checkpoint directory it can be
pointed at, it never trains, and it writes no files. It is meant to be run
AGAINST A PARTIALLY-COMPLETE MATRIX WHILE TRAINING IS STILL WRITING NEW
CHECKPOINTS -- so every run directory and every checkpoint file it looks for is
treated as optional, and a missing one is reported, not raised.

  A7 (§14.5) -- does the corrected embedding input also fix the dead-gradient
  budget? §12/A4a measured, on the v1 (uncorrected) reservoir arm, that 865 of
  8192 `in_proj` columns (10.5591%) never received a single Adam update by the
  final checkpoint, that the dead set is a strict superset of the permanently
  silent set (`dead \\ silent = 0`), and that it only ever SHRINKS during
  training (`dead(t+1) subset dead(t)`, `newly_dead = 0` at every one of 9
  transitions). A7 replicates that exact procedure -- Adam `exp_avg_sq` exactly
  0 -- on the v2 corrected-input runs, to see whether removing the silent units
  (the fix §12 shipped) also removes the dead columns downstream of them.

  A9 (§15.6) -- where does the reservoir's operating point actually end up over
  a FULL run? §15.5 found, on the complete v1 run, that centring the embedding
  fixes the *initial* operating point but nothing regulates where it drifts to
  afterwards: by the final checkpoint the residual DC had grown 6x and the
  membrane-offset std had grown from 0.94 to 5.63. A9 measures, per checkpoint
  of every v2 run, whether the *corrected* input's silent-fraction advantage
  over legacy (~46%) survives to the end of a full run or decays back toward it.

WHY THIS REUSES `analysis/pilot_diagnostics.py`'S MACHINERY RATHER THAN
REIMPLEMENTING IT. That module already contains reviewed, working
implementations of the three primitives both ablations need:

  * `silent_fraction` -- VERBATIM from `tests/test_embedding_centering.py`,
    which that test's own docstring explains at length is not interchangeable
    with a synthetic surrogate (24.12% silent on an i.i.d.-matched Gaussian vs
    1.66% on the real fixture, and pointing the WRONG direction) because real
    observations are strongly temporally correlated and a beta=0.9 LIF membrane
    integrates that low-frequency energy with gain up to 1/(1-beta)=10. Silently
    re-deriving this measurement here would risk measuring something subtly
    different from the number both fixes were accepted on.
  * `reservoir_at` -- reconstructs the reservoir-arm model exactly as
    `run_training` built it (`torch.manual_seed(seed)` then `build_model(...,
    seed=seed)`, because the trainable init draws from the GLOBAL RNG while the
    frozen reservoir takes its own `seed=` argument, and only doing both
    reproduces what `--seed s` actually produced), loads a checkpoint into it,
    and diffs the frozen `reservoir.*` buffers against a pre-load reference
    copy -- an independent, per-load re-verification of spec §3's frozen
    invariant, not a one-time assumption. `pilot_diagnostics.py` found 0.0e+00
    max abs diff across 27 such loads; this module repeats that check on every
    checkpoint A9 touches and reports the running max, per the task that
    "verifying while loading anyway" is the cheap way to keep re-confirming it.
  * `load_ckpt` -- `torch.load(..., weights_only=True)`, for the same
    code-execution-risk reason `training/train.py` uses it.

`reservoir_at`'s `embed_init_mode`/`embed_scale` arguments are always passed as
`"legacy", 1.0` below, REGARDLESS of what a given checkpoint was actually
trained under. This is deliberate and provably harmless, not an oversight: those
two arguments only affect (a) the std used to draw `embedding.weight`'s initial
values and (b) which deterministic (non-random) function initialises
`embedding.bias` -- neither changes the NUMBER of draws taken from the global
RNG, so the RNG state handed to the subsequently-constructed `readout` module
(and therefore its would-be random init) is identical either way, and the
`reservoir.*` buffers being diffed come from `SpikingReservoir`'s OWN seeded
generator, which never touches the global RNG at all. Once `ckpt_path` is given
(every call in this module supplies one), `load_state_dict` overwrites
`embedding.weight`, `embedding.bias` and every `readout.*` tensor wholesale
regardless of what they were constructed with, and the measurements below (spike
statistics, weight norms, DC drift) are all read off the tensors AFTER that
overwrite. So the two arguments are inert here; hardcoding them avoids re-reading
each checkpoint a second time merely to peek at its own `embed_init_mode` label
before `reservoir_at` reads it a third time internally.

ORIENTATION COMMITMENT (`docs/EXPERIMENT_LOG.md` §11.1, restated here because
getting this backwards produces a wrong-but-plausible number and silently
invalidates A7). `ActorCriticReadout.in_proj` is `nn.Linear(reservoir_size,
d_model)` (`models/actor_critic_readout.py:21`), so `in_proj.weight` has shape
`(d_model, reservoir_size) = (16, 8192)`. A reservoir unit therefore indexes a
COLUMN of `in_proj.weight` -- dim 1 -- not a row. Unit `j` owns the 16 entries
`in_proj.weight[:, j]`, and a column is "dead" iff ALL 16 of those entries have
Adam `exp_avg_sq` exactly 0. `dead_mask_from_exp_avg_sq` below asserts the shape
looks the right way round (d_model « reservoir_size) before reducing over dim 0,
and `dead_column_mask` additionally asserts the concrete expected shape
`(16, 8192)`, so a construction change that silently altered either dimension
fails loudly here rather than producing a quietly-wrong fraction.

GRACEFUL PARTIAL OPERATION. The v2 matrix is a live 10-process training run at
the time this module is written (`docs/EXPERIMENT_LOG.md` §17), and every run
directory this module can be pointed at may not exist yet, or may exist with a
`train_log.jsonl` but zero `step_*.pt` checkpoint files (this was true of every
`checkpoints_v2/reservoir_seed{0..9}/` directory when this module was written --
verified, not assumed). `find_run` reports one of three statuses
(`"ok"`/`"missing_dir"`/`"no_checkpoints"`) for every seed instead of raising, so
a run against an in-progress matrix produces a smaller-but-honest report rather
than a crash, and the aggregate statistics and verdicts below are computed only
over seeds that actually contributed data -- with the seed count that went into
each aggregate printed alongside it, so a mean of 2 seeds is never mistaken for a
mean of 10.

DETERMINISM. No randomness is drawn at measurement time beyond the SAME
`torch.manual_seed(seed)` / `SpikingReservoir(seed=seed)` calls that constructed
the checkpoint's own reservoir in the first place (needed to reconstruct the
model to load weights into, and to re-verify the frozen buffers); `model.eval()`
is set before any forward pass. Same checkpoints on disk => same report, every
time. This module writes no files.
"""
import argparse
import glob
import os
import re
import statistics
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from analysis.pilot_diagnostics import (REAL_OBS_PATH, load_ckpt,  # noqa: E402
                                        reservoir_at, silent_fraction,
                                        trainable_names)
from envs.mario_land_env import OBS_MEAN  # noqa: E402
from training.train import neuron_config_from_checkpoint  # noqa: E402

# --------------------------------------------------------------------------- #
# constants -- the production geometry (training/train.py:build_model,
# models/policy_value_reservoir.py) and the two pre-registrations' own numbers
# --------------------------------------------------------------------------- #

RESERVOIR_SIZE = 8192
D_MODEL = 16
TRAINABLE_BUDGET = 139_179          # §11.1 / models/policy_value_reservoir.py
IN_PROJ_PARAM_NAME = "readout.in_proj.weight"

# The step a full v1-length run's final checkpoint carries (§12: 1,000,064 =
# 7,813 updates * 128 steps/update). Informational only: a v2 run interrupted
# before this step still gets measured, and the report says so rather than
# silently treating an earlier checkpoint as "the" final one.
EXPECTED_FINAL_STEP = 1_000_064

# A7 (§14.5): confirmed if mean dead fraction over seeds < 2% of 8192 (~164
# columns); falsified if >= 5% (~410 columns); the band between is ambiguous.
A7_CONFIRMED_BELOW = 0.02
A7_FALSIFIED_AT_OR_ABOVE = 0.05

# A9 (§15.6): confirmed if mean final silent fraction over seeds < 40%;
# falsified if >= 46% (legacy's measured ~46%); the band between is ambiguous.
A9_CONFIRMED_BELOW = 0.40
A9_FALSIFIED_AT_OR_ABOVE = 0.46

# Both pre-registrations use this exact phrase for their ambiguous band and
# require it be "reported in exactly those words".
AMBIGUOUS_PHRASE = "confirms the direction while falsifying the magnitude"

_CHECKPOINT_FILENAME_RE = re.compile(r"^step_(?P<step>\d+)\.pt$")

# A9's trajectory table layout, factored out of `section_a9` rather than left
# inline, because §23.8 requires the resonate-and-fire pilot's own trajectories be
# printed in "the same columns `results_v2_health.txt` reports, so the two are
# readable side by side". `analysis/rf_pilot.py` imports these; a second
# hand-copied format string would drift on the first column-width tweak and the
# side-by-side reading is the whole point of the requirement.
A9_TRAJECTORY_HEADER = (f"  {'run':<10}{'step':>10}{'silent':>10}{'spike rate':>13}"
                        f"{'saturated':>11}{'||W||':>9}{'|W@mu+b|':>11}"
                        f"{'offset std':>12}{'frozen drift':>14}")


def a9_trajectory_row(label: str, row: dict) -> str:
    """One line of the A9 trajectory table. `row["step"]` may be an int or a
    string (`"init"`), and `row["frozen_drift"]` may be None for a model that was
    built rather than loaded -- both render at the same widths, so a table that
    mixes them still lines up with the committed one."""
    drift = row["frozen_drift"]
    drift_s = "n/a (built)" if drift is None else f"{drift:.1e}"
    return (f"  {label:<10}{row['step']:>10}{row['silent']:>9.4%}"
            f"{row['spike_rate']:>13.6f}{row['saturated']:>10.4%}"
            f"{row['w_norm']:>9.4f}{row['dc_norm']:>11.4f}"
            f"{row['offset_std']:>12.4f}{drift_s:>14}")


# --------------------------------------------------------------------------- #
# pure logic -- no torch checkpoint I/O, fully covered by
# tests/test_reservoir_health.py on synthetic fixtures
# --------------------------------------------------------------------------- #

def parse_seed_spec(spec: str) -> list:
    """`"0-9"` -> `[0..9]`; also accepts comma-separated ints/ranges (`"0,2,5-7"`)."""
    seeds = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(chunk))
    return seeds


def select_subset(items: list, max_n) -> list:
    """Cap a sorted-by-step checkpoint list to `max_n` items, keeping the FIRST
    and LAST always (so a capped run still reports its final checkpoint -- the
    one both verdicts are computed from -- rather than silently dropping it)
    and spreading the rest evenly across the index range in between.

    `max_n=None` or `max_n >= len(items)` returns `items` unchanged.
    """
    n = len(items)
    if max_n is None or n <= max_n:
        return list(items)
    if max_n <= 1:
        return [items[-1]]
    idxs = sorted({round(i * (n - 1) / (max_n - 1)) for i in range(max_n)})
    return [items[i] for i in idxs]


def dead_mask_from_exp_avg_sq(exp_avg_sq: torch.Tensor) -> torch.Tensor:
    """The orientation commitment (§11.1), made executable and asserted.

    `exp_avg_sq` is expected shaped `(d_model, reservoir_units)` -- a reservoir
    unit indexes dim 1 (a COLUMN), never dim 0. A column is dead iff EVERY entry
    in it (all `d_model` rows) is exactly 0.0, i.e. that `in_proj` column never
    received a single nonzero gradient. Returns a `(reservoir_units,)` bool mask.

    The shape assertion is deliberately generic (`d_model < reservoir_units`)
    rather than hardcoding `(16, 8192)`, so this function is testable on small
    synthetic tensors; `dead_column_mask` below additionally asserts the
    concrete production shape once real checkpoints are involved.
    """
    assert exp_avg_sq.dim() == 2, (
        f"expected a 2D weight-shaped tensor, got shape {tuple(exp_avg_sq.shape)}"
    )
    d_model, reservoir_units = exp_avg_sq.shape
    assert d_model < reservoir_units, (
        f"exp_avg_sq shape {(d_model, reservoir_units)} looks backwards: d_model "
        "must be « reservoir_size, and a reservoir unit indexes dim 1 (columns), "
        "never dim 0 -- see docs/EXPERIMENT_LOG.md §11.1's orientation commitment. "
        "Getting this axis backwards silently produces a wrong-but-plausible number."
    )
    return (exp_avg_sq == 0).all(dim=0)


def nesting_and_newly_dead(mask_sequence: list):
    """Given dead-masks in ascending-step order, return `(newly_dead, holds)`.

    `newly_dead[i]` is the count of columns dead at transition i that were NOT
    dead at the previous checkpoint -- i.e. a column that died mid-training,
    which A4a found never happens (`newly_dead = 0` at all 9 v1 transitions,
    `dead(t+1) subset dead(t)`). `holds` is True iff every transition's
    `newly_dead` is 0, independent of the magnitude of either mask.
    """
    newly_dead = []
    for prev, nxt in zip(mask_sequence, mask_sequence[1:]):
        newly_dead.append(int((nxt & ~prev).sum()))
    return newly_dead, all(n == 0 for n in newly_dead)


def dc_offset_factor(beta, omega=None):
    """Per-unit factor turning a CONSTANT input current into the standing membrane
    offset it induces -- the REAL PART of the complex DC gain (§23.10(b)).

    WHY THE REAL PART AND NOT THE MAGNITUDE, which is the quantity §23.3's table
    and G0a are stated in. A resonate-and-fire unit carries a two-dimensional
    state `(u, v)`, and `|1/(1 - beta*e^{i w})|` measures how much DC energy that
    two-dimensional state holds. But spiking is thresholded on `u` ALONE, so what
    actually shifts a unit's firing threshold is the `u`-component of the
    steady state, i.e. the real part

        (1 - beta*cos w) / (1 - 2*beta*cos w + beta^2)

    which is strictly smaller than the magnitude at every frequency where the
    pole is genuinely complex. §23.10(b) records that reporting the magnitude in
    the offset column instead would overstate the resonate-and-fire arm's standing
    offset by roughly the same factor H10 predicts it to FALL by -- an arithmetic
    error indistinguishable from the result it was measuring. Both quantities are
    reported (see `checkpoint_operating_point`); this is the one the offset column
    uses.

    `omega=None` IS the LIF arm and returns the scalar `1/(1 - beta)`. Not a
    tensor of zeros: `SpikingReservoir.omega` deliberately RAISES on a LIF
    reservoir rather than handing back zeros, precisely so rf-specific analysis
    cannot run silently against the control arm, and this signature mirrors that.

    THE `w == 0` BRANCH IS EXPLICIT, AND IT IS NOT REDUNDANT. At `cos w = 1` the
    general expression is algebraically `(1-beta)/(1-beta)^2 = 1/(1-beta)`, but
    IEEE-754 does not respect that algebra: `(1-beta)` and `(1-beta)^2` are each
    rounded, and their quotient can miss `1/(1-beta)` by an ULP -- in float64 it
    does. The whole point of §23.2's construction is that LIF is the `w = 0` point
    of this family BIT FOR BIT, so the zero entries are taken from the LIF formula
    directly and the identity becomes a tested property (tests/test_rf_pilot.py)
    rather than an appeal to the algebra. The general branch is written over
    `1 - cos w` rather than `cos w` for the same reason it is written that way in
    the numerator: at small omega, `1 - cos w` is the well-conditioned form and
    subtracting two nearly-equal numbers is not.
    """
    beta = float(beta)
    lif_factor = 1.0 / (1.0 - beta)
    if omega is None:
        return lif_factor
    omega = torch.as_tensor(omega)
    one_minus_cos = 1.0 - torch.cos(omega)
    d = 1.0 - beta
    general = (d + beta * one_minus_cos) / (d * d + 2.0 * beta * one_minus_cos)
    return torch.where(one_minus_cos == 0,
                       torch.full_like(general, lif_factor), general)


def induced_membrane_offset(drive, beta, omega=None):
    """The standing per-unit membrane offset a DC input current induces.

    `drive` is `W_in @ (W @ mu + b)` -- the constant part of the reservoir's input
    current, the quantity §12 measured at std 0.943583 against a threshold of 1.0
    for the legacy init. This is a LINEARISED input-driven estimate in both neuron
    models: it omits the frozen recurrent TT term, so it explains the silent
    fraction's direction and magnitude without equalling it.

    THE ZERO-FREQUENCY UNITS ARE A DIVISION, NOT A MULTIPLICATION BY
    `dc_offset_factor`, and that is load-bearing rather than stylistic.
    `x / (1-beta)` and `x * (1/(1-beta))` differ by up to one ULP because the
    reciprocal is itself rounded. Two things ride on closing that gap:

      * A9's trajectory table is committed to four decimal places in
        `results_v2_health.txt` and this change has to reproduce it byte for
        byte, so the LIF arm has to keep the exact expression it always had;
      * §23.2's claim is that LIF is the `w = 0` point of the resonate-and-fire
        family BIT FOR BIT, and G0e-ii tests exactly that at the cell. Carrying
        the same property up to the reported column costs one elementwise select
        on a diagnostic path and makes "an rf reservoir drawn at omega == 0
        reports the identical offset column" a tested identity rather than an
        agreement to within rounding.
    """
    denominator = 1.0 - float(beta)
    if omega is None:
        return drive / denominator
    return torch.where(torch.as_tensor(omega) == 0, drive / denominator,
                       drive * dc_offset_factor(beta, omega))


def band_verdict(value: float, confirmed_below: float, falsified_at_or_above: float) -> str:
    """`"CONFIRMED"` / `"FALSIFIED"` / `"AMBIGUOUS"`, per a pre-registration's
    own three-way band. `value < confirmed_below` confirms, `value >=
    falsified_at_or_above` falsifies, the closed-open band between is ambiguous.
    """
    if value < confirmed_below:
        return "CONFIRMED"
    if value >= falsified_at_or_above:
        return "FALSIFIED"
    return "AMBIGUOUS"


# --------------------------------------------------------------------------- #
# on-disk discovery -- every one of these degrades gracefully, never raises,
# because the matrix this module is pointed at may still be training
# --------------------------------------------------------------------------- #

def run_dir(checkpoint_dir: str, arm: str, seed: int) -> str:
    """`{checkpoint_dir}/{arm}_seed{seed}` -- matches `training.train.run_dir_for`
    with `run_tag=None`, which is what every v1 AND v2 run uses (§14.9: v2
    deliberately puts its version coordinate in the parent directory, not a
    run-tag suffix, so this shape is unchanged from v1)."""
    return os.path.join(checkpoint_dir, f"{arm}_seed{seed}")


def find_run(checkpoint_dir: str, arm: str, seed: int):
    """`(status, directory, checkpoints)`.

    `status` is one of:
      "ok"             -- directory exists and has >=1 `step_*.pt` file.
      "missing_dir"    -- the run directory does not exist at all.
      "no_checkpoints" -- the directory exists (e.g. `train_log.jsonl` and/or
                          `launcher.log` are already there) but training has
                          not written a checkpoint yet.
    `checkpoints` is a `[(step:int, path:str), ...]` list sorted ascending by
    step, parsed from the filename (never assumed round -- `training/train.py`
    saves whatever step a rollout boundary landed on).
    """
    d = run_dir(checkpoint_dir, arm, seed)
    if not os.path.isdir(d):
        return "missing_dir", d, []
    found = []
    for path in glob.glob(os.path.join(d, "step_*.pt")):
        m = _CHECKPOINT_FILENAME_RE.match(os.path.basename(path))
        if m:
            found.append((int(m.group("step")), path))
    if not found:
        return "no_checkpoints", d, []
    return "ok", d, sorted(found)


# --------------------------------------------------------------------------- #
# A7 -- dead-gradient in_proj columns
# --------------------------------------------------------------------------- #

def dead_column_mask(ckpt: dict, names: list) -> torch.Tensor:
    """`in_proj.weight`'s dead-column mask from one checkpoint's stored
    `optimizer.state_dict()`, replicating A4a's exact procedure (§12): Adam
    `exp_avg_sq` exactly 0 means that parameter never received a nonzero
    gradient across the whole run, no gradient re-run or emulator needed --
    the stored moment IS the evidence.
    """
    opt = ckpt["optimizer"]
    groups = opt["param_groups"]
    assert len(groups) == 1, f"expected one Adam param group, found {len(groups)}"
    idx = names.index(IN_PROJ_PARAM_NAME)
    exp_avg_sq = opt["state"][idx]["exp_avg_sq"]
    assert exp_avg_sq.shape == (D_MODEL, RESERVOIR_SIZE), (
        f"{IN_PROJ_PARAM_NAME} exp_avg_sq has shape {tuple(exp_avg_sq.shape)}, "
        f"expected the production geometry ({D_MODEL}, {RESERVOIR_SIZE})"
    )
    return dead_mask_from_exp_avg_sq(exp_avg_sq)


def a7_run_stats(checkpoint_dir: str, arm: str, seed: int, names: list, max_ckpts):
    """One seed's A7 trajectory, or a skip record if the run isn't ready yet."""
    status, d, ckpts = find_run(checkpoint_dir, arm, seed)
    if status != "ok":
        return {"seed": seed, "status": status, "dir": d}
    selected = select_subset(ckpts, max_ckpts)
    trajectory = []
    for step, path in selected:
        mask = dead_column_mask(load_ckpt(path), names)
        count = int(mask.sum())
        trajectory.append({"step": step, "mask": mask, "count": count,
                            "fraction": count / RESERVOIR_SIZE})
    newly_dead, nesting_holds = nesting_and_newly_dead([r["mask"] for r in trajectory])
    final = trajectory[-1]
    return {
        "seed": seed, "status": "ok", "dir": d, "trajectory": trajectory,
        "newly_dead": newly_dead, "nesting_holds": nesting_holds, "final": final,
        "final_is_expected_final_step": final["step"] == EXPECTED_FINAL_STEP,
        "n_checkpoints_on_disk": len(ckpts), "n_checkpoints_used": len(selected),
    }


# --------------------------------------------------------------------------- #
# A9 -- the reservoir's operating point over a full run
# --------------------------------------------------------------------------- #

def checkpoint_operating_point(seed: int, ckpt_path: str, obs: torch.Tensor, mu: torch.Tensor):
    """One checkpoint's A9 measurements plus the frozen-buffer drift found while
    reconstructing the model to take them (see module docstring for why
    `"legacy", 1.0` are hardcoded here rather than read off the checkpoint).

    THE NEURON MODEL, UNLIKE `embed_init_mode`/`embed_scale`, IS READ OFF THE
    CHECKPOINT (§23). The module docstring's argument for hardcoding the embedding
    arguments turns on `load_state_dict` overwriting whatever they produced; that
    argument does NOT extend to `neuron_model`, because an `rf` checkpoint carries
    five `reservoir.rf.*` buffers a LIF model does not have, so the load REFUSES
    the file instead of overwriting it. There is no build-it-wrong-and-let-the-
    load-fix-it path here, which is why the file has to be opened before anything
    is constructed. `neuron_config_from_checkpoint` supplies lif/2.0/32.0 for the
    400 committed files that predate the keys, so A7 and A9 on `checkpoints_v2/`
    are byte-for-byte what they were. The extra `torch.load` costs ~5 ms against
    the ~1.8 s this function then spends stepping the fixture.

    BOTH DC QUANTITIES ARE REPORTED, per §23.10(b):
      `dc_gain_mean`          -- mean of `|1/(1 - beta*e^{i w})|`, the MAGNITUDE.
                                 This is what §23.3's table quotes and what G0a
                                 gates on, and it is a flat 1/(1-beta) = 10.0 on
                                 the LIF arm.
      `dc_offset_factor_mean` -- mean of the REAL PART, which is what actually
                                 shifts a threshold applied to `u` alone, and
                                 hence what `offset_std` below is computed with.
    They coincide exactly on the LIF arm and the real part is the smaller of the
    two wherever the pole is genuinely complex; reporting only the magnitude in
    the offset column would overstate the rf arm's standing offset by roughly the
    factor H10 predicts it to fall by.
    """
    neuron = neuron_config_from_checkpoint(load_ckpt(ckpt_path))
    model, drift = reservoir_at(seed, "legacy", 1.0, ckpt_path, **neuron)
    silent, spike_rate, saturated = silent_fraction(model, obs)
    weight = model.embedding.weight.detach()
    bias = model.embedding.bias.detach()
    dc = weight @ mu + bias
    beta = float(model.reservoir.lif.beta)
    # `SpikingReservoir.omega` RAISES on a LIF reservoir rather than returning
    # zeros, so `None` -- not `getattr(..., None)` on a tensor -- is what the LIF
    # arm hands the offset formula, and the LIF branch there is the historical
    # division, unchanged to the ULP.
    omega = model.reservoir.omega if neuron["neuron_model"] == "rf" else None
    offset = induced_membrane_offset(model.reservoir.W_in @ dc, beta, omega)
    # Both DC summaries are reduced from a full (reservoir_size,) tensor in the
    # model's own dtype, mirroring what `SpikingReservoir.dc_gain` does and for
    # the same stated reason: one code path, not two. It also makes "the two
    # quantities coincide on the LIF arm" exactly true rather than true to five
    # decimals -- beta is stored as float32, so 1/(1-beta) is 9.999998 and not the
    # 10.0 §23.3's prose quotes, and two paths rounding that differently would
    # look like a real discrepancy.
    dc_gain = model.reservoir.dc_gain()
    factor = (dc_offset_factor(beta, omega) if omega is not None
              else torch.full_like(dc_gain, dc_offset_factor(beta, None)))
    return {
        "silent": silent, "spike_rate": spike_rate, "saturated": saturated,
        "w_norm": weight.norm().item(), "dc_norm": dc.norm().item(),
        "offset_std": offset.std().item(), "frozen_drift": drift,
        "neuron_model": neuron["neuron_model"],
        "dc_gain_mean": float(dc_gain.mean()),
        "dc_offset_factor_mean": float(factor.mean()),
    }


def a9_run_stats(checkpoint_dir: str, arm: str, seed: int, obs: torch.Tensor,
                 mu: torch.Tensor, max_ckpts):
    """One seed's A9 trajectory, or a skip record if the run isn't ready yet."""
    status, d, ckpts = find_run(checkpoint_dir, arm, seed)
    if status != "ok":
        return {"seed": seed, "status": status, "dir": d}
    selected = select_subset(ckpts, max_ckpts)
    trajectory = []
    for step, path in selected:
        stats = checkpoint_operating_point(seed, path, obs, mu)
        stats["step"] = step
        trajectory.append(stats)
    final = trajectory[-1]
    return {
        "seed": seed, "status": "ok", "dir": d, "trajectory": trajectory, "final": final,
        "final_is_expected_final_step": final["step"] == EXPECTED_FINAL_STEP,
        "n_checkpoints_on_disk": len(ckpts), "n_checkpoints_used": len(selected),
    }


# --------------------------------------------------------------------------- #
# report sections
# --------------------------------------------------------------------------- #

def _print_skip(seed, status, d):
    if status == "missing_dir":
        print(f"  seed{seed}: run directory does not exist yet -- skipped ({d})")
    else:
        print(f"  seed{seed}: directory exists but no checkpoints written yet -- "
              f"skipped, training in progress ({d})")


def section_a7(checkpoint_dir: str, arm: str, seeds: list, max_ckpts):
    print("\n" + "=" * 78)
    print("A7 -- dead-gradient in_proj columns (docs/EXPERIMENT_LOG.md §14.5)")
    print(f"    prediction: mean dead fraction over seeds < {A7_CONFIRMED_BELOW:.0%} of "
          f"{RESERVOIR_SIZE} (~{int(A7_CONFIRMED_BELOW * RESERVOIR_SIZE)} columns)")
    print(f"    falsified at or above {A7_FALSIFIED_AT_OR_ABOVE:.0%} "
          f"(~{int(A7_FALSIFIED_AT_OR_ABOVE * RESERVOIR_SIZE)} columns)")
    print("=" * 78)
    if arm != "reservoir":
        print(f"\n  arm={arm!r} has no in_proj readout column to measure -- A7 is "
              "reservoir-arm-only. Skipped entirely.")
        return None

    names = trainable_names("reservoir", 0)  # architecture is seed-independent
    results = []
    print(f"\n  {'run':<12}{'step':>10}{'dead cols':>11}{'% of 8192':>11}"
          f"{'params':>9}{'% of budget':>13}")
    print("  " + "-" * 68)
    for seed in seeds:
        r = a7_run_stats(checkpoint_dir, arm, seed, names, max_ckpts)
        results.append(r)
        if r["status"] != "ok":
            _print_skip(seed, r["status"], r["dir"])
            continue
        for i, row in enumerate(r["trajectory"]):
            label = f"seed{seed}" if i == 0 else ""
            params = row["count"] * D_MODEL
            print(f"  {label:<12}{row['step']:>10}{row['count']:>11}"
                  f"{row['fraction']:>10.4%}{params:>9}{params / TRAINABLE_BUDGET:>12.4%}")
        note = "" if r["final_is_expected_final_step"] else (
            f"  (PARTIAL RUN -- final on disk is step {r['final']['step']}, "
            f"expected final is {EXPECTED_FINAL_STEP})")
        print(f"    newly_dead per transition: {r['newly_dead']}  "
              f"nesting_holds={r['nesting_holds']}{note}")
        print()

    ok = [r for r in results if r["status"] == "ok"]
    print(f"  {len(ok)}/{len(seeds)} requested seeds contributed data.")
    if not ok:
        print("\n  A7 verdict: NO DATA -- no run in the requested set has a checkpoint yet.")
        return None

    finals = [r["final"]["fraction"] for r in ok]
    nesting_all_hold = all(r["nesting_holds"] for r in ok)
    partial = [r["seed"] for r in ok if not r["final_is_expected_final_step"]]
    print(f"\n  final dead fraction per seed ({len(ok)} seeds): "
          + ", ".join(f"seed{r['seed']}={r['final']['fraction']:.4%}" for r in ok))
    mean_final = sum(finals) / len(finals)
    if len(finals) > 1:
        print(f"  mean final dead fraction: {mean_final:.4%} (sd {statistics.pstdev(finals):.4%})")
    else:
        print(f"  mean final dead fraction: {mean_final:.4%} (n=1, no sd)")
    print(f"  nesting property (newly_dead=0 at every transition) holds for ALL "
          f"{len(ok)} seeds: {nesting_all_hold}")
    if partial:
        print(f"  NOTE: seeds {partial} have not yet reached the expected final step "
              f"{EXPECTED_FINAL_STEP} -- their 'final' above is their latest checkpoint "
              "so far, not necessarily the run's eventual final value.")

    v = band_verdict(mean_final, A7_CONFIRMED_BELOW, A7_FALSIFIED_AT_OR_ABOVE)
    if v == "AMBIGUOUS":
        print(f"\n  A7 verdict: AMBIGUOUS -- mean {mean_final:.4%} {AMBIGUOUS_PHRASE} "
              f"(band is [{A7_CONFIRMED_BELOW:.0%}, {A7_FALSIFIED_AT_OR_ABOVE:.0%}))")
    else:
        print(f"\n  A7 verdict: {v} -- mean final dead fraction {mean_final:.4%} against "
              f"confirmed<{A7_CONFIRMED_BELOW:.0%} / falsified>={A7_FALSIFIED_AT_OR_ABOVE:.0%}")
    return v, mean_final, len(ok)


def section_a9(checkpoint_dir: str, arm: str, seeds: list, max_ckpts):
    print("\n" + "=" * 78)
    print("A9 -- the reservoir's operating point over a full run "
          "(docs/EXPERIMENT_LOG.md §15.6)")
    print(f"    prediction: mean final silent fraction over seeds < {A9_CONFIRMED_BELOW:.0%}")
    print(f"    falsified at or above {A9_FALSIFIED_AT_OR_ABOVE:.0%} (legacy's measured ~46%)")
    print("=" * 78)
    if arm != "reservoir":
        print(f"\n  arm={arm!r} has no frozen reservoir to measure -- A9 is "
              "reservoir-arm-only. Skipped entirely.")
        return None

    obs = torch.as_tensor(np.load(REAL_OBS_PATH), dtype=torch.float32)
    mu = torch.tensor(OBS_MEAN, dtype=torch.float32)
    print(f"\n  fixture: {os.path.relpath(REAL_OBS_PATH, REPO_ROOT)}  shape={tuple(obs.shape)}")

    results = []
    all_drifts = []
    print("\n" + A9_TRAJECTORY_HEADER)
    print("  " + "-" * 100)
    for seed in seeds:
        r = a9_run_stats(checkpoint_dir, arm, seed, obs, mu, max_ckpts)
        results.append(r)
        if r["status"] != "ok":
            _print_skip(seed, r["status"], r["dir"])
            continue
        for i, row in enumerate(r["trajectory"]):
            label = f"seed{seed}" if i == 0 else ""
            all_drifts.append(row["frozen_drift"])
            print(a9_trajectory_row(label, row))
        # §23.10(b) requires BOTH DC quantities be reported. On the LIF arm they
        # are the same constant for every unit of every checkpoint, so printing
        # them would add a redundant line to a table whose committed form
        # (`results_v2_health.txt`) this module must still reproduce byte for
        # byte -- the annotation is therefore emitted only where the two differ,
        # i.e. on a run with a genuinely rotated pole.
        rotated = [row for row in r["trajectory"] if row["neuron_model"] != "lif"]
        if rotated:
            head = rotated[0]
            print(f"  {'':<10}neuron_model={head['neuron_model']}  "
                  f"mean DC gain |1/(1-beta*e^iw)| = {head['dc_gain_mean']:.4f} "
                  f"(LIF: 10.0)  mean real-part factor = "
                  f"{head['dc_offset_factor_mean']:.4f}  "
                  f"-- the offset column above uses the REAL PART (§23.10(b))")
        note = "" if r["final_is_expected_final_step"] else (
            f"  (PARTIAL RUN -- final on disk is step {r['final']['step']}, "
            f"expected final is {EXPECTED_FINAL_STEP})")
        print(f"  {'':<10}{note}")
        print()

    ok = [r for r in results if r["status"] == "ok"]
    print(f"  {len(ok)}/{len(seeds)} requested seeds contributed data.")
    if all_drifts:
        print(f"  frozen-reservoir invariant: max abs diff across {len(all_drifts)} "
              f"checkpoint loads = {max(all_drifts):.1e}")
    if not ok:
        print("\n  A9 verdict: NO DATA -- no run in the requested set has a checkpoint yet.")
        return None

    finals = [r["final"]["silent"] for r in ok]
    partial = [r["seed"] for r in ok if not r["final_is_expected_final_step"]]
    print(f"\n  final silent fraction per seed ({len(ok)} seeds): "
          + ", ".join(f"seed{r['seed']}={r['final']['silent']:.4%}" for r in ok))
    mean_final = sum(finals) / len(finals)
    if len(finals) > 1:
        print(f"  mean final silent fraction: {mean_final:.4%} (sd {statistics.pstdev(finals):.4%})")
    else:
        print(f"  mean final silent fraction: {mean_final:.4%} (n=1, no sd)")
    if partial:
        print(f"  NOTE: seeds {partial} have not yet reached the expected final step "
              f"{EXPECTED_FINAL_STEP} -- their 'final' above is their latest checkpoint "
              "so far, not necessarily the run's eventual final value.")

    v = band_verdict(mean_final, A9_CONFIRMED_BELOW, A9_FALSIFIED_AT_OR_ABOVE)
    if v == "AMBIGUOUS":
        print(f"\n  A9 verdict: AMBIGUOUS -- mean {mean_final:.4%} {AMBIGUOUS_PHRASE} "
              f"(band is [{A9_CONFIRMED_BELOW:.0%}, {A9_FALSIFIED_AT_OR_ABOVE:.0%}))")
    else:
        print(f"\n  A9 verdict: {v} -- mean final silent fraction {mean_final:.4%} against "
              f"confirmed<{A9_CONFIRMED_BELOW:.0%} / falsified>={A9_FALSIFIED_AT_OR_ABOVE:.0%}")
    return v, mean_final, len(ok)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="A7 (dead in_proj columns) and A9 (reservoir operating point) "
                     "measurements against the v2 corrected-input matrix. Read-only, "
                     "writes no files, safe to run against a partially-complete matrix.")
    parser.add_argument("--checkpoint-dir", default="checkpoints_v2",
                        help="directory holding {arm}_seed{n}/ run subdirectories "
                             "(default: checkpoints_v2)")
    parser.add_argument("--arm", default="reservoir", choices=["baseline", "reservoir"])
    parser.add_argument("--seeds", default="0-9",
                        help="e.g. '0-9' (default), '0-0', '0,2,5-7'")
    parser.add_argument("--only", choices=["a7", "a9"], default=None,
                        help="run only one measurement (default: both)")
    parser.add_argument("--max-checkpoints-per-run", type=int, default=None,
                        help="cap checkpoints loaded per run (always keeps the "
                             "first and last); omit for the full trajectory")
    args = parser.parse_args(argv)

    checkpoint_dir = os.path.join(REPO_ROOT, args.checkpoint_dir) \
        if not os.path.isabs(args.checkpoint_dir) else args.checkpoint_dir
    seeds = parse_seed_spec(args.seeds)

    print("=" * 78)
    print("GameSpike reservoir health -- A7 (dead columns) + A9 (operating point)")
    print(f"  repo            : {REPO_ROOT}")
    print(f"  torch           : {torch.__version__}")
    print(f"  checkpoint dir  : {checkpoint_dir}")
    print(f"  arm             : {args.arm}")
    print(f"  seeds requested : {seeds}")
    print(f"  max ckpts/run   : {args.max_checkpoints_per_run or 'unlimited'}")
    print("  READ-ONLY: this module writes nothing and trains nothing. A run "
          "directory or checkpoint that does not exist yet is skipped and "
          "reported, never raised.")
    print("=" * 78)

    a7 = a9 = None
    if args.only in (None, "a7"):
        a7 = section_a7(checkpoint_dir, args.arm, seeds, args.max_checkpoints_per_run)
    if args.only in (None, "a9"):
        a9 = section_a9(checkpoint_dir, args.arm, seeds, args.max_checkpoints_per_run)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    if a7 is not None:
        v, mean_final, n = a7
        print(f"  A7  {v:<10} mean final dead fraction {mean_final:.4%} (n={n} seeds) "
              f"(threshold: <{A7_CONFIRMED_BELOW:.0%} confirms, "
              f">={A7_FALSIFIED_AT_OR_ABOVE:.0%} falsifies)")
    elif args.only in (None, "a7"):
        print("  A7  NO DATA")
    if a9 is not None:
        v, mean_final, n = a9
        print(f"  A9  {v:<10} mean final silent fraction {mean_final:.4%} (n={n} seeds) "
              f"(threshold: <{A9_CONFIRMED_BELOW:.0%} confirms, "
              f">={A9_FALSIFIED_AT_OR_ABOVE:.0%} falsifies)")
    elif args.only in (None, "a9"):
        print("  A9  NO DATA")


if __name__ == "__main__":
    main()
