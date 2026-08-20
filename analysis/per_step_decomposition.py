"""Reproduces `docs/RESULTS.md` §5's "per-step decomposition" table: the
measurement that shows the episode-return scoreboard in §1/§3 is the metric
MOST FLATTERING to the reservoir arm, not a neutral one.

WHAT THE TABLE IS, IN ONE SENTENCE. §1/§3 rank the two arms on episode
RETURN, an integral over however long an episode happens to last. §5 found
that the reservoir arm's episodes last roughly six times longer than the
baseline's (it "learned to survive without progressing" rather than "learned
to progress and die quickly" -- see §5's prose), so its longer return
integrates dense progress reward over ~6x more steps and closes part of a
per-step gap it never actually closed. Normalising by episode length -- reward
PER STEP -- is the fairer per-unit-of-experience comparison, and it makes the
baseline's advantage roughly 6x rather than the ~1.27x visible in raw return.
This module computes exactly that normalisation, and the two-sided exact
permutation tests §5 reports alongside it, from the same
`eval_{arm}_seed{N}_{regime}.json` files `analysis/aggregate_results.py`
already knows how to load.

THE ORDERING THAT MUST NOT BE CHANGED: PER-SEED, THEN AVERAGE.
Reward per step for one arm is computed as

    mean( mean_extrinsic_return[seed] / mean_episode_length[seed]  for seed in seeds )

and NEVER as

    mean(mean_extrinsic_return[seed] for seed in seeds) / mean(mean_episode_length[seed] for seed in seeds)

i.e. mean-of-ratios, not ratio-of-means. These are NOT interchangeable, and
the difference is not a rounding footnote -- division is a nonlinear
operation, so by Jensen's inequality the two orderings generally disagree, and
they disagree here by enough to matter. A two-line illustration: seed A
returns 10 over 100 steps (0.1/step) and seed B returns 10 over 1 step
(10.0/step); mean-of-ratios is (0.1 + 10.0)/2 = 5.05/step, while ratio-of-means
is 20/101 = 0.198/step -- a 25x difference in the same toy example, driven
entirely by which seed's episode-length denominator happens to be small. Both
readings are legitimate ways to define "reward per step" at the level of the
POOLED data, but they answer different questions: mean-of-ratios treats each
of the 10 TRAINING SEEDS as one independent observation of "this arm's reward
rate" and averages across observations (matching this project's unit of
analysis, see below); ratio-of-means treats the 10 seeds' total reward and
total steps as if they came from one pooled run, which lets whichever seed
happened to rack up the most steps dominate the estimate. §5's published
numbers use mean-of-ratios (per-seed-then-average), and this module
reproduces that choice deliberately, not by accident -- see
`decompose`'s docstring for where the division actually happens.

THE UNIT OF ANALYSIS IS THE TRAINING SEED, NEVER THE EPISODE. This is not
restated at length here because `analysis/aggregate_results.py`'s module
docstring (point 2) and `docs/RESULTS.md` §2.4 already say it in full: each of
the 30 evaluation episodes behind one `eval_*.json` file is POLICY-SAMPLING
noise from one already-frozen checkpoint, not independent evidence about the
architecture. `mean_extrinsic_return` and `mean_episode_length` are each
already a mean over one checkpoint's 30 episodes by the time this module ever
sees them (that reduction happens once, in `training/evaluate.py`, when the
JSON is written); this module only ever averages those 10 per-seed numbers
further, exactly as `aggregate_by_arm` does for every other metric in this
project. It never touches `extrinsic_returns` or `episode_lengths` (the
raw 30-episode lists) directly, and doing so would be the exact mistake this
paragraph, and both of the documents above, warn against.

REUSE, NOT REIMPLEMENTATION. `aggregate_results.load_eval_results` (parsing +
loud validation of the `eval_{arm}_seed{N}_{regime}.json` naming/schema
convention) and `aggregate_results.aggregate_by_arm` (per-arm, per-metric
reduction to one number per seed, in ascending-seed order) do exactly the
per-seed reduction this module needs for `mean_extrinsic_return` and
`mean_episode_length` already; `_records_with_reward_per_step` below only adds
ONE derived per-record field (`reward_per_step`, computed per record --
i.e. per seed -- BEFORE `aggregate_by_arm` ever averages anything) so that the
existing reducer can be reused unmodified for the third metric too, rather
than hand-rolling a second aggregation path that could quietly drift out of
sync with the first. `aggregate_results.exact_permutation_test` is imported
and called directly (see its own docstring: full C(20,10)=184,756-way exact
enumeration at n=10-vs-10, no reimplementation here) using its established
CONVENTION -- `a` is the reservoir arm's per-seed values, `b` is the
baseline's, so a positive `observed_diff` means reservoir > baseline and a
negative one means baseline > reservoir. §5 reports the p-value on both the
reward-per-step difference and the episode-length difference under this same
convention, and this module reports the exact same two numbers the exact same
way, not a re-derivation.

GRACEFUL PARTIAL OPERATION. This module is meant to be pointed at whichever
`results_dir` an experiment happens to be reporting at the time -- `final`,
`init`, `best`, or later a `results_v2/*` sibling with the identical file
naming convention (`results_v2/final`, etc. -- nothing here is hardcoded to
the `results` directory name, only to the `eval_{arm}_seed{N}_{regime}.json`
convention `load_eval_results` already enforces). Any of the following is a
routine, expected state rather than a bug, and none of them raises:
  * `results_dir` does not exist yet (e.g. an evaluation pass has not run) --
    `decompose` returns a dict with `"skipped": True` and a clear `"error"`
    message instead of computing anything.
  * one arm has zero eval files for the requested regime (e.g. only the
    baseline half of a matrix has been evaluated so far) -- the OTHER arm's
    per-arm numbers are still computed and returned; only the cross-arm
    ratio/permutation numbers, which need both arms, are skipped, with an
    `"error"` explaining why.
  * an arm has SOME but not all 10 training seeds' files present (a matrix
    still filling in) -- the arm's numbers are computed over however many
    seeds are actually on disk, and `missing_seeds` lists exactly which
    training seeds (0-9) are absent, so a mean of 4 seeds is never mistaken
    for a mean of 10.
A malformed or mislabelled eval file (wrong `arm`/`regime` inside a
suspiciously-named JSON) is a DIFFERENT kind of problem -- silent data
corruption, not routine incompleteness -- and `load_eval_results` continues to
raise loudly on that, exactly as it does for every other caller in this
project. "Missing" is tolerated; "wrong" is not.
"""
import argparse
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from analysis.aggregate_results import (  # noqa: E402
    aggregate_by_arm,
    exact_permutation_test,
    load_eval_results,
)

ARMS = ("baseline", "reservoir")
REGIMES = ("continuous", "reset128")
ALL_TRAIN_SEEDS = tuple(range(10))


def _records_with_reward_per_step(records) -> list:
    """Returns a NEW list of records (the input is left untouched) with one
    derived field added to each: `reward_per_step = mean_extrinsic_return /
    mean_episode_length`, computed PER RECORD -- i.e. per training seed --
    which is what makes it safe to hand the result to `aggregate_by_arm`
    unmodified and get the per-seed-then-average reduction this module's
    docstring insists on, rather than a ratio of already-averaged sums.

    Raises `ValueError` (loudly, not a silent zero/inf) if any record's
    `mean_episode_length` is exactly 0.0 -- an episode of zero length across
    all 30 evaluation episodes of a checkpoint would itself be a serious
    anomaly in the evaluation harness, and dividing by it silently would
    either crash confusingly downstream or, worse, produce `inf`/`nan` that
    then poisons a mean over seeds without any indication of where it came
    from.
    """
    out = []
    for r in records:
        length = r["mean_episode_length"]
        if length == 0.0:
            raise ValueError(
                f"_records_with_reward_per_step: record for arm={r.get('arm')!r} "
                f"train_seed={r.get('train_seed')!r} regime={r.get('regime')!r} has "
                f"mean_episode_length == 0.0 (source file: {r.get('source_file')!r}) "
                f"-- cannot compute a reward-per-step rate from a zero-length episode "
                f"mean; this would need investigating, not dividing through."
            )
        r2 = dict(r)
        r2["reward_per_step"] = r["mean_extrinsic_return"] / length
        out.append(r2)
    return out


def decompose(results_dir, regime: str, seeds=ALL_TRAIN_SEEDS) -> dict:
    """Computes `docs/RESULTS.md` §5's per-step decomposition for one
    `results_dir` (e.g. `results/final`, `results/init`, or later
    `results_v2/final`) and one `regime` (`"continuous"` or `"reset128"`).

    For each arm found (`baseline`, `reservoir`), returns:
      * `mean_episode_return`  -- mean over seeds of `mean_extrinsic_return`
      * `mean_episode_length`  -- mean over seeds of `mean_episode_length`
      * `mean_reward_per_step` -- mean over seeds of the PER-SEED ratio
        `mean_extrinsic_return / mean_episode_length` (see module docstring
        for why this is not interchangeable with a ratio of the two means
        above)
      * `n`, `train_seeds`, `missing_seeds` -- exactly which of `seeds`
        (default 0-9) actually contributed a file, so a partial arm is never
        silently reported as if it were complete.

    When BOTH arms have at least one contributing seed, also returns:
      * `ratio_baseline_over_reservoir_reward_per_step` -- `docs/RESULTS.md`
        §5's headline ratio (baseline / reservoir; > 1 means the baseline
        earns more reward per step).
      * `permutation_reward_per_step`, `permutation_episode_length` -- exact
        two-sided permutation tests (via
        `aggregate_results.exact_permutation_test`, reservoir as `a`,
        baseline as `b` -- see module docstring) on the per-seed
        reward-per-step values and the per-seed mean-episode-length values
        respectively. Both are what §5 reports as "the floor"
        (p=2/184756=0.000011) on the published `results/final` data.

    Returns `{"skipped": True, "error": "..."}`  (plus `results_dir`,
    `regime`, and `"arms": {"baseline": None, "reservoir": None}`) if
    `results_dir` does not exist at all -- see module docstring's "GRACEFUL
    PARTIAL OPERATION" section. Raises `ValueError` if `regime` is not one of
    `"continuous"`/`"reset128"` (a caller/programming error, not a data-
    availability one -- refusing to guess is better than silently no-op'ing on
    a typo'd regime name).
    """
    if regime not in REGIMES:
        raise ValueError(
            f"decompose: regime must be one of {REGIMES}, got {regime!r}"
        )
    results_dir = str(results_dir)

    if not os.path.isdir(results_dir):
        return {
            "results_dir": results_dir,
            "regime": regime,
            "arms": {arm: None for arm in ARMS},
            "skipped": True,
            "error": (
                f"decompose: results_dir {results_dir!r} does not exist -- skipping "
                f"rather than raising (e.g. this checkpoint selection has not been "
                f"evaluated yet). No numbers computed."
            ),
        }

    records = load_eval_results(results_dir)  # loud on malformed/mislabelled files
    records = _records_with_reward_per_step(records)

    return_agg = aggregate_by_arm(records, regime, metric="mean_extrinsic_return")
    length_agg = aggregate_by_arm(records, regime, metric="mean_episode_length")
    rps_agg = aggregate_by_arm(records, regime, metric="reward_per_step")
    arms_present = set(rps_agg)  # aggregate_by_arm only keys arms with >=1 record

    out = {"results_dir": results_dir, "regime": regime, "arms": {}}
    for arm in ARMS:
        if arm not in arms_present:
            out["arms"][arm] = None
            continue
        missing_seeds = sorted(set(seeds) - set(rps_agg[arm]["train_seeds"]))
        out["arms"][arm] = {
            "arm": arm,
            "n": rps_agg[arm]["n"],
            "train_seeds": rps_agg[arm]["train_seeds"],
            "missing_seeds": missing_seeds,
            "mean_episode_return": return_agg[arm]["mean"],
            "mean_episode_length": length_agg[arm]["mean"],
            "mean_reward_per_step": rps_agg[arm]["mean"],
            "reward_per_step_values": rps_agg[arm]["values"],
            "episode_length_values": length_agg[arm]["values"],
        }

    if ARMS[0] in arms_present and ARMS[1] in arms_present:
        baseline_rps = out["arms"]["baseline"]["mean_reward_per_step"]
        reservoir_rps = out["arms"]["reservoir"]["mean_reward_per_step"]
        out["ratio_baseline_over_reservoir_reward_per_step"] = baseline_rps / reservoir_rps
        out["permutation_reward_per_step"] = exact_permutation_test(
            rps_agg["reservoir"]["values"], rps_agg["baseline"]["values"]
        )._asdict()
        out["permutation_episode_length"] = exact_permutation_test(
            length_agg["reservoir"]["values"], length_agg["baseline"]["values"]
        )._asdict()
    else:
        missing_arms = sorted(set(ARMS) - arms_present)
        out["error"] = (
            f"decompose: no eval results for arm(s) {missing_arms} under "
            f"regime={regime!r} in {results_dir!r} -- per-arm numbers above (where "
            f"present) are still valid, but the cross-arm ratio and permutation "
            f"tests need both arms and were skipped."
        )
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _fmt(x, prec: int = 5) -> str:
    """Same non-fatal float formatting convention as
    `aggregate_results._fmt` (NaN/inf print as words, anything else via
    plain `str()`), duplicated rather than imported because it is three
    lines and importing a private underscore-prefixed helper across modules
    would be a worse coupling than repeating it."""
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return f"{x:.{prec}f}"
    return str(x)


def format_table(result: dict) -> str:
    """Renders `decompose`'s return value as the human-readable table
    `docs/RESULTS.md` §5 publishes (one row per arm: episode return, mean
    episode length, reward per step), followed by the ratio and both exact
    permutation tests when both arms are present. Prints numbers only --
    no "arm X wins" line, matching `aggregate_results._format_comparison`'s
    documented reason (module docstring point 3 there): a small-n comparison
    like this one is for the reader to judge, not for the tool to announce.
    """
    lines = [f"--- results_dir={result['results_dir']}  regime={result['regime']} ---"]

    if result.get("skipped"):
        lines.append(f"  SKIPPED: {result['error']}")
        return "\n".join(lines)

    header = f"  {'arm':<12}{'n':>4}{'episode return':>18}{'mean ep. length':>18}{'reward per step':>18}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for arm in ARMS:
        a = result["arms"][arm]
        if a is None:
            lines.append(f"  {arm:<12}{'(no eval results for this regime)':>58}")
            continue
        row = (f"  {arm:<12}{a['n']:>4}{_fmt(a['mean_episode_return']):>18}"
               f"{_fmt(a['mean_episode_length']):>18}{_fmt(a['mean_reward_per_step'], 6):>18}")
        lines.append(row)
        if a["missing_seeds"]:
            lines.append(f"      NOTE: missing training seeds {a['missing_seeds']} "
                         f"(n={a['n']} of {a['n'] + len(a['missing_seeds'])} expected)")

    if "error" in result:
        lines.append(f"  {result['error']}")
        return "\n".join(lines)

    lines.append("")
    lines.append(f"  baseline/reservoir reward-per-step ratio: "
                 f"{_fmt(result['ratio_baseline_over_reservoir_reward_per_step'], 4)}x")

    p_rps = result["permutation_reward_per_step"]
    lines.append(f"  exact permutation test, reward per step (reservoir - baseline): "
                 f"diff={_fmt(p_rps['observed_diff'], 6)}  p={_fmt(p_rps['p_value'], 6)}  "
                 f"(method={p_rps['method']}, n_permutations={p_rps['n_permutations']})")

    p_len = result["permutation_episode_length"]
    lines.append(f"  exact permutation test, mean episode length (reservoir - baseline): "
                 f"diff={_fmt(p_len['observed_diff'], 6)}  p={_fmt(p_len['p_value'], 6)}  "
                 f"(method={p_len['method']}, n_permutations={p_len['n_permutations']})")

    lines.append("")
    lines.append("  NOTE: reward per step is computed PER TRAINING SEED as "
                 "mean_extrinsic_return / mean_episode_length,")
    lines.append("        THEN averaged over seeds -- not a ratio of the two mean "
                 "columns to its left. See this module's")
    lines.append("        docstring for why the two orderings are not interchangeable.")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Reproduce docs/RESULTS.md §5's per-step decomposition table "
                    "(episode return, mean episode length, reward per step, and the "
                    "baseline/reservoir gap) for one results directory and regime."
    )
    parser.add_argument("--results-dir", required=True,
                        help="directory of eval_{arm}_seed{N}_{regime}.json files, "
                             "e.g. results/final, results/init, results_v2/final")
    parser.add_argument("--regime", required=True, choices=REGIMES,
                        help="'continuous' or 'reset128'")
    args = parser.parse_args(argv)

    result = decompose(args.results_dir, args.regime)
    print(format_table(result))


if __name__ == "__main__":
    main()
