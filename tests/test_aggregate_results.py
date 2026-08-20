"""Tests for `analysis/aggregate_results.py`.

This module's output IS the scientific verdict for design doc §5 (frozen
reservoir vs. matched-parameter GRU baseline), so the statistical primitives in
particular are tested against hand-checkable reference values, not merely
"runs without crashing". Section map mirrors the module: primitives first
(each independently pinned against a known-correct number), then loading /
aggregation (including the loud-failure paths that exist specifically to catch
silent mislabelling), then the training-log summariser's truncated-line
handling.
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from analysis.aggregate_results import (
    _json_safe,
    aggregate_by_arm,
    bootstrap_ci_diff,
    build_eval_manifest,
    build_report,
    cohens_d,
    compare_arms,
    exact_permutation_test,
    list_checkpoints,
    load_eval_results,
    mann_whitney_u,
    select_best_checkpoint,
    select_final_checkpoint,
    summarise_training_logs,
    welch_ttest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# The reference pair used throughout: two disjoint, evenly-spaced groups of 5,
# chosen because EVERY statistic below has a hand-checkable closed form for it
# (see each test for the derivation) -- exactly the property needed to catch a
# sign error or an off-by-one in the exact-enumeration machinery, which a
# random/fuzzed input could not.
A = [1, 2, 3, 4, 5]
B = [6, 7, 8, 9, 10]


# ---------------------------------------------------------------------------
# welch_ttest
# ---------------------------------------------------------------------------

def test_welch_ttest_matches_hand_computed_case():
    """a and b both have sample variance 2.5 (n=5): deviations from the mean
    are +-2,+-1,0, squared-summed=10, /(5-1)=2.5. Equal variances and equal n
    means Welch's df collapses to n_a+n_b-2=8 exactly, and
    t = (3-8) / sqrt(2.5/5 + 2.5/5) = -5 / sqrt(1.0) = -5.0 exactly."""
    result = welch_ttest(A, B)
    assert result.t_statistic == pytest.approx(-5.0, abs=1e-9)
    assert result.df == pytest.approx(8.0, abs=1e-9)
    # p-value for t=+-5, df=8, two-sided. Cross-checked independently two ways
    # (mpmath's regularized incomplete beta at 50-digit precision, AND direct
    # numerical integration of the Student-t PDF's tail -- a totally separate
    # code path from the incomplete-beta identity this module itself uses):
    # both give 0.00105282579... . The task brief's own recalled value of
    # ~0.00106 was checked and found to be off in the fourth significant digit;
    # this is the verified number.
    assert result.p_value == pytest.approx(0.0010528257933665393, abs=1e-9)


def test_welch_ttest_identical_samples_give_p_one_and_t_zero():
    result = welch_ttest(A, A)
    assert result.t_statistic == pytest.approx(0.0, abs=1e-9)
    assert result.p_value == pytest.approx(1.0, abs=1e-9)


def test_welch_ttest_below_two_samples_returns_nan_not_an_exception():
    """A sample variance needs >=2 points; NaN says 'not measurable', matching
    training/evaluate.py's own single-episode convention, rather than raising
    on an input that is weak but not actually invalid."""
    result = welch_ttest([1.0], B)
    assert math.isnan(result.t_statistic)
    assert math.isnan(result.df)
    assert math.isnan(result.p_value)


def test_welch_ttest_rejects_empty_group():
    with pytest.raises(ValueError):
        welch_ttest([], B)


# ---------------------------------------------------------------------------
# exact_permutation_test
# ---------------------------------------------------------------------------

def test_exact_permutation_test_matches_known_p_value():
    """With a and b fully separated (every a < every b), only the observed
    split itself and its mirror image (b and a swapped) are AS extreme as the
    observed |diff of means| -- 2 out of C(10,5)=252 splits."""
    result = exact_permutation_test(A, B)
    assert result.method == "exact"
    assert result.n_permutations == 252
    assert result.observed_diff == pytest.approx(3.0 - 8.0, abs=1e-9)
    assert result.p_value == pytest.approx(2 / 252, abs=1e-9)


def test_exact_permutation_test_identical_samples_give_p_one():
    """Every split is at least as extreme as the (zero) observed difference."""
    result = exact_permutation_test(A, A)
    assert result.observed_diff == pytest.approx(0.0, abs=1e-9)
    assert result.p_value == pytest.approx(1.0, abs=1e-9)
    assert result.method == "exact"


def test_exact_permutation_test_falls_back_to_seeded_monte_carlo_above_threshold():
    """Forcing a tiny exact_threshold (rather than actually using >20 seeds,
    which would make this test itself slow) exercises the Monte Carlo branch
    directly and pins its two defining properties: it reports which method ran,
    and it is reproducible from its seed."""
    result = exact_permutation_test(A, B, exact_threshold=1, n_resamples=20_000, seed=42)
    assert result.method == "monte_carlo"
    assert result.n_permutations == 20_000
    # The true probability a RANDOM split is at least as extreme as the fully-
    # separated observed one is 2/252 (from the exact test above) -- the Monte
    # Carlo estimate should land close to it. Standard error of a proportion at
    # p~0.0079, n=20000 is ~0.00063, so 8 SE is a very safe margin against
    # flakiness while still catching a badly broken implementation (e.g. one
    # that always returns p~1.0 or p~0.0).
    assert result.p_value == pytest.approx(2 / 252, abs=0.005)

    repeat = exact_permutation_test(A, B, exact_threshold=1, n_resamples=20_000, seed=42)
    assert repeat.p_value == result.p_value


def test_exact_permutation_test_rejects_empty_group():
    with pytest.raises(ValueError):
        exact_permutation_test([], B)


# ---------------------------------------------------------------------------
# mann_whitney_u
# ---------------------------------------------------------------------------

def test_mann_whitney_u_matches_known_p_value():
    """U_a convention: U_a = 0 means every value of a is below every value of
    b, which is exactly this case (a=[1..5], b=[6..10]). The exact null
    distribution over the same 252 tie-free rank-splits gives the identical
    2/252 the permutation test found -- expected, since with distinct
    integer data the rank-based and mean-based extremeness orderings coincide
    for this particular fully-separated case."""
    result = mann_whitney_u(A, B)
    assert result.method == "exact"
    assert result.U == pytest.approx(0.0, abs=1e-9)
    assert result.p_value == pytest.approx(2 / 252, abs=1e-9)


def test_mann_whitney_u_identical_samples_give_p_near_one():
    """a and b identical means every value is tied, forcing the
    normal-approximation branch (the exact recurrence assumes tie-free data).
    U_a sits exactly at its null mean (n_a*n_b/2), so after the continuity
    correction the z-score is 0 and p is exactly 1.0."""
    result = mann_whitney_u(A, A)
    assert result.method == "normal_approximation"
    assert result.p_value == pytest.approx(1.0, abs=1e-9)


def test_mann_whitney_u_ties_use_normal_approximation_even_below_the_exact_threshold():
    a = [1, 2, 2, 4, 5]
    b = [2, 6, 7, 8, 9]  # shares the value 2 with `a` -> a tie
    result = mann_whitney_u(a, b)
    assert result.method == "normal_approximation"


def test_mann_whitney_u_rejects_empty_group():
    with pytest.raises(ValueError):
        mann_whitney_u([], B)


# ---------------------------------------------------------------------------
# cohens_d
# ---------------------------------------------------------------------------

def test_cohens_d_matches_hand_computed_case():
    """pooled_var = (4*2.5 + 4*2.5)/8 = 2.5, pooled_sd = sqrt(2.5) = 1.5811388;
    d = (3-8)/1.5811388 = -3.16227766...  Sign convention: NEGATIVE here
    because a's mean (3) is below b's mean (8) -- positive means `a` scored
    higher, matching compare_arms' reservoir-as-a/baseline-as-b convention."""
    d = cohens_d(A, B)
    assert d == pytest.approx(-3.16227766, abs=1e-6)


def test_cohens_d_identical_samples_is_zero():
    assert cohens_d(A, A) == pytest.approx(0.0, abs=1e-12)


def test_cohens_d_below_two_samples_returns_nan():
    assert math.isnan(cohens_d([1.0], B))


def test_cohens_d_zero_pooled_variance_with_differing_means_is_signed_infinity():
    d = cohens_d([5.0, 5.0, 5.0], [1.0, 1.0, 1.0])
    assert math.isinf(d) and d > 0
    d2 = cohens_d([1.0, 1.0, 1.0], [5.0, 5.0, 5.0])
    assert math.isinf(d2) and d2 < 0


# ---------------------------------------------------------------------------
# bootstrap_ci_diff
# ---------------------------------------------------------------------------

def test_bootstrap_ci_diff_is_seeded_and_reproducible():
    r1 = bootstrap_ci_diff(A, B, n_resamples=5000, seed=7)
    r2 = bootstrap_ci_diff(A, B, n_resamples=5000, seed=7)
    assert r1 == r2  # identical seed -> byte-identical resampling -> identical CI


def test_bootstrap_ci_diff_different_seed_gives_a_different_interval():
    r1 = bootstrap_ci_diff(A, B, n_resamples=2000, seed=1)
    r2 = bootstrap_ci_diff(A, B, n_resamples=2000, seed=2)
    assert (r1.ci_low, r1.ci_high) != (r2.ci_low, r2.ci_high)


def test_bootstrap_ci_diff_contains_the_observed_difference():
    """Not a proof of correctness by itself, but a real bootstrap CI must at
    minimum contain its own observed point estimate -- the percentile method
    can only fail this if something upstream (RNG, resampling shape) is broken."""
    result = bootstrap_ci_diff(A, B, n_resamples=20000, seed=0)
    assert result.observed_diff == pytest.approx(-5.0, abs=1e-9)
    assert result.ci_low <= result.observed_diff <= result.ci_high


def test_bootstrap_ci_diff_is_centred_on_zero_for_identical_samples():
    """a and b are the SAME values, so the OBSERVED difference is exactly 0 --
    but the bootstrap resamples each group independently with replacement, so
    the resampled difference distribution still has real spread (this is
    correct: two independent resamples of the same underlying data are not
    guaranteed to match each other even though they're drawn from identical
    populations). What must hold is that the interval is centred on 0 and
    contains the observed difference."""
    result = bootstrap_ci_diff(A, A, n_resamples=20000, seed=0)
    assert result.observed_diff == pytest.approx(0.0, abs=1e-9)
    assert result.ci_low < 0.0 < result.ci_high
    # Symmetric by construction (a and b are resampled from the identical
    # population), so the interval should be roughly symmetric around 0.
    assert result.ci_low == pytest.approx(-result.ci_high, abs=0.5)


# ---------------------------------------------------------------------------
# load_eval_results
# ---------------------------------------------------------------------------

def _write_eval_json(results_dir: Path, arm: str, train_seed: int, regime: str,
                      *, json_arm: str = None, state_reset_interval="__default__",
                      mean_extrinsic_return: float = 42.0):
    """Writes one fake eval JSON, matching training/evaluate.py's --json shape
    closely enough for the loader/aggregator under test (arm, seed keys,
    state_reset_interval, and whichever metric a test needs)."""
    if json_arm is None:
        json_arm = arm
    if state_reset_interval == "__default__":
        state_reset_interval = None if regime == "continuous" else 128
    payload = {
        "arm": json_arm,
        "n_episodes": 3,
        "seed": 999,  # EVALUATION seed -- deliberately different from train_seed,
                      # to prove the loader never confuses the two.
        "state_reset_interval": state_reset_interval,
        "mean_extrinsic_return": mean_extrinsic_return,
        "std_extrinsic_return": 1.0,
        "sem_extrinsic_return": 0.5,
        "extrinsic_returns": [mean_extrinsic_return] * 3,
        "mean_combined_return": mean_extrinsic_return + 1.0,
        "std_combined_return": 1.0,
        "sem_combined_return": 0.5,
        "combined_returns": [mean_extrinsic_return + 1.0] * 3,
        "mean_episode_length": 100.0,
        "std_episode_length": 1.0,
        "sem_episode_length": 0.5,
        "episode_lengths": [100.0] * 3,
    }
    filename = f"eval_{arm}_seed{train_seed}_{regime}.json"
    (results_dir / filename).write_text(json.dumps(payload))
    return filename


def test_load_eval_results_parses_arm_seed_regime_from_filename(tmp_path):
    _write_eval_json(tmp_path, "reservoir", 3, "continuous")
    _write_eval_json(tmp_path, "baseline", 7, "reset128")

    records = load_eval_results(tmp_path)
    assert len(records) == 2
    by_arm = {r["arm"]: r for r in records}

    assert by_arm["reservoir"]["train_seed"] == 3
    assert by_arm["reservoir"]["regime"] == "continuous"
    # The EVALUATION seed (JSON's own "seed" key) must survive untouched and
    # distinct from train_seed -- these are two different axes.
    assert by_arm["reservoir"]["seed"] == 999
    assert by_arm["reservoir"]["train_seed"] != by_arm["reservoir"]["seed"]

    assert by_arm["baseline"]["train_seed"] == 7
    assert by_arm["baseline"]["regime"] == "reset128"


def test_load_eval_results_raises_on_filename_json_arm_mismatch(tmp_path):
    """The file is named for the reservoir arm but its JSON payload claims to
    be a baseline result -- a real copy/rename mistake that must not pass
    silently through into a comparison."""
    _write_eval_json(tmp_path, "reservoir", 0, "continuous", json_arm="baseline")
    with pytest.raises(ValueError, match="arm"):
        load_eval_results(tmp_path)


def test_load_eval_results_raises_on_malformed_filename(tmp_path):
    (tmp_path / "eval_reservoir_continuous.json").write_text(
        json.dumps({"arm": "reservoir", "state_reset_interval": None}))
    with pytest.raises(ValueError):
        load_eval_results(tmp_path)


def test_load_eval_results_raises_on_regime_state_reset_interval_mismatch(tmp_path):
    """Filename says 'continuous' (implies state_reset_interval=None) but the
    payload was actually produced with state_reset_interval=128 -- a
    mislabelled-regime file, the same failure class as the arm check above."""
    _write_eval_json(tmp_path, "reservoir", 0, "continuous", state_reset_interval=128)
    with pytest.raises(ValueError, match="regime"):
        load_eval_results(tmp_path)


def test_load_eval_results_on_empty_dir_returns_empty_list(tmp_path):
    assert load_eval_results(tmp_path) == []


# ---------------------------------------------------------------------------
# aggregate_by_arm / compare_arms
# ---------------------------------------------------------------------------

def test_aggregate_by_arm_sorts_by_train_seed_and_computes_ddof1_spread(tmp_path):
    # Written out of seed order deliberately, to prove aggregate_by_arm sorts
    # rather than trusting file-listing order.
    _write_eval_json(tmp_path, "reservoir", 5, "continuous", mean_extrinsic_return=30.0)
    _write_eval_json(tmp_path, "reservoir", 1, "continuous", mean_extrinsic_return=10.0)
    _write_eval_json(tmp_path, "reservoir", 3, "continuous", mean_extrinsic_return=20.0)
    records = load_eval_results(tmp_path)

    agg = aggregate_by_arm(records, "continuous")
    r = agg["reservoir"]
    assert r["train_seeds"] == [1, 3, 5]
    assert r["values"] == [10.0, 20.0, 30.0]
    assert r["n"] == 3
    assert r["mean"] == pytest.approx(20.0)
    # sample stdev (ddof=1) of [10,20,30]: mean=20, sq devs 100+0+100=200, /(3-1)=100, sqrt=10
    assert r["std"] == pytest.approx(10.0)
    assert r["sem"] == pytest.approx(10.0 / math.sqrt(3))
    assert r["min"] == 10.0 and r["max"] == 30.0


def test_aggregate_by_arm_ignores_other_regimes(tmp_path):
    _write_eval_json(tmp_path, "reservoir", 0, "continuous", mean_extrinsic_return=1.0)
    _write_eval_json(tmp_path, "reservoir", 0, "reset128", mean_extrinsic_return=999.0)
    records = load_eval_results(tmp_path)
    agg = aggregate_by_arm(records, "continuous")
    assert agg["reservoir"]["values"] == [1.0]


def test_aggregate_by_arm_raises_on_duplicate_train_seed(tmp_path):
    """Two eval results claiming the same (arm, train_seed, regime) -- e.g.
    evaluate.py was re-run and both outputs were kept -- is an ambiguous
    duplicate, not additional evidence; aggregate_by_arm must refuse to guess
    which one counts. Simulated by duplicating a loaded record directly,
    since the filename pattern itself only allows one file per (arm, seed,
    regime) triple."""
    _write_eval_json(tmp_path, "reservoir", 0, "continuous", mean_extrinsic_return=1.0)
    records = load_eval_results(tmp_path)
    records.append(dict(records[0]))
    with pytest.raises(ValueError, match="more than one"):
        aggregate_by_arm(records, "continuous")


def test_compare_arms_bundles_both_arms_and_every_statistic(tmp_path):
    for seed, val in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
        _write_eval_json(tmp_path, "reservoir", seed, "continuous", mean_extrinsic_return=val)
    for seed, val in enumerate([6.0, 7.0, 8.0, 9.0, 10.0]):
        _write_eval_json(tmp_path, "baseline", seed, "continuous", mean_extrinsic_return=val)
    records = load_eval_results(tmp_path)

    result = compare_arms(records, "continuous")
    assert result["arm_a"] == "reservoir" and result["arm_b"] == "baseline"
    assert result["reservoir"]["values"] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert result["baseline"]["values"] == [6.0, 7.0, 8.0, 9.0, 10.0]
    for key in ("welch_ttest", "permutation_test", "mann_whitney_u", "bootstrap_ci_diff"):
        assert isinstance(result[key], dict)
    assert isinstance(result["cohens_d"], float)
    # This is exactly the A vs B reference case (reservoir=A, baseline=B).
    assert result["welch_ttest"]["t_statistic"] == pytest.approx(-5.0, abs=1e-9)
    assert result["permutation_test"]["p_value"] == pytest.approx(2 / 252, abs=1e-9)


def test_compare_arms_raises_when_an_arm_is_entirely_missing(tmp_path):
    _write_eval_json(tmp_path, "reservoir", 0, "continuous")
    records = load_eval_results(tmp_path)
    with pytest.raises(ValueError, match="baseline"):
        compare_arms(records, "continuous")


# ---------------------------------------------------------------------------
# summarise_training_logs
# ---------------------------------------------------------------------------

def _log_line(step, update, mean_extrinsic_reward, grad_norm):
    return json.dumps({
        "arm": "baseline", "seed": 0, "step": step, "update": update,
        "mean_reward": mean_extrinsic_reward, "mean_extrinsic_reward": mean_extrinsic_reward,
        "policy_loss": 0.1, "value_loss": 0.1, "entropy": 2.3, "total_loss": 0.2,
        "grad_norm": grad_norm,
    })


def test_summarise_training_logs_skips_unparseable_trailing_line(tmp_path):
    """Simulates a run killed mid-write: 10 clean lines, then a truncated
    half-written JSON object with no closing brace."""
    ckpt_dir = tmp_path / "baseline_seed0"
    ckpt_dir.mkdir()
    lines = [_log_line(step=(i + 1) * 100, update=i + 1,
                       mean_extrinsic_reward=float(i), grad_norm=float(i) + 1.0)
             for i in range(10)]
    lines.append('{"arm": "baseline", "seed": 0, "step": 1100, "update": 11, "mean_r')
    (ckpt_dir / "train_log.jsonl").write_text("\n".join(lines) + "\n")

    summary = summarise_training_logs(tmp_path)
    s = summary["baseline"][0]
    assert s["n_updates"] == 10
    assert s["n_skipped_lines"] == 1
    assert s["final_step"] == 1000  # from the 10th clean line, not the truncated 11th


def test_summarise_training_logs_first_and_last_10_percent_and_grad_norm_stats(tmp_path):
    ckpt_dir = tmp_path / "reservoir_seed2"
    ckpt_dir.mkdir()
    # 10 updates -> 10% window = 1 update each side. Rewards climb 0..9, so the
    # first-10% mean is 0.0 and the last-10% mean is 9.0: an unambiguous signal
    # this function must recover exactly, not merely "improve".
    lines = [_log_line(step=(i + 1) * 100, update=i + 1,
                       mean_extrinsic_reward=float(i), grad_norm=float(i))
             for i in range(10)]
    (ckpt_dir / "train_log.jsonl").write_text("\n".join(lines) + "\n")

    summary = summarise_training_logs(tmp_path)
    s = summary["reservoir"][2]
    assert s["n_updates"] == 10
    assert s["n_skipped_lines"] == 0
    assert s["final_step"] == 1000
    assert s["mean_extrinsic_reward_first_10pct"] == pytest.approx(0.0)
    assert s["mean_extrinsic_reward_last_10pct"] == pytest.approx(9.0)
    assert s["grad_norm_min"] == pytest.approx(0.0)
    assert s["grad_norm_median"] == pytest.approx(4.5)
    assert s["grad_norm_max"] == pytest.approx(9.0)


def test_summarise_training_logs_skips_dirs_without_a_log_and_unrelated_dirs(tmp_path):
    (tmp_path / "baseline_seed0").mkdir()  # no train_log.jsonl inside
    (tmp_path / "not_a_checkpoint_dir").mkdir()
    (tmp_path / "some_file.txt").write_text("irrelevant")
    assert summarise_training_logs(tmp_path) == {}


def test_summarise_training_logs_on_missing_dir_returns_empty_dict(tmp_path):
    assert summarise_training_logs(tmp_path / "does_not_exist") == {}


# ---------------------------------------------------------------------------
# list_checkpoints / select_best_checkpoint / select_final_checkpoint /
# build_eval_manifest
#
# All fake checkpoints below are EMPTY files (Path.touch()). That is a
# deliberate assertion in itself, not laziness: these functions operate only
# on filenames and on train_log.jsonl, and must never need to actually load a
# torch checkpoint's contents -- an empty file is the strongest available
# proof of that, since a real torch.load could not even succeed on one.
# ---------------------------------------------------------------------------

def _touch_checkpoints(run_dir: Path, steps):
    """Creates one empty `step_{N}.pt` per step in `steps` under `run_dir`
    (created if needed). See the section comment above for why empty."""
    run_dir.mkdir(parents=True, exist_ok=True)
    for step in steps:
        (run_dir / f"step_{step}.pt").touch()


def test_list_checkpoints_sorts_numerically_not_lexicographically(tmp_path):
    """The exact case a lexicographic sort gets wrong: as strings,
    'step_1000064.pt' < 'step_900864.pt' (comparing the first differing
    character, '1' < '9'), which is the OPPOSITE of the correct numeric
    order. This is not a contrived edge case -- it is the last checkpoint of
    every 1,000,000-step run this project produces, which crosses exactly
    this 6-digit -> 7-digit boundary."""
    run_dir = tmp_path / "reservoir_seed0"
    _touch_checkpoints(run_dir, [900864, 1000064])

    # Confirm the premise: a naive string sort really would get this wrong.
    assert sorted(["step_900864.pt", "step_1000064.pt"])[-1] == "step_900864.pt"

    checkpoints = list_checkpoints(run_dir)
    assert [step for step, _path in checkpoints] == [900864, 1000064]
    # Empty files -- proves no checkpoint content was ever read.
    for _step, path in checkpoints:
        assert os.path.getsize(path) == 0


def test_list_checkpoints_raises_on_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_checkpoints(tmp_path / "does_not_exist")


def test_list_checkpoints_raises_on_directory_with_no_checkpoints(tmp_path):
    run_dir = tmp_path / "baseline_seed0"
    run_dir.mkdir()
    (run_dir / "train_log.jsonl").write_text("")  # a log but zero .pt files
    with pytest.raises(ValueError, match="no step_\\*\\.pt"):
        list_checkpoints(run_dir)


def test_select_best_checkpoint_picks_the_peak_not_the_last(tmp_path):
    """Hand-constructed to mirror the reservoir arm's real failure mode:
    reward rises, peaks at the MIDDLE checkpoint, then regresses. Windows
    (default, checkpoint-spacing-based) are:
      checkpoint@300: updates with step in (0, 300]   -> rewards [1, 2, 3]  -> mean 2
      checkpoint@600: updates with step in (300, 600]  -> rewards [10,10,10] -> mean 10  (peak)
      checkpoint@900: updates with step in (600, 900]  -> rewards [1, 1, 1]  -> mean 1  (regressed)
    The best checkpoint must be the MIDDLE one -- that is the entire point of
    this selection rule existing instead of just taking the last checkpoint.
    """
    run_dir = tmp_path / "reservoir_seed0"
    _touch_checkpoints(run_dir, [300, 600, 900])
    lines = [
        _log_line(step=100, update=1, mean_extrinsic_reward=1.0, grad_norm=1.0),
        _log_line(step=200, update=2, mean_extrinsic_reward=2.0, grad_norm=1.0),
        _log_line(step=300, update=3, mean_extrinsic_reward=3.0, grad_norm=1.0),
        _log_line(step=400, update=4, mean_extrinsic_reward=10.0, grad_norm=1.0),
        _log_line(step=500, update=5, mean_extrinsic_reward=10.0, grad_norm=1.0),
        _log_line(step=600, update=6, mean_extrinsic_reward=10.0, grad_norm=1.0),
        _log_line(step=700, update=7, mean_extrinsic_reward=1.0, grad_norm=1.0),
        _log_line(step=800, update=8, mean_extrinsic_reward=1.0, grad_norm=1.0),
        _log_line(step=900, update=9, mean_extrinsic_reward=1.0, grad_norm=1.0),
    ]
    (run_dir / "train_log.jsonl").write_text("\n".join(lines) + "\n")

    result = select_best_checkpoint(run_dir)
    assert result["step"] == 600
    assert result["path"] == str(run_dir / "step_600.pt")
    assert result["score"] == pytest.approx(10.0)
    assert result["n_updates_in_window"] == 3
    assert result["all_scores"] == [
        (300, pytest.approx(2.0)),
        (600, pytest.approx(10.0)),
        (900, pytest.approx(1.0)),
    ]


def test_select_best_checkpoint_tie_breaks_to_the_earlier_step(tmp_path):
    """Two checkpoints whose windows score IDENTICALLY (mean=5.0 both): the
    earlier (step=300) must win, not the later one."""
    run_dir = tmp_path / "reservoir_seed1"
    _touch_checkpoints(run_dir, [300, 600])
    lines = [
        _log_line(step=100, update=1, mean_extrinsic_reward=5.0, grad_norm=1.0),
        _log_line(step=200, update=2, mean_extrinsic_reward=5.0, grad_norm=1.0),
        _log_line(step=300, update=3, mean_extrinsic_reward=5.0, grad_norm=1.0),
        _log_line(step=400, update=4, mean_extrinsic_reward=5.0, grad_norm=1.0),
        _log_line(step=500, update=5, mean_extrinsic_reward=5.0, grad_norm=1.0),
        _log_line(step=600, update=6, mean_extrinsic_reward=5.0, grad_norm=1.0),
    ]
    (run_dir / "train_log.jsonl").write_text("\n".join(lines) + "\n")

    result = select_best_checkpoint(run_dir)
    assert result["step"] == 300
    assert result["score"] == pytest.approx(5.0)


def test_select_best_checkpoint_scores_empty_window_as_negative_infinity(tmp_path):
    """checkpoint@600's window (300, 600] has zero logged updates (the log
    only has entries up to step 300) -- must score -inf with
    n_updates_in_window=0, and must NOT crash. checkpoint@300 has a real
    score, so it wins overall (any finite score beats -inf)."""
    run_dir = tmp_path / "baseline_seed0"
    _touch_checkpoints(run_dir, [300, 600])
    lines = [
        _log_line(step=100, update=1, mean_extrinsic_reward=1.0, grad_norm=1.0),
        _log_line(step=200, update=2, mean_extrinsic_reward=2.0, grad_norm=1.0),
        _log_line(step=300, update=3, mean_extrinsic_reward=3.0, grad_norm=1.0),
    ]
    (run_dir / "train_log.jsonl").write_text("\n".join(lines) + "\n")

    result = select_best_checkpoint(run_dir)
    assert result["all_scores"][1] == (600, float("-inf"))
    assert result["step"] == 300  # only checkpoint with any evidence at all


def test_select_best_checkpoint_handles_missing_train_log_without_crashing(tmp_path):
    """No train_log.jsonl at all (e.g. checkpoint written before the first
    logged update) -- every window is empty, every score is -inf, and the
    function must still return a result (the earliest checkpoint, by the tie
    rule), not raise."""
    run_dir = tmp_path / "reservoir_seed2"
    _touch_checkpoints(run_dir, [128, 256])
    result = select_best_checkpoint(run_dir)
    assert result["step"] == 128
    assert result["score"] == float("-inf")
    assert result["n_updates_in_window"] == 0


def test_select_final_checkpoint_returns_highest_step_not_lexicographic_max(tmp_path):
    run_dir = tmp_path / "baseline_seed3"
    _touch_checkpoints(run_dir, [900864, 1000064])

    # Confirm the premise again: naive string max picks the wrong file here.
    assert max("step_900864.pt", "step_1000064.pt") == "step_900864.pt"

    result = select_final_checkpoint(run_dir)
    assert result["step"] == 1000064
    assert result["path"] == str(run_dir / "step_1000064.pt")


def test_build_eval_manifest_skips_missing_run_dir_and_reports_it(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    baseline_dir = checkpoint_dir / "baseline_seed0"
    _touch_checkpoints(baseline_dir, [128, 256])
    (baseline_dir / "train_log.jsonl").write_text(
        _log_line(step=128, update=1, mean_extrinsic_reward=1.0, grad_norm=1.0) + "\n"
        + _log_line(step=256, update=2, mean_extrinsic_reward=2.0, grad_norm=1.0) + "\n"
    )
    # reservoir_seed0 is deliberately never created -- its run has not started.

    result = build_eval_manifest(checkpoint_dir, arms=("baseline", "reservoir"),
                                   seeds=[0], selection="final")
    assert result["manifest"] == [{
        "arm": "baseline", "seed": 0, "step": 256,
        "path": str(baseline_dir / "step_256.pt"), "selection": "final",
    }]
    assert len(result["missing"]) == 1
    assert result["missing"][0]["arm"] == "reservoir"
    assert result["missing"][0]["seed"] == 0
    assert "does not exist" in result["missing"][0]["reason"]


def test_build_eval_manifest_skips_run_dir_with_no_checkpoints_yet(tmp_path):
    """A run directory that EXISTS (e.g. train_log.jsonl has started) but has
    not written its first checkpoint yet -- also must be skipped into
    `missing`, not crash, since build_eval_manifest is explicitly meant to be
    called mid-experiment."""
    checkpoint_dir = tmp_path / "checkpoints"
    started_dir = checkpoint_dir / "reservoir_seed0"
    started_dir.mkdir(parents=True)
    (started_dir / "train_log.jsonl").write_text(
        _log_line(step=128, update=1, mean_extrinsic_reward=1.0, grad_norm=1.0) + "\n"
    )

    result = build_eval_manifest(checkpoint_dir, arms=("reservoir",), seeds=[0],
                                   selection="best")
    assert result["manifest"] == []
    assert len(result["missing"]) == 1
    assert "no checkpoints yet" in result["missing"][0]["reason"]


def test_build_eval_manifest_rejects_unknown_selection(tmp_path):
    with pytest.raises(ValueError, match="selection"):
        build_eval_manifest(tmp_path, selection="not_a_real_rule")


# ---------------------------------------------------------------------------
# JSON safety (non-finite floats -> null)
# ---------------------------------------------------------------------------

def test_json_safe_converts_nonfinite_floats_to_none():
    payload = {
        "a": float("nan"),
        "b": float("inf"),
        "c": -float("inf"),
        "d": 1.5,
        "e": [float("nan"), 2, "text"],
        "f": {"g": float("inf")},
    }
    safe = _json_safe(payload)
    # Must be genuinely JSON-serialisable now, not merely "look right" -- round
    # trip it for real.
    round_tripped = json.loads(json.dumps(safe))
    assert round_tripped["a"] is None
    assert round_tripped["b"] is None
    assert round_tripped["c"] is None
    assert round_tripped["d"] == 1.5
    assert round_tripped["e"] == [None, 2, "text"]
    assert round_tripped["f"]["g"] is None


# ---------------------------------------------------------------------------
# End-to-end: build_report + the CLI's --json / human-readable output
# ---------------------------------------------------------------------------

def _populate_full_scenario(results_dir: Path, checkpoint_dir: Path):
    for seed, val in enumerate([10.0, 12.0, 11.0, 13.0, 9.0]):
        _write_eval_json(results_dir, "reservoir", seed, "continuous", mean_extrinsic_return=val)
    for seed, val in enumerate([8.0, 7.0, 9.0, 6.0, 10.0]):
        _write_eval_json(results_dir, "baseline", seed, "continuous", mean_extrinsic_return=val)
    for arm in ("reservoir", "baseline"):
        d = checkpoint_dir / f"{arm}_seed0"
        d.mkdir(parents=True)
        lines = [_log_line(step=(i + 1) * 100, update=i + 1,
                           mean_extrinsic_reward=float(i), grad_norm=1.0)
                 for i in range(10)]
        (d / "train_log.jsonl").write_text("\n".join(lines) + "\n")


def test_build_report_end_to_end(tmp_path):
    results_dir = tmp_path / "results"
    checkpoint_dir = tmp_path / "checkpoints"
    results_dir.mkdir()
    checkpoint_dir.mkdir()
    _populate_full_scenario(results_dir, checkpoint_dir)

    bundle = build_report(str(results_dir), str(checkpoint_dir))
    assert "error" not in bundle["continuous"]
    assert bundle["reset128"] == {
        "error": (
            "compare_arms: no eval results for arm(s) ['reservoir', 'baseline'] under "
            "regime='reset128', metric='mean_extrinsic_return' -- cannot compare. "
            "Arms found: []"
        )
    }
    assert "reservoir" in bundle["training_logs"]
    assert "baseline" in bundle["training_logs"]

    # Round-trips through the exact JSON machinery the CLI uses.
    from analysis.aggregate_results import _json_safe as json_safe
    dumped = json.dumps(json_safe(bundle))
    json.loads(dumped)


def test_human_readable_report_never_prints_a_verdict(tmp_path):
    """This project's culture (see module docstring) requires reporting the
    numbers and letting the reader judge -- a printed 'wins'/'beats' line would
    invite exactly the overstatement that culture exists to avoid."""
    from analysis.aggregate_results import _format_comparison

    results_dir = tmp_path / "results"
    checkpoint_dir = tmp_path / "checkpoints"
    results_dir.mkdir()
    checkpoint_dir.mkdir()
    _populate_full_scenario(results_dir, checkpoint_dir)
    bundle = build_report(str(results_dir), str(checkpoint_dir))

    report_text = _format_comparison(bundle["continuous"])
    lowered = report_text.lower()
    for banned in ("wins", "beats", "winner", "superior", " better\n", "victory"):
        assert banned not in lowered, f"report should not editorialise, found {banned!r}"
    # But it must still contain the actual numbers a reader needs to judge for themselves.
    assert "p=" in report_text
    assert "n=5" in report_text


def test_cli_runs_end_to_end_and_produces_valid_json(tmp_path):
    """Exercises the real `python -m analysis.aggregate_results --json` entry
    point in a subprocess -- the only way to confirm the CLI is actually wired
    up (argparse defaults, __main__ block) rather than merely that its
    underlying functions work in-process."""
    results_dir = tmp_path / "results"
    checkpoint_dir = tmp_path / "checkpoints"
    results_dir.mkdir()
    checkpoint_dir.mkdir()
    _populate_full_scenario(results_dir, checkpoint_dir)

    proc = subprocess.run(
        [sys.executable, "-m", "analysis.aggregate_results",
         "--results-dir", str(results_dir), "--checkpoint-dir", str(checkpoint_dir), "--json"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert "continuous" in payload and "reset128" in payload and "training_logs" in payload
    assert "error" not in payload["continuous"]


def test_cli_human_readable_mode_also_runs_end_to_end(tmp_path):
    results_dir = tmp_path / "results"
    checkpoint_dir = tmp_path / "checkpoints"
    results_dir.mkdir()
    checkpoint_dir.mkdir()
    _populate_full_scenario(results_dir, checkpoint_dir)

    proc = subprocess.run(
        [sys.executable, "-m", "analysis.aggregate_results",
         "--results-dir", str(results_dir), "--checkpoint-dir", str(checkpoint_dir)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "regime=continuous" in proc.stdout
    assert "training log summary" in proc.stdout
