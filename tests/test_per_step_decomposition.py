"""Tests for `analysis/per_step_decomposition.py`.

Two kinds of test live here, and they check different things:

1. SYNTHETIC-DATA tests (small, hand-constructed `tmp_path` fixtures) pin down
   the MECHANICS: the per-seed-then-average ordering, graceful handling of
   missing files/arms/seeds, the permutation-test wiring, and the CLI. These
   use tiny numbers chosen so every expected value has a hand-checkable closed
   form, the same discipline `tests/test_aggregate_results.py` uses for the
   primitives it pins.

2. ORACLE tests, against the REAL `results/final` and `results/init`
   directories committed to this repo, assert this module reproduces the
   exact numbers `docs/RESULTS.md` §5 publishes. This is the whole point of
   the module: new tooling that reproduces an already-published measurement
   is trustworthy in a way that new tooling producing only new numbers is
   not (see `docs/EXPERIMENT_LOG.md` §17.11 for the precedent of what
   happens when tooling and prose DISAGREE -- it gets reported, not
   smoothed over with a looser tolerance). If any oracle assertion below
   ever fails, the fix is to investigate the discrepancy, not to widen the
   tolerance or edit the published number.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from analysis.per_step_decomposition import (
    _records_with_reward_per_step,
    decompose,
    format_table,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_eval_json(results_dir: Path, arm: str, train_seed: int, regime: str,
                      *, mean_extrinsic_return: float, mean_episode_length: float,
                      n_episodes: int = 3):
    """Writes one fake eval JSON, matching `training/evaluate.py --json`'s
    schema closely enough for `decompose` under test: the `arm`/`state_reset_
    interval` cross-checks `load_eval_results` performs, plus whichever
    `mean_extrinsic_return`/`mean_episode_length` pair a test needs. Mirrors
    `tests/test_aggregate_results.py`'s own `_write_eval_json` helper closely
    on purpose -- same fixture shape, same convention -- extended with a
    controllable `mean_episode_length` since that is the second input this
    module's division needs and the sibling module's helper hardcodes it."""
    state_reset_interval = None if regime == "continuous" else 128
    payload = {
        "arm": arm,
        "n_episodes": n_episodes,
        "seed": 999,
        "state_reset_interval": state_reset_interval,
        "mean_extrinsic_return": mean_extrinsic_return,
        "std_extrinsic_return": 1.0,
        "sem_extrinsic_return": 0.5,
        "extrinsic_returns": [mean_extrinsic_return] * n_episodes,
        "mean_combined_return": mean_extrinsic_return + 1.0,
        "std_combined_return": 1.0,
        "sem_combined_return": 0.5,
        "combined_returns": [mean_extrinsic_return + 1.0] * n_episodes,
        "mean_episode_length": mean_episode_length,
        "std_episode_length": 1.0,
        "sem_episode_length": 0.5,
        "episode_lengths": [mean_episode_length] * n_episodes,
    }
    filename = f"eval_{arm}_seed{train_seed}_{regime}.json"
    (results_dir / filename).write_text(json.dumps(payload))
    return filename


def _populate(results_dir: Path, arm: str, regime: str, returns: list, lengths: list,
              seeds=None):
    """Writes one eval JSON per (return, length) pair, seeds defaulting to
    0..len(returns)-1."""
    if seeds is None:
        seeds = range(len(returns))
    for seed, ret, length in zip(seeds, returns, lengths):
        _write_eval_json(results_dir, arm, seed, regime,
                          mean_extrinsic_return=ret, mean_episode_length=length)


# ---------------------------------------------------------------------------
# _records_with_reward_per_step
# ---------------------------------------------------------------------------

def test_records_with_reward_per_step_computes_per_record_ratio():
    records = [
        {"arm": "baseline", "train_seed": 0, "mean_extrinsic_return": 10.0,
         "mean_episode_length": 100.0},
        {"arm": "reservoir", "train_seed": 0, "mean_extrinsic_return": 9.0,
         "mean_episode_length": 3.0},
    ]
    out = _records_with_reward_per_step(records)
    assert out[0]["reward_per_step"] == pytest.approx(0.1, abs=1e-12)
    assert out[1]["reward_per_step"] == pytest.approx(3.0, abs=1e-12)


def test_records_with_reward_per_step_does_not_mutate_input():
    records = [{"arm": "baseline", "train_seed": 0, "mean_extrinsic_return": 10.0,
                "mean_episode_length": 100.0}]
    _records_with_reward_per_step(records)
    assert "reward_per_step" not in records[0]


def test_records_with_reward_per_step_rejects_zero_length():
    records = [{"arm": "baseline", "train_seed": 0, "mean_extrinsic_return": 10.0,
                "mean_episode_length": 0.0, "source_file": "eval_baseline_seed0_continuous.json"}]
    with pytest.raises(ValueError, match="mean_episode_length == 0.0"):
        _records_with_reward_per_step(records)


# ---------------------------------------------------------------------------
# decompose: the load-bearing per-seed-then-average ordering
# ---------------------------------------------------------------------------

def test_decompose_uses_mean_of_ratios_not_ratio_of_means(tmp_path):
    """THE test that matters most in this file. Two seeds, deliberately
    constructed so mean-of-ratios and ratio-of-means give DIFFERENT answers
    (the toy example from the module's own docstring):

      seed 0: return=10, length=100 -> ratio 0.1
      seed 1: return=10, length=1   -> ratio 10.0

    mean-of-ratios  (per-seed-then-average, what this module must compute):
      (0.1 + 10.0) / 2 = 5.05

    ratio-of-means (what it must NOT compute):
      mean(returns)/mean(lengths) = 10 / 50.5 = 0.198019...

    These differ by more than 25x, so a wrong ordering fails this test loudly
    rather than by a rounding-sized margin -- exactly what a wrong-but-close
    number would NOT do.
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _populate(results_dir, "baseline", "continuous",
              returns=[10.0, 10.0], lengths=[100.0, 1.0])
    _populate(results_dir, "reservoir", "continuous",
              returns=[10.0, 10.0], lengths=[100.0, 1.0])

    result = decompose(results_dir, "continuous")

    mean_of_ratios = 5.05
    ratio_of_means = 10.0 / 50.5
    assert ratio_of_means == pytest.approx(0.19801980198019803, abs=1e-9)

    for arm in ("baseline", "reservoir"):
        got = result["arms"][arm]["mean_reward_per_step"]
        assert got == pytest.approx(mean_of_ratios, abs=1e-9)
        assert got != pytest.approx(ratio_of_means, abs=1e-3)

    # The mean-return and mean-length columns are still plain means (that
    # part legitimately IS a ratio of means territory, since they are not
    # rates) -- confirms the module isn't applying the per-seed trick where
    # it doesn't belong either.
    assert result["arms"]["baseline"]["mean_episode_return"] == pytest.approx(10.0, abs=1e-9)
    assert result["arms"]["baseline"]["mean_episode_length"] == pytest.approx(50.5, abs=1e-9)


# ---------------------------------------------------------------------------
# decompose: graceful partial operation
# ---------------------------------------------------------------------------

def test_decompose_skips_gracefully_when_results_dir_missing(tmp_path):
    missing = tmp_path / "does_not_exist"
    result = decompose(missing, "continuous")
    assert result["skipped"] is True
    assert "error" in result and "does not exist" in result["error"]
    assert result["arms"]["baseline"] is None
    assert result["arms"]["reservoir"] is None


def test_decompose_rejects_bad_regime(tmp_path):
    with pytest.raises(ValueError, match="regime"):
        decompose(tmp_path, "not_a_real_regime")


def test_decompose_handles_one_arm_entirely_missing(tmp_path):
    """Only the baseline arm was ever evaluated for this regime (e.g. a
    still-running results_v2 matrix). The baseline's own numbers must still
    be reported; the cross-arm ratio/permutation numbers cannot exist without
    both arms and must be absent, with a clear reason recorded."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _populate(results_dir, "baseline", "continuous",
              returns=[10.0] * 3, lengths=[100.0] * 3)

    result = decompose(results_dir, "continuous")

    assert result["arms"]["baseline"] is not None
    assert result["arms"]["baseline"]["n"] == 3
    assert result["arms"]["reservoir"] is None
    assert "ratio_baseline_over_reservoir_reward_per_step" not in result
    assert "permutation_reward_per_step" not in result
    assert "reservoir" in result["error"]


def test_decompose_reports_missing_seeds_without_raising(tmp_path):
    """Only 4 of the (default) expected 10 training seeds are present for the
    reservoir arm. decompose must compute over those 4 rather than raising,
    and must say exactly which 6 seeds are absent."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _populate(results_dir, "baseline", "continuous",
              returns=[10.0] * 10, lengths=[100.0] * 10, seeds=range(10))
    _populate(results_dir, "reservoir", "continuous",
              returns=[5.0] * 4, lengths=[50.0] * 4, seeds=[0, 2, 4, 6])

    result = decompose(results_dir, "continuous")

    res = result["arms"]["reservoir"]
    assert res["n"] == 4
    assert res["train_seeds"] == [0, 2, 4, 6]
    assert res["missing_seeds"] == [1, 3, 5, 7, 8, 9]

    base = result["arms"]["baseline"]
    assert base["n"] == 10
    assert base["missing_seeds"] == []


# ---------------------------------------------------------------------------
# decompose: ratio + permutation wiring, cross-checked against a fully
# hand-verifiable case (mean_episode_length pinned to 1.0 so reward_per_step
# == mean_extrinsic_return exactly, then reusing the SAME fully-separated
# A=[1..5] vs B=[6..10] case tests/test_aggregate_results.py pins its own
# exact_permutation_test reference value against: 2/252 = 0.0079365...).
# A match here means this module's call into aggregate_results.
# exact_permutation_test is wired correctly, not merely "doesn't crash".
# ---------------------------------------------------------------------------

def test_decompose_permutation_and_ratio_match_hand_checked_reference(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    reservoir_returns = [1.0, 2.0, 3.0, 4.0, 5.0]
    baseline_returns = [6.0, 7.0, 8.0, 9.0, 10.0]
    _populate(results_dir, "reservoir", "continuous",
              returns=reservoir_returns, lengths=[1.0] * 5)
    _populate(results_dir, "baseline", "continuous",
              returns=baseline_returns, lengths=[1.0] * 5)

    result = decompose(results_dir, "continuous")

    assert result["arms"]["reservoir"]["mean_reward_per_step"] == pytest.approx(3.0, abs=1e-9)
    assert result["arms"]["baseline"]["mean_reward_per_step"] == pytest.approx(8.0, abs=1e-9)
    assert result["ratio_baseline_over_reservoir_reward_per_step"] == pytest.approx(
        8.0 / 3.0, abs=1e-9)

    p_rps = result["permutation_reward_per_step"]
    assert p_rps["method"] == "exact"
    assert p_rps["n_permutations"] == 252
    # CONVENTION per aggregate_results.exact_permutation_test / compare_arms:
    # a=reservoir, b=baseline -> observed_diff = mean(reservoir) - mean(baseline).
    assert p_rps["observed_diff"] == pytest.approx(3.0 - 8.0, abs=1e-9)
    assert p_rps["p_value"] == pytest.approx(2 / 252, abs=1e-9)

    # Lengths are identical (1.0) across every seed of both arms here, so the
    # length permutation test has zero spread and the maximally-uncertain
    # p=1.0 is the only correct answer -- exercises that this module calls
    # exact_permutation_test a SECOND, independent time for length rather than
    # reusing the reward-per-step result.
    p_len = result["permutation_episode_length"]
    assert p_len["observed_diff"] == pytest.approx(0.0, abs=1e-9)
    assert p_len["p_value"] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# format_table
# ---------------------------------------------------------------------------

def test_format_table_skipped_case(tmp_path):
    result = decompose(tmp_path / "nope", "continuous")
    text = format_table(result)
    assert "SKIPPED" in text
    assert "does not exist" in text


def test_format_table_full_case_contains_key_numbers(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _populate(results_dir, "reservoir", "continuous",
              returns=[1.0, 2.0, 3.0, 4.0, 5.0], lengths=[1.0] * 5)
    _populate(results_dir, "baseline", "continuous",
              returns=[6.0, 7.0, 8.0, 9.0, 10.0], lengths=[1.0] * 5)
    result = decompose(results_dir, "continuous")
    text = format_table(result)

    assert "baseline" in text
    assert "reservoir" in text
    assert "reward-per-step ratio" in text
    assert "permutation test, reward per step" in text
    assert "permutation test, mean episode length" in text
    # The per-seed-then-average disclaimer must survive into the printed
    # table, not just live in the docstring -- a reader of the CLI output
    # should not have to open the source to learn the ordering convention.
    assert "PER TRAINING SEED" in text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_prints_table_for_synthetic_data(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _populate(results_dir, "reservoir", "continuous",
              returns=[1.0, 2.0, 3.0, 4.0, 5.0], lengths=[1.0] * 5)
    _populate(results_dir, "baseline", "continuous",
              returns=[6.0, 7.0, 8.0, 9.0, 10.0], lengths=[1.0] * 5)

    proc = subprocess.run(
        [sys.executable, "-m", "analysis.per_step_decomposition",
         "--results-dir", str(results_dir), "--regime", "continuous"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "baseline" in proc.stdout
    assert "reservoir" in proc.stdout
    assert "reward-per-step ratio" in proc.stdout


def test_cli_requires_results_dir_and_regime():
    proc = subprocess.run(
        [sys.executable, "-m", "analysis.per_step_decomposition"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode != 0


def test_cli_rejects_bad_regime_choice():
    proc = subprocess.run(
        [sys.executable, "-m", "analysis.per_step_decomposition",
         "--results-dir", "results/final", "--regime", "bogus"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode != 0
    assert "invalid choice" in proc.stderr


# ---------------------------------------------------------------------------
# ORACLE: reproduces docs/RESULTS.md §5's published v1 numbers exactly, from
# the real results/final and results/init directories committed to this repo.
# See this file's module docstring: a failure here is a real finding to be
# reported, not a tolerance to be loosened.
# ---------------------------------------------------------------------------

FINAL_DIR = REPO_ROOT / "results" / "final"
INIT_DIR = REPO_ROOT / "results" / "init"

# The permutation design's resolution floor: 2 (the observed split plus its
# mirror image) out of C(20,10)=184,756 possible 10-vs-10 splits. §2.6 states
# this is what a reported "0.000011" means -- "at the floor", not "vanishingly
# small". Computed once here (rather than hardcoding the rounded 0.000011) so
# the oracle assertions below compare against the exact rational value.
PERMUTATION_FLOOR_P = 2 / 184756


@pytest.mark.skipif(not FINAL_DIR.is_dir(), reason="results/final not present in this checkout")
def test_oracle_final_continuous_matches_results_md_section_5():
    result = decompose(FINAL_DIR, "continuous")

    baseline = result["arms"]["baseline"]
    reservoir = result["arms"]["reservoir"]
    assert baseline["n"] == 10 and reservoir["n"] == 10

    # baseline trained: return 36.134, mean length 314.9, reward/step 0.11455
    assert baseline["mean_episode_return"] == pytest.approx(36.134, abs=1e-3)
    assert baseline["mean_episode_length"] == pytest.approx(314.9, abs=0.05)
    assert baseline["mean_reward_per_step"] == pytest.approx(0.11455, abs=1e-4)

    # reservoir trained: return 28.417, mean length 1917.0, reward/step 0.01921
    assert reservoir["mean_episode_return"] == pytest.approx(28.417, abs=1e-3)
    assert reservoir["mean_episode_length"] == pytest.approx(1917.0, abs=0.05)
    assert reservoir["mean_reward_per_step"] == pytest.approx(0.01921, abs=1e-4)

    # ratio 5.96x
    assert result["ratio_baseline_over_reservoir_reward_per_step"] == pytest.approx(
        5.96, abs=0.01)

    # exact permutation p = 0.000011 (the floor) on BOTH the reward-per-step
    # difference and the episode-length difference.
    p_rps = result["permutation_reward_per_step"]
    assert p_rps["method"] == "exact"
    assert p_rps["n_permutations"] == 184756
    assert p_rps["p_value"] == pytest.approx(PERMUTATION_FLOOR_P, abs=1e-9)

    p_len = result["permutation_episode_length"]
    assert p_len["method"] == "exact"
    assert p_len["p_value"] == pytest.approx(PERMUTATION_FLOOR_P, abs=1e-9)


@pytest.mark.skipif(not FINAL_DIR.is_dir(), reason="results/final not present in this checkout")
def test_oracle_final_reset128_matches_results_md_section_5():
    result = decompose(FINAL_DIR, "reset128")

    baseline = result["arms"]["baseline"]
    reservoir = result["arms"]["reservoir"]

    # per step 0.11322 vs 0.01937, ratio 5.85x; lengths 315.4 vs 1906.6
    assert baseline["mean_reward_per_step"] == pytest.approx(0.11322, abs=1e-4)
    assert reservoir["mean_reward_per_step"] == pytest.approx(0.01937, abs=1e-4)
    assert result["ratio_baseline_over_reservoir_reward_per_step"] == pytest.approx(
        5.85, abs=0.01)
    assert baseline["mean_episode_length"] == pytest.approx(315.4, abs=0.05)
    assert reservoir["mean_episode_length"] == pytest.approx(1906.6, abs=0.05)


@pytest.mark.skipif(not INIT_DIR.is_dir(), reason="results/init not present in this checkout")
def test_oracle_init_continuous_matches_results_md_section_5():
    result = decompose(INIT_DIR, "continuous")

    baseline = result["arms"]["baseline"]
    reservoir = result["arms"]["reservoir"]

    # baseline untrained: return 8.123, length 2825.8, reward/step 0.00287
    assert baseline["mean_episode_return"] == pytest.approx(8.123, abs=1e-3)
    assert baseline["mean_episode_length"] == pytest.approx(2825.8, abs=0.05)
    assert baseline["mean_reward_per_step"] == pytest.approx(0.00287, abs=1e-5)

    # reservoir untrained: return 9.963, length 2491.1, reward/step 0.00527
    assert reservoir["mean_episode_return"] == pytest.approx(9.963, abs=1e-3)
    assert reservoir["mean_episode_length"] == pytest.approx(2491.1, abs=0.05)
    assert reservoir["mean_reward_per_step"] == pytest.approx(0.00527, abs=1e-5)


@pytest.mark.skipif(not FINAL_DIR.is_dir(), reason="results/final not present in this checkout")
def test_oracle_cli_output_for_final_continuous(tmp_path):
    """The CLI, invoked exactly as a user would, against the real committed
    results/final directory -- not a synthetic fixture. Confirms the module
    is actually wired up end to end (argparse defaults, __main__ block,
    decompose, format_table) against the real oracle data, not merely that
    the library function returns the right dict."""
    proc = subprocess.run(
        [sys.executable, "-m", "analysis.per_step_decomposition",
         "--results-dir", "results/final", "--regime", "continuous"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "36.13" in proc.stdout   # baseline mean return
    assert "0.1145" in proc.stdout  # baseline reward per step
    assert "1916.9" in proc.stdout or "1917.0" in proc.stdout  # reservoir mean length
    assert "5.96" in proc.stdout    # ratio
