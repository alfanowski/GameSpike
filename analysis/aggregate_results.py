"""Turns per-checkpoint `training/evaluate.py --json` output into the arm-vs-arm
statistical comparison design doc §5 mandates. This module's output IS the
scientific verdict for this project's central experiment (frozen reservoir vs.
matched-parameter trained GRU), so several decisions below are deliberate and
recorded here rather than left implicit:

1. STDLIB + NUMPY ONLY, AND THAT IS NOT A COMPROMISE.
   No scipy. At the sample sizes this project actually has (a handful of
   independently-trained checkpoints per arm -- see point 2), scipy's own
   asymptotic shortcuts (the normal approximation to Mann-Whitney's null, the
   t-distribution's CDF via a library call that most builds compute the exact
   same way anyway) are not more correct than what is implemented here by hand:
   `welch_ttest`'s p-value is EXACT (via the regularized incomplete beta
   function, Numerical Recipes' Lentz's-method continued fraction -- the same
   algorithm scipy itself effectively wraps for this case), and
   `exact_permutation_test`/`mann_whitney_u` use EXACT enumeration whenever the
   sample is small enough to enumerate, which at n=10-vs-10 or smaller it always
   is (C(20,10)=184,756 splits -- milliseconds to walk on a laptop). An "exact
   test done by hand" beats an "asymptotic test from a library" at n=5-vs-5.

2. THE UNIT OF ANALYSIS IS THE TRAINING SEED, NOT THE EPISODE.
   `training/evaluate.py`'s own module docstring is explicit that its reported
   spread is POLICY-SAMPLING variance from one checkpoint, and that comparing
   arms honestly needs several INDEPENDENTLY-TRAINED checkpoints per arm,
   compared across those checkpoints. Pooling raw per-episode returns across
   training seeds here would silently undo that warning: it would treat
   within-checkpoint sampling noise (how lucky one already-frozen policy got on
   one episode) as if it were independent evidence about the ARCHITECTURE, which
   inflates the effective sample size and understates the true uncertainty.
   `aggregate_by_arm` therefore reduces every checkpoint to exactly ONE number
   (that checkpoint's own mean over its evaluation episodes) before any
   statistic in section 1 ever sees the data.

3. THE CLI NEVER PRINTS A VERDICT.
   Both prior projects in this lineage (see `docs/DESIGN.md` §1) report negative
   results as negative results, not reframed ones. A function that prints
   "arm X wins" invites exactly the overstatement that culture exists to avoid --
   especially here, where n is small enough that any single comparison is
   fragile. The human-readable formatter below prints numbers, spreads, sample
   sizes and p-values, in full, and stops there.

Section map: (1) statistical primitives, independently unit-tested; (2) loading
and aggregation; (2.5) checkpoint selection for evaluation (which checkpoint
of a run gets evaluated at all -- see that section's own header comment for
why this is its own careful piece of logic and not just "take the last one");
(3) a CLI wiring all of it together.
"""
import argparse
import glob
import itertools
import json
import math
import os
import re
import statistics
from collections import defaultdict, namedtuple
from functools import lru_cache

import numpy as np

# ---------------------------------------------------------------------------
# Section 1: statistical primitives
# ---------------------------------------------------------------------------

# Thresholds are module-level constants (not magic numbers buried in function
# bodies) so the "why 200,000" and "why 30" reasoning has one place to live and
# one place to change if the project ever runs with far more training seeds.
EXACT_PERMUTATION_THRESHOLD = 200_000  # C(n_total, n_a) at/below this -> exact
DEFAULT_MONTE_CARLO_RESAMPLES = 100_000
MWU_EXACT_TOTAL_THRESHOLD = 30  # combined sample size at/below this -> exact DP
DEFAULT_BOOTSTRAP_RESAMPLES = 20_000
DEFAULT_BOOTSTRAP_ALPHA = 0.05

WelchTTestResult = namedtuple("WelchTTestResult", ["t_statistic", "df", "p_value"])
PermutationTestResult = namedtuple(
    "PermutationTestResult", ["observed_diff", "p_value", "n_permutations", "method"])
MannWhitneyResult = namedtuple("MannWhitneyResult", ["U", "p_value", "method"])
BootstrapCIResult = namedtuple(
    "BootstrapCIResult", ["observed_diff", "ci_low", "ci_high", "n_resamples", "seed", "alpha"])


# --- regularized incomplete beta function (Numerical Recipes betai/betacf) ---
# This is the only nontrivial special-function machinery the module needs: the
# Student-t survival function has an EXACT closed form in terms of it
# (`p = I_{df/(df+t^2)}(df/2, 1/2)`, used by `welch_ttest` below), so
# implementing this one pair of functions buys an exact two-sided t-test
# p-value with zero external dependencies.
_BETA_MAXIT = 200
_BETA_EPS = 3e-16
_BETA_FPMIN = 1e-300


def _betacf(a: float, b: float, x: float) -> float:
    """Continued-fraction evaluation used by the regularized incomplete beta
    function, via Lentz's method (Numerical Recipes 3rd ed., §6.4). `_betai`
    below handles the prefactor and picks whichever tail this converges fastest
    on; this function does the actual continued-fraction sum.
    """
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _BETA_FPMIN:
        d = _BETA_FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, _BETA_MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _BETA_FPMIN:
            d = _BETA_FPMIN
        c = 1.0 + aa / c
        if abs(c) < _BETA_FPMIN:
            c = _BETA_FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _BETA_FPMIN:
            d = _BETA_FPMIN
        c = 1.0 + aa / c
        if abs(c) < _BETA_FPMIN:
            c = _BETA_FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _BETA_EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b) in [0, 1].

    x==0 and x==1 are handled by exact short-circuits (bt=0 exactly, matching
    I_0(a,b)=0 and I_1(a,b)=1 by definition) rather than trusting the continued
    fraction to converge at the domain edge -- this is what makes
    `welch_ttest`'s "identical samples -> p==1.0 exactly" behaviour exact rather
    than merely close.
    """
    if x < 0.0 or x > 1.0:
        raise ValueError(f"_betai: x must be in [0, 1], got {x}")
    if x == 0.0 or x == 1.0:
        bt = 0.0
    else:
        bt = math.exp(
            math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
            + a * math.log(x) + b * math.log(1.0 - x)
        )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _standard_normal_cdf(z: float) -> float:
    """Phi(z), via math.erf (stdlib) -- the one place a normal approximation is
    used at all (mann_whitney_u's large/tied-sample fallback)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def welch_ttest(a, b) -> WelchTTestResult:
    """Welch's t-test for a difference of means, safe for UNEQUAL variances --
    there is no reason to assume the reservoir and baseline arms' across-seed
    variances match, and the classic pooled-variance t-test silently assumes
    they do. Degrees of freedom via the Welch-Satterthwaite approximation;
    p-value EXACT via `_betai` (see module docstring point 1). Two-sided.

    Requires >=1 observation per group to even define a mean, but a SAMPLE
    VARIANCE needs >=2. With fewer than 2 in either group this returns NaN for
    every field rather than raising -- the same "cannot be measured, and NaN
    says so honestly" convention `training/evaluate.py`'s own `_summarise`
    uses for a single-episode run, applied here to a single-seed arm (an early,
    legitimate, if statistically weak, state for this project to be in).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 1 or n2 < 1:
        raise ValueError("welch_ttest needs at least 1 observation per group")
    if n1 < 2 or n2 < 2:
        return WelchTTestResult(t_statistic=float("nan"), df=float("nan"),
                                 p_value=float("nan"))

    mean1, mean2 = float(a.mean()), float(b.mean())
    var1, var2 = float(a.var(ddof=1)), float(b.var(ddof=1))
    se1_sq, se2_sq = var1 / n1, var2 / n2
    se_sum = se1_sq + se2_sq

    if se1_sq == 0.0 and se2_sq == 0.0:
        # Both groups have literally zero across-seed spread (every seed in
        # both arms landed on the exact same value). df is undefined (0/0);
        # NaN says so. The comparison itself is not undefined though: a zero
        # gap under zero noise is maximal certainty of "no difference" (t=0,
        # p=1), and a nonzero gap under zero noise is maximal certainty of
        # SOME difference (t=+-inf, p=0) -- there is no noise left to explain
        # a gap with, however small.
        if mean1 == mean2:
            return WelchTTestResult(t_statistic=0.0, df=float("nan"), p_value=1.0)
        return WelchTTestResult(t_statistic=math.copysign(math.inf, mean1 - mean2),
                                 df=float("nan"), p_value=0.0)

    t_stat = (mean1 - mean2) / math.sqrt(se_sum)
    df = (se_sum ** 2) / ((se1_sq ** 2) / (n1 - 1) + (se2_sq ** 2) / (n2 - 1))
    p_value = _betai(df / 2.0, 0.5, df / (df + t_stat ** 2))
    return WelchTTestResult(t_statistic=t_stat, df=df, p_value=p_value)


def exact_permutation_test(a, b, seed: int = 0,
                            n_resamples: int = DEFAULT_MONTE_CARLO_RESAMPLES,
                            exact_threshold: int = EXACT_PERMUTATION_THRESHOLD
                            ) -> PermutationTestResult:
    """Two-sided permutation test on the difference of means, mean(a) - mean(b).

    Under H0 ("arm label is exchangeable"), every way of splitting the pooled
    n_a+n_b values into groups of size n_a and n_b is equally likely. When the
    number of such splits, C(n_a+n_b, n_a), is at or below `exact_threshold`
    (default 200,000 -- C(20,10)=184,756 is comfortably under this, so two
    10-seed arms enumerate exactly), every split is walked and the two-sided
    p-value is EXACT: the fraction of splits whose |difference of means| is >=
    the observed one (the observed split itself always counts, since it is
    trivially >= itself). Above the threshold, a seeded Monte Carlo estimate
    with >= `n_resamples` resamples is used instead, with the returned
    `.method` recording which branch actually ran -- never leave the caller to
    guess whether a p-value is exact or estimated.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 1 or n2 < 1:
        raise ValueError("exact_permutation_test needs at least 1 observation per group")

    observed_diff = float(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    n_total = n1 + n2
    total_splits = math.comb(n_total, n1)

    if total_splits <= exact_threshold:
        # Pure-Python sum-over-indices beats numpy fancy-indexing here: with up
        # to ~185k iterations of a tight loop over <=~20 elements each, per-call
        # numpy overhead would dominate: plain list summation is both simpler
        # and faster at this scale.
        total_sum = float(pooled.sum())
        pooled_list = pooled.tolist()
        target = abs(observed_diff) - 1e-9  # tolerance against float round-off
        count = 0
        for idx in itertools.combinations(range(n_total), n1):
            sum_a = sum(pooled_list[i] for i in idx)
            mean_a = sum_a / n1
            mean_b = (total_sum - sum_a) / n2
            if abs(mean_a - mean_b) >= target:
                count += 1
        p_value = count / total_splits
        return PermutationTestResult(observed_diff, p_value, total_splits, "exact")

    rng = np.random.default_rng(seed)
    # Vectorized Monte Carlo: argsort of independent random keys is a cheap way
    # to draw `n_resamples` independent random permutations of the pooled array
    # at once, without a Python-level loop over each resample.
    random_keys = rng.random((n_resamples, n_total))
    perm_idx = np.argsort(random_keys, axis=1, kind="stable")
    a_means = pooled[perm_idx[:, :n1]].mean(axis=1)
    b_means = pooled[perm_idx[:, n1:]].mean(axis=1)
    diffs = a_means - b_means
    count = int(np.sum(np.abs(diffs) >= abs(observed_diff) - 1e-9))
    # +1/+1 (Davison & Hinkley bias correction): the observed data split is
    # ITSELF a valid draw under H0 and belongs in both the numerator and the
    # denominator. Without it, a Monte Carlo run can report p=0.0 exactly,
    # which claims a certainty no finite resample count can actually deliver --
    # exactly the overstatement this project's culture (see module docstring
    # point 3) exists to avoid.
    p_value = (count + 1) / (n_resamples + 1)
    return PermutationTestResult(observed_diff, p_value, n_resamples, "monte_carlo")


def _midranks(values):
    """1-based MIDRANKS of `values`: a group of k tied values that would jointly
    occupy ranks r..r+k-1 all receive the average rank (2r+k-1)/2. Returns
    `(ranks, tie_group_sizes)`; `tie_group_sizes` lists every group's size
    (including singletons, size 1) so `sum(t**3 - t for t in tie_group_sizes)`
    -- the standard tie-correction term -- is 0 wherever there is no tie and the
    caller does not need to filter singletons out first.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    order = np.argsort(values, kind="mergesort")  # stable: ties keep input order
    ranks = np.empty(n, dtype=float)
    tie_group_sizes = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0  # average of the 1-based positions i+1..j+1
        ranks[order[i:j + 1]] = avg_rank
        tie_group_sizes.append(j - i + 1)
        i = j + 1
    return ranks, tie_group_sizes


@lru_cache(maxsize=None)
def _mwu_exact_counts(n1: int, n2: int) -> tuple:
    """Exact rank-permutation distribution of the Mann-Whitney U_a statistic
    (see `mann_whitney_u` for the convention), for TIE-FREE data of group sizes
    n1, n2. Classical Mann & Whitney (1947) recurrence, computed bottom-up:

        f(u; n1, n2) = f(u - n2; n1-1, n2) + f(u; n1, n2-1)
        f(u; i, 0) = f(u; 0, j) = [u == 0]

    (`f(u; n1, n2)` = number of ways to assign n1+n2 distinct ranks to two
    groups of size n1, n2 such that the resulting U_a equals u.) Depends ONLY
    on the group sizes, never the data, so `lru_cache` makes every call after
    the first for a given (n1, n2) free -- the same pair recurs across every
    checkpoint/regime/metric combination a real analysis run asks about.

    Returns a tuple of length n1*n2+1 where index u holds f(u; n1, n2); it sums
    to C(n1+n2, n1), the total number of splits.
    """
    table = [[None] * (n2 + 1) for _ in range(n1 + 1)]
    for i in range(n1 + 1):
        table[i][0] = (1,)
    for j in range(n2 + 1):
        table[0][j] = (1,)
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            prev_i = table[i - 1][j]   # f(*; i-1, j)
            prev_j = table[i][j - 1]   # f(*; i, j-1)
            max_u = i * j
            counts = [0] * (max_u + 1)
            for u in range(max_u + 1):
                total = 0
                uu = u - j
                if 0 <= uu < len(prev_i):
                    total += prev_i[uu]
                if u < len(prev_j):
                    total += prev_j[u]
                counts[u] = total
            table[i][j] = tuple(counts)
    return table[n1][n2]


def mann_whitney_u(a, b, exact_total_threshold: int = MWU_EXACT_TOTAL_THRESHOLD
                    ) -> MannWhitneyResult:
    """Mann-Whitney U test (a rank-based, distribution-free alternative to
    Welch's t-test -- useful precisely because it does NOT assume the per-seed
    metric is normally distributed, which a handful of training seeds gives
    little power to check).

    CONVENTION: `U` returned is U_a = R_a - n_a(n_a+1)/2, where R_a is the sum
    of `a`'s MIDRANKS in the pooled sample -- i.e. the number of pairs (a_i,
    b_j) with a_i > b_j (ties contributing 0.5 each). U_a == 0 therefore means
    every value in `a` was below every value in `b`; U_a == n_a*n_b means the
    opposite. (The other common convention, U_b = n_a*n_b - U_a, is the mirror
    image; this module always reports U_a.)

    Exact null distribution (via `_mwu_exact_counts`) is used when the combined
    sample is tie-free AND its size is at or below `exact_total_threshold`
    (default 30 -- comfortably covers this project's per-arm seed counts): the
    classical recurrence assumes a continuous, tie-free population, so ties
    fall back to the normal approximation below regardless of n. Otherwise,
    the normal approximation with BOTH a tie correction (to the variance) and a
    continuity correction (to the z-score) is used. `.method` reports which.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 1 or n2 < 1:
        raise ValueError("mann_whitney_u needs at least 1 observation per group")

    combined = np.concatenate([a, b])
    ranks, tie_group_sizes = _midranks(combined)
    R_a = float(ranks[:n1].sum())
    U_a = R_a - n1 * (n1 + 1) / 2.0
    n_total = n1 + n2
    has_ties = any(t > 1 for t in tie_group_sizes)

    if n_total <= exact_total_threshold and not has_ties:
        counts = _mwu_exact_counts(n1, n2)
        total = sum(counts)
        u_int = int(round(U_a))  # exact integer here: no ties means no midrank fractions
        obs_dist2 = abs(2 * u_int - n1 * n2)  # x2 to compare against a possibly-odd mean exactly in ints
        extreme = sum(c for u, c in enumerate(counts) if abs(2 * u - n1 * n2) >= obs_dist2)
        return MannWhitneyResult(U=float(u_int), p_value=extreme / total, method="exact")

    mean_u = n1 * n2 / 2.0
    tie_term = sum(t ** 3 - t for t in tie_group_sizes)
    variance_u = (n1 * n2 / 12.0) * ((n_total + 1) - tie_term / (n_total * (n_total - 1)))
    if variance_u <= 0.0:
        # Every value across BOTH groups is identical: zero spread, nothing to
        # test. There is no evidence of any difference and no way to manufacture
        # a p-value out of zero variance, so report the maximally-uncertain 1.0.
        return MannWhitneyResult(U=U_a, p_value=1.0, method="normal_approximation")
    sigma_u = math.sqrt(variance_u)
    z = max(0.0, abs(U_a - mean_u) - 0.5) / sigma_u  # continuity correction
    p_value = min(1.0, max(0.0, 2.0 * (1.0 - _standard_normal_cdf(z))))
    return MannWhitneyResult(U=U_a, p_value=p_value, method="normal_approximation")


def cohens_d(a, b) -> float:
    """Pooled-SD standardised mean difference: (mean(a) - mean(b)) / s_pooled,
    s_pooled = sqrt(((n_a-1)*var(a) + (n_b-1)*var(b)) / (n_a+n_b-2)) (ddof=1
    both sides).

    SIGN CONVENTION: POSITIVE means `a` scored higher than `b` -- the same
    convention `compare_arms` uses (reservoir as `a`, baseline as `b`), so a
    positive d here reads the same direction as a positive `welch_ttest`
    t-statistic and a positive `bootstrap_ci_diff` centre.

    Needs >=2 observations per group for a variance to exist at all; with fewer
    returns NaN (same convention as `welch_ttest`, see its docstring). A pooled
    SD of exactly 0.0 with means that still differ is a genuinely infinite
    effect size (a nonzero gap measured in zero SDs), reported as a signed
    +-inf rather than fabricated as some large finite number -- and it survives
    into `--json` output as `null` via `_json_safe` below, the same non-finite
    handling `training/evaluate.py` uses for NaN.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 1 or n2 < 1:
        raise ValueError("cohens_d needs at least 1 observation per group")
    if n1 < 2 or n2 < 2:
        return float("nan")

    mean1, mean2 = float(a.mean()), float(b.mean())
    var1, var2 = float(a.var(ddof=1)), float(b.var(ddof=1))
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    if pooled_var == 0.0:
        if mean1 == mean2:
            return 0.0
        return math.copysign(math.inf, mean1 - mean2)
    pooled_sd = math.sqrt(pooled_var)
    return (mean1 - mean2) / pooled_sd


def bootstrap_ci_diff(a, b, n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES, seed: int = 0,
                       alpha: float = DEFAULT_BOOTSTRAP_ALPHA) -> BootstrapCIResult:
    """Percentile bootstrap confidence interval on mean(a) - mean(b).

    Makes NO distributional assumption (unlike `welch_ttest`) -- resamples each
    group WITH replacement `n_resamples` times, recomputes the mean difference
    each time, and reports the [alpha/2, 1-alpha/2] percentiles of that
    distribution as the CI. `numpy.random.default_rng(seed)` is used explicitly
    (not the module-global numpy RNG) so the reported interval reproduces
    exactly from the returned `.seed` -- the same "own your randomness, report
    the seed" discipline `training/evaluate.py` applies to episode sampling.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 1 or n2 < 1:
        raise ValueError("bootstrap_ci_diff needs at least 1 observation per group")

    observed_diff = float(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    a_idx = rng.integers(0, n1, size=(n_resamples, n1))
    b_idx = rng.integers(0, n2, size=(n_resamples, n2))
    diffs = a[a_idx].mean(axis=1) - b[b_idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [100 * alpha / 2.0, 100 * (1.0 - alpha / 2.0)])
    return BootstrapCIResult(observed_diff, float(lo), float(hi), n_resamples, seed, alpha)


# ---------------------------------------------------------------------------
# Section 2: loading and aggregation
# ---------------------------------------------------------------------------

# `training/evaluate.py --json` writes one of these per checkpoint, saved as
# `results/eval_{arm}_seed{trainseed}_{regime}.json`. NOTE the seed in the
# FILENAME is the TRAINING seed (which checkpoint this is); the JSON payload's
# own `seed` key is the EVALUATION seed (which episode-action-sampling stream
# was used) -- a different axis entirely. Conflating the two would silently
# mislabel every aggregate below, so `load_eval_results` keeps them as
# separate, explicitly-named fields (`train_seed` vs. the untouched `seed`).
_EVAL_FILENAME_RE = re.compile(
    r"^eval_(?P<arm>[A-Za-z0-9]+)_seed(?P<train_seed>\d+)_(?P<regime>continuous|reset128)\.json$"
)
_CHECKPOINT_DIRNAME_RE = re.compile(r"^(?P<arm>[A-Za-z0-9]+)_seed(?P<seed>\d+)$")

# `training/train.py` names checkpoints `step_{N}.pt`, where N lands on the
# first multiple of rollout_len (128) at or past whatever round threshold
# triggered the save (e.g. `step_100096.pt`, not `step_100000.pt`) -- so N is
# never assumed round anywhere in this module, only ever discovered from the
# filename itself.
_CHECKPOINT_FILENAME_RE = re.compile(r"^step_(?P<step>\d+)\.pt$")


def load_eval_results(results_dir) -> list:
    """Loads every `eval_*.json` file in `results_dir` into a list of dicts,
    each the original JSON payload plus `train_seed` (int, parsed from the
    filename), `regime` (parsed from the filename) and `source_file`.

    Fails LOUDLY (raises `ValueError` with a specific, actionable message) on:
      * a filename that does not match the
        `eval_{arm}_seed{trainseed}_{continuous|reset128}.json` convention --
        refusing to guess an arm/seed/regime from a filename that does not
        parse is better than silently skipping or silently guessing wrong;
      * a JSON payload whose own `arm` key disagrees with the filename's arm --
        exactly the "renamed or copy-pasted the wrong file" failure mode that
        would otherwise silently corrupt an arm-vs-arm comparison;
      * a JSON payload whose `state_reset_interval` disagrees with what the
        filename's regime implies (`continuous` -> None, `reset128` -> 128) --
        the same failure mode as the arm check, one level down: a file renamed
        into the wrong regime bucket would silently mix two different
        recurrent-state regimes into one "comparison".
    A malformed or corrupted (non-JSON) file also raises, via the underlying
    `json.load` call -- there is no silent-skip path anywhere in this loader,
    unlike `summarise_training_logs`'s deliberately more lenient handling of a
    live training log's truncated last line (see that function's docstring for
    why that case is different).
    """
    results_dir = str(results_dir)
    paths = sorted(glob.glob(os.path.join(results_dir, "eval_*.json")))
    records = []
    for path in paths:
        filename = os.path.basename(path)
        match = _EVAL_FILENAME_RE.match(filename)
        if match is None:
            raise ValueError(
                f"load_eval_results: {filename!r} does not match the required "
                f"'eval_{{arm}}_seed{{trainseed}}_{{continuous|reset128}}.json' naming "
                f"convention -- refusing to guess arm/seed/regime from a filename that "
                f"does not parse."
            )
        filename_arm = match.group("arm")
        train_seed = int(match.group("train_seed"))
        regime = match.group("regime")

        with open(path) as f:
            data = json.load(f)

        json_arm = data.get("arm")
        if json_arm != filename_arm:
            raise ValueError(
                f"load_eval_results: {filename!r} claims arm={filename_arm!r} in its "
                f"name, but the JSON payload's own 'arm' key says {json_arm!r}. A "
                f"silently mislabelled result would corrupt the whole comparison -- "
                f"fix whichever one is wrong before trusting this file."
            )

        expected_reset_interval = None if regime == "continuous" else 128
        actual_reset_interval = data.get("state_reset_interval")
        if actual_reset_interval != expected_reset_interval:
            raise ValueError(
                f"load_eval_results: {filename!r} claims regime={regime!r} (implies "
                f"state_reset_interval={expected_reset_interval!r}) but the JSON "
                f"payload recorded state_reset_interval={actual_reset_interval!r} -- "
                f"this file has been mislabelled or misplaced."
            )

        record = dict(data)
        record["train_seed"] = train_seed
        record["regime"] = regime
        record["source_file"] = filename
        records.append(record)
    return records


def aggregate_by_arm(records, regime: str, metric: str = "mean_extrinsic_return") -> dict:
    """Reduces `records` (as returned by `load_eval_results`) to a per-arm
    summary of ONE metric, filtered to `regime`.

    THE UNIT OF ANALYSIS IS THE TRAINING SEED, NOT THE EPISODE (see module
    docstring point 2): each training seed contributes exactly ONE number here
    (its checkpoint's own `metric`, e.g. `mean_extrinsic_return` -- already an
    average over that checkpoint's evaluation episodes). Averaging or pooling
    raw per-episode values across different seeds would treat one checkpoint's
    internal sampling noise as if it were independent evidence about the
    architecture, exactly the mistake `training/evaluate.py`'s own docstring
    warns a real arm comparison must not make.

    Returns `{arm: {..., "train_seeds": [...], "values": [...], "n":, "mean":,
    "std":, "sem":, "min":, "max":}}`, values sorted by ascending train seed.
    `std`/`sem` are NaN for n<2 (nothing to estimate a spread from -- same
    convention as `training/evaluate.py`'s `_summarise`).

    Raises if the same (arm, train_seed) pair appears more than once for this
    regime: that is not more evidence, it is an ambiguous duplicate (which
    checkpoint's re-evaluation should count?) and must be resolved by the
    caller, not silently averaged away.
    """
    by_arm = defaultdict(list)
    for r in records:
        if r.get("regime") != regime:
            continue
        arm = r["arm"]
        seed = r["train_seed"]
        if metric not in r:
            raise ValueError(
                f"aggregate_by_arm: record for arm={arm!r} train_seed={seed} "
                f"regime={regime!r} has no {metric!r} key (source file: "
                f"{r.get('source_file')!r})"
            )
        by_arm[arm].append((seed, r[metric]))

    out = {}
    for arm, pairs in by_arm.items():
        seeds_seen = [s for s, _ in pairs]
        if len(seeds_seen) != len(set(seeds_seen)):
            dupes = sorted({s for s in seeds_seen if seeds_seen.count(s) > 1})
            raise ValueError(
                f"aggregate_by_arm: arm={arm!r} regime={regime!r} has more than one "
                f"eval result for training seed(s) {dupes} -- each training seed must "
                f"contribute exactly one number (see this function's docstring); "
                f"remove or reconcile the duplicate file(s) before aggregating."
            )
        pairs.sort(key=lambda p: p[0])
        seeds = [s for s, _ in pairs]
        values = [float(v) for _, v in pairs]
        n = len(values)
        std = statistics.stdev(values) if n >= 2 else float("nan")
        sem = std / math.sqrt(n) if n >= 2 else float("nan")
        out[arm] = {
            "arm": arm,
            "regime": regime,
            "metric": metric,
            "train_seeds": seeds,
            "values": values,
            "n": n,
            "mean": float(statistics.fmean(values)),
            "std": float(std),
            "sem": float(sem),
            "min": float(min(values)),
            "max": float(max(values)),
        }
    return out


def compare_arms(records, regime: str, metric: str = "mean_extrinsic_return",
                  arm_a: str = "reservoir", arm_b: str = "baseline",
                  permutation_seed: int = 0, bootstrap_seed: int = 0) -> dict:
    """Bundles both arms' `aggregate_by_arm` summaries plus every statistic
    from section 1, computed on their per-seed values.

    CONVENTION: `arm_a` (default "reservoir") is always `a`, `arm_b` (default
    "baseline") is always `b`, in every section-1 call. A POSITIVE
    `welch_ttest.t_statistic`, `cohens_d`, `permutation_test.observed_diff`, or
    `bootstrap_ci_diff` centre therefore all mean the SAME thing: `arm_a`
    (reservoir) scored higher than `arm_b` (baseline) on `metric`. Stated once
    here and in the returned dict's own `"convention"` key rather than left for
    a reader to infer from argument order.

    Raises if either arm has zero eval results for this `regime`/`metric`
    combination -- there is nothing to compare, and continuing with a
    single-arm "comparison" would silently produce a meaningless result rather
    than an obvious error.
    """
    agg = aggregate_by_arm(records, regime, metric)
    missing = [arm for arm in (arm_a, arm_b) if arm not in agg]
    if missing:
        raise ValueError(
            f"compare_arms: no eval results for arm(s) {missing} under "
            f"regime={regime!r}, metric={metric!r} -- cannot compare. Arms found: "
            f"{sorted(agg)}"
        )

    a_values = agg[arm_a]["values"]
    b_values = agg[arm_b]["values"]

    return {
        "regime": regime,
        "metric": metric,
        "arm_a": arm_a,
        "arm_b": arm_b,
        "convention": (
            f"every signed statistic below is (arm_a={arm_a}) minus (arm_b={arm_b}); "
            f"positive means {arm_a} scored higher on {metric!r}"
        ),
        arm_a: agg[arm_a],
        arm_b: agg[arm_b],
        "welch_ttest": welch_ttest(a_values, b_values)._asdict(),
        "permutation_test": exact_permutation_test(
            a_values, b_values, seed=permutation_seed)._asdict(),
        "mann_whitney_u": mann_whitney_u(a_values, b_values)._asdict(),
        "cohens_d": cohens_d(a_values, b_values),
        "bootstrap_ci_diff": bootstrap_ci_diff(
            a_values, b_values, seed=bootstrap_seed)._asdict(),
    }


def _read_train_log(path: str):
    """Reads one `train_log.jsonl`, one JSON object per line.

    A run killed mid-write (e.g. the machine was needed for something else, or
    the process was SIGKILLed) leaves a half-written trailing line. That line
    fails `json.loads` and is SKIPPED here, not fatal -- unlike
    `load_eval_results`'s checks above, an interrupted LIVE training log is an
    expected, routine consequence of stopping a training run, not a sign the
    data was mislabelled or corrupted. The skip count is still returned so a
    caller can see exactly how many lines were dropped, rather than that
    information silently vanishing.
    """
    records = []
    skipped = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1
    return records, skipped


def summarise_training_logs(checkpoint_dir) -> dict:
    """Walks `checkpoint_dir` for `{arm}_seed{N}/train_log.jsonl` files (the
    format `training/train.py` writes, one JSON object per PPO update) and
    returns `{arm: {seed: {...}}}` with, per (arm, seed):

      * `n_updates`: number of successfully-parsed log lines;
      * `n_skipped_lines`: unparseable lines skipped (see `_read_train_log`);
      * `final_step`: the last logged `step` (None if the log was empty);
      * `mean_extrinsic_reward_first_10pct` / `_last_10pct`: mean of
        `mean_extrinsic_reward` over the first/last ceil(10%) of UPDATES (by
        count, not by wall-clock or by step) -- a crude but honest
        learning-progress indicator: a run whose last tenth is no better than
        its first tenth learned nothing worth reporting, whatever its final
        checkpoint's eval score says. This is diagnostic only, about ONE run's
        own trajectory -- it is not a seed-level statistical comparison (that
        is what `compare_arms` is for).
      * `grad_norm_min` / `_median` / `_max` over every parsed update.

    Directories not matching `{arm}_seed{N}` (or matching but missing a
    `train_log.jsonl`) are silently skipped -- `checkpoints/` may legitimately
    contain other things (e.g. only `.pt` files saved before this analysis
    module existed), and that is not this function's concern. Returns `{}` if
    `checkpoint_dir` itself does not exist.
    """
    checkpoint_dir = str(checkpoint_dir)
    out = defaultdict(dict)
    if not os.path.isdir(checkpoint_dir):
        return {}

    for entry in sorted(os.listdir(checkpoint_dir)):
        match = _CHECKPOINT_DIRNAME_RE.match(entry)
        if match is None:
            continue
        arm = match.group("arm")
        seed = int(match.group("seed"))
        log_path = os.path.join(checkpoint_dir, entry, "train_log.jsonl")
        if not os.path.isfile(log_path):
            continue

        records, skipped = _read_train_log(log_path)
        n = len(records)
        if n == 0:
            out[arm][seed] = {
                "n_updates": 0,
                "n_skipped_lines": skipped,
                "final_step": None,
                "mean_extrinsic_reward_first_10pct": float("nan"),
                "mean_extrinsic_reward_last_10pct": float("nan"),
                "grad_norm_min": float("nan"),
                "grad_norm_median": float("nan"),
                "grad_norm_max": float("nan"),
            }
            continue

        k = max(1, round(n * 0.1))
        first, last = records[:k], records[-k:]
        grad_norms = sorted(float(r["grad_norm"]) for r in records)
        out[arm][seed] = {
            "n_updates": n,
            "n_skipped_lines": skipped,
            "final_step": records[-1].get("step"),
            "mean_extrinsic_reward_first_10pct": float(
                statistics.fmean(r["mean_extrinsic_reward"] for r in first)),
            "mean_extrinsic_reward_last_10pct": float(
                statistics.fmean(r["mean_extrinsic_reward"] for r in last)),
            "grad_norm_min": grad_norms[0],
            "grad_norm_median": float(statistics.median(grad_norms)),
            "grad_norm_max": grad_norms[-1],
        }
    return dict(out)


# ---------------------------------------------------------------------------
# Section 2.5: checkpoint selection for evaluation
#
# WHY THIS SECTION EXISTS: one experimental arm (the frozen reservoir) has
# turned out to be OPTIMIZATION-UNSTABLE. Its training reward rises, peaks
# partway through the run, then regresses back toward its starting value,
# while its gradient norms grow by two orders of magnitude over the same run
# (see `train_log.jsonl`'s `grad_norm` column for any reservoir run). If
# evaluation only ever looked at each run's FINAL checkpoint, the reservoir
# arm would be scored at an essentially arbitrary point of its own
# oscillation -- which would understate it for a reason that has nothing to
# do with the actual scientific question this project is asking (does a
# frozen reservoir make a good feature extractor?).
#
# THE FIX, AND THE ONE RULE THAT MUST NEVER BE VIOLATED: pick each run's best
# checkpoint by its TRAINING reward (`mean_extrinsic_reward` in
# `train_log.jsonl`), applied IDENTICALLY to every arm -- never by its
# EVALUATION score. Selecting on the evaluation measure would be testing on
# the training set of the selection procedure itself: whichever checkpoint
# happened to sample a lucky handful of evaluation episodes would be
# preferred FOR having gotten lucky, which biases the reported arm-vs-arm
# comparison upward in a way that has nothing to do with which architecture
# actually generalises better. Selecting on training reward, identically for
# both arms, keeps the comparison honest: neither arm gets to cherry-pick
# post hoc on the very metric being compared.
# ---------------------------------------------------------------------------

def list_checkpoints(run_dir) -> list:
    """Lists every `step_*.pt` checkpoint file directly inside `run_dir`,
    sorted NUMERICALLY ascending by step -- NOT lexicographically, which
    silently gets this wrong: the string 'step_900864.pt' sorts AFTER
    'step_1000064.pt' character-by-character (because '9' > '1'), even though
    900864 < 1000064 as numbers. A full 1,000,000-step run crosses exactly
    this 9xx,xxx -> 1,0xx,xxx digit-count boundary at its LAST checkpoint, so
    this is not a hypothetical edge case -- a lexicographic sort would get
    the final checkpoint of every complete run in this project wrong.

    Returns a list of `(step: int, path: str)` tuples ascending by step;
    `path` is the full path to the file (`os.path.join(run_dir, filename)`),
    so a caller never has to reconstruct it.

    Raises `FileNotFoundError` if `run_dir` does not exist, and `ValueError`
    if it exists but contains no `step_*.pt` files -- both are "nothing to
    select a checkpoint from" states, and `select_best_checkpoint` /
    `select_final_checkpoint` below rely on this raising rather than
    returning an empty list, so a caller cannot accidentally "select" from
    nothing without an obvious, immediate error.
    """
    run_dir = str(run_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"list_checkpoints: no such directory {run_dir!r}")

    checkpoints = []
    for entry in os.listdir(run_dir):
        match = _CHECKPOINT_FILENAME_RE.match(entry)
        if match is None:
            continue
        step = int(match.group("step"))
        checkpoints.append((step, os.path.join(run_dir, entry)))

    if not checkpoints:
        raise ValueError(
            f"list_checkpoints: {run_dir!r} exists but contains no step_*.pt "
            f"checkpoint files"
        )
    checkpoints.sort(key=lambda pair: pair[0])  # numeric, via the int step -- see docstring
    return checkpoints


def select_best_checkpoint(run_dir, window=None) -> dict:
    """Selects the checkpoint whose recent TRAINING reward is highest -- the
    model-selection rule this project applies to EVERY arm alike. See this
    section's header comment for the full "why": in short, evaluating only
    the last checkpoint would score an optimization-unstable run (the frozen
    reservoir arm) at an arbitrary point of its own reward oscillation, and
    selecting on the EVALUATION score instead of the TRAINING score would
    bias the comparison by testing on the training set of the selection
    procedure itself. This function only ever looks at `train_log.jsonl`.

    For each checkpoint at step S (walked in ascending step order), its score
    is the mean of `mean_extrinsic_reward` over every training-log update in
    the window ENDING at S:
      * default (`window=None`): every update with
        `prev_checkpoint_step < step <= S`, where `prev_checkpoint_step` is
        the PRECEDING checkpoint's step (0 for the first checkpoint in the
        run). This partitions every logged update into exactly one
        checkpoint's window -- none double-counted, none dropped -- since the
        windows are the (checkpoint_step, previous_checkpoint_step] intervals
        tiling the run.
      * `window=<int>`: every update with `S - window < step <= S` instead --
        a fixed-width trailing window, independent of checkpoint spacing.

    TIES: if two or more checkpoints tie for the highest score, the EARLIER
    (smaller-step) one wins. A smaller step reaching the same training reward
    is the more conservative choice -- it is evidence the run already reached
    that level of performance and did not need the extra training steps to
    do so -- and it avoids silently preferring a later checkpoint that may
    just be riding noise rather than genuine further improvement.

    A checkpoint whose window contains ZERO training-log updates (possible if
    checkpointing is denser than logging) is scored `-inf` with
    `n_updates_in_window=0`, rather than raising or dividing by zero -- an
    empty window means "no evidence this checkpoint is any good", which
    simply cannot win the selection (any real window outscores -inf); it must
    not crash the whole manifest build over what is a routine consequence of
    checkpoint/log-interval mismatch, not a data-corruption error.

    Returns a dict: `step`, `path` (of the SELECTED checkpoint), `score`,
    `n_updates_in_window`, and `all_scores` -- a list of `(step, score)` for
    EVERY checkpoint in the run, ascending by step, so a caller can audit or
    plot the full selection trace rather than trusting only the winner.
    """
    checkpoints = list_checkpoints(run_dir)  # raises if run_dir has none

    log_path = os.path.join(str(run_dir), "train_log.jsonl")
    updates = _read_train_log(log_path)[0] if os.path.isfile(log_path) else []

    all_scores = []
    n_in_window_by_step = {}
    prev_checkpoint_step = 0
    for step, _path in checkpoints:
        lo = prev_checkpoint_step if window is None else step - window
        window_rewards = [u["mean_extrinsic_reward"] for u in updates if lo < u["step"] <= step]
        if window_rewards:
            score = float(statistics.fmean(window_rewards))
            n_in_window = len(window_rewards)
        else:
            score = float("-inf")
            n_in_window = 0
        all_scores.append((step, score))
        n_in_window_by_step[step] = n_in_window
        prev_checkpoint_step = step

    # Ascending-step walk, updating only on a STRICT improvement: this is what
    # makes an equal-scoring later checkpoint lose to the already-recorded
    # earlier one (see TIES above), and it correctly leaves the very first
    # checkpoint selected in the degenerate case where every window is empty
    # (every score is -inf, so nothing ever beats the initial choice).
    best_step, best_score = all_scores[0]
    for step, score in all_scores[1:]:
        if score > best_score:
            best_step, best_score = step, score

    paths_by_step = dict(checkpoints)
    return {
        "step": best_step,
        "path": paths_by_step[best_step],
        "score": best_score,
        "n_updates_in_window": n_in_window_by_step[best_step],
        "all_scores": all_scores,
    }


def select_final_checkpoint(run_dir) -> dict:
    """Returns the run's LAST checkpoint (the highest step) -- the naive
    selection rule `select_best_checkpoint` exists as a deliberate alternative
    to (see that function's docstring, and this section's header comment, for
    why "just take the last one" silently mis-scores the optimization-unstable
    reservoir arm).

    Returns `{"step": ..., "path": ...}` -- the SAME two keys
    `select_best_checkpoint` uses, so a caller (in particular
    `build_eval_manifest`) can apply either selection rule through one code
    path without branching on the return shape.
    """
    checkpoints = list_checkpoints(run_dir)  # raises if none; numerically sorted ascending
    step, path = checkpoints[-1]
    return {"step": step, "path": path}


def build_eval_manifest(checkpoint_dir, arms=("baseline", "reservoir"),
                         seeds=range(10), selection: str = "final") -> dict:
    """Builds the checkpoint-selection manifest for the next evaluation pass:
    for every (arm, seed) in `arms` x `seeds`, applies `selection` ("final" ->
    `select_final_checkpoint`, "best" -> `select_best_checkpoint`) to
    `checkpoint_dir/{arm}_seed{seed}/` and records which checkpoint that run
    should be evaluated at.

    Designed to be called MID-EXPERIMENT, while some runs are still training
    or have not started at all: a run directory that does not exist, or
    exists but has no `step_*.pt` files yet (`list_checkpoints` raising
    `FileNotFoundError` or `ValueError` respectively, via
    `select_final_checkpoint`/`select_best_checkpoint`), is SKIPPED rather
    than fatal -- its (arm, seed) is instead recorded in the returned
    `missing` list, so a caller sees exactly which runs are not ready yet
    without the whole manifest build failing because of them.

    Returns `{"manifest": [...], "missing": [...]}`:
      * `manifest`: one dict per run directory that DID yield a selection --
        `{"arm", "seed", "step", "path", "selection"}` (the last echoing back
        which rule -- "final" or "best" -- was used, so a manifest saved to
        disk is self-describing about how it was produced).
      * `missing`: one dict per (arm, seed) that was skipped --
        `{"arm", "seed", "reason"}`.

    Raises `ValueError` up front if `selection` is not `"final"` or `"best"`
    -- refusing to guess what an unrecognised selection rule should do is
    better than silently defaulting to one of them.
    """
    if selection not in ("final", "best"):
        raise ValueError(
            f"build_eval_manifest: selection must be 'final' or 'best', got {selection!r}"
        )
    checkpoint_dir = str(checkpoint_dir)

    manifest = []
    missing = []
    for arm in arms:
        for seed in seeds:
            run_dir = os.path.join(checkpoint_dir, f"{arm}_seed{seed}")
            if not os.path.isdir(run_dir):
                missing.append({
                    "arm": arm, "seed": seed,
                    "reason": f"run directory {run_dir!r} does not exist",
                })
                continue
            try:
                picked = (select_final_checkpoint(run_dir) if selection == "final"
                          else select_best_checkpoint(run_dir))
            except ValueError:
                # list_checkpoints' "exists but has no step_*.pt files yet"
                # case -- e.g. a run whose train_log.jsonl has started but
                # hasn't hit its first checkpoint interval. Routine mid-run
                # state, not an error.
                missing.append({
                    "arm": arm, "seed": seed,
                    "reason": f"run directory {run_dir!r} has no checkpoints yet",
                })
                continue
            manifest.append({
                "arm": arm,
                "seed": seed,
                "step": picked["step"],
                "path": picked["path"],
                "selection": selection,
            })
    return {"manifest": manifest, "missing": missing}


# ---------------------------------------------------------------------------
# Section 3: CLI
# ---------------------------------------------------------------------------

def _json_safe(obj):
    """Recursively replaces non-finite floats (NaN, +inf, -inf) with `None` so
    `json.dumps` produces valid JSON -- `json.dumps` happily emits the literal
    tokens `NaN`/`Infinity`, which is NOT valid JSON per spec and which
    `JSON.parse` (and plenty of other consumers) reject outright. This mirrors
    `training/evaluate.py`'s own `--json` handling in its `__main__` block
    exactly, generalised here to walk nested dicts/lists/tuples since this
    module's output is nested rather than flat.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _fmt(x, prec: int = 4) -> str:
    """Formats a number for the human-readable report. NaN/inf print as the
    literal words 'nan'/'inf'/'-inf' rather than through the numeric format
    spec (which renders them as e.g. 'nan' anyway, but inconsistently across
    precisions) -- explicit is clearer than relying on that. Anything that
    isn't a float (e.g. a list of seeds) passes through via plain `str()`."""
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return f"{x:.{prec}f}"
    return str(x)


def _format_arm_summary(agg: dict) -> str:
    return (f"{agg['arm']:10s} n={agg['n']:<3d} seeds={agg['train_seeds']}  "
            f"mean={_fmt(agg['mean']):>9s}  std={_fmt(agg['std']):>9s}  "
            f"sem={_fmt(agg['sem']):>9s}  min={_fmt(agg['min']):>9s}  "
            f"max={_fmt(agg['max']):>9s}")


def _format_comparison(cmp: dict) -> str:
    """Prints every number section 1 produced, and nothing that looks like a
    verdict -- see module docstring point 3. The reader is handed the means,
    the spreads, the sample sizes and every p-value, in full, and is left to
    judge whether any of it clears the bar for a real claim."""
    lines = [f"--- regime={cmp['regime']}  metric={cmp['metric']} ---"]
    lines.append(f"  {_format_arm_summary(cmp[cmp['arm_a']])}")
    lines.append(f"  {_format_arm_summary(cmp[cmp['arm_b']])}")
    lines.append(f"  convention: {cmp['convention']}")

    w = cmp["welch_ttest"]
    lines.append(f"  Welch t-test:         t={_fmt(w['t_statistic'])}  "
                 f"df={_fmt(w['df'])}  p={_fmt(w['p_value'], 6)}")

    p = cmp["permutation_test"]
    lines.append(f"  Permutation test:     observed_diff={_fmt(p['observed_diff'])}  "
                 f"p={_fmt(p['p_value'], 6)}  (method={p['method']}, "
                 f"n_permutations={p['n_permutations']})")

    m = cmp["mann_whitney_u"]
    lines.append(f"  Mann-Whitney U:       U={_fmt(m['U'])}  p={_fmt(m['p_value'], 6)}  "
                 f"(method={m['method']})")

    lines.append(f"  Cohen's d (pooled SD): {_fmt(cmp['cohens_d'])}")

    b = cmp["bootstrap_ci_diff"]
    lines.append(f"  Bootstrap {100 * (1 - b['alpha']):.0f}% CI on diff of means: "
                 f"[{_fmt(b['ci_low'])}, {_fmt(b['ci_high'])}]  "
                 f"(observed_diff={_fmt(b['observed_diff'])}, "
                 f"n_resamples={b['n_resamples']}, seed={b['seed']})")

    lines.append("  NOTE: n above counts independently-trained CHECKPOINTS (training "
                 "seeds), never episodes --")
    lines.append("        pooling raw episodes across seeds would treat one checkpoint's "
                 "sampling noise as")
    lines.append("        independent evidence about the architecture. See "
                 "aggregate_by_arm's docstring.")
    return "\n".join(lines)


def _format_training_logs(summary: dict) -> str:
    if not summary:
        return "(no checkpoint train logs found)"
    lines = ["--- training log summary ---"]
    for arm in sorted(summary):
        for seed in sorted(summary[arm]):
            s = summary[arm][seed]
            lines.append(
                f"  {arm}_seed{seed}: updates={s['n_updates']} "
                f"(skipped {s['n_skipped_lines']} unparseable line(s))  "
                f"final_step={s['final_step']}"
            )
            lines.append(
                f"      mean_extrinsic_reward  first 10%={_fmt(s['mean_extrinsic_reward_first_10pct'])}  "
                f"last 10%={_fmt(s['mean_extrinsic_reward_last_10pct'])}"
            )
            lines.append(
                f"      grad_norm  min={_fmt(s['grad_norm_min'])}  "
                f"median={_fmt(s['grad_norm_median'])}  max={_fmt(s['grad_norm_max'])}"
            )
    return "\n".join(lines)


def build_report(results_dir: str, checkpoint_dir: str) -> dict:
    """Loads everything and builds the full analysis bundle: a `compare_arms`
    result (or a recorded skip reason) per known regime, plus the training-log
    summary. Missing data for a regime (e.g. only one arm has been evaluated so
    far, or that regime hasn't been run at all yet) is a normal, expected state
    early in a research project -- recorded as `{"error": "..."}` for that
    regime rather than raised, so a partial run still produces a usable report.
    """
    records = load_eval_results(results_dir)
    bundle = {}
    for regime in ("continuous", "reset128"):
        try:
            bundle[regime] = compare_arms(records, regime)
        except ValueError as exc:
            bundle[regime] = {"error": str(exc)}
    bundle["training_logs"] = summarise_training_logs(checkpoint_dir)
    return bundle


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-checkpoint evaluation results into the "
                    "arm-vs-arm statistical comparison design doc §5 requires."
    )
    parser.add_argument("--results-dir", default="results",
                        help="directory of eval_{arm}_seed{N}_{regime}.json files "
                             "(default: results)")
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="directory of {arm}_seed{N}/train_log.jsonl files "
                             "(default: checkpoints)")
    parser.add_argument("--json", action="store_true",
                        help="print the raw results dict as JSON instead of a "
                             "human-readable summary")
    parser.add_argument("--manifest", action="store_true",
                        help="print a checkpoint-selection manifest (JSON) built by "
                             "build_eval_manifest, instead of the aggregation report "
                             "-- see --selection")
    parser.add_argument("--selection", choices=("final", "best"), default="final",
                        help="checkpoint selection rule for --manifest: 'final' (highest "
                             "step) or 'best' (highest training reward -- see "
                             "select_best_checkpoint's docstring for why this exists) "
                             "(default: final)")
    args = parser.parse_args()

    if args.manifest:
        result = build_eval_manifest(args.checkpoint_dir, selection=args.selection)
        print(json.dumps(_json_safe(result), indent=2))
        return

    bundle = build_report(args.results_dir, args.checkpoint_dir)

    if args.json:
        print(json.dumps(_json_safe(bundle), indent=2))
        return

    for regime in ("continuous", "reset128"):
        r = bundle[regime]
        if "error" in r:
            print(f"--- regime={regime}: skipped ({r['error']}) ---")
        else:
            print(_format_comparison(r))
        print()
    print(_format_training_logs(bundle["training_logs"]))
    # Deliberately no "arm X wins" line here or anywhere above -- see module
    # docstring point 3.


if __name__ == "__main__":
    main()
