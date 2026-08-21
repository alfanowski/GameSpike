"""Phase 2a SPEC-A/SPEC-B analysis: the pre-registered go/no-go, and the
performance matrix's first row (docs/DESIGN_ROADMAP_PHASE2.md §15).

Reuses this project's own statistics rather than re-deriving them
(`analysis/aggregate_results.py`: exact permutation test, Cohen's d, bootstrap CI
on the difference of means), so Phase 2a is measured with the instrument Phase 1
was measured with -- `EXPERIMENT_LOG.md` §17.11's rule.

UNIT OF ANALYSIS IS THE TRAINING SEED, never the episode (`RESULTS.md` §2.4):
each of the 160 evaluation files is already one checkpoint's mean over 30
episodes, and this module reduces each to exactly one number per training seed
before any statistic sees it.

The pre-registered gate, quoted from §15.3 so it cannot drift:

    GO for task j requires BOTH:
      1. the specialist beats its own untrained anchor on mean_extrinsic_return,
         exact two-sided permutation test, p < 0.05, n = 10 vs n = 10; AND
      2. Cohen's d >= 1.0 between the specialist and init seed-level means.
    Phase 2a's full matrix proceeds only if BOTH tasks pass.
"""
import argparse
import glob
import json
import os
import re

from analysis.aggregate_results import (bootstrap_ci_diff, cohens_d,
                                        exact_permutation_test)

# eval_{arm}_ckpt{W-L}_on{W-L}_seed{N}_{regime}.json
_NAME = re.compile(
    r"^eval_(?P<arm>[A-Za-z0-9]+)_ckpt(?P<ckpt_task>\d+-\d+)_on(?P<eval_task>\d+-\d+)"
    r"_seed(?P<seed>\d+)_(?P<regime>continuous|reset128)\.json$")

P_THRESHOLD = 0.05      # §15.3 gate 1
D_THRESHOLD = 1.0       # §15.3 gate 2


def load(results_dir):
    """Every evaluation cell, keyed by its five coordinates."""
    out = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "*.json"))):
        m = _NAME.match(os.path.basename(path))
        if not m:
            raise ValueError(f"unparseable result filename: {path!r}")
        with open(path) as fh:
            payload = json.load(fh)
        out.append({
            "selection": os.path.basename(os.path.dirname(path)),
            "arm": m["arm"], "ckpt_task": m["ckpt_task"], "eval_task": m["eval_task"],
            "seed": int(m["seed"]), "regime": m["regime"],
            "mean_extrinsic_return": payload["mean_extrinsic_return"],
            "mean_episode_length": payload["mean_episode_length"],
        })
    return out


def series(records, *, selection, ckpt_task, eval_task, regime):
    """One number per training seed, seed-ordered. The §2.4 reduction."""
    rows = [r for r in records
            if r["selection"] == selection and r["ckpt_task"] == ckpt_task
            and r["eval_task"] == eval_task and r["regime"] == regime]
    by_seed = {}
    for r in rows:
        if r["seed"] in by_seed:
            raise ValueError(f"duplicate seed {r['seed']} for {selection}/{ckpt_task}/"
                             f"{eval_task}/{regime}")
        by_seed[r["seed"]] = r["mean_extrinsic_return"]
    return [by_seed[s] for s in sorted(by_seed)]


def gate(records, task, regime):
    """§15.3's two-part gate for one task, in one regime."""
    spec = series(records, selection="final", ckpt_task=task, eval_task=task, regime=regime)
    init = series(records, selection="init", ckpt_task=task, eval_task=task, regime=regime)
    perm = exact_permutation_test(spec, init)
    d = cohens_d(spec, init)
    ci = bootstrap_ci_diff(spec, init)
    mean = lambda v: sum(v) / len(v)
    return {
        "task": task, "regime": regime, "n_spec": len(spec), "n_init": len(init),
        "spec_mean": mean(spec), "init_mean": mean(init),
        "denominator": mean(spec) - mean(init),
        "p": perm.p_value, "method": perm.method, "d": d,
        "ci_low": ci.ci_low, "ci_high": ci.ci_high,
        "pass_p": perm.p_value < P_THRESHOLD,
        "pass_d": abs(d) >= D_THRESHOLD and d > 0,
    }


def normalized(records, *, ckpt_task, eval_task, regime, anchors):
    """§8.1's specialist-normalized score, seed-level.

    `anchors[eval_task]` supplies (R_init, R_spec) for that task -- frozen means
    over ten seeds, per §8.1, so the denominator cannot drift with what is being
    measured.
    """
    r_init, r_spec = anchors[eval_task]
    denom = r_spec - r_init
    vals = series(records, selection="final", ckpt_task=ckpt_task,
                  eval_task=eval_task, regime=regime)
    return [(v - r_init) / denom for v in vals]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_p2a")
    ap.add_argument("--tasks", default="1-1,2-1")
    args = ap.parse_args(argv)
    tasks = args.tasks.split(",")
    records = load(args.results_dir)
    mean = lambda v: sum(v) / len(v)

    print(f"loaded {len(records)} evaluation cells from {args.results_dir}\n")

    print("=" * 78)
    print("PRE-REGISTERED GO/NO-GO (docs/DESIGN_ROADMAP_PHASE2.md §15.3)")
    print("  GO requires p < 0.05 AND Cohen's d >= 1.0, specialist vs its own init")
    print("=" * 78)
    verdicts = {}
    for regime in ("continuous", "reset128"):
        print(f"\n-- regime: {regime} --")
        for task in tasks:
            g = gate(records, task, regime)
            ok = g["pass_p"] and g["pass_d"]
            verdicts[(task, regime)] = ok
            print(f"  task {task}:  spec {g['spec_mean']:8.3f}   init {g['init_mean']:8.3f}   "
                  f"denominator {g['denominator']:8.3f}")
            print(f"            p = {g['p']:.6g} ({g['method']})"
                  f"  d = {g['d']:+.3f}   95% CI [{g['ci_low']:.3f}, {g['ci_high']:.3f}]")
            print(f"            gate p<0.05: {'PASS' if g['pass_p'] else 'FAIL'}    "
                  f"gate d>=1.0: {'PASS' if g['pass_d'] else 'FAIL'}    "
                  f"=> {'GO' if ok else 'NO-GO'}")

    headline_regime = "reset128"
    overall = all(verdicts[(t, headline_regime)] for t in tasks)
    print("\n" + "=" * 78)
    print(f"VERDICT (on {headline_regime}, the regime training actually used): "
          f"{'GO -- both tasks pass' if overall else 'NO-GO'}")
    both = all(verdicts[k] for k in verdicts)
    print(f"  (both regimes agree: {both})")
    print("=" * 78)

    # §8.1's anchors, frozen from the ten-seed means, then the performance matrix.
    for regime in ("continuous", "reset128"):
        anchors = {}
        for t in tasks:
            g = gate(records, t, regime)
            anchors[t] = (g["init_mean"], g["spec_mean"])
        print(f"\n-- performance matrix, normalized (§8.1/§8.2), regime {regime} --")
        print(f"  anchors: " + "   ".join(
            f"{t}: init {anchors[t][0]:.3f} spec {anchors[t][1]:.3f}" for t in tasks))
        print(f"  {'trained on':>12} " + "".join(f"{'eval ' + t:>18}" for t in tasks))
        for ct in tasks:
            cells = []
            for et in tasks:
                v = normalized(records, ckpt_task=ct, eval_task=et, regime=regime,
                               anchors=anchors)
                cells.append(f"{mean(v):+.3f}")
            print(f"  {ct:>12} " + "".join(f"{c:>18}" for c in cells))

        print(f"  -- raw mean_extrinsic_return, same cells --")
        for ct in tasks:
            cells = []
            for et in tasks:
                v = series(records, selection="final", ckpt_task=ct, eval_task=et,
                           regime=regime)
                cells.append(f"{mean(v):.3f}")
            print(f"  {ct:>12} " + "".join(f"{c:>18}" for c in cells))

    # §15.4 / §8.3: forward transfer. Under §8.1's normalization the untrained
    # reference is 0.0 by construction, so FWT is just the zero-shot normalized score.
    print("\n" + "=" * 78)
    print("FORWARD TRANSFER (§15.4, descriptive -- gates nothing)")
    print("  zero-shot normalized score of a specialist on the task it never saw")
    print("=" * 78)
    for regime in ("continuous", "reset128"):
        anchors = {}
        for t in tasks:
            g = gate(records, t, regime)
            anchors[t] = (g["init_mean"], g["spec_mean"])
        for ct in tasks:
            for et in tasks:
                if ct == et:
                    continue
                v = normalized(records, ckpt_task=ct, eval_task=et, regime=regime,
                               anchors=anchors)
                raw = series(records, selection="final", ckpt_task=ct, eval_task=et,
                             regime=regime)
                init = series(records, selection="init", ckpt_task=et, eval_task=et,
                              regime=regime)
                perm = exact_permutation_test(raw, init)
                print(f"  [{regime}] trained {ct} -> evaluated {et}: "
                      f"normalized {mean(v):+.3f}  raw {mean(raw):.3f} vs init "
                      f"{mean(init):.3f}  p = {perm.p_value:.6g}  d = {cohens_d(raw, init):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
