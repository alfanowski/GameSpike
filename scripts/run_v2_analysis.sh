#!/bin/bash
# Statistics + health measurements for the v2 matrix, run after the training and
# evaluation stages complete. Read-only with respect to checkpoints and results;
# it only writes report files at the repository root.
#
# WHY THIS IS A SEPARATE SCRIPT FROM run_v2_pipeline.sh, AND WHY IT EXISTS AT ALL.
# docs/EXPERIMENT_LOG.md §14.11's Step 5 gives the aggregation command as
#
#     python -m analysis.aggregate_results --results-dir results_v2 --checkpoint-dir checkpoints_v2
#
# and that command SILENTLY PRODUCES NO COMPARISON. `analysis/aggregate_results.py`
# reads evaluation JSONs from the directory it is given, but the evaluation driver
# writes them one level down, into `{results-dir}/{final,best,init}/` (§7 of the
# ledger describes exactly this layout). Pointed at the parent, the aggregator finds
# zero eval files, prints
#
#     regime=continuous: skipped (... Arms found: [])
#
# for both regimes, and then prints a perfectly healthy-looking training-log summary
# underneath -- so the output looks like a successful run with a couple of skipped
# lines rather than like a failure. Verified against v1: pointed at `results/` it
# compares nothing; pointed at `results/final` it reproduces RESULTS.md v1 §3
# digit-for-digit (28.4169 +/- 7.0593 vs 36.1335 +/- 3.4082, diff -7.7167,
# p=0.000996, d=-1.3922). §14.11 Step 5 is corrected in §19 of the ledger.
#
# The primary result table needs THREE aggregator runs, not one -- `final`, `best`
# and `init` are three separate evaluation passes written into three directories,
# and v1's six-row §3 table is those three crossed with the two recurrent-state
# regimes each run already reports.
#
# NOTE: `--selection` is NOT the flag that picks between them. It only affects
# `--manifest` output (verified in `main`); the report is built from `--results-dir`
# and `--checkpoint-dir` alone. Passing `--selection best` while pointing at the
# wrong directory changes nothing at all, which is another way to get a silently
# empty comparison.
#
# Usage (from the repository root):
#   scripts/run_v2_analysis.sh
set -u

REPO="/Users/alfanowski/Desktop/Projects/GameSpike"
PY="$REPO/.venv/bin/python"
LOG="$REPO/pipeline_v2.log"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

cd "$REPO" || exit 1

# Guard: refuse to aggregate a partial matrix. §17.10 -- aggregating an incomplete
# matrix is a WRONG RESULT rather than a failed run.
#
# EXACT ENUMERATION, not a glob. An audit of this pipeline's sibling script found
# that a `find ... -path "*/{arm}_seed*"` pattern is not anchored: a stray tagged
# run directory (e.g. `reservoir_seed0_clipemb/`, which `--run-tag` exists to
# create and which §14.11's manual-recovery fallback would produce) also matches,
# so ONE stray directory can silently stand in for ONE genuinely missing seed and
# make the guard report a full count. That is precisely the false pass the guard
# exists to prevent. Enumerating the 20 exact paths removes the ambiguity entirely
# rather than patching the pattern. See §19.4.
expected_evals=120
n_eval=0
for sel in final best init; do
  for arm in baseline reservoir; do
    for s in 0 1 2 3 4 5 6 7 8 9; do
      for regime in continuous reset128; do
        [ -f "$REPO/results_v2/$sel/eval_${arm}_seed${s}_${regime}.json" ] && n_eval=$((n_eval+1))
      done
    done
  done
done
if [ "$n_eval" -ne "$expected_evals" ]; then
  log "ANALYSIS ABORT: $n_eval/$expected_evals evaluation results present. Nothing aggregated."
  echo "ABORT: $n_eval/$expected_evals evaluation results. Refusing to aggregate a partial matrix." >&2
  exit 1
fi
log "analysis: guard PASS, $n_eval/$expected_evals evaluations"

# Every final checkpoint must actually LOAD, not merely exist. `save_checkpoint`
# is a bare `torch.save` with no temp-file-then-`os.replace`, so a process killed
# mid-write leaves a present-but-corrupt file that a existence-only guard accepts.
# This machine has already lost power once mid-matrix (§18), which is exactly when
# that would happen.
log "analysis: verifying all 20 final checkpoints load"
if ! "$PY" - <<'PYEOF' >> "$LOG" 2>&1
import sys, torch
bad = []
for arm in ("baseline", "reservoir"):
    for s in range(10):
        p = f"checkpoints_v2/{arm}_seed{s}/step_1000064.pt"
        try:
            c = torch.load(p, map_location="cpu", weights_only=True)
            assert c["step"] == 1000064 and c["arm"] == arm and c["seed"] == s, f"label mismatch in {p}"
            assert c["grad_clip_mode"] == "per-group", f"wrong clip mode in {p}"
            assert c["embed_init_mode"] == "centered" and c["embed_scale"] == 3.0, f"wrong embed config in {p}"
        except Exception as exc:
            bad.append(f"{p}: {type(exc).__name__}: {exc}")
if bad:
    print("CHECKPOINT VERIFICATION FAILED:")
    for b in bad:
        print("  " + b)
    sys.exit(1)
print("all 20 final checkpoints load and self-identify correctly")
PYEOF
then
  log "ANALYSIS ABORT: final-checkpoint verification failed. Nothing aggregated."
  echo "ABORT: final-checkpoint verification failed, see $LOG" >&2
  exit 1
fi

for sel in final best init; do
  log "analysis: aggregating results_v2/$sel"
  if ! "$PY" -m analysis.aggregate_results \
    --results-dir "results_v2/$sel" --checkpoint-dir checkpoints_v2 \
    > "$REPO/results_v2_report_$sel.txt" 2>> "$LOG"; then
    log "ANALYSIS ABORT: aggregation failed for selection '$sel'"
    echo "ABORT: aggregation failed for '$sel', see $LOG" >&2
    exit 1
  fi
  if ! "$PY" -m analysis.aggregate_results \
    --results-dir "results_v2/$sel" --checkpoint-dir checkpoints_v2 --json \
    > "$REPO/results_v2_report_$sel.json" 2>> "$LOG"; then
    log "ANALYSIS ABORT: JSON aggregation failed for selection '$sel'"
    echo "ABORT: JSON aggregation failed for '$sel', see $LOG" >&2
    exit 1
  fi
done
log "analysis: three aggregation reports written"

# A7 (dead-gradient in_proj columns, pre-registered §14.5) and A9 (operating-point
# trajectory, pre-registered §15.6). Read-only; computes each verdict against its
# pre-registered band in code, so the verdict is not a judgement call made while
# looking at the number (§17.11).
log "analysis: A7/A9 reservoir health"
if ! "$PY" -m analysis.reservoir_health \
  --checkpoint-dir checkpoints_v2 --arm reservoir --seeds 0-9 \
  > "$REPO/results_v2_health.txt" 2>> "$LOG"; then
  log "ANALYSIS ABORT: reservoir_health failed"
  echo "ABORT: reservoir_health failed, see $LOG" >&2
  exit 1
fi
log "analysis: health written"

log "=== v2 analysis COMPLETE ==="
echo "done: results_v2_report_{final,best,init}.{txt,json}, results_v2_health.txt"
