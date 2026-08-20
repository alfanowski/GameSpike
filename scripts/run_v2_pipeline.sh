#!/bin/bash
# End-to-end driver for the v2 (corrected) training matrix.
#
# WHY THIS FILE IS IN THE REPOSITORY AND NOT IN /tmp. Its predecessor was
# /tmp/gs_pipeline.sh (docs/EXPERIMENT_LOG.md §17.12). At 23:38:53 on 2026-08-20
# the machine lost power, /tmp was cleared on reboot, and the recovery script was
# destroyed by exactly the incident class it existed to recover from (§18.1). A
# committed script survives a reboot, is legible in git, and is reviewable.
#
# WHAT IT DOES:
#   reservoir arm (10 runs)
#     -> [guard: 10 final checkpoints?] -> baseline arm (10 runs)
#       -> [guard: 20 final checkpoints?] -> evaluation matrix (120 evals)
#         -> [guard: 120 result JSONs?] -> aggregation -> A7/A9 health
#
# EVERY ARROW IS GUARDED AND EVERY GUARD REFUSES RATHER THAN DEGRADING.
# §17.10's argument, cashed once already at §17.12: evaluating a partial matrix
# produces a comparison with fewer seeds than it claims, which is a WRONG RESULT
# rather than a failed run. The guards never continue on a short count.
#
# THREE HAZARDS THIS FILE IS WRITTEN AGAINST, all already paid for once:
#   1. Guard counts use `find`, never a bare glob. zsh aborts a command on an
#      unmatched glob BEFORE redirection applies, so `ls pattern 2>/dev/null | wc -l`
#      silently yields 0 (§17.12 lesson 3). `find` takes a literal path and a
#      -path pattern it expands itself, so there is no shell glob to fail.
#   2. Never `pgrep -f`/`grep` on a pattern that appears in the watcher's own
#      command line -- it matches itself and loops forever (§17.4). This script
#      runs its stages in the foreground and needs no PID watcher at all.
#   3. Launch it `nohup`-ed and disowned from a short-lived wrapper so an agent
#      harness reaping its task cannot kill it (§17.12 lesson 1). `setsid` does
#      not exist on macOS (lesson 2).
#
# The training stages execute with cwd inside the PINNED WORKTREE (§17.1) so
# every import resolves from commit dc966a3 while --checkpoint-dir points at the
# main repository. The worktree is disposable; the data is not.
#
# Usage (from the repository root):
#   nohup caffeinate -is scripts/run_v2_pipeline.sh >> pipeline_v2.log 2>&1 &
#   disown
set -u

REPO="/Users/alfanowski/Desktop/Projects/GameSpike"
WORKTREE="$REPO/.claude/worktrees/v2train"
PY="$REPO/.venv/bin/python"
ROM="/Users/alfanowski/Desktop/Super Mario Land (World).gb"
LOG="$REPO/pipeline_v2.log"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

V2FLAGS=(--grad-clip-mode per-group --embed-init-mode centered --embed-scale 3.0)
# --embed-scale 3.0 is NOT optional: --embed-init-mode centered without it is
# WORSE than doing nothing (65.9% silent units vs 45.6% for the legacy default).
# See §14.8 -- this is the exact omission that would reproduce a disconfirmed
# configuration.

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

# Count final checkpoints for one arm. Literal path + -path pattern, no shell glob.
count_final() {
  find "$REPO/checkpoints_v2" -type f -name 'step_1000064.pt' -path "*/$1_seed*" 2>/dev/null | wc -l | tr -d ' '
}

# Count evaluation result JSONs across all selection directories.
count_evals() {
  find "$REPO/results_v2" -type f -name 'eval_*_seed*_*.json' 2>/dev/null | wc -l | tr -d ' '
}

log "=== v2 pipeline start (pid $$) ==="

# ---------------------------------------------------------------- stage 1: reservoir
if [ "$(count_final reservoir)" -eq 10 ]; then
  log "stage 1 SKIP: reservoir arm already complete (10/10)"
else
  log "stage 1: launching reservoir arm, 10 seeds, jobs=10"
  cd "$WORKTREE" || { log "FATAL: worktree $WORKTREE missing"; exit 1; }
  "$PY" scripts/run_training_matrix.py \
    --arms reservoir --seeds 0-9 --rom "$ROM" \
    --steps 1000000 --checkpoint-every 100000 \
    --checkpoint-dir "$REPO/checkpoints_v2" \
    "${V2FLAGS[@]}" --jobs 10 >> "$REPO/checkpoints_v2_reservoir_launch.log" 2>&1
  log "stage 1: launcher exited rc=$?"
fi

n=$(count_final reservoir)
if [ "$n" -ne 10 ]; then
  log "ABORT at guard 1: reservoir $n/10 final checkpoints. Baseline arm NOT launched."
  exit 1
fi
log "guard 1 PASS: reservoir 10/10"

# ---------------------------------------------------------------- stage 2: baseline
if [ "$(count_final baseline)" -eq 10 ]; then
  log "stage 2 SKIP: baseline arm already complete (10/10)"
else
  log "stage 2: launching baseline arm, 10 seeds, jobs=10"
  cd "$WORKTREE" || { log "FATAL: worktree $WORKTREE missing"; exit 1; }
  "$PY" scripts/run_training_matrix.py \
    --arms baseline --seeds 0-9 --rom "$ROM" \
    --steps 1000000 --checkpoint-every 100000 \
    --checkpoint-dir "$REPO/checkpoints_v2" \
    "${V2FLAGS[@]}" --jobs 10 >> "$REPO/checkpoints_v2_baseline_launch.log" 2>&1
  log "stage 2: launcher exited rc=$?"
fi

nb=$(count_final baseline)
if [ "$nb" -ne 10 ]; then
  log "ABORT at guard 2: baseline $nb/10 final checkpoints. Evaluation NOT run."
  exit 1
fi
log "guard 2 PASS: baseline 10/10, matrix complete (20/20)"

# ---------------------------------------------------------------- stage 3: evaluation
# Run from the main repository, not the pinned worktree: v1's evaluation ran from
# its own worktree pinned at 64839a9 (RESULTS.md §2.7), and the eval driver on the
# current branch is that same code plus committed fixes. Resumable, atomic writes.
log "stage 3: evaluation matrix (120 evals, jobs=8)"
cd "$REPO" || exit 1
"$PY" scripts/run_eval_matrix.py --rom "$ROM" \
  --episodes 30 --eval-seed 0 --jobs 8 \
  --checkpoint-dir checkpoints_v2 --init-checkpoint-dir checkpoints_v2_init \
  --results-dir results_v2 >> "$REPO/results_v2_eval.log" 2>&1
log "stage 3: eval driver exited rc=$?"

ne=$(count_evals)
if [ "$ne" -ne 120 ]; then
  log "ABORT at guard 3: $ne/120 evaluation results. Aggregation NOT run."
  exit 1
fi
log "guard 3 PASS: 120/120 evaluations"

# ---------------------------------------------------------------- stage 4: statistics
log "stage 4: aggregation"
cd "$REPO" || exit 1
"$PY" -m analysis.aggregate_results \
  --results-dir results_v2 --checkpoint-dir checkpoints_v2 \
  > "$REPO/results_v2_report.txt" 2>> "$LOG"
"$PY" -m analysis.aggregate_results \
  --results-dir results_v2 --checkpoint-dir checkpoints_v2 --json \
  > "$REPO/results_v2_report.json" 2>> "$LOG"
"$PY" -m analysis.aggregate_results \
  --results-dir results_v2 --checkpoint-dir checkpoints_v2 --selection best \
  > "$REPO/results_v2_report_best.txt" 2>> "$LOG"
log "stage 4: aggregation written"

# ---------------------------------------------------------------- stage 5: A7 / A9
# Pre-registered at §14.5 (A7) and §15.6 (A9). Read-only, writes nothing itself.
log "stage 5: A7/A9 reservoir health"
"$PY" -m analysis.reservoir_health \
  --checkpoint-dir checkpoints_v2 --arm reservoir --seeds 0-9 \
  > "$REPO/results_v2_health.txt" 2>> "$LOG"
log "stage 5: health written"

log "=== v2 pipeline COMPLETE ==="
