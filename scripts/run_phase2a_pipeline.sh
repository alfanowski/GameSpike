#!/usr/bin/env bash
# Phase 2a SPEC-A / SPEC-B pipeline: the two single-task specialist references
# and their untrained anchors, pre-registered in docs/DESIGN_ROADMAP_PHASE2.md §15.
#
# SCOPE, and it is deliberately narrow: this runs SPEC-A, SPEC-B and INIT only.
# It does NOT run INT, SEQ, the Q3 ablation or any mitigation arm -- those are
# §7.1's remaining conditions and are NOT authorised. Do not add them here; write
# a separate pipeline when and if they are.
#
# Lessons this script is shaped by, all learned the hard way in Phase 1:
#   * docs/EXPERIMENT_LOG.md §19.4 -- a completeness guard that pattern-matched
#     directories passed FALSELY. Every guard below ENUMERATES the exact expected
#     paths instead of globbing.
#   * docs/EXPERIMENT_LOG.md §19.3 -- a running bash script must not be edited in
#     place. If this needs changing mid-run, stop it first.
#   * docs/EXPERIMENT_LOG.md §19.1 -- a stage that checks no exit code can log
#     COMPLETE behind a crashed child. Every stage's status is checked.
#   * docs/EXPERIMENT_LOG.md §18 -- the machine lost power mid-matrix once. The
#     launcher's own resume/skip logic is left intact so a re-run continues rather
#     than restarting.
set -u -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

PY="${PY:-/Users/alfanowski/Desktop/Projects/GameSpike/.venv/bin/python}"
ROM="${MARIO_LAND_ROM_PATH:?set MARIO_LAND_ROM_PATH}"
JOBS="${JOBS:-8}"
SEEDS="${SEEDS:-0-9}"
STEPS="${STEPS:-1000000}"
CKPT_DIR="${CKPT_DIR:-checkpoints_p2a}"
INIT_DIR="${INIT_DIR:-checkpoints_p2a_init}"
RESULTS_DIR="${RESULTS_DIR:-results_p2a}"
LOG="${LOG:-$REPO_ROOT/phase2a_pipeline.log}"

# Phase 1 v2's flag set, inherited unchanged -- §15.1 commits to not tuning it.
FLAGS=(--grad-clip-mode per-group --embed-init-mode centered --embed-scale 3.0)
TASKS=(1-1 2-1)
FINAL_CKPT="step_1000064.pt"   # NOT step_1000000.pt -- RESULTS.md §23's last line

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
die() { say "FATAL: $*"; exit 1; }

# Enumerate exact expected paths. No globs: see §19.4 above.
guard_final() {
    local dir="$1" task="$2" want="$3" n=0 missing=()
    for s in $(seq 0 9); do
        local p="$dir/baseline_task${task}_seed${s}/$want"
        if [[ -f "$p" ]]; then n=$((n+1)); else missing+=("$p"); fi
    done
    if [[ $n -ne 10 ]]; then
        say "GUARD FAIL ($dir, task $task): $n/10 present; missing: ${missing[*]}"
        return 1
    fi
    say "GUARD PASS ($dir, task $task): 10/10 $want present"
    return 0
}

say "=== Phase 2a SPEC pipeline start (jobs=$JOBS, seeds=$SEEDS, steps=$STEPS) ==="
say "repo=$REPO_ROOT rom=$ROM"

# --- Stage 1: untrained anchors (§15.1 INIT, control C2). Cheap, and every later
# --- normalized score divides by them, so they come first.
for t in "${TASKS[@]}"; do
    say "--- INIT anchors, task $t ---"
    "$PY" -m scripts.run_training_matrix --arms baseline --seeds "$SEEDS" --rom "$ROM" \
        --steps 0 --checkpoint-dir "$INIT_DIR" --task "$t" "${FLAGS[@]}" --jobs "$JOBS" \
        >>"$LOG" 2>&1 || die "INIT task $t launcher exited non-zero"
    guard_final "$INIT_DIR" "$t" "step_0.pt" || die "INIT task $t incomplete"
done

# --- Stage 2: §15.3.1's pre-registered invariant. The anchors must be
# --- task-independent; if they are not, C2 is not the control §2.2 specifies and
# --- nothing downstream is interpretable. Checked BEFORE spending 20M env steps.
say "--- verifying §15.3.1: INIT anchors are task-independent ---"
"$PY" - "$INIT_DIR" <<'EOF' >>"$LOG" 2>&1 || die "§15.3.1 invariant VIOLATED -- stopping"
import hashlib, sys, torch
init_dir = sys.argv[1]
bad = []
for seed in range(10):
    digs = []
    for task in ("1-1", "2-1"):
        blob = torch.load(f"{init_dir}/baseline_task{task}_seed{seed}/step_0.pt",
                          map_location="cpu", weights_only=True)
        h = hashlib.sha256()
        for k in sorted(blob["model"]):
            h.update(k.encode()); h.update(blob["model"][k].numpy().tobytes())
        digs.append(h.hexdigest())
    if digs[0] != digs[1]:
        bad.append(seed)
    print(f"seed {seed}: {'identical' if digs[0]==digs[1] else 'DIFFERENT'} ({digs[0][:16]})")
if bad:
    print(f"VIOLATED at seeds {bad}"); sys.exit(1)
print("INVARIANT HOLDS across all 10 seeds")
EOF
say "§15.3.1 invariant HOLDS"

# --- Stage 3: the specialists themselves.
for t in "${TASKS[@]}"; do
    say "--- SPEC task $t: 10 seeds x $STEPS steps ---"
    "$PY" -m scripts.run_training_matrix --arms baseline --seeds "$SEEDS" --rom "$ROM" \
        --steps "$STEPS" --checkpoint-every 100000 --checkpoint-dir "$CKPT_DIR" \
        --task "$t" "${FLAGS[@]}" --jobs "$JOBS" >>"$LOG" 2>&1 \
        || die "SPEC task $t launcher exited non-zero"
    guard_final "$CKPT_DIR" "$t" "$FINAL_CKPT" || die "SPEC task $t incomplete"
    say "SPEC task $t COMPLETE"
done

# --- Stage 4: evaluation, including the off-diagonal (§15.4): every specialist is
# --- scored on BOTH tasks, which is the zero-shot transfer number.
say "--- evaluation: checkpoint_task x eval_task, both regimes, final+init ---"
"$PY" -m scripts.run_phase2a_eval --rom "$ROM" --tasks 1-1,2-1 --arms baseline \
    --seeds "$SEEDS" --episodes 30 --eval-seed 0 --jobs "$JOBS" \
    --checkpoint-dir "$CKPT_DIR" --init-checkpoint-dir "$INIT_DIR" \
    --results-dir "$RESULTS_DIR" >>"$LOG" 2>&1 || die "evaluation exited non-zero"

n_eval=$(find "$RESULTS_DIR" -name '*.json' -type f | wc -l | tr -d ' ')
say "evaluation wrote $n_eval result files"
say "=== Phase 2a SPEC pipeline COMPLETE ==="
