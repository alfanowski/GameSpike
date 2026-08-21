"""Resumable, parallel launcher for the Phase 2 training matrix.

Launches `training/train.py` once per (arm, seed) cell of the matrix --
2 arms x 10 seeds x 1,000,064 env steps by default, ~3.5 hours total on a
10-core M4 -- and is the first COMMITTED launcher for this project. The v1
runs were launched ad hoc from a shell one-liner, not from a script this repo
tracks, and that ad-hoc approach is what produced the damage recorded in
`docs/EXPERIMENT_LOG.md` §9 ("Operational hazards already hit"). This module
exists to make each of those hazards structurally impossible to repeat, not
merely to remember not to repeat them:

  * **Written in Python, not shell.** macOS ships bash 3.2, which has no
    `wait -n`: a job-queue semaphore built against it cannot block on a job
    slot becoming free, so it busy-spins polling instead, which both wastes
    CPU that PPO training needs and spams its own log at roughly 1MB/min.
    `concurrent.futures.ThreadPoolExecutor` (below) has no such problem --
    every hazard in this module's history is a bash-3.2-specific one, so the
    fix is "don't write this in bash," not "write more careful bash."

  * **Two independent guards against two launchers racing each other.**
    §9 records that two queue-runner instances were once live simultaneously.
    An atomic single-instance LOCK (`acquire_lock`, `os.mkdir` on a lock
    directory -- atomic on any local filesystem, unlike "check a file, then
    create it") stops a second `run_training_matrix.py` invocation from ever
    starting. But the lock alone is not sufficient on its own: it protects
    against two *launcher processes*, not against one launcher being asked
    to redo work a previous, unlocked invocation already produced (e.g. a
    resumed run after a crash, or an operator re-running the same command
    by hand outside this lock's lifetime). So every job ALSO carries its own
    resume guard (`resume_status`) that inspects the run directory directly,
    independent of whatever process is asking. Two independent mechanisms,
    because they defend against two independent failure modes: concurrent
    processes (lock) and repeated invocations (per-job resume check).

  * **Never `git checkout`/`git switch` while training is live** is a hazard
    about the CALLER's behaviour, not this script's -- this module does not
    touch git at all, and this docstring exists partly so a future reader
    editing this file (or a future orchestrator invoking it) knows why that
    matters: `training/train.py` and everything it imports load from this
    checkout, so switching branches under 10 live subprocesses is a footgun
    for whatever they next read off disk. Use a worktree for any repo work
    done while this launcher's jobs are running.

RESUME GUARD, precisely -- and the trap that motivates it (see
`docs/EXPERIMENT_LOG.md` §2 for the full derivation): `training/train.py`'s
step counter advances in `--rollout-len` increments (default 128), not 1, so
the FINAL checkpoint of a `--steps 1000000` run is `step_1000064.pt`
(ceil(1,000,000 / 128) * 128), never `step_1000000.pt`. A resume guard that
globbed for the round number would find nothing for every completed run and
silently re-launch all of them, burning another 3.5 hours and, far worse,
launching a SECOND `train.py` process against the SAME run directory as
nothing stops it from writing into (see "skip-if-in-progress" below). This
module never hardcodes 1,000,064 anywhere: `final_step_for` derives it from
`(total_steps, rollout_len)` the same way `training/train.py`'s own loop
(`while step < total_steps: ...; step += rollout_len`) does, and has its own
direct test pinning exactly this case.

SKIP-IF-IN-PROGRESS, the other half of the resume guard: a run directory
that EXISTS but does not contain its final checkpoint is not evidence that
nothing is running there -- it is exactly what a crashed run OR a still-live
run both look like from the filesystem, and this launcher cannot tell those
two apart from outside. The safe default is therefore to REFUSE to touch it
and report it (`resume_status` -> `"incomplete"`, `run_job` returns a
`"failed"` outcome naming `--restart-incomplete`), not to guess. Silently
resuming would risk two processes writing into the same `train_log.jsonl`
and racing each other's `torch.save` (which is NOT atomic in
`training/train.py`'s `save_checkpoint` -- a direct `torch.save(..., path)`,
no temp-file-then-`os.replace` the way this project's other write paths use)
-- precisely the corruption §9 describes. `--restart-incomplete` is the
explicit, opt-in escape hatch: it deletes the incomplete run directory and
re-launches that job from scratch, and is opt-in specifically so it is never
the ACCIDENTAL default while a run might still be alive.

CONCURRENCY: `concurrent.futures.ThreadPoolExecutor`, exactly the reasoning
`scripts/run_eval_matrix.py`'s own module docstring gives for its identical
choice -- the actual work (PyBoy stepping, PyTorch forward/backward passes)
happens in the `train.py` child process, so the parent thread spends its
time blocked in `Popen.communicate()`/`subprocess.run`, during which the GIL
is released; threads, not processes, are the right tool because there is no
CPU-bound work in THIS process to parallelise. Each child gets
`OMP_NUM_THREADS=1`/`MKL_NUM_THREADS=1` so N parallel `train.py` processes
don't each ALSO fan out their own BLAS/OpenMP thread pool -- without this,
`--jobs 10` would silently oversubscribe far past 10 cores.

LOGGING TO DISK, NOT MEMORY: unlike `run_eval_matrix.py` (whose child
processes run for seconds and can afford `capture_output=True`, buffering
stdout/stderr in memory until the process exits), a `train.py` job runs for
hours. Capturing its output in memory and only writing it out after
`subprocess.run` returns would mean a job killed partway through -- by an
operator, a crash, a `--restart-incomplete` re-run, the machine sleeping --
loses its ENTIRE diagnostic history, right when it is needed most. So
`_run_subprocess` instead opens `{run_dir}/launcher.log` once, in append
mode, and hands that open file directly to `subprocess.run` as BOTH `stdout`
and `stderr` -- the OS writes into it as the child produces output, with no
buffering in this process at all, so whatever the child managed to print
before being killed is already durably on disk. Appended (not truncated) so
a `--restart-incomplete` re-run's log sits after the failed attempt's, and
the file's own history is never destroyed by this launcher.

RESULT VALIDATION, not just exit code: a `train.py` subprocess returning 0
is necessary but not sufficient evidence the run actually completed --
`run_job` also checks that `job.final_checkpoint_path` now exists on disk
before reporting `"ran"`, the same "don't trust the wrapper, check the
artifact" discipline `run_eval_matrix.validate_result` applies to its own
child's JSON payload.

Every flag `training/train.py`'s own argparse block accepts that this
launcher forwards (`--arm`, `--rom`, `--steps`, `--checkpoint-every`,
`--checkpoint-dir`, `--seed`, `--grad-clip-mode`, `--embed-init-mode`,
`--embed-scale`, `--run-tag`, `--task`) is its own flag here too, verified
against `training/train.py`'s argparse block directly (around lines
717-780) rather than assumed -- so nothing about the matrix's configuration
is hardcoded. The one exception is `--rollout-len`: this launcher does not
expose it (it is not in the set of flags the task that produced this file
asked to be forwarded), so every job runs under `training/train.py`'s own
default of 128. `DEFAULT_ROLLOUT_LEN` below exists only so `final_step_for`
-- used for the resume guard -- agrees with that default; if a future
caller ever needs to run this matrix with a non-default `--rollout-len`,
that constant AND a new CLI flag both need to change together, since
nothing currently forwards the value to the child process to keep the two
in sync automatically. This is a real, if narrow, seam -- see this module's
own tests for how it is pinned.

`--task` (docs/DESIGN_ROADMAP_PHASE2.md §9 item 4): Phase 2a's task axis.
Unset (default) leaves every job's behaviour, command line and run directory
EXACTLY as they were before this flag existed. Set (`1-1` or `2-1`, the same
two values `training/train.py`'s own `--task` accepts), it is forwarded
verbatim to every job's child process AND used when this launcher computes
each job's `run_dir` -- through `training.train.run_dir_for`, the SAME
function `training/train.py` itself uses, imported rather than
reimplemented (see the import below), so this launcher's resume/skip guards
can never disagree with where `train.py` actually writes. That agreement is
not a nicety: `docs/EXPERIMENT_LOG.md` §19.4 records a completeness guard
elsewhere in this project that passed FALSELY because its directory pattern
and the real naming convention had quietly drifted apart -- two independent
copies of "what does a run directory look like" is exactly the shape of bug
that produced. `--task` and `--run-tag` compose (both are separate
coordinates `run_dir_for` already knows how to combine; this launcher does
not additionally reconcile them itself).
"""
import argparse
import concurrent.futures
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# See scripts/run_eval_matrix.py's own module docstring for why this is
# needed regardless of invocation style (`python scripts/run_training_matrix.py`
# vs. `python -m scripts.run_training_matrix`): the former puts THIS file's
# own directory on sys.path[0], not the repo root, so `import training` would
# otherwise fail with ModuleNotFoundError depending on cwd.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Imported, not re-declared: this launcher must never invent its own opinion
# about what a valid --grad-clip-mode/--embed-init-mode is, or about how a
# run directory is named. `training/train.py` is the single source of truth
# for all three (its own CLI validates against the first two; `run_dir_for`
# IS the naming convention every checkpoint on disk and every other tool in
# this repo -- `analysis/aggregate_results.py`'s anchored regex included --
# already agrees on). Re-declaring any of them here would create exactly the
# "two independent copies of the same rule, silently drifting apart" failure
# mode this project's other drivers go out of their way to avoid.
from training.train import (EMBED_INIT_MODES, GRAD_CLIP_MODES, NEURON_MODELS,
                            RF_PERIOD_MAX_DEFAULT, RF_PERIOD_MIN_DEFAULT, TASKS,
                            format_task, parse_task, run_dir_for)

ARMS = ("baseline", "reservoir")  # matches training/train.py's --arm choices
DEFAULT_SEEDS = tuple(range(10))
# training/train.py's own --rollout-len default (argparse block, ~line 723).
# This launcher does not expose --rollout-len as a flag of its own (see
# module docstring's last paragraph) -- every job therefore runs under this
# value, and it exists here ONLY so the resume guard's final-step derivation
# (final_step_for) agrees with what the child process will actually do.
DEFAULT_ROLLOUT_LEN = 128

LAUNCHER_LOG_NAME = "launcher.log"
DEFAULT_LOCK_DIR = os.path.join(REPO_ROOT, ".run_training_matrix.lock")
_LOG_TAIL_LINES = 40

_print_lock = threading.Lock()


def _log(msg: str) -> None:
    """Thread-safe `print`: `run_matrix` calls `run_job` from multiple worker
    threads at once, and each prints its own START/DONE line -- without a
    shared lock, two threads' `print` calls can interleave mid-line on some
    platforms/buffering configurations, producing a garbled log line that is
    useless for the "diagnosable after the fact" goal this launcher exists
    to serve."""
    with _print_lock:
        print(msg)


# ---------------------------------------------------------------------------
# --arms / --seeds parsing
# ---------------------------------------------------------------------------

def parse_arms(spec: str) -> tuple:
    """Comma-separated subset of `ARMS` (e.g. `"reservoir"`,
    `"baseline,reservoir"`). Returns them in CANONICAL order
    (`ARMS`'s own order), de-duplicated, regardless of the order the caller
    typed them in -- so `--dry-run` output and job numbering are reproducible
    across equivalent invocations that merely spelled the list differently.
    """
    requested = {a.strip() for a in spec.split(",") if a.strip()}
    for arm in requested:
        if arm not in ARMS:
            raise ValueError(f"--arms: unknown arm {arm!r}, must be one of {ARMS}")
    if not requested:
        raise ValueError(f"--arms: {spec!r} produced no arms")
    return tuple(a for a in ARMS if a in requested)


def parse_seeds(spec: str) -> tuple:
    """Comma-separated list where each item is either a plain integer or an
    inclusive `A-B` range (e.g. `"0-9"`, `"1,3,5"`, `"0-2,7"`). Returns a
    sorted, de-duplicated tuple -- the manifest never depends on whether the
    caller wrote overlapping or out-of-order tokens.
    """
    seeds = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token[1:]:  # ignore a leading '-' so this never misparses a lone negative
            start_str, _, end_str = token.partition("-")
            try:
                start, end = int(start_str), int(end_str)
            except ValueError:
                raise ValueError(f"--seeds: bad range {token!r}, expected INT-INT")
            if start > end:
                raise ValueError(f"--seeds: bad range {token!r}, start must be <= end")
            seeds.update(range(start, end + 1))
        else:
            try:
                seeds.add(int(token))
            except ValueError:
                raise ValueError(f"--seeds: {token!r} is not an integer or an A-B range")
    if not seeds:
        raise ValueError(f"--seeds: {spec!r} produced no seeds")
    return tuple(sorted(seeds))


# ---------------------------------------------------------------------------
# Final-step derivation -- see module docstring "RESUME GUARD, precisely"
# ---------------------------------------------------------------------------

def final_step_for(total_steps: int, rollout_len: int) -> int:
    """The exact step number `training/train.py`'s unconditional final save
    lands on, derived the same way its own loop does it -- NOT a hardcoded
    number, and NOT `total_steps` itself unless `total_steps` happens to be
    an exact multiple of `rollout_len`.

    `run_training`'s loop (`training/train.py`, inside `run_training`) is:
        step = start_step  # 0 for a fresh run
        while step < total_steps:
            ...
            step += rollout_len
    which only ever leaves `step` on a multiple of `rollout_len`, advancing
    STRICTLY PAST `total_steps` on its last iteration unless `total_steps`
    was already such a multiple. That is exactly `ceil(total_steps /
    rollout_len) * rollout_len`, computed here with integer (`-(-a // b)`)
    ceiling division so it is exact for every input, never subject to
    floating-point rounding.

    For this project's own `--steps 1000000` (train.py's `--rollout-len`
    default of 128): ceil(1_000_000 / 128) = 7813, so the final checkpoint is
    `step_1000064.pt` -- see docs/EXPERIMENT_LOG.md §2, and this function's
    own tests, for the trap a launcher that assumed round step numbers falls
    into (it silently finds zero completed runs and re-launches everything).

    `total_steps <= 0` (the `checkpoints_init/` "untrained reference"
    convention, `--steps 0`) is handled as a special case returning 0: the
    loop condition `0 < 0` is false on the very first check, so the loop
    body -- and therefore any increment of `step` -- never runs at all, and
    `training/train.py` writes only `step_0.pt`.
    """
    if total_steps <= 0:
        return 0
    if rollout_len <= 0:
        raise ValueError(f"final_step_for: rollout_len must be positive, got {rollout_len!r}")
    updates = -(-total_steps // rollout_len)  # ceil(total_steps / rollout_len), exact integer math
    return updates * rollout_len


def final_checkpoint_path_for(run_dir: str, final_step: int) -> str:
    """`training/train.py`'s own final-checkpoint naming
    (`save_checkpoint(..., os.path.join(run_dir, f"step_{step}.pt"))`),
    applied to the step `final_step_for` derives. The one place this
    launcher's naming assumption is defined, so a future drift in
    `train.py`'s own `f"step_{step}.pt"` template has exactly one call site
    here to update, not several scattered ones.
    """
    return os.path.join(run_dir, f"step_{final_step}.pt")


# ---------------------------------------------------------------------------
# Manifest: RunConfig (broadcast, identical across every job) + Job (per-cell)
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """Every setting this launcher broadcasts IDENTICALLY to every job in one
    invocation -- everything `training/train.py` accepts EXCEPT `--arm` and
    `--seed`, which vary per job and therefore live on `Job` instead (see
    `build_job_matrix`). Kept as its own dataclass, separate from `Job`,
    because it is the same object passed to every one of the 20 jobs in a
    typical invocation: one place to construct it (`main`, from parsed CLI
    args), one place to read it (`build_command`), no risk of one job
    silently seeing a different `--embed-scale` than its siblings because
    the eight broadcast values were threaded through as eight separate
    positional parameters that could be reordered or partially forwarded by
    mistake.
    """
    rom: str
    steps: int
    checkpoint_every: int
    checkpoint_dir: str
    grad_clip_mode: str
    embed_init_mode: str
    embed_scale: float
    run_tag: Optional[str] = None
    # Phase 2a's task axis (docs/DESIGN_ROADMAP_PHASE2.md §9 item 4). A parsed
    # (world, level) tuple, e.g. (2, 1) -- NOT the raw "--task" spec string; see
    # parse_args, which converts it the same way it already converts --arms/
    # --seeds before RunConfig is ever constructed. None (default) is Phase 1's
    # task-less matrix, unchanged.
    task: Optional[tuple] = None
    # Defaulted, unlike the fields above, so that every existing caller (and every
    # existing test) that constructs a RunConfig without them keeps building the
    # historical configuration rather than failing -- the same courtesy `run_tag`
    # already gets, and the same defaults train.py's own CLI applies.
    neuron_model: str = "lif"
    rf_period_min: float = RF_PERIOD_MIN_DEFAULT
    rf_period_max: float = RF_PERIOD_MAX_DEFAULT


@dataclass
class Job:
    """One (arm, seed) cell of the matrix, fully resolved: where its run
    directory is, what step its final checkpoint will land on, and where its
    checkpoint/log paths are. Everything here is a pure function of
    `(arm, seed, RunConfig)` -- see `build_job_matrix` -- so two callers
    building a `Job` for the same (arm, seed, config) always get the
    byte-identical paths back, which is what lets `resume_status` and a
    LATER, independent invocation of this launcher agree on what "already
    done" means without sharing any in-memory state.
    """
    arm: str
    seed: int
    run_dir: str
    final_step: int
    final_checkpoint_path: str
    log_path: str


def build_job_matrix(config: RunConfig, arms=ARMS, seeds=DEFAULT_SEEDS,
                     rollout_len: int = DEFAULT_ROLLOUT_LEN) -> list:
    """Expands `arms x seeds` into one `Job` each, in canonical arm order
    (`ARMS`'s own order, not necessarily the order `arms` was passed in) x
    ascending seed order -- stable, so `--dry-run` output and job numbering
    are reproducible across runs of the SAME on-disk state and the SAME
    logical request, independent of how `--arms`/`--seeds` happened to be
    spelled on the command line (see `parse_arms`/`parse_seeds`).

    Every job in the returned list shares `config.steps` (hence the same
    `final_step`) -- there is exactly one training budget per launcher
    invocation, applied uniformly, matching `training/train.py`'s own
    single-run-at-a-time `--steps` flag.
    """
    arms = tuple(arms)
    for arm in arms:
        if arm not in ARMS:
            raise ValueError(f"build_job_matrix: unknown arm {arm!r}, must be one of {ARMS}")
    seeds = tuple(sorted(set(seeds)))
    final_step = final_step_for(config.steps, rollout_len)

    jobs = []
    for arm in ARMS:
        if arm not in arms:
            continue
        for seed in seeds:
            run_dir = run_dir_for(config.checkpoint_dir, arm, seed, config.run_tag,
                                  task=config.task)
            jobs.append(Job(
                arm=arm, seed=seed, run_dir=run_dir, final_step=final_step,
                final_checkpoint_path=final_checkpoint_path_for(run_dir, final_step),
                log_path=os.path.join(run_dir, LAUNCHER_LOG_NAME),
            ))
    return jobs


# ---------------------------------------------------------------------------
# Resume guard / skip-if-in-progress guard
# ---------------------------------------------------------------------------

def resume_status(job: Job) -> str:
    """One of `"complete"` (the final checkpoint is already on disk -- skip),
    `"incomplete"` (the run directory exists but the final checkpoint does
    not -- a crashed or still-live run; see module docstring
    "SKIP-IF-IN-PROGRESS"), or `"absent"` (nothing here yet -- safe to start
    fresh). A pure filesystem read, safe to call from `--dry-run` (which
    must never write anything) as well as from a real run.

    Deliberately an EXISTENCE check on the final checkpoint, not a content
    validation: `training/train.py`'s `save_checkpoint` writes with a direct
    `torch.save(..., path)` (no temp-file-then-`os.replace`, unlike this
    project's other write paths -- e.g. `run_eval_matrix._atomic_write_json`
    -- see this module's own docstring for why that makes "incomplete"
    directories dangerous rather than merely untidy), so validating a
    checkpoint's content would mean importing torch and attempting to load
    every candidate file on every invocation. `analysis.aggregate_results`'s
    own checkpoint discovery (`list_checkpoints`, `select_final_checkpoint`)
    makes the identical choice for the identical reason -- checkpoint
    presence, not checkpoint validity, is what this project's tooling
    checks -- so this stays consistent with that precedent rather than
    inventing a stricter rule found nowhere else in the codebase.
    """
    if os.path.isfile(job.final_checkpoint_path):
        return "complete"
    if os.path.isdir(job.run_dir):
        return "incomplete"
    return "absent"


# ---------------------------------------------------------------------------
# Command construction and subprocess execution
# ---------------------------------------------------------------------------

def build_command(python_exe: str, job: Job, config: RunConfig) -> list:
    """The exact argv this launcher hands to `training/train.py` for `job`.
    A pure function of its arguments (no filesystem/env access) so
    `--dry-run` and a real run print/execute the identical command -- exactly
    the same discipline `run_eval_matrix.build_command` follows for its own
    child.

    `-m training.train` (not `training/train.py` as a bare script path):
    matches `run_eval_matrix.build_command`'s own `-m training.evaluate`, and
    means both drivers are insensitive to the child process's own cwd
    resolution of a relative script path -- `-m` always resolves the module
    from `sys.path`, which `REPO_ROOT` is already on for THIS process (see
    top of file) and which `subprocess.run(..., cwd=REPO_ROOT, ...)` in
    `_run_subprocess` guarantees for the child too.
    """
    cmd = [python_exe, "-m", "training.train",
           "--arm", job.arm,
           "--rom", config.rom,
           "--steps", str(config.steps),
           "--checkpoint-every", str(config.checkpoint_every),
           "--checkpoint-dir", config.checkpoint_dir,
           "--seed", str(job.seed),
           "--grad-clip-mode", config.grad_clip_mode,
           "--embed-init-mode", config.embed_init_mode,
           "--embed-scale", str(config.embed_scale),
           # Always emitted, including at the default -- same rule as every flag
           # above it. This argv is what lands in the launcher log and in
           # `--dry-run`'s preview, and a run whose neuron model has to be
           # inferred from an ABSENT flag is a run nobody can label afterwards.
           "--neuron-model", config.neuron_model,
           "--rf-period-min", str(config.rf_period_min),
           "--rf-period-max", str(config.rf_period_max)]
    if config.run_tag:
        cmd += ["--run-tag", config.run_tag]
    if config.task is not None:
        cmd += ["--task", format_task(config.task)]
    return cmd


def _run_subprocess(cmd, env, log_path: str):
    """The ONE place this module actually spawns a child process. Kept as
    its own function (rather than inlined `subprocess.run`) so tests can
    monkeypatch exactly this seam with a plain Python stand-in instead of
    mocking `subprocess` globally -- see `tests/test_run_training_matrix.py`,
    which mirrors how `tests/test_run_eval_matrix.py` isolates its own
    identical seam.

    Opens `log_path` in APPEND mode and hands that file directly to
    `subprocess.run` as both `stdout` and `stderr` -- see module docstring
    "LOGGING TO DISK, NOT MEMORY" for why a multi-hour child's output must
    never be buffered in this process's memory instead. Returns anything
    with `.returncode` (`subprocess.CompletedProcess`'s own shape); unlike
    `run_eval_matrix._run_subprocess`, `.stdout`/`.stderr` on the result are
    NOT populated (they were redirected to `log_path`, not captured) -- a
    caller that needs the child's output reads `log_path` instead (see
    `_tail_log`).
    """
    directory = os.path.dirname(log_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(log_path, "a") as log_fh:
        return subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=log_fh,
                              stderr=subprocess.STDOUT)


def _tail_log(log_path: str, n: int = _LOG_TAIL_LINES) -> str:
    """Last `n` lines of `log_path`, for inlining into a failure message --
    the same "make the failure diagnosable from the message alone" goal
    `run_eval_matrix._extract_json_result`'s docstring describes, adapted to
    a launcher whose child's output lives in a file, not in memory."""
    try:
        with open(log_path) as f:
            lines = f.readlines()
    except OSError:
        return "    (launcher.log not found)"
    tail = lines[-n:] if lines else []
    if not tail:
        return "    (launcher.log is empty)"
    return "\n".join(f"    {line.rstrip(chr(10))}" for line in tail)


# ---------------------------------------------------------------------------
# Per-job execution
# ---------------------------------------------------------------------------

def _job_label(job: Job) -> str:
    return f"{job.arm}/seed{job.seed}"


def run_job(job: Job, index: int, total: int, python_exe: str, config: RunConfig,
           restart_incomplete: bool = False) -> dict:
    """Runs one job to completion, or determines it's already done, or
    refuses to touch an in-progress-looking directory. Returns an outcome
    dict `{"job", "status", "elapsed", ...}` where status is one of
    `"skipped"` (already complete, see `resume_status`), `"ran"` (subprocess
    succeeded AND the final checkpoint was verified on disk), or `"failed"`
    (refused due to an incomplete directory, subprocess launch/exit failure,
    or a clean exit that still didn't produce the final checkpoint -- `error`
    is always present and self-contained enough to act on without re-running
    anything). Never raises: every failure mode is caught and turned into a
    `"failed"` outcome, so one bad job cannot abort the ones running
    alongside it in the thread pool (see `run_matrix`).

    Prints one line when the job starts and one when it finishes (with
    elapsed wall-clock) -- required so a multi-hour unattended run is
    followable without waiting hours for the first line, unlike
    `run_eval_matrix`'s jobs (seconds each), which only print on completion.
    """
    start = time.monotonic()
    label = _job_label(job)
    _log(f"[{index}/{total}] START {label} -> {job.run_dir}")

    def _done(status: str, **extra) -> dict:
        elapsed = time.monotonic() - start
        if status == "skipped":
            _log(f"[{index}/{total}] DONE  {label} ({elapsed:.1f}s) -> "
                f"SKIPPED (final checkpoint already on disk: {job.final_checkpoint_path})")
        elif status == "ran":
            _log(f"[{index}/{total}] DONE  {label} ({elapsed:.1f}s) -> "
                f"completed, final checkpoint {job.final_checkpoint_path}")
        else:
            _log(f"[{index}/{total}] DONE  {label} ({elapsed:.1f}s) -> FAILED: {extra['error']}")
        return {"job": job, "status": status, "elapsed": elapsed, **extra}

    status = resume_status(job)

    if status == "complete":
        return _done("skipped")

    if status == "incomplete" and not restart_incomplete:
        error = (
            f"{job.run_dir} exists but does not contain its final checkpoint "
            f"({os.path.basename(job.final_checkpoint_path)}) -- refusing to touch a "
            f"directory that may belong to a crashed OR a still-live run (this launcher "
            f"cannot tell the two apart from the filesystem alone). If you have confirmed "
            f"no training process is writing to {job.run_dir}, pass --restart-incomplete "
            f"to delete it and re-run this job from scratch."
        )
        return _done("failed", error=error)

    if status == "incomplete" and restart_incomplete:
        shutil.rmtree(job.run_dir)

    os.makedirs(job.run_dir, exist_ok=True)
    env = dict(os.environ)
    # See module docstring "CONCURRENCY": stop each child from fanning out
    # its own BLAS/OpenMP thread pool on top of this launcher's own --jobs cap.
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    cmd = build_command(python_exe, job, config)
    try:
        proc = _run_subprocess(cmd, env, job.log_path)
    except OSError as exc:
        return _done("failed", cmd=cmd,
                    error=f"failed to launch child process: {exc}")

    if proc.returncode != 0:
        error = (f"child exited with status {proc.returncode}. Last "
                f"{_LOG_TAIL_LINES} line(s) of {job.log_path}:\n{_tail_log(job.log_path)}")
        return _done("failed", cmd=cmd, error=error)

    if not os.path.isfile(job.final_checkpoint_path):
        # See module docstring "RESULT VALIDATION, not just exit code": a 0
        # exit status is not, on its own, proof the run actually completed.
        error = (f"child exited 0 but the final checkpoint "
                f"{job.final_checkpoint_path!r} was not created. Last "
                f"{_LOG_TAIL_LINES} line(s) of {job.log_path}:\n{_tail_log(job.log_path)}")
        return _done("failed", cmd=cmd, error=error)

    return _done("ran", cmd=cmd)


def run_matrix(jobs, python_exe: str, config: RunConfig, max_workers: int,
               restart_incomplete: bool = False) -> dict:
    """Runs every job in `jobs` concurrently (up to `max_workers` at a time)
    and returns `{"outcomes", "ran", "skipped", "failed"}` (the last three
    are sub-lists of `outcomes`, partitioned by status). One
    `ThreadPoolExecutor`, not a two-phase split like `run_eval_matrix`'s
    primary/dedup phases -- there is no dedup concept in the training
    matrix, every job is independent of every other job's outcome.
    """
    total = len(jobs)
    outcomes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_job = {
            pool.submit(run_job, job, i, total, python_exe, config, restart_incomplete): job
            for i, job in enumerate(jobs, start=1)
        }
        for future in concurrent.futures.as_completed(future_to_job):
            outcomes.append(future.result())

    return {
        "outcomes": outcomes,
        "ran": [o for o in outcomes if o["status"] == "ran"],
        "skipped": [o for o in outcomes if o["status"] == "skipped"],
        "failed": [o for o in outcomes if o["status"] == "failed"],
    }


# ---------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------

def format_dry_run_lines(jobs, python_exe: str, config: RunConfig,
                         restart_incomplete: bool = False) -> list:
    """Everything `--dry-run` prints: one line per job in matrix order,
    stating exactly what a real run would do for it -- RUN the given
    command, SKIP because the final checkpoint already exists, REFUSE
    because the run directory looks in-progress (unless `restart_incomplete`
    is set, in which case that job shows as a RUN preceded by the deletion
    it would perform) -- and NOTHING here executes anything, including the
    resume check, which is a read-only `os.path` check of files already on
    disk. Followed by a final job-count line.

    Every line names the job's derived FINAL checkpoint filename
    (`step_{final_step}.pt`) up front, not just in the SKIP case -- this is
    deliberate: the whole point of a human eyeballing this output before
    committing hours of compute is to catch exactly the
    docs/EXPERIMENT_LOG.md §2 trap (`step_1000064.pt`, not the "obvious"
    `step_1000000.pt`) before it happens, and that is only checkable if the
    derived number is visible for every row, not only for rows that happen
    to already be complete.
    """
    lines = []
    total = len(jobs)
    for i, job in enumerate(jobs, start=1):
        label = _job_label(job)
        final_name = os.path.basename(job.final_checkpoint_path)
        status = resume_status(job)
        if status == "complete":
            lines.append(f"[{i}/{total}] {label} -> {job.run_dir} (final: {final_name}) :: "
                        f"SKIP (final checkpoint already on disk)")
        elif status == "incomplete" and not restart_incomplete:
            lines.append(f"[{i}/{total}] {label} -> {job.run_dir} (final: {final_name}) :: "
                        f"REFUSE (run directory exists without its final checkpoint -- "
                        f"pass --restart-incomplete to delete it and re-run)")
        else:
            cmd = build_command(python_exe, job, config)
            if status == "incomplete":
                prefix = "RESTART (deletes the existing incomplete directory first), then RUN"
            else:
                prefix = "RUN"
            lines.append(f"[{i}/{total}] {label} -> {job.run_dir} (final: {final_name}) :: "
                        f"{prefix}: {shlex.join(cmd)}")

    lines.append("")
    lines.append(f"TOTAL JOBS: {total}")
    return lines


# ---------------------------------------------------------------------------
# Single-instance lock -- see module docstring "Two independent guards"
# ---------------------------------------------------------------------------

class LockHeld(Exception):
    """Raised by `acquire_lock` when another instance already holds the lock."""


def _read_lock_owner(lock_dir: str) -> Optional[str]:
    try:
        with open(os.path.join(lock_dir, "pid")) as f:
            return f.read().strip()
    except OSError:
        return None


def acquire_lock(lock_dir: str) -> None:
    """`os.mkdir` on `lock_dir` -- atomic on any local filesystem (unlike
    "check whether a marker file exists, then create it," which has a race
    window between the check and the create that `mkdir` does not), so two
    launcher instances started at the exact same moment cannot both believe
    they acquired it. Records the acquiring PID inside the lock directory
    (`{lock_dir}/pid`) so a stale lock -- left behind by a launcher that was
    killed rather than exiting cleanly -- is diagnosable: an operator can
    check whether that PID is still a live `run_training_matrix.py` process
    before reaching for `--force-unlock`.

    Raises `LockHeld` (never a bare `FileExistsError`) with a message that
    names the recorded owner PID and the `--force-unlock` escape hatch, so
    the caller (`main`) can print something actionable without having to
    know this function's internals.
    """
    try:
        os.mkdir(lock_dir)
    except FileExistsError:
        owner = _read_lock_owner(lock_dir)
        raise LockHeld(
            f"another run_training_matrix.py instance appears to be running: lock "
            f"directory {lock_dir!r} already exists (owner pid={owner!r}). If that "
            f"process is confirmed not running, pass --force-unlock to remove the "
            f"stale lock and try again."
        )
    with open(os.path.join(lock_dir, "pid"), "w") as f:
        f.write(str(os.getpid()))


def release_lock(lock_dir: str) -> None:
    """Removes `lock_dir` if present. `ignore_errors=True`: release is called
    from both normal exit (`main`'s `finally`) and from a signal handler
    (see `_install_signal_handlers`) -- in neither case should a filesystem
    hiccup while releasing turn a successful (or already-failing) run into a
    traceback instead of the exit code it already decided on.
    """
    shutil.rmtree(lock_dir, ignore_errors=True)


def force_unlock(lock_dir: str) -> None:
    """The explicit, opt-in escape hatch for a stale lock (see
    `acquire_lock`'s docstring): removes `lock_dir` unconditionally, after
    printing who it was recorded as owned by, so an operator invoking
    `--force-unlock` gets a record of what they just discarded even if they
    didn't check the PID first. A no-op (not an error) when no lock exists,
    since "make sure nothing is locked" is the intent either way.
    """
    if os.path.isdir(lock_dir):
        owner = _read_lock_owner(lock_dir)
        print(f"--force-unlock: removing lock directory {lock_dir!r} "
             f"(was owned by pid={owner!r})")
        shutil.rmtree(lock_dir, ignore_errors=True)


def _install_signal_handlers(lock_dir: str) -> None:
    """Releases the lock on SIGINT (Ctrl-C) or SIGTERM before the process
    exits. Necessary because SIGTERM has no default Python-level exception
    the way SIGINT does (SIGINT's default handler raises `KeyboardInterrupt`,
    which a `try/finally` around `main`'s body would already catch; SIGTERM's
    default action terminates the process directly, with no exception raised
    and therefore no `finally` block ever running) -- so both need an
    explicit handler here for the lock to be reliably released either way,
    exactly the "release it on normal exit AND on SIGINT/SIGTERM" requirement
    this launcher exists to satisfy.

    The handler calls `sys.exit(128 + signum)` (the conventional shell exit
    code for "terminated by signal N") rather than re-delivering the raw
    signal via `os.kill` -- simpler, avoids re-entering signal delivery from
    inside a handler, and is directly unit-testable (a test can invoke the
    installed handler function and assert `SystemExit`, without literally
    sending itself a process signal).
    """
    def _handler(signum, frame):
        release_lock(lock_dir)
        sys.exit(128 + signum)
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Resumable, parallel launcher for the Phase 2 training matrix "
                    "(arm x seed) over training/train.py.")
    parser.add_argument("--arms", default=",".join(ARMS),
                        help=f"comma-separated subset of {ARMS} (default: both)")
    parser.add_argument("--seeds", default="0-9",
                        help="comma-separated seeds and/or inclusive A-B ranges "
                             "(default: 0-9)")
    parser.add_argument("--rom", required=True,
                        help="path to the Game Boy ROM, forwarded to train.py's --rom")
    parser.add_argument("--steps", type=int, default=100_000,
                        help="forwarded to train.py's --steps (default: 100000, "
                             "matching train.py's own default)")
    parser.add_argument("--checkpoint-every", type=int, default=10_000,
                        help="forwarded to train.py's --checkpoint-every (default: "
                             "10000, matching train.py's own default)")
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="forwarded to train.py's --checkpoint-dir; also where "
                             "this launcher's resume/skip guards look for existing "
                             "runs (default: checkpoints, matching train.py's own "
                             "default)")
    parser.add_argument("--grad-clip-mode", choices=list(GRAD_CLIP_MODES), default="global",
                        help="forwarded to train.py's --grad-clip-mode (default: "
                             "global, matching train.py's own default; see "
                             "train.py's own help text for what each mode does)")
    parser.add_argument("--embed-init-mode", choices=list(EMBED_INIT_MODES), default="legacy",
                        help="forwarded to train.py's --embed-init-mode (default: "
                             "legacy, matching train.py's own default)")
    parser.add_argument("--embed-scale", type=float, default=1.0,
                        help="forwarded to train.py's --embed-scale (default: 1.0, "
                             "matching train.py's own default)")
    parser.add_argument("--neuron-model", choices=list(NEURON_MODELS), default="lif",
                        help="forwarded to train.py's --neuron-model (default: lif, "
                             "matching train.py's own default). 'rf' is the "
                             "resonate-and-fire pilot of docs/EXPERIMENT_LOG.md §23 "
                             "and is RESERVOIR-ONLY -- train.py raises on --arm "
                             "baseline, so a matrix that spans both arms cannot be "
                             "launched with it")
    parser.add_argument("--rf-period-min", type=float, default=RF_PERIOD_MIN_DEFAULT,
                        help="forwarded to train.py's --rf-period-min (default: 2.0, "
                             "matching train.py's own default)")
    parser.add_argument("--rf-period-max", type=float, default=RF_PERIOD_MAX_DEFAULT,
                        help="forwarded to train.py's --rf-period-max (default: 32.0, "
                             "matching train.py's own default)")
    parser.add_argument("--run-tag", default=None,
                        help="forwarded to train.py's --run-tag if given (omitted "
                             "entirely from the child command otherwise); also "
                             "changes this launcher's own resume-guard run directory "
                             "the same way, since both go through train.py's own "
                             "run_dir_for")
    parser.add_argument("--task", default=None,
                        help=f"Phase 2a's task axis: one of {sorted(TASKS)} (default: "
                             f"unset, i.e. Phase 1's task-less matrix, unchanged). "
                             f"Forwarded to train.py's --task if given (omitted "
                             f"entirely from the child command otherwise); also "
                             f"changes this launcher's own resume-guard run directory "
                             f"the same way, since both go through train.py's own "
                             f"run_dir_for -- see module docstring")
    parser.add_argument("--jobs", type=int, default=10,
                        help="max concurrent train.py subprocesses (default: 10, "
                             "this machine's core count)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print every job's action/command, in order, and exit "
                             "without running, locking, or writing anything")
    parser.add_argument("--restart-incomplete", action="store_true",
                        help="delete and re-run any run directory that exists but "
                             "lacks its final checkpoint, instead of refusing to "
                             "touch it (see module docstring 'SKIP-IF-IN-PROGRESS')")
    parser.add_argument("--force-unlock", action="store_true",
                        help="remove a pre-existing lock directory before acquiring "
                             "a fresh one -- use only after confirming the owning "
                             "PID (recorded inside the lock directory) is not "
                             "actually a running launcher")
    parser.add_argument("--lock-dir", default=DEFAULT_LOCK_DIR,
                        help="atomic single-instance lock directory (default: "
                             f"{DEFAULT_LOCK_DIR})")
    args = parser.parse_args(argv)

    try:
        args.arms = parse_arms(args.arms)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.seeds = parse_seeds(args.seeds)
    except ValueError as exc:
        parser.error(str(exc))
    # Same pattern as --arms/--seeds above: converted here, once, so args.task
    # is already the typed (world, level) tuple (or None) by the time main() --
    # or a test -- reads it, not a string every caller has to reparse itself.
    if args.task is not None:
        try:
            args.task = parse_task(args.task)
        except ValueError as exc:
            parser.error(str(exc))
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    python_exe = sys.executable

    config = RunConfig(
        rom=args.rom, steps=args.steps, checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir, grad_clip_mode=args.grad_clip_mode,
        embed_init_mode=args.embed_init_mode, embed_scale=args.embed_scale,
        run_tag=args.run_tag, task=args.task, neuron_model=args.neuron_model,
        rf_period_min=args.rf_period_min, rf_period_max=args.rf_period_max,
    )
    jobs = build_job_matrix(config, arms=args.arms, seeds=args.seeds)

    # --dry-run short-circuits before the lock is even touched: it is a
    # read-only preview (see format_dry_run_lines's docstring) and must be
    # usable to eyeball a matrix's commands WHILE a real matrix is running
    # under the same lock, not blocked by it.
    if args.dry_run:
        for line in format_dry_run_lines(jobs, python_exe, config, args.restart_incomplete):
            print(line)
        return 0

    if args.force_unlock:
        force_unlock(args.lock_dir)

    try:
        acquire_lock(args.lock_dir)
    except LockHeld as exc:
        print(str(exc), file=sys.stderr)
        return 1

    _install_signal_handlers(args.lock_dir)
    try:
        report = run_matrix(jobs, python_exe, config, args.jobs, args.restart_incomplete)

        print()
        print(f"{len(jobs)} total jobs: {len(report['ran'])} completed, "
             f"{len(report['skipped'])} skipped (already had a final checkpoint), "
             f"{len(report['failed'])} failed")

        if report["failed"]:
            print("\n=== FAILURES ===")
            for outcome in report["failed"]:
                job = outcome["job"]
                print(f"[{_job_label(job)}] {outcome['error']}")
                if outcome.get("cmd"):
                    print(f"  cmd: {shlex.join(outcome['cmd'])}")
            return 1
        return 0
    finally:
        release_lock(args.lock_dir)


if __name__ == "__main__":
    sys.exit(main())
