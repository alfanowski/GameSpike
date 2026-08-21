"""Phase 2a's evaluation driver: training task x evaluation task cross product.

`scripts/run_eval_matrix.py` is NOT modified or reused as a base for this file
-- `docs/RESULTS.md` §23 pins it byte-identical to commit `64839a9`, and Phase 2a
needs a genuinely different shape anyway. Phase 1's matrix is `selection x arm x
seed x regime`: one checkpoint, evaluated on the one task it implicitly always
ran on (1-1). Phase 2a needs `selection x arm x checkpoint_task x seed x
eval_task x regime`, because a 1-1 specialist scored ONLY on 1-1 tells you
nothing about whether it generalises -- the number the study actually wants is
the OFF-DIAGONAL cell, a checkpoint trained on one task evaluated on the other
(zero-shot transfer). Bolting that cross product onto `run_eval_matrix.py` would
mean either destabilising a pinned, published artefact or growing a second,
incompatible code path inside it; a new, smaller driver is the honest shape.

WHAT IS AND IS NOT REUSED, deliberately, per the task that produced this file
("steal the useful patterns... rather than importing wholesale"):
  * `analysis.aggregate_results.list_checkpoints` / `select_final_checkpoint` --
    these ARE generic over `run_dir` (they take a directory and look for
    `step_*.pt` files in it; they do not hardcode any naming convention), so
    they are imported and reused as-is. `analysis.aggregate_results.
    build_eval_manifest`, by contrast, hardcodes `{arm}_seed{seed}` internally
    rather than calling `training.train.run_dir_for` -- it is Phase 1's own
    task-less naming, unrelated to (and not extended for) Phase 2a's task-aware
    directories, so this driver does not go through it at all; it resolves
    checkpoints itself, via `run_dir_for` (below), the single source of truth
    every other Phase 2a tool already uses.
  * `scripts.run_eval_matrix.REGIMES` / `state_reset_interval_for` -- public,
    generic, and exactly the "continuous"/"reset128" spelling this driver must
    match (per the task brief: "reuse how run_eval_matrix.py expresses this,
    don't invent a new spelling"). Imported, not reimplemented.
  * `scripts.run_training_matrix.parse_arms` / `parse_seeds` -- public,
    generic CLI-parsing helpers with no Phase-1-specific naming baked in.
    Imported rather than re-derived a third time in this repo.
  * Everything else here (`_extract_json_result`'s stdout-scanning, atomic
    writes, the resume check, the job runner) is a SMALL, LOCAL adaptation,
    not an import of `run_eval_matrix.py`'s own (underscore-prefixed, private)
    internals -- this driver does not need Phase 1's locking machinery, its
    best/final dedup concept (Phase 2a's selections are `final`/`init` only;
    see SELECTIONS below), or its two-phase primary/dedup scheduling, and
    coupling to a module whose docstring is pinned byte-identical to a
    specific commit is the wrong kind of dependency for a driver that WILL
    keep changing as Phase 2a's task set grows (§12 OPEN-3's deferred 2-3).

ARMS DEFAULT TO BASELINE ONLY. `docs/DESIGN_ROADMAP_PHASE2.md` §10's own
closing line is explicit: "Phase 2 runs the GRU only." `--arms` still accepts
`reservoir` if an operator has such checkpoints, but the default reflects what
this phase's own compute-budget table actually counts.

SELECTIONS ARE `final`/`init`, NOT Phase 1's `final`/`best`/`init`. Phase 1's
`best` selection exists because the reservoir arm's training reward oscillates
(see `run_eval_matrix.py`'s own header comment); Phase 2a's compute budget
(§10) never asks for it, and adding it back is a small, later extension to
this driver, not a structural change.

OUTPUT NAMING carries BOTH tasks unambiguously, in the filename AND inside the
JSON: `eval_{arm}_ckpt{W-L}_on{W-L}_seed{N}_{regime}.json`, where `ckpt{...}`
is the level the checkpoint TRAINED on and `on{...}` is the level it was
SCORED on. `training/evaluate.py --json`'s own payload already carries `task`
(the eval task, since `--task` is what this driver passes it); this driver
additionally stamps `checkpoint_task` into the written JSON so a reader is
never left to infer the training task from the filename alone.

CONCURRENCY / resume / atomic writes: same reasoning as `run_eval_matrix.py`'s
own module docstring gives for its identical choices (`ThreadPoolExecutor`
because the work is in the child process, not this one; temp-file-then-
`os.replace` so a killed process never leaves a half-written result that a
resume check could mistake for valid).
"""
import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
import sys
import uuid
from dataclasses import dataclass
from typing import Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Same reasoning as every other driver in this repo (see run_eval_matrix.py's
# and run_training_matrix.py's own module docstrings): `python
# scripts/run_phase2a_eval.py` puts THIS file's own directory on sys.path[0],
# not the repo root, so `import training`/`import analysis` would otherwise
# fail depending on invocation style and cwd.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from analysis.aggregate_results import list_checkpoints, select_final_checkpoint
from scripts.run_eval_matrix import REGIMES, state_reset_interval_for
from scripts.run_training_matrix import parse_arms, parse_seeds
from training.train import TASKS, format_task, parse_task, run_dir_for

# Phase 2a's own selections -- see module docstring "SELECTIONS ARE final/init".
SELECTIONS = ("final", "init")

# The keys training/evaluate.py --json's _summarise guarantees for every run --
# same three run_eval_matrix.py checks, for the same reason: enough to prove
# the payload is a real, well-formed evaluate.py result before it is trusted.
REQUIRED_RESULT_KEYS = ("mean_extrinsic_return", "std_extrinsic_return", "episode_lengths")


@dataclass
class Job:
    """One (selection, arm, checkpoint_task, eval_task, seed, regime) cell,
    fully resolved: which checkpoint to load, which task to evaluate it on, and
    where the result must land."""
    selection: str
    arm: str
    checkpoint_task: tuple
    eval_task: tuple
    seed: int
    regime: str
    checkpoint_path: str
    checkpoint_step: Optional[int]
    output_path: str


def output_path_for(results_dir: str, selection: str, arm: str, checkpoint_task: tuple,
                    eval_task: tuple, seed: int, regime: str) -> str:
    """The one place this driver's naming convention is defined -- see module
    docstring "OUTPUT NAMING". `ckpt{...}` before `on{...}` reads left-to-right
    as "trained here, scored there," matching how the brief itself phrases the
    two coordinates."""
    return os.path.join(
        results_dir, selection,
        f"eval_{arm}_ckpt{format_task(checkpoint_task)}_on{format_task(eval_task)}"
        f"_seed{seed}_{regime}.json"
    )


def _job_label(job: Job) -> str:
    return (f"{job.selection}/{job.arm}/ckpt{format_task(job.checkpoint_task)}"
           f"/on{format_task(job.eval_task)}/seed{job.seed}/{job.regime}")


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------

def resolve_checkpoint(checkpoint_root: str, arm: str, checkpoint_task: tuple, seed: int,
                       selection: str):
    """Resolves the ONE checkpoint a (selection, arm, checkpoint_task, seed)
    combination scores -- `run_dir_for` computes the same directory
    `training/train.py` itself writes to, so this driver can never disagree
    with where a checkpoint actually lives (the exact class of bug
    `docs/EXPERIMENT_LOG.md` §19.4 records happening once already, see
    `training.train.run_dir_for`'s own docstring).

    `selection="final"`: the run's highest-step checkpoint, via
    `analysis.aggregate_results.select_final_checkpoint` (generic over
    `run_dir`, so reused rather than reimplemented -- see module docstring).
    `selection="init"`: `training/train.py --steps 0`'s convention, a single
    `step_0.pt` written before any gradient step -- mirrors
    `run_eval_matrix.resolve_init_checkpoints`'s own logic, just task-aware.

    Returns `{"step", "path"}`, or `None` if nothing is there yet (mid-
    experiment; a routine, non-fatal state -- see `build_job_matrix`).
    """
    run_dir = run_dir_for(checkpoint_root, arm, seed, task=checkpoint_task)
    if selection == "final":
        try:
            picked = select_final_checkpoint(run_dir)
        except (FileNotFoundError, ValueError):
            return None
        return {"step": picked["step"], "path": picked["path"]}
    if selection == "init":
        path = os.path.join(run_dir, "step_0.pt")
        if os.path.isfile(path):
            return {"step": 0, "path": path}
        return None
    raise ValueError(f"resolve_checkpoint: unknown selection {selection!r}, "
                     f"must be one of {SELECTIONS}")


def build_job_matrix(checkpoint_dir, init_checkpoint_dir, results_dir,
                     tasks=tuple(TASKS.values()), arms=("baseline",), seeds=range(10),
                     regimes=REGIMES, selections=SELECTIONS):
    """Resolves every (selection, arm, checkpoint_task, seed) to a checkpoint,
    then expands each into one `Job` per (eval_task, regime) -- THE cross
    product this driver exists for (module docstring). A checkpoint that
    cannot be resolved contributes ZERO jobs (not one job per eval_task with a
    missing checkpoint_path) and is recorded once in `missing` instead --
    mirrors `analysis.aggregate_results.build_eval_manifest`'s own "skip, don't
    crash" philosophy for a mid-experiment driver run.

    `selection="final"` resolves against `checkpoint_dir`; `selection="init"`
    against `init_checkpoint_dir` -- the same two-tree split
    `scripts/run_eval_matrix.py` uses, since a trained run and its untrained
    reference are written by two separate `training/train.py --steps N` /
    `--steps 0` invocations with (by convention) different `--checkpoint-dir`
    roots.

    Returns `(jobs, missing)`.
    """
    for s in selections:
        if s not in SELECTIONS:
            raise ValueError(
                f"build_job_matrix: unknown selection {s!r}, must be one of {SELECTIONS}")

    tasks = tuple(tasks)
    arms = tuple(arms)
    seeds = tuple(seeds)
    regimes = tuple(regimes)

    jobs = []
    missing = []
    for selection in selections:
        checkpoint_root = checkpoint_dir if selection == "final" else init_checkpoint_dir
        for arm in arms:
            for checkpoint_task in tasks:
                for seed in seeds:
                    entry = resolve_checkpoint(checkpoint_root, arm, checkpoint_task,
                                               seed, selection)
                    if entry is None:
                        run_dir = run_dir_for(checkpoint_root, arm, seed, task=checkpoint_task)
                        missing.append({
                            "selection": selection, "arm": arm,
                            "checkpoint_task": checkpoint_task, "seed": seed,
                            "reason": f"no checkpoint found under {run_dir!r}",
                        })
                        continue
                    for eval_task in tasks:
                        for regime in regimes:
                            jobs.append(Job(
                                selection=selection, arm=arm,
                                checkpoint_task=checkpoint_task, eval_task=eval_task,
                                seed=seed, regime=regime,
                                checkpoint_path=entry["path"], checkpoint_step=entry["step"],
                                output_path=output_path_for(
                                    results_dir, selection, arm, checkpoint_task,
                                    eval_task, seed, regime),
                            ))
    return jobs, missing


# ---------------------------------------------------------------------------
# Result validation, atomic writes, resume check -- adapted from
# scripts/run_eval_matrix.py's own (private) helpers; see module docstring
# "WHAT IS AND IS NOT REUSED" for why these are local copies, not imports.
# ---------------------------------------------------------------------------

def _resume_check(path: str) -> bool:
    """True iff `path` exists, parses as JSON, and contains
    `mean_extrinsic_return` -- shallow on purpose, mirroring
    `run_eval_matrix._resume_check`'s own docstring: this only needs to catch a
    file a killed process left truncated, not re-validate a file this driver's
    own atomic-write path already accepted."""
    if not os.path.isfile(path):
        return False
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return False
    return isinstance(data, dict) and "mean_extrinsic_return" in data


def validate_result(data, job: Job):
    """Raises `ValueError` if `data` does not match what `job` asked for. Arm
    mismatch is the hard-stop case: a checkpoint trained for one architecture
    scored as if it were the other is a wrong-experiment result, not a data-
    quality nit."""
    if not isinstance(data, dict):
        raise ValueError(f"result is not a JSON object (got {type(data).__name__})")
    json_arm = data.get("arm")
    if json_arm != job.arm:
        raise ValueError(
            f"arm mismatch: job expects arm={job.arm!r} (checkpoint={job.checkpoint_path!r}) "
            f"but the result JSON's own 'arm' key says {json_arm!r}"
        )
    for key in REQUIRED_RESULT_KEYS:
        if key not in data:
            raise ValueError(
                f"result for {_job_label(job)!r} is missing required key {key!r} -- "
                f"not a well-formed training/evaluate.py --json payload"
            )


def _atomic_write_json(path: str, data) -> None:
    """Temp-file-then-`os.replace`, so a reader (including this driver's own
    next run) sees either the OLD content or the FULLY WRITTEN new content,
    never a partial write -- same pattern as `run_eval_matrix._atomic_write_json`."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


# ---------------------------------------------------------------------------
# Command construction and subprocess execution
# ---------------------------------------------------------------------------

def build_command(python_exe: str, job: Job, rom: str, episodes: int, eval_seed: int) -> list:
    """The exact argv handed to `training/evaluate.py` for `job`. Note
    `--task` is `job.eval_task`, NOT `job.checkpoint_task` -- the whole point
    of this driver is scoring a checkpoint on a task that may differ from the
    one it trained on."""
    cmd = [python_exe, "-m", "training.evaluate",
           "--arm", job.arm,
           "--checkpoint", job.checkpoint_path,
           "--rom", rom,
           "--episodes", str(episodes),
           "--seed", str(eval_seed),
           "--task", format_task(job.eval_task),
           "--json"]
    interval = state_reset_interval_for(job.regime)
    if interval is not None:
        cmd += ["--state-reset-interval", str(interval)]
    return cmd


def _run_subprocess(cmd, env):
    """The one place this module spawns a child process -- kept as its own
    function so tests can monkeypatch exactly this seam (mirrors both other
    drivers' identical seam)."""
    return subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)


def _extract_json_result(stdout: str):
    """Scans `stdout` from the END for the first line that parses as a JSON
    OBJECT carrying an `arm` key -- adapted from
    `run_eval_matrix._extract_json_result`, whose own docstring explains WHY a
    naive whole-stdout `json.loads` is wrong: PyBoy writes "Missing dependency
    Pillow" WARNING lines to stdout, not stderr, ahead of `training/evaluate.py
    --json`'s own single result line."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "arm" in candidate:
            return candidate
    return None


def run_job(job: Job, python_exe: str, rom: str, episodes: int, eval_seed: int) -> dict:
    """Runs one job to completion, or determines it is already done. Returns
    an outcome dict `{"job", "status", ...}`, status one of "skipped" (valid
    result already present), "ran" (subprocess succeeded, result validated and
    written, self-describing fields added), or "failed" (`error`, plus
    `cmd`/`stdout`/`stderr` where available). Never raises -- one bad job must
    not abort the ones running alongside it (see `run_matrix`).
    """
    if _resume_check(job.output_path):
        return {"job": job, "status": "skipped"}

    env = dict(os.environ)
    # Stop each child fanning out its own BLAS/OpenMP pool on top of this
    # driver's own --jobs cap -- same reasoning as both other drivers.
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    cmd = build_command(python_exe, job, rom, episodes, eval_seed)
    try:
        proc = _run_subprocess(cmd, env)
    except OSError as exc:
        return {"job": job, "status": "failed", "cmd": cmd,
                "error": f"failed to launch child process: {exc}"}

    if proc.returncode != 0:
        return {"job": job, "status": "failed", "cmd": cmd,
                "error": f"child exited with status {proc.returncode}",
                "stdout": proc.stdout, "stderr": proc.stderr}

    data = _extract_json_result(proc.stdout)
    if data is None:
        tail_lines = proc.stdout.splitlines()[-20:]
        tail = "\n".join(f"    {line}" for line in tail_lines) if tail_lines else "    (empty)"
        stderr_block = ("\n".join(f"    {line}" for line in proc.stderr.splitlines())
                        if proc.stderr else "    (empty)")
        error = (
            f"child exited {proc.returncode} but no line of stdout parsed as a JSON "
            f"object containing an 'arm' key. Last {len(tail_lines)} line(s) of stdout:\n"
            f"{tail}\n  stderr:\n{stderr_block}"
        )
        return {"job": job, "status": "failed", "cmd": cmd, "error": error,
                "stdout": proc.stdout, "stderr": proc.stderr}

    try:
        validate_result(data, job)
    except ValueError as exc:
        return {"job": job, "status": "failed", "cmd": cmd, "error": str(exc),
                "stdout": proc.stdout, "stderr": proc.stderr}

    # Self-describing: `data` already carries evaluate.py's own "task" (the
    # EVAL task, since --task above was job.eval_task). This driver adds the
    # coordinate evaluate.py cannot know on its own -- which task the
    # CHECKPOINT was trained on -- plus the matrix coordinates a reader would
    # otherwise have to reconstruct from the filename alone.
    enriched = dict(data)
    enriched["checkpoint_task"] = list(job.checkpoint_task)
    enriched["training_seed"] = job.seed
    enriched["selection"] = job.selection
    enriched["checkpoint_step"] = job.checkpoint_step
    enriched["checkpoint_path"] = job.checkpoint_path

    _atomic_write_json(job.output_path, enriched)
    return {"job": job, "status": "ran", "mean_extrinsic_return": enriched["mean_extrinsic_return"]}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _format_progress(index: int, total: int, job: Job, outcome: dict) -> str:
    label = _job_label(job)
    status = outcome["status"]
    if status == "ran":
        return (f"[{index}/{total}] {label} -> "
               f"mean_extrinsic_return={outcome['mean_extrinsic_return']:.4f}")
    if status == "skipped":
        return f"[{index}/{total}] {label} -> SKIPPED (valid result already on disk)"
    if status == "failed":
        return f"[{index}/{total}] {label} -> FAILED: {outcome['error']}"
    raise ValueError(f"_format_progress: unknown status {status!r}")  # pragma: no cover


def run_matrix(jobs, python_exe: str, rom: str, episodes: int, eval_seed: int,
               max_workers: int) -> dict:
    """Runs every job concurrently (up to `max_workers`), returns
    `{"outcomes", "ran", "skipped", "failed"}`. One flat `ThreadPoolExecutor`
    -- no dedup phase, unlike `run_eval_matrix.run_matrix`: Phase 2a's
    selections (`final`/`init`) have no best-equals-final concept to dedup."""
    total = len(jobs)
    outcomes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_job = {
            pool.submit(run_job, job, python_exe, rom, episodes, eval_seed): job
            for job in jobs
        }
        for future in concurrent.futures.as_completed(future_to_job):
            outcome = future.result()
            outcomes.append(outcome)
            print(_format_progress(len(outcomes), total, outcome["job"], outcome))

    return {
        "outcomes": outcomes,
        "ran": [o for o in outcomes if o["status"] == "ran"],
        "skipped": [o for o in outcomes if o["status"] == "skipped"],
        "failed": [o for o in outcomes if o["status"] == "failed"],
    }


def format_dry_run_lines(jobs, missing, python_exe: str, rom: str, episodes: int,
                         eval_seed: int) -> list:
    """Everything `--dry-run` prints: one line per job (RUN the command, or
    SKIP because a valid result already exists), then the missing checkpoints,
    then a job-count line. Nothing here executes anything or writes anything."""
    lines = []
    total = len(jobs)
    for i, job in enumerate(jobs, start=1):
        label = _job_label(job)
        if _resume_check(job.output_path):
            lines.append(f"[{i}/{total}] {label} -> {job.output_path} :: "
                         f"SKIP (valid result already on disk)")
        else:
            cmd = build_command(python_exe, job, rom, episodes, eval_seed)
            lines.append(f"[{i}/{total}] {label} -> {job.output_path} :: {shlex.join(cmd)}")

    if missing:
        lines.append("")
        lines.append(f"{len(missing)} (arm, checkpoint_task, seed) combo(s) skipped "
                     f"entirely -- no checkpoint found:")
        for m in missing:
            lines.append(
                f"  selection={m['selection']} arm={m['arm']} "
                f"checkpoint_task={format_task(m['checkpoint_task'])} seed={m['seed']}: "
                f"{m['reason']}"
            )

    lines.append("")
    lines.append(f"TOTAL JOBS: {total}")
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_tasks(spec: str) -> tuple:
    """Comma-separated W-L specs, e.g. '1-1,2-1'. Validated against
    `training.train.TASKS` (via `parse_task`) -- the same allow-list every
    other Phase 2a tool uses, so a typo here fails exactly the way it would in
    `training/train.py --task`. De-duplicated, order preserved (first
    occurrence wins), rather than sorted -- there is no canonical order to
    sort against beyond "the order the caller asked for."
    """
    tasks = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        tasks.append(parse_task(token))
    if not tasks:
        raise ValueError(f"--tasks: {spec!r} produced no tasks")
    seen = []
    for t in tasks:
        if t not in seen:
            seen.append(t)
    return tuple(seen)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Phase 2a's evaluation driver: checkpoint_task x eval_task x "
                    "arm x seed x regime over training/evaluate.py.")
    parser.add_argument("--rom", required=True, help="path to the Game Boy ROM")
    parser.add_argument("--tasks", default=",".join(TASKS),
                        help=f"comma-separated subset of {sorted(TASKS)} -- every "
                             f"checkpoint_task is ALSO evaluated on every eval_task in "
                             f"this same set (default: all of them)")
    parser.add_argument("--arms", default="baseline",
                        help="comma-separated subset of ('baseline', 'reservoir') "
                             "(default: baseline only -- docs/DESIGN_ROADMAP_PHASE2.md "
                             "§10: 'Phase 2 runs the GRU only')")
    parser.add_argument("--seeds", default="0-9",
                        help="comma-separated seeds and/or inclusive A-B ranges "
                             "(default: 0-9)")
    parser.add_argument("--regimes", default=",".join(REGIMES),
                        help=f"comma-separated subset of {REGIMES} (default: both)")
    parser.add_argument("--selections", default=",".join(SELECTIONS),
                        help=f"comma-separated subset of {SELECTIONS} (default: both; "
                             f"NOTE: no 'best' here, unlike run_eval_matrix.py -- see "
                             f"module docstring)")
    parser.add_argument("--episodes", type=int, default=30,
                        help="episodes per evaluation (default: 30, matching "
                             "run_eval_matrix.py's own default)")
    parser.add_argument("--eval-seed", type=int, default=0,
                        help="base evaluation seed passed as evaluate.py's --seed "
                             "(default: 0) -- NOT the training seed")
    parser.add_argument("--results-dir", default="results_phase2a",
                        help="output root; writes to {results_dir}/{selection}/... "
                             "(default: results_phase2a, distinct from Phase 1's "
                             "results/ tree)")
    parser.add_argument("--checkpoint-dir", default="checkpoints_phase2a",
                        help="trained-run checkpoints for 'final' selection "
                             "(default: checkpoints_phase2a)")
    parser.add_argument("--init-checkpoint-dir", default="checkpoints_phase2a_init",
                        help="untrained reference checkpoints for 'init' selection "
                             "(default: checkpoints_phase2a_init)")
    parser.add_argument("--jobs", type=int, default=8,
                        help="max concurrent evaluate.py subprocesses (default: 8, "
                             "matching run_eval_matrix.py's own default)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print every command/action this run would take, in "
                             "order, and exit without executing or writing anything")
    args = parser.parse_args(argv)

    try:
        args.tasks = parse_tasks(args.tasks)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.arms = parse_arms(args.arms)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.seeds = parse_seeds(args.seeds)
    except ValueError as exc:
        parser.error(str(exc))

    regimes = tuple(r.strip() for r in args.regimes.split(",") if r.strip())
    for r in regimes:
        if r not in REGIMES:
            parser.error(f"--regimes: unknown regime {r!r}, must be one of {REGIMES}")
    args.regimes = regimes

    selections = tuple(s.strip() for s in args.selections.split(",") if s.strip())
    for s in selections:
        if s not in SELECTIONS:
            parser.error(f"--selections: unknown selection {s!r}, must be one of {SELECTIONS}")
    args.selections = selections
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    python_exe = sys.executable

    jobs, missing = build_job_matrix(
        checkpoint_dir=args.checkpoint_dir, init_checkpoint_dir=args.init_checkpoint_dir,
        results_dir=args.results_dir, tasks=args.tasks, arms=args.arms, seeds=args.seeds,
        regimes=args.regimes, selections=args.selections,
    )

    if args.dry_run:
        for line in format_dry_run_lines(jobs, missing, python_exe, args.rom,
                                         args.episodes, args.eval_seed):
            print(line)
        return 0

    report = run_matrix(jobs, python_exe, args.rom, args.episodes, args.eval_seed, args.jobs)

    print()
    print(f"{len(jobs)} total jobs: {len(report['ran'])} ran, "
         f"{len(report['skipped'])} skipped (already had a valid result), "
         f"{len(report['failed'])} failed")
    if missing:
        print(f"{len(missing)} (arm, checkpoint_task, seed) combo(s) had no checkpoint "
             f"to evaluate at all (skipped, not failed):")
        for m in missing:
            print(f"  selection={m['selection']} arm={m['arm']} "
                 f"checkpoint_task={format_task(m['checkpoint_task'])} seed={m['seed']}: "
                 f"{m['reason']}")

    if report["failed"]:
        print("\n=== FAILURES ===")
        for outcome in report["failed"]:
            job = outcome["job"]
            print(f"[{_job_label(job)}] {outcome['error']}")
            if outcome.get("cmd"):
                print(f"  cmd: {shlex.join(outcome['cmd'])}")
            if outcome.get("stderr"):
                print("  stderr:")
                for line in outcome["stderr"].splitlines():
                    print(f"    {line}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
