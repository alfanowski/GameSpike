"""Resumable, parallel driver for the Phase 1 evaluation matrix.

Runs `training/evaluate.py` over every checkpoint the experiment cares about
and writes one JSON per (selection, arm, seed, regime) to
`{results_dir}/{selection}/eval_{arm}_seed{seed}_{regime}.json` -- the exact
naming `analysis.aggregate_results.load_eval_results` parses (arm, train_seed,
regime out of the filename; see that function's docstring for why it refuses
to guess from anything that doesn't match). This module never reimplements
checkpoint discovery: `final`/`best` selection go through
`analysis.aggregate_results.build_eval_manifest`, which is already the single
source of truth for "which checkpoint of this run gets evaluated."

THE MATRIX: 3 selections x 2 arms x 10 seeds x 2 regimes = up to 120 jobs.

  * `final` / `best` -- see `aggregate_results.py`'s own section-2.5 header
    comment for the full "why": in short, the reservoir arm's training reward
    oscillates (peaks then regresses) with grad-norm blowup, so evaluating
    only the LAST checkpoint would score it at an arbitrary point of that
    oscillation. `best` fixes that by picking the checkpoint with the highest
    mean TRAINING reward. Selection is always on the TRAINING metric, NEVER
    the evaluation metric that comes out of this driver -- selecting on the
    evaluation measure would test on the training set of the selection
    procedure itself (whichever checkpoint got a lucky handful of eval
    episodes would be preferred for having gotten lucky), biasing the
    reported arm comparison upward for a reason that has nothing to do with
    which architecture actually generalises. `aggregate_results.py` calls
    this "the one rule that must never be violated" and this driver does not
    violate it: it only ever asks `build_eval_manifest` to select on
    `train_log.jsonl`'s `mean_extrinsic_reward`.
  * `init` -- the untrained reference (`checkpoints_init/{arm}_seed{N}/
    step_0.pt`, one per run, produced by `training/train.py` at step 0). This
    is the experimental control: whatever the trained checkpoints score, the
    comparison is only meaningful next to what a freshly-initialised model of
    the same arm scores under an identical evaluation procedure.
  * regimes `continuous` (no `--state-reset-interval`) and `reset128`
    (`--state-reset-interval 128`, matching `training/train.py`'s
    `rollout_len` default) -- see `training/evaluate.py`'s module docstring,
    "THE RECURRENT-STATE REGIME HERE IS NOT THE ONE EITHER ARM WAS TRAINED
    IN," for why both get run and reported rather than just one.

DEDUP (best == final): for a run whose training reward never regressed,
`select_best_checkpoint` and `select_final_checkpoint` land on the identical
step, i.e. the identical `.pt` file. Evaluation is a pure function of
(checkpoint weights, rom, episodes, eval seed, regime) -- `evaluate.py`'s own
module docstring is explicit that its only randomness is a private
per-episode `torch.Generator` seeded from the `--seed` this driver passes
identically to both the `final` and `best` invocations for the same
(arm, seed, regime). Re-running the same checkpoint through the same harness
with the same seed does not produce a second independent measurement; it
reproduces the same file, byte-for-byte modulo the `_copied_from` key added
below, at the cost of several real minutes of emulator time per job. So when
the resolved paths match, this driver copies the `final` result to the `best`
location instead of re-running, and stamps the copy with `_copied_from` so a
later reader can never mistake it for an independently-executed measurement.

RESUMABILITY: every job checks for a pre-existing, validly-parsing output file
before doing any work (subprocess OR copy) -- an unattended multi-hour run can
be killed and restarted without redoing completed jobs. "Validly parsing"
is checked for real (JSON loads, has `mean_extrinsic_return`), not merely
"the path exists" -- a file left behind by a killed process mid-write must not
be trusted. This driver never produces such a file itself: every write goes to
a temp file in the same directory, then `os.replace` (atomic on a local
filesystem), so a result file this driver wrote is either fully absent or
fully valid; the leftover temp file from an interrupted run doesn't even match
the `eval_*.json` glob `load_eval_results`/the resume check use, so it's
inert.

CONCURRENCY: `concurrent.futures.ThreadPoolExecutor` driving `subprocess.run`
-- the actual work (PyBoy stepping, torch forward passes) happens in the
child process, so the parent thread spends its time blocked on
`Popen.communicate()`, during which the GIL is released; threads are the
right tool here, not multiprocessing, because there is no CPU-bound work in
THIS process to parallelise. Each child gets `OMP_NUM_THREADS=1` and
`MKL_NUM_THREADS=1` so that N parallel evaluate.py processes don't each also
fan out their own BLAS/OpenMP thread pool -- without this, `--jobs 8` would
silently oversubscribe far past 8 cores, which is exactly the
saturated-machine problem this driver exists to run into deliberately (via
`--jobs`) rather than accidentally (via nested, uncapped thread pools).
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
# Invoked two different ways: `python -m scripts.run_eval_matrix` (repo root
# already on sys.path) and, per this project's own documented usage, directly
# as `python scripts/run_eval_matrix.py` -- which makes Python put THIS
# file's own directory (scripts/) on sys.path[0], not the repo root, so
# `import analysis` would otherwise fail with ModuleNotFoundError regardless
# of the current working directory. Inserted before the local import below
# so both invocation styles resolve `analysis`/`training`/`envs` identically.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from analysis.aggregate_results import build_eval_manifest

ARMS = ("baseline", "reservoir")
SEEDS = tuple(range(10))
# "continuous" -> no --state-reset-interval; "reset128" -> --state-reset-interval 128.
# Order matters only for deterministic output ordering (dry-run / progress).
REGIMES = ("continuous", "reset128")
SELECTIONS = ("final", "best", "init")

# The keys `training/evaluate.py --json`'s `_summarise` guarantees exist for
# every run, checked against that module directly rather than assumed:
# `mean_{name}`/`std_{name}` per metric, plus the raw per-episode list
# `{name}s`. These three are enough to prove the payload is a real,
# well-formed evaluation result (arm is checked separately -- see
# `validate_result`, since a wrong arm is a distinct failure mode from a
# malformed payload).
REQUIRED_RESULT_KEYS = ("mean_extrinsic_return", "std_extrinsic_return", "episode_lengths")


@dataclass
class Job:
    """One (selection, arm, seed, regime) cell of the matrix, fully resolved:
    which checkpoint to load and where the result must land. `dedup_source`,
    when set, is the `final` job's output path this job is byte-identical to
    (see module docstring "DEDUP") -- its presence is what tells `run_matrix`
    to copy instead of spawning a subprocess.
    """
    selection: str
    arm: str
    seed: int
    regime: str
    checkpoint_path: str
    checkpoint_step: Optional[int]
    output_path: str
    dedup_source: Optional[str] = None


def state_reset_interval_for(regime: str):
    """None for `continuous` (one uninterrupted playthrough); 128 for
    `reset128` (training's own rollout boundary -- see module docstring)."""
    if regime == "continuous":
        return None
    if regime == "reset128":
        return 128
    raise ValueError(f"state_reset_interval_for: unknown regime {regime!r}")


def output_path_for(results_dir: str, selection: str, arm: str, seed: int, regime: str) -> str:
    """The one place this driver's naming convention is defined. MUST stay in
    lockstep with `analysis.aggregate_results._EVAL_FILENAME_RE`
    (`eval_{arm}_seed{trainseed}_{continuous|reset128}.json`) -- that regex is
    what `load_eval_results` uses to parse the training seed and regime back
    out of a filename it did not write. `tests/test_run_eval_matrix.py` pins
    this round-trip directly against `load_eval_results`, not just against
    this function, so a drift here is caught by an actual parse, not by two
    copies of the same string template agreeing with each other.
    """
    return os.path.join(results_dir, selection, f"eval_{arm}_seed{seed}_{regime}.json")


def _job_label(job: Job) -> str:
    return f"{job.selection}/{job.arm}/seed{job.seed}/{job.regime}"


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------

def resolve_init_checkpoints(init_checkpoint_dir, arms, seeds):
    """The `init` selection's counterpart to `build_eval_manifest`: every run
    under `init_checkpoint_dir` has exactly one checkpoint, `step_0.pt`
    (`training/train.py` writes it once, at construction, before any gradient
    step). There is no "final vs. best" question here -- there is exactly one
    file -- so this is a directory existence check, not a selection rule.

    Returns `(resolved, missing)` in the same shape as `build_eval_manifest`'s
    two return values, so `build_job_matrix` can treat all three selections
    uniformly: `resolved` maps `(arm, seed) -> {"step": 0, "path": ...}`,
    `missing` lists `{"selection": "init", "arm", "seed", "reason"}` for any
    (arm, seed) whose `step_0.pt` isn't there yet -- e.g. mid-experiment,
    before the untrained reference checkpoints have been generated for every
    seed. Skipped, not fatal, matching `build_eval_manifest`'s own philosophy
    of "this driver may run before every run exists."
    """
    resolved = {}
    missing = []
    for arm in arms:
        for seed in seeds:
            path = os.path.join(str(init_checkpoint_dir), f"{arm}_seed{seed}", "step_0.pt")
            if os.path.isfile(path):
                resolved[(arm, seed)] = {"step": 0, "path": path}
            else:
                missing.append({
                    "selection": "init", "arm": arm, "seed": seed,
                    "reason": f"{path!r} does not exist",
                })
    return resolved, missing


def build_job_matrix(checkpoint_dir, init_checkpoint_dir, results_dir,
                      selections=SELECTIONS, arms=ARMS, seeds=SEEDS, regimes=REGIMES):
    """Resolves every requested (selection, arm, seed) to a checkpoint, then
    expands each into one `Job` per regime. Returns `(jobs, missing)`:
    `jobs` is a flat list in `selection -> arm -> seed -> regime` order
    (stable, so `--dry-run` output and progress numbering are reproducible
    across runs of the SAME on-disk state); `missing` lists every (arm, seed)
    under a requested selection that has no checkpoint to evaluate yet
    (mid-experiment runs that haven't started, or haven't checkpointed yet --
    see `build_eval_manifest`'s and `resolve_init_checkpoints`'s own
    docstrings). A missing (arm, seed) simply contributes no jobs; it is not
    an error this function raises.

    DEDUP resolution happens HERE, once, from the resolved paths alone (before
    any job runs) -- see module docstring "DEDUP". It only fires when `final`
    is ALSO among `selections`: if the caller runs `--selections best` alone,
    there is no `final` job in THIS invocation to copy from (an on-disk
    `results/final/...` file from a past run is deliberately not consulted
    here, since this function's contract is about what job structure a given
    `selections` request implies, not about probing the filesystem for
    unrelated past runs), so `best` jobs run for real in that case.
    """
    for s in selections:
        if s not in SELECTIONS:
            raise ValueError(
                f"build_job_matrix: unknown selection {s!r}, must be one of {SELECTIONS}")

    arms = tuple(arms)
    seeds = tuple(seeds)
    regimes = tuple(regimes)

    resolved_by_selection = {}
    missing = []
    for selection in selections:
        if selection in ("final", "best"):
            result = build_eval_manifest(checkpoint_dir, arms=arms, seeds=seeds,
                                         selection=selection)
            resolved_by_selection[selection] = {
                (m["arm"], m["seed"]): m for m in result["manifest"]
            }
            for miss in result["missing"]:
                missing.append({"selection": selection, **miss})
        else:  # "init"
            resolved, init_missing = resolve_init_checkpoints(init_checkpoint_dir, arms, seeds)
            resolved_by_selection["init"] = resolved
            missing.extend(init_missing)

    final_map = resolved_by_selection.get("final", {})
    jobs = []
    for selection in selections:
        sel_map = resolved_by_selection.get(selection, {})
        for arm in arms:
            for seed in seeds:
                entry = sel_map.get((arm, seed))
                if entry is None:
                    continue  # already recorded in `missing` above
                for regime in regimes:
                    dedup_source = None
                    if selection == "best" and "final" in selections:
                        final_entry = final_map.get((arm, seed))
                        # Byte-identical PATH, not merely equal step: the path
                        # is what actually gets loaded, so comparing paths is
                        # the literal statement of "this would load the exact
                        # same weights."
                        if final_entry is not None and final_entry["path"] == entry["path"]:
                            dedup_source = output_path_for(
                                results_dir, "final", arm, seed, regime)
                    jobs.append(Job(
                        selection=selection, arm=arm, seed=seed, regime=regime,
                        checkpoint_path=entry["path"], checkpoint_step=entry.get("step"),
                        output_path=output_path_for(results_dir, selection, arm, seed, regime),
                        dedup_source=dedup_source,
                    ))
    return jobs, missing


# ---------------------------------------------------------------------------
# Result validation, atomic writes, resume check
# ---------------------------------------------------------------------------

def _resume_check(path: str) -> bool:
    """True iff `path` exists, parses as JSON, and contains the key
    `mean_extrinsic_return`. This is deliberately a SHALLOW check (not the
    full `validate_result` below, which also checks the arm): the file was
    written by THIS driver's own atomic-write path (or an equally careful
    predecessor run), so an arm mismatch inside an already-accepted file
    would be a bug in a past run's validation, not something a resume check
    should re-litigate. What it must catch is a file a killed process left
    truncated -- `json.load` raising is exactly that signal.
    """
    if not os.path.isfile(path):
        return False
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return False
    return isinstance(data, dict) and "mean_extrinsic_return" in data


def validate_result(data, job: Job):
    """Raises `ValueError` if `data` (parsed JSON, from a child process or a
    dedup copy source) does not match what `job` asked for. Arm mismatch is
    the hard-stop case called out in the task brief: it would mean a
    checkpoint trained for one architecture got loaded and scored as if it
    were the other, which is not a data-quality nit, it's a wrong-experiment
    result that must never silently land in `results/`.
    """
    if not isinstance(data, dict):
        raise ValueError(f"result is not a JSON object (got {type(data).__name__})")
    json_arm = data.get("arm")
    if json_arm != job.arm:
        raise ValueError(
            f"arm mismatch: job expects arm={job.arm!r} (checkpoint={job.checkpoint_path!r}) "
            f"but the result JSON's own 'arm' key says {json_arm!r} -- refusing to accept "
            f"a checkpoint scored under the wrong architecture"
        )
    for key in REQUIRED_RESULT_KEYS:
        if key not in data:
            raise ValueError(
                f"result for {_job_label(job)!r} is missing required key {key!r} -- "
                f"not a well-formed training/evaluate.py --json payload"
            )


def _atomic_write_json(path: str, data) -> None:
    """Writes `data` to `path` such that any reader (including this driver's
    own next run) sees either the OLD content or the FULLY WRITTEN new
    content, never a partial write. Writes to a sibling temp file first, then
    `os.replace` (atomic on any single local filesystem, POSIX or NTFS) onto
    the real path. The temp filename deliberately does not match `eval_*.json`
    (it's `eval_*.json.tmp-<pid>-<uuid>`), so if the process is killed between
    the write and the rename, the leftover temp file is invisible to both
    `_resume_check`'s glob-free direct-path check and to
    `analysis.aggregate_results.load_eval_results`'s `eval_*.json` glob.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, path)
    except BaseException:
        # Best-effort cleanup so a failed write doesn't litter the results
        # directory with orphaned temp files across a long unattended run.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


# ---------------------------------------------------------------------------
# Command construction and subprocess execution
# ---------------------------------------------------------------------------

def build_command(python_exe: str, job: Job, rom: str, episodes: int, eval_seed: int) -> list:
    """The exact argv this driver hands to `training/evaluate.py` for `job`.
    A pure function of its arguments (no filesystem/env access) so
    `--dry-run` and the real run print/execute the identical command."""
    cmd = [python_exe, "-m", "training.evaluate",
           "--arm", job.arm,
           "--checkpoint", job.checkpoint_path,
           "--rom", rom,
           "--episodes", str(episodes),
           "--seed", str(eval_seed),
           "--json"]
    interval = state_reset_interval_for(job.regime)
    if interval is not None:
        cmd += ["--state-reset-interval", str(interval)]
    return cmd


def _run_subprocess(cmd, env):
    """The ONE place this module actually spawns a child process. Kept as its
    own function (rather than inlined `subprocess.run`) so tests can
    monkeypatch exactly this seam with a plain Python stand-in instead of
    mocking `subprocess` globally -- see `tests/test_run_eval_matrix.py`.
    Returns anything with `.returncode`, `.stdout`, `.stderr`
    (`subprocess.CompletedProcess`'s own shape).
    """
    return subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)


def _extract_json_result(stdout: str) -> Optional[dict]:
    """Finds `training/evaluate.py --json`'s result line inside `stdout`,
    which is NOT guaranteed to be pure JSON top to bottom.

    A subprocess's stdout is not a private channel that only the code we
    wrote gets to write to -- any library the child imports can, and does,
    write to it too. PyBoy is exactly such a library: `pyboy.api.screen`,
    `pyboy.plugins.screen_recorder` and `pyboy.plugins.screenshot_recorder`
    each log a "Missing dependency Pillow" WARNING line to stdout (not
    stderr) once per episode, entirely independent of anything this driver
    or `evaluate.py` does. That is legitimate behaviour on PyBoy's part, not
    a bug this driver gets to demand a fix for -- so treating the whole of
    stdout as one JSON document (the original, naive `json.loads(proc.stdout)`)
    was always wrong, and it is why every one of the 120 evaluation-matrix
    jobs failed with "Expecting value: line 1 column 1 (char 0)" the moment
    PyBoy logged anything before the result line: the parser choked on the
    FIRST line of stdout, a warning, never even reaching the JSON.

    The one guarantee this driver DOES get from `evaluate.py --json` is that
    the result dict is printed as the last thing the process does, as a
    single `print(json.dumps(...))` call with no embedded newlines (see
    `training/evaluate.py`'s `--json` branch). So rather than parse the
    whole blob, scan lines from the END backwards and take the first one
    that (a) parses as JSON at all, (b) is a JSON *object* (dict), and
    (c) carries an `arm` key -- (b)+(c) together are what distinguish an
    actual result line from any other stray JSON-shaped text a library
    might happen to emit (e.g. a bare number or list), since `arm` is a key
    only `evaluate.py`'s own result payload has any reason to contain.
    Scanning backwards, rather than taking the first JSON-shaped line found
    forwards, is what makes this robust to warnings that themselves look
    JSON-adjacent or to multiple JSON-parseable lines: the real result is
    always the LAST one written, by construction of how `evaluate.py` prints.

    Returns the parsed dict, or `None` if no line in `stdout` qualifies --
    the caller is responsible for turning that into a diagnosable failure
    (see `run_job`), since silently returning `None` here would leave the
    caller no way to tell "no result line" apart from "result line was
    itself `null`".
    """
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
    """Executes one non-dedup job to completion (or determines it's already
    done). Returns an outcome dict with at least `{"job": job, "status": ...}`
    where status is one of "skipped" (valid result already present -- see
    `_resume_check`), "ran" (subprocess succeeded, result validated and
    written -- `mean_extrinsic_return` is included), or "failed"
    (`error` plus, when available, `cmd`/`stdout`/`stderr` for the FAILURES
    report). Never raises: every failure mode the child or the validation can
    produce is caught and turned into a "failed" outcome, so one bad job
    cannot abort the ones running alongside it in the thread pool.
    """
    if _resume_check(job.output_path):
        return {"job": job, "status": "skipped"}

    env = dict(os.environ)
    # See module docstring "CONCURRENCY": stop each child from fanning out its
    # own BLAS/OpenMP thread pool on top of this driver's own --jobs cap.
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

    # See `_extract_json_result`'s docstring for why this is a line-scan and
    # not `json.loads(proc.stdout)`: PyBoy legitimately writes warning lines
    # to stdout alongside the child's JSON result line, so the whole of
    # stdout is not itself one JSON document.
    data = _extract_json_result(proc.stdout)
    if data is None:
        # Every one of the original 120 failures produced only
        # "Expecting value: line 1 column 1 (char 0)" -- utterly
        # undiagnosable without re-running the job by hand. This message is
        # built to be the opposite: exit code, the tail of stdout, and the
        # full stderr, all inline, so a future failure of this shape can be
        # root-caused straight from the run's own log.
        tail_lines = proc.stdout.splitlines()[-20:]
        tail = "\n".join(f"    {line}" for line in tail_lines) if tail_lines else "    (empty)"
        stderr_block = ("\n".join(f"    {line}" for line in proc.stderr.splitlines())
                        if proc.stderr else "    (empty)")
        error = (
            f"child exited {proc.returncode} but no line of stdout parsed as a JSON "
            f"object containing an 'arm' key (see _extract_json_result). "
            f"Last {len(tail_lines)} line(s) of stdout:\n{tail}\n"
            f"  stderr:\n{stderr_block}"
        )
        return {"job": job, "status": "failed", "cmd": cmd, "error": error,
                "stdout": proc.stdout, "stderr": proc.stderr}

    try:
        validate_result(data, job)
    except ValueError as exc:
        return {"job": job, "status": "failed", "cmd": cmd, "error": str(exc),
                "stdout": proc.stdout, "stderr": proc.stderr}

    _atomic_write_json(job.output_path, data)
    return {"job": job, "status": "ran", "mean_extrinsic_return": data["mean_extrinsic_return"]}


def run_dedup_job(job: Job) -> dict:
    """Executes one dedup ('best' == 'final') job: no subprocess, ever -- see
    module docstring "DEDUP". Copies `job.dedup_source` (the `final` job's
    already-written output) to `job.output_path`, with `_copied_from` stamped
    into the payload so the copy is permanently distinguishable from an
    independently-run measurement. If the `final` counterpart never actually
    produced a valid result (e.g. it failed), there is nothing safe to copy,
    and that is recorded as a "failed" outcome for THIS job too, rather than
    silently skipped or fabricated.
    """
    if _resume_check(job.output_path):
        return {"job": job, "status": "skipped"}

    if not os.path.isfile(job.dedup_source):
        return {"job": job, "status": "failed",
                "error": f"dedup source {job.dedup_source!r} does not exist -- the "
                         f"corresponding 'final' job must run (and succeed) first"}

    try:
        with open(job.dedup_source) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return {"job": job, "status": "failed",
                "error": f"dedup source {job.dedup_source!r} is not a valid result: {exc}"}

    try:
        validate_result(data, job)
    except ValueError as exc:
        return {"job": job, "status": "failed",
                "error": f"dedup source {job.dedup_source!r} failed validation: {exc}"}

    copied = dict(data)
    copied["_copied_from"] = job.dedup_source
    _atomic_write_json(job.output_path, copied)
    return {"job": job, "status": "copied", "mean_extrinsic_return": copied["mean_extrinsic_return"]}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _format_progress(index: int, total: int, job: Job, outcome: dict) -> str:
    """`[k/N] selection/arm/seed/regime -> ...` -- one line per completed job,
    so a log from an unattended multi-hour run is followable without waiting
    for it to finish."""
    label = _job_label(job)
    status = outcome["status"]
    if status == "ran":
        return f"[{index}/{total}] {label} -> mean_extrinsic_return={outcome['mean_extrinsic_return']:.4f}"
    if status == "copied":
        return (f"[{index}/{total}] {label} -> "
                f"mean_extrinsic_return={outcome['mean_extrinsic_return']:.4f} "
                f"(copied from final checkpoint's result -- dedup, see module docstring)")
    if status == "skipped":
        return f"[{index}/{total}] {label} -> SKIPPED (valid result already on disk)"
    if status == "failed":
        return f"[{index}/{total}] {label} -> FAILED: {outcome['error']}"
    raise ValueError(f"_format_progress: unknown status {status!r}")  # pragma: no cover


def run_matrix(jobs, python_exe: str, rom: str, episodes: int, eval_seed: int,
                max_workers: int) -> dict:
    """Runs every job in `jobs`, printing one progress line per completion,
    and returns `{"outcomes", "ran", "copied", "skipped", "failed"}` (the
    last four are sub-lists of `outcomes`, partitioned by status).

    Two phases, not one flat thread pool: primary (non-dedup) jobs first,
    concurrently, up to `max_workers` at a time; dedup jobs after, since a
    dedup job's source is its `final` counterpart's OUTPUT FILE, which for a
    non-resumed final job only exists once that job's subprocess above has
    completed. Dedup jobs themselves are pure local file I/O (no subprocess),
    so running them sequentially after the pool drains costs milliseconds,
    not minutes -- no reason to parallelise them.
    """
    primary = [j for j in jobs if j.dedup_source is None]
    dedup = [j for j in jobs if j.dedup_source is not None]
    total = len(jobs)
    outcomes = []
    completed = 0

    if primary:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_job = {
                pool.submit(run_job, j, python_exe, rom, episodes, eval_seed): j
                for j in primary
            }
            for future in concurrent.futures.as_completed(future_to_job):
                outcome = future.result()
                completed += 1
                print(_format_progress(completed, total, outcome["job"], outcome))
                outcomes.append(outcome)

    for j in dedup:
        outcome = run_dedup_job(j)
        completed += 1
        print(_format_progress(completed, total, j, outcome))
        outcomes.append(outcome)

    return {
        "outcomes": outcomes,
        "ran": [o for o in outcomes if o["status"] == "ran"],
        "copied": [o for o in outcomes if o["status"] == "copied"],
        "skipped": [o for o in outcomes if o["status"] == "skipped"],
        "failed": [o for o in outcomes if o["status"] == "failed"],
    }


def format_dry_run_lines(jobs, missing, python_exe: str, rom: str, episodes: int,
                          eval_seed: int) -> list:
    """Everything `--dry-run` prints: one line per job in matrix order, stating
    exactly what a real run would do for it (RUN the given command, SKIP
    because a valid result already exists, or COPY from the dedup source) --
    NOTHING here executes anything, including the resume check, which is a
    read-only JSON parse of files already on disk. Followed by the missing
    (arm, seed) runs (checkpoint not found) and a final job-count line.
    """
    lines = []
    total = len(jobs)
    for i, job in enumerate(jobs, start=1):
        label = _job_label(job)
        if _resume_check(job.output_path):
            lines.append(f"[{i}/{total}] {label} -> {job.output_path} :: "
                         f"SKIP (valid result already on disk)")
        elif job.dedup_source is not None:
            lines.append(f"[{i}/{total}] {label} -> {job.output_path} :: "
                         f"COPY <- {job.dedup_source} (dedup: best checkpoint == final checkpoint)")
        else:
            cmd = build_command(python_exe, job, rom, episodes, eval_seed)
            lines.append(f"[{i}/{total}] {label} -> {job.output_path} :: {shlex.join(cmd)}")

    if missing:
        lines.append("")
        lines.append(f"{len(missing)} (arm, seed) run(s) skipped entirely -- no checkpoint found:")
        for m in missing:
            lines.append(f"  selection={m['selection']} arm={m['arm']} seed={m['seed']}: {m['reason']}")

    lines.append("")
    lines.append(f"TOTAL JOBS: {total}")
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Resumable, parallel driver for the Phase 1 evaluation matrix "
                    "(selection x arm x seed x regime) over training/evaluate.py.")
    parser.add_argument("--rom", required=True, help="path to the Game Boy ROM")
    parser.add_argument("--episodes", type=int, default=30,
                        help="episodes per evaluation (default: 30)")
    parser.add_argument("--eval-seed", type=int, default=0,
                        help="base evaluation seed passed as evaluate.py's --seed "
                             "(default: 0) -- NOT the training seed, which is the "
                             "run being evaluated, encoded in the output filename")
    parser.add_argument("--results-dir", default="results",
                        help="output root; writes to {results_dir}/{selection}/... "
                             "(default: results)")
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="trained-run checkpoints for 'final'/'best' selection "
                             "(default: checkpoints)")
    parser.add_argument("--init-checkpoint-dir", default="checkpoints_init",
                        help="untrained reference checkpoints for 'init' selection "
                             "(default: checkpoints_init)")
    parser.add_argument("--jobs", type=int, default=8,
                        help="max concurrent evaluate.py subprocesses (default: 8)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print every command/action this run would take, in "
                             "order, and exit without executing or writing anything")
    parser.add_argument("--selections", default=",".join(SELECTIONS),
                        help=f"comma-separated subset of {SELECTIONS} "
                             f"(default: all three)")
    args = parser.parse_args(argv)

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
        checkpoint_dir=args.checkpoint_dir,
        init_checkpoint_dir=args.init_checkpoint_dir,
        results_dir=args.results_dir,
        selections=args.selections,
    )

    if args.dry_run:
        for line in format_dry_run_lines(jobs, missing, python_exe, args.rom,
                                         args.episodes, args.eval_seed):
            print(line)
        return 0

    report = run_matrix(jobs, python_exe, args.rom, args.episodes, args.eval_seed, args.jobs)

    print()
    print(f"{len(jobs)} total jobs: {len(report['ran'])} ran, "
         f"{len(report['copied'])} copied (dedup), "
         f"{len(report['skipped'])} skipped (already had a valid result), "
         f"{len(report['failed'])} failed")
    if missing:
        print(f"{len(missing)} (arm, seed) run(s) had no checkpoint to evaluate at all "
             f"(skipped, not failed):")
        for m in missing:
            print(f"  selection={m['selection']} arm={m['arm']} seed={m['seed']}: {m['reason']}")

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
