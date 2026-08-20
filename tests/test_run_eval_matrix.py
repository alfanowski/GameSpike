"""Tests for `scripts/run_eval_matrix.py`.

ALMOST NEVER invokes `training/evaluate.py` for real: this driver's whole job
is to launch that module as a child process, and nearly every test here
exercises that boundary through `run_eval_matrix._run_subprocess`,
monkeypatched to a plain Python stand-in that returns a
`subprocess.CompletedProcess`-shaped result without ever touching PyBoy or a
ROM. This is deliberate, not merely convenient: with 10 real training
processes saturating the machine while these tests run, spawning a real
evaluation on every test would be both slow and an unwelcome 11th CPU hog.

The ONE exception is `test_real_evaluate_subprocess_stdout_is_correctly_
extracted` in section 7, and it exists precisely BECAUSE of that rule, not
despite it: a mock defines the child's stdout shape by fiat, so a suite built
entirely on `_run_subprocess` mocks can prove this driver handles whatever
shape its authors imagined stdout would have, and nothing about whether that
imagined shape matches reality. It didn't -- PyBoy writes warning lines to
real stdout, `training.evaluate --json`'s ACTUAL output is "warnings, then
one JSON line," and every one of this suite's 15 original tests (all mocked)
stayed green while all 120 real evaluation-matrix jobs failed on exactly that
gap. One narrow, `--episodes 1`, skip-if-no-ROM-or-checkpoint integration test
is the only thing that closes it, because only a real child process's real
stdout can contradict an assumption baked into every mock in this file.

Section map: (1) job-matrix construction, including the 3x2x10x2=120 default
count and that output filenames really do round-trip through
`analysis.aggregate_results.load_eval_results` -- the integration point most
likely to silently break, since that function's naming convention and this
driver's `output_path_for` are two independent pieces of code that must agree
byte-for-byte; (2) resume logic; (3) dedup (best == final); (4) failure
handling (non-zero exit, arm mismatch), including that one bad job never
takes down the others; (5) [see below] stdout JSON extraction, both the real
end-to-end case and the synthetic PyBoy-warnings-then-JSON shape that broke
every job before the fix.
"""
import json
import os
import subprocess
import sys

import pytest

from analysis.aggregate_results import load_eval_results
from scripts import run_eval_matrix
from scripts.run_eval_matrix import (
    Job,
    build_job_matrix,
    output_path_for,
    run_dedup_job,
    run_job,
    run_matrix,
    state_reset_interval_for,
    validate_result,
    _resume_check,
)

# Verbatim from a real 2-episode `training.evaluate --json` run (see the task
# that produced this fix): PyBoy logs these three WARNING lines to STDOUT,
# not stderr, once per episode, entirely independent of anything this driver
# or evaluate.py does -- see `run_eval_matrix._extract_json_result`'s
# docstring for why that makes a naive whole-stdout `json.loads` wrong.
_PYBOY_WARNING_LINES = (
    'pyboy.api.screen               WARNING  Cannot generate screen image. '
    'Missing dependency "Pillow".',
    'pyboy.plugins.screen_recorder  WARNING  pyboy.plugins.screen_recorder: '
    'Missing dependency "Pillow". Recording disabled',
    'pyboy.plugins.screenshot_recorder WARNING  pyboy.plugins.screenshot_recorder: '
    'Missing dependency "Pillow". Screenshots disabled',
)

ROM_PATH = os.environ.get("MARIO_LAND_ROM_PATH")
_CHECKPOINT_ROOT = os.path.join(run_eval_matrix.REPO_ROOT, "checkpoints")


def _find_a_real_checkpoint():
    """Returns `(arm, path)` for the first real, already-trained checkpoint
    found under `checkpoints/`, or `None` if the tree doesn't exist or is
    empty -- e.g. a fresh clone that hasn't trained anything yet. Picking
    ANY checkpoint (not a specific arm/seed) keeps this test independent of
    which runs happen to exist on the machine it's run on.
    """
    if not os.path.isdir(_CHECKPOINT_ROOT):
        return None
    for run_name in sorted(os.listdir(_CHECKPOINT_ROOT)):
        arm = run_name.split("_seed")[0]
        if arm not in ("baseline", "reservoir"):
            continue
        run_dir = os.path.join(_CHECKPOINT_ROOT, run_name)
        if not os.path.isdir(run_dir):
            continue
        steps = sorted(
            (f for f in os.listdir(run_dir) if f.startswith("step_") and f.endswith(".pt")),
            key=lambda f: int(f[len("step_"):-len(".pt")]),
        )
        if steps:
            return arm, os.path.join(run_dir, steps[-1])
    return None


def _touch_run(checkpoint_dir, arm, seed, step=100, reward=1.0):
    """One trained-run directory with a single checkpoint and a matching
    one-line train_log.jsonl -- `select_final_checkpoint` and
    `select_best_checkpoint` both resolve to the SAME (only) checkpoint here,
    which is exactly what most of these tests want: it makes 'final' and
    'best' byte-identical, the dedup precondition, without needing a second
    checkpoint or a richer log. Only `step` and `mean_extrinsic_reward` are
    read by `select_best_checkpoint` (see that function's own source), so the
    log line carries nothing else.
    """
    run_dir = os.path.join(str(checkpoint_dir), f"{arm}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    open(os.path.join(run_dir, f"step_{step}.pt"), "a").close()
    with open(os.path.join(run_dir, "train_log.jsonl"), "w") as f:
        f.write(json.dumps({"step": step, "mean_extrinsic_reward": reward}) + "\n")
    return run_dir


def _touch_init(init_checkpoint_dir, arm, seed):
    run_dir = os.path.join(str(init_checkpoint_dir), f"{arm}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    open(os.path.join(run_dir, "step_0.pt"), "a").close()
    return run_dir


def _completed(cmd, returncode=0, payload=None, stderr=""):
    stdout = json.dumps(payload) if payload is not None else ""
    return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)


def _valid_payload(arm="baseline", mean=1.0, lengths=(10.0,)):
    return {"arm": arm, "mean_extrinsic_return": mean, "std_extrinsic_return": 0.1,
            "episode_lengths": list(lengths)}


# ---------------------------------------------------------------------------
# 1. Job-matrix construction
# ---------------------------------------------------------------------------

def _full_matrix_dirs(tmp_path):
    """Every (arm, seed) present for both trained runs and the init reference
    -- the state where the full 3x2x10x2=120 matrix is actually resolvable."""
    checkpoint_dir = tmp_path / "checkpoints"
    init_checkpoint_dir = tmp_path / "checkpoints_init"
    for arm in ("baseline", "reservoir"):
        for seed in range(10):
            _touch_run(checkpoint_dir, arm, seed, step=100, reward=1.0)
            _touch_init(init_checkpoint_dir, arm, seed)
    return checkpoint_dir, init_checkpoint_dir


def test_default_configuration_yields_120_jobs_with_correct_output_paths(tmp_path):
    checkpoint_dir, init_checkpoint_dir = _full_matrix_dirs(tmp_path)
    results_dir = tmp_path / "results"

    jobs, missing = build_job_matrix(str(checkpoint_dir), str(init_checkpoint_dir),
                                     str(results_dir))

    assert missing == []
    assert len(jobs) == 120
    # every job's output path is unique -- no two (selection, arm, seed, regime)
    # cells collide on disk.
    assert len({j.output_path for j in jobs}) == 120

    one = next(j for j in jobs if j.selection == "final" and j.arm == "baseline"
              and j.seed == 3 and j.regime == "reset128")
    assert one.output_path == str(results_dir / "final" / "eval_baseline_seed3_reset128.json")
    assert one.checkpoint_path == str(checkpoint_dir / "baseline_seed3" / "step_100.pt")

    init_job = next(j for j in jobs if j.selection == "init" and j.arm == "reservoir"
                    and j.seed == 7 and j.regime == "continuous")
    assert init_job.output_path == str(results_dir / "init" / "eval_reservoir_seed7_continuous.json")
    assert init_job.checkpoint_path == str(init_checkpoint_dir / "reservoir_seed7" / "step_0.pt")
    assert init_job.checkpoint_step == 0

    # 3 selections x 2 arms x 10 seeds x 2 regimes, each contributing exactly once.
    for selection in ("final", "best", "init"):
        assert sum(1 for j in jobs if j.selection == selection) == 40
    for arm in ("baseline", "reservoir"):
        assert sum(1 for j in jobs if j.arm == arm) == 60
    for regime in ("continuous", "reset128"):
        assert sum(1 for j in jobs if j.regime == regime) == 60


def test_missing_run_is_skipped_not_fatal(tmp_path):
    """A run directory that doesn't exist yet (mid-experiment) contributes no
    jobs and shows up in `missing` instead of crashing the whole build."""
    checkpoint_dir = tmp_path / "checkpoints"
    init_checkpoint_dir = tmp_path / "checkpoints_init"
    _touch_run(checkpoint_dir, "baseline", 0)
    _touch_init(init_checkpoint_dir, "baseline", 0)
    # reservoir_seed0 never created for either tree.

    jobs, missing = build_job_matrix(str(checkpoint_dir), str(init_checkpoint_dir),
                                     str(tmp_path / "results"),
                                     arms=("baseline", "reservoir"), seeds=(0,))
    assert {j.selection for j in jobs} == {"final", "best", "init"}
    assert all(j.arm == "baseline" for j in jobs)
    assert len(jobs) == 6  # 3 selections x 1 arm x 1 seed x 2 regimes
    assert len(missing) == 3  # reservoir_seed0 missing under each of the 3 selections


# ---------------------------------------------------------------------------
# 2. Filename round-trip through analysis.aggregate_results.load_eval_results
# ---------------------------------------------------------------------------

def test_output_filenames_round_trip_through_load_eval_results(tmp_path):
    """The exact integration point the task calls out as most likely to
    silently break: this driver's own naming (`output_path_for`) and
    `load_eval_results`'s parsing regex are two independent pieces of code
    that must agree byte-for-byte on `eval_{arm}_seed{N}_{regime}.json`.
    """
    results_dir = str(tmp_path / "results")
    cases = [
        ("final", "baseline", 3, "continuous"),
        ("best", "reservoir", 7, "reset128"),
        ("init", "baseline", 0, "reset128"),
    ]
    for selection, arm, seed, regime in cases:
        path = output_path_for(results_dir, selection, arm, seed, regime)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = _valid_payload(arm=arm)
        payload["state_reset_interval"] = state_reset_interval_for(regime)
        with open(path, "w") as f:
            json.dump(payload, f)

        records = load_eval_results(os.path.join(results_dir, selection))
        assert len(records) == 1, f"expected exactly one eval_*.json under {selection!r}"
        record = records[0]
        assert record["train_seed"] == seed
        assert record["arm"] == arm
        assert record["regime"] == regime


# ---------------------------------------------------------------------------
# 3. Resume logic
# ---------------------------------------------------------------------------

def test_resume_check_accepts_valid_and_rejects_truncated_or_missing(tmp_path):
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps(_valid_payload()))
    assert _resume_check(str(valid)) is True

    truncated = tmp_path / "truncated.json"
    truncated.write_text('{"arm": "baseline", "mean_extrinsic_ret')  # cut off mid-write
    assert _resume_check(str(truncated)) is False

    no_key = tmp_path / "no_key.json"
    no_key.write_text(json.dumps({"arm": "baseline"}))  # valid JSON, missing the required key
    assert _resume_check(str(no_key)) is False

    assert _resume_check(str(tmp_path / "does_not_exist.json")) is False


def test_run_job_skips_subprocess_for_an_already_valid_result(tmp_path, monkeypatch):
    def boom(cmd, env):
        raise AssertionError("a resumed job must never spawn a subprocess")
    monkeypatch.setattr(run_eval_matrix, "_run_subprocess", boom)

    output_path = tmp_path / "eval_baseline_seed0_continuous.json"
    output_path.write_text(json.dumps(_valid_payload(mean=3.0)))
    job = Job(selection="final", arm="baseline", seed=0, regime="continuous",
              checkpoint_path="ckpt.pt", checkpoint_step=100, output_path=str(output_path))

    outcome = run_job(job, python_exe="python", rom="rom.gb", episodes=30, eval_seed=0)
    assert outcome["status"] == "skipped"


def test_run_job_reruns_a_truncated_existing_result(tmp_path, monkeypatch):
    output_path = tmp_path / "eval_baseline_seed0_continuous.json"
    output_path.write_text('{"arm": "baseline"')  # truncated, as if killed mid-write

    calls = []

    def fake_run_subprocess(cmd, env):
        calls.append(cmd)
        return _completed(cmd, 0, _valid_payload(mean=42.0))
    monkeypatch.setattr(run_eval_matrix, "_run_subprocess", fake_run_subprocess)

    job = Job(selection="final", arm="baseline", seed=0, regime="continuous",
              checkpoint_path="ckpt.pt", checkpoint_step=100, output_path=str(output_path))
    outcome = run_job(job, python_exe="python", rom="rom.gb", episodes=30, eval_seed=0)

    assert len(calls) == 1, "a truncated existing file must trigger a real re-run"
    assert outcome["status"] == "ran"
    assert outcome["mean_extrinsic_return"] == 42.0
    with open(output_path) as f:
        assert json.load(f)["mean_extrinsic_return"] == 42.0


# ---------------------------------------------------------------------------
# 4. Dedup: best == final
# ---------------------------------------------------------------------------

def test_dedup_best_equal_final_copies_without_a_second_subprocess(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "checkpoints"
    _touch_run(checkpoint_dir, "baseline", 0, step=100, reward=1.0)
    results_dir = tmp_path / "results"

    jobs, missing = build_job_matrix(str(checkpoint_dir), str(tmp_path / "checkpoints_init"),
                                     str(results_dir), selections=("final", "best"),
                                     arms=("baseline",), seeds=(0,), regimes=("continuous",))
    assert missing == []
    assert len(jobs) == 2
    final_job = next(j for j in jobs if j.selection == "final")
    best_job = next(j for j in jobs if j.selection == "best")
    assert final_job.checkpoint_path == best_job.checkpoint_path  # same only checkpoint in the run
    assert best_job.dedup_source == final_job.output_path

    calls = []

    def fake_run_subprocess(cmd, env):
        calls.append(cmd)
        return _completed(cmd, 0, _valid_payload(arm="baseline", mean=7.5))
    monkeypatch.setattr(run_eval_matrix, "_run_subprocess", fake_run_subprocess)

    report = run_matrix(jobs, python_exe="python", rom="rom.gb", episodes=30, eval_seed=0,
                        max_workers=2)

    assert len(calls) == 1, "only the 'final' job may spawn a subprocess; 'best' must be a copy"
    assert len(report["ran"]) == 1
    assert len(report["copied"]) == 1

    with open(best_job.output_path) as f:
        best_data = json.load(f)
    assert best_data["_copied_from"] == final_job.output_path
    assert best_data["mean_extrinsic_return"] == 7.5


def test_dedup_does_not_fire_when_best_and_final_checkpoints_differ(tmp_path):
    """Sanity check on the other side of the dedup condition: two DIFFERENT
    checkpoints (best mid-run, final at the end) must NOT be deduped."""
    checkpoint_dir = tmp_path / "checkpoints"
    run_dir = os.path.join(str(checkpoint_dir), "reservoir_seed0")
    os.makedirs(run_dir, exist_ok=True)
    open(os.path.join(run_dir, "step_100.pt"), "a").close()
    open(os.path.join(run_dir, "step_200.pt"), "a").close()
    with open(os.path.join(run_dir, "train_log.jsonl"), "w") as f:
        # reward peaks at step 100 then regresses -- 'best' picks 100, 'final' picks 200.
        f.write(json.dumps({"step": 100, "mean_extrinsic_reward": 9.0}) + "\n")
        f.write(json.dumps({"step": 200, "mean_extrinsic_reward": 1.0}) + "\n")

    jobs, missing = build_job_matrix(str(checkpoint_dir), str(tmp_path / "checkpoints_init"),
                                     str(tmp_path / "results"), selections=("final", "best"),
                                     arms=("reservoir",), seeds=(0,), regimes=("continuous",))
    best_job = next(j for j in jobs if j.selection == "best")
    final_job = next(j for j in jobs if j.selection == "final")
    assert best_job.checkpoint_path != final_job.checkpoint_path
    assert best_job.dedup_source is None


def test_dedup_not_applied_when_final_is_not_in_the_requested_selections(tmp_path):
    """If the caller only asks for 'best' (no 'final' in this invocation),
    there is no final job in this run to copy from, so 'best' must run for
    real rather than silently referencing an unrelated past run's file."""
    checkpoint_dir = tmp_path / "checkpoints"
    _touch_run(checkpoint_dir, "baseline", 0, step=100, reward=1.0)

    jobs, missing = build_job_matrix(str(checkpoint_dir), str(tmp_path / "checkpoints_init"),
                                     str(tmp_path / "results"), selections=("best",),
                                     arms=("baseline",), seeds=(0,), regimes=("continuous",))
    assert len(jobs) == 1
    assert jobs[0].dedup_source is None


# ---------------------------------------------------------------------------
# 5. Failure handling: non-zero exit, keeps going, exits non-zero overall
# ---------------------------------------------------------------------------

def test_failed_child_is_recorded_and_does_not_abort_other_jobs(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "checkpoints"
    _touch_run(checkpoint_dir, "baseline", 0, step=100)
    _touch_run(checkpoint_dir, "baseline", 1, step=100)

    jobs, missing = build_job_matrix(str(checkpoint_dir), str(tmp_path / "checkpoints_init"),
                                     str(tmp_path / "results"), selections=("final",),
                                     arms=("baseline",), seeds=(0, 1), regimes=("continuous",))
    assert len(jobs) == 2

    def fake_run_subprocess(cmd, env):
        checkpoint_arg = cmd[cmd.index("--checkpoint") + 1]
        if "baseline_seed0" in checkpoint_arg:
            return _completed(cmd, 1, stderr="boom: simulated crash")
        return _completed(cmd, 0, _valid_payload(mean=9.0))
    monkeypatch.setattr(run_eval_matrix, "_run_subprocess", fake_run_subprocess)

    report = run_matrix(jobs, python_exe="python", rom="rom.gb", episodes=30, eval_seed=0,
                        max_workers=2)

    assert len(report["failed"]) == 1
    assert len(report["ran"]) == 1
    failure = report["failed"][0]
    assert "exited with status 1" in failure["error"]
    assert "boom: simulated crash" in failure["stderr"]

    # the surviving job's result really was written despite the other failing
    assert os.path.isfile(report["ran"][0]["job"].output_path)
    # the failed job never produced a (partial or otherwise) result file
    assert not os.path.isfile(failure["job"].output_path)


def test_main_exits_nonzero_when_any_job_fails(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "checkpoints"
    _touch_run(checkpoint_dir, "baseline", 0, step=100)

    def fake_run_subprocess(cmd, env):
        return _completed(cmd, 1, stderr="crash")
    monkeypatch.setattr(run_eval_matrix, "_run_subprocess", fake_run_subprocess)

    argv = [
        "--rom", "fake.gb",
        "--checkpoint-dir", str(checkpoint_dir),
        "--init-checkpoint-dir", str(tmp_path / "checkpoints_init"),
        "--results-dir", str(tmp_path / "results"),
        "--selections", "final",
        "--jobs", "2",
    ]
    assert run_eval_matrix.main(argv) == 1


def test_main_exits_zero_when_everything_succeeds(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "checkpoints"
    _touch_run(checkpoint_dir, "baseline", 0, step=100)

    def fake_run_subprocess(cmd, env):
        return _completed(cmd, 0, _valid_payload(mean=1.0))
    monkeypatch.setattr(run_eval_matrix, "_run_subprocess", fake_run_subprocess)

    argv = [
        "--rom", "fake.gb",
        "--checkpoint-dir", str(checkpoint_dir),
        "--init-checkpoint-dir", str(tmp_path / "checkpoints_init"),
        "--results-dir", str(tmp_path / "results"),
        "--selections", "final",
        "--jobs", "2",
    ]
    assert run_eval_matrix.main(argv) == 0


# ---------------------------------------------------------------------------
# 6. Arm mismatch: hard failure
# ---------------------------------------------------------------------------

def test_arm_mismatch_is_a_hard_failure_and_writes_nothing(tmp_path, monkeypatch):
    job = Job(selection="final", arm="baseline", seed=0, regime="continuous",
              checkpoint_path="ckpt.pt", checkpoint_step=100,
              output_path=str(tmp_path / "eval_baseline_seed0_continuous.json"))

    def fake_run_subprocess(cmd, env):
        return _completed(cmd, 0, _valid_payload(arm="reservoir"))  # wrong arm
    monkeypatch.setattr(run_eval_matrix, "_run_subprocess", fake_run_subprocess)

    outcome = run_job(job, python_exe="python", rom="rom.gb", episodes=30, eval_seed=0)
    assert outcome["status"] == "failed"
    assert "arm mismatch" in outcome["error"]
    assert not os.path.isfile(job.output_path)


def test_validate_result_rejects_arm_mismatch_directly():
    job = Job(selection="final", arm="baseline", seed=0, regime="continuous",
              checkpoint_path="ckpt.pt", checkpoint_step=100, output_path="unused.json")
    with pytest.raises(ValueError, match="arm mismatch"):
        validate_result(_valid_payload(arm="reservoir"), job)


def test_dedup_copy_also_enforces_arm_match(tmp_path):
    """The dedup path re-validates too -- a corrupted or mislabelled 'final'
    result must not be silently propagated into 'best' by the copy shortcut."""
    final_path = tmp_path / "eval_baseline_seed0_continuous.json"
    final_path.write_text(json.dumps(_valid_payload(arm="reservoir")))  # wrong arm

    best_job = Job(selection="best", arm="baseline", seed=0, regime="continuous",
                   checkpoint_path="ckpt.pt", checkpoint_step=100,
                   output_path=str(tmp_path / "eval_baseline_seed0_continuous_best.json"),
                   dedup_source=str(final_path))
    outcome = run_dedup_job(best_job)
    assert outcome["status"] == "failed"
    assert not os.path.isfile(best_job.output_path)


# ---------------------------------------------------------------------------
# 7. stdout JSON extraction -- the bug this whole file's task fixed:
#    `training.evaluate --json`'s stdout is not a pure JSON document (PyBoy
#    writes warning lines to it too), so the driver has to find the result
#    line, not `json.loads` the blob whole.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not ROM_PATH or not os.path.exists(ROM_PATH),
    reason="MARIO_LAND_ROM_PATH not set or file missing; set it to your own legally-dumped ROM",
)
def test_real_evaluate_subprocess_stdout_is_correctly_extracted(tmp_path):
    """THE regression guard for the entire class of bug in this task: every
    other test in this file monkeypatches `_run_subprocess`, so none of them
    can ever notice a contract mismatch between what the real child actually
    prints and what a mock was written to pretend it prints -- which is
    exactly how all 120 real evaluation-matrix jobs failed while this whole
    mocked suite stayed green. This test runs the real
    `python -m training.evaluate --json` subprocess, for real, against a real
    checkpoint and the real ROM, with `--episodes 1` to keep it fast, and
    asserts `_extract_json_result` (the driver's own extraction function,
    not a re-implementation of it in the test) pulls a valid result out of
    the real stdout.

    Skips cleanly (not a failure) on a fresh clone: no ROM configured, or no
    checkpoint yet trained -- both legitimate states this suite must still
    pass in.
    """
    if not os.path.isdir(_CHECKPOINT_ROOT):
        pytest.skip(f"no {_CHECKPOINT_ROOT!r} directory -- nothing has been trained yet")
    found = _find_a_real_checkpoint()
    if found is None:
        pytest.skip(f"no real checkpoint (step_*.pt) found under {_CHECKPOINT_ROOT!r}")
    arm, checkpoint_path = found

    job = Job(selection="final", arm=arm, seed=0, regime="continuous",
              checkpoint_path=checkpoint_path, checkpoint_step=None,
              output_path=str(tmp_path / "eval_result.json"))
    cmd = run_eval_matrix.build_command(sys.executable, job, ROM_PATH, episodes=1, eval_seed=0)

    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    # The real seam, unmocked -- see the module docstring's explanation of
    # why this is the one test in the file allowed to do this.
    proc = run_eval_matrix._run_subprocess(cmd, env)

    assert proc.returncode == 0, (
        f"real `training.evaluate` child exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    result = run_eval_matrix._extract_json_result(proc.stdout)
    assert result is not None, (
        f"_extract_json_result found no JSON result line in the real child's "
        f"stdout -- either the extractor or evaluate.py's --json output "
        f"format has regressed. Full stdout:\n{proc.stdout}"
    )
    assert result["arm"] == arm
    assert "mean_extrinsic_return" in result


def test_extract_json_result_skips_real_pyboy_warning_lines_and_finds_the_json():
    """The exact real-world stdout shape that broke every one of the original
    120 jobs: several PyBoy WARNING lines (verbatim text, see
    `_PYBOY_WARNING_LINES` above) precede the single JSON result line that
    `training.evaluate --json` prints last, once per episode -- so a
    2-episode run repeats the warning block twice before the JSON.
    """
    payload = {"arm": "baseline", "n_episodes": 2, "mean_extrinsic_return": 53.375}
    stdout = "\n".join([*_PYBOY_WARNING_LINES, *_PYBOY_WARNING_LINES, json.dumps(payload)])

    result = run_eval_matrix._extract_json_result(stdout)
    assert result == payload


def test_extract_json_result_returns_none_when_no_line_qualifies():
    """No JSON object line anywhere in stdout -- the extractor must report
    'nothing found' (`None`), not raise or fabricate a match."""
    stdout = "\n".join(_PYBOY_WARNING_LINES)
    assert run_eval_matrix._extract_json_result(stdout) is None


def test_extract_json_result_rejects_a_json_line_missing_the_arm_key():
    """A JSON object that parses fine but has no `arm` key must NOT be
    accepted -- `arm` is what tells a real result line apart from some other
    stray JSON-shaped text a library might emit; without requiring it, this
    driver could silently accept the wrong line."""
    no_arm_payload = {"n_episodes": 2, "mean_extrinsic_return": 53.375}  # no 'arm' key
    stdout = "\n".join([*_PYBOY_WARNING_LINES, json.dumps(no_arm_payload)])
    assert run_eval_matrix._extract_json_result(stdout) is None


def test_run_job_fails_with_a_diagnostic_including_stderr_when_no_json_line_is_found(
    tmp_path, monkeypatch
):
    """When stdout has no qualifying JSON line, `run_job`'s failure message
    must be self-contained: exit code, stdout tail, AND stderr, inline in
    `error`, so a future failure of this shape is debuggable from the log
    alone (see the task this fix came from -- the ORIGINAL message,
    "Expecting value: line 1 column 1 (char 0)", was not)."""
    def fake_run_subprocess(cmd, env):
        stdout = "\n".join(_PYBOY_WARNING_LINES)  # no JSON line at all
        return subprocess.CompletedProcess(
            cmd, 0, stdout=stdout,
            stderr="Traceback (most recent call last):\nRuntimeError: simulated crash",
        )
    monkeypatch.setattr(run_eval_matrix, "_run_subprocess", fake_run_subprocess)

    job = Job(selection="final", arm="baseline", seed=0, regime="continuous",
              checkpoint_path="ckpt.pt", checkpoint_step=100,
              output_path=str(tmp_path / "eval_baseline_seed0_continuous.json"))
    outcome = run_job(job, python_exe="python", rom="rom.gb", episodes=1, eval_seed=0)

    assert outcome["status"] == "failed"
    assert "RuntimeError: simulated crash" in outcome["error"]
    assert "Traceback (most recent call last)" in outcome["error"]
    for warning_line in _PYBOY_WARNING_LINES:
        assert warning_line in outcome["error"]
    assert not os.path.isfile(job.output_path)
