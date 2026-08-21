"""Tests for `scripts/run_training_matrix.py`.

NEVER invokes `training/train.py` for real -- not even `--steps 0`. Every test
that exercises job execution goes through `run_training_matrix._run_subprocess`,
monkeypatched to a plain Python stand-in, mirroring exactly how
`tests/test_run_eval_matrix.py` isolates its own subprocess boundary (see that
file's module docstring). This is stricter than that file's own rule (which
allows ONE real-subprocess test gated on a real ROM/checkpoint): the task this
suite verifies explicitly forbids starting any real training run, short or
not, so there is no exception here. The one test that does exercise a REAL
`_run_subprocess` call (`test_real_run_subprocess_streams_output_to_log_file`)
runs a trivial `python -c` one-liner, never `training.train`, purely to prove
the stdout/stderr-to-log-file redirection plumbing actually works end to end.

Section map: (1) --arms/--seeds parsing; (2) final-step derivation, including
the 1,000,064 trap from docs/EXPERIMENT_LOG.md §2; (3) job-matrix construction
(run_dir/final_checkpoint_path/log_path correctness, --run-tag); (4) resume
guard (skip a complete run without spawning a subprocess); (5) skip-if-
in-progress guard (refuse an incomplete run dir by default; --restart-
incomplete overrides it); (6) the single-instance lock (mkdir atomicity,
stale-lock PID diagnosis, --force-unlock, signal-triggered release);
(7) concurrency/failure handling (one bad job doesn't abort the others, exit
code reflects any failure); (8) --dry-run (correct RUN/SKIP/REFUSE lines,
nothing executed, nothing written); (9) command construction; (10) the real
subprocess/log-file redirection plumbing (the one unmocked test, see above).
"""
import json
import os
import signal
import subprocess
import sys

import pytest

from scripts import run_training_matrix as rtm
from scripts.run_training_matrix import (
    Job,
    RunConfig,
    LockHeld,
    acquire_lock,
    build_command,
    build_job_matrix,
    final_step_for,
    force_unlock,
    parse_arms,
    parse_seeds,
    release_lock,
    resume_status,
    run_job,
    run_matrix,
    format_dry_run_lines,
)


def _config(tmp_path, **overrides):
    defaults = dict(
        rom="fake.gb",
        steps=1_000_000,
        checkpoint_every=100_000,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        grad_clip_mode="global",
        embed_init_mode="legacy",
        embed_scale=1.0,
        run_tag=None,
    )
    defaults.update(overrides)
    return RunConfig(**defaults)


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "a").close()


# ---------------------------------------------------------------------------
# 1. --arms / --seeds parsing
# ---------------------------------------------------------------------------

def test_parse_arms_default_both_in_canonical_order():
    assert parse_arms("baseline,reservoir") == ("baseline", "reservoir")
    # canonical order regardless of input order
    assert parse_arms("reservoir,baseline") == ("baseline", "reservoir")


def test_parse_arms_single():
    assert parse_arms("reservoir") == ("reservoir",)


def test_parse_arms_rejects_unknown():
    with pytest.raises(ValueError, match="unknown arm"):
        parse_arms("baseline,rogue")


def test_parse_seeds_range():
    assert parse_seeds("0-9") == tuple(range(10))


def test_parse_seeds_comma_list():
    assert parse_seeds("1,3,5") == (1, 3, 5)


def test_parse_seeds_mixed_range_and_list_deduped_and_sorted():
    assert parse_seeds("5,0-2,2") == (0, 1, 2, 5)


def test_parse_seeds_rejects_garbage():
    with pytest.raises(ValueError):
        parse_seeds("abc")


def test_parse_seeds_rejects_backwards_range():
    with pytest.raises(ValueError):
        parse_seeds("9-0")


# ---------------------------------------------------------------------------
# 2. final_step_for -- the docs/EXPERIMENT_LOG.md §2 trap
# ---------------------------------------------------------------------------

def test_final_step_for_the_documented_1000064_trap():
    """--steps 1000000 --rollout-len 128 (train.py's own default) must land on
    step_1000064.pt, not step_1000000.pt -- see docs/EXPERIMENT_LOG.md §2."""
    assert final_step_for(1_000_000, 128) == 1_000_064


def test_final_step_for_exact_multiple_stays_put():
    """When total_steps is already an exact multiple of rollout_len, the loop
    (`while step < total_steps: step += rollout_len`) stops exactly there --
    it does not overshoot by one more rollout_len."""
    assert final_step_for(1280, 128) == 1280


def test_final_step_for_zero_steps_is_the_untrained_reference_case():
    """--steps 0 (checkpoints_init/) never enters the loop body at all."""
    assert final_step_for(0, 128) == 0


def test_final_step_for_small_remainder():
    # ceil(100000/128) = 782 (since 781*128=99968 < 100000, 782*128=100096) --
    # this is also the first --checkpoint-every 100000 boundary documented in
    # docs/EXPERIMENT_LOG.md §2 (step_100096.pt).
    assert final_step_for(100_000, 128) == 100_096


def test_final_step_for_rejects_nonpositive_rollout_len():
    with pytest.raises(ValueError):
        final_step_for(1_000_000, 0)


# ---------------------------------------------------------------------------
# 3. Job-matrix construction
# ---------------------------------------------------------------------------

def test_build_job_matrix_default_yields_20_jobs(tmp_path):
    config = _config(tmp_path)
    jobs = build_job_matrix(config, arms=("baseline", "reservoir"), seeds=tuple(range(10)))
    assert len(jobs) == 20
    assert len({j.run_dir for j in jobs}) == 20
    for arm in ("baseline", "reservoir"):
        assert sum(1 for j in jobs if j.arm == arm) == 10


def test_build_job_matrix_run_dir_and_final_checkpoint_naming(tmp_path):
    config = _config(tmp_path, steps=1_000_000)
    jobs = build_job_matrix(config, arms=("reservoir",), seeds=(3,))
    job = jobs[0]
    assert job.run_dir == os.path.join(str(tmp_path / "checkpoints"), "reservoir_seed3")
    assert job.final_step == 1_000_064
    assert job.final_checkpoint_path == os.path.join(job.run_dir, "step_1000064.pt")
    assert job.log_path == os.path.join(job.run_dir, "launcher.log")


def test_build_job_matrix_honours_run_tag(tmp_path):
    """--run-tag changes the run directory the same way train.py's own
    run_dir_for does -- this launcher must not reimplement that naming, it
    imports run_dir_for from training.train (see module docstring)."""
    config = _config(tmp_path, run_tag="per-group")
    jobs = build_job_matrix(config, arms=("baseline",), seeds=(0,))
    assert jobs[0].run_dir == os.path.join(
        str(tmp_path / "checkpoints"), "baseline_seed0_per-group")


def test_build_job_matrix_arm_seed_major_minor_order(tmp_path):
    config = _config(tmp_path)
    jobs = build_job_matrix(config, arms=("reservoir", "baseline"), seeds=(1, 0))
    # canonical arm order (baseline, reservoir) x ascending seed order,
    # regardless of the order arms/seeds were passed in.
    assert [(j.arm, j.seed) for j in jobs] == [
        ("baseline", 0), ("baseline", 1), ("reservoir", 0), ("reservoir", 1),
    ]


# ---------------------------------------------------------------------------
# 4. Resume guard
# ---------------------------------------------------------------------------

def test_resume_status_absent_when_run_dir_does_not_exist(tmp_path):
    config = _config(tmp_path)
    job = build_job_matrix(config, arms=("baseline",), seeds=(0,))[0]
    assert resume_status(job) == "absent"


def test_resume_status_complete_when_final_checkpoint_present(tmp_path):
    config = _config(tmp_path, steps=100_000)
    job = build_job_matrix(config, arms=("baseline",), seeds=(0,))[0]
    _touch(job.final_checkpoint_path)
    assert resume_status(job) == "complete"


def test_resume_status_incomplete_when_dir_exists_without_final_checkpoint(tmp_path):
    config = _config(tmp_path, steps=100_000)
    job = build_job_matrix(config, arms=("baseline",), seeds=(0,))[0]
    os.makedirs(job.run_dir)
    _touch(os.path.join(job.run_dir, "step_50048.pt"))  # an earlier, non-final checkpoint
    assert resume_status(job) == "incomplete"


def test_run_job_skips_subprocess_for_a_complete_run(tmp_path, monkeypatch):
    def boom(cmd, env, log_path):
        raise AssertionError("a completed run must never spawn a subprocess")
    monkeypatch.setattr(rtm, "_run_subprocess", boom)

    config = _config(tmp_path, steps=100_000)
    job = build_job_matrix(config, arms=("baseline",), seeds=(0,))[0]
    _touch(job.final_checkpoint_path)

    outcome = run_job(job, 1, 1, "python", config, restart_incomplete=False)
    assert outcome["status"] == "skipped"


# ---------------------------------------------------------------------------
# 5. Skip-if-in-progress guard
# ---------------------------------------------------------------------------

def test_run_job_refuses_an_incomplete_run_dir_by_default(tmp_path, monkeypatch):
    def boom(cmd, env, log_path):
        raise AssertionError("a refused job must never spawn a subprocess")
    monkeypatch.setattr(rtm, "_run_subprocess", boom)

    config = _config(tmp_path, steps=100_000)
    job = build_job_matrix(config, arms=("baseline",), seeds=(0,))[0]
    os.makedirs(job.run_dir)
    _touch(os.path.join(job.run_dir, "step_50048.pt"))

    outcome = run_job(job, 1, 1, "python", config, restart_incomplete=False)
    assert outcome["status"] == "failed"
    assert "--restart-incomplete" in outcome["error"]
    # nothing was deleted
    assert os.path.isfile(os.path.join(job.run_dir, "step_50048.pt"))


def test_run_job_restart_incomplete_deletes_and_reruns(tmp_path, monkeypatch):
    config = _config(tmp_path, steps=100_000)
    job = build_job_matrix(config, arms=("baseline",), seeds=(0,))[0]
    os.makedirs(job.run_dir)
    stale_file = os.path.join(job.run_dir, "step_50048.pt")
    _touch(stale_file)

    calls = []

    def fake_run_subprocess(cmd, env, log_path):
        calls.append((cmd, log_path))
        # simulate train.py actually finishing: it writes the final checkpoint
        _touch(job.final_checkpoint_path)
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(rtm, "_run_subprocess", fake_run_subprocess)

    outcome = run_job(job, 1, 1, "python", config, restart_incomplete=True)
    assert outcome["status"] == "ran"
    assert len(calls) == 1
    # the stale, non-final checkpoint from the old incomplete run is gone
    assert not os.path.isfile(stale_file)
    assert os.path.isfile(job.final_checkpoint_path)


# ---------------------------------------------------------------------------
# 6. Single-instance lock
# ---------------------------------------------------------------------------

def test_acquire_lock_then_second_acquire_raises_lock_held(tmp_path):
    lock_dir = str(tmp_path / "lock")
    acquire_lock(lock_dir)
    try:
        with pytest.raises(LockHeld):
            acquire_lock(lock_dir)
    finally:
        release_lock(lock_dir)


def test_lock_records_owning_pid_for_diagnosis(tmp_path):
    lock_dir = str(tmp_path / "lock")
    acquire_lock(lock_dir)
    try:
        with open(os.path.join(lock_dir, "pid")) as f:
            recorded = f.read().strip()
        assert recorded == str(os.getpid())
    finally:
        release_lock(lock_dir)


def test_release_lock_allows_reacquisition(tmp_path):
    lock_dir = str(tmp_path / "lock")
    acquire_lock(lock_dir)
    release_lock(lock_dir)
    acquire_lock(lock_dir)  # must not raise
    release_lock(lock_dir)


def test_force_unlock_removes_a_stale_lock(tmp_path):
    lock_dir = str(tmp_path / "lock")
    acquire_lock(lock_dir)
    assert os.path.isdir(lock_dir)
    force_unlock(lock_dir)
    assert not os.path.isdir(lock_dir)
    acquire_lock(lock_dir)  # proves it's really gone
    release_lock(lock_dir)


def test_force_unlock_is_a_noop_when_no_lock_exists(tmp_path):
    lock_dir = str(tmp_path / "lock")
    force_unlock(lock_dir)  # must not raise
    assert not os.path.isdir(lock_dir)


def test_main_refuses_to_run_when_lock_already_held(tmp_path, monkeypatch, capsys):
    checkpoint_dir = tmp_path / "checkpoints"
    lock_dir = tmp_path / "lock"
    os.makedirs(lock_dir)  # simulate another live instance
    with open(os.path.join(lock_dir, "pid"), "w") as f:
        f.write("999999")

    def boom(cmd, env, log_path):
        raise AssertionError("must never spawn a subprocess when the lock is held")
    monkeypatch.setattr(rtm, "_run_subprocess", boom)

    argv = [
        "--rom", "fake.gb",
        "--checkpoint-dir", str(checkpoint_dir),
        "--lock-dir", str(lock_dir),
        "--arms", "baseline", "--seeds", "0",
    ]
    rc = rtm.main(argv)
    assert rc == 1
    captured = capsys.readouterr()
    assert "999999" in (captured.out + captured.err)
    # the pre-existing lock must be left alone (not owned by us, not deleted)
    assert os.path.isdir(lock_dir)


def test_signal_handler_releases_lock_before_exiting(tmp_path):
    lock_dir = str(tmp_path / "lock")
    acquire_lock(lock_dir)
    orig_sigint = signal.getsignal(signal.SIGINT)
    orig_sigterm = signal.getsignal(signal.SIGTERM)
    rtm._install_signal_handlers(lock_dir)
    try:
        handler = signal.getsignal(signal.SIGTERM)
        with pytest.raises(SystemExit):
            handler(signal.SIGTERM, None)
    finally:
        signal.signal(signal.SIGINT, orig_sigint)
        signal.signal(signal.SIGTERM, orig_sigterm)
        if os.path.isdir(lock_dir):
            release_lock(lock_dir)
    assert not os.path.isdir(lock_dir)


# ---------------------------------------------------------------------------
# 7. Concurrency / failure handling
# ---------------------------------------------------------------------------

def test_failed_job_does_not_abort_the_others_and_report_reflects_both(tmp_path, monkeypatch):
    config = _config(tmp_path, steps=100_000)
    jobs = build_job_matrix(config, arms=("baseline",), seeds=(0, 1))

    def fake_run_subprocess(cmd, env, log_path):
        if "seed0" in log_path:
            with open(log_path, "a") as f:
                f.write("boom: simulated crash\n")
            return subprocess.CompletedProcess(cmd, 1)
        # seed1 succeeds: simulate train.py writing the final checkpoint
        job1 = next(j for j in jobs if j.seed == 1)
        _touch(job1.final_checkpoint_path)
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(rtm, "_run_subprocess", fake_run_subprocess)

    report = run_matrix(jobs, "python", config, max_workers=2, restart_incomplete=False)
    assert len(report["failed"]) == 1
    assert len(report["ran"]) == 1
    assert report["failed"][0]["job"].seed == 0
    assert report["ran"][0]["job"].seed == 1


def test_run_job_fails_when_subprocess_exits_zero_but_no_final_checkpoint_appears(
    tmp_path, monkeypatch
):
    """A returncode of 0 alone is not proof of success -- the artifact
    (the final checkpoint) must actually exist, mirroring how
    run_eval_matrix.validate_result never trusts a bare exit code either."""
    config = _config(tmp_path, steps=100_000)
    job = build_job_matrix(config, arms=("baseline",), seeds=(0,))[0]

    def fake_run_subprocess(cmd, env, log_path):
        return subprocess.CompletedProcess(cmd, 0)  # exits clean, writes nothing
    monkeypatch.setattr(rtm, "_run_subprocess", fake_run_subprocess)

    outcome = run_job(job, 1, 1, "python", config, restart_incomplete=False)
    assert outcome["status"] == "failed"
    assert "final checkpoint" in outcome["error"] or job.final_checkpoint_path in outcome["error"]


def test_main_exits_nonzero_when_any_job_fails(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "checkpoints"

    def fake_run_subprocess(cmd, env, log_path):
        return subprocess.CompletedProcess(cmd, 1)
    monkeypatch.setattr(rtm, "_run_subprocess", fake_run_subprocess)

    argv = [
        "--rom", "fake.gb",
        "--checkpoint-dir", str(checkpoint_dir),
        "--lock-dir", str(tmp_path / "lock"),
        "--arms", "baseline", "--seeds", "0",
        "--steps", "100000",
        "--jobs", "2",
    ]
    assert rtm.main(argv) == 1


def test_main_exits_zero_when_everything_succeeds(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "checkpoints"

    def fake_run_subprocess(cmd, env, log_path):
        run_dir = os.path.dirname(log_path)
        _touch(os.path.join(run_dir, "step_100096.pt"))  # final_step_for(100000,128)
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(rtm, "_run_subprocess", fake_run_subprocess)

    argv = [
        "--rom", "fake.gb",
        "--checkpoint-dir", str(checkpoint_dir),
        "--lock-dir", str(tmp_path / "lock"),
        "--arms", "baseline", "--seeds", "0",
        "--steps", "100000",
        "--jobs", "2",
    ]
    assert rtm.main(argv) == 0
    # the lock must be released after a clean run
    assert not os.path.isdir(str(tmp_path / "lock"))


# ---------------------------------------------------------------------------
# 8. --dry-run
# ---------------------------------------------------------------------------

def test_dry_run_prints_commands_and_touches_nothing(tmp_path, monkeypatch, capsys):
    def boom(cmd, env, log_path):
        raise AssertionError("--dry-run must never spawn a subprocess")
    monkeypatch.setattr(rtm, "_run_subprocess", boom)

    checkpoint_dir = tmp_path / "checkpoints"
    argv = [
        "--dry-run",
        "--arms", "reservoir", "--seeds", "0-1",
        "--rom", "/some/rom.gb",
        "--steps", "1000000", "--checkpoint-every", "100000",
        "--checkpoint-dir", str(checkpoint_dir),
        "--grad-clip-mode", "per-group", "--embed-init-mode", "centered",
        "--embed-scale", "3.0",
        "--jobs", "10",
    ]
    rc = rtm.main(argv)
    assert rc == 0

    out = capsys.readouterr().out
    assert "reservoir_seed0" in out
    assert "reservoir_seed1" in out
    assert "step_1000064.pt" in out or "1000064" in out
    assert "--grad-clip-mode" in out and "per-group" in out
    assert "--embed-scale" in out and "3.0" in out
    # nothing on disk was created
    assert not os.path.isdir(checkpoint_dir)
    # no lock was left behind
    assert not os.path.isdir(str(tmp_path / ".run_training_matrix.lock"))


def test_dry_run_reports_skip_and_refuse_correctly(tmp_path, capsys):
    checkpoint_dir = tmp_path / "checkpoints"
    config = _config(tmp_path, steps=100_000, checkpoint_dir=str(checkpoint_dir))
    complete_job, incomplete_job = build_job_matrix(
        config, arms=("baseline",), seeds=(0, 1))
    _touch(complete_job.final_checkpoint_path)
    os.makedirs(incomplete_job.run_dir)
    _touch(os.path.join(incomplete_job.run_dir, "step_50048.pt"))

    argv = [
        "--dry-run",
        "--arms", "baseline", "--seeds", "0,1",
        "--rom", "/some/rom.gb", "--steps", "100000",
        "--checkpoint-dir", str(checkpoint_dir),
    ]
    rtm.main(argv)
    out = capsys.readouterr().out
    assert "SKIP" in out
    assert "REFUSE" in out
    assert "--restart-incomplete" in out


def test_format_dry_run_lines_shows_restart_when_flag_given(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    config = _config(tmp_path, steps=100_000, checkpoint_dir=str(checkpoint_dir))
    job = build_job_matrix(config, arms=("baseline",), seeds=(0,))[0]
    os.makedirs(job.run_dir)
    _touch(os.path.join(job.run_dir, "step_50048.pt"))

    lines_no_restart = format_dry_run_lines([job], "python", config, restart_incomplete=False)
    lines_restart = format_dry_run_lines([job], "python", config, restart_incomplete=True)
    assert any("REFUSE" in l for l in lines_no_restart)
    assert not any("REFUSE" in l for l in lines_restart)
    assert any("RUN" in l for l in lines_restart)


# ---------------------------------------------------------------------------
# 9. Command construction
# ---------------------------------------------------------------------------

def test_build_command_includes_every_passthrough_flag(tmp_path):
    config = _config(
        tmp_path, rom="/roms/mario.gb", steps=1_000_000, checkpoint_every=100_000,
        checkpoint_dir=str(tmp_path / "checkpoints"), grad_clip_mode="per-group",
        embed_init_mode="centered", embed_scale=3.0, run_tag=None,
    )
    job = build_job_matrix(config, arms=("reservoir",), seeds=(7,))[0]
    cmd = build_command("python3", job, config)

    assert cmd[0] == "python3"
    assert cmd[1:3] == ["-m", "training.train"]

    def flag(name):
        return cmd[cmd.index(name) + 1]

    assert flag("--arm") == "reservoir"
    assert flag("--rom") == "/roms/mario.gb"
    assert flag("--steps") == "1000000"
    assert flag("--checkpoint-every") == "100000"
    assert flag("--checkpoint-dir") == str(tmp_path / "checkpoints")
    assert flag("--seed") == "7"
    assert flag("--grad-clip-mode") == "per-group"
    assert flag("--embed-init-mode") == "centered"
    assert flag("--embed-scale") == "3.0"
    assert "--run-tag" not in cmd


def test_build_command_includes_run_tag_only_when_given(tmp_path):
    config = _config(tmp_path, run_tag="per-group")
    job = build_job_matrix(config, arms=("baseline",), seeds=(0,))[0]
    cmd = build_command("python3", job, config)
    assert "--run-tag" in cmd
    assert cmd[cmd.index("--run-tag") + 1] == "per-group"


# ---------------------------------------------------------------------------
# 10. Real subprocess / log-file redirection (the one unmocked test)
# ---------------------------------------------------------------------------

def test_real_run_subprocess_streams_output_to_log_file(tmp_path):
    """The one test in this file that calls the REAL `_run_subprocess` --
    never against training.train (forbidden by this task), just a trivial
    `python -c` one-liner, to prove stdout+stderr really land in the log
    file, appended, rather than being buffered in memory for the whole
    process lifetime (which would lose everything if a multi-hour job were
    killed before it returned)."""
    log_path = str(tmp_path / "run_dir" / "launcher.log")
    cmd = [sys.executable, "-c",
           "import sys; print('hello stdout'); print('hello stderr', file=sys.stderr)"]
    env = dict(os.environ)

    proc = rtm._run_subprocess(cmd, env, log_path)
    assert proc.returncode == 0

    with open(log_path) as f:
        content = f.read()
    assert "hello stdout" in content
    assert "hello stderr" in content

    # a second call appends rather than truncating
    proc2 = rtm._run_subprocess(cmd, env, log_path)
    assert proc2.returncode == 0
    with open(log_path) as f:
        content2 = f.read()
    assert content2.count("hello stdout") == 2
