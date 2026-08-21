"""Tests for `scripts/run_phase2a_eval.py`.

Mirrors `tests/test_run_eval_matrix.py`'s own mocking discipline: nearly every
test exercises the `_run_subprocess` boundary through a monkeypatched stand-in,
never a real `training.evaluate` child (this suite must not start real emulation).
`test_real_evaluate_subprocess_produces_a_self_describing_result` is the one
exception, gated on both a real ROM AND a real Phase 2a checkpoint existing on
disk (neither is expected to exist yet -- this task does not train anything --
so it is expected to skip cleanly for now and is here for when that checkpoint
exists).

This driver's whole reason to exist is the cross product `training/evaluate.py`
--task alone does not give you: a checkpoint trained on ONE task scored on BOTH
tasks, which is the zero-shot transfer number Phase 2a's performance matrix
needs. Section map: (1) job-matrix construction, the cross product and missing-
checkpoint handling; (2) output filename shape (both tasks unambiguous); (3)
resume; (4) failure handling; (5) build_command; (6) --dry-run; (7) CLI parsing.
"""
import json
import os
import subprocess
import sys

import pytest

from scripts import run_phase2a_eval as rpe
from scripts.run_phase2a_eval import (
    Job,
    build_command,
    build_job_matrix,
    output_path_for,
    parse_args,
    parse_tasks,
    run_job,
    run_matrix,
    validate_result,
    _resume_check,
)
from training.train import run_dir_for

ROM_PATH = os.environ.get("MARIO_LAND_ROM_PATH")


def _touch_run(checkpoint_dir, arm, task, seed, step=64):
    """One (arm, task, seed) checkpoint directory with a single final checkpoint
    -- enough for select_final_checkpoint to resolve, nothing else needed since
    this driver never reads train_log.jsonl (no 'best' selection here)."""
    run_dir = run_dir_for(str(checkpoint_dir), arm, seed, task=task)
    os.makedirs(run_dir, exist_ok=True)
    open(os.path.join(run_dir, f"step_{step}.pt"), "a").close()
    return run_dir


def _touch_init(init_checkpoint_dir, arm, task, seed):
    run_dir = run_dir_for(str(init_checkpoint_dir), arm, seed, task=task)
    os.makedirs(run_dir, exist_ok=True)
    open(os.path.join(run_dir, "step_0.pt"), "a").close()
    return run_dir


def _completed(cmd, returncode=0, payload=None, stderr=""):
    stdout = json.dumps(payload) if payload is not None else ""
    return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)


def _valid_payload(arm="baseline", eval_task=(1, 1), mean=1.0):
    return {"arm": arm, "task": list(eval_task), "mean_extrinsic_return": mean,
            "std_extrinsic_return": 0.1, "episode_lengths": [10.0]}


# ---------------------------------------------------------------------------
# 1. Job-matrix construction: the cross product, and missing-checkpoint handling
# ---------------------------------------------------------------------------

def test_cross_product_scores_every_checkpoint_task_on_every_eval_task(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    for task in ((1, 1), (2, 1)):
        for seed in (0, 1):
            _touch_run(checkpoint_dir, "baseline", task, seed)

    jobs, missing = build_job_matrix(
        checkpoint_dir=str(checkpoint_dir), init_checkpoint_dir=str(tmp_path / "init"),
        results_dir=str(tmp_path / "results"), tasks=((1, 1), (2, 1)),
        arms=("baseline",), seeds=(0, 1), regimes=("continuous",), selections=("final",),
    )
    assert missing == []
    # 1 arm x 2 checkpoint_tasks x 2 seeds x 2 eval_tasks x 1 regime x 1 selection
    assert len(jobs) == 8
    # Every (checkpoint_task, eval_task) combination is present, including the
    # off-diagonal zero-shot-transfer cells.
    combos = {(j.checkpoint_task, j.eval_task) for j in jobs}
    assert combos == {((1, 1), (1, 1)), ((1, 1), (2, 1)),
                      ((2, 1), (1, 1)), ((2, 1), (2, 1))}
    assert len({j.output_path for j in jobs}) == 8


def test_missing_checkpoint_is_recorded_not_fatal_and_skips_its_whole_cross_product(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    _touch_run(checkpoint_dir, "baseline", (1, 1), 0)
    # (2, 1) seed 0 never trained.

    jobs, missing = build_job_matrix(
        checkpoint_dir=str(checkpoint_dir), init_checkpoint_dir=str(tmp_path / "init"),
        results_dir=str(tmp_path / "results"), tasks=((1, 1), (2, 1)),
        arms=("baseline",), seeds=(0,), regimes=("continuous",), selections=("final",),
    )
    # Only the (1,1)-trained checkpoint contributes jobs, evaluated on both tasks.
    assert len(jobs) == 2
    assert all(j.checkpoint_task == (1, 1) for j in jobs)
    assert len(missing) == 1
    assert missing[0]["checkpoint_task"] == (2, 1)
    assert missing[0]["arm"] == "baseline"
    assert missing[0]["seed"] == 0


def test_init_selection_resolves_the_untrained_reference(tmp_path):
    init_dir = tmp_path / "init"
    _touch_init(init_dir, "baseline", (1, 1), 0)

    jobs, missing = build_job_matrix(
        checkpoint_dir=str(tmp_path / "checkpoints"), init_checkpoint_dir=str(init_dir),
        results_dir=str(tmp_path / "results"), tasks=((1, 1),),
        arms=("baseline",), seeds=(0,), regimes=("continuous",), selections=("init",),
    )
    assert missing == []
    assert len(jobs) == 1
    assert jobs[0].checkpoint_step == 0
    assert jobs[0].checkpoint_path == os.path.join(
        run_dir_for(str(init_dir), "baseline", 0, task=(1, 1)), "step_0.pt")


# ---------------------------------------------------------------------------
# 2. Output filenames: both tasks unambiguous
# ---------------------------------------------------------------------------

def test_output_path_names_both_tasks_unambiguously(tmp_path):
    path = output_path_for(str(tmp_path), "final", "baseline", (1, 1), (2, 1), 3, "reset128")
    assert path == os.path.join(
        str(tmp_path), "final", "eval_baseline_ckpt1-1_on2-1_seed3_reset128.json")


def test_output_path_distinguishes_same_task_from_cross_task(tmp_path):
    """ckpt1-1_on1-1 (in-distribution) must not collide with ckpt1-1_on2-1
    (transfer) or with ckpt2-1_on1-1 (the reverse transfer direction)."""
    a = output_path_for(str(tmp_path), "final", "baseline", (1, 1), (1, 1), 0, "continuous")
    b = output_path_for(str(tmp_path), "final", "baseline", (1, 1), (2, 1), 0, "continuous")
    c = output_path_for(str(tmp_path), "final", "baseline", (2, 1), (1, 1), 0, "continuous")
    assert len({a, b, c}) == 3


# ---------------------------------------------------------------------------
# 3. Resume
# ---------------------------------------------------------------------------

def test_resume_check_accepts_valid_and_rejects_truncated_or_missing(tmp_path):
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps(_valid_payload()))
    assert _resume_check(str(valid)) is True

    truncated = tmp_path / "truncated.json"
    truncated.write_text('{"arm": "baseline"')
    assert _resume_check(str(truncated)) is False

    assert _resume_check(str(tmp_path / "missing.json")) is False


def test_run_job_skips_subprocess_for_an_already_valid_result(tmp_path, monkeypatch):
    def boom(cmd, env):
        raise AssertionError("a resumed job must never spawn a subprocess")
    monkeypatch.setattr(rpe, "_run_subprocess", boom)

    output_path = tmp_path / "eval_baseline_ckpt1-1_on2-1_seed0_continuous.json"
    output_path.write_text(json.dumps(_valid_payload()))
    job = Job(selection="final", arm="baseline", checkpoint_task=(1, 1), eval_task=(2, 1),
             seed=0, regime="continuous", checkpoint_path="ckpt.pt", checkpoint_step=64,
             output_path=str(output_path))

    outcome = run_job(job, python_exe="python", rom="rom.gb", episodes=5, eval_seed=0)
    assert outcome["status"] == "skipped"


def test_run_job_reruns_a_truncated_existing_result(tmp_path, monkeypatch):
    output_path = tmp_path / "eval_baseline_ckpt1-1_on2-1_seed0_continuous.json"
    output_path.write_text('{"arm": "baseline"')

    calls = []

    def fake_run_subprocess(cmd, env):
        calls.append(cmd)
        # eval_task=(2, 1) here mirrors what a REAL training.evaluate child
        # would report: build_command passes --task job.eval_task, so a real
        # child's own "task" key would read back as job.eval_task, not
        # job.checkpoint_task.
        return _completed(cmd, 0, _valid_payload(mean=42.0, eval_task=(2, 1)))
    monkeypatch.setattr(rpe, "_run_subprocess", fake_run_subprocess)

    job = Job(selection="final", arm="baseline", checkpoint_task=(1, 1), eval_task=(2, 1),
             seed=0, regime="continuous", checkpoint_path="ckpt.pt", checkpoint_step=64,
             output_path=str(output_path))
    outcome = run_job(job, python_exe="python", rom="rom.gb", episodes=5, eval_seed=0)

    assert len(calls) == 1
    assert outcome["status"] == "ran"
    with open(output_path) as f:
        written = json.load(f)
    assert written["mean_extrinsic_return"] == 42.0
    # Self-describing: the checkpoint's training task is stamped in, distinct
    # from evaluate.py's own "task" key (the EVAL task).
    assert written["checkpoint_task"] == [1, 1]
    assert written["task"] == [2, 1]
    assert written["training_seed"] == 0
    assert written["selection"] == "final"


# ---------------------------------------------------------------------------
# 4. Failure handling
# ---------------------------------------------------------------------------

def test_failed_child_is_recorded_and_does_not_abort_other_jobs(tmp_path, monkeypatch):
    good_path = tmp_path / "good.json"
    bad_path = tmp_path / "bad.json"

    def fake_run_subprocess(cmd, env):
        if "bad" in " ".join(cmd):
            return _completed(cmd, returncode=1, stderr="boom")
        return _completed(cmd, 0, _valid_payload(mean=7.0))
    monkeypatch.setattr(rpe, "_run_subprocess", fake_run_subprocess)

    good_job = Job(selection="final", arm="baseline", checkpoint_task=(1, 1), eval_task=(1, 1),
                   seed=0, regime="continuous", checkpoint_path="good.pt", checkpoint_step=64,
                   output_path=str(good_path))
    bad_job = Job(selection="final", arm="baseline", checkpoint_task=(1, 1), eval_task=(1, 1),
                  seed=0, regime="continuous", checkpoint_path="bad.pt", checkpoint_step=64,
                  output_path=str(bad_path))

    report = run_matrix([good_job, bad_job], python_exe="python", rom="rom.gb",
                        episodes=5, eval_seed=0, max_workers=2)
    assert len(report["ran"]) == 1
    assert len(report["failed"]) == 1
    assert report["failed"][0]["job"] is bad_job


def test_arm_mismatch_is_rejected(tmp_path):
    job = Job(selection="final", arm="baseline", checkpoint_task=(1, 1), eval_task=(1, 1),
             seed=0, regime="continuous", checkpoint_path="ckpt.pt", checkpoint_step=64,
             output_path=str(tmp_path / "out.json"))
    with pytest.raises(ValueError, match="arm"):
        validate_result(_valid_payload(arm="reservoir"), job)


# ---------------------------------------------------------------------------
# 5. build_command
# ---------------------------------------------------------------------------

def test_build_command_evaluates_on_the_eval_task_not_the_checkpoint_task():
    job = Job(selection="final", arm="baseline", checkpoint_task=(1, 1), eval_task=(2, 1),
             seed=0, regime="reset128", checkpoint_path="/ckpt/step_64.pt", checkpoint_step=64,
             output_path="out.json")
    cmd = build_command("python3", job, rom="/roms/mario.gb", episodes=30, eval_seed=5)

    def flag(name):
        return cmd[cmd.index(name) + 1]

    assert cmd[1:3] == ["-m", "training.evaluate"]
    assert flag("--arm") == "baseline"
    assert flag("--checkpoint") == "/ckpt/step_64.pt"
    assert flag("--rom") == "/roms/mario.gb"
    assert flag("--episodes") == "30"
    assert flag("--seed") == "5"
    assert flag("--task") == "2-1"  # the EVAL task, not checkpoint_task (1-1)
    assert flag("--state-reset-interval") == "128"
    assert "--json" in cmd


def test_build_command_omits_state_reset_interval_for_continuous():
    job = Job(selection="final", arm="baseline", checkpoint_task=(1, 1), eval_task=(1, 1),
             seed=0, regime="continuous", checkpoint_path="ckpt.pt", checkpoint_step=64,
             output_path="out.json")
    cmd = build_command("python3", job, rom="rom.gb", episodes=30, eval_seed=0)
    assert "--state-reset-interval" not in cmd


# ---------------------------------------------------------------------------
# 6. --dry-run
# ---------------------------------------------------------------------------

def test_dry_run_touches_nothing_and_prints_every_job(tmp_path, monkeypatch, capsys):
    checkpoint_dir = tmp_path / "checkpoints"
    _touch_run(checkpoint_dir, "baseline", (1, 1), 0)

    monkeypatch.setattr(rpe, "_run_subprocess",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("dry-run must not execute anything")))
    argv = ["--rom", "fake.gb", "--tasks", "1-1", "--arms", "baseline", "--seeds", "0",
           "--checkpoint-dir", str(checkpoint_dir), "--init-checkpoint-dir", str(tmp_path / "init"),
           "--results-dir", str(tmp_path / "results"), "--selections", "final",
           "--regimes", "continuous", "--dry-run"]
    assert rpe.main(argv) == 0
    out = capsys.readouterr().out
    assert "TOTAL JOBS: 1" in out
    assert not os.path.isdir(tmp_path / "results")


# ---------------------------------------------------------------------------
# 7. CLI parsing
# ---------------------------------------------------------------------------

def test_parse_tasks_default_and_explicit():
    assert parse_tasks("1-1,2-1") == ((1, 1), (2, 1))
    assert parse_tasks("2-1") == ((2, 1),)


def test_parse_tasks_dedups_preserving_order():
    assert parse_tasks("2-1,1-1,2-1") == ((2, 1), (1, 1))


def test_parse_tasks_rejects_unknown():
    with pytest.raises(ValueError):
        parse_tasks("1-1,9-9")


def test_parse_args_defaults():
    args = parse_args(["--rom", "fake.gb"])
    assert args.tasks == ((1, 1), (2, 1))
    # Phase 2 runs the GRU only (docs/DESIGN_ROADMAP_PHASE2.md §10) -- the
    # default must reflect that, not silently also schedule reservoir jobs.
    assert args.arms == ("baseline",)
    assert args.selections == ("final", "init")
    assert set(args.regimes) == {"continuous", "reset128"}


def test_parse_args_rejects_bad_task():
    with pytest.raises(SystemExit):
        parse_args(["--rom", "fake.gb", "--tasks", "9-9"])


def test_parse_args_rejects_bad_selection():
    with pytest.raises(SystemExit):
        parse_args(["--rom", "fake.gb", "--selections", "best"])


def test_parse_args_rejects_bad_regime():
    with pytest.raises(SystemExit):
        parse_args(["--rom", "fake.gb", "--regimes", "reset999"])


def test_parse_args_arms_can_include_reservoir_explicitly():
    """Not hard-removed -- an operator who DOES have reservoir checkpoints for
    Phase 2a can still ask for them; the default just doesn't assume it."""
    args = parse_args(["--rom", "fake.gb", "--arms", "baseline,reservoir"])
    assert args.arms == ("baseline", "reservoir")


# ---------------------------------------------------------------------------
# 8. Real subprocess (skips cleanly: no Phase 2a checkpoint exists on disk yet)
# ---------------------------------------------------------------------------

def test_real_evaluate_subprocess_produces_a_self_describing_result(tmp_path):
    checkpoint_root = os.path.join(rpe.REPO_ROOT, "checkpoints_phase2a")
    run_dir = run_dir_for(checkpoint_root, "baseline", 0, task=(1, 1))
    if not ROM_PATH or not os.path.exists(ROM_PATH) or not os.path.isdir(run_dir):
        pytest.skip("no real ROM and/or Phase 2a checkpoint available yet")

    jobs, _missing = build_job_matrix(
        checkpoint_dir=checkpoint_root, init_checkpoint_dir=checkpoint_root,
        results_dir=str(tmp_path / "results"), tasks=((1, 1), (2, 1)),
        arms=("baseline",), seeds=(0,), regimes=("continuous",), selections=("final",),
    )
    job = next(j for j in jobs if j.eval_task == (2, 1))
    outcome = run_job(job, sys.executable, ROM_PATH, episodes=1, eval_seed=0)
    assert outcome["status"] == "ran"
