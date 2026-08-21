"""Phase 2a's task axis: everything that does NOT need a real ROM.

`run_dir_for`'s task argument, the checkpoint dict's `task` key (including
backward compatibility with the pre-Phase-2a checkpoint format), `OBS_MEAN_PHASE2A`,
and `--task` argument parsing for training/train.py and training/evaluate.py. All
of `build_model`/`save_checkpoint`/`load_checkpoint` are pure PyTorch (no envs/
import, no PyBoy), so this file needs no ROM and is not skipped without one --
mirroring tests/test_grad_clip_modes.py's own split between ROM-free and
ROM-gated tests in one file.

Boot- and env-level task behaviour (does world_level=(2, 1) actually reach 2-1) is
covered separately in tests/test_boot.py and tests/test_mario_land_env_world_level.py,
both of which DO require a ROM.
"""
import os

import pytest
import torch

from envs.mario_land_env import OBS_DIM, OBS_MEAN, OBS_MEAN_PHASE2A
from training.evaluate import parse_task as evaluate_parse_task
from training.train import (
    TASKS,
    build_model,
    format_task,
    load_checkpoint,
    parse_task,
    run_dir_for,
    save_checkpoint,
)

# --------------------------------------------------------------------------- #
# 1. run_dir_for's task argument
# --------------------------------------------------------------------------- #

def test_run_dir_for_without_a_task_is_byte_identical_to_the_historical_path():
    """task=None (the default) must leave every existing call site's output
    unchanged -- Phase 1's entire on-disk layout depends on this."""
    assert run_dir_for("checkpoints", "baseline", 0) == os.path.join(
        "checkpoints", "baseline_seed0")
    assert run_dir_for("checkpoints", "baseline", 0, task=None) == os.path.join(
        "checkpoints", "baseline_seed0")
    assert run_dir_for("checkpoints", "baseline", 0, None, None) == os.path.join(
        "checkpoints", "baseline_seed0")


def test_run_dir_for_with_a_task_includes_it_in_the_directory_name():
    assert run_dir_for("checkpoints", "baseline", 0, task=(1, 1)) == os.path.join(
        "checkpoints", "baseline_task1-1_seed0")
    assert run_dir_for("checkpoints", "baseline", 3, task=(2, 1)) == os.path.join(
        "checkpoints", "baseline_task2-1_seed3")


def test_two_different_tasks_at_the_same_seed_never_collide():
    """The failure mode the brief calls out as the worst one here -- see
    docs/EXPERIMENT_LOG.md §19.4 for the precedent of a guard that passed falsely
    on exactly this class of directory-naming bug."""
    dir_1_1 = run_dir_for("checkpoints", "baseline", 0, task=(1, 1))
    dir_2_1 = run_dir_for("checkpoints", "baseline", 0, task=(2, 1))
    dir_none = run_dir_for("checkpoints", "baseline", 0, task=None)
    assert len({dir_1_1, dir_2_1, dir_none}) == 3, (
        f"task-labelled run directories collided: {dir_1_1!r}, {dir_2_1!r}, {dir_none!r}"
    )
    # And neither task-labelled directory is a sub/parent path of the other or of
    # the task-less one -- a naive prefix glob (the exact §19.4 failure class)
    # must not be able to match across tasks either.
    for a, b in ((dir_1_1, dir_2_1), (dir_1_1, dir_none), (dir_2_1, dir_none)):
        assert not a.startswith(b + os.sep) and not b.startswith(a + os.sep)


def test_task_and_run_tag_compose_rather_than_collide():
    tagged_1_1 = run_dir_for("checkpoints", "baseline", 0, run_tag="clip", task=(1, 1))
    tagged_2_1 = run_dir_for("checkpoints", "baseline", 0, run_tag="clip", task=(2, 1))
    untagged_1_1 = run_dir_for("checkpoints", "baseline", 0, task=(1, 1))
    assert os.path.basename(tagged_1_1) == "baseline_task1-1_seed0_clip"
    assert len({tagged_1_1, tagged_2_1, untagged_1_1}) == 3


def test_run_dir_for_task_positional_run_tag_still_works():
    """run_tag stays the 4th POSITIONAL argument -- every existing caller that
    passes it positionally (training/train.py, scripts/run_training_matrix.py)
    must keep working unchanged after task is added."""
    assert run_dir_for("checkpoints", "reservoir", 0, "per-group") == os.path.join(
        "checkpoints", "reservoir_seed0_per-group")


# --------------------------------------------------------------------------- #
# 2. Checkpoint task round-trip + backward compatibility
# --------------------------------------------------------------------------- #

def test_checkpoint_records_the_task(tmp_path):
    model, optimizer = build_model("baseline")
    model._task = (2, 1)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, optimizer, step=0, path=str(path))

    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    assert ckpt["task"] == (2, 1)

    # And the round trip is enforceable on load, not just readable off the dict.
    model2, optimizer2 = build_model("baseline")
    restored_step = load_checkpoint(model2, optimizer2, path=str(path), expected_task=(2, 1))
    assert restored_step == 0


def test_checkpoint_without_a_task_records_none(tmp_path):
    """The default -- a checkpoint from a task-less run must record task=None,
    not omit the key or record some other sentinel."""
    model, optimizer = build_model("baseline")
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, optimizer, step=0, path=str(path))
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    assert ckpt["task"] is None


def test_expected_task_mismatch_is_rejected(tmp_path):
    model, optimizer = build_model("baseline")
    model._task = (1, 1)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, optimizer, step=0, path=str(path))

    model2, optimizer2 = build_model("baseline")
    with pytest.raises(ValueError, match="task"):
        load_checkpoint(model2, optimizer2, path=str(path), expected_task=(2, 1))


def test_an_old_format_checkpoint_without_a_task_key_still_loads(tmp_path):
    """Backward compatibility: a checkpoint written before `task` existed has no
    such key at all (not even task=None) -- this must not become unloadable."""
    model, optimizer = build_model("baseline")
    path = tmp_path / "old_ckpt.pt"
    # Simulates a pre-Phase-2a checkpoint: no 'task' key whatsoever, exactly the
    # shape training/train.py's save_checkpoint produced before this change.
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "step": 5, "arm": "baseline", "seed": 0}, str(path))

    model2, optimizer2 = build_model("baseline")
    step = load_checkpoint(model2, optimizer2, path=str(path))
    assert step == 5
    # expected_task=None is the "unchecked" sentinel, exactly like
    # expected_arm/expected_seed's existing None default -- so it must not
    # spuriously reject an old checkpoint that has no 'task' key at all.
    load_checkpoint(model2, optimizer2, path=str(path), expected_task=None)


# --------------------------------------------------------------------------- #
# 3. OBS_MEAN_PHASE2A
# --------------------------------------------------------------------------- #

def test_obs_mean_phase2a_has_one_entry_per_observation_slot():
    assert len(OBS_MEAN_PHASE2A) == OBS_DIM


def test_obs_mean_phase2a_is_distinct_from_obs_mean_and_leaves_it_untouched():
    """OBS_MEAN (1-1 only) must stay exactly what it was -- Phase 1's centered-init
    numbers depend on it. OBS_MEAN_PHASE2A is a genuinely different measurement,
    not a copy or an alias."""
    assert OBS_MEAN == (
        0.006229, 0.831424, 0.001833, -0.000167, 0.136833, 0.755903,
        0.193796, 0.111167, 0.000400, 0.000000, 0.000000, 0.000000,
    )
    assert OBS_MEAN_PHASE2A != OBS_MEAN


def test_obs_mean_phase2a_reserved_slots_are_zero():
    """Slots 9-11 are documented reserved-zero slots for every task -- a mixture
    over any task set must still show exactly zero there."""
    assert tuple(OBS_MEAN_PHASE2A[9:]) == (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------- #
# 4. --task argument parsing (training/train.py and training/evaluate.py)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("spec,expected", [("1-1", (1, 1)), ("2-1", (2, 1))])
def test_parse_task_accepts_the_phase2a_task_set(spec, expected):
    assert parse_task(spec) == expected
    assert evaluate_parse_task(spec) == expected  # same function, imported in both modules


@pytest.mark.parametrize("spec", ["1-2", "3-1", "2-3", "bogus", "1", "1-1-1", ""])
def test_parse_task_rejects_anything_outside_the_task_set(spec):
    with pytest.raises(ValueError):
        parse_task(spec)


def test_format_task_round_trips_with_parse_task():
    for spec, task in TASKS.items():
        assert format_task(task) == spec
        assert parse_task(format_task(task)) == task


def test_tasks_is_exactly_the_phase2a_task_set():
    assert set(TASKS) == {"1-1", "2-1"}
    assert TASKS["1-1"] == (1, 1)
    assert TASKS["2-1"] == (2, 1)
