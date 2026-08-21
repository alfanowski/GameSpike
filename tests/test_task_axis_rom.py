"""Phase 2a's task axis: end-to-end behaviour that DOES need a real ROM.

Split out from tests/test_task_axis.py (which covers everything task-related that
does not need PyBoy at all) so the ROM-free half of the suite stays importable and
runnable without MARIO_LAND_ROM_PATH set.

The single most important test in this file is
`test_default_task_constructs_env_with_world_level_none`: docs/RESULTS.md §23
claims training/evaluate.py is byte-identical to commit 64839a9, and that claim no
longer holds after this change -- what MUST still hold is BEHAVIOURAL identity when
--task is unset, and this test is the direct evidence for it.
"""
import os

import pytest

import training.evaluate as evaluate_module
from envs.mario_land_env import MarioLandEnv
from training.evaluate import run_evaluation
from training.train import build_model, run_dir_for, run_training, save_checkpoint

ROM_PATH = os.environ.get("MARIO_LAND_ROM_PATH")
pytestmark = pytest.mark.skipif(
    not ROM_PATH or not os.path.exists(ROM_PATH),
    reason="MARIO_LAND_ROM_PATH not set or file missing; set it to your own legally-dumped ROM",
)


def _checkpoint(tmp_path, arm="baseline", task=None):
    model, optimizer = build_model(arm)
    model._task = task
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, optimizer, step=0, path=str(path))
    return str(path)


class _CapturingEnv(MarioLandEnv):
    """Records every constructor call's kwargs, split by instance -- mirrors
    tests/test_evaluate.py's own _SpyEnv pattern."""
    calls = []

    def __init__(self, *args, **kwargs):
        type(self).calls.append(kwargs)
        super().__init__(*args, **kwargs)


# --------------------------------------------------------------------------- #
# training/evaluate.py
# --------------------------------------------------------------------------- #

def test_default_task_constructs_env_with_world_level_none(tmp_path, monkeypatch):
    """THE deliverable this brief calls out by name: no --task must mean the env
    is constructed with world_level=None, i.e. behaviourally identical to every
    evaluation run before this change existed."""
    _CapturingEnv.calls = []
    monkeypatch.setattr(evaluate_module, "MarioLandEnv", _CapturingEnv)
    checkpoint = _checkpoint(tmp_path)

    run_evaluation("baseline", checkpoint, ROM_PATH, n_episodes=1, max_steps_per_episode=5)

    assert len(_CapturingEnv.calls) == 1
    assert _CapturingEnv.calls[0]["world_level"] is None


def test_task_2_1_constructs_env_with_that_world_level(tmp_path, monkeypatch):
    _CapturingEnv.calls = []
    monkeypatch.setattr(evaluate_module, "MarioLandEnv", _CapturingEnv)
    checkpoint = _checkpoint(tmp_path)

    run_evaluation("baseline", checkpoint, ROM_PATH, n_episodes=1, max_steps_per_episode=5,
                   task=(2, 1))

    assert _CapturingEnv.calls[0]["world_level"] == (2, 1)


def test_evaluation_results_record_the_task(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    results_none = run_evaluation("baseline", checkpoint, ROM_PATH, n_episodes=1,
                                  max_steps_per_episode=5)
    results_task = run_evaluation("baseline", checkpoint, ROM_PATH, n_episodes=1,
                                  max_steps_per_episode=5, task=(2, 1))
    assert results_none["task"] is None
    assert results_task["task"] == (2, 1)


def test_evaluating_on_2_1_regardless_of_what_the_checkpoint_was_trained_on(tmp_path):
    """Zero-shot transfer scoring is the point, not an error -- a checkpoint
    labelled task=(1, 1) can legitimately be evaluated with --task 2-1."""
    checkpoint = _checkpoint(tmp_path, task=(1, 1))
    results = run_evaluation("baseline", checkpoint, ROM_PATH, n_episodes=1,
                             max_steps_per_episode=5, task=(2, 1))
    assert results["task"] == (2, 1)


# --------------------------------------------------------------------------- #
# training/train.py -- short real run_training calls
# --------------------------------------------------------------------------- #

def test_run_training_without_a_task_is_unaffected(tmp_path):
    stats = run_training(arm="baseline", rom_path=ROM_PATH, total_steps=32,
                         n_envs=1, rollout_len=16, checkpoint_every=1_000_000,
                         checkpoint_dir=str(tmp_path))
    assert stats["task"] is None
    assert stats["run_dir"] == run_dir_for(str(tmp_path), "baseline", 0)


def test_run_training_with_task_2_1_writes_a_task_labelled_run_dir_and_checkpoint(tmp_path):
    stats = run_training(arm="baseline", rom_path=ROM_PATH, total_steps=32,
                         n_envs=1, rollout_len=16, checkpoint_every=1_000_000,
                         checkpoint_dir=str(tmp_path), task=(2, 1))
    assert stats["task"] == (2, 1)
    expected_dir = run_dir_for(str(tmp_path), "baseline", 0, task=(2, 1))
    assert stats["run_dir"] == expected_dir
    assert os.path.isdir(expected_dir)

    import torch
    final_ckpt = os.path.join(expected_dir, f"step_{stats['final_step']}.pt")
    ckpt = torch.load(final_ckpt, map_location="cpu", weights_only=True)
    assert ckpt["task"] == (2, 1)


def test_run_training_1_1_and_2_1_at_the_same_seed_land_in_different_directories(tmp_path):
    """The collision the brief calls out as the worst failure mode, exercised
    end to end through the real training loop rather than only through
    run_dir_for in isolation (tests/test_task_axis.py already covers that)."""
    stats_1_1 = run_training(arm="baseline", rom_path=ROM_PATH, total_steps=32,
                             n_envs=1, rollout_len=16, checkpoint_every=1_000_000,
                             checkpoint_dir=str(tmp_path), task=(1, 1))
    stats_2_1 = run_training(arm="baseline", rom_path=ROM_PATH, total_steps=32,
                             n_envs=1, rollout_len=16, checkpoint_every=1_000_000,
                             checkpoint_dir=str(tmp_path), task=(2, 1))
    assert stats_1_1["run_dir"] != stats_2_1["run_dir"]
    assert os.path.isdir(stats_1_1["run_dir"])
    assert os.path.isdir(stats_2_1["run_dir"])
