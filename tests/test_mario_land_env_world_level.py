"""MarioLandEnv's `world_level` task axis (Phase 2a, docs/DESIGN_ROADMAP_PHASE2.md §9).

Deliberately narrow in scope: the observation/reward/termination machinery itself
is exhaustively covered against 1-1 in tests/test_mario_land_env.py and must not be
re-litigated here. This file only checks that (a) world_level=None is unchanged,
(b) world_level=(2, 1) actually resets into 2-1 rather than silently staying on
1-1, and (c) the standard observation contract (shape, dtype, [-1, 1] bounds) still
holds on the new level -- since 2-1 dies fast (frame ~235 under hold-right, per
§14.5), tests here only ever run a handful of steps.
"""
import os

import numpy as np
import pytest

from envs.mario_land_env import OBS_DIM, MarioLandEnv

ROM_PATH = os.environ.get("MARIO_LAND_ROM_PATH")

pytestmark = pytest.mark.skipif(
    not ROM_PATH or not os.path.exists(ROM_PATH),
    reason="MARIO_LAND_ROM_PATH not set or file missing; set it to your own legally-dumped ROM",
)


def _env(**kwargs):
    kwargs.setdefault("verify_control", False)  # gate is covered in tests/test_boot.py
    return MarioLandEnv(rom_path=ROM_PATH, **kwargs)


def test_world_level_none_still_starts_at_1_1():
    env = _env()
    _, info = env.reset()
    try:
        assert info["world_level"] == (1, 1)
    finally:
        env.close()


def test_world_level_default_is_none():
    """The constructor's default must be the historical, task-less behaviour --
    a caller that never heard of Phase 2a gets exactly what it always got."""
    env = _env()
    assert env.world_level is None
    env.close()


def test_world_level_2_1_resets_into_2_1():
    env = _env(world_level=(2, 1))
    _, info = env.reset()
    try:
        assert info["world_level"] == (2, 1)
    finally:
        env.close()


def test_world_level_2_1_observation_shape_and_bounds():
    env = _env(world_level=(2, 1))
    obs, _ = env.reset()
    try:
        assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
        assert env.observation_space.contains(obs)
        for action in range(env.action_space.n):
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
            assert env.observation_space.contains(obs), f"observation escaped its box: {obs}"
            assert isinstance(reward, float)
            if terminated or truncated:
                break
    finally:
        env.close()


def test_world_level_2_1_runs_a_few_steps_without_error():
    """Not a survival test -- §14.5 measured a hold-right policy dying around
    frame 235 in 2-1, so this deliberately runs FEWER steps than that and does
    not assert the episode is still alive at the end."""
    env = _env(world_level=(2, 1))
    env.reset()
    try:
        right = env.action_index("right")
        for _ in range(20):
            obs, reward, terminated, truncated, info = env.step(right)
            if terminated or truncated:
                break
    finally:
        env.close()


def test_two_envs_with_different_world_levels_do_not_interfere():
    """Guards against the world_level being read once at import time or cached
    across instances instead of being a genuine per-instance setting."""
    env_a = _env(world_level=(1, 1))
    env_b = _env(world_level=(2, 1))
    try:
        _, info_a = env_a.reset()
        _, info_b = env_b.reset()
        assert info_a["world_level"] == (1, 1)
        assert info_b["world_level"] == (2, 1)
    finally:
        env_a.close()
        env_b.close()
