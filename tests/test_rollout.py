"""Multi-process rollout mechanics, against a real ROM.

Random-policy only, deliberately: this pins down the multiprocessing/shape
contract in isolation, before Task 11 wires a real model into the same
collection loop.
"""
import os

import numpy as np
import pytest

from training.rollout import collect_rollout_random_policy

ROM_PATH = os.environ.get("MARIO_LAND_ROM_PATH")
pytestmark = pytest.mark.skipif(
    not ROM_PATH or not os.path.exists(ROM_PATH),
    reason="MARIO_LAND_ROM_PATH not set or file missing; set it to your own legally-dumped ROM",
)


def test_collect_rollout_shapes():
    data = collect_rollout_random_policy(ROM_PATH, n_envs=2, n_steps=16)
    assert data["obs"].shape == (2, 16, 12)
    assert data["actions"].shape == (2, 16)
    assert data["rewards"].shape == (2, 16)
    assert data["dones"].shape == (2, 16)
    assert data["actions"].max() < 10 and data["actions"].min() >= 0
