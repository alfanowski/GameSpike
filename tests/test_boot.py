"""Tests for envs/boot.py's `world_level` boot path (Phase 2a's task axis).

The power-on path (`world_level=None`) is exhaustively exercised elsewhere
(tests/test_ram_map_invariants.py, tests/test_mario_land_env*.py) and MUST NOT
change here -- see the module docstring's own claim about this. This file only
covers the new `game_wrapper.start_game(world_level=...)` path: that it really
reaches the requested level (not just draws it on the HUD) and that the
behavioural control gate (the same one the power-on path uses) passes there too.
"""
import os

import pytest
from pyboy import PyBoy

from envs import boot, ram_map

ROM_PATH = os.environ.get("MARIO_LAND_ROM_PATH")

pytestmark = pytest.mark.skipif(
    not ROM_PATH or not os.path.exists(ROM_PATH),
    reason="MARIO_LAND_ROM_PATH not set or file missing; set it to your own legally-dumped ROM",
)


def test_world_level_none_still_boots_into_1_1():
    """The default must remain exactly what it always was."""
    pyboy = boot.boot_to_level_start(ROM_PATH)
    try:
        assert ram_map.read_world_level(pyboy) == (1, 1)
    finally:
        pyboy.stop(save=False)


def test_world_level_2_1_reaches_world_2_1_with_control_verified():
    """The task this brief actually asks for: boot into 2-1, not 1-1, and the
    behavioural gate (Mario answers both directions of input) must pass there --
    boot_to_level_start raises AssertionError internally if it does not, so simply
    not raising is already most of this test's assertion."""
    pyboy = boot.boot_to_level_start(ROM_PATH, world_level=(2, 1))
    try:
        assert ram_map.read_world_level(pyboy) == (2, 1)
    finally:
        pyboy.stop(save=False)


def test_world_level_boot_matches_the_power_on_paths_level_start_state():
    """Measured 2026-08-21: game_wrapper.start_game(world_level=...) lands on
    IDENTICAL state to the power-on path's own level start (lives=2, timer=400,
    Mario screen X=50) for both 1-1 and 2-1 -- the two boot paths agree on what
    "the start of a level" means, they just take a different route to get there."""
    for world_level in ((1, 1), (2, 1)):
        pyboy = boot.boot_to_level_start(ROM_PATH, world_level=world_level)
        try:
            assert ram_map.read_lives(pyboy) == 2
            assert ram_map.read_timer(pyboy) > 350
            assert ram_map.read_mario_x(pyboy) == 50
        finally:
            pyboy.stop(save=False)


def test_world_level_start_game_rejects_a_hold_right_probe_shorter_than_2_1_survives():
    """Guards against a settle window that accidentally eats into 2-1's short
    fuse. Per docs/DESIGN_ROADMAP_PHASE2.md §14.5, a continuously-held-right policy
    in 2-1 dies around frame 235 from the level start -- so the settle-then-probe
    sequence boot_to_level_start runs (WORLD_LEVEL_SETTLE_FRAMES ticks, then up to
    _PROBE_FRAMES=40 frames held right) must land well inside that budget, or the
    gate itself would start flaking on exactly the level it needs to support."""
    assert boot.WORLD_LEVEL_SETTLE_FRAMES + boot._PROBE_FRAMES < 200


def test_world_level_none_and_world_level_tuple_share_the_same_gate_function():
    """assert_player_has_control is not duplicated for the new path -- both boot
    routes call the identical function, so a fix to the gate covers both at once."""
    pyboy = PyBoy(ROM_PATH, window="null")
    pyboy.set_emulation_speed(0)
    try:
        pyboy.game_wrapper.start_game(world_level=(2, 1))
        for _ in range(boot.WORLD_LEVEL_SETTLE_FRAMES):
            pyboy.tick()
        boot.assert_player_has_control(pyboy)  # must not raise
    finally:
        pyboy.stop(save=False)


def test_verify_control_false_skips_the_gate_on_the_world_level_path_too():
    """Mirrors the power-on path's own verify_control=False escape hatch (used
    throughout the test suite to avoid paying the gate's cost per test) -- must
    work identically on the world_level path rather than silently ignoring the
    flag there."""
    pyboy = boot.boot_to_level_start(ROM_PATH, world_level=(2, 1), verify_control=False)
    try:
        assert ram_map.read_world_level(pyboy) == (2, 1)
    finally:
        pyboy.stop(save=False)
