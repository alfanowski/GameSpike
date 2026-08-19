import os
import pytest
from pyboy import PyBoy
from envs import ram_map

ROM_PATH = os.environ.get("MARIO_LAND_ROM_PATH")

pytestmark = pytest.mark.skipif(
    not ROM_PATH or not os.path.exists(ROM_PATH),
    reason="MARIO_LAND_ROM_PATH not set or file missing",
)


def _boot(frames=30):
    pyboy = PyBoy(ROM_PATH, window="null")
    pyboy.set_emulation_speed(0)
    for _ in range(frames):
        pyboy.tick()
    return pyboy


def test_mario_x_increases_while_holding_right():
    pyboy = _boot()
    x_before = ram_map.read_mario_x(pyboy)
    pyboy.button_press("right")
    for _ in range(120):
        pyboy.tick()
    x_after = ram_map.read_mario_x(pyboy)
    pyboy.button_release("right")
    pyboy.stop(save=False)
    assert x_after > x_before, (
        f"expected rightward movement to increase X (got {x_before} -> {x_after}); "
        "ADDR_MARIO_X is likely wrong -- re-run envs/ram_scan_tool.py"
    )


def test_timer_counts_down_when_idle():
    pyboy = _boot()
    t_before = ram_map.read_timer(pyboy)
    for _ in range(600):  # several real seconds of game time
        pyboy.tick()
    t_after = ram_map.read_timer(pyboy)
    pyboy.stop(save=False)
    assert t_after < t_before, (
        f"expected the level timer to count down (got {t_before} -> {t_after}); "
        "timer addresses are likely wrong -- re-run envs/ram_scan_tool.py"
    )


def test_lives_in_plausible_range():
    pyboy = _boot()
    lives = ram_map.read_lives(pyboy)
    pyboy.stop(save=False)
    assert 0 <= lives <= 99, f"lives={lives} outside a plausible range -- ADDR_LIVES likely wrong"
