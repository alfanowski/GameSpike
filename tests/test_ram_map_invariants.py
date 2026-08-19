"""Invariants that re-confirm envs/ram_map.py against a real Super Mario Land ROM.

Each test drives the live game into a state where a *predicted* change must
appear at the mapped address. A silently-wrong address reads a plausible-looking
number forever, so "it returned an int" proves nothing -- only a predicted
change does. Several tests additionally cross-check against the background
tilemap (the on-screen HUD), which is an observation channel independent of the
RAM read itself.
"""
import os
import pytest
from pyboy import PyBoy
from envs import ram_map

ROM_PATH = os.environ.get("MARIO_LAND_ROM_PATH")

pytestmark = pytest.mark.skipif(
    not ROM_PATH or not os.path.exists(ROM_PATH),
    reason="MARIO_LAND_ROM_PATH not set or file missing",
)

# Background-tilemap encoding of the HUD, confirmed empirically: digit glyphs are
# tiles 256..265 for '0'..'9', and 300 is the blank/padding tile.
_DIGIT_BASE = 256
_BLANK_TILE = 300
_ADDR_WORLD_LEVEL = 0xFFB4  # 0x11 == world 1-1; used only to assert the test setup worked


def _start_game():
    """Boot to the start of world 1-1 with the player actually in control.

    Booting and ticking is NOT enough: Super Mario Land sits on the title screen
    and then plays its own attract-mode demo, during which the ROM -- not the
    test -- drives Mario. START has to be pressed before the demo takes over.
    """
    pyboy = PyBoy(ROM_PATH, window="null")
    pyboy.set_emulation_speed(0)
    for _ in range(200):  # past the Nintendo logo, onto the title screen
        pyboy.tick()
    pyboy.button_press("start")
    pyboy.tick()
    pyboy.button_release("start")
    for _ in range(200):  # level intro -> player in control
        pyboy.tick()
    assert pyboy.memory[_ADDR_WORLD_LEVEL] == 0x11, (
        "test setup failed: expected to be in world 1-1, got "
        f"0x{pyboy.memory[_ADDR_WORLD_LEVEL]:02X} -- the boot/START sequence needs re-timing"
    )
    return pyboy


def _hud_number(pyboy, x0, y, n):
    """Read an n-digit number straight off the HUD's background tilemap."""
    total = 0
    for i in range(n):
        tile = pyboy.tilemap_background[x0 + i, y]
        total = total * 10 + (0 if tile == _BLANK_TILE else tile - _DIGIT_BASE)
    return total


def _run(pyboy, frames, buttons=("right",), jump_period=None, jump_len=16, sample=None):
    """Hold `buttons` for `frames`, optionally jumping rhythmically.

    Returns one `sample(pyboy)` result per frame; callers that only care about a
    prefix (e.g. up to the first death) slice the result themselves.
    """
    out = []
    for b in buttons:
        pyboy.button_press(b)
    try:
        for f in range(frames):
            if jump_period is not None:
                phase = f % jump_period
                if phase == 0:
                    pyboy.button_press("a")
                elif phase == jump_len:
                    pyboy.button_release("a")
            pyboy.tick()
            if sample is not None:
                out.append(sample(pyboy))
    finally:
        for b in buttons:
            pyboy.button_release(b)
        pyboy.button_release("a")
    return out


# --------------------------------------------------------------------- Mario X
def test_mario_x_increases_while_holding_right():
    pyboy = _start_game()
    try:
        x_before = ram_map.read_mario_x(pyboy)
        _run(pyboy, 120)
        x_after = ram_map.read_mario_x(pyboy)
    finally:
        pyboy.stop(save=False)
    assert x_after > x_before, (
        f"expected rightward movement to increase screen X (got {x_before} -> {x_after}); "
        "ADDR_MARIO_X is likely wrong -- re-run envs/ram_scan_tool.py"
    )


def test_mario_x_saturates_and_is_not_a_progress_signal():
    """Guards the reason read_level_progress() exists.

    ADDR_MARIO_X pins at the camera-lock position and stops tracking how far
    Mario has actually travelled. If this test ever starts failing because X kept
    growing, the warning in ram_map.py (and Task 5's reward) needs revisiting.
    """
    pyboy = _start_game()
    try:
        x_start = ram_map.read_mario_x(pyboy)
        progress_start = ram_map.read_level_progress(pyboy)
        xs = _run(pyboy, 500, jump_period=40, sample=ram_map.read_mario_x)
        progress_gain = ram_map.read_level_progress(pyboy) - progress_start
    finally:
        pyboy.stop(save=False)
    x_gain = max(xs) - x_start
    assert x_gain < 64, f"screen X grew by {x_gain}; it is supposed to saturate at camera lock"
    assert progress_gain > 3 * x_gain, (
        f"level progress only moved {progress_gain} while screen X moved {x_gain}; "
        "the whole point is that real progress far outruns screen X"
    )


# --------------------------------------------------------------------- Mario Y
def test_mario_y_decreases_during_jump():
    pyboy = _start_game()
    try:
        y_ground = ram_map.read_mario_y(pyboy)
        ys = _run(pyboy, 40, buttons=("a",), sample=ram_map.read_mario_y)
        for _ in range(60):  # let him land again
            pyboy.tick()
        y_landed = ram_map.read_mario_y(pyboy)
    finally:
        pyboy.stop(save=False)
    assert min(ys) < y_ground - 10, (
        f"expected a jump to raise Mario (smaller Y) from {y_ground}, apex was {min(ys)}; "
        "ADDR_MARIO_Y is likely wrong"
    )
    assert y_landed == y_ground, f"expected Mario back on the ground at {y_ground}, got {y_landed}"


# ----------------------------------------------------------------------- timer
def test_timer_counts_down_when_idle():
    pyboy = _start_game()
    try:
        t_before = ram_map.read_timer(pyboy)
        for _ in range(600):  # several real seconds of game time
            pyboy.tick()
        t_after = ram_map.read_timer(pyboy)
    finally:
        pyboy.stop(save=False)
    assert t_after < t_before, (
        f"expected the level timer to count down (got {t_before} -> {t_after}); "
        "timer addresses are likely wrong -- re-run envs/ram_scan_tool.py"
    )


def test_timer_decoding_matches_the_on_screen_hud():
    """The decode is BCD seconds + a hundreds digit, NOT three decimal digits.

    Cross-checked against the TIME digits the game itself draws, so a wrong
    formula cannot pass by accident.
    """
    pyboy = _start_game()
    mismatches = []
    try:
        assert ram_map.read_timer(pyboy) > 350, "world 1-1 should start near 400 on the clock"
        for _ in range(400):
            pyboy.tick()
            decoded = ram_map.read_timer(pyboy)
            on_screen = _hud_number(pyboy, 17, 1, 3)
            if decoded != on_screen:
                mismatches.append((decoded, on_screen))
    finally:
        pyboy.stop(save=False)
    assert not mismatches, f"read_timer() disagreed with the HUD on {len(mismatches)} frames: {mismatches[:5]}"


def test_timer_frame_subcounter_wraps_within_a_second():
    """0xDA00 is a 40-frame sub-counter, which is why read_timer() ignores it."""
    pyboy = _start_game()
    vals = []
    try:
        for _ in range(120):
            pyboy.tick()
            vals.append(ram_map.read_timer_frames(pyboy))
    finally:
        pyboy.stop(save=False)
    assert 1 <= min(vals) and max(vals) <= 40, f"frame sub-counter out of the 1..40 range: {min(vals)}..{max(vals)}"
    assert any(vals[i] > vals[i - 1] for i in range(1, len(vals))), "sub-counter never wrapped in 120 frames"


# ----------------------------------------------------------------------- lives
def test_lives_in_plausible_range():
    pyboy = _start_game()
    try:
        lives = ram_map.read_lives(pyboy)
    finally:
        pyboy.stop(save=False)
    assert 0 <= lives <= 99, f"lives={lives} outside a plausible range -- ADDR_LIVES likely wrong"


def test_lives_match_the_hud_and_decrement_on_death():
    pyboy = _start_game()
    try:
        start_lives = ram_map.read_lives(pyboy)
        assert start_lives == _hud_number(pyboy, 6, 0, 2), "lives disagree with the HUD at level start"
        after = _run(pyboy, 600, sample=ram_map.read_lives)
    finally:
        pyboy.stop(save=False)
    assert min(after) < start_lives, (
        f"walking blindly right should have cost a life (stayed at {start_lives}); "
        "ADDR_LIVES is likely wrong"
    )


def test_lives_are_bcd_encoded_not_binary():
    """Natural play never exceeds 9 lives, where BCD and binary read the same.

    Poking a discriminating value settles it: 0x15 is 15 in BCD and 21 in binary,
    and after a death the game's own HUD shows which one it thinks it is.
    """
    pyboy = _start_game()
    try:
        pyboy.memory[ram_map.ADDR_LIVES] = 0x15
        pyboy.button_press("right")
        for _ in range(900):
            pyboy.tick()
            if pyboy.memory[ram_map.ADDR_LIVES] != 0x15:
                break
        else:
            pytest.fail("Mario never died, so the lives encoding could not be discriminated")
        pyboy.button_release("right")
        for _ in range(120):  # let the game redraw the lives digits
            pyboy.tick()
        decoded = ram_map.read_lives(pyboy)
        on_screen = _hud_number(pyboy, 6, 0, 2)
    finally:
        pyboy.stop(save=False)
    assert decoded == on_screen, (
        f"read_lives() returned {decoded} but the HUD shows {on_screen}; "
        "0xDA15 is BCD-encoded, so read_lives() must decode it as BCD (a raw byte read gives 20 here)"
    )


# --------------------------------------------------------------------- powerup
def test_powerup_is_small_at_level_start():
    pyboy = _start_game()
    try:
        state = ram_map.read_powerup_state(pyboy)
    finally:
        pyboy.stop(save=False)
    assert state == 0, f"Mario starts 1-1 small, so the powerup state should be 0, got {state}"


def test_powerup_state_actually_drives_marios_form():
    """Causal check: the game must *read* this byte, not just happen to sit near it.

    Small Mario dies to the first enemy on a blind rightward walk; if the byte is
    really the super status then forcing it non-zero turns that same collision
    into a shrink, with no life lost.
    """
    control = _start_game()
    try:
        control_lives = _run(control, 600, sample=ram_map.read_lives)
    finally:
        control.stop(save=False)
    assert min(control_lives) < control_lives[0], "control run was supposed to die; test setup drifted"

    pyboy = _start_game()
    try:
        tile_small = pyboy.get_sprite(3).tile_identifier
        pyboy.memory[ram_map.ADDR_POWERUP_STATE] = 1
        for _ in range(20):
            pyboy.tick()
        tile_super = pyboy.get_sprite(3).tile_identifier
        state_after_growth = ram_map.read_powerup_state(pyboy)
        lives = _run(pyboy, 600, sample=ram_map.read_lives)
    finally:
        pyboy.stop(save=False)

    assert tile_super != tile_small, (
        "forcing the powerup state did not change how Mario is drawn -- "
        "ADDR_POWERUP_STATE is probably not the super status"
    )
    assert state_after_growth == 2, f"expected the growing state to settle on 2 (super), got {state_after_growth}"
    assert min(lives) == lives[0], (
        f"super Mario should have been shrunk rather than killed, but lives went {lives[0]} -> {min(lives)}"
    )


# ----------------------------------------------------------------------- score
def test_score_starts_at_zero_and_increases_matching_the_hud():
    pyboy = _start_game()
    try:
        assert ram_map.read_score(pyboy) == 0, "score should start at 0"
        scores = _run(pyboy, 700, jump_period=40, sample=ram_map.read_score)
        final_ram = ram_map.read_score(pyboy)
        final_hud = _hud_number(pyboy, 0, 1, 6)
    finally:
        pyboy.stop(save=False)
    assert max(scores) > 0, "expected to stomp at least one enemy and score points -- ADDR_SCORE_START likely wrong"
    assert final_ram == final_hud, (
        f"decoded score {final_ram} disagrees with the HUD's {final_hud}; "
        "the BCD byte order in read_score() is wrong"
    )


def test_score_byte_order_is_least_significant_first():
    """Ordinary play only ever produces multiples of 100, which leaves the outer
    two score bytes at zero and so cannot distinguish the byte order. Poking a
    distinct pattern into all three and letting the game redraw the HUD can."""
    pyboy = _start_game()
    try:
        pyboy.memory[ram_map.ADDR_SCORE_START + 0] = 0x12
        pyboy.memory[ram_map.ADDR_SCORE_START + 1] = 0x34
        pyboy.memory[ram_map.ADDR_SCORE_START + 2] = 0x56
        # The HUD is only redrawn when the score changes, so earn some points.
        _run(pyboy, 700, jump_period=40)
        decoded = ram_map.read_score(pyboy)
        on_screen = _hud_number(pyboy, 0, 1, 6)
    finally:
        pyboy.stop(save=False)
    assert on_screen > 500000, (
        f"the poked score never reached the HUD (shows {on_screen}); no enemy was stomped"
    )
    assert decoded == on_screen, (
        f"read_score() returned {decoded} but the game drew {on_screen}; "
        "the three BCD bytes go least-significant first (0xC0A0 is the ones/tens pair)"
    )


# ------------------------------------------------------------- scroll/progress
def test_scroll_x_advances_while_holding_right():
    pyboy = _start_game()
    try:
        before = ram_map.read_scroll_x(pyboy)
        _run(pyboy, 150)
        after = ram_map.read_scroll_x(pyboy)
    finally:
        pyboy.stop(save=False)
    assert after > before, (
        f"expected the camera to scroll right (got {before} -> {after}); ADDR_SCROLL_X is likely wrong"
    )


def test_scroll_x_wraps_and_so_cannot_be_diffed_raw():
    """Documents why read_level_progress() exists rather than diffing scroll X.

    ADDR_SCROLL_X is only the low byte of the camera position, so it rolls over
    every 256 pixels -- twice inside one ordinary life.
    """
    pyboy = _start_game()
    try:
        vals = _run(pyboy, 500, jump_period=40, sample=ram_map.read_scroll_x)
    finally:
        pyboy.stop(save=False)
    wraps = [(vals[i - 1], vals[i]) for i in range(1, len(vals)) if vals[i] < vals[i - 1] - 100]
    assert wraps, "scroll X never wrapped; the 'wraps mod 256' warning in ram_map.py needs re-checking"


def test_level_progress_is_monotonic_across_a_scroll_wrap():
    pyboy = _start_game()
    try:
        start_lives = ram_map.read_lives(pyboy)
        samples = _run(
            pyboy, 800, jump_period=40,
            sample=lambda p: (ram_map.read_level_progress(p), ram_map.read_lives(p),
                              ram_map.read_scroll_x(p)),
        )
    finally:
        pyboy.stop(save=False)

    # Only the stretch before the first death is meaningful: dying resets the camera.
    alive = []
    for progress, lives, scroll in samples:
        if lives < start_lives:
            break
        alive.append((progress, scroll))
    assert len(alive) > 300, f"died too early ({len(alive)} frames) to cross a scroll wrap"

    scrolls = [s for _, s in alive]
    assert any(scrolls[i] < scrolls[i - 1] - 100 for i in range(1, len(scrolls))), \
        "the run did not cross a scroll wrap, so this test proved nothing"

    progresses = [p for p, _ in alive]
    drops = [(i, progresses[i - 1], progresses[i]) for i in range(1, len(progresses))
             if progresses[i] < progresses[i - 1]]
    assert not drops, f"level progress went backwards while running right: {drops[:5]}"
    assert progresses[-1] - progresses[0] > 200, (
        f"level progress barely moved ({progresses[0]} -> {progresses[-1]}) over a long rightward run"
    )
