"""The one place that knows how to get Super Mario Land into playable gameplay.

Booting the ROM and ticking is NOT enough, and getting this wrong is silent
rather than loud, which is why it lives in exactly one module:

  * The game sits on the title screen for ~770 frames and then starts its own
    **attract-mode demo**, in which the ROM drives Mario and your held buttons do
    nothing. Anything measured during the demo looks like real gameplay -- Mario
    runs, the camera scrolls, the timer counts down -- but is uncorrelated with
    the input you think you are applying.
  * Checking a world/level register does NOT rule that out. Measured on this ROM,
    0xFFB4 already reads 0x11 ("world 1-1") on the **title screen**, at frame 120,
    long before START is pressed, and the attract demo plays 1-1 as well. A
    `world == 1-1` gate therefore passes in all three states and discriminates
    nothing.

So the gate here is behavioural: it proves the *player* is in control by checking
that Mario answers to both directions of input. The title screen has no Mario
responding to anything, and the demo's Mario ignores the pad, so only real
gameplay passes. The probe is then rolled back via a savestate, leaving the
caller at an untouched level start.
"""
import io

from envs import ram_map

TITLE_SCREEN_FRAMES = 200  # past the Nintendo logo, onto the title screen
LEVEL_INTRO_FRAMES = 200   # level intro -> player in control
_PROBE_FRAMES = 40         # long enough to move Mario several pixels either way
_MIN_PROBE_TRAVEL = 4      # pixels; real movement clears this easily (measured ~25)


def press_start(pyboy):
    """Tap START for a single frame."""
    pyboy.button_press("start")
    pyboy.tick()
    pyboy.button_release("start")


def _probe(pyboy, button):
    """Hold `button` for the probe window and return (x_before, x_after)."""
    before = ram_map.read_mario_x(pyboy)
    pyboy.button_press(button)
    try:
        for _ in range(_PROBE_FRAMES):
            pyboy.tick()
    finally:
        pyboy.button_release(button)
    return before, ram_map.read_mario_x(pyboy)


def assert_player_has_control(pyboy):
    """Fail loudly unless Mario actually answers the pad. Leaves state untouched.

    Right is probed first so that Mario is guaranteed room to move back left
    afterwards, which would not hold if he were standing against the level's
    left edge.

    Note the deliberate circularity: this uses ADDR_MARIO_X, the very address
    whose responsiveness it is establishing. That is fine and is covered --
    corrupting ADDR_MARIO_X makes this gate raise, so the mutation test still
    catches it -- but it does mean this gate assumes ADDR_MARIO_X is confirmed
    and cannot be used to discover it.
    """
    state = io.BytesIO()
    pyboy.save_state(state)
    try:
        r_before, r_after = _probe(pyboy, "right")
        assert r_after - r_before >= _MIN_PROBE_TRAVEL, (
            f"holding right moved Mario {r_before} -> {r_after}; the player is not in "
            "control -- the emulator is most likely on the title screen or in the "
            "attract-mode demo, not in real gameplay"
        )
        l_before, l_after = _probe(pyboy, "left")
        assert l_before - l_after >= _MIN_PROBE_TRAVEL, (
            f"holding left moved Mario {l_before} -> {l_after}; Mario responded to right "
            "but not to left, so this is not player-controlled gameplay"
        )
    finally:
        state.seek(0)
        pyboy.load_state(state)


def start_game(pyboy, verify_control=True):
    """Drive `pyboy` from power-on to the start of world 1-1, player in control."""
    for _ in range(TITLE_SCREEN_FRAMES):
        pyboy.tick()
    press_start(pyboy)
    for _ in range(LEVEL_INTRO_FRAMES):
        pyboy.tick()
    if verify_control:
        assert_player_has_control(pyboy)
    return pyboy


def boot_to_level_start(rom_path, window="null", verify_control=True):
    """Construct a headless PyBoy and leave it at the start of world 1-1."""
    from pyboy import PyBoy

    pyboy = PyBoy(rom_path, window=window)
    pyboy.set_emulation_speed(0)
    try:
        start_game(pyboy, verify_control=verify_control)
    except BaseException:
        pyboy.stop(save=False)
        raise
    return pyboy
