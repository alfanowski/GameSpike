"""Locked, empirically-confirmed Super Mario Land RAM addresses.

STATUS: CONFIRMED against `Super Mario Land (World).gb` (cartridge title
"SUPER MARIOLAND") on 2026-08-20, Task 3 Steps 2/5. Every constant below was
verified by driving the live game in PyBoy and observing a predicted state
change -- not by trusting a community RAM map. Where a byte's *encoding* was
ambiguous it was pinned down by poking a discriminating value into RAM and
reading back what the game's own HUD-rendering code then drew on screen (the
background tilemap is an observation channel independent of the RAM read).

How each one was confirmed:
  * ADDR_LIVES         2 at the start of 1-1, matching the HUD's "02"; drops
                       2 -> 1 -> 0 on successive deaths. Encoding: poking 0x15
                       then dying yielded 0x14 and a HUD reading "14" (a binary
                       reading would have shown "20"), so the field is BCD.
  * ADDR_POWERUP_STATE 0 while small. Forcing it to 1 makes the game advance it
                       to 2 on its own, swaps Mario's sprite tiles (+32), and --
                       decisively -- turns an enemy collision that kills small
                       Mario into a mere shrink (2 -> 3 -> 4 -> 0, lives
                       unchanged). That is the defining behaviour of the super
                       state, so the game really does read this byte.
  * ADDR_TIMER_*       See read_timer(): the decoded value matched the on-screen
                       TIME digits on all 900 frames sampled.
  * ADDR_SCORE_START   Poking C0A0/C0A1/C0A2 = 0x12/0x34/0x56 made the game draw
                       "563512", proving 3 BCD byte-pairs, least-significant
                       first. Each enemy stomp then incremented C0A1 by one BCD
                       unit (= +100 points), matching the HUD.
  * ADDR_MARIO_Y       134 standing, 101 at the apex of a jump, back to 134 on
                       landing. Y grows downward (screen coordinates).
  * ADDR_MARIO_X       50 at level start, rises to 81, then pins there forever
                       as the camera takes over. SCREEN-relative -- see below.
  * ADDR_SCROLL_X      Rises with rightward movement, but it is only the low
                       byte of the camera position: it wrapped 255 -> 0 twice
                       within a single 868-frame life. See below.
  * ADDR_LEVEL_BLOCK   12 at the start of 1-1, rising to 46 with zero decreases
                       over the same 868 frames, including across both scroll
                       wraps. This is the un-wrapped part of the camera position.
  * ADDR_WORLD_LEVEL   Poking 0x34 and forcing a status-bar redraw (by dying)
                       made the game draw "3-4", so it renders the world and
                       level straight out of this byte's two nibbles. It also
                       advances on a real, player-driven level completion: 1-1
                       was played through to the exit (progress reached 2592,
                       lives unchanged) and the byte went 0x11 -> 0x12 with the
                       HUD following "1-1" -> "1-2", after which the game started
                       1-2 with the progress counter re-based. That completion is
                       committed as a test -- see
                       tests/test_mario_land_env.py::test_completing_a_level_pays_the_bonus_and_terminates.
                       CORRECTION (Task 5): an earlier note here cited a
                       0x11 -> 0x12 transition seen during a long unattended run
                       as a "natural level change". Re-running that scenario shows
                       the run had already hit a game over (lives 2 -> 1 -> 0, then
                       0 -> 2 as the ROM restarted) and the byte moved because the
                       *attract-mode demo* went on to play 1-2 -- no level was
                       completed. The byte still tracks the current level, so the
                       conclusion drawn from it stood, but it was not evidence of a
                       completion, and reading a change in it as "level finished" is
                       only safe while the episode can never reach the game over.
                       NOTE it is also useless as a "are we in gameplay yet" gate --
                       it already reads 0x11 on the title screen at frame 120,
                       before START is pressed. See envs/boot.py for the gate that
                       actually works.

Do not add or change an address here without repeating that empirical
confirmation; a wrong address fails silently (it just reads a plausible-looking
wrong number), which is worse than a crash. The invariant tests in
tests/test_ram_map_invariants.py re-check these against a real ROM.
"""

# --- Mario, screen-relative ---------------------------------------------------
# WARNING: ADDR_MARIO_X is Mario's X *within the visible screen*, not within the
# level. Once the camera locks on he sits at 81 and stays there no matter how far
# he runs: over one measured 868-frame life he advanced ~575px through the level
# while this byte moved a total of +31. A naive `x_now - x_prev` progress reward
# built on it is therefore near-zero for almost the whole level, and swings
# hugely negative on respawn (81 -> 50). Use read_level_progress() instead.
ADDR_MARIO_X = 0xC202
ADDR_MARIO_Y = 0xC201

# --- HUD / game state ---------------------------------------------------------
ADDR_LIVES = 0xDA15          # BCD-encoded, matches the HUD's two lives digits
ADDR_POWERUP_STATE = 0xFF99  # HRAM. 0=small 1=growing 2=super 3=shrinking 4=post-hit
ADDR_SCORE_START = 0xC0A0    # 3 bytes of BCD digit-pairs, least significant first

# --- Level timer --------------------------------------------------------------
# NOT three decimal digits of one number (the pre-confirmation stub assumed that
# and was wrong). It is a frame sub-counter plus a BCD seconds field:
ADDR_TIMER_FRAMES = 0xDA00       # counts down 40 -> 1, wraps; one game-second per 40 frames
ADDR_TIMER_SECONDS_BCD = 0xDA01  # BCD, the tens and ones digits of the displayed time
ADDR_TIMER_HUNDREDS = 0xDA02     # the hundreds digit (BCD; 0x03 at the start of 1-1 = 400)

# --- Camera / world progress --------------------------------------------------
# ADDR_SCROLL_X is the camera's horizontal scroll, but only its low 8 bits: it
# wraps 255 -> 0 every 256 pixels of level, so it is NOT a usable progress signal
# on its own either. ADDR_LEVEL_BLOCK is the camera's position counted in 16-pixel
# blocks and does not wrap; read_level_progress() composes the two safely.
ADDR_SCROLL_X = 0xFFA4     # HRAM, low byte of camera scroll X -- wraps mod 256
ADDR_LEVEL_BLOCK = 0xC0AB  # camera position in 16px blocks; monotonic within a life

# --- Which level we are in ----------------------------------------------------
# HRAM. High nibble = world, low nibble = level (0x11 == world 1-1). Useful to
# Task 5 for spotting an episode boundary, but NOT usable to detect whether
# gameplay has started -- see the note in the module docstring.
ADDR_WORLD_LEVEL = 0xFFB4


def _bcd_to_int(value: int) -> int:
    """Decode one binary-coded-decimal byte (two decimal digits) to an int."""
    return (value >> 4) * 10 + (value & 0x0F)


def read_mario_x(pyboy) -> int:
    """Mario's X *within the screen* (0..81-ish). Not level progress -- see above."""
    assert ADDR_MARIO_X is not None, "ADDR_MARIO_X not confirmed -- see Task 3 Step 2"
    return pyboy.memory[ADDR_MARIO_X]


def read_mario_y(pyboy) -> int:
    """Mario's Y within the screen; smaller means higher up (134 ground, ~101 jump apex)."""
    assert ADDR_MARIO_Y is not None, "ADDR_MARIO_Y not confirmed -- see Task 3 Step 2"
    return pyboy.memory[ADDR_MARIO_Y]


def read_lives(pyboy) -> int:
    assert ADDR_LIVES is not None, "ADDR_LIVES not confirmed -- see Task 3 Step 2"
    return _bcd_to_int(pyboy.memory[ADDR_LIVES])


def read_timer(pyboy) -> int:
    """The level timer exactly as the HUD shows it (400 at the start of 1-1).

    0xDA00 is a 40-frame sub-counter and deliberately does not contribute: it
    would make the returned value non-monotonic within each game-second.
    """
    assert ADDR_TIMER_SECONDS_BCD is not None, "timer addresses not confirmed -- see Task 3 Step 2"
    hundreds = _bcd_to_int(pyboy.memory[ADDR_TIMER_HUNDREDS])
    seconds = _bcd_to_int(pyboy.memory[ADDR_TIMER_SECONDS_BCD])
    return hundreds * 100 + seconds


def read_timer_frames(pyboy) -> int:
    """Sub-second frame counter of the level timer: counts down 40 -> 1, then wraps."""
    assert ADDR_TIMER_FRAMES is not None, "timer addresses not confirmed -- see Task 3 Step 2"
    return pyboy.memory[ADDR_TIMER_FRAMES]


def read_powerup_state(pyboy) -> int:
    """0=small, 1=growing, 2=super, 3=shrinking, 4=post-hit invulnerable."""
    assert ADDR_POWERUP_STATE is not None, "ADDR_POWERUP_STATE not confirmed -- see Task 3 Step 2"
    return pyboy.memory[ADDR_POWERUP_STATE]


def read_score(pyboy) -> int:
    """Score as displayed: three BCD digit-pairs, least significant byte first."""
    assert ADDR_SCORE_START is not None, "ADDR_SCORE_START not confirmed -- see Task 3 Step 2"
    low, mid, high = (pyboy.memory[ADDR_SCORE_START + i] for i in range(3))
    return _bcd_to_int(low) + _bcd_to_int(mid) * 100 + _bcd_to_int(high) * 10000


def read_scroll_x(pyboy) -> int:
    """Low byte of the camera's scroll X. Wraps mod 256 -- never diff this raw."""
    assert ADDR_SCROLL_X is not None, "ADDR_SCROLL_X not confirmed -- see Task 3 Step 2"
    return pyboy.memory[ADDR_SCROLL_X]


def read_level_block(pyboy) -> int:
    """Camera position in 16-pixel blocks. Does not wrap; resets on respawn."""
    assert ADDR_LEVEL_BLOCK is not None, "ADDR_LEVEL_BLOCK not confirmed -- see Task 3 Step 2"
    return pyboy.memory[ADDR_LEVEL_BLOCK]


def read_world_level(pyboy):
    """The current level as a (world, level) pair -- (1, 1) at the start of a run."""
    assert ADDR_WORLD_LEVEL is not None, "ADDR_WORLD_LEVEL not confirmed -- see Task 3 Step 2"
    packed = pyboy.memory[ADDR_WORLD_LEVEL]
    return packed >> 4, packed & 0x0F


def read_level_progress(pyboy) -> int:
    """Mario's approximate position *within the level*, in pixels.

    This is the signal a progress reward should be built on. The camera's
    un-wrapped block index supplies the coarse part and Mario's screen X the
    fine part, so the value survives camera scrolling (measured: strictly
    non-decreasing across an 868-frame rightward run that included two wraps of
    ADDR_SCROLL_X) while still falling when Mario genuinely walks back left
    (measured: -48 over 60 frames of holding left).

    Accurate to within one 16px block; it collapses when Mario dies, because a
    death reloads the level, so a life loss must be treated as an episode/segment
    boundary rather than fed into a reward as a giant negative delta. That is what
    envs/mario_land_env.py does.

    It does NOT necessarily collapse to the *level's start*: 1-1 has an
    intermediate checkpoint, and a death past it reloads there instead (measured:
    a death at progress 929 respawned at 882, while a death earlier in the same
    level respawned at 242). Never assume "respawn == level start" -- read the
    post-reload value back and re-baseline on it, whatever it turns out to be.
    """
    return read_level_block(pyboy) * 16 + read_mario_x(pyboy)
