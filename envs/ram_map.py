"""Locked, empirically-confirmed Super Mario Land RAM addresses.

Every constant here was confirmed by running envs/ram_scan_tool.py against a
real ROM and cross-checking against a public disassembly reference -- see
Task 3, Step 2 of docs/superpowers/plans/2026-08-19-mario-ppo-reservoir.md.
Do not add or change an address here without re-running that empirical
confirmation; a wrong address fails silently (it just reads a plausible-looking
wrong number), which is worse than a crash.

STATUS: NOT YET CONFIRMED. No Super Mario Land ROM was available when this file
was created (MARIO_LAND_ROM_PATH unset), so Step 2's scan could not be run and
every ADDR_* below is deliberately left as None. Filling these in from memory,
a community RAM map, or any other non-empirical source is explicitly forbidden
by this project's design doc (Sec. 2): addresses differ across ROM revisions and
regions, so an address that was not verified against *this* ROM is not locked.
Run the scan against your own legally-dumped ROM, then hand-write the results.
"""

# --- CONFIRM AND FILL IN before Task 5 depends on this file ---
ADDR_MARIO_X = None       # world-relative X position (confirm: monotonic while holding right)
ADDR_MARIO_Y = None       # screen-relative Y position (confirm: decreases while jumping)
ADDR_LIVES = None         # confirm: decrements on death, visible on the lives-lost screen
ADDR_TIMER_HUNDREDS = None
ADDR_TIMER_TENS = None
ADDR_TIMER_ONES = None    # confirm: BCD-or-binary digits counting down; verify encoding empirically
ADDR_POWERUP_STATE = None # confirm: changes on mushroom/flower pickup and on taking damage
ADDR_SCORE_START = None   # confirm: multi-byte score field, verify byte order empirically


def read_mario_x(pyboy) -> int:
    assert ADDR_MARIO_X is not None, "ADDR_MARIO_X not yet confirmed -- see Task 3 Step 2"
    return pyboy.memory[ADDR_MARIO_X]


def read_mario_y(pyboy) -> int:
    assert ADDR_MARIO_Y is not None, "ADDR_MARIO_Y not yet confirmed -- see Task 3 Step 2"
    return pyboy.memory[ADDR_MARIO_Y]


def read_lives(pyboy) -> int:
    assert ADDR_LIVES is not None, "ADDR_LIVES not yet confirmed -- see Task 3 Step 2"
    return pyboy.memory[ADDR_LIVES]


def read_timer(pyboy) -> int:
    assert ADDR_TIMER_ONES is not None, "timer addresses not yet confirmed -- see Task 3 Step 2"
    h = pyboy.memory[ADDR_TIMER_HUNDREDS]
    t = pyboy.memory[ADDR_TIMER_TENS]
    o = pyboy.memory[ADDR_TIMER_ONES]
    return h * 100 + t * 10 + o


def read_powerup_state(pyboy) -> int:
    assert ADDR_POWERUP_STATE is not None, "ADDR_POWERUP_STATE not yet confirmed -- see Task 3 Step 2"
    return pyboy.memory[ADDR_POWERUP_STATE]


def read_score(pyboy) -> int:
    assert ADDR_SCORE_START is not None, "ADDR_SCORE_START not yet confirmed -- see Task 3 Step 2"
    b0, b1, b2 = (pyboy.memory[ADDR_SCORE_START + i] for i in range(3))
    return b0 * 65536 + b1 * 256 + b2
