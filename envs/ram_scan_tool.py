"""Standalone tool (not imported by production code) to empirically discover
Super Mario Land's RAM addresses for a given metric.

Run it holding a fixed input and it prints the addresses whose value moved the
way you'd expect for the metric under test (e.g. never decreasing while holding
right on a flat stretch of 1-1).

    python -m envs.ram_scan_tool --rom /path/to/rom.gb --hold right --frames 240

Two things this deliberately does that a naive scanner does not:

  * It reaches real gameplay first, via envs/boot.py, which verifies the player
    is actually in control. Booting and ticking leaves the game on the title
    screen and then in its own attract-mode demo, where the ROM drives Mario and
    your held button does nothing -- a scan run that way reports pure boot-time
    initialisation noise and no real candidates at all.
  * It scans HRAM (0xFF80-0xFFFF) as well as WRAM. Two of the addresses this
    project depends on -- the powerup status and the camera scroll -- live in
    HRAM, so a WRAM-only sweep would silently never see them.

Note the ordering dependency: the boot gate uses an already-confirmed address
(Mario's X) to prove the pad is live, so this tool cannot be used to discover
that first address. It was found by hand; everything after it can use this.

It still does NOT commit anything to envs/ram_map.py. A human must read the
candidate list, confirm the address by driving the live game into a state where
a *predicted* change has to appear there, and hand-write the result. That manual
confirmation, plus the invariant tests in tests/test_ram_map_invariants.py, is
what makes an address "locked."
"""
import argparse

from envs import boot

WRAM_START = 0xC000
WRAM_END = 0xE000  # PyBoy/Game Boy work RAM range
HRAM_START = 0xFF80
HRAM_END = 0x10000  # high RAM + the interrupt-enable byte


def scanned_addresses():
    """Every address this tool watches, in order."""
    return list(range(WRAM_START, WRAM_END)) + list(range(HRAM_START, HRAM_END))


def snapshot(pyboy, addresses):
    return [pyboy.memory[a] for a in addresses]


def scan_for_monotonic_increase(rom_path: str, hold: str, frames: int):
    """Addresses that never decreased, and ended higher, while `hold` was held.

    Sampling every frame (rather than just comparing the first and last frame)
    is what separates a real progress-like counter from a byte that merely
    happened to land on a bigger value -- animation counters and the like churn
    up and down constantly and are filtered out here.
    """
    pyboy = boot.boot_to_level_start(rom_path)

    addresses = scanned_addresses()
    first = snapshot(pyboy, addresses)
    prev = first
    alive = [True] * len(addresses)  # still monotonically non-decreasing

    pyboy.button_press(hold)
    for _ in range(frames):
        pyboy.tick()
        cur = snapshot(pyboy, addresses)
        for i, (p, c) in enumerate(zip(prev, cur)):
            if alive[i] and c < p:
                alive[i] = False
        prev = cur
    pyboy.button_release(hold)
    pyboy.stop(save=False)

    return [(addresses[i], first[i], prev[i])
            for i in range(len(addresses))
            if alive[i] and prev[i] > first[i]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rom", required=True)
    parser.add_argument("--hold", default="right")
    parser.add_argument("--frames", type=int, default=240)
    args = parser.parse_args()
    candidates = scan_for_monotonic_increase(args.rom, args.hold, args.frames)
    print(f"{len(candidates)} monotonically-increasing candidates while holding "
          f"'{args.hold}' for {args.frames} frames:")
    for addr, before, after in candidates:
        region = "HRAM" if addr >= HRAM_START else "WRAM"
        print(f"  0x{addr:04X} [{region}]: {before} -> {after}")
