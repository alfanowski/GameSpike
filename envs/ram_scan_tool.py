"""Standalone tool (not imported by production code) to empirically discover
Super Mario Land's RAM addresses for a given metric.

Usage: run interactively while manually playing, or scripted by holding a
fixed input sequence, and it prints addresses whose value changed in the way
you'd expect for the metric under test (e.g. monotonically increasing while
holding right on a flat, obstacle-free stretch of 1-1).

    python -m envs.ram_scan_tool --rom /path/to/rom.gb --hold right --frames 120

This does NOT commit any address to envs/ram_map.py automatically -- a human
must read the candidate list, cross-check it against a public disassembly
reference for Super Mario Land, and hand-write the confirmed address into
ram_map.py. That manual confirmation step, and the invariant tests in
tests/test_ram_map_invariants.py, are what makes an address "locked."
"""
import argparse
from pyboy import PyBoy

WRAM_START = 0xC000
WRAM_END = 0xE000  # PyBoy/Game Boy work RAM range


def snapshot(pyboy):
    return bytes(pyboy.memory[WRAM_START:WRAM_END])


def scan_for_monotonic_increase(rom_path: str, hold: str, frames: int, settle_frames: int = 30):
    pyboy = PyBoy(rom_path, window="null")
    pyboy.set_emulation_speed(0)
    for _ in range(settle_frames):
        pyboy.tick()
    before = snapshot(pyboy)
    pyboy.button_press(hold)
    for _ in range(frames):
        pyboy.tick()
    pyboy.button_release(hold)
    after = snapshot(pyboy)
    candidates = []
    for offset in range(len(before)):
        b, a = before[offset], after[offset]
        if a > b:  # single-byte monotonic increase candidate
            candidates.append((WRAM_START + offset, b, a))
    pyboy.stop(save=False)
    return candidates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rom", required=True)
    parser.add_argument("--hold", default="right")
    parser.add_argument("--frames", type=int, default=120)
    args = parser.parse_args()
    for addr, before, after in scan_for_monotonic_increase(args.rom, args.hold, args.frames):
        print(f"0x{addr:04X}: {before} -> {after}")
