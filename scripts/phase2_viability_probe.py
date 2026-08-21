"""Roadmap Phase 2 testbed-viability probe (docs/DESIGN_ROADMAP_PHASE2.md §13).

Standalone, read-only with respect to the repository: it trains nothing, writes no
checkpoint, and imports production code only to reuse already-confirmed constants.
It answers the seven questions §13 says must be answered BEFORE any Phase 2 code is
built on top of them, because every one of them fails *silently* if assumed:

  SML-1  does game_wrapper.start_game(world_level=...) really load the level, as
         opposed to only making the status bar draw it?
  SML-2  does a save_state() captured at a level start reload bit-identically, so
         the environment stays deterministic (RESULTS.md §9 rests on this)?
  SML-3  how much of read_level_progress() in the two vehicle stages is FREE --
         i.e. accrues while pressing nothing? That number decides whether Phase 1's
         progress reward can be reused there at all (§4.1.1).
  SML-4  do ADDR_MARIO_X / ADDR_LEVEL_BLOCK / the timer / the on-ground heuristic
         still mean what envs/ram_map.py says in the vehicle stages (§4.4)?
  KDL-1  does PyBoy's Kirby wrapper bind to this cartridge and reach live gameplay?
  KDL-2  do Kirby's published addresses behave as documented (§5.2, §5.3)?
  KDL-3  does a level-progress signal COMPOSE for Kirby the way it does for Mario --
         unwrapped camera coarse part plus local X fine part (§6.1 slot 0)?

Every check is behavioural. "The HUD says 2-3" is not accepted as evidence that
level 2-3 loaded; "a submarine sprite is on screen" is. That is the same standard
envs/ram_map.py's docstring holds its own addresses to.

Usage:
    python -m scripts.phase2_viability_probe \
        --mario-rom "/path/to/Super Mario Land (World).gb" \
        --kirby-rom "/path/to/Kirby's Dream Land (USA, Europe).gb"

Costs seconds of emulation, not hours. It is NOT a training script and must never
grow into one.
"""
import argparse
import hashlib
import io
import json
import sys

from envs import ram_map

# --- Super Mario Land ---------------------------------------------------------
# Sprite tile identifiers, taken from PyBoy's own Super Mario Land game wrapper
# (pyboy/plugins/game_wrapper_super_mario_land.py). They are an INDEPENDENT channel
# from any RAM read: if a submarine sprite is on screen, level 2-3 genuinely loaded,
# regardless of what the status bar or 0xFFB4 claim.
SML_PLANE_TILES = list(range(99, 110))
SML_SUBMARINE_TILES = list(range(112, 122))
SML_MARIO_TILES = list(range(81))

# --- Kirby's Dream Land -------------------------------------------------------
# From PyBoy's own Kirby wrapper (score/health/lives) and DataCrystal's RAM map
# (position/scroll). PUBLISHED, THEREFORE HYPOTHESES -- envs/ram_map.py:63-66.
KDL_ADDR_SCORE_START = 0xD070   # 4 bytes, wrapper decodes as base-10 digits
KDL_ADDR_HEALTH = 0xD086
KDL_ADDR_LIVES = 0xD089
KDL_ADDR_X = 0xD05C             # DataCrystal
KDL_ADDR_Y = 0xD05D             # DataCrystal
KDL_ADDR_SCROLL_X = 0xD051      # DataCrystal

WRAM = (0xC000, 0xE000)
HRAM = (0xFF80, 0x10000)


def _new_pyboy(rom_path):
    from pyboy import PyBoy
    pyboy = PyBoy(rom_path, window="null")
    pyboy.set_emulation_speed(0)
    return pyboy


def _hold(pyboy, buttons, frames):
    """Hold a set of buttons for `frames` ticks, always releasing them."""
    for b in buttons:
        pyboy.button_press(b)
    try:
        for _ in range(frames):
            pyboy.tick(1, False)
    finally:
        for b in buttons:
            pyboy.button_release(b)


def _ram_digest(pyboy):
    """A hash over all of WRAM+HRAM -- the state that gameplay actually depends on."""
    h = hashlib.sha256()
    for lo, hi in (WRAM, HRAM):
        h.update(bytes(pyboy.memory[lo:hi]))
    return h.hexdigest()


# ==============================================================================
# SML-1 / SML-3 / SML-4
# ==============================================================================

def _sml_start_at(rom_path, world_level):
    """Boot Super Mario Land straight into `world_level` via PyBoy's own wrapper."""
    pyboy = _new_pyboy(rom_path)
    pyboy.game_wrapper.start_game(world_level=world_level, timer_div=0)
    return pyboy


def _sprite_present(pyboy, tile_ids):
    found = pyboy.get_sprite_by_tile_identifier(tile_ids, on_screen=True)
    return sum(len(group) for group in found)


def check_sml_levels_really_load(rom_path, levels):
    """SML-1: independent, non-HUD evidence that each requested level loaded."""
    out = []
    for wl in levels:
        pyboy = _sml_start_at(rom_path, wl)
        try:
            pyboy.tick(60, False)  # let the level settle
            hud = ram_map.read_world_level(pyboy)
            row = {
                "requested": list(wl),
                "ram_world_level": list(hud),
                "ram_agrees": list(hud) == list(wl),
                "mario_sprites": _sprite_present(pyboy, SML_MARIO_TILES),
                "submarine_sprites": _sprite_present(pyboy, SML_SUBMARINE_TILES),
                "plane_sprites": _sprite_present(pyboy, SML_PLANE_TILES),
                "level_block_at_start": ram_map.read_level_block(pyboy),
                "progress_at_start": ram_map.read_level_progress(pyboy),
                "timer_at_start": ram_map.read_timer(pyboy),
                "lives_at_start": ram_map.read_lives(pyboy),
            }
            # The independent verdict: a vehicle stage must show its vehicle.
            if tuple(wl) == (2, 3):
                row["independent_evidence"] = "submarine" if row["submarine_sprites"] else "NONE"
            elif tuple(wl) == (4, 3):
                row["independent_evidence"] = "plane" if row["plane_sprites"] else "NONE"
            else:
                row["independent_evidence"] = "mario" if row["mario_sprites"] else "NONE"
            out.append(row)
        finally:
            pyboy.stop(save=False)
    return out


def check_sml_autoscroll(rom_path, levels, frames=240):
    """SML-3: how much progress accrues while pressing NOTHING, vs. holding right."""
    out = []
    for wl in levels:
        row = {"level": list(wl), "frames": frames}
        for label, buttons in (("idle", []), ("hold_right", ["right"])):
            pyboy = _sml_start_at(rom_path, wl)
            try:
                pyboy.tick(60, False)
                before = ram_map.read_level_progress(pyboy)
                before_block = ram_map.read_level_block(pyboy)
                before_timer = ram_map.read_timer(pyboy)
                _hold(pyboy, buttons, frames)
                row[label] = {
                    "progress_delta": ram_map.read_level_progress(pyboy) - before,
                    "level_block_delta": ram_map.read_level_block(pyboy) - before_block,
                    "timer_delta": ram_map.read_timer(pyboy) - before_timer,
                    "mario_x": ram_map.read_mario_x(pyboy),
                    "mario_y": ram_map.read_mario_y(pyboy),
                }
            finally:
                pyboy.stop(save=False)
        idle = row["idle"]["progress_delta"]
        held = row["hold_right"]["progress_delta"]
        row["free_fraction"] = (idle / held) if held else None
        out.append(row)
    return out


def check_sml_axis_control(rom_path, levels, frames=60):
    """SML-4: does the player still drive 0xC202 (X) and 0xC201 (Y) in this level?

    In the vehicle stages the craft flies freely in 2D, so `up`/`down` should move
    Y -- something that is meaningless in a platformer level and is exactly why
    §6.3 has to widen the action space.
    """
    out = []
    for wl in levels:
        row = {"level": list(wl)}
        for label, buttons in (("right", ["right"]), ("left", ["left"]),
                               ("up", ["up"]), ("down", ["down"])):
            pyboy = _sml_start_at(rom_path, wl)
            try:
                pyboy.tick(60, False)
                x0, y0 = ram_map.read_mario_x(pyboy), ram_map.read_mario_y(pyboy)
                _hold(pyboy, buttons, frames)
                row[label] = {
                    "dx": ram_map.read_mario_x(pyboy) - x0,
                    "dy": ram_map.read_mario_y(pyboy) - y0,
                }
            finally:
                pyboy.stop(save=False)
        out.append(row)
    return out


# ==============================================================================
# SML-2 -- determinism, the check the whole statistical design rests on
# ==============================================================================

def check_savestate_determinism(rom_path, world_level, replay_frames=300, trials=3):
    """SML-2: does a state captured at a level start replay bit-identically?

    Three separate questions, because they can fail independently:
      same_process  -- reload into the SAME emulator, replay, compare RAM digests
      fresh_process -- reload into a BRAND-NEW emulator, replay, compare
      boot_repeat   -- boot from power-on twice and compare, i.e. is start_game
                       itself deterministic (PyBoy randomises timer DIV by default)
    """
    script = [([], 20), (["right"], 60), (["right", "a"], 30), ([], 20),
              (["left"], 40), (["a"], 30), ([], 20), (["right", "b"], 60), ([], 20)]

    def replay(pyboy):
        for buttons, n in script:
            _hold(pyboy, buttons, n)
        return _ram_digest(pyboy)

    pyboy = _sml_start_at(rom_path, world_level)
    try:
        pyboy.tick(60, False)
        buf = io.BytesIO()
        pyboy.save_state(buf)
        state_bytes = buf.getvalue()
        same_process = []
        for _ in range(trials):
            pyboy.load_state(io.BytesIO(state_bytes))
            same_process.append(replay(pyboy))
    finally:
        pyboy.stop(save=False)

    fresh_process = []
    for _ in range(trials):
        p = _new_pyboy(rom_path)
        try:
            p.load_state(io.BytesIO(state_bytes))
            fresh_process.append(replay(p))
        finally:
            p.stop(save=False)

    boot_repeat = []
    for _ in range(trials):
        p = _sml_start_at(rom_path, world_level)
        try:
            p.tick(60, False)
            boot_repeat.append(replay(p))
        finally:
            p.stop(save=False)

    return {
        "level": list(world_level),
        "replay_frames": sum(n for _, n in script),
        "state_bytes": len(state_bytes),
        "same_process_identical": len(set(same_process)) == 1,
        "fresh_process_identical": len(set(fresh_process)) == 1,
        "same_matches_fresh": set(same_process) == set(fresh_process),
        "boot_repeat_identical": len(set(boot_repeat)) == 1,
        "boot_matches_savestate": set(boot_repeat) == set(same_process),
        "digests": {
            "same_process": sorted(set(same_process)),
            "fresh_process": sorted(set(fresh_process)),
            "boot_repeat": sorted(set(boot_repeat)),
        },
    }


# ==============================================================================
# KDL-1 / KDL-2 / KDL-3
# ==============================================================================

def check_kirby_wrapper(rom_path):
    """KDL-1: does PyBoy's Kirby wrapper bind, and does start_game reach gameplay?"""
    pyboy = _new_pyboy(rom_path)
    try:
        wrapper = pyboy.game_wrapper
        info = {
            "wrapper_class": type(wrapper).__name__ if wrapper is not None else None,
            "wrapper_bound": type(wrapper).__name__ == "GameWrapperKirbyDreamLand",
        }
        if not info["wrapper_bound"]:
            return info
        wrapper.start_game(timer_div=0)
        pyboy.tick(30, False)
        info.update({
            "score": wrapper.score,
            "health": wrapper.health,
            "lives_left": wrapper.lives_left,
            "game_over": wrapper.game_over(),
        })
        # Behavioural gate, the same standard envs/boot.py holds Mario to: the pad
        # must actually move the player, or this is a menu/demo and not gameplay.
        x0 = pyboy.memory[KDL_ADDR_X]
        _hold(pyboy, ["right"], 40)
        x_right = pyboy.memory[KDL_ADDR_X]
        _hold(pyboy, ["left"], 60)
        x_left = pyboy.memory[KDL_ADDR_X]
        info.update({
            "x_at_start": x0, "x_after_right": x_right, "x_after_left": x_left,
            "player_has_control": (x_right != x0) and (x_left != x_right),
        })
        return info
    finally:
        pyboy.stop(save=False)


def _kirby_boot(rom_path):
    pyboy = _new_pyboy(rom_path)
    pyboy.game_wrapper.start_game(timer_div=0)
    pyboy.tick(30, False)
    return pyboy


def check_kirby_addresses(rom_path, frames=60):
    """KDL-2: do the published Kirby addresses behave the way they are documented?"""
    pyboy = _kirby_boot(rom_path)
    try:
        def snap():
            return {
                "x": pyboy.memory[KDL_ADDR_X],
                "y": pyboy.memory[KDL_ADDR_Y],
                "scroll_x": pyboy.memory[KDL_ADDR_SCROLL_X],
                "health": pyboy.memory[KDL_ADDR_HEALTH],
                "lives": pyboy.memory[KDL_ADDR_LIVES],
                "score_bytes": [pyboy.memory[KDL_ADDR_SCORE_START + i] for i in range(4)],
            }
        out = {"at_start": snap()}
        for label, buttons in (("hold_right", ["right"]), ("hold_left", ["left"]),
                               ("hold_up", ["up"]), ("hold_down", ["down"]),
                               ("hold_a", ["a"])):
            before = snap()
            _hold(pyboy, buttons, frames)
            after = snap()
            out[label] = {
                "dx": after["x"] - before["x"],
                "dy": after["y"] - before["y"],
                "dscroll": after["scroll_x"] - before["scroll_x"],
                "health": after["health"],
                "lives": after["lives"],
            }
        out["wrapper_agrees"] = {
            "wrapper_health": pyboy.game_wrapper.health,
            "ram_health": pyboy.memory[KDL_ADDR_HEALTH],
            "wrapper_lives": pyboy.game_wrapper.lives_left,
            "ram_lives_raw": pyboy.memory[KDL_ADDR_LIVES],
        }
        return out
    finally:
        pyboy.stop(save=False)


def check_kirby_progress_composes(rom_path, frames=900):
    """KDL-3: is there an unwrapped camera counter to compose a progress signal from?

    Method, mirroring how ADDR_LEVEL_BLOCK was found for Mario: hold right for a
    long run, record whether the documented scroll byte wraps, and independently
    scan every WRAM/HRAM byte for one that never decreased and ended higher. A
    coarse counter that survives the wrap is what slot 0 of §6.1's schema needs.
    """
    pyboy = _kirby_boot(rom_path)
    try:
        addresses = list(range(*WRAM)) + list(range(*HRAM))
        first = [pyboy.memory[a] for a in addresses]
        prev = list(first)
        alive = [True] * len(addresses)
        scroll_series = []
        wraps = 0
        prev_scroll = pyboy.memory[KDL_ADDR_SCROLL_X]

        pyboy.button_press("right")
        try:
            for _ in range(frames):
                pyboy.tick(1, False)
                cur = [pyboy.memory[a] for a in addresses]
                for i in range(len(addresses)):
                    if alive[i] and cur[i] < prev[i]:
                        alive[i] = False
                prev = cur
                s = pyboy.memory[KDL_ADDR_SCROLL_X]
                if s < prev_scroll - 128:
                    wraps += 1
                prev_scroll = s
                scroll_series.append(s)
        finally:
            pyboy.button_release("right")

        candidates = [
            {"addr": f"0x{addresses[i]:04X}",
             "region": "HRAM" if addresses[i] >= HRAM[0] else "WRAM",
             "start": first[i], "end": prev[i], "delta": prev[i] - first[i]}
            for i in range(len(addresses)) if alive[i] and prev[i] > first[i]
        ]
        candidates.sort(key=lambda c: -c["delta"])
        return {
            "frames": frames,
            "scroll_wraps_observed": wraps,
            "scroll_start": scroll_series[0] if scroll_series else None,
            "scroll_end": scroll_series[-1] if scroll_series else None,
            "monotonic_candidate_count": len(candidates),
            "monotonic_candidates": candidates[:25],
        }
    finally:
        pyboy.stop(save=False)


def check_kirby_traversal(rom_path, scripted_frames=2400, random_steps=2000, seeds=4):
    """KDL-3b: can ANY simple policy traverse Kirby's first level?

    Added after the first probe run found `hold right` stalling. This is the
    question that actually decides Phase 2b's cost: Mario's RAM map was discovered
    with a blind hold-right scan, and that method only works if holding right makes
    the player travel. `scroll_x` is the progress proxy under test.
    """
    import numpy as np

    def scripted(script, frames):
        pyboy = _kirby_boot(rom_path)
        try:
            held, best = set(), pyboy.memory[KDL_ADDR_SCROLL_X]
            for f in range(frames):
                want = set(script(f))
                for b in want - held:
                    pyboy.button_press(b)
                for b in held - want:
                    pyboy.button_release(b)
                held = want
                pyboy.tick(1, False)
                best = max(best, pyboy.memory[KDL_ADDR_SCROLL_X])
            for b in held:
                pyboy.button_release(b)
            return {"max_scroll": best, "final_scroll": pyboy.memory[KDL_ADDR_SCROLL_X],
                    "x": pyboy.memory[KDL_ADDR_X], "score": pyboy.game_wrapper.score,
                    "health": pyboy.game_wrapper.health}
        finally:
            pyboy.stop(save=False)

    out = {"scripted": {
        "hold_right": scripted(lambda f: ["right"], scripted_frames),
        "right_plus_jump": scripted(
            lambda f: ["right", "a"] if f % 40 < 6 else ["right"], scripted_frames),
        "right_plus_fly": scripted(
            lambda f: ["right", "up"] if f % 16 < 8 else ["right"], scripted_frames),
    }, "random": []}

    # The proposed §6.3 union action space, exercised as an RL agent would at init.
    dirs = [[], ["left"], ["right"], ["up"], ["down"]]
    mods = [[], ["a"], ["b"], ["a", "b"]]
    actions = [d + m for d in dirs for m in mods]
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        pyboy = _kirby_boot(rom_path)
        try:
            best = pyboy.memory[KDL_ADDR_SCROLL_X]
            for _ in range(random_steps):
                buttons = actions[int(rng.integers(0, len(actions)))]
                for b in buttons:
                    pyboy.button_press(b)
                for _ in range(4):  # env frame_skip
                    pyboy.tick(1, False)
                for b in buttons:
                    pyboy.button_release(b)
                best = max(best, pyboy.memory[KDL_ADDR_SCROLL_X])
                if pyboy.game_wrapper.game_over():
                    break
            out["random"].append({
                "seed": seed, "max_scroll": best,
                "score": pyboy.game_wrapper.score,
                "health": pyboy.game_wrapper.health,
                "game_over": bool(pyboy.game_wrapper.game_over()),
            })
        finally:
            pyboy.stop(save=False)
    out["union_action_space_size"] = len(actions)
    return out


def check_sml_hold_right_survival(rom_path, levels, frames=480):
    """Characterises the SML-3 surprise: holding right in 2-1 scored zero progress.

    It was a death-and-reload artefact, not a broken level, and the difference in
    *how fast* a naive rightward policy dies is itself a task-difficulty signal
    worth having before 2-1 is adopted as the near-transfer task.
    """
    out = []
    for wl in levels:
        pyboy = _sml_start_at(rom_path, wl)
        try:
            pyboy.tick(60, False)
            lives0 = ram_map.read_lives(pyboy)
            peak = ram_map.read_level_progress(pyboy)
            first_death = None
            pyboy.button_press("right")
            try:
                for f in range(frames):
                    pyboy.tick(1, False)
                    peak = max(peak, ram_map.read_level_progress(pyboy))
                    if first_death is None and ram_map.read_lives(pyboy) < lives0:
                        first_death = f
            finally:
                pyboy.button_release("right")
            out.append({
                "level": list(wl), "frames": frames, "lives_at_start": lives0,
                "lives_at_end": ram_map.read_lives(pyboy),
                "first_death_frame": first_death,
                "peak_progress": peak,
                "peak_progress_gain": peak - 242,
            })
        finally:
            pyboy.stop(save=False)
    return out


def check_timer_div_determinism(rom_path, world_level=(1, 1), trials=4):
    """Does PyBoy's DIV randomisation (timer_div=None, the default) break determinism?

    Checked rather than assumed, because pinning it would otherwise be cargo-cult.
    """
    script = [([], 20), (["right"], 60), (["right", "a"], 30), ([], 20),
              (["left"], 40), (["a"], 30), ([], 20), (["right", "b"], 60), ([], 20)]
    result = {}
    for label, td in (("pinned_0", 0), ("randomised_none", None)):
        digests = []
        for _ in range(trials):
            pyboy = _new_pyboy(rom_path)
            try:
                pyboy.game_wrapper.start_game(world_level=world_level, timer_div=td)
                pyboy.tick(60, False)
                for buttons, n in script:
                    _hold(pyboy, buttons, n)
                digests.append(_ram_digest(pyboy))
            finally:
                pyboy.stop(save=False)
        result[label] = {"trials": trials, "distinct": len(set(digests)),
                         "deterministic": len(set(digests)) == 1,
                         "digest": sorted(set(digests))[0][:16]}
    result["pinning_changes_outcome"] = (
        result["pinned_0"]["digest"] != result["randomised_none"]["digest"])
    return result


# ==============================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mario-rom", required=True)
    parser.add_argument("--kirby-rom", required=True)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    levels = [(1, 1), (2, 1), (2, 3), (4, 3)]
    report = {
        "SML-1_levels_really_load": check_sml_levels_really_load(args.mario_rom, levels),
        "SML-3_autoscroll": check_sml_autoscroll(args.mario_rom, levels),
        "SML-4_axis_control": check_sml_axis_control(args.mario_rom, levels),
        "SML-2_determinism": [
            check_savestate_determinism(args.mario_rom, (1, 1)),
            check_savestate_determinism(args.mario_rom, (2, 3)),
        ],
        "SML-2b_timer_div": check_timer_div_determinism(args.mario_rom),
        "SML-5_hold_right_survival": check_sml_hold_right_survival(args.mario_rom, levels),
        "KDL-1_wrapper": check_kirby_wrapper(args.kirby_rom),
        "KDL-2_addresses": check_kirby_addresses(args.kirby_rom),
        "KDL-3_progress_composes": check_kirby_progress_composes(args.kirby_rom),
        "KDL-3b_traversal": check_kirby_traversal(args.kirby_rom),
    }

    text = json.dumps(report, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
