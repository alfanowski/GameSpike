# Spiking Reservoir RL — Mario Land PPO (Phase 0+1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and scientifically compare (mandatory-control design) a frozen-spiking-reservoir PPO agent against a matched-parameter trained-GRU baseline on Super Mario Land (Game Boy), answering the core question: does the reservoir contribute anything over a conventional trained RL feature extractor at the same parameter budget?

**Architecture:** RAM-state observation → small trainable embedding → [frozen spiking reservoir (TT-native, vendored from `spiking-reservoir-lm`) OR trained GRU baseline] → windowed causal-attention actor/critic readout → PPO. A trajectory-novelty write-gate produces an intrinsic curiosity reward from the feature-extractor's own state trajectory, included identically in both arms.

**Tech Stack:** Python 3.9, PyTorch, `snntorch` (LIF reservoir), PyBoy 2.x (headless Game Boy emulation), Gymnasium (env interface), pytest.

**Spec:** `~/Desktop/Projects/GameSpike/docs/DESIGN.md` — this plan implements Phase 0 (§7.1, environment plumbing) and Phase 1 (§7.2, reservoir vs. baseline core comparison) only. Phase 2 (resonate-and-fire ablation) and Phase 3 (DLIF/RSSR stretch) are explicitly out of scope for this plan — separate future plans, per the spec's own build order.

## Global Constraints

- The reservoir's parameters are NEVER trained: zero `nn.Parameter`s on `SpikingReservoir`, optimizer built only over embedding/readout/actor/critic components, verified by a runtime tripwire (spec §3, carried over from `spiking-reservoir-lm`'s own invariant).
- Mandatory scientific control (spec §5): baseline and reservoir arms MUST have matched trainable-parameter counts (tolerance: within 10%), verified by an explicit test, not eyeballed.
- No hardcoded/assumed RAM addresses (spec §2): all Super Mario Land RAM addresses used by this codebase must be empirically verified against the actual ROM before being committed to `envs/ram_map.py`.
- No cloud GPU rental (spec §2, §10): local M4 development is the default; Kaggle (credentials already at `~/.kaggle/kaggle.json`) is available but not required for this plan's scope.
- Every training/rollout script must checkpoint and be resumable (spec §6) — no long-running script may assume it completes uninterrupted.
- The ROM file itself is never committed to git (copyright) — loaded via a path the user supplies (env var `MARIO_LAND_ROM_PATH` or CLI flag), and `.gitignore` must exclude `*.gb`.

---

## Task 1: Project scaffolding and vendored reservoir core

**Files:**
- Create: `GameSpike/.gitignore`
- Create: `GameSpike/.python-version`
- Create: `GameSpike/requirements.txt`
- Create: `GameSpike/README.md`
- Create: `GameSpike/LICENSE`
- Create: `GameSpike/models/__init__.py`
- Create: `GameSpike/models/spiking_reservoir.py` (vendored, unmodified, from `spiking-reservoir-lm/models/spiking_reservoir.py`)
- Create: `GameSpike/models/baseline_transformer.py` (vendored, unmodified, from `spiking-reservoir-lm/models/baseline_transformer.py` — provides the `Block` causal-attention module later tasks reuse)
- Test: `GameSpike/tests/test_vendored_reservoir.py`

**Interfaces:**
- Produces: `SpikingReservoir` class (constructor signature: `SpikingReservoir(vocab_size=256, reservoir_size=2048, spectral_radius=1.0, beta=0.9, seed=0, use_tensor_train=False, tt_rank=8, tt_n_cores=4, tt_core_std=None, input_dim=None, soft_spike=False, soft_spike_sigma_frac=0.1)`); methods `.step(x_t, mem, spk) -> (spk_next, mem_next)`, `.readout_feature(spk, mem) -> tensor`, `.reservoir_size`, `.input_dim`. `Block` class (constructor `Block(dim, n_heads, context_len)`, `.forward(x) -> x` for `x: (B, T, dim)`).

- [ ] **Step 1: Create the repo skeleton and non-code scaffolding**

```bash
cd ~/Desktop/Projects/GameSpike
mkdir -p models envs training tests checkpoints
touch models/__init__.py envs/__init__.py training/__init__.py tests/__init__.py
git init
```

`.gitignore`:
```
__pycache__/
*.pyc
.venv/
checkpoints/*.pt
checkpoints/*.json
*.gb
*.gbc
.pytest_cache/
.DS_Store
```

`.python-version`:
```
3.9
```

`requirements.txt`:
```
torch>=2.8.0
snntorch>=0.9.1
pyboy>=2.0
gymnasium>=0.29
numpy
pytest
```

`LICENSE`: copy verbatim from `~/Desktop/Projects/spike/spiking-reservoir-lm/LICENSE` (Apache License 2.0, matching the sibling project).

**Status note (partially complete — read before executing this task):**
`README.md`, `LICENSE`, `.gitignore`, `.python-version`, `requirements.txt`, and
`docs/DESIGN.md` already exist and are committed/pushed to
`https://github.com/alfanowski/GameSpike` (public, repo root — do not recreate
them; read the real `README.md` rather than reconstructing it from a draft).
**Still NOT done, still owed by this task's remaining steps:** the
`models/`, `envs/`, `training/`, `tests/`, `checkpoints/` package directories and
their `__init__.py` files, `git init` inside them (the repo root is already a git
repo, so skip that specific sub-step), the vendored `spiking_reservoir.py` /
`baseline_transformer.py`, and the vendoring-verification smoke test. Start Step 1
from the `mkdir -p models envs training tests checkpoints` line onward, skip the
already-created files, then proceed through Steps 2-5 as written.

## Running the test suite

\`\`\`bash
python -m pytest tests/ -q
\`\`\`
```

- [ ] **Step 2: Vendor the reservoir core, unmodified**

```bash
cp ~/Desktop/Projects/spike/spiking-reservoir-lm/models/spiking_reservoir.py \
   ~/Desktop/Projects/GameSpike/models/spiking_reservoir.py
cp ~/Desktop/Projects/spike/spiking-reservoir-lm/models/baseline_transformer.py \
   ~/Desktop/Projects/GameSpike/models/baseline_transformer.py
```

Do not edit either file in this task. They are reused as-is (per design doc §4);
later tasks build new modules that *import* from them, never modify them in place.

- [ ] **Step 3: Write the vendoring-verification smoke test**

```python
# tests/test_vendored_reservoir.py
import torch
from models.spiking_reservoir import SpikingReservoir
from models.baseline_transformer import Block


def test_reservoir_has_zero_trainable_parameters():
    res = SpikingReservoir(reservoir_size=256, input_dim=16, use_tensor_train=True,
                            tt_rank=4, tt_n_cores=2, seed=0)
    trainable = [p for p in res.parameters() if p.requires_grad]
    assert trainable == [], "vendored reservoir must have zero trainable parameters"


def test_reservoir_step_shapes_and_dtype():
    B, input_dim, N = 4, 16, 256
    res = SpikingReservoir(reservoir_size=N, input_dim=input_dim, use_tensor_train=True,
                            tt_rank=4, tt_n_cores=2, seed=0)
    mem = torch.zeros(B, N)
    spk = torch.zeros(B, N)
    x_t = torch.randn(B, input_dim)
    spk_next, mem_next = res.step(x_t, mem, spk)
    assert spk_next.shape == (B, N)
    assert mem_next.shape == (B, N)
    feat = res.readout_feature(spk_next, mem_next)
    assert feat.shape == (B, N)


def test_block_forward_shape():
    B, T, dim = 2, 8, 32
    block = Block(dim=dim, n_heads=4, context_len=16)
    x = torch.randn(B, T, dim)
    out = block(x)
    assert out.shape == (B, T, dim)
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `cd GameSpike && python -m pytest tests/test_vendored_reservoir.py -v`
Expected: `3 passed`. If any fail, the vendored copy diverged from the source file —
re-copy rather than debug (this is verbatim-reuse code, not new logic).

- [ ] **Step 5: Commit**

```bash
git add .gitignore .python-version requirements.txt README.md LICENSE \
        models/ tests/test_vendored_reservoir.py envs/__init__.py training/__init__.py
git commit -m "feat: scaffold project, vendor frozen reservoir core from spiking-reservoir-lm"
```

---

## Task 2: PyBoy environment skeleton

**Files:**
- Create: `envs/mario_land_env.py`
- Test: `tests/test_mario_land_env_skeleton.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `MarioLandEnv(gymnasium.Env)` class, constructor `MarioLandEnv(rom_path: str, headless: bool = True, frame_skip: int = 4)`; `.reset() -> (obs, info)`; `.step(action: int) -> (obs, reward, terminated, truncated, info)`; `.close()`. In this task `obs` is a placeholder zero-vector and `reward` is always `0.0` — real observation/reward logic is Task 4. This task only proves the emulator boots headless and accepts input without crashing.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mario_land_env_skeleton.py
import os
import pytest
from envs.mario_land_env import MarioLandEnv

ROM_PATH = os.environ.get("MARIO_LAND_ROM_PATH")

pytestmark = pytest.mark.skipif(
    not ROM_PATH or not os.path.exists(ROM_PATH),
    reason="MARIO_LAND_ROM_PATH not set or file missing; set it to your own legally-dumped ROM",
)


def test_reset_returns_correct_shape():
    env = MarioLandEnv(rom_path=ROM_PATH)
    obs, info = env.reset()
    assert obs.shape == (env.observation_space.shape[0],)
    env.close()


def test_step_runs_without_crashing():
    env = MarioLandEnv(rom_path=ROM_PATH)
    env.reset()
    for action in range(env.action_space.n):
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (env.observation_space.shape[0],)
        assert isinstance(reward, float)
    env.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mario_land_env_skeleton.py -v`
Expected: FAIL (`ModuleNotFoundError` or `ImportError`, `envs.mario_land_env` does
not exist yet). If `MARIO_LAND_ROM_PATH` is unset, the test skips instead — that's
expected too; set the env var to your own ROM to actually exercise this.

- [ ] **Step 3: Write the minimal environment skeleton**

```python
# envs/mario_land_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pyboy import PyBoy

OBS_DIM = 12  # locked in Task 4; placeholder zero-vector until then
N_ACTIONS = 10  # locked in Task 5; placeholder discrete space until then


class MarioLandEnv(gym.Env):
    """Gymnasium-style wrapper around PyBoy running Super Mario Land.

    Observation and reward are placeholders in this skeleton (Task 2) --
    verified to boot headless and accept input without crashing. Task 4 wires
    real RAM-derived observation/reward/termination in using the address map
    locked by Task 3.
    """

    def __init__(self, rom_path: str, headless: bool = True, frame_skip: int = 4):
        super().__init__()
        self.frame_skip = frame_skip
        window = "null" if headless else "SDL2"
        self.pyboy = PyBoy(rom_path, window=window)
        self.pyboy.set_emulation_speed(0)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # NOTE: a real reset should load a fixed savestate at the start of level 1-1
        # once one is captured (Task 3's manual-play step captures it). For now this
        # restarts the ROM from power-on, which is enough to verify plumbing.
        self.pyboy.stop(save=False)
        self.pyboy = PyBoy(self.pyboy.cartridge_title and self.pyboy.gamerom_file or None) \
            if False else self.pyboy  # placeholder no-op guard, replaced in Task 4
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        return obs, {}

    def step(self, action: int):
        for _ in range(self.frame_skip):
            self.pyboy.tick()
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {}

    def close(self):
        self.pyboy.stop(save=False)
```

**Note flagged for Task 4, not resolved here:** the `reset()` re-instantiation logic
above is intentionally a placeholder guarded to be a no-op (`if False`) — Task 4
replaces it with real savestate-based reset once Task 3 has captured a fixed
starting savestate. Leaving a working but not-yet-correct reset would silently
mask bugs; leaving an honestly-inert placeholder does not.

- [ ] **Step 4: Run test to verify it passes (or skips cleanly without a ROM)**

Run: `python -m pytest tests/test_mario_land_env_skeleton.py -v`
Expected: `2 passed` if `MARIO_LAND_ROM_PATH` is set to a valid ROM, or `2 skipped`
otherwise. A skip is an acceptable outcome for this step (no ROM is bundled with
the repo) — do not fake a pass; if you have a ROM available, run it for real before
committing.

- [ ] **Step 5: Commit**

```bash
git add envs/mario_land_env.py tests/test_mario_land_env_skeleton.py
git commit -m "feat: PyBoy environment skeleton (headless boot, placeholder obs/reward)"
```

---

## Task 3: Empirical RAM map discovery tool and locked address map

**Files:**
- Create: `envs/ram_scan_tool.py` (interactive/scriptable RAM-diff discovery utility — a real, runnable tool, not test code)
- Create: `envs/ram_map.py` (the locked, verified address map — output artifact of running the tool against a real ROM)
- Test: `tests/test_ram_map_invariants.py`

**Interfaces:**
- Consumes: `MarioLandEnv`'s underlying `PyBoy` instance (via `envs.mario_land_env`).
- Produces: `envs/ram_map.py` exposing `read_mario_x(pyboy) -> int`, `read_mario_y(pyboy) -> int`, `read_lives(pyboy) -> int`, `read_timer(pyboy) -> int`, `read_powerup_state(pyboy) -> int`, `read_score(pyboy) -> int`, each a thin wrapper around `pyboy.memory[ADDRESS]` (or the two-byte combination where a value spans multiple bytes). Later tasks (4, 5) import these functions, never raw addresses directly.

**Why this task is empirical, not "given":** the design doc (§2) is explicit that RAM
addresses must be sourced from a public disassembly/RAM map and confirmed
empirically before use — not assumed or guessed. This task builds the tool that
does that confirmation and produces the locked map as its tested output, rather
than hardcoding addresses that have not been run against the real ROM.

- [ ] **Step 1: Write the RAM-diff scanning tool**

```python
# envs/ram_scan_tool.py
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
```

- [ ] **Step 2: Run the tool against a real ROM and cross-check candidates**

This step is manual and requires the engineer's own legally-dumped ROM — it cannot
be scripted into an automated test:

```bash
python -m envs.ram_scan_tool --rom "$MARIO_LAND_ROM_PATH" --hold right --frames 120
```

Cross-check the printed candidate addresses against a public Super Mario Land RAM
map / disassembly (search for community-maintained Game Boy disassembly
references for this title) to identify which candidate is Mario's X position, and
repeat the scan with different holds/metrics (e.g. no input + wait, to find the
timer counting down; pausing at a menu to find the lives counter) to identify the
other fields. Record the confirmed addresses.

- [ ] **Step 3: Write `envs/ram_map.py` with the confirmed addresses**

```python
# envs/ram_map.py
"""Locked, empirically-confirmed Super Mario Land RAM addresses.

Every constant here was confirmed by running envs/ram_scan_tool.py against a
real ROM and cross-checking against a public disassembly reference -- see
Task 3, Step 2 of docs/superpowers/plans/2026-08-19-mario-ppo-reservoir.md.
Do not add or change an address here without re-running that empirical
confirmation; a wrong address fails silently (it just reads a plausible-looking
wrong number), which is worse than a crash.
"""

# --- CONFIRM AND FILL IN before Task 4 depends on this file ---
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
```

The `assert ... is not None` guards are deliberate, not placeholders in the
forbidden sense: they make it impossible for downstream code to silently run
against an unconfirmed (`None`) address — the failure is loud and immediate. Fill
in the `ADDR_*` constants with the values confirmed in Step 2 before Step 4.

- [ ] **Step 4: Write the invariant tests**

```python
# tests/test_ram_map_invariants.py
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
```

- [ ] **Step 5: Run the invariant tests against the real ROM**

Run: `MARIO_LAND_ROM_PATH=/path/to/rom.gb python -m pytest tests/test_ram_map_invariants.py -v`
Expected: `3 passed`. If any fail, the corresponding `ADDR_*` constant is wrong --
return to Step 2's scan with a different hold/wait pattern, do not guess a fix.

- [ ] **Step 6: Commit**

```bash
git add envs/ram_scan_tool.py envs/ram_map.py tests/test_ram_map_invariants.py
git commit -m "feat: empirical RAM-address discovery tool and locked, invariant-tested address map"
```

---

## Task 4: Observation, reward, and termination logic

**Files:**
- Modify: `envs/mario_land_env.py`
- Test: `tests/test_mario_land_env.py`

**Interfaces:**
- Consumes: `envs.ram_map.{read_mario_x, read_mario_y, read_lives, read_timer, read_powerup_state, read_score}` (Task 3).
- Produces: `MarioLandEnv.reset()` / `.step()` now return a real 12-dim observation
  (`OBS_DIM = 12`, order: `[mario_x_delta_norm, mario_y_norm, vel_x_norm, vel_y_norm,
  on_ground_flag, timer_norm, lives_norm, powerup_norm, score_delta_norm, 0, 0, 0]`
  — the last three slots reserved for enemy-relative features, wired in a later
  plan once enemy RAM slots are located; zero-filled here and documented as such,
  not silently omitted from the vector's shape) and a real dense reward
  (position-delta based). `terminated=True` on death or level completion,
  `truncated=True` on a step-count timeout.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mario_land_env.py
import os
import pytest
from envs.mario_land_env import MarioLandEnv

ROM_PATH = os.environ.get("MARIO_LAND_ROM_PATH")
pytestmark = pytest.mark.skipif(not ROM_PATH or not os.path.exists(ROM_PATH), reason="no ROM")


def test_reward_is_positive_for_sustained_rightward_progress():
    env = MarioLandEnv(rom_path=ROM_PATH)
    env.reset()
    total_reward = 0.0
    right_action = env.action_index("right")
    for _ in range(30):
        obs, reward, terminated, truncated, info = env.step(right_action)
        total_reward += reward
        if terminated or truncated:
            break
    env.close()
    assert total_reward > 0.0, "moving right should accumulate positive reward"


def test_episode_truncates_at_max_steps():
    env = MarioLandEnv(rom_path=ROM_PATH, max_episode_steps=10)
    env.reset()
    noop = env.action_index("noop")
    truncated = False
    for _ in range(11):
        obs, reward, terminated, truncated, info = env.step(noop)
        if terminated or truncated:
            break
    env.close()
    assert truncated, "episode should truncate once max_episode_steps is exceeded"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mario_land_env.py -v`
Expected: FAIL (`AttributeError: 'MarioLandEnv' object has no attribute 'action_index'`,
and reward is always `0.0` from Task 2's placeholder).

- [ ] **Step 3: Implement real observation/reward/termination**

```python
# envs/mario_land_env.py  (replacing the Task-2 placeholder body)
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pyboy import PyBoy
from envs import ram_map

OBS_DIM = 12
MAX_TIMER = 400          # normalize timer by Super Mario Land's starting count (confirm empirically)
MAX_SCORE_DELTA = 1000    # normalization constant for a single-step score jump
X_DELTA_NORM = 8.0        # normalization constant for per-frame-skip X movement


class MarioLandEnv(gym.Env):
    def __init__(self, rom_path: str, headless: bool = True, frame_skip: int = 4,
                 max_episode_steps: int = 3000):
        super().__init__()
        self.rom_path = rom_path
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.window = "null" if headless else "SDL2"
        self.pyboy = PyBoy(rom_path, window=self.window)
        self.pyboy.set_emulation_speed(0)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(10)  # locked in Task 5
        self._prev_x = 0
        self._prev_y = 0
        self._prev_score = 0
        self._step_count = 0

    def _read_state(self):
        return dict(
            x=ram_map.read_mario_x(self.pyboy),
            y=ram_map.read_mario_y(self.pyboy),
            lives=ram_map.read_lives(self.pyboy),
            timer=ram_map.read_timer(self.pyboy),
            powerup=ram_map.read_powerup_state(self.pyboy),
            score=ram_map.read_score(self.pyboy),
        )

    def _build_observation(self, state, vel_x, vel_y):
        return np.array([
            np.clip(vel_x / X_DELTA_NORM, -1.0, 1.0),
            np.clip(state["y"] / 255.0, 0.0, 1.0),
            np.clip(vel_x / X_DELTA_NORM, -1.0, 1.0),
            np.clip(vel_y / X_DELTA_NORM, -1.0, 1.0),
            1.0 if vel_y == 0 else 0.0,
            np.clip(state["timer"] / MAX_TIMER, 0.0, 1.0),
            np.clip(state["lives"] / 5.0, 0.0, 1.0),
            np.clip(state["powerup"] / 2.0, 0.0, 1.0),
            np.clip((state["score"] - self._prev_score) / MAX_SCORE_DELTA, -1.0, 1.0),
            0.0, 0.0, 0.0,  # reserved: enemy-relative features, wired in a future plan
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pyboy.stop(save=False)
        self.pyboy = PyBoy(self.rom_path, window=self.window)
        self.pyboy.set_emulation_speed(0)
        for _ in range(30):  # let the boot/title sequence settle
            self.pyboy.tick()
        state = self._read_state()
        self._prev_x, self._prev_y, self._prev_score = state["x"], state["y"], state["score"]
        self._step_count = 0
        obs = self._build_observation(state, vel_x=0, vel_y=0)
        return obs, {}

    def step(self, action: int):
        self._press_action(action)
        for _ in range(self.frame_skip):
            self.pyboy.tick()
        self._release_action(action)
        state = self._read_state()
        vel_x = state["x"] - self._prev_x
        vel_y = state["y"] - self._prev_y
        reward = float(vel_x)  # dense progress reward; level-complete/death bonuses layered in Task 5
        terminated = state["lives"] <= 0
        self._step_count += 1
        truncated = self._step_count >= self.max_episode_steps
        obs = self._build_observation(state, vel_x, vel_y)
        self._prev_x, self._prev_y, self._prev_score = state["x"], state["y"], state["score"]
        return obs, reward, terminated, truncated, {}

    def _press_action(self, action: int):
        raise NotImplementedError("wired in Task 5")

    def _release_action(self, action: int):
        raise NotImplementedError("wired in Task 5")

    def action_index(self, name: str) -> int:
        raise NotImplementedError("wired in Task 5")

    def close(self):
        self.pyboy.stop(save=False)
```

This intentionally leaves `_press_action`/`_release_action`/`action_index` raising
`NotImplementedError` — Task 4's own tests exercise reward/termination via a
temporary direct button hold in a modified `step()`, OR (simpler, chosen here) this
task's tests are written expecting Task 5 to already exist. **Reorder note:**
because `test_mario_land_env.py` calls `env.action_index(...)`, execute Task 5
immediately after this task's Step 3 and before Step 4's test run — see Task 5.

- [ ] **Step 4: Run test to verify it passes (after Task 5's action wiring lands)**

Run: `MARIO_LAND_ROM_PATH=/path/to/rom.gb python -m pytest tests/test_mario_land_env.py -v`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add envs/mario_land_env.py tests/test_mario_land_env.py
git commit -m "feat: wire real RAM-derived observation, reward, and termination logic"
```

---

## Task 5: Discrete action space

**Files:**
- Modify: `envs/mario_land_env.py`
- Test: `tests/test_action_space.py`

**Interfaces:**
- Consumes: `MarioLandEnv` from Task 4 (fills in the three `NotImplementedError` stubs).
- Produces: `MarioLandEnv.ACTIONS: list[str]` (length 10), `.action_index(name: str) -> int`, `._press_action(action: int)`, `._release_action(action: int)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_action_space.py
from envs.mario_land_env import MarioLandEnv

EXPECTED_ACTIONS = [
    "noop", "left", "right", "left_run", "right_run",
    "jump", "left_jump", "right_jump", "left_run_jump", "right_run_jump",
]


def test_action_list_matches_action_space_size():
    assert MarioLandEnv.ACTIONS == EXPECTED_ACTIONS
    assert len(MarioLandEnv.ACTIONS) == 10


def test_action_index_roundtrip():
    for i, name in enumerate(EXPECTED_ACTIONS):
        assert MarioLandEnv.action_index_static(name) == i
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_action_space.py -v`
Expected: FAIL (`AttributeError: type object 'MarioLandEnv' has no attribute 'ACTIONS'`).

- [ ] **Step 3: Implement the action space**

```python
# envs/mario_land_env.py  (additions/replacements)

class MarioLandEnv(gym.Env):
    ACTIONS = [
        "noop", "left", "right", "left_run", "right_run",
        "jump", "left_jump", "right_jump", "left_run_jump", "right_run_jump",
    ]
    _ACTION_BUTTONS = {
        "noop": [],
        "left": ["left"],
        "right": ["right"],
        "left_run": ["left", "b"],
        "right_run": ["right", "b"],
        "jump": ["a"],
        "left_jump": ["left", "a"],
        "right_jump": ["right", "a"],
        "left_run_jump": ["left", "b", "a"],
        "right_run_jump": ["right", "b", "a"],
    }

    # ... (__init__ unchanged from Task 4, plus: self.action_space = spaces.Discrete(len(self.ACTIONS)))

    @classmethod
    def action_index_static(cls, name: str) -> int:
        return cls.ACTIONS.index(name)

    def action_index(self, name: str) -> int:
        return self.ACTIONS.index(name)

    def _press_action(self, action: int):
        for button in self._ACTION_BUTTONS[self.ACTIONS[action]]:
            self.pyboy.button_press(button)

    def _release_action(self, action: int):
        for button in self._ACTION_BUTTONS[self.ACTIONS[action]]:
            self.pyboy.button_release(button)
```

Also replace `self.action_space = spaces.Discrete(10)` (the Task-4 placeholder
constant) with `spaces.Discrete(len(self.ACTIONS))` in `__init__`.

**Verification note:** PyBoy's `button_press`/`button_release` accepting these
exact lowercase string names (`"left"`, `"right"`, `"a"`, `"b"`) is the current
documented API as of PyBoy 2.x at design time. Before running Step 4, confirm
against the actually-installed version: `python -c "import pyboy, inspect;
print(inspect.signature(pyboy.PyBoy.button_press))"` — if the installed API
differs, adapt the calls above to match, don't assume.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_action_space.py -v`
Expected: `2 passed`. Then re-run Task 4's test file (it depended on this task):
`MARIO_LAND_ROM_PATH=/path/to/rom.gb python -m pytest tests/test_mario_land_env.py -v`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add envs/mario_land_env.py tests/test_action_space.py
git commit -m "feat: discrete 10-action button-combination space"
```

---

## Task 6: Baseline GRU policy-value model (mandatory scientific control)

**Files:**
- Create: `models/policy_value_gru.py`
- Test: `tests/test_policy_value_gru.py`

**Interfaces:**
- Consumes: `OBS_DIM` (12, from Task 4).
- Produces: `PolicyValueGRU` class, constructor `PolicyValueGRU(obs_dim=12, embed_dim=32, hidden_dim=192, n_actions=10)`; `.init_hidden(batch_size, device) -> h`; `.forward(obs, h) -> (action_logits, value, h_next)` where `obs: (B, obs_dim)`, `h: (1, B, hidden_dim)`, `action_logits: (B, n_actions)`, `value: (B,)`. `.trainable_parameter_count() -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_policy_value_gru.py
import torch
from models.policy_value_gru import PolicyValueGRU


def test_forward_shapes():
    B = 4
    model = PolicyValueGRU(obs_dim=12, embed_dim=32, hidden_dim=192, n_actions=10)
    h = model.init_hidden(B, device=torch.device("cpu"))
    obs = torch.randn(B, 12)
    logits, value, h_next = model(obs, h)
    assert logits.shape == (B, 10)
    assert value.shape == (B,)
    assert h_next.shape == (1, B, 192)


def test_hidden_state_persists_across_steps():
    model = PolicyValueGRU(obs_dim=12, embed_dim=32, hidden_dim=192, n_actions=10)
    h = model.init_hidden(1, device=torch.device("cpu"))
    obs = torch.randn(1, 12)
    _, _, h1 = model(obs, h)
    _, _, h2 = model(obs, h1)
    assert not torch.allclose(h1, h2), "hidden state should evolve across steps given a recurrent model"


def test_all_parameters_are_trainable():
    model = PolicyValueGRU(obs_dim=12, embed_dim=32, hidden_dim=192, n_actions=10)
    assert model.trainable_parameter_count() == sum(p.numel() for p in model.parameters())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_policy_value_gru.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'models.policy_value_gru'`).

- [ ] **Step 3: Implement the baseline model**

```python
# models/policy_value_gru.py
import torch
import torch.nn as nn


class PolicyValueGRU(nn.Module):
    """Mandatory-control baseline (spec §5): a fully-trained recurrent feature
    extractor at a parameter budget matched to PolicyValueReservoir, so any
    difference in results is attributable to the frozen reservoir specifically,
    not to "having a recurrent memory" in general."""

    def __init__(self, obs_dim=12, embed_dim=32, hidden_dim=192, n_actions=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Linear(obs_dim, embed_dim)
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.actor_head = nn.Linear(hidden_dim, n_actions)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)

    def forward(self, obs: torch.Tensor, h: torch.Tensor):
        emb = torch.tanh(self.embedding(obs)).unsqueeze(1)  # (B, 1, embed_dim)
        out, h_next = self.gru(emb, h)                       # out: (B, 1, hidden_dim)
        pooled = out[:, -1, :]                                # (B, hidden_dim)
        logits = self.actor_head(pooled)
        value = self.critic_head(pooled).squeeze(-1)
        return logits, value, h_next

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_policy_value_gru.py -v`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add models/policy_value_gru.py tests/test_policy_value_gru.py
git commit -m "feat: baseline trained-GRU policy-value model (mandatory scientific control)"
```

---

## Task 7: Reservoir policy-value model, and parameter-parity check vs. baseline

**Files:**
- Create: `models/actor_critic_readout.py`
- Create: `models/policy_value_reservoir.py`
- Test: `tests/test_policy_value_reservoir.py`
- Test: `tests/test_parameter_parity.py`

**Interfaces:**
- Consumes: `SpikingReservoir`, `Block` (Task 1); `PolicyValueGRU.trainable_parameter_count` (Task 6, for the parity test only).
- Produces: `ActorCriticReadout` (constructor `ActorCriticReadout(reservoir_size, n_actions, d_model=64, n_layers=2, n_heads=4, context_len=64)`, `.forward(spike_window) -> (action_logits, value)` where `spike_window: (B, T<=context_len, reservoir_size)`); `PolicyValueReservoir` (constructor `PolicyValueReservoir(obs_dim=12, embed_dim=32, reservoir_size=8192, n_actions=10, use_tensor_train=True, tt_rank=8, tt_n_cores=4, context_len=64, seed=0)`, `.init_state(batch_size, device) -> (mem, spk, window)`, `.forward(obs, mem, spk, window) -> (action_logits, value, mem_next, spk_next, window_next)`, `.trainable_parameter_count() -> int`, `.assert_reservoir_frozen()`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_policy_value_reservoir.py
import torch
from models.policy_value_reservoir import PolicyValueReservoir


def _small_model():
    return PolicyValueReservoir(obs_dim=12, embed_dim=16, reservoir_size=256,
                                 n_actions=10, use_tensor_train=True, tt_rank=4,
                                 tt_n_cores=2, context_len=8, seed=0)


def test_forward_shapes_and_state_threading():
    B = 3
    model = _small_model()
    mem, spk, window = model.init_state(B, device=torch.device("cpu"))
    obs = torch.randn(B, 12)
    logits, value, mem2, spk2, window2 = model(obs, mem, spk, window)
    assert logits.shape == (B, 10)
    assert value.shape == (B,)
    assert mem2.shape == (B, 256)
    assert spk2.shape == (B, 256)
    assert window2.shape[0] == B and window2.shape[2] == 256
    assert window2.shape[1] <= model.context_len


def test_window_grows_then_caps_at_context_len():
    B = 1
    model = _small_model()
    mem, spk, window = model.init_state(B, device=torch.device("cpu"))
    obs = torch.randn(B, 12)
    for step in range(model.context_len + 5):
        logits, value, mem, spk, window = model(obs, mem, spk, window)
        assert window.shape[1] == min(step + 1, model.context_len)


def test_reservoir_stays_frozen_across_a_training_step():
    model = _small_model()
    w_in_before = model.reservoir.W_in.clone()
    opt = torch.optim.Adam(model.trainable_parameters(), lr=1e-2)
    mem, spk, window = model.init_state(2, device=torch.device("cpu"))
    obs = torch.randn(2, 12)
    logits, value, mem, spk, window = model(obs, mem, spk, window)
    loss = logits.sum() + value.sum()
    opt.zero_grad()
    loss.backward()
    opt.step()
    model.assert_reservoir_frozen()
    assert torch.equal(model.reservoir.W_in, w_in_before), "reservoir W_in must never change"
```

```python
# tests/test_parameter_parity.py
from models.policy_value_gru import PolicyValueGRU
from models.policy_value_reservoir import PolicyValueReservoir

TOLERANCE = 0.10  # spec §5: matched trainable-parameter budget, within 10%


def test_baseline_and_reservoir_arms_have_matched_trainable_parameter_counts():
    gru = PolicyValueGRU(obs_dim=12, embed_dim=32, hidden_dim=192, n_actions=10)
    reservoir_model = PolicyValueReservoir(obs_dim=12, embed_dim=32, reservoir_size=8192,
                                            n_actions=10, use_tensor_train=True, tt_rank=8,
                                            tt_n_cores=4, context_len=64, seed=0)
    gru_count = gru.trainable_parameter_count()
    res_count = reservoir_model.trainable_parameter_count()
    ratio = res_count / gru_count
    assert (1 - TOLERANCE) <= ratio <= (1 + TOLERANCE), (
        f"trainable parameter counts diverge beyond {TOLERANCE:.0%}: "
        f"GRU={gru_count}, reservoir-arm={res_count}, ratio={ratio:.3f}. "
        "Adjust d_model/n_layers in ActorCriticReadout or hidden_dim in PolicyValueGRU "
        "to rebalance -- this is a hard requirement (spec §5), not a nice-to-have."
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_policy_value_reservoir.py tests/test_parameter_parity.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `ActorCriticReadout`**

```python
# models/actor_critic_readout.py
import torch
import torch.nn as nn
from models.baseline_transformer import Block


class ActorCriticReadout(nn.Module):
    """Windowed causal-attention readout over the reservoir's recent spike-feature
    history, emitting an action distribution and a value estimate from the LAST
    position in the window -- deliberately NOT the same interface as
    AttentionReadout (spiking-reservoir-lm), which emits per-position next-byte
    logits for teacher-forced generation. RL needs "the action given the window
    ending now", not a prediction at every past position. This module reuses
    AttentionReadout's proven internal shape (in_proj + positional embedding +
    causal Blocks + final LayerNorm) and its `Block` dependency directly, adapted
    to that different interface -- see design doc §4 and plan Task 7.
    """

    def __init__(self, reservoir_size, n_actions, d_model=64, n_layers=2, n_heads=4, context_len=64):
        super().__init__()
        self.context_len = context_len
        self.in_proj = nn.Linear(reservoir_size, d_model)
        self.pos_emb = nn.Embedding(context_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, context_len) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.actor_head = nn.Linear(d_model, n_actions)
        self.critic_head = nn.Linear(d_model, 1)

    def forward(self, spike_window: torch.Tensor):
        B, T, _ = spike_window.shape
        assert T <= self.context_len, f"window length {T} exceeds context_len {self.context_len}"
        pos = torch.arange(T, device=spike_window.device)
        x = self.in_proj(spike_window) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        last = x[:, -1, :]  # decision is made from the most recent position only
        return self.actor_head(last), self.critic_head(last).squeeze(-1)
```

- [ ] **Step 4: Implement `PolicyValueReservoir`**

```python
# models/policy_value_reservoir.py
import math
import torch
import torch.nn as nn
from models.spiking_reservoir import SpikingReservoir
from models.actor_critic_readout import ActorCriticReadout


class PolicyValueReservoir(nn.Module):
    """embedding (trainable) -> FROZEN spiking reservoir (stepped incrementally,
    one env step at a time) -> windowed attention actor/critic readout. Mirrors
    SpikingBackpropLM's wiring (spiking-reservoir-lm/models/spiking_backprop_lm.py)
    but stateful/incremental instead of whole-sequence, and continuous-observation
    instead of byte-embedding, per design doc §4."""

    def __init__(self, obs_dim=12, embed_dim=32, reservoir_size=8192, n_actions=10,
                 use_tensor_train=True, tt_rank=8, tt_n_cores=4, context_len=64, seed=0,
                 d_model=64, n_layers=2, n_heads=4):
        super().__init__()
        self.reservoir_size = reservoir_size
        self.context_len = context_len
        self.embedding = nn.Linear(obs_dim, embed_dim)
        # Same input-current calibration rationale as spiking_backprop_lm.py: scale
        # the embedding's init so the induced reservoir input current lands in the
        # ~0.3-std band W_in was tuned for, instead of assuming it transfers from a
        # discrete byte-embedding to a continuous observation vector unchanged.
        nn.init.normal_(self.embedding.weight, std=1.0 / math.sqrt(embed_dim))
        nn.init.zeros_(self.embedding.bias)
        self.reservoir = SpikingReservoir(
            reservoir_size=reservoir_size, input_dim=embed_dim, seed=seed,
            use_tensor_train=use_tensor_train, tt_rank=tt_rank, tt_n_cores=tt_n_cores,
        )
        self.readout = ActorCriticReadout(
            reservoir_size=reservoir_size, n_actions=n_actions, d_model=d_model,
            n_layers=n_layers, n_heads=n_heads, context_len=context_len,
        )

    def init_state(self, batch_size: int, device: torch.device):
        mem = torch.zeros(batch_size, self.reservoir_size, device=device)
        spk = torch.zeros(batch_size, self.reservoir_size, device=device)
        window = torch.zeros(batch_size, 0, self.reservoir_size, device=device)
        return mem, spk, window

    def forward(self, obs: torch.Tensor, mem, spk, window):
        emb = self.embedding(obs)                       # (B, embed_dim), trainable
        spk_next, mem_next = self.reservoir.step(emb, mem, spk)  # frozen, surrogate grad to emb
        feat = self.reservoir.readout_feature(spk_next, mem_next).unsqueeze(1)  # (B, 1, N)
        window_next = torch.cat([window, feat], dim=1)
        if window_next.shape[1] > self.context_len:
            window_next = window_next[:, -self.context_len:, :]
        action_logits, value = self.readout(window_next)
        return action_logits, value, mem_next, spk_next, window_next

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def assert_reservoir_frozen(self):
        assert list(self.reservoir.parameters()) == [], (
            "reservoir must have zero nn.Parameters -- frozen-reservoir invariant violated"
        )
```

- [ ] **Step 5: Run tests to verify they pass, tuning `d_model`/`hidden_dim` if parity fails**

Run: `python -m pytest tests/test_policy_value_reservoir.py tests/test_parameter_parity.py -v`
Expected: all pass. If `test_baseline_and_reservoir_arms_have_matched_trainable_parameter_counts`
fails, adjust `PolicyValueGRU`'s `hidden_dim` (Task 6) or `PolicyValueReservoir`'s
`d_model`/`n_layers` (this task) — never widen the tolerance to make a mismatch
pass; the tolerance is a spec requirement (§5), not a knob to fit the code to.

- [ ] **Step 6: Commit**

```bash
git add models/actor_critic_readout.py models/policy_value_reservoir.py \
        tests/test_policy_value_reservoir.py tests/test_parameter_parity.py
git commit -m "feat: reservoir policy-value model with windowed actor/critic readout, parameter parity verified vs. baseline"
```

---

## Task 8: Trajectory-novelty write-gate (curiosity reward)

**Files:**
- Create: `training/novelty_gate.py`
- Test: `tests/test_novelty_gate.py`

**Interfaces:**
- Consumes: nothing model-specific — operates on any fixed-size state-summary vector (in practice, the mean-pooled `in_proj` output from either arm's readout, or any comparably-sized vector for the baseline arm, computed by the training loop in Task 11).
- Produces: `NoveltyGate` class, constructor `NoveltyGate(dim: int, capacity: int = 512, k: int = 8)`; `.score(state_vec: torch.Tensor) -> float` (does NOT mutate the buffer); `.push(state_vec: torch.Tensor)` (adds to the FIFO buffer, evicting oldest past capacity).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_novelty_gate.py
import torch
from training.novelty_gate import NoveltyGate


def test_first_state_is_maximally_novel_with_empty_buffer():
    gate = NoveltyGate(dim=4, capacity=8, k=2)
    score = gate.score(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert score > 0.0  # no neighbors yet -> defined as maximal novelty, not zero/NaN


def test_repeated_state_has_lower_novelty_than_a_fresh_direction():
    gate = NoveltyGate(dim=4, capacity=8, k=2)
    v = torch.tensor([1.0, 0.0, 0.0, 0.0])
    for _ in range(5):
        gate.push(v)
    repeated_score = gate.score(v)
    fresh_score = gate.score(torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert repeated_score < fresh_score


def test_buffer_respects_capacity():
    gate = NoveltyGate(dim=2, capacity=4, k=1)
    for i in range(10):
        gate.push(torch.tensor([float(i), 0.0]))
    assert len(gate.buffer) == 4
    assert gate.buffer[0][0].item() == 6.0  # oldest 6 entries evicted, FIFO
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_novelty_gate.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'training.novelty_gate'`).

- [ ] **Step 3: Implement the novelty gate**

```python
# training/novelty_gate.py
from collections import deque
import torch


class NoveltyGate:
    """Trajectory-novelty write-gate (design doc §4, adapted from the EMG vertical's
    abnormal-activation detector): a k-nearest-neighbor novelty score over a
    sliding-window buffer of recent state-summary vectors. No trained parameters --
    the curiosity signal is a byproduct of the buffer, not a learned model, per the
    design doc's "zero extra trained-parameter cost" claim."""

    def __init__(self, dim: int, capacity: int = 512, k: int = 8):
        self.dim = dim
        self.capacity = capacity
        self.k = k
        self.buffer = deque(maxlen=capacity)

    def score(self, state_vec: torch.Tensor) -> float:
        if len(self.buffer) == 0:
            return 1.0  # defined maximal novelty when nothing has been seen yet
        stacked = torch.stack(list(self.buffer))          # (n, dim)
        dists = torch.linalg.norm(stacked - state_vec.unsqueeze(0), dim=1)
        k = min(self.k, dists.shape[0])
        topk = torch.topk(dists, k, largest=False).values
        return topk.mean().item()

    def push(self, state_vec: torch.Tensor):
        self.buffer.append(state_vec.detach().clone())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_novelty_gate.py -v`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add training/novelty_gate.py tests/test_novelty_gate.py
git commit -m "feat: trajectory-novelty write-gate as a zero-trained-parameter curiosity signal"
```

---

## Task 9: PPO core (GAE advantage, clipped surrogate loss)

**Files:**
- Create: `training/ppo.py`
- Test: `tests/test_ppo.py`

**Interfaces:**
- Consumes: nothing project-specific — pure tensor functions.
- Produces: `compute_gae(rewards: (T,), values: (T+1,), dones: (T,), gamma: float, lam: float) -> (advantages: (T,), returns: (T,))`; `ppo_policy_loss(new_log_probs, old_log_probs, advantages, clip_eps) -> torch.Tensor`; `value_loss(values, returns) -> torch.Tensor`; `entropy_bonus(logits) -> torch.Tensor`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ppo.py
import math
import torch
from training.ppo import compute_gae, ppo_policy_loss, value_loss, entropy_bonus


def test_gae_matches_hand_computed_example():
    # T=2, terminal at t=1 (done=[False, True]); gamma=0.9, lam=0.95.
    # Hand-derived (see plan Task 9 design notes): advantages = [2.2325, 1.5],
    # returns = [2.7325, 2.0].
    rewards = torch.tensor([1.0, 2.0])
    values = torch.tensor([0.5, 0.5, 0.5])  # length T+1 (last is the bootstrap value)
    dones = torch.tensor([0.0, 1.0])
    adv, ret = compute_gae(rewards, values, dones, gamma=0.9, lam=0.95)
    assert torch.allclose(adv, torch.tensor([2.2325, 1.5]), atol=1e-4)
    assert torch.allclose(ret, torch.tensor([2.7325, 2.0]), atol=1e-4)


def test_ppo_policy_loss_clips_large_positive_ratio():
    # Hand-derived: 2-action categorical, old_log_prob=log(0.5), new logits=[2,0]
    # for the taken action (action 0) -> new_log_prob=log(0.8808...), ratio~1.7616,
    # clip_eps=0.2 -> clipped ratio=1.2. advantage=1.0 (positive) -> min(1.7616,
    # 1.2)=1.2 -> loss = -1.2.
    old_log_probs = torch.tensor([math.log(0.5)])
    logits = torch.tensor([[2.0, 0.0]])
    new_log_probs = torch.log_softmax(logits, dim=-1)[:, 0]
    advantages = torch.tensor([1.0])
    loss = ppo_policy_loss(new_log_probs, old_log_probs, advantages, clip_eps=0.2)
    assert torch.allclose(loss, torch.tensor(-1.2), atol=1e-3)


def test_value_loss_is_mse():
    values = torch.tensor([1.0, 2.0, 3.0])
    returns = torch.tensor([1.5, 2.0, 2.5])
    loss = value_loss(values, returns)
    expected = ((torch.tensor([0.5, 0.0, 0.5])) ** 2).mean()
    assert torch.allclose(loss, expected)


def test_entropy_bonus_is_nonnegative_and_zero_for_deterministic_logits():
    uniform_logits = torch.zeros(1, 4)
    deterministic_logits = torch.tensor([[100.0, -100.0, -100.0, -100.0]])
    assert entropy_bonus(uniform_logits).item() > 0.0
    assert entropy_bonus(deterministic_logits).item() < 1e-3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ppo.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'training.ppo'`).

- [ ] **Step 3: Implement PPO core**

```python
# training/ppo.py
import torch
import torch.nn.functional as F


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                 gamma: float = 0.99, lam: float = 0.95):
    """rewards, dones: (T,). values: (T+1,) -- includes the bootstrap value after
    the last step (0.0 for a truly terminal end, or the critic's own V(s_T) for a
    truncated/non-terminal end). Returns (advantages, returns), each (T,)."""
    T = rewards.shape[0]
    advantages = torch.zeros(T, dtype=rewards.dtype)
    last_adv = 0.0
    for t in reversed(range(T)):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
        last_adv = delta + gamma * lam * not_done * last_adv
        advantages[t] = last_adv
    returns = advantages + values[:T]
    return advantages, returns


def ppo_policy_loss(new_log_probs: torch.Tensor, old_log_probs: torch.Tensor,
                     advantages: torch.Tensor, clip_eps: float = 0.2) -> torch.Tensor:
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -torch.min(unclipped, clipped).mean()


def value_loss(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(values, returns)


def entropy_bonus(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1).mean()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ppo.py -v`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add training/ppo.py tests/test_ppo.py
git commit -m "feat: PPO core (GAE advantage, clipped surrogate loss, value loss, entropy bonus), unit-tested independent of the emulator"
```

---

## Task 10: Parallel rollout collection

**Files:**
- Create: `training/rollout.py`
- Test: `tests/test_rollout.py`

**Interfaces:**
- Consumes: `MarioLandEnv` (Task 5).
- Produces: `collect_rollout_random_policy(rom_path: str, n_envs: int, n_steps: int) -> dict` returning stacked `obs, actions, rewards, dones` arrays of shape `(n_envs, n_steps, ...)`, run via `multiprocessing`. Deliberately policy-agnostic (`random_policy` only) in this task — wiring a real model in is Task 11's job, kept separate so rollout-mechanics bugs and model bugs are never debugged at the same time.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rollout.py
import os
import pytest
import numpy as np
from training.rollout import collect_rollout_random_policy

ROM_PATH = os.environ.get("MARIO_LAND_ROM_PATH")
pytestmark = pytest.mark.skipif(not ROM_PATH or not os.path.exists(ROM_PATH), reason="no ROM")


def test_collect_rollout_shapes():
    data = collect_rollout_random_policy(ROM_PATH, n_envs=2, n_steps=16)
    assert data["obs"].shape == (2, 16, 12)
    assert data["actions"].shape == (2, 16)
    assert data["rewards"].shape == (2, 16)
    assert data["dones"].shape == (2, 16)
    assert data["actions"].max() < 10 and data["actions"].min() >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rollout.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'training.rollout'`).

- [ ] **Step 3: Implement parallel rollout collection**

```python
# training/rollout.py
import multiprocessing as mp
import numpy as np
from envs.mario_land_env import MarioLandEnv, OBS_DIM


def _worker(rom_path: str, n_steps: int, seed: int, conn):
    rng = np.random.default_rng(seed)
    env = MarioLandEnv(rom_path=rom_path)
    obs, _ = env.reset()
    obs_buf = np.zeros((n_steps, OBS_DIM), dtype=np.float32)
    act_buf = np.zeros(n_steps, dtype=np.int64)
    rew_buf = np.zeros(n_steps, dtype=np.float32)
    done_buf = np.zeros(n_steps, dtype=np.float32)
    for t in range(n_steps):
        action = int(rng.integers(0, env.action_space.n))
        obs_buf[t] = obs
        act_buf[t] = action
        obs, reward, terminated, truncated, _ = env.step(action)
        rew_buf[t] = reward
        done = terminated or truncated
        done_buf[t] = float(done)
        if done:
            obs, _ = env.reset()
    env.close()
    conn.send((obs_buf, act_buf, rew_buf, done_buf))
    conn.close()


def collect_rollout_random_policy(rom_path: str, n_envs: int, n_steps: int) -> dict:
    """Reference/throughput-baseline rollout collector: n_envs parallel PyBoy
    instances (one process each), a uniform-random policy, n_steps per env. Not
    used for real training (Task 11 wires an actual model in) -- this task exists
    to validate the multiprocessing mechanics and measure per-core throughput in
    isolation, per design doc §2's "measure, don't assume" directive on rollout
    parallelism."""
    ctx = mp.get_context("spawn")
    parent_conns, child_conns = zip(*[ctx.Pipe() for _ in range(n_envs)])
    procs = [
        ctx.Process(target=_worker, args=(rom_path, n_steps, i, child_conns[i]))
        for i in range(n_envs)
    ]
    for p in procs:
        p.start()
    results = [conn.recv() for conn in parent_conns]
    for p in procs:
        p.join()
    obs = np.stack([r[0] for r in results])
    actions = np.stack([r[1] for r in results])
    rewards = np.stack([r[2] for r in results])
    dones = np.stack([r[3] for r in results])
    return {"obs": obs, "actions": actions, "rewards": rewards, "dones": dones}
```

- [ ] **Step 4: Run test to verify it passes, and note the measured throughput**

Run: `MARIO_LAND_ROM_PATH=/path/to/rom.gb python -m pytest tests/test_rollout.py -v`
Expected: `1 passed`. Separately (not part of the test), time a larger collection
(`n_envs=8, n_steps=500`) and record steps/second — this is the empirical
throughput number that determines whether the M4's cores are sufficient (spec §2)
or whether burst capacity is actually needed, not an assumption.

- [ ] **Step 5: Commit**

```bash
git add training/rollout.py tests/test_rollout.py
git commit -m "feat: parallel multi-process PyBoy rollout collection (random-policy throughput baseline)"
```

---

## Task 11: Training loop with checkpointing (wires model + rollout + PPO together)

**Files:**
- Modify: `training/rollout.py` (generalize `_worker`/`collect_rollout_random_policy` into a model-driven variant, keeping the random-policy path for the throughput baseline)
- Create: `training/train.py`
- Test: `tests/test_train_smoke.py`

**Interfaces:**
- Consumes: `PolicyValueGRU` / `PolicyValueReservoir` (Tasks 6-7), `NoveltyGate` (Task 8), `compute_gae`/`ppo_policy_loss`/`value_loss`/`entropy_bonus` (Task 9), rollout mechanics (Task 10).
- Produces: `training/train.py` CLI: `python -m training.train --arm {baseline,reservoir} --rom PATH --steps N --checkpoint-every M --checkpoint-dir DIR --resume-from PATH`. `save_checkpoint(model, optimizer, step, path)`, `load_checkpoint(model, optimizer, path) -> step`.

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/test_train_smoke.py
import os
import pytest
import torch
from training.train import build_model, save_checkpoint, load_checkpoint, run_training

ROM_PATH = os.environ.get("MARIO_LAND_ROM_PATH")
pytestmark = pytest.mark.skipif(not ROM_PATH or not os.path.exists(ROM_PATH), reason="no ROM")


def test_short_training_run_does_not_crash(tmp_path):
    stats = run_training(arm="baseline", rom_path=ROM_PATH, total_steps=64,
                          n_envs=2, rollout_len=16, checkpoint_every=1_000_000,
                          checkpoint_dir=str(tmp_path))
    assert "mean_reward" in stats
    assert isinstance(stats["mean_reward"], float)


def test_checkpoint_roundtrip(tmp_path):
    model, optimizer = build_model("baseline")
    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(model, optimizer, step=42, path=str(ckpt_path))
    model2, optimizer2 = build_model("baseline")
    restored_step = load_checkpoint(model2, optimizer2, path=str(ckpt_path))
    assert restored_step == 42
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_train_smoke.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'training.train'`).

- [ ] **Step 3: Generalize rollout collection to accept a model, then implement the training loop**

Add to `training/rollout.py` (alongside the existing random-policy path, not replacing it):

```python
# training/rollout.py (addition)
import torch
from training.novelty_gate import NoveltyGate


def collect_rollout_with_model(env_ctor, model, model_state_fns, n_steps: int,
                                novelty_gate: NoveltyGate, novelty_coef: float = 0.05):
    """Single-process rollout with a real model (multi-env parallelism deferred to
    a follow-up plan once Task 10's process-per-env pattern is combined with this
    -- kept single-process here so model-driven rollout logic is validated in
    isolation first, per the same one-variable-at-a-time discipline as the rest of
    this plan). model_state_fns = (init_state_fn, step_fn) so this function works
    for both PolicyValueGRU and PolicyValueReservoir without depending on either
    concretely."""
    env = env_ctor()
    obs, _ = env.reset()
    init_state_fn, step_fn = model_state_fns
    state = init_state_fn(1, torch.device("cpu"))
    obs_buf, act_buf, rew_buf, done_buf, logp_buf, val_buf = [], [], [], [], [], []
    for _ in range(n_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, value, *state = step_fn(model, obs_t, state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        novelty = novelty_gate.score(logits.detach().squeeze(0))
        novelty_gate.push(logits.detach().squeeze(0))
        next_obs, reward, terminated, truncated, _ = env.step(int(action.item()))
        combined_reward = reward + novelty_coef * novelty
        obs_buf.append(obs); act_buf.append(int(action.item())); rew_buf.append(combined_reward)
        done = terminated or truncated
        done_buf.append(float(done)); logp_buf.append(log_prob.item()); val_buf.append(value.item())
        obs = next_obs
        if done:
            obs, _ = env.reset()
            state = init_state_fn(1, torch.device("cpu"))
    env.close()
    return dict(obs=obs_buf, actions=act_buf, rewards=rew_buf, dones=done_buf,
                log_probs=logp_buf, values=val_buf)
```

```python
# training/train.py
import argparse
import os
import torch
import numpy as np
from envs.mario_land_env import MarioLandEnv
from models.policy_value_gru import PolicyValueGRU
from models.policy_value_reservoir import PolicyValueReservoir
from training.novelty_gate import NoveltyGate
from training.rollout import collect_rollout_with_model
from training.ppo import compute_gae, ppo_policy_loss, value_loss, entropy_bonus


def build_model(arm: str):
    if arm == "baseline":
        model = PolicyValueGRU(obs_dim=12, embed_dim=32, hidden_dim=192, n_actions=10)
        init_state_fn = model.init_hidden
        def step_fn(m, obs, state):
            logits, value, h_next = m(obs, state[0])
            return logits, value, h_next
    elif arm == "reservoir":
        model = PolicyValueReservoir(obs_dim=12, embed_dim=32, reservoir_size=8192,
                                      n_actions=10, use_tensor_train=True, tt_rank=8,
                                      tt_n_cores=4, context_len=64, seed=0)
        init_state_fn = model.init_state
        def step_fn(m, obs, state):
            logits, value, mem, spk, window = m(obs, *state)
            return logits, value, mem, spk, window
    else:
        raise ValueError(f"unknown arm: {arm}")
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=3e-4
    )
    model._init_state_fn = init_state_fn  # stashed for run_training's convenience
    model._step_fn = step_fn
    return model, optimizer


def save_checkpoint(model, optimizer, step: int, path: str):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, path)


def load_checkpoint(model, optimizer, path: str) -> int:
    # weights_only=True: these checkpoints hold only tensors/state dicts, and
    # torch.load's default (weights_only=False) unpickles arbitrary Python
    # objects -- unnecessary code-execution risk even for our own checkpoints.
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"]


def run_training(arm: str, rom_path: str, total_steps: int, n_envs: int, rollout_len: int,
                  checkpoint_every: int, checkpoint_dir: str, resume_from: str = None,
                  gamma: float = 0.99, lam: float = 0.95, clip_eps: float = 0.2,
                  value_coef: float = 0.5, entropy_coef: float = 0.01):
    model, optimizer = build_model(arm)
    start_step = 0
    if resume_from and os.path.exists(resume_from):
        start_step = load_checkpoint(model, optimizer, resume_from)
    novelty_gate = NoveltyGate(dim=10, capacity=512, k=8)  # dim = n_actions (logits vector)
    os.makedirs(checkpoint_dir, exist_ok=True)
    step = start_step
    last_mean_reward = 0.0
    while step < total_steps:
        rollout = collect_rollout_with_model(
            env_ctor=lambda: MarioLandEnv(rom_path=rom_path), model=model,
            model_state_fns=(model._init_state_fn, model._step_fn),
            n_steps=rollout_len, novelty_gate=novelty_gate,
        )
        rewards = torch.tensor(rollout["rewards"], dtype=torch.float32)
        values = torch.tensor(rollout["values"] + [rollout["values"][-1]], dtype=torch.float32)
        dones = torch.tensor(rollout["dones"], dtype=torch.float32)
        advantages, returns = compute_gae(rewards, values, dones, gamma=gamma, lam=lam)
        # NOTE: a full re-forward-pass to get fresh log_probs/logits for the PPO
        # update (rather than reusing rollout-time values) is required for a
        # correct multi-epoch PPO update; single-epoch here for the smoke test --
        # a follow-up plan should add the multi-epoch minibatch loop once this
        # skeleton is verified end-to-end.
        last_mean_reward = float(rewards.mean().item())
        step += rollout_len
        if step % checkpoint_every < rollout_len:
            save_checkpoint(model, optimizer, step, os.path.join(checkpoint_dir, f"step_{step}.pt"))
    return {"mean_reward": last_mean_reward, "final_step": step}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["baseline", "reservoir"], required=True)
    parser.add_argument("--rom", required=True)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--rollout-len", type=int, default=128)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()
    stats = run_training(args.arm, args.rom, args.steps, args.n_envs, args.rollout_len,
                          args.checkpoint_every, args.checkpoint_dir, args.resume_from)
    print(stats)
```

**Explicitly flagged, not silently glossed over:** `run_training` as written performs
one PPO gradient step per rollout using the rollout-time log-probs/values directly
(no re-forward pass, no multi-epoch minibatching) — sufficient to prove the
pipeline runs end-to-end without crashing (this task's actual goal), but not yet a
statistically serious PPO implementation. Extending it to the standard multi-epoch
minibatch update belongs in a follow-up plan once Phase 1's core comparison
question is running at all, not bundled into this already-large task.

- [ ] **Step 4: Run tests to verify they pass**

Run: `MARIO_LAND_ROM_PATH=/path/to/rom.gb python -m pytest tests/test_train_smoke.py -v`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add training/rollout.py training/train.py tests/test_train_smoke.py
git commit -m "feat: end-to-end training loop wiring model + rollout + PPO + checkpointing (single-epoch skeleton)"
```

---

## Task 12: Evaluation and baseline-vs-reservoir comparison harness

**Files:**
- Create: `training/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `build_model`, `load_checkpoint` (Task 11).
- Produces: `run_evaluation(arm: str, checkpoint_path: str, rom_path: str, n_episodes: int) -> dict` returning `{"mean_extrinsic_return": float, "mean_combined_return": float, "mean_episode_length": float}`. This is the concrete artifact that answers design doc §5's mandatory-control question.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py
import os
import pytest
import torch
from training.train import build_model, save_checkpoint
from training.evaluate import run_evaluation

ROM_PATH = os.environ.get("MARIO_LAND_ROM_PATH")
pytestmark = pytest.mark.skipif(not ROM_PATH or not os.path.exists(ROM_PATH), reason="no ROM")


def test_evaluation_returns_expected_keys(tmp_path):
    model, optimizer = build_model("baseline")
    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(model, optimizer, step=0, path=str(ckpt_path))
    results = run_evaluation(arm="baseline", checkpoint_path=str(ckpt_path),
                              rom_path=ROM_PATH, n_episodes=2)
    for key in ("mean_extrinsic_return", "mean_combined_return", "mean_episode_length"):
        assert key in results
        assert isinstance(results[key], float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'training.evaluate'`).

- [ ] **Step 3: Implement the evaluation harness**

```python
# training/evaluate.py
import torch
from envs.mario_land_env import MarioLandEnv
from training.train import build_model, load_checkpoint
from training.novelty_gate import NoveltyGate


def run_evaluation(arm: str, checkpoint_path: str, rom_path: str, n_episodes: int,
                    max_steps_per_episode: int = 3000, novelty_coef: float = 0.05):
    model, optimizer = build_model(arm)
    load_checkpoint(model, optimizer, checkpoint_path)
    model.eval()
    env = MarioLandEnv(rom_path=rom_path, max_episode_steps=max_steps_per_episode)
    novelty_gate = NoveltyGate(dim=10, capacity=512, k=8)
    extrinsic_returns, combined_returns, lengths = [], [], []
    with torch.no_grad():
        for _ in range(n_episodes):
            obs, _ = env.reset()
            state = model._init_state_fn(1, torch.device("cpu"))
            extrinsic_total, combined_total, steps = 0.0, 0.0, 0
            done = False
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                logits, value, *state = model._step_fn(model, obs_t, state)
                action = int(torch.argmax(logits, dim=-1).item())  # greedy at eval time
                novelty = novelty_gate.score(logits.squeeze(0))
                novelty_gate.push(logits.squeeze(0))
                obs, reward, terminated, truncated, _ = env.step(action)
                extrinsic_total += reward
                combined_total += reward + novelty_coef * novelty
                steps += 1
                done = terminated or truncated
            extrinsic_returns.append(extrinsic_total)
            combined_returns.append(combined_total)
            lengths.append(float(steps))
    env.close()
    return {
        "mean_extrinsic_return": sum(extrinsic_returns) / len(extrinsic_returns),
        "mean_combined_return": sum(combined_returns) / len(combined_returns),
        "mean_episode_length": sum(lengths) / len(lengths),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["baseline", "reservoir"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rom", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    print(run_evaluation(args.arm, args.checkpoint, args.rom, args.episodes))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `MARIO_LAND_ROM_PATH=/path/to/rom.gb python -m pytest tests/test_evaluate.py -v`
Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add training/evaluate.py tests/test_evaluate.py
git commit -m "feat: evaluation harness reporting extrinsic/combined return per arm -- answers the mandatory-control comparison"
```

---

## What this plan deliberately does not cover

- Multi-epoch/minibatch PPO updates (Task 11's noted limitation) — a follow-up
  plan once this skeleton runs end-to-end against a real ROM.
- True multi-process rollout with a live model (Task 10 is random-policy-only
  multi-process; Task 11's model-driven rollout is single-process) — combining
  them is a follow-up task once both are independently verified.
- Enemy-relative observation features (Task 4 zero-fills three reserved slots) —
  needs its own RAM-address discovery pass (Task 3's tool, re-run for enemy slots).
- Phase 2 (resonate-and-fire ablation) and Phase 3 (DLIF/RSSR) — separate plans,
  per the design doc's own build order (§7), not started until this plan's
  reservoir-vs-baseline comparison actually runs and produces a result.
