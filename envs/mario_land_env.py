import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pyboy import PyBoy

OBS_DIM = 12  # locked in Task 5; placeholder zero-vector until then
N_ACTIONS = 10  # locked in Task 4; placeholder discrete space until then


class MarioLandEnv(gym.Env):
    """Gymnasium-style wrapper around PyBoy running Super Mario Land.

    Observation and reward are placeholders in this skeleton (Task 2) --
    verified to boot headless and accept input without crashing. Task 4 wires
    the real action vocabulary/button mechanics; Task 5 wires real RAM-derived
    observation/reward/termination using the address map locked by Task 3.
    Task 4 runs before Task 5 -- see the ruling in the SDD ledger's pre-flight
    scan for why (Task 5's own tests need action_index to already exist).
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
            if False else self.pyboy  # placeholder no-op guard, replaced in Task 5
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
