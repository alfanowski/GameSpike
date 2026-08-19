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
