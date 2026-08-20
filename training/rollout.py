"""Multi-process, random-policy rollout collection -- the throughput baseline.

Deliberately policy-agnostic: this module only ever samples uniform-random
actions. Wiring a real (reservoir or GRU) policy in is Task 11's job, kept in a
separate module so that a rollout-mechanics bug (e.g. a multiprocessing
deadlock, a shape mismatch) and a model bug (e.g. a bad forward pass) are never
debugged at the same time.

One PyBoy instance per OS process (not thread): PyBoy wraps a C emulator core
with its own mutable global-ish state per instance, so processes -- with their
own memory space -- are the safe unit of parallelism here, not threads.
"""
import multiprocessing as mp

import numpy as np

from envs.mario_land_env import MarioLandEnv, OBS_DIM


def _worker(rom_path: str, n_steps: int, seed: int, conn):
    """Runs in its own process: boots one PyBoy instance, plays n_steps with a
    uniform-random policy, and ships the collected arrays back over the pipe."""
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
    parallelism.

    Returns a dict of stacked float32/int64 arrays, each shaped (n_envs, n_steps,
    ...): "obs" (..., OBS_DIM), "actions", "rewards", "dones".
    """
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
