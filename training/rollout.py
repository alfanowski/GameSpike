"""Rollout collection: a multi-process random-policy throughput baseline, and a
single-process model-driven collector used by the actual training loop.

The two paths are kept side by side, not merged:

  * `collect_rollout_random_policy` is the throughput baseline (Task 10). It is
    deliberately policy-agnostic -- uniform-random actions only -- so that a
    rollout-mechanics bug (a multiprocessing deadlock, a shape mismatch) and a
    model bug (a bad forward pass) are never debugged at the same time.
  * `collect_rollout_with_model` (Task 11) drives a real policy, single-process.
    Combining the process-per-env parallelism with a real model is explicitly
    deferred: a model-driven multi-process collector needs either per-process
    model replicas kept in sync with the learner or a batched action server, and
    neither belongs in the task that first proves the learning loop works.

One PyBoy instance per OS process (not thread): PyBoy wraps a C emulator core
with its own mutable global-ish state per instance, so processes -- with their
own memory space -- are the safe unit of parallelism here, not threads.
"""
import multiprocessing as mp

import numpy as np
import torch

from envs.mario_land_env import MarioLandEnv, OBS_DIM
from training.novelty_gate import NoveltyGate


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


def collect_rollout_with_model(env_ctor, model, model_state_fns, n_steps: int,
                               novelty_gate: NoveltyGate, novelty_coef: float = 0.05) -> dict:
    """Single-process rollout driven by a real policy-value model.

    `model_state_fns = (init_state_fn, step_fn)` so this works for both
    PolicyValueGRU and PolicyValueReservoir without depending on either
    concretely: `init_state_fn(batch_size, device)` returns the model's initial
    recurrent state as a TUPLE, and `step_fn(model, obs, state)` returns
    `(logits, value, *next_state)` -- the trailing elements being whatever that
    model threads forward (a GRU hidden state; the reservoir's mem/spk/window).

    Episode boundaries reset BOTH the env and the model state, so no recurrent
    state ever leaks across an episode.

    The whole loop runs under `torch.no_grad()`: every value this function keeps
    is turned into a Python float, so the PPO-relevant quantities are frozen
    "old" values by construction, and building an autograd graph here would only
    pin the entire rollout's activations in memory (the reservoir threads its
    state forward, so the graph would span all n_steps).

    Returns a dict with `obs`, `actions`, `rewards`, `dones`, `log_probs`,
    `values` (each length n_steps), plus `extrinsic_rewards` and `last_value`.

    `rewards` is what PPO optimises: extrinsic + novelty_coef * novelty.
    `extrinsic_rewards` is the env's reward alone, kept separately because it is
    the only fair scoreboard for the baseline-vs-reservoir comparison -- the
    intrinsic term rewards an arm for having diverse logits, which is precisely
    the kind of thing the two arms differ in, so scoring the experiment on the
    combined reward would confound the result.

    `last_value` is the bootstrap V(s_T) that
    compute_gae wants as the (T+1)-th value -- the critic's own estimate of the
    state the rollout stopped in, or exactly 0.0 when the final step ended the
    episode (there is no future to bootstrap from). Reusing values[-1] as the
    bootstrap instead, as a stand-in, would silently bias every truncated
    rollout's last few advantages.
    """
    env = env_ctor()
    init_state_fn, step_fn = model_state_fns
    device = torch.device("cpu")
    obs_buf, act_buf, rew_buf, done_buf, logp_buf, val_buf = [], [], [], [], [], []
    ext_rew_buf = []
    try:
        obs, _ = env.reset()
        state = init_state_fn(1, device)
        done = False
        with torch.no_grad():
            for _ in range(n_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                logits, value, *state = step_fn(model, obs_t, state)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                # The novelty signal summarises the policy's own reaction to the
                # state (the logits vector), matching NoveltyGate's dim=n_actions.
                # Score BEFORE push, so a state is never counted as its own
                # nearest neighbour (which would drive every score to 0).
                state_vec = logits.squeeze(0)
                novelty = novelty_gate.score(state_vec)
                novelty_gate.push(state_vec)
                next_obs, reward, terminated, truncated, _ = env.step(int(action.item()))
                done = bool(terminated or truncated)
                obs_buf.append(obs)
                act_buf.append(int(action.item()))
                ext_rew_buf.append(float(reward))
                rew_buf.append(float(reward) + novelty_coef * novelty)
                done_buf.append(float(done))
                logp_buf.append(float(log_prob.item()))
                val_buf.append(float(value.item()))
                obs = next_obs
                if done:
                    obs, _ = env.reset()
                    state = init_state_fn(1, device)
            if done or n_steps == 0:
                last_value = 0.0
            else:
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                _, boot_value, *_ = step_fn(model, obs_t, state)
                last_value = float(boot_value.item())
    finally:
        env.close()
    return dict(obs=obs_buf, actions=act_buf, rewards=rew_buf, dones=done_buf,
                log_probs=logp_buf, values=val_buf, extrinsic_rewards=ext_rew_buf,
                last_value=last_value)
