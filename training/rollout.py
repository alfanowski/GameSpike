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


def collect_rollout_with_model(env, obs, model, model_state_fns, n_steps: int,
                               novelty_gate: NoveltyGate, novelty_coef: float = 0.05) -> dict:
    """Collect `n_steps` from an ALREADY-RUNNING env, driven by a real model.

    LIFECYCLE: the caller owns the env. It constructs it, performs the initial
    `reset()`, passes the resulting observation in as `obs`, feeds `final_obs`
    back in on the next call, and closes it when training ends. This function
    calls `env.reset()` ONLY when an episode genuinely ends. The earlier design
    took an `env_ctor` and built/closed a fresh env per call, which silently
    restarted world 1-1 at every rollout boundary -- so however long training
    ran, the agent could never experience more than `n_steps` of the level.

    MODEL STATE, by contrast, IS reset at every rollout boundary, here and
    identically in the gradient replay: standard truncated BPTT at a fixed
    sequence boundary. The two are independent -- only the emulator persists --
    and resetting model state on both sides at the same logical point is what
    keeps the replay bit-exact against collection.

    `model_state_fns = (init_state_fn, step_fn)` so this works for both
    PolicyValueGRU and PolicyValueReservoir without depending on either
    concretely: `init_state_fn(batch_size, device)` returns the model's initial
    recurrent state as a TUPLE, and `step_fn(model, obs, state)` returns
    `(logits, value, *next_state)` -- the trailing elements being whatever that
    model threads forward (a GRU hidden state; the reservoir's mem/spk/window).

    The whole loop runs under `torch.no_grad()`: every value this function keeps
    is turned into a Python float, so the PPO-relevant quantities are frozen
    "old" values by construction, and building an autograd graph here would only
    pin the entire rollout's activations in memory (the reservoir threads its
    state forward, so the graph would span all n_steps).

    Returns a dict of per-step lists (`obs`, `actions`, `rewards`,
    `extrinsic_rewards`, `dones`, `terminateds`, `truncateds`,
    `truncation_values`, `log_probs`, `values`), plus scalars `last_value` and
    `final_obs`.

    TERMINATED vs TRUNCATED are kept apart, not collapsed into `done`, because
    Task 9's `compute_gae` contract treats them differently: a true terminal
    state has no future to bootstrap from (value 0.0), while a step-limit
    truncation is an ongoing episode whose future must be bootstrapped with a
    real critic estimate V(s_T). `dones` (their OR) is still returned, because
    the GAE recursion and the replay's state resets must both cut at either kind
    of boundary -- what differs is only the value that gets bootstrapped.

    `truncation_values[t]` is V(s_{t+1}) at a truncated (non-terminal) step and
    0.0 everywhere else. The caller folds `gamma * truncation_values` into the
    reward before calling compute_gae: with `dones[t] = 1` the GAE recursion
    zeroes its own next-state term, so adding gamma*V(s_T) to the reward
    reproduces the correct delta exactly (see the identity test in
    tests/test_rollout.py). That is how a mid-rollout truncation gets a proper
    bootstrap without changing compute_gae's (T+1,)-values contract.

    `last_value` is the end-of-rollout bootstrap: V of the state the rollout
    stopped in when it stopped mid-episode, or 0.0 when the final step ended the
    episode (in which case GAE multiplies it by not_done = 0 anyway).
    """
    init_state_fn, step_fn = model_state_fns
    device = torch.device("cpu")
    obs_buf, act_buf, rew_buf, ext_rew_buf = [], [], [], []
    done_buf, term_buf, trunc_buf, trunc_val_buf = [], [], [], []
    logp_buf, val_buf = [], []
    state = init_state_fn(1, device)
    episode_ended = False
    with torch.no_grad():
        for _ in range(n_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits, value, *state = step_fn(model, obs_t, state)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            next_obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            terminated, truncated = bool(terminated), bool(truncated)

            # Curiosity scores the OBSERVATION the agent actually reached -- the
            # game state -- not the policy's logits. Scoring logits rewarded
            # "having an unusual action distribution" rather than "seeing an
            # unusual game state", which is (a) not curiosity and (b) a moving
            # target: the buffer goes stale on every policy update, because the
            # same state starts producing different logits, so stored neighbours
            # stop meaning anything. Worse, the two arms differ in exactly that
            # property, so it perturbed the reward FUNCTION per arm, not just the
            # reported metric. Score BEFORE push, so a state is never counted as
            # its own nearest neighbour.
            novelty_vec = torch.as_tensor(next_obs, dtype=torch.float32)
            novelty = novelty_gate.score(novelty_vec)
            novelty_gate.push(novelty_vec)

            # A step-limit truncation is an ongoing episode: its future is worth
            # a real critic estimate. Computed here, BEFORE the reset, because it
            # needs the pre-reset observation and the recurrent state that
            # produced it.
            truncation_value = 0.0
            if truncated and not terminated:
                boot_t = torch.as_tensor(next_obs, dtype=torch.float32).unsqueeze(0)
                _, trunc_value, *_ = step_fn(model, boot_t, state)
                truncation_value = float(trunc_value.item())

            episode_ended = terminated or truncated
            obs_buf.append(obs)
            act_buf.append(int(action.item()))
            ext_rew_buf.append(float(reward))
            rew_buf.append(float(reward) + novelty_coef * novelty)
            term_buf.append(float(terminated))
            trunc_buf.append(float(truncated))
            done_buf.append(float(episode_ended))
            trunc_val_buf.append(truncation_value)
            logp_buf.append(float(log_prob.item()))
            val_buf.append(float(value.item()))

            if episode_ended:
                obs, _ = env.reset()
                state = init_state_fn(1, device)
            else:
                obs = next_obs
        if episode_ended or n_steps == 0:
            last_value = 0.0
        else:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            _, boot_value, *_ = step_fn(model, obs_t, state)
            last_value = float(boot_value.item())
    return dict(obs=obs_buf, actions=act_buf, rewards=rew_buf,
                extrinsic_rewards=ext_rew_buf, dones=done_buf,
                terminateds=term_buf, truncateds=trunc_buf,
                truncation_values=trunc_val_buf, log_probs=logp_buf,
                values=val_buf, last_value=last_value, final_obs=obs)
