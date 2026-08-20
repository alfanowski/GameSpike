"""End-to-end PPO training loop: model + rollout + PPO losses + checkpointing.

This is where every earlier task meets: the real Mario Land env (Task 5), the two
competing policy-value models (Tasks 6-7), the novelty gate (Task 8), the PPO
core math (Task 9) and rollout collection (Task 10).

THE UPDATE, in full (the part that has to be right, so it is written down here
rather than left implicit):

  1. Collect `rollout_len` steps with the CURRENT policy, under no_grad. The
     log-probs and values stored there are the "old" (behaviour-policy)
     quantities PPO's ratio is measured against.
  2. Compute GAE advantages/returns from the stored rewards/values/dones.
  3. RE-RUN the model over the same stored observations, in the same order, with
     gradients this time, mirroring exactly how collection threaded recurrent
     state (fresh state at the start, fresh state after every `done`). Both
     models' `forward` is a SINGLE-TIMESTEP function that explicitly threads
     state -- there is no whole-sequence entry point -- so this replay is a
     sequential loop, not a batched call.
  4. Losses: clipped policy surrogate + value_coef * value MSE - entropy_coef *
     entropy. The entropy term is SUBTRACTED: PPO's objective is
     `L_clip - c1*L_vf + c2*S` to be MAXIMISED, and `ppo_policy_loss` already
     returns the negated surrogate (a loss to minimise), so flipping the whole
     objective's sign leaves the value term positive and the entropy term
     negative. Getting this sign backwards would drive the policy toward
     determinism and kill exploration, so it is stated explicitly.
  5. One optimizer step per rollout.

Deliberately NOT here (kept for a follow-up plan, because each one is a separate
scientific knob and this task is the one that proves the loop learns at all):
multi-epoch minibatched updates over the same rollout, advantage normalisation,
a learning-rate schedule, and multi-process model-driven collection.

KNOWN LIMITATION, flagged rather than buried: every rollout constructs a fresh
env (that is `collect_rollout_with_model`'s contract) and therefore restarts
world 1-1 from the beginning, so the agent only ever experiences the first
`rollout_len` env steps of the level. That is exactly what makes the gradient
replay in step 3 correct -- collection and replay both start from a fresh model
state -- so the two are coupled and must be changed together: persisting the env
across rollouts requires snapshotting the recurrent state at rollout start and
replaying from that snapshot instead of from `init_state_fn`. Fix both together
before running the real baseline-vs-reservoir comparison at long horizons.
"""
import argparse
import os

import numpy as np
import torch

from envs.mario_land_env import MarioLandEnv, OBS_DIM
from models.policy_value_gru import PolicyValueGRU
from models.policy_value_reservoir import PolicyValueReservoir
from training.novelty_gate import NoveltyGate
from training.rollout import collect_rollout_with_model
from training.ppo import compute_gae, ppo_policy_loss, value_loss, entropy_bonus

N_ACTIONS = len(MarioLandEnv.ACTIONS)
LEARNING_RATE = 3e-4

# Standard PPO gradient clipping. It matters more here than in a feed-forward
# PPO: the replay backpropagates through `rollout_len` sequential recurrent steps
# (and, on the reservoir arm, through surrogate spike gradients), which is
# exactly the setting where a single outlier advantage produces an exploding
# update. Clipping rescales the gradient, it never zeroes it.
MAX_GRAD_NORM = 0.5

DEVICE = torch.device("cpu")


def build_model(arm: str):
    """Construct one experimental arm's model plus its optimizer.

    Both arms are exposed through the same `(init_state_fn, step_fn)` pair so the
    rollout collector and the gradient replay below never branch on which arm
    they are driving.
    """
    if arm == "baseline":
        model = PolicyValueGRU(obs_dim=OBS_DIM, embed_dim=32, hidden_dim=192, n_actions=N_ACTIONS)

        # PolicyValueGRU.init_hidden returns a BARE tensor, but the collector and
        # the replay both unpack state as `logits, value, *state = step_fn(...)`,
        # i.e. always a sequence. Wrap it in a 1-tuple so `state[0]` is valid on
        # the very first call as well as every call after -- without this wrapper
        # the first call would pass a raw (1, B, hidden) tensor whose `state[0]`
        # slices to (B, hidden), a silent shape bug only a real forward surfaces.
        def init_state_fn(batch_size, device):
            return (model.init_hidden(batch_size, device),)

        def step_fn(m, obs, state):
            logits, value, h_next = m(obs, state[0])
            return logits, value, h_next
    elif arm == "reservoir":
        model = PolicyValueReservoir(obs_dim=OBS_DIM, embed_dim=32, reservoir_size=8192,
                                     n_actions=N_ACTIONS, use_tensor_train=True, tt_rank=8,
                                     tt_n_cores=4, context_len=64, seed=0)
        init_state_fn = model.init_state

        def step_fn(m, obs, state):
            logits, value, mem, spk, window = m(obs, *state)
            return logits, value, mem, spk, window
    else:
        raise ValueError(f"unknown arm: {arm}")
    # requires_grad filter is what keeps the frozen reservoir frozen: the
    # optimizer is never even handed its buffers, so no update rule can touch
    # them (SpikingReservoir holds zero nn.Parameters in the first place --
    # belt and braces, verified in tests/test_train_smoke.py).
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE
    )
    model._init_state_fn = init_state_fn
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


def replay_rollout(model, obs_seq: torch.Tensor, dones: torch.Tensor):
    """Re-run the model over stored observations WITH gradients.

    Mirrors `collect_rollout_with_model`'s state threading exactly: start from a
    fresh state, feed the stored observations one at a time in the collected
    order, and reset the state after every step whose stored `done` flag is set.
    Any deviation here (a batched forward, a state that is not reset on episode
    boundaries) would evaluate the stored actions under a DIFFERENT hidden state
    than the one that actually produced them, which silently corrupts the PPO
    ratio while still looking like it trains.

    Nothing is detached: these outputs are what carries the gradient.
    Returns (logits (T, n_actions), values (T,)).
    """
    init_state_fn, step_fn = model._init_state_fn, model._step_fn
    state = init_state_fn(1, DEVICE)
    logits_seq, values_seq = [], []
    for t in range(obs_seq.shape[0]):
        logits, value, *state = step_fn(model, obs_seq[t:t + 1], state)
        logits_seq.append(logits.squeeze(0))   # (n_actions,)
        values_seq.append(value.squeeze(0))    # scalar
        if bool(dones[t]):
            state = init_state_fn(1, DEVICE)
    return torch.stack(logits_seq), torch.stack(values_seq)


def run_training(arm: str, rom_path: str, total_steps: int, n_envs: int, rollout_len: int,
                 checkpoint_every: int, checkpoint_dir: str, resume_from: str = None,
                 gamma: float = 0.99, lam: float = 0.95, clip_eps: float = 0.2,
                 value_coef: float = 0.5, entropy_coef: float = 0.01,
                 novelty_coef: float = 0.05):
    """Collect -> GAE -> replay-with-gradients -> one PPO update, repeated.

    `n_envs` is accepted (it is part of the CLI/interface contract and of the
    resumable run config) but currently unused: collection is single-process, see
    training/rollout.py's module docstring. It is not silently dropped from the
    signature so that adding parallel model-driven collection later does not
    change every caller.
    """
    model, optimizer = build_model(arm)
    start_step = 0
    if resume_from and os.path.exists(resume_from):
        start_step = load_checkpoint(model, optimizer, resume_from)
    novelty_gate = NoveltyGate(dim=N_ACTIONS, capacity=512, k=8)
    os.makedirs(checkpoint_dir, exist_ok=True)

    step = start_step
    last_checkpoint_step = start_step
    stats = {"mean_reward": 0.0, "final_step": step, "updates": 0}
    while step < total_steps:
        rollout = collect_rollout_with_model(
            env_ctor=lambda: MarioLandEnv(rom_path=rom_path), model=model,
            model_state_fns=(model._init_state_fn, model._step_fn),
            n_steps=rollout_len, novelty_gate=novelty_gate, novelty_coef=novelty_coef,
        )

        rewards = torch.tensor(rollout["rewards"], dtype=torch.float32)
        dones = torch.tensor(rollout["dones"], dtype=torch.float32)
        # (T+1,): the stored per-step values plus the bootstrap for the state the
        # rollout stopped in (0.0 if it stopped because the episode ended).
        values = torch.tensor(rollout["values"] + [rollout["last_value"]], dtype=torch.float32)
        advantages, returns = compute_gae(rewards, values, dones, gamma=gamma, lam=lam)

        obs_seq = torch.as_tensor(np.asarray(rollout["obs"], dtype=np.float32))
        actions = torch.as_tensor(np.asarray(rollout["actions"], dtype=np.int64))
        old_log_probs = torch.tensor(rollout["log_probs"], dtype=torch.float32)

        new_logits, new_values = replay_rollout(model, obs_seq, dones)
        # log-probability of the SAME actions that were actually taken, under the
        # freshly recomputed distribution.
        new_log_probs = torch.distributions.Categorical(logits=new_logits).log_prob(actions)

        p_loss = ppo_policy_loss(new_log_probs, old_log_probs, advantages, clip_eps=clip_eps)
        v_loss = value_loss(new_values, returns)
        entropy = entropy_bonus(new_logits)
        total_loss = p_loss + value_coef * v_loss - entropy_coef * entropy

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], MAX_GRAD_NORM)
        optimizer.step()

        step += rollout_len
        stats = {
            "mean_reward": float(rewards.mean().item()),
            "final_step": step,
            "updates": stats["updates"] + 1,
            "policy_loss": float(p_loss.item()),
            "value_loss": float(v_loss.item()),
            "entropy": float(entropy.item()),
            "total_loss": float(total_loss.item()),
            # Pre-clip gradient norm: the cheapest honest evidence that the
            # update was real rather than a no-op.
            "grad_norm": float(grad_norm.item()),
        }
        if step - last_checkpoint_step >= checkpoint_every:
            save_checkpoint(model, optimizer, step, os.path.join(checkpoint_dir, f"step_{step}.pt"))
            last_checkpoint_step = step
    return stats


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
