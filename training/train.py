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

TWO LIFETIMES, deliberately different -- this is the part that is easy to get
wrong, so it is stated explicitly:

  * The ENV persists for the whole run. It is constructed once, before the loop,
    and closed once, after it. An earlier version built a fresh env per rollout,
    which restarted world 1-1 at every rollout boundary: the agent could never
    experience more than `rollout_len` env steps of the level however long
    training ran, which would have made the eventual arm comparison meaningless.
  * The MODEL's recurrent state is reset at every rollout boundary, on the
    collection side and identically in the replay -- standard truncated BPTT at
    a fixed sequence boundary. These two lifetimes are independent: persisting
    the emulator does not require persisting (or snapshotting) recurrent state,
    and resetting model state at the same logical point on both sides is exactly
    what keeps the replay bit-exact. NOTE: `training/evaluate.py` does NOT reset
    state within an episode by default, which is a deliberate but real train/eval
    regime mismatch -- see that module's "WHAT THIS HARNESS CANNOT TELL YOU".

SEEDING, AND WHY IT COVERS BOTH ARMS SYMMETRICALLY. `seed` drives three things at
once: `torch.manual_seed` (the trainable weights' random init on both arms, and
the action sampling `collect_rollout_with_model` draws from the global RNG), and
-- separately threaded through `build_model` -- `PolicyValueReservoir`'s own
`seed=` argument, i.e. the frozen reservoir's W_in and TT cores. Hardcoding the
latter (as an earlier version did) would have meant that across "different"
training seeds only the GRU arm's init actually varied while the reservoir arm
was always the exact same frozen instance -- an asymmetry in the very thing §5's
control is supposed to isolate. `evaluate.py`'s own docstring spells out why a
real §5 verdict needs several independently-trained checkpoints PER ARM; this is
what makes producing them possible, labelled (arm+seed in the path AND in the
checkpoint dict) and reproducible (same seed => same run).

EVERY UPDATE IS LOGGED, not just the last one. `run_training` returns a summary of
its FINAL update; a 100k-step run is ~780 updates, and a summary of the last one
is no learning curve, no divergence detector and nothing to plot. Each update
therefore appends one JSON object to `{run_dir}/train_log.jsonl`, flushed as it
goes, so a running job is inspectable mid-flight rather than only post-mortem.

GRADIENT CLIPPING IS A SCIENTIFIC KNOB, NOT AN IMPLEMENTATION DETAIL
(`--grad-clip-mode`, default `global` = the historical behaviour). A diagnostic on
`checkpoints/reservoir_seed0/step_500480.pt` measured the following, and it is
written down here because it is the reason the `per-group` mode exists at all:

  * The reservoir arm backpropagates through `rollout_len`=128 sequential steps of
    a FROZEN spiking reservoir. The gradient reaching the trainable `embedding`
    grows exponentially in the replay-chain length L: global pre-clip norm 2.171 at
    L=1, 52.19 at L=32, 1.111e4 at L=64, 3.988e6 at L=96, 1.258e9 at L=128 -- a
    per-step multiplier of ~1.22. The readout's OWN gradient grows only ~sqrt(L)
    (1.5 -> 8.9), and the baseline GRU arm is flat across the same sweep
    (29.97 / 51.71 / 48.18). So the explosion is not "the loss", it is one path.
  * Consequence: `embedding.weight` + `embedding.bias` -- 416 parameters, 0.3% of
    the trainable budget -- carry 100.0000% of the global gradient norm. The 29
    readout/head tensors (138,763 params, 99.7% of the budget) contribute ~5e-15%.
  * ONE `clip_grad_norm_` over the whole parameter list therefore computes a clip
    coefficient of 3.976e-10 from the exploding 0.3% and applies it to the 99.7%
    that is not exploding, taking the readout's post-clip gradient norm to
    3.52e-09.
  * Adam is invariant to a CONSTANT rescaling of the gradient but NOT to a
    time-varying one. The clip coefficient's max/median ratio over 1000 updates is
    2.63e5, so `sqrt(v_hat)` (beta2=0.999, long memory) is set by the rare
    non-exploding updates while `m_hat` (beta1=0.9, short memory) tracks the
    typical one. The readout's median `|m_hat|/sqrt(v_hat)` collapses to 7.475e-04
    versus 1.346e-01 on the baseline GRU: the readout is, in effect, frozen.
  * Measured counterfactual on one step, same gradients and same restored Adam
    state: per-group clipping raises the readout's median `||dp||/||p||` from
    1.9034e-05 to 6.4186e-03 -- a factor of 337. Raising Adam's eps from 1e-8 to
    1e-12 gives 1.11x, so the eps floor is NOT the mechanism; the shared clip
    coefficient is.

An arm comparison run under the global rule is therefore not cleanly testing the
architectural question: it is partly measuring which arm's optimiser survived its
own clipping. `per-group` clips each parameter group to MAX_GRAD_NORM separately,
so an exploding group cannot suppress a non-exploding one. It is applied
IDENTICALLY to both arms (the baseline GRU does not need it, but a control that
only one arm receives is not a control).

Default stays `global` and BIT-IDENTICAL: 20 completed runs / 200 checkpoints on
disk have to stay exactly reproducible, and a results write-up depends on them.
`--run-tag` exists for the same reason -- see `run_dir_for`.
"""
import argparse
import json
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

# `global`   -- one clip_grad_norm_ over every trainable parameter at once. The
#               historical behaviour, and the default, because the 20 completed
#               runs on disk were produced under it and must stay reproducible.
# `per-group` -- one clip_grad_norm_ per parameter GROUP (see
#               `group_trainable_parameters`). Fixes the coupling documented at
#               length in the module docstring: a 0.3%-of-the-budget embedding
#               whose gradient explodes 1.22^128 cannot then scale the readout's
#               update to 3.5e-09.
GRAD_CLIP_MODES = ("global", "per-group")

DEVICE = torch.device("cpu")

# One line of JSON per PPO update, appended live. `.jsonl` (not `.json`) because a
# run is appended to incrementally and may be killed at any point: a half-written
# JSON array is unparseable, a half-written JSONL file is just shorter.
TRAIN_LOG_NAME = "train_log.jsonl"


def run_dir_for(checkpoint_dir: str, arm: str, seed: int, run_tag: str = None) -> str:
    """Per-run output directory: `{checkpoint_dir}/{arm}_seed{seed}[_{run_tag}]`.

    Both coordinates are in the path because both collide otherwise. Writing every
    arm's checkpoints as `{checkpoint_dir}/step_{step}.pt` meant running `--arm
    baseline` then `--arm reservoir` with default args silently overwrote the
    first arm's checkpoints with the second's -- and the seed is in there for the
    same reason one step further out, since §5's comparison needs SEVERAL
    independently-trained checkpoints per arm sitting on disk at once.

    `run_tag` is the third coordinate, and it is a DATA-SAFETY requirement rather
    than a convenience. arm+seed is no longer a unique run identity now that
    `--grad-clip-mode` exists: re-running `--arm reservoir --seed 0` under
    `per-group` would land on `reservoir_seed0/`, i.e. straight on top of the 20
    completed runs (200 checkpoints) the results write-up is built from, and
    torch.save would overwrite each `step_N.pt` in place. `--run-tag per-group`
    sends the corrected re-run to `reservoir_seed0_per-group/` instead.

    With no tag the path is BYTE-IDENTICAL to what it has always been, so every
    existing run directory, resume path and analysis script still resolves. An
    empty string is treated as no tag (an empty `--run-tag ""` must not produce a
    trailing-underscore directory that silently forks the run layout).
    """
    if run_tag:
        return os.path.join(checkpoint_dir, f"{arm}_seed{seed}_{run_tag}")
    return os.path.join(checkpoint_dir, f"{arm}_seed{seed}")


def group_trainable_parameters(model):
    """Trainable parameters bucketed by the FIRST dot-separated component of their
    `named_parameters()` name -- i.e. by top-level submodule.

    Discovered, never hardcoded, so this keeps working if a module is added; the
    names it currently yields are asserted in tests/test_grad_clip_modes.py so a
    rename fails loudly there instead of silently collapsing everything into one
    group (which would quietly restore the global-clip bug under a per-group flag).

    Today:
      baseline  -> {embedding, gru, actor_head, critic_head}   (10 tensors)
      reservoir -> {embedding, readout}                        (31 tensors)

    Note the asymmetry is REAL and is exactly the point: on the reservoir arm the
    two groups are the exploding one (embedding, 416 params) and the frozen-out one
    (readout, 138,763 params). The frozen reservoir itself holds zero nn.Parameters
    and no gradients, so it cannot appear here at all.

    Returns a dict preserving `named_parameters()` order, so the clipping order is
    deterministic and a run stays reproducible.
    """
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        groups.setdefault(name.split(".")[0], []).append(param)
    return groups


def apply_grad_clipping(model, grad_clip_mode: str = "global"):
    """Clip gradients in place. Returns (global_pre_clip_norm, group_pre_clip_norms).

    `global_pre_clip_norm` is a tensor and always means the same thing in both
    modes -- the norm over ALL trainable gradients, before any clipping -- so the
    `grad_norm` field of an old log and a new one remain directly comparable.
    `group_pre_clip_norms` is a `{group_name: float}` dict in `per-group` mode and
    None in `global` mode.

    Why None rather than "computed anyway for the logs": the `global` branch below
    is the path 20 completed runs were produced under, and its bit-exactness is
    worth more than a richer log line. Not one extra tensor op runs in it. The
    branch itself is the ONLY thing between `backward()` and `optimizer.step()`
    that differs between the modes.
    """
    if grad_clip_mode not in GRAD_CLIP_MODES:
        raise ValueError(
            f"unknown grad_clip_mode: {grad_clip_mode!r}; expected one of {GRAD_CLIP_MODES}"
        )
    trainable = [p for p in model.parameters() if p.requires_grad]
    if grad_clip_mode == "global":
        # VERBATIM the historical call. Do not "clean this up".
        return torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM), None

    # per-group. Measure the global norm FIRST and without touching the gradients:
    # `get_total_norm` is the exact function `clip_grad_norm_` computes its return
    # value with, so `grad_norm` is the same number it would have logged, and the
    # measurement provably cannot perturb the update.
    grads = [p.grad for p in trainable if p.grad is not None]
    total_norm = (torch.nn.utils.get_total_norm(grads) if grads
                  else torch.tensor(0.0, device=DEVICE))
    group_norms = {}
    for group_name, params in group_trainable_parameters(model).items():
        # Each group gets its OWN clip coefficient, so a group whose norm is 1.2e9
        # scales only itself down and leaves a group whose norm is 8.9 alone.
        # Applied identically on BOTH arms: the baseline GRU's gradients do not
        # explode and will mostly pass through untouched, but a treatment given to
        # one arm and not the other stops being a control -- the whole point of
        # this experiment is that the two arms differ ONLY in architecture.
        group_norms[group_name] = float(
            torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM).item()
        )
    return total_norm, group_norms


def build_model(arm: str, seed: int = 0):
    """Construct one experimental arm's model plus its optimizer.

    Both arms are exposed through the same `(init_state_fn, step_fn)` pair so the
    rollout collector and the gradient replay below never branch on which arm
    they are driving.

    `seed` is the reservoir arm's FROZEN-WEIGHT seed (W_in and the TT cores). It is
    threaded in rather than hardcoded so a multi-seed comparison varies both arms,
    not just the GRU's trainable init -- see the module docstring. It does NOT seed
    the trainable init on either arm: that comes from the global RNG, which
    `run_training` seeds before calling this.
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
                                     tt_n_cores=4, context_len=64, seed=seed)
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
    # Carried on the model rather than passed to save_checkpoint separately: a
    # checkpoint's arm/seed labels are then structurally incapable of disagreeing
    # with the model that produced them.
    model._arm = arm
    model._seed = seed
    return model, optimizer


def save_checkpoint(model, optimizer, step: int, path: str):
    """Write one checkpoint, arm/seed-labelled, after the frozen tripwire passes.

    The tripwire (spec §3: "a runtime tripwire asserting the reservoir's weights are
    bit-identical to their initialization at every checkpoint") runs BEFORE the
    write, on the reservoir arm only -- the baseline GRU has no frozen component to
    check. Before the write specifically, so a reservoir that has somehow been
    mutated can never reach disk and be evaluated later as if it were frozen.

    The `arm`/`seed` labels are read off the model (`build_model` stamps them
    there, so they cannot disagree with the weights) and written into the dict, so a
    checkpoint file self-identifies. Without them, loading a reservoir checkpoint
    into a baseline model surfaced as a state-dict shape mismatch several frames
    away from the actual mistake.

    `grad_clip_mode`/`run_tag` are stamped the same way, for the same reason: the
    clipping rule is now a knob, and two checkpoints with identical arm+seed but
    different clipping rules are NOT the same experiment. A file that does not
    carry the rule it was trained under cannot be placed in a results table. The
    getattr defaults ("global"/None) are the historical behaviour, so a model built
    outside `run_training` (every direct `build_model` caller, including tests)
    self-labels exactly as the 200 checkpoints already on disk implicitly are.
    """
    if getattr(model, "_arm", None) == "reservoir":
        model.assert_reservoir_frozen()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "arm": getattr(model, "_arm", None),
                "seed": getattr(model, "_seed", None),
                "grad_clip_mode": getattr(model, "_grad_clip_mode", "global"),
                "run_tag": getattr(model, "_run_tag", None)}, path)


def load_checkpoint(model, optimizer, path: str, expected_arm: str = None,
                    expected_seed: int = None, expected_grad_clip_mode: str = None) -> int:
    """Restore a checkpoint into `model`/`optimizer`, returning its step.

    `expected_arm`/`expected_seed` are checked BEFORE `load_state_dict`, so a
    mismatch fails with a sentence naming the actual problem instead of a shape
    error thrown from somewhere inside torch. Both default to None (unchecked) for
    callers that genuinely do not care which run a file came from; every production
    caller passes them.

    BACKWARD COMPATIBILITY, load-bearing: the 200 checkpoints already on disk were
    written before `grad_clip_mode`/`run_tag` existed and contain NEITHER key.
    Every read of the new keys therefore goes through `.get(...)` with the
    historical default -- indexing them directly would turn all 20 completed runs
    into unloadable files (and take `training/evaluate.py`, the eval matrix and the
    write-up with them). A pre-existing checkpoint reads back as
    grad_clip_mode="global", run_tag=None, which is precisely what it is.

    `expected_grad_clip_mode` WARNS rather than raises: resuming a run under a
    different clipping rule produces a checkpoint whose optimiser state was
    accumulated under the other rule, which is a scientific hazard but a legitimate
    thing to do deliberately -- and raising here would break resuming any of the
    existing runs.
    """
    # weights_only=True: these checkpoints hold only tensors/state dicts, and
    # torch.load's default (weights_only=False) unpickles arbitrary Python
    # objects -- unnecessary code-execution risk even for our own checkpoints.
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    recorded_arm = ckpt.get("arm")
    if expected_arm is not None and recorded_arm != expected_arm:
        raise ValueError(
            f"checkpoint arm mismatch: {path!r} was written by arm "
            f"{recorded_arm!r}, but it is being loaded into arm {expected_arm!r}. "
            "The two arms are architecturally different models; loading one into "
            "the other is never valid. Check --arm, or the checkpoint path."
        )
    recorded_seed = ckpt.get("seed")
    if expected_seed is not None and recorded_seed != expected_seed:
        raise ValueError(
            f"checkpoint seed mismatch: {path!r} was written by seed "
            f"{recorded_seed!r}, but the run resuming it is labelled seed "
            f"{expected_seed!r}. Continuing would produce checkpoints labelled "
            f"{expected_seed} whose frozen reservoir is in fact seed "
            f"{recorded_seed}'s -- pass --seed {recorded_seed} to continue that run."
        )
    # .get, never ckpt["grad_clip_mode"]: pre-existing checkpoints have no such key.
    recorded_clip_mode = ckpt.get("grad_clip_mode", "global")
    if expected_grad_clip_mode is not None and recorded_clip_mode != expected_grad_clip_mode:
        print(
            f"WARNING: {path!r} was trained with grad_clip_mode={recorded_clip_mode!r} "
            f"but is being resumed with grad_clip_mode={expected_grad_clip_mode!r}. "
            "The restored Adam moments were accumulated under the other rule; the "
            "resulting run is not a clean instance of either. Use --run-tag so this "
            "does not land in a directory labelled as one of them."
        )
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    # The loaded frozen weights are now this run's frozen weights, so the §3
    # tripwire's reference point moves to them. Without this, a resumed reservoir
    # run would compare against the weights this process constructed and then
    # immediately overwrote, and trip on its very first checkpoint.
    if hasattr(model, "snapshot_frozen_weights"):
        model.snapshot_frozen_weights()
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


def _append_log(log_path: str, record: dict):
    """Append one JSON object as a line. Reopened per call rather than held open:
    ~780 opens over a 100k-step run is free next to the emulator, and every line is
    on disk (not in a buffer) the instant it is written, which is the whole point of
    a log you can watch a running job through."""
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def run_training(arm: str, rom_path: str, total_steps: int, n_envs: int, rollout_len: int,
                 checkpoint_every: int, checkpoint_dir: str, resume_from: str = None,
                 seed: int = 0,
                 gamma: float = 0.99, lam: float = 0.95, clip_eps: float = 0.2,
                 value_coef: float = 0.5, entropy_coef: float = 0.01,
                 novelty_coef: float = 0.05,
                 grad_clip_mode: str = "global", run_tag: str = None):
    """Collect -> GAE -> replay-with-gradients -> one PPO update, repeated.

    `n_envs` is accepted (it is part of the CLI/interface contract and of the
    resumable run config) but currently unused: collection is single-process, see
    training/rollout.py's module docstring. It is not silently dropped from the
    signature so that adding parallel model-driven collection later does not
    change every caller.

    `grad_clip_mode` selects the clipping rule, one of GRAD_CLIP_MODES:

      "global"    (default) one clip over all trainable parameters. Bit-identical
                  to the behaviour every existing checkpoint was produced under --
                  this default is what keeps those runs reproducible, so it does
                  not change, ever.
      "per-group" one clip per top-level submodule. Applied identically to both
                  arms. Exists because under "global" the reservoir arm's readout
                  receives a gradient scaled by 3.976e-10 and effectively stops
                  learning; see the module docstring for the measurements (337x
                  larger readout updates, clip-coefficient max/median 2.63e5).

    `run_tag` appends a third coordinate to the output directory so a re-run under
    different settings cannot overwrite the completed matrix. Both are recorded in
    every checkpoint and every log line, because a run whose clipping rule is not
    written down next to its numbers cannot be interpreted later.

    Checkpoints and the per-update JSONL log go to `run_dir_for(checkpoint_dir,
    arm, seed, run_tag)`, not to `checkpoint_dir` itself, so concurrent/sequential
    runs of different arms, seeds or tags never overwrite each other. The returned
    stats dict carries `run_dir`/`log_path` so a caller never has to re-derive that.
    """
    if grad_clip_mode not in GRAD_CLIP_MODES:
        raise ValueError(
            f"unknown grad_clip_mode: {grad_clip_mode!r}; expected one of {GRAD_CLIP_MODES}"
        )
    # Seeded before build_model, so it covers the trainable init on BOTH arms (and
    # the action sampling that follows). The reservoir's own frozen-weight seed is
    # a separate argument, threaded below -- see the module docstring.
    torch.manual_seed(seed)
    model, optimizer = build_model(arm, seed=seed)
    # Stamped onto the model for the same reason arm/seed are (build_model): a
    # checkpoint's labels are then structurally incapable of disagreeing with the
    # run that produced it.
    model._grad_clip_mode = grad_clip_mode
    model._run_tag = run_tag
    start_step = 0
    if resume_from and os.path.exists(resume_from):
        start_step = load_checkpoint(model, optimizer, resume_from,
                                     expected_arm=arm, expected_seed=seed,
                                     expected_grad_clip_mode=grad_clip_mode)
    # dim=OBS_DIM: novelty is scored on the game state the agent reached, not on
    # the policy's logits. See collect_rollout_with_model's own note -- scoring
    # logits changes the reward FUNCTION per arm, not merely the reported metric.
    novelty_gate = NoveltyGate(dim=OBS_DIM, capacity=512, k=8)
    run_dir = run_dir_for(checkpoint_dir, arm, seed, run_tag)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, TRAIN_LOG_NAME)

    step = start_step
    last_checkpoint_step = start_step
    stats = {"mean_reward": 0.0, "mean_extrinsic_reward": 0.0, "final_step": step, "updates": 0}
    # ONE env for the whole run: rollouts continue the same episode instead of
    # restarting world 1-1 every `rollout_len` steps.
    env = MarioLandEnv(rom_path=rom_path)
    try:
        obs, _ = env.reset()
        while step < total_steps:
            rollout = collect_rollout_with_model(
                env=env, obs=obs, model=model,
                model_state_fns=(model._init_state_fn, model._step_fn),
                n_steps=rollout_len, novelty_gate=novelty_gate, novelty_coef=novelty_coef,
            )
            obs = rollout["final_obs"]  # next rollout resumes exactly where this one stopped

            rewards = torch.tensor(rollout["rewards"], dtype=torch.float32)
            extrinsic_rewards = torch.tensor(rollout["extrinsic_rewards"], dtype=torch.float32)
            dones = torch.tensor(rollout["dones"], dtype=torch.float32)
            truncation_values = torch.tensor(rollout["truncation_values"], dtype=torch.float32)
            # (T+1,): the stored per-step values plus the bootstrap for the state
            # the rollout stopped in (0.0 if the episode ended on the last step,
            # where GAE multiplies it by not_done = 0 anyway).
            values = torch.tensor(rollout["values"] + [rollout["last_value"]], dtype=torch.float32)
            # Truncation bootstrap. `dones` cuts the GAE recursion at BOTH kinds
            # of boundary (the next buffer entry belongs to a different episode
            # either way), which would silently zero the future of a step-limit
            # truncation -- an episode that was still very much alive. Folding
            # gamma * V(s_T) into that step's reward restores exactly the delta a
            # real bootstrap would have produced, without changing compute_gae's
            # (T+1,)-values contract. Zero everywhere except truncated steps, so
            # this is a no-op on rollouts that contain none.
            gae_rewards = rewards + gamma * truncation_values
            advantages, returns = compute_gae(gae_rewards, values, dones, gamma=gamma, lam=lam)

            obs_seq = torch.as_tensor(np.asarray(rollout["obs"], dtype=np.float32))
            actions = torch.as_tensor(np.asarray(rollout["actions"], dtype=np.int64))
            old_log_probs = torch.tensor(rollout["log_probs"], dtype=torch.float32)

            new_logits, new_values = replay_rollout(model, obs_seq, dones)
            # log-probability of the SAME actions that were actually taken, under
            # the freshly recomputed distribution.
            new_log_probs = torch.distributions.Categorical(logits=new_logits).log_prob(actions)

            p_loss = ppo_policy_loss(new_log_probs, old_log_probs, advantages, clip_eps=clip_eps)
            v_loss = value_loss(new_values, returns)
            entropy = entropy_bonus(new_logits)
            total_loss = p_loss + value_coef * v_loss - entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            # The one line the two modes differ in. In "global" this is the exact
            # historical call; in "per-group" the gradients are clipped submodule by
            # submodule so the exploding embedding cannot scale the readout's update
            # into the noise floor (module docstring).
            grad_norm, grad_norm_groups = apply_grad_clipping(model, grad_clip_mode)
            optimizer.step()

            step += rollout_len
            stats = {
                # `mean_reward` is the optimised reward (extrinsic + novelty),
                # which is what the loss actually saw. `mean_extrinsic_reward` is
                # the one to compare arms on: the novelty term is an exploration
                # subsidy, not a score, so it must not be part of the scoreboard.
                "mean_reward": float(rewards.mean().item()),
                "mean_extrinsic_reward": float(extrinsic_rewards.mean().item()),
                "final_step": step,
                "updates": stats["updates"] + 1,
                "policy_loss": float(p_loss.item()),
                "value_loss": float(v_loss.item()),
                "entropy": float(entropy.item()),
                "total_loss": float(total_loss.item()),
                # Pre-clip GLOBAL gradient norm: the cheapest honest evidence that
                # the update was real rather than a no-op. Same meaning in both
                # modes, so this field stays comparable against every log line the
                # 20 completed runs wrote.
                "grad_norm": float(grad_norm.item()),
                # Per-group pre-clip norms, or None in "global" mode. This is the
                # field that makes the pathology visible while a run is in flight:
                # under "global", embedding ~1e9 next to readout ~9 is the whole
                # bug in one line.
                "grad_norm_groups": grad_norm_groups,
            }
            # The learning curve, written as it happens. `final_step`/`updates` are
            # renamed to `step`/`update` here because in a per-update record they
            # are this row's coordinates, not a run summary's end state.
            # grad_clip_mode/run_tag are on EVERY line, not just the checkpoint: a
            # log file gets copied and plotted on its own, and a learning curve
            # whose clipping rule is unknown cannot be compared with another.
            _append_log(log_path, {
                "arm": arm, "seed": seed, "step": step, "update": stats["updates"],
                "grad_clip_mode": grad_clip_mode, "run_tag": run_tag,
                **{k: v for k, v in stats.items() if k not in ("final_step", "updates")},
            })
            if step - last_checkpoint_step >= checkpoint_every:
                save_checkpoint(model, optimizer, step,
                                os.path.join(run_dir, f"step_{step}.pt"))
                last_checkpoint_step = step
    finally:
        env.close()
    # Unconditional final save: `total_steps` is rarely an exact multiple of
    # `checkpoint_every`, and without this the last (i.e. best-trained) weights of
    # a run would never reach disk. Same naming scheme as the periodic saves, so
    # --resume-from takes it directly; re-writing an identical file is harmless
    # when the periodic save already landed on this exact step.
    save_checkpoint(model, optimizer, step, os.path.join(run_dir, f"step_{step}.pt"))
    stats.update({"arm": arm, "seed": seed, "run_dir": run_dir, "log_path": log_path,
                  "grad_clip_mode": grad_clip_mode, "run_tag": run_tag})
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
    parser.add_argument("--seed", type=int, default=0,
                        help="training seed: trainable init + action sampling on both "
                             "arms, AND the reservoir arm's frozen weights. Outputs go "
                             "to {checkpoint-dir}/{arm}_seed{seed}/. Comparing the two "
                             "arms needs several seeds per arm (see training/evaluate.py)")
    parser.add_argument("--grad-clip-mode", choices=list(GRAD_CLIP_MODES), default="global",
                        help="gradient-clipping rule. 'global' (default) is one clip over "
                             "all trainable parameters -- the rule every existing "
                             "checkpoint was trained under, kept bit-identical so those "
                             "runs stay reproducible. 'per-group' clips each top-level "
                             "submodule separately, on BOTH arms: under 'global' the "
                             "reservoir arm's 416-parameter embedding explodes to a "
                             "gradient norm of ~1e9 and drags the clip coefficient to "
                             "3.976e-10, which scales the readout's own (non-exploding) "
                             "gradient down to 3.5e-09 and effectively freezes 99.7%% of "
                             "the trainable budget. Measured effect of switching: 337x "
                             "larger readout parameter updates")
    parser.add_argument("--run-tag", default=None,
                        help="optional third coordinate on the output directory: "
                             "{checkpoint-dir}/{arm}_seed{seed}_{run-tag}/. USE IT for any "
                             "run that is not a plain default-settings run -- arm+seed "
                             "alone is no longer a unique identity now that "
                             "--grad-clip-mode exists, so an untagged corrected re-run "
                             "would overwrite the completed matrix in place")
    args = parser.parse_args()
    stats = run_training(arm=args.arm, rom_path=args.rom, total_steps=args.steps,
                         n_envs=args.n_envs, rollout_len=args.rollout_len,
                         checkpoint_every=args.checkpoint_every,
                         checkpoint_dir=args.checkpoint_dir,
                         resume_from=args.resume_from, seed=args.seed,
                         grad_clip_mode=args.grad_clip_mode, run_tag=args.run_tag)
    print(stats)
