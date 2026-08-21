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

THE EMBEDDING INITIALISATION IS A SCIENTIFIC KNOB TOO (`--embed-init-mode`,
default `legacy` = the historical behaviour; `--embed-scale`, default 1.0).

A diagnostic over 6,000 real observation steps (3,000 trained-policy, 3,000
random-policy) found the observation is DC-dominated: 77.70% of its energy is its
own mean, so 76.11% of the reservoir's input-current variance is DC. The LIF
neuron integrates DC with gain 1/(1-beta)=10.0 against AC's 1/sqrt(1-beta^2)=
2.2942 -- a 4.3589x amplification of exactly the component that carries no
information. The result is a frozen per-unit membrane offset (std 0.943583 across
units, range [-3.5080, +3.4847], threshold 1.0) that leaves 14.93% of units
permanently below -threshold (silent forever) and 14.50% permanently saturated.

`--embed-scale` alone cannot fix that -- it multiplies DC and AC together, and a
scale sweep floors at ~20% silent even at 32x. `--embed-init-mode centered` sets
the embedding's bias to -(W @ obs_mean), which is exactly "embed the centred
observation" because the embedding is linear, costs zero new parameters, and
leaves the bias trainable. Measured over 8 seeds: silent fraction 44.7403% ->
1.7532%, mean spike rate 0.024351 -> 0.020912, zero saturated units.

Applied IDENTICALLY to both arms, for the same reason per-group clipping is; see
models/embedding_init.py. Both settings are recorded in every checkpoint and every
log line, and both default to the historical behaviour.

THE NEURON MODEL IS THE THIRD SUCH KNOB (`--neuron-model`, default `lif` = the
historical behaviour), and it is the one knob here that is NOT applied to both
arms -- because it cannot be. A GRU has no spiking neuron model, so `--arm
baseline --neuron-model rf` RAISES rather than being quietly ignored: silence
would put a checkpoint and a `train_log.jsonl` labelled `neuron_model="rf"` on
disk for a run that trained a GRU, and a mislabelled run gets tabulated rather
than noticed.

`rf` is docs/EXPERIMENT_LOG.md §23's resonate-and-fire pilot: one frozen complex
pole per unit with |lambda| = beta = 0.9 held identical to LIF, so the memory
horizon is unchanged by construction and only a rotation is added. Zero new
trainable parameters (parameter parity stays 139,179 vs 132,715). The hypothesis
it tests, H10, is about the operating point rather than about task reward: the
measured defect is that the reservoir's spike rate runs away over a full run
(0.0209 at step 100,096 -> 0.1615 at step 1,000,064) because the trained
embedding's DC component grows 8.4x and nothing regulates it -- `centered` fixes
the STARTING point of a quantity nothing holds. A resonate-and-fire unit's DC
gain is a property of the frozen pole, not of a trainable bias, so it cannot
decay the same way: mean DC gain 10.0 -> 1.7846 with the AC accumulation gain
1/sqrt(1-beta^2) = 2.2942 exactly unchanged.

Unlike the two knobs above, `rf` is NOT a candidate default. The LIF path must
stay bit-identical to the code that produced `checkpoints_v2/`, because §23.9
performs no new LIF or GRU runs and uses the published ones as the pilot's
control; `tests/test_neuron_model_flag.py` reproduces a committed v2 training-log
prefix float for float to prove it (G0e-i). All three of `neuron_model`,
`rf_period_min` and `rf_period_max` are recorded in every checkpoint and every log
line -- and in the checkpoint they are more than metadata, since an `rf` state
dict holds buffers a LIF model does not, so a reader has to consult them BEFORE it
constructs anything (`neuron_config_from_checkpoint`).
"""
import argparse
import json
import os

import numpy as np
import torch

from envs.mario_land_env import MarioLandEnv, OBS_DIM, OBS_MEAN, OBS_MEAN_PHASE2A
from models.embedding_init import EMBED_INIT_MODES as _EMBED_INIT_MODES
from models.policy_value_gru import PolicyValueGRU
from models.policy_value_reservoir import PolicyValueReservoir
from models.spiking_reservoir import NEURON_MODELS as _NEURON_MODELS
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

# `legacy`   -- the embedding init every existing checkpoint was produced under
#               (zeroed bias on the reservoir arm, nn.Linear's default on the
#               baseline). THE DEFAULT, and it stays the default.
# `centered` -- embedding.bias := -(W @ OBS_MEAN), i.e. the embedding of the
#               DC-removed observation. Both arms, always. See the module
#               docstring and models/embedding_init.py.
# Imported rather than re-declared: the CLI's choices and the models' validation
# must be the same tuple, or a mode could be accepted here and rejected there.
EMBED_INIT_MODES = _EMBED_INIT_MODES

# Phase 2a's task axis (docs/DESIGN_ROADMAP_PHASE2.md §9 item 3, §12 OPEN-3,
# resolved 2026-08-21: the two-task set {1-1, 2-1}; 2-3 is DEFERRED, not dropped
# -- see that section for why). Keyed by the exact "--task" spelling, so
# `TASKS[spec]` and `parse_task(spec)` are the same lookup and `format_task` is
# its exact inverse. Deliberately an ALLOW-LIST, not a generic "any W-L pair"
# parser: envs/boot.py's world_level path is only EMPIRICALLY CONFIRMED
# (§14.1/§14.5) for the levels this project actually measured, and accepting an
# arbitrary pair would let a typo silently boot into an unverified level and
# train against nobody-knows-what -- exactly the failure envs/ram_map.py's own
# "do not add or trust an address without empirical confirmation" rule exists to
# prevent, applied to a level rather than an address.
TASKS = {"1-1": (1, 1), "2-1": (2, 1)}


def parse_task(spec: str) -> tuple:
    """'1-1' -> (1, 1). Raises ValueError naming the valid choices for anything
    not in TASKS -- see TASKS's own comment for why this is an allow-list rather
    than a generic parser."""
    if spec not in TASKS:
        raise ValueError(f"--task: unknown task {spec!r}; must be one of {sorted(TASKS)}")
    return TASKS[spec]


def format_task(task: tuple) -> str:
    """(1, 1) -> '1-1'. The exact inverse of parse_task/TASKS, so every tool that
    renders a task into a path fragment or a re-typeable CLI value (run_dir_for,
    scripts/run_training_matrix.py, scripts/run_phase2a_eval.py) agrees on the
    same spelling."""
    world, level = task
    return f"{world}-{level}"


# `lif` -- snntorch's Leaky, the neuron every existing checkpoint was produced
#          under. THE DEFAULT, and it stays the default: docs/EXPERIMENT_LOG.md
#          §23.5's G0e-i makes the published v2 LIF runs the resonate-and-fire
#          pilot's experimental control, which is only legitimate while this path
#          is bit-identical to the one that produced them.
# `rf`   -- resonate-and-fire (§23.2): one frozen complex pole per unit, |lambda|
#          = beta = 0.9 held identical to LIF so the memory horizon is unchanged
#          by construction, and only the rotation is added. Reservoir arm only.
# Imported for the same reason EMBED_INIT_MODES is: one tuple, so a mode cannot be
# accepted by the CLI and rejected by the model.
NEURON_MODELS = _NEURON_MODELS

# §23.2 fixes the resonant-period support BEFORE measurement and does not search
# it: T_min = 2 is the Nyquist period of the discrete-time system (nothing faster
# is representable), T_max = 32 is one quarter of the 128-step truncated-BPTT
# window, so every unit completes at least four cycles inside the horizon the
# readout ever receives a gradient over. They are defaults here AND in
# `SpikingReservoir.__init__`'s signature; `tests/test_neuron_model_flag.py`
# asserts the two agree via `inspect`, so the second copy cannot drift silently.
RF_PERIOD_MIN_DEFAULT = 2.0
RF_PERIOD_MAX_DEFAULT = 32.0

DEVICE = torch.device("cpu")

# One line of JSON per PPO update, appended live. `.jsonl` (not `.json`) because a
# run is appended to incrementally and may be killed at any point: a half-written
# JSON array is unparseable, a half-written JSONL file is just shorter.
TRAIN_LOG_NAME = "train_log.jsonl"


def run_dir_for(checkpoint_dir: str, arm: str, seed: int, run_tag: str = None,
                task: tuple = None) -> str:
    """Per-run output directory:
    `{checkpoint_dir}/{arm}[_task{W}-{L}]_seed{seed}[_{run_tag}]`.

    Both `arm` and `seed` are in the path because both collide otherwise. Writing
    every arm's checkpoints as `{checkpoint_dir}/step_{step}.pt` meant running
    `--arm baseline` then `--arm reservoir` with default args silently overwrote
    the first arm's checkpoints with the second's -- and the seed is in there for
    the same reason one step further out, since §5's comparison needs SEVERAL
    independently-trained checkpoints per arm sitting on disk at once.

    `run_tag` is the third coordinate, and it is a DATA-SAFETY requirement rather
    than a convenience. arm+seed is no longer a unique run identity now that
    `--grad-clip-mode` and `--embed-init-mode`/`--embed-scale` exist: re-running
    `--arm reservoir --seed 0` under `per-group` would land on `reservoir_seed0/`,
    i.e. straight on top of the 20
    completed runs (200 checkpoints) the results write-up is built from, and
    torch.save would overwrite each `step_N.pt` in place. `--run-tag per-group`
    sends the corrected re-run to `reservoir_seed0_per-group/` instead.

    With no tag AND no task the path is BYTE-IDENTICAL to what it has always
    been, so every existing run directory, resume path and analysis script still
    resolves. An empty string is treated as no tag (an empty `--run-tag ""` must
    not produce a trailing-underscore directory that silently forks the run
    layout).

    `task` is Phase 2a's fourth coordinate (docs/DESIGN_ROADMAP_PHASE2.md §9 item
    4), a `(world, level)` tuple, e.g. `(2, 1)`. `task=None` (the default) omits
    it entirely -- the historical, task-less path, unchanged. This is the SINGLE
    MOST IMPORTANT property of this function for Phase 2a: two tasks silently
    colliding on the same directory (a 2-1 specialist's checkpoints landing on
    top of a 1-1 specialist's) is exactly the class of bug
    docs/EXPERIMENT_LOG.md §19.4 found already happening once, in a shell
    completeness guard's unanchored glob -- see tests/test_task_axis.py for the
    explicit collision tests this function is held to.

    Positioned BEFORE `_seed`, not after: `{arm}_task1-1_seed0` and
    `{arm}_task2-1_seed0` therefore share no `_seed`-prefixed substring at all,
    so even a naive, unanchored glob for `{arm}_seed*` (the historical,
    task-less pattern this repo's other tooling was written against) cannot
    accidentally match a task-labelled directory -- the §19.4 substitution
    failure is structurally impossible here, not merely unlikely.
    """
    name = arm
    if task is not None:
        name += f"_task{format_task(task)}"
    name += f"_seed{seed}"
    if run_tag:
        name += f"_{run_tag}"
    return os.path.join(checkpoint_dir, name)


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


def build_model(arm: str, seed: int = 0, embed_init_mode: str = "legacy",
                embed_scale: float = 1.0, obs_mean=OBS_MEAN,
                neuron_model: str = "lif",
                rf_period_min: float = RF_PERIOD_MIN_DEFAULT,
                rf_period_max: float = RF_PERIOD_MAX_DEFAULT):
    """Construct one experimental arm's model plus its optimizer.

    Both arms are exposed through the same `(init_state_fn, step_fn)` pair so the
    rollout collector and the gradient replay below never branch on which arm
    they are driving.

    `seed` is the reservoir arm's FROZEN-WEIGHT seed (W_in and the TT cores). It is
    threaded in rather than hardcoded so a multi-seed comparison varies both arms,
    not just the GRU's trainable init -- see the module docstring. It does NOT seed
    the trainable init on either arm: that comes from the global RNG, which
    `run_training` seeds before calling this.

    `embed_init_mode`/`embed_scale` are handed to BOTH arms with the same values --
    that is a control requirement, argued at length in models/embedding_init.py, not
    a convenience. `obs_mean` defaults to `OBS_MEAN` (1-1's own measurement, Phase
    1's historical value) and is passed in from `envs.mario_land_env` rather than
    imported by the models, so the models stay game-agnostic (they take `obs_dim` as
    an argument) and the measured constant stays next to the observation
    construction it describes. `run_training` passes `OBS_MEAN_PHASE2A` instead
    when `--task` is set (docs/DESIGN_ROADMAP_PHASE2.md §9 item 3) -- callers that
    never heard of Phase 2a (every existing caller, including
    `training/evaluate.py`'s `build_model(arm)`) get the historical default
    unchanged. Only `embed_init_mode="centered"` ever reads this value; on
    `"legacy"` it is accepted and ignored, so passing the "wrong" mean there
    changes nothing.

    `neuron_model`/`rf_period_min`/`rf_period_max` are RESERVOIR-ARM ONLY, and the
    baseline arm REFUSES anything but the default rather than ignoring it. A GRU
    has no neuron model at all, so there is no behaviour a non-default value could
    select; accepting it silently would let `save_checkpoint` stamp
    `neuron_model="rf"` onto a checkpoint that trained an ordinary GRU, and a
    mislabelled run is worse than a crashed one because it gets tabulated rather
    than noticed. This is the one place that can catch it: every caller below reads
    the label off the model, not off its own arguments.
    """
    if embed_init_mode not in EMBED_INIT_MODES:
        raise ValueError(
            f"unknown embed_init_mode: {embed_init_mode!r}; expected one of "
            f"{EMBED_INIT_MODES}"
        )
    if neuron_model not in NEURON_MODELS:
        raise ValueError(
            f"unknown neuron_model: {neuron_model!r}; expected one of {NEURON_MODELS}"
        )
    if arm == "baseline" and neuron_model != "lif":
        raise ValueError(
            f"arm 'baseline' does not accept neuron_model={neuron_model!r}: the "
            "baseline is a trained GRU and has no spiking neuron model to choose. "
            "Ignoring the flag here would put a checkpoint and a train_log.jsonl "
            f"labelled neuron_model={neuron_model!r} on disk for a run that trained "
            "a GRU. Pass --arm reservoir, or leave --neuron-model at 'lif'."
        )
    if arm == "baseline":
        model = PolicyValueGRU(obs_dim=OBS_DIM, embed_dim=32, hidden_dim=192, n_actions=N_ACTIONS,
                               embed_init_mode=embed_init_mode, embed_scale=embed_scale,
                               obs_mean=obs_mean)

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
                                     tt_n_cores=4, context_len=64, seed=seed,
                                     embed_init_mode=embed_init_mode, embed_scale=embed_scale,
                                     obs_mean=obs_mean, neuron_model=neuron_model,
                                     rf_period_min=rf_period_min,
                                     rf_period_max=rf_period_max)
        init_state_fn = model.init_state

        # FOUR state components in BOTH neuron models -- `imem` is the
        # resonate-and-fire quadrature companion, an inert zeros tensor under LIF
        # that nothing reads (see PolicyValueReservoir.init_state). Unpacked
        # positionally here, so an arity that depended on the neuron model would
        # silently drop a component on one path instead of raising on either.
        def step_fn(m, obs, state):
            logits, value, mem, imem, spk, window = m(obs, *state)
            return logits, value, mem, imem, spk, window
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
    # Same reasoning as arm/seed: the embedding init is now part of a run's identity,
    # so it is stamped where `save_checkpoint` can read it off the model that was
    # actually built, not passed separately and possibly disagreeing.
    model._embed_init_mode = embed_init_mode
    model._embed_scale = float(embed_scale)
    # Same reasoning again, and it carries more weight here than for any other
    # label: the neuron model changes which BUFFERS the state dict contains, so a
    # checkpoint that did not carry it could not even be reconstructed -- see
    # `neuron_config_from_checkpoint`, which is how training/evaluate.py reads it
    # back before it builds anything.
    model._neuron_model = neuron_model
    model._rf_period_min = float(rf_period_min)
    model._rf_period_max = float(rf_period_max)
    return model, optimizer


def neuron_config_from_checkpoint(ckpt: dict) -> dict:
    """The `build_model` neuron-model keyword arguments a checkpoint was written
    under, as a dict ready to splat: `{"neuron_model", "rf_period_min",
    "rf_period_max"}`.

    THE READ ORDER THIS FUNCTION EXISTS TO ENFORCE: an `rf` checkpoint carries
    five buffers (`reservoir.rf.{omega,cos_omega,sin_omega,beta,threshold}`) that a
    LIF model does not have, so `load_state_dict` at default strictness REFUSES it.
    A loader therefore cannot build the model first and let the load overwrite the
    buffers, the way it can (and does) for every other label -- it has to read the
    file's own construction arguments before constructing anything.

    BACKWARD COMPATIBILITY, load-bearing, same rule as `load_checkpoint`'s: the 400
    checkpoints under `checkpoints/` and `checkpoints_v2/` predate all three keys
    and contain none of them. Every read goes through `.get(...)` with the
    historical default, so a pre-existing file reads back as
    lif/2.0/32.0 -- which is precisely what it is -- and evaluates exactly as it
    did before this flag existed. A direct index would make all 40 completed runs
    unloadable and take the published v1/v2 results with them.
    """
    return {
        "neuron_model": ckpt.get("neuron_model", "lif"),
        "rf_period_min": float(ckpt.get("rf_period_min", RF_PERIOD_MIN_DEFAULT)),
        "rf_period_max": float(ckpt.get("rf_period_max", RF_PERIOD_MAX_DEFAULT)),
    }


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

    `embed_init_mode`/`embed_scale` are stamped for the identical reason, with the
    identical getattr defaults ("legacy"/1.0 = the historical init): two checkpoints
    with the same arm+seed but different embedding initialisations are not the same
    experiment either.

    `task` is stamped the same way again, getattr default None (= the historical,
    task-less run -- see `run_dir_for`'s own docstring for why a task is a
    directory coordinate; here it is the same fact recorded INSIDE the file, so a
    checkpoint self-identifies even if it is ever copied or renamed out of its
    `run_dir_for`-derived directory).

    `neuron_model`/`rf_period_min`/`rf_period_max` go one step further than
    metadata: they are the arguments a READER has to reconstruct the model from
    before it can load this file at all, because an `rf` state dict contains
    buffers a LIF model does not have. See `neuron_config_from_checkpoint`. Same
    getattr defaults as everywhere else, so a model built outside `run_training`
    self-labels exactly as the 400 checkpoints already on disk implicitly are.
    """
    if getattr(model, "_arm", None) == "reservoir":
        model.assert_reservoir_frozen()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "arm": getattr(model, "_arm", None),
                "seed": getattr(model, "_seed", None),
                "grad_clip_mode": getattr(model, "_grad_clip_mode", "global"),
                "run_tag": getattr(model, "_run_tag", None),
                "embed_init_mode": getattr(model, "_embed_init_mode", "legacy"),
                "embed_scale": float(getattr(model, "_embed_scale", 1.0)),
                "task": getattr(model, "_task", None),
                "neuron_model": getattr(model, "_neuron_model", "lif"),
                "rf_period_min": float(
                    getattr(model, "_rf_period_min", RF_PERIOD_MIN_DEFAULT)),
                "rf_period_max": float(
                    getattr(model, "_rf_period_max", RF_PERIOD_MAX_DEFAULT))}, path)


def load_checkpoint(model, optimizer, path: str, expected_arm: str = None,
                    expected_seed: int = None, expected_grad_clip_mode: str = None,
                    expected_task: tuple = None,
                    expected_neuron_model: str = None) -> int:
    """Restore a checkpoint into `model`/`optimizer`, returning its step.

    `expected_arm`/`expected_seed` are checked BEFORE `load_state_dict`, so a
    mismatch fails with a sentence naming the actual problem instead of a shape
    error thrown from somewhere inside torch. Both default to None (unchecked) for
    callers that genuinely do not care which run a file came from; every production
    caller passes them.

    BACKWARD COMPATIBILITY, load-bearing: the 200 checkpoints already on disk were
    written before `grad_clip_mode`/`run_tag`/`embed_init_mode`/`embed_scale`/`task`
    existed and contain NONE of those keys. Every read of the new keys therefore
    goes through `.get(...)` with the historical default -- indexing them directly
    would turn all 20 completed runs into unloadable files (and take
    `training/evaluate.py`, the eval matrix and the write-up with them). A
    pre-existing checkpoint reads back as grad_clip_mode="global", run_tag=None,
    embed_init_mode="legacy", embed_scale=1.0, task=None, which is precisely what
    it is.

    Note the new keys change nothing about the RESTORE itself: `embed_init_mode`
    describes how the embedding was INITIALISED, and `load_state_dict` overwrites
    the embedding wholesale, so the labels are metadata for the results table rather
    than something this function has to act on.

    `expected_grad_clip_mode` WARNS rather than raises: resuming a run under a
    different clipping rule produces a checkpoint whose optimiser state was
    accumulated under the other rule, which is a scientific hazard but a legitimate
    thing to do deliberately -- and raising here would break resuming any of the
    existing runs.

    `expected_task` follows `expected_arm`/`expected_seed`'s pattern, not
    `expected_grad_clip_mode`'s: it RAISES, like arm/seed, because a task
    mismatch is architecturally the same class of mistake arm mismatch is -- a
    2-1 specialist's frozen weights and embedding-centering are not a valid
    starting point to keep training as a 1-1 run, any more than a reservoir
    checkpoint is a valid starting point for the baseline arm. `None` is the
    SAME "unchecked" sentinel `expected_arm`/`expected_seed` use, which is a
    genuine ambiguity here (None is also task-less runs' real, recorded value)
    -- `run_training` resolves it by always passing its own `task` argument
    (None for a task-less run, a tuple for a Phase 2a run) as `expected_task`,
    so the check activates exactly when Phase 2a's task flag is in use and is
    silently absent otherwise, matching this function's behaviour before `task`
    existed at all.

    `expected_neuron_model` RAISES, and deliberately not by the same rule. A
    grad-clip mismatch changes OPTIMISATION: both checkpoints are instances of the
    same architecture, the restored tensors all fit, and a deliberate switch
    mid-run is a coherent (if hazardous) experiment someone might mean to perform.
    A neuron-model mismatch changes the ARCHITECTURE: the two models do not have
    the same buffers (`rf` carries `reservoir.rf.omega` and four more), so the
    `load_state_dict` below cannot even succeed -- warning would buy nothing except
    that torch's unexpected-key dump lands three frames later with no mention of
    the flag that caused it. And in the one direction where the shapes DO happen to
    line up, silence would be worse still: the frozen reservoir's dynamics would
    change mid-run and the resulting checkpoint would be a valid instance of
    neither model. That is the same class of error as an `arm` mismatch, which has
    always raised, and it gets the same treatment.
    Backward compatibility is unaffected: a checkpoint with no `neuron_model` key
    reads back as "lif" (see `neuron_config_from_checkpoint`), which is what the
    400 existing files are, so resuming any of them under the default is a match.
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
    # .get, never ckpt["task"]: pre-existing checkpoints have no such key.
    recorded_task = ckpt.get("task")
    if expected_task is not None and recorded_task != expected_task:
        raise ValueError(
            f"checkpoint task mismatch: {path!r} was written under task "
            f"{recorded_task!r}, but it is being loaded for task {expected_task!r}. "
            "A checkpoint's frozen weights and embedding centering were shaped by "
            "the level it trained on; loading it as a different task's starting "
            "point is never valid. Check --task, or the checkpoint path."
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
    # .get via neuron_config_from_checkpoint, for the same reason: pre-existing
    # checkpoints have no such key and read back as "lif", which is what they are.
    recorded_neuron_model = neuron_config_from_checkpoint(ckpt)["neuron_model"]
    if expected_neuron_model is not None and recorded_neuron_model != expected_neuron_model:
        raise ValueError(
            f"checkpoint neuron_model mismatch: {path!r} was written by "
            f"neuron_model={recorded_neuron_model!r}, but it is being loaded into a "
            f"neuron_model={expected_neuron_model!r} model. Unlike the clipping rule, "
            "the neuron model is part of the ARCHITECTURE -- the two reservoirs do "
            "not even hold the same buffers -- so there is no state to carry across "
            "and the resulting run would be a valid instance of neither. Pass "
            f"--neuron-model {recorded_neuron_model} to continue that run."
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
                 grad_clip_mode: str = "global", run_tag: str = None,
                 embed_init_mode: str = "legacy", embed_scale: float = 1.0,
                 task: tuple = None,
                 neuron_model: str = "lif",
                 rf_period_min: float = RF_PERIOD_MIN_DEFAULT,
                 rf_period_max: float = RF_PERIOD_MAX_DEFAULT):
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

    `embed_init_mode`/`embed_scale` select the embedding initialisation, applied to
    BOTH arms with the same values:

      "legacy", 1.0  (default) the historical init, bit-identical to what every
                  existing checkpoint was produced under. Does not change, ever.
      "centered"  embedding.bias := -(W @ OBS_MEAN). Removes the observation's DC
                  component, which the LIF neuron otherwise amplifies 4.3589x over
                  the AC component and freezes into a per-unit membrane offset.
                  Measured over 8 seeds: silent-unit fraction 44.7403% -> 1.7532%.
      embed_scale multiplies the weight-init std (3.0 -> 3/sqrt(obs_dim*embed_dim)
                  on the reservoir arm). On its own it is a palliative, not a fix:
                  it scales DC and AC together and floors at ~20% silent.

    `neuron_model` selects the reservoir arm's frozen neuron dynamics, one of
    NEURON_MODELS, and is REJECTED on the baseline arm unless it is the default
    (see `build_model`):

      "lif"       (default) snntorch's Leaky, bit-identical to the path every
                  existing checkpoint was produced under. Does not change, ever --
                  docs/EXPERIMENT_LOG.md §23.5's G0e-i makes the published v2 LIF
                  runs the resonate-and-fire pilot's experimental control, and a
                  control that has drifted is not a control.
      "rf"        resonate-and-fire (§23.2): one frozen complex pole per unit,
                  |lambda| = beta = 0.9 held identical to LIF so the memory horizon
                  is unchanged by construction. Zero new trainable parameters.
                  Hypothesis H10 (§23.3): mean DC gain falls 10.0 -> 1.7846 while
                  the AC accumulation gain 1/sqrt(1-beta^2) = 2.2942 does not move
                  at all, so the DC/AC ratio flips from 4.3589 to 0.7779 -- and
                  because that is a property of the FROZEN pole rather than of a
                  trainable bias, it cannot decay as the embedding trains, which is
                  the failure mode `--embed-init-mode centered` has (§21.5).
      rf_period_min/rf_period_max bound the log-uniform resonant-period draw, in
                  env steps. §23.2 fixes them at 2 (the Nyquist period) and 32 (one
                  quarter of the 128-step BPTT window) BEFORE measurement and does
                  not search them; they are arguments only so the pre-registered
                  values are visible and recorded rather than buried in a default.

    `run_tag` appends a third coordinate to the output directory so a re-run under
    different settings cannot overwrite the completed matrix. All of these are
    recorded in every checkpoint and every log line, because a run whose clipping
    rule and embedding init are not written down next to its numbers cannot be
    interpreted later.

    Checkpoints and the per-update JSONL log go to `run_dir_for(checkpoint_dir,
    arm, seed, run_tag, task)`, not to `checkpoint_dir` itself, so concurrent/
    sequential runs of different arms, seeds, tags or tasks never overwrite each
    other. The returned stats dict carries `run_dir`/`log_path` so a caller never
    has to re-derive that.

    `task` is Phase 2a's axis (docs/DESIGN_ROADMAP_PHASE2.md §9 item 3):

      None            (default) EXACTLY Phase 1's behaviour -- power-on boot to
                      world 1-1 (`MarioLandEnv(world_level=None)`), `OBS_MEAN`
                      for `--embed-init-mode centered`, run dir
                      `{arm}_seed{N}[_{tag}]`. Does not change, ever.
      (world, level)  the env boots straight into that level instead
                      (`MarioLandEnv(world_level=task)`, envs/boot.py's
                      `game_wrapper.start_game` path), `OBS_MEAN_PHASE2A` (the
                      {1-1, 2-1} mixture mean) is used for `--embed-init-mode
                      centered` instead of `OBS_MEAN`, and the run dir gains a
                      task coordinate: `{arm}_task{W}-{L}_seed{N}[_{tag}]`.
                      Recorded in every checkpoint (`save_checkpoint`) so a
                      checkpoint self-identifies which level it was trained on.

    Nothing else branches on `task`: 2-1 is an ordinary platformer level and
    reuses the observation, reward, action set and termination logic unchanged
    (envs/mario_land_env.py's own docstring).
    """
    if grad_clip_mode not in GRAD_CLIP_MODES:
        raise ValueError(
            f"unknown grad_clip_mode: {grad_clip_mode!r}; expected one of {GRAD_CLIP_MODES}"
        )
    if embed_init_mode not in EMBED_INIT_MODES:
        raise ValueError(
            f"unknown embed_init_mode: {embed_init_mode!r}; expected one of "
            f"{EMBED_INIT_MODES}"
        )
    # Seeded before build_model, so it covers the trainable init on BOTH arms (and
    # the action sampling that follows). The reservoir's own frozen-weight seed is
    # a separate argument, threaded below -- see the module docstring.
    torch.manual_seed(seed)
    # task=None -> OBS_MEAN (Phase 1's historical value, unchanged); a real task
    # -> OBS_MEAN_PHASE2A (the {1-1, 2-1} mixture). See build_model's own
    # docstring for why this is the only place that decides between them.
    obs_mean = OBS_MEAN if task is None else OBS_MEAN_PHASE2A
    model, optimizer = build_model(arm, seed=seed, embed_init_mode=embed_init_mode,
                                   embed_scale=embed_scale, obs_mean=obs_mean,
                                   neuron_model=neuron_model,
                                   rf_period_min=rf_period_min,
                                   rf_period_max=rf_period_max)
    # Stamped onto the model for the same reason arm/seed are (build_model): a
    # checkpoint's labels are then structurally incapable of disagreeing with the
    # run that produced it.
    model._grad_clip_mode = grad_clip_mode
    model._run_tag = run_tag
    model._task = task
    start_step = 0
    if resume_from and os.path.exists(resume_from):
        start_step = load_checkpoint(model, optimizer, resume_from,
                                     expected_arm=arm, expected_seed=seed,
                                     expected_grad_clip_mode=grad_clip_mode,
                                     expected_task=task,
                                     expected_neuron_model=neuron_model)
    # dim=OBS_DIM: novelty is scored on the game state the agent reached, not on
    # the policy's logits. See collect_rollout_with_model's own note -- scoring
    # logits changes the reward FUNCTION per arm, not merely the reported metric.
    novelty_gate = NoveltyGate(dim=OBS_DIM, capacity=512, k=8)
    run_dir = run_dir_for(checkpoint_dir, arm, seed, run_tag, task=task)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, TRAIN_LOG_NAME)

    step = start_step
    last_checkpoint_step = start_step
    stats = {"mean_reward": 0.0, "mean_extrinsic_reward": 0.0, "final_step": step, "updates": 0}
    # ONE env for the whole run: rollouts continue the same episode instead of
    # restarting the level every `rollout_len` steps. world_level=task: None
    # reproduces the historical power-on-to-1-1 path unchanged; a real task boots
    # straight into that level instead (see this function's own docstring).
    env = MarioLandEnv(rom_path=rom_path, world_level=task)
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
                # On every line for the same reason grad_clip_mode is: a learning
                # curve whose embedding init is unknown cannot be compared with
                # another. Pre-existing log files simply lack these keys, which reads
                # back as the historical ("legacy", 1.0) exactly as it should.
                "embed_init_mode": embed_init_mode, "embed_scale": float(embed_scale),
                # Same reasoning again: pre-existing log lines simply lack "task",
                # which reads back as the historical task=None exactly as it should.
                "task": task,
                # On every line for the third time and the same reason: a learning
                # curve whose NEURON MODEL is unknown cannot be compared with
                # another, and the rf pilot's whole output is a comparison of two
                # curves. The period bounds ride along because two rf runs with
                # different bounds are different experiments, not one experiment
                # with a footnote. Pre-existing log files simply lack these keys,
                # which reads back as ("lif", 2.0, 32.0) exactly as it should.
                "neuron_model": neuron_model,
                "rf_period_min": float(rf_period_min),
                "rf_period_max": float(rf_period_max),
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
                  "grad_clip_mode": grad_clip_mode, "run_tag": run_tag,
                  "embed_init_mode": embed_init_mode, "embed_scale": float(embed_scale),
                  "task": task,
                  "neuron_model": neuron_model,
                  "rf_period_min": float(rf_period_min),
                  "rf_period_max": float(rf_period_max)})
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
    parser.add_argument("--embed-init-mode", choices=list(EMBED_INIT_MODES), default="legacy",
                        help="observation-embedding initialisation, applied identically to "
                             "BOTH arms. 'legacy' (default) is the init every existing "
                             "checkpoint was produced under, kept bit-identical so those "
                             "runs stay reproducible. 'centered' sets embedding.bias to "
                             "-(W @ OBS_MEAN), i.e. embeds the DC-removed observation -- "
                             "exact because the embedding is linear, and free because the "
                             "bias already exists. Real observations are 77.70%% DC energy, "
                             "and the LIF neuron amplifies DC over AC by 4.3589x, which "
                             "freezes a per-unit membrane offset (std 0.943583, threshold "
                             "1.0) that leaves 14.93%% of units silent forever and 14.50%% "
                             "saturated. Measured effect over 8 seeds: silent-unit fraction "
                             "44.7403%% -> 1.7532%%, spike rate 0.024351 -> 0.020912, zero "
                             "saturated units")
    parser.add_argument("--embed-scale", type=float, default=1.0,
                        help="multiplier on the embedding's weight-init std (default 1.0 = "
                             "the historical init; 3.0 gives 3/sqrt(obs_dim*embed_dim) on "
                             "the reservoir arm). A PALLIATIVE ON ITS OWN, not a fix: it "
                             "scales the DC and AC components together, so a sweep floors "
                             "at ~20%% silent units even at 32x. Use it with "
                             "--embed-init-mode centered, not instead of it")
    parser.add_argument("--neuron-model", choices=list(NEURON_MODELS), default="lif",
                        help="the RESERVOIR arm's frozen neuron dynamics; REJECTED on "
                             "--arm baseline, which is a GRU and has no neuron model "
                             "to choose. 'lif' (default) is snntorch's Leaky, the path "
                             "every existing checkpoint was produced under, kept "
                             "bit-identical so the published v2 runs stay this pilot's "
                             "experimental control (docs/EXPERIMENT_LOG.md §23.5, "
                             "G0e-i). 'rf' is resonate-and-fire (§23.2): one frozen "
                             "complex pole per unit at |lambda| = beta = 0.9, so the "
                             "memory horizon is unchanged by construction and only the "
                             "rotation is added. Zero new trainable parameters. "
                             "Predicted effect (H10, §23.3): mean DC gain 10.0 -> "
                             "1.7846 with the AC accumulation gain 2.2942 exactly "
                             "unchanged, flipping the DC/AC ratio 4.3589 -> 0.7779 -- "
                             "as a property of the frozen pole, so unlike a bias "
                             "initialisation it cannot decay as the embedding trains")
    parser.add_argument("--rf-period-min", type=float, default=RF_PERIOD_MIN_DEFAULT,
                        help="lower bound of the log-uniform resonant-period draw, in "
                             "env steps (default: 2.0 = the Nyquist period of the "
                             "discrete-time system; nothing faster is representable). "
                             "Ignored unless --neuron-model rf")
    parser.add_argument("--rf-period-max", type=float, default=RF_PERIOD_MAX_DEFAULT,
                        help="upper bound of the log-uniform resonant-period draw, in "
                             "env steps (default: 32.0 = one quarter of the 128-step "
                             "rollout / truncated-BPTT window, so every unit completes "
                             "at least four cycles inside the horizon the readout ever "
                             "receives a gradient over). Ignored unless --neuron-model "
                             "rf. Both bounds are pre-registered in §23.2 and are NOT "
                             "searched -- they are flags so the values are recorded, "
                             "not so they are tuned")
    parser.add_argument("--run-tag", default=None,
                        help="optional third coordinate on the output directory: "
                             "{checkpoint-dir}/{arm}_seed{seed}_{run-tag}/. USE IT for any "
                             "run that is not a plain default-settings run -- arm+seed "
                             "alone is no longer a unique identity now that "
                             "--grad-clip-mode exists, so an untagged corrected re-run "
                             "would overwrite the completed matrix in place. This now "
                             "covers --embed-init-mode/--embed-scale too")
    parser.add_argument("--task", choices=list(TASKS), default=None,
                        help="Phase 2a's task axis (docs/DESIGN_ROADMAP_PHASE2.md §9 item "
                             "3): which Super Mario Land level to train on. Unset "
                             "(default) is EXACTLY Phase 1's behaviour -- power-on boot "
                             "to world 1-1, OBS_MEAN, run dir {arm}_seed{seed}. Set, the "
                             "env boots straight into that level instead (envs/boot.py's "
                             "game_wrapper.start_game path), OBS_MEAN_PHASE2A (the "
                             "{1-1,2-1} mixture mean) is used for --embed-init-mode "
                             "centered in its place, and the run dir gains a task "
                             "coordinate: {arm}_task{world}-{level}_seed{seed}")
    args = parser.parse_args()
    task = parse_task(args.task) if args.task is not None else None
    stats = run_training(arm=args.arm, rom_path=args.rom, total_steps=args.steps,
                         n_envs=args.n_envs, rollout_len=args.rollout_len,
                         checkpoint_every=args.checkpoint_every,
                         checkpoint_dir=args.checkpoint_dir,
                         resume_from=args.resume_from, seed=args.seed,
                         grad_clip_mode=args.grad_clip_mode, run_tag=args.run_tag,
                         embed_init_mode=args.embed_init_mode, embed_scale=args.embed_scale,
                         task=task,
                         neuron_model=args.neuron_model,
                         rf_period_min=args.rf_period_min,
                         rf_period_max=args.rf_period_max)
    print(stats)
