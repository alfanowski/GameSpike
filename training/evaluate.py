"""Evaluation harness: plays a trained checkpoint and reports what it scored.

This is the artifact that answers design doc §5's mandatory-control question --
does the frozen reservoir beat the matched-parameter trained GRU, or not -- so
its job is not merely to produce a number but to produce a number that can be
argued with. Three decisions carry that weight, and each one is a deliberate
correction of the original task brief's reference code:

1. ACTIONS ARE SAMPLED FROM THE POLICY, NOT ARGMAXED.
   The brief picked `argmax(logits)`. Everything downstream of that choice is
   deterministic: PyBoy is deterministic, `envs/boot.py` is frame-deterministic,
   and `MarioLandEnv.reset()` reboots to a bit-identical state (its own docstring
   says the `seed` argument changes nothing). A greedy policy against that env
   replays the SAME episode every time -- identical actions, identical
   trajectory, identical return. `n_episodes=10` would then average ten copies of
   one number and report a spread of exactly zero, which is not a weak
   measurement but a fake one. Sampling from `Categorical(logits)` is also what
   `collect_rollout_with_model` does during training, so the evaluated policy is
   the policy that was actually trained, not a different (greedy) one.

2. THE RANDOMNESS IS OWNED, SEEDED PER EPISODE, AND REPORTED.
   Episode i draws its actions from a private `torch.Generator` seeded
   `seed + i`. Private, so an unrelated `torch.manual_seed` elsewhere in the
   process cannot move the published numbers and so evaluation does not perturb
   the caller's RNG; per-episode, so episodes genuinely differ from each other
   while the whole run reproduces exactly from the reported `seed`. The seeds
   used come back in the results dict. `torch.multinomial(dist.probs, ...,
   generator=g)` is precisely what `Categorical.sample()` executes internally --
   the same draw, just from a stream this module controls.

3. SPREAD IS REPORTED ALONGSIDE EVERY MEAN.
   A bare mean cannot support "arm A beats arm B": with the only stochasticity
   being the policy's own sampling, a 5% gap between two means may be entirely
   noise. Each metric therefore comes with a sample standard deviation and the
   raw per-episode list, plus the standard error on the extrinsic return, which
   is the quantity an actual comparison claim has to clear. With `n_episodes=1`
   these are NaN, not 0.0: no spread exists to estimate from one sample, and 0.0
   would read as "measured, and there was none".

Two smaller lifetime rules, both per-episode:

  * The model's recurrent state is re-initialised for every episode (an episode
    is a fresh playthrough; carrying hidden state across a reboot would evaluate
    episode 2 under a state that belongs to episode 1).
  * The novelty gate is rebuilt for every episode, unlike training, which keeps
    one sliding buffer for the whole run. Under a shared gate, episode 1 scores
    against an empty buffer and episode 3 against a full one, so the combined
    return would carry an ordering artefact that shows up as spread while being
    no measurement at all. A per-episode gate makes the episodes exchangeable,
    which is the property the reported standard deviation assumes.

`mean_extrinsic_return` is the scoreboard. `mean_combined_return` is the reward
the loss optimised (extrinsic + novelty subsidy) and is reported for diagnosis
only -- per training/train.py's own note, curiosity is an exploration subsidy,
not a score, and must never be what the arms are ranked on.

WHAT THIS HARNESS CANNOT TELL YOU. Read this before quoting any number it prints
as the answer to design doc §5, because the spread it reports is narrower than
the spread that actually matters:

  * THE REPORTED SPREAD IS POLICY-SAMPLING VARIANCE, AND NOTHING ELSE. The
    emulator, the boot sequence and `reset()` are all deterministic, and every
    episode starts from a bit-identical state, so the only thing that differs
    between episodes is which actions the policy happened to draw. There is no
    environment variance, no start-state variance, no opponent variance -- none
    of it exists in this setup, so none of it is in the error bars.
  * IT SAYS NOTHING ABOUT TRAINING-RUN VARIANCE, WHICH IN DEEP RL USUALLY
    DOMINATES. Every number here comes from ONE checkpoint, i.e. one weight
    init and one rollout ordering. Two identically-configured training runs of
    the same arm routinely land far apart -- further apart than the two arms'
    means will differ. Evaluating a single checkpoint per arm cannot separate
    "this architecture is better" from "this particular run got lucky", however
    many episodes it averages over, because more episodes shrink the wrong error
    bar.
  * SO A REAL ARM COMPARISON NEEDS SEVERAL INDEPENDENTLY-TRAINED CHECKPOINTS PER
    ARM (several training seeds), each evaluated here, with the arms compared
    ACROSS training seeds -- not one checkpoint per arm evaluated over many
    episodes. This module is the per-checkpoint instrument for that experiment,
    not the experiment. Running it twice, once per arm, produces a comparison
    that looks publishable and is not. (`training/train.py --seed N` produces
    those checkpoints: the seed drives both arms' trainable init AND the
    reservoir arm's frozen weights, and lands in `{arm}_seed{N}/`.)
  * THE RECURRENT-STATE REGIME HERE IS NOT THE ONE EITHER ARM WAS TRAINED IN.
    Training resets the model's recurrent state at every rollout boundary --
    every `rollout_len` steps, 128 by default -- as deliberate truncated BPTT
    (see training/train.py's "TWO LIFETIMES"). This harness, by default,
    initialises state ONCE per episode and never resets it again, so the policy
    runs continuously for up to `max_steps_per_episode` (3000) steps: more than
    20x the horizon it ever saw a gradient over. Both arms are therefore scored
    in a regime neither was trained under, and there is no reason to assume that
    penalises them equally -- memory horizon is precisely the axis on which a
    frozen 8192-dim reservoir and a trained 192-dim GRU are most likely to
    differ, so this mismatch sits directly on top of the quantity §5 is trying
    to measure. It is not a bug (evaluating an episode as one continuous
    playthrough is the honest thing to measure) but it IS an uncontrolled
    variable, and a number quoted from here without it stated is a number
    quoted without its main caveat. `state_reset_interval=rollout_len` runs the
    matched-regime counterpart; reporting BOTH is what actually separates "this
    arm is better" from "this arm degrades more slowly past its training
    horizon".
"""
import argparse
import json
import math
from statistics import fmean, stdev

import torch

from envs.mario_land_env import MarioLandEnv, OBS_DIM
from training.novelty_gate import NoveltyGate
from training.train import build_model, load_checkpoint

DEVICE = torch.device("cpu")

# Same buffer GEOMETRY as the training loop's gate. NOT the same scale: training
# runs one gate for the whole run, so its buffer is permanently warm, while this
# harness starts every episode with an empty one (see the per-episode rationale
# above -- a shared gate across identical restarts would measure "have I replayed
# this level before", which is worse). A cold buffer scores higher, because early
# steps have few stored neighbours to be close to: measured on a 200-step episode,
# the first 8 steps average 0.545 novelty against 0.141 for the last 8. Re-scoring
# that same trajectory against a buffer already holding it inflates the total
# subsidy by +174% at 200 steps and +14.8% at 1000 -- i.e. the distortion is
# severe on short evaluation episodes and fades as they lengthen. The defined
# maximal 1.0 of the very first step is NOT the main term (2.4% of a 200-step
# episode's subsidy, 0.8% of a 1000-step one); the warm-up window is.
#
# Consequence: `mean_combined_return` is comparable ACROSS ARMS (both are scored
# by this identical procedure) and comparable across episodes of equal length, but
# it is NOT on the same scale as the reward the training loss actually optimised,
# and it inflates as `max_steps_per_episode` shrinks. `mean_extrinsic_return` --
# the scoreboard -- is untouched by any of this.
NOVELTY_CAPACITY = 512
NOVELTY_K = 8


def _summarise(name: str, values) -> dict:
    """mean + sample spread for one metric, plus the raw per-episode values.

    ddof=1 (sample, not population): these episodes are a sample drawn from the
    policy's action distribution, not the whole population of possible episodes.
    Below two samples there is nothing to estimate, and NaN says so honestly.
    """
    n = len(values)
    std = stdev(values) if n >= 2 else float("nan")
    return {
        f"{name}s": list(values),
        f"mean_{name}": float(fmean(values)),
        f"std_{name}": float(std),
        f"sem_{name}": float(std / math.sqrt(n)),
    }


def run_evaluation(arm: str, checkpoint_path: str, rom_path: str, n_episodes: int,
                   max_steps_per_episode: int = 3000, novelty_coef: float = 0.05,
                   seed: int = 0, state_reset_interval: int = None) -> dict:
    """Play `n_episodes` with `arm`'s checkpoint and report per-episode statistics.

    Episode i is driven by a private generator seeded `seed + i`, so the run is
    reproducible from `seed` alone while the episodes still differ from one
    another. Returns the keys documented in `_summarise` for `extrinsic_return`,
    `combined_return` and `episode_length`, plus `arm`, `n_episodes`, `seed`,
    `episode_seeds` and `state_reset_interval`.

    `state_reset_interval` (default None = never reset within an episode) re-inits
    the model's recurrent state every N env steps, mirroring what training does at
    every rollout boundary. Set it to the run's `rollout_len` to score the arms in
    the recurrent-state regime they were actually trained in; leave it None to
    score one continuous playthrough. These measure different things and the
    difference between them is itself informative -- see the module docstring's
    "THE RECURRENT-STATE REGIME HERE IS NOT THE ONE EITHER ARM WAS TRAINED IN".
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")
    if state_reset_interval is not None and state_reset_interval < 1:
        raise ValueError(
            f"state_reset_interval must be >= 1 or None, got {state_reset_interval}")

    # seed=0 is build_model's own default and is irrelevant here: `load_checkpoint`
    # overwrites every buffer, the reservoir's frozen W_in/TT cores included, with
    # the ones the checkpoint was actually trained with (they are persistent
    # buffers, so they are in the state dict). The construction seed only decides
    # what gets thrown away.
    model, optimizer = build_model(arm)
    load_checkpoint(model, optimizer, checkpoint_path, expected_arm=arm)
    model.eval()
    init_state_fn, step_fn = model._init_state_fn, model._step_fn

    env = MarioLandEnv(rom_path=rom_path, max_episode_steps=max_steps_per_episode)
    episode_seeds = [seed + i for i in range(n_episodes)]
    extrinsic_returns, combined_returns, lengths = [], [], []
    try:
        with torch.no_grad():
            for episode_seed in episode_seeds:
                generator = torch.Generator(device=DEVICE)
                generator.manual_seed(episode_seed)
                # dim=OBS_DIM: novelty is scored on the game state the agent
                # reached, not on the policy's logits (dim=N_ACTIONS). Scoring
                # logits rewards "having an unusual action distribution" rather
                # than "seeing an unusual game state" -- and since the two arms
                # differ in exactly that property, it would perturb the compared
                # quantity per arm. Mirrors collect_rollout_with_model exactly.
                novelty_gate = NoveltyGate(dim=OBS_DIM, capacity=NOVELTY_CAPACITY, k=NOVELTY_K)
                obs, _ = env.reset()
                state = init_state_fn(1, DEVICE)
                extrinsic_total, combined_total, steps = 0.0, 0.0, 0
                done = False
                while not done:
                    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    logits, _value, *state = step_fn(model, obs_t, state)
                    # Identical to Categorical(logits=logits).sample(), drawn from
                    # this episode's own generator. See point 2 in the module
                    # docstring: greedy selection here would make every episode
                    # the same episode.
                    dist = torch.distributions.Categorical(logits=logits)
                    action = int(torch.multinomial(dist.probs, 1, generator=generator).item())

                    obs, reward, terminated, truncated, _ = env.step(action)

                    # Score BEFORE push, so a state is never its own nearest
                    # neighbour -- same ordering as the training collector.
                    novelty_vec = torch.as_tensor(obs, dtype=torch.float32)
                    novelty = novelty_gate.score(novelty_vec)
                    novelty_gate.push(novelty_vec)

                    extrinsic_total += float(reward)
                    combined_total += float(reward) + novelty_coef * novelty
                    steps += 1
                    # Optional matched-regime mode: reset recurrent state on the
                    # same cadence training does, so the policy is scored over the
                    # horizon it actually received gradients over. Placed after the
                    # step so an interval of N gives runs of exactly N steps.
                    if state_reset_interval is not None and steps % state_reset_interval == 0:
                        state = init_state_fn(1, DEVICE)
                    done = bool(terminated) or bool(truncated)
                extrinsic_returns.append(extrinsic_total)
                combined_returns.append(combined_total)
                lengths.append(float(steps))
    finally:
        env.close()

    results = {"arm": arm, "n_episodes": n_episodes, "seed": seed,
               "episode_seeds": episode_seeds,
               "state_reset_interval": state_reset_interval}
    results.update(_summarise("extrinsic_return", extrinsic_returns))
    results.update(_summarise("combined_return", combined_returns))
    results.update(_summarise("episode_length", lengths))
    return results


def _format(results: dict) -> str:
    """Human-readable summary: every mean carries its spread, so nobody reads a
    single number off this output and calls it a comparison."""
    lines = [f"arm={results['arm']}  episodes={results['n_episodes']}  "
             f"seed={results['seed']} (per-episode seeds {results['episode_seeds']})"]
    for name, label in (("extrinsic_return", "extrinsic return (SCOREBOARD)"),
                        ("combined_return", "combined return (extrinsic+novelty)"),
                        ("episode_length", "episode length")):
        lines.append(
            f"  {label:36s} mean {results['mean_' + name]:10.3f}  "
            f"std {results['std_' + name]:9.3f}  sem {results['sem_' + name]:9.3f}"
        )
        lines.append(f"  {'per episode':36s} "
                     f"{[round(v, 3) for v in results[name + 's']]}")
    # Printed with the numbers, not buried in a docstring: this caveat is the
    # difference between a result and a press release. See the module docstring's
    # "WHAT THIS HARNESS CANNOT TELL YOU".
    lines.append("  NOTE: one checkpoint, deterministic env -- this spread is "
                 "POLICY-SAMPLING variance only.")
    lines.append("        It is not training-seed variance, which usually dominates. "
                 "Comparing the two arms")
    lines.append("        needs several independently-trained checkpoints per arm, "
                 "compared across those seeds.")
    reset = results.get("state_reset_interval")
    if reset is None:
        lines.append("  NOTE: recurrent state was NEVER reset within an episode, while "
                     "training resets it every")
        lines.append("        rollout_len (128) steps -- both arms are scored outside "
                     "the memory regime they were")
        lines.append("        trained in, and not necessarily by the same amount. "
                     "Re-run with --state-reset-interval 128")
        lines.append("        for the matched-regime counterpart.")
    else:
        lines.append(f"  NOTE: recurrent state was reset every {reset} steps within each "
                     "episode (matched-regime mode);")
        lines.append("        this is NOT the same measurement as a continuous "
                     "playthrough (--state-reset-interval unset).")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", choices=["baseline", "reservoir"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rom", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=3000,
                        help="per-episode step limit (env truncation)")
    parser.add_argument("--novelty-coef", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0,
                        help="base seed; episode i is played under seed+i, so the "
                             "whole run reproduces from this one number")
    parser.add_argument("--state-reset-interval", type=int, default=None,
                        help="reset the model's recurrent state every N steps within an "
                             "episode, mirroring training's rollout boundary. Unset "
                             "(default) = one continuous playthrough, which is NOT the "
                             "regime either arm was trained in -- see this module's "
                             "docstring")
    parser.add_argument("--json", action="store_true",
                        help="print the raw results dict as JSON instead of a summary")
    args = parser.parse_args()
    results = run_evaluation(args.arm, args.checkpoint, args.rom, args.episodes,
                             max_steps_per_episode=args.max_steps,
                             novelty_coef=args.novelty_coef, seed=args.seed,
                             state_reset_interval=args.state_reset_interval)
    if args.json:
        # NaN is not valid JSON (JS's JSON.parse rejects it), and an unmeasurable
        # spread is exactly what null means, so the single-episode case
        # serialises as null rather than as a token half the world cannot read.
        print(json.dumps({k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                          for k, v in results.items()}))
    else:
        print(_format(results))
