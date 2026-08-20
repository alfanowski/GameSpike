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

# Same buffer geometry as the training loop's gate, so the combined return
# reported here is measured on the same scale as the one that was optimised.
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
                   seed: int = 0) -> dict:
    """Play `n_episodes` with `arm`'s checkpoint and report per-episode statistics.

    Episode i is driven by a private generator seeded `seed + i`, so the run is
    reproducible from `seed` alone while the episodes still differ from one
    another. Returns the keys documented in `_summarise` for `extrinsic_return`,
    `combined_return` and `episode_length`, plus `arm`, `n_episodes`, `seed` and
    `episode_seeds`.
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")

    model, optimizer = build_model(arm)
    load_checkpoint(model, optimizer, checkpoint_path)
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
                    done = bool(terminated) or bool(truncated)
                extrinsic_returns.append(extrinsic_total)
                combined_returns.append(combined_total)
                lengths.append(float(steps))
    finally:
        env.close()

    results = {"arm": arm, "n_episodes": n_episodes, "seed": seed,
               "episode_seeds": episode_seeds}
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
    parser.add_argument("--json", action="store_true",
                        help="print the raw results dict as JSON instead of a summary")
    args = parser.parse_args()
    results = run_evaluation(args.arm, args.checkpoint, args.rom, args.episodes,
                             max_steps_per_episode=args.max_steps,
                             novelty_coef=args.novelty_coef, seed=args.seed)
    if args.json:
        # NaN is not valid JSON (JS's JSON.parse rejects it), and an unmeasurable
        # spread is exactly what null means, so the single-episode case
        # serialises as null rather than as a token half the world cannot read.
        print(json.dumps({k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                          for k, v in results.items()}))
    else:
        print(_format(results))
