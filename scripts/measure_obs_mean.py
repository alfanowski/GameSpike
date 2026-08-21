"""Phase 2a's mixture OBS_MEAN measurement tool.

`envs.mario_land_env.OBS_MEAN` is a per-slot mean of the real observation
distribution, used by `--embed-init-mode centered` to remove the observation's DC
component before it ever reaches the frozen reservoir (see that constant's own
docstring in envs/mario_land_env.py, and training/train.py's module docstring for
why the DC component matters at all). It was measured on 1-1 only, over 3,000
trained-policy + 3,000 uniform-random steps, pooled.

A run spanning {1-1, 2-1} needs the per-slot mean of the POOLED observation
distribution across BOTH tasks -- not either task's mean alone, which would over-
or under-correct depending which task a given rollout happens to be running.
`OBS_MEAN_PHASE2A` in envs/mario_land_env.py is that mixture mean, and this script
is how it was produced (docs/DESIGN_ROADMAP_PHASE2.md §9 item 3).

METHODOLOGY, and where it deliberately departs from the original measurement:
uniform-random policy ONLY, not trained+random pooled. There is no trained
checkpoint for 2-1 (there cannot be yet -- this script's own output is what a
Phase 2a training run's centered-init depends on, so it cannot depend on a Phase 2a
training run first), and a uniform-random policy is a reproducible, defensible
stand-in: it is exactly the policy the original measurement's "random" half used,
just without the "trained" half alongside it. `--steps-per-task` defaults to 3,000,
matching the step count of that random half, applied IDENTICALLY to every task
(not proportioned by anything) so the mixture is not implicitly weighted toward
whichever task happens to run more steps.

Usage:
    python -m scripts.measure_obs_mean --rom "/path/to/Super Mario Land (World).gb" \\
        --tasks 1-1,2-1 --steps-per-task 3000

Costs a few seconds of emulation per 1,000 steps requested (see this project's own
measured env-steps/s figures), not hours -- like scripts/phase2_viability_probe.py,
this is a diagnostic tool and must never grow into a training script.
"""
import argparse
import sys

import numpy as np

from envs.mario_land_env import OBS_DIM, MarioLandEnv

DEFAULT_STEPS_PER_TASK = 3000


def parse_task(spec: str) -> tuple:
    """'1-1' -> (1, 1). Kept as an independent, minimal parser rather than
    importing training.train.parse_task: this script has to run standalone, without
    pulling in torch/models just to parse two integers, and it deliberately does
    NOT restrict itself to training/train.py's TASKS allow-list -- measuring the
    mixture mean for a level Phase 2a hasn't adopted yet (e.g. while scoping a
    future task set) is a legitimate use of this tool even before that level is
    wired into training/evaluate.py's own --task flag.
    """
    parts = spec.split("-")
    if len(parts) != 2:
        raise ValueError(f"bad task spec {spec!r}, expected 'W-L' e.g. '1-1'")
    try:
        world, level = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(f"bad task spec {spec!r}, expected two integers separated by '-'")
    return (world, level)


def collect_task_observations(rom_path: str, world_level: tuple, n_steps: int, seed: int):
    """`n_steps` observations from a uniform-random policy on `world_level`.

    A fresh env per task, mirroring production (`MarioLandEnv(world_level=...)` is
    one task's env). Episodes that terminate mid-collection are reset immediately
    rather than truncating the sample early -- 2-1 in particular dies fast under
    undirected random play, so without this a naive single-episode collection could
    fall well short of `n_steps`.
    """
    rng = np.random.default_rng(seed)
    env = MarioLandEnv(rom_path=rom_path, world_level=world_level, verify_control=False)
    observations = []
    try:
        obs, _ = env.reset()
        observations.append(obs)
        while len(observations) < n_steps:
            action = int(rng.integers(0, env.action_space.n))
            obs, _reward, terminated, truncated, _info = env.step(action)
            observations.append(obs)
            if terminated or truncated:
                obs, _ = env.reset()
    finally:
        env.close()
    return np.stack(observations[:n_steps], axis=0)


def format_constant(mean: np.ndarray, name: str = "OBS_MEAN_PHASE2A") -> str:
    """Renders `mean` as a Python tuple literal in the same 6-decimal style as
    envs/mario_land_env.py's own OBS_MEAN, so the printed output can be pasted
    straight into that module."""
    lines = [f"{name} = ("]
    for i in range(0, len(mean), 6):
        row = ", ".join(f"{v:.6f}" for v in mean[i:i + 6])
        lines.append(f"    {row},")
    lines.append(")")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", required=True, help="path to the Game Boy ROM")
    parser.add_argument("--tasks", required=True,
                        help="comma-separated W-L specs, e.g. '1-1,2-1'")
    parser.add_argument("--steps-per-task", type=int, default=DEFAULT_STEPS_PER_TASK,
                        help=f"observations collected per task, pooled before "
                             f"averaging (default: {DEFAULT_STEPS_PER_TASK}, matching "
                             f"the original OBS_MEAN measurement's random-policy half)")
    parser.add_argument("--seed", type=int, default=0,
                        help="base seed; task i is collected under seed+i, so the "
                             "whole measurement reproduces from this one number")
    args = parser.parse_args(argv)

    try:
        tasks = [parse_task(t.strip()) for t in args.tasks.split(",") if t.strip()]
    except ValueError as exc:
        parser.error(str(exc))
    if not tasks:
        parser.error("--tasks produced no tasks")
    if args.steps_per_task < 1:
        parser.error(f"--steps-per-task must be >= 1, got {args.steps_per_task}")

    per_task_obs = []
    for i, task in enumerate(tasks):
        obs = collect_task_observations(args.rom, task, args.steps_per_task, args.seed + i)
        per_task_obs.append(obs)
        print(f"task {task[0]}-{task[1]}: collected {obs.shape[0]} steps "
             f"(seed {args.seed + i})", file=sys.stderr)

    pooled = np.concatenate(per_task_obs, axis=0)
    assert pooled.shape == (args.steps_per_task * len(tasks), OBS_DIM)
    mean = pooled.mean(axis=0).astype(np.float64)

    print(format_constant(mean))
    print(f"# measured over {pooled.shape[0]} steps "
         f"({args.steps_per_task} per task x {len(tasks)} tasks), "
         f"tasks={['-'.join(map(str, t)) for t in tasks]}, seed={args.seed}",
         file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
