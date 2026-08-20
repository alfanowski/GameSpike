"""Evaluation-harness tests, against a real ROM.

The first test is the brief's contract check. The rest exist because the brief's
own reference implementation had two defects that a keys-only test passes with
flying colours:

  * it scored curiosity on the policy's LOGITS (dim=10) instead of the
    environment's OBSERVATION (dim=12) -- the same bug Task 11 already fixed in
    the training loop, which changes the reward FUNCTION per arm rather than a
    reported metric (`test_novelty_is_scored_on_observations_not_logits`);
  * it picked actions with `argmax`. This emulator and boot sequence are fully
    deterministic and `reset()` returns to a bit-identical state, so a greedy
    policy replays the SAME episode n_episodes times: `n_episodes > 1` would be
    pure theatre and the reported "mean" would be one number with no sample
    behind it (`test_episodes_within_one_run_are_genuinely_different`).

The reproducibility tests pin the other half of that fix: sampling only buys an
honest spread if a stated seed still reproduces the exact reported numbers.
"""
import os

import pytest
import torch

import training.evaluate as evaluate_module
from envs.mario_land_env import MarioLandEnv, OBS_DIM
from training.evaluate import run_evaluation
from training.novelty_gate import NoveltyGate
from training.train import build_model, save_checkpoint

ROM_PATH = os.environ.get("MARIO_LAND_ROM_PATH")
pytestmark = pytest.mark.skipif(
    not ROM_PATH or not os.path.exists(ROM_PATH),
    reason="MARIO_LAND_ROM_PATH not set or file missing; set it to your own legally-dumped ROM",
)

# Long enough for random play to produce genuinely different trajectories, short
# enough that a test run costs milliseconds rather than minutes.
EVAL_STEPS = 40


def _checkpoint(tmp_path, arm="baseline"):
    """An untrained checkpoint of `arm`, saved to disk -- exactly what the harness
    consumes in production, so the load path is exercised rather than bypassed."""
    model, optimizer = build_model(arm)
    path = tmp_path / f"{arm}.pt"
    save_checkpoint(model, optimizer, step=0, path=str(path))
    return str(path)


class _SpyEnv(MarioLandEnv):
    """Records the action sequence of every episode, split on reset()."""

    episodes = []

    def reset(self, **kwargs):
        type(self).episodes.append([])
        return super().reset(**kwargs)

    def step(self, action):
        type(self).episodes[-1].append(action)
        return super().step(action)


def test_evaluation_returns_expected_keys(tmp_path):
    results = run_evaluation(arm="baseline", checkpoint_path=_checkpoint(tmp_path),
                             rom_path=ROM_PATH, n_episodes=2,
                             max_steps_per_episode=EVAL_STEPS)
    for key in ("mean_extrinsic_return", "mean_combined_return", "mean_episode_length"):
        assert key in results
        assert isinstance(results[key], float)


def test_results_report_spread_not_just_a_mean(tmp_path):
    """A bare mean cannot support a baseline-vs-reservoir claim: without a spread
    there is no way to tell a real gap from sampling noise."""
    results = run_evaluation(arm="baseline", checkpoint_path=_checkpoint(tmp_path),
                             rom_path=ROM_PATH, n_episodes=3,
                             max_steps_per_episode=EVAL_STEPS)
    for key in ("std_extrinsic_return", "std_combined_return", "std_episode_length",
                "sem_extrinsic_return"):
        assert key in results, f"no spread reported for {key}"
        assert isinstance(results[key], float)

    # The raw per-episode numbers must survive into the results, so anyone can
    # recompute the statistics (or run a different test) without re-running the
    # emulator.
    for key in ("extrinsic_returns", "combined_returns", "episode_lengths"):
        assert len(results[key]) == 3
    assert results["mean_extrinsic_return"] == pytest.approx(
        sum(results["extrinsic_returns"]) / 3)
    assert results["std_extrinsic_return"] > 0.0, (
        "zero spread across three sampled episodes -- the episodes are identical"
    )


def test_single_episode_reports_undefined_spread_not_zero(tmp_path):
    """One episode has no spread to estimate. Reporting 0.0 would read as
    'measured, and it was zero'; NaN says 'not measurable', which is the truth."""
    results = run_evaluation(arm="baseline", checkpoint_path=_checkpoint(tmp_path),
                             rom_path=ROM_PATH, n_episodes=1,
                             max_steps_per_episode=EVAL_STEPS)
    assert results["std_extrinsic_return"] != results["std_extrinsic_return"]  # NaN
    assert results["sem_extrinsic_return"] != results["sem_extrinsic_return"]


def test_episodes_within_one_run_are_genuinely_different(tmp_path, monkeypatch):
    """THE test for defect 2. The env is deterministic and reset() returns to a
    bit-identical state, so with greedy action selection every episode would be
    the same episode: same actions, same trajectory, same return. Actions must be
    SAMPLED from the policy, under a per-episode seed."""
    _SpyEnv.episodes = []
    monkeypatch.setattr(evaluate_module, "MarioLandEnv", _SpyEnv)

    results = run_evaluation(arm="baseline", checkpoint_path=_checkpoint(tmp_path),
                             rom_path=ROM_PATH, n_episodes=3,
                             max_steps_per_episode=EVAL_STEPS, seed=0)

    traces = _SpyEnv.episodes
    assert len(traces) == 3, f"{len(traces)} episodes played, expected 3"
    assert len({tuple(t) for t in traces}) == 3, (
        "episodes repeated the same action sequence -- action selection is not "
        "sampling from the policy (greedy argmax against a deterministic env)"
    )
    assert len(set(results["extrinsic_returns"])) > 1, (
        f"all episodes returned the same score: {results['extrinsic_returns']}"
    )


def test_same_seed_reproduces_the_same_numbers(tmp_path):
    """Reproducibility is what makes a reported result checkable by someone else.

    The global torch RNG is deliberately disturbed between the two runs: the
    harness must own its randomness (a private generator), so that evaluating
    after some unrelated torch sampling still reproduces the published numbers.
    """
    ckpt = _checkpoint(tmp_path)
    kwargs = dict(arm="baseline", checkpoint_path=ckpt, rom_path=ROM_PATH,
                  n_episodes=2, max_steps_per_episode=EVAL_STEPS, seed=123)
    first = run_evaluation(**kwargs)

    torch.manual_seed(999)
    torch.randn(64)

    second = run_evaluation(**kwargs)
    assert first["extrinsic_returns"] == second["extrinsic_returns"]
    assert first["combined_returns"] == second["combined_returns"]
    assert first["episode_lengths"] == second["episode_lengths"]


def test_different_seed_changes_the_episodes(tmp_path):
    """The counterpart to the test above: if the seed did not actually drive
    action sampling, seeding would be reproducible and meaningless at once."""
    ckpt = _checkpoint(tmp_path)
    kwargs = dict(arm="baseline", checkpoint_path=ckpt, rom_path=ROM_PATH,
                  n_episodes=2, max_steps_per_episode=EVAL_STEPS)
    a = run_evaluation(seed=0, **kwargs)
    b = run_evaluation(seed=1000, **kwargs)
    assert a["extrinsic_returns"] != b["extrinsic_returns"]


def test_novelty_is_scored_on_observations_not_logits(tmp_path, monkeypatch):
    """THE test for defect 1. Curiosity is scored over the 12-dim observation --
    the game state the agent reached -- not the 10-dim logits vector. Same
    invariant tests/test_train_smoke.py pins for the training loop."""
    dims, pushed_shapes = [], []

    class SpyGate(NoveltyGate):
        def __init__(self, dim, **kwargs):
            dims.append(dim)
            super().__init__(dim=dim, **kwargs)

        def push(self, state_vec):
            pushed_shapes.append(tuple(state_vec.shape))
            super().push(state_vec)

    monkeypatch.setattr(evaluate_module, "NoveltyGate", SpyGate)
    run_evaluation(arm="baseline", checkpoint_path=_checkpoint(tmp_path),
                   rom_path=ROM_PATH, n_episodes=2, max_steps_per_episode=EVAL_STEPS)

    assert set(dims) == {OBS_DIM}, f"novelty gate built over dim={dims}, expected {OBS_DIM}"
    assert pushed_shapes, "nothing was ever pushed into the novelty gate"
    assert set(pushed_shapes) == {(OBS_DIM,)}, (
        f"novelty vectors were {set(pushed_shapes)}, not observations"
    )


def test_novelty_buffer_does_not_leak_between_episodes(tmp_path, monkeypatch):
    """Episodes are the unit of comparison, so each must start from the same
    novelty state. A gate shared across episodes makes episode 1 systematically
    more 'novel' than episode 3 purely by ordering, which shows up as spread in
    the combined return that is an artefact, not a measurement."""
    gates = []
    real_gate = evaluate_module.NoveltyGate

    def spy_gate(*args, **kwargs):
        gate = real_gate(*args, **kwargs)
        gates.append(gate)
        return gate

    monkeypatch.setattr(evaluate_module, "NoveltyGate", spy_gate)
    run_evaluation(arm="baseline", checkpoint_path=_checkpoint(tmp_path),
                   rom_path=ROM_PATH, n_episodes=3, max_steps_per_episode=EVAL_STEPS)

    assert len(gates) == 3, f"{len(gates)} novelty gates built for 3 episodes"


def test_reservoir_arm_evaluates_too(tmp_path):
    """The harness exists to compare two arms; a baseline-only harness answers
    nothing. Both arms must run through the identical code path."""
    results = run_evaluation(arm="reservoir", checkpoint_path=_checkpoint(tmp_path, "reservoir"),
                             rom_path=ROM_PATH, n_episodes=2,
                             max_steps_per_episode=EVAL_STEPS, seed=7)
    assert results["arm"] == "reservoir"
    assert isinstance(results["mean_extrinsic_return"], float)
    assert len(results["extrinsic_returns"]) == 2


def test_combined_return_contains_the_novelty_subsidy(tmp_path):
    """`mean_extrinsic_return` is the scoreboard; `mean_combined_return` is the
    reward the loss actually optimised. Reporting the same number twice would
    quietly put the exploration subsidy on the scoreboard."""
    results = run_evaluation(arm="baseline", checkpoint_path=_checkpoint(tmp_path),
                             rom_path=ROM_PATH, n_episodes=2,
                             max_steps_per_episode=EVAL_STEPS, novelty_coef=0.05)
    assert results["mean_combined_return"] != results["mean_extrinsic_return"]

    zero_coef = run_evaluation(arm="baseline", checkpoint_path=_checkpoint(tmp_path),
                               rom_path=ROM_PATH, n_episodes=2,
                               max_steps_per_episode=EVAL_STEPS, novelty_coef=0.0)
    assert zero_coef["combined_returns"] == zero_coef["extrinsic_returns"]
