"""End-to-end training-loop smoke tests, against a real ROM.

The third test is the one that matters. "Does not crash" is a weak claim for a
training loop: a loop that collects rollouts, computes advantages and then throws
them away -- never calling backward() or optimizer.step() -- passes a
does-not-crash test perfectly while never changing a single weight. So the third
test snapshots every trainable parameter before training and asserts at least one
of them actually moved, on BOTH arms, and additionally asserts that the
reservoir arm's frozen weights survived a REAL optimizer step untouched (not just
the isolated forward-pass invariant tests/test_policy_value_reservoir.py covers).
"""
import os

import pytest
import torch

import training.train as train_module
from envs.mario_land_env import MarioLandEnv, OBS_DIM
from training.novelty_gate import NoveltyGate
from training.train import build_model, save_checkpoint, load_checkpoint, run_training

ROM_PATH = os.environ.get("MARIO_LAND_ROM_PATH")
pytestmark = pytest.mark.skipif(
    not ROM_PATH or not os.path.exists(ROM_PATH),
    reason="MARIO_LAND_ROM_PATH not set or file missing; set it to your own legally-dumped ROM",
)


def test_short_training_run_does_not_crash(tmp_path):
    stats = run_training(arm="baseline", rom_path=ROM_PATH, total_steps=64,
                         n_envs=2, rollout_len=16, checkpoint_every=1_000_000,
                         checkpoint_dir=str(tmp_path))
    assert "mean_reward" in stats
    assert isinstance(stats["mean_reward"], float)


def test_checkpoint_roundtrip(tmp_path):
    model, optimizer = build_model("baseline")
    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(model, optimizer, step=42, path=str(ckpt_path))
    model2, optimizer2 = build_model("baseline")
    restored_step = load_checkpoint(model2, optimizer2, path=str(ckpt_path))
    assert restored_step == 42
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)


@pytest.mark.parametrize("arm", ["baseline", "reservoir"])
def test_training_actually_updates_parameters(tmp_path, monkeypatch, arm):
    """run_training must perform a REAL gradient update, not a no-op pass.

    run_training builds its own model internally, so the pre-training snapshot is
    taken by spying on build_model rather than by changing the production
    signature to hand the model back out.
    """
    captured = {}
    real_build_model = train_module.build_model

    def spy_build_model(a):
        model, optimizer = real_build_model(a)
        captured["model"] = model
        captured["params_before"] = {
            name: p.detach().clone()
            for name, p in model.named_parameters() if p.requires_grad
        }
        if a == "reservoir":
            captured["reservoir_buffers_before"] = {
                name: b.detach().clone()
                for name, b in model.reservoir.named_buffers()
            }
        return model, optimizer

    monkeypatch.setattr(train_module, "build_model", spy_build_model)

    stats = run_training(arm=arm, rom_path=ROM_PATH, total_steps=8, n_envs=1,
                         rollout_len=8, checkpoint_every=1_000_000,
                         checkpoint_dir=str(tmp_path))

    assert stats["updates"] >= 1, "no PPO update ran at all"
    assert stats["grad_norm"] > 0.0, "gradient was exactly zero -- nothing was learned from"

    model = captured["model"]
    params_before = captured["params_before"]
    assert params_before, "model reported no trainable parameters"
    changed = [
        name for name, p in model.named_parameters()
        if p.requires_grad and not torch.equal(params_before[name], p.detach())
    ]
    assert changed, (
        f"[{arm}] no trainable parameter changed after run_training -- the loop "
        f"collected rollouts but never applied a gradient step"
    )

    if arm == "reservoir":
        # The frozen-reservoir invariant, checked through a real optimizer step
        # rather than in isolation. `lif.mem` is excluded deliberately: it is
        # snntorch's transient membrane-state slot, mutated by ANY forward pass
        # (including the pure-inference ones during collection) and never read by
        # this codebase, which threads `mem` explicitly through step(). It is
        # state, not a weight. Everything else under the reservoir -- W_in, the
        # TT cores, the LIF constants -- must be bit-identical.
        buffers_before = captured["reservoir_buffers_before"]
        for name, buf in model.reservoir.named_buffers():
            if name == "lif.mem":
                continue
            assert torch.equal(buffers_before[name], buf), (
                f"frozen reservoir buffer {name} changed during training"
            )
        assert torch.equal(buffers_before["W_in"], model.reservoir.W_in)
        assert list(model.reservoir.parameters()) == [], (
            "reservoir grew an nn.Parameter -- it would be trained by the optimizer"
        )
        # The readout is the arm's actual trainable capacity; if only the
        # embedding moved, something is wrong with the path through the reservoir.
        assert any(name.startswith("readout.") for name in changed), (
            f"no readout parameter changed; only {changed} moved"
        )


def test_env_persists_across_rollouts(tmp_path, monkeypatch):
    """One env for the whole run, advancing continuously across rollout boundaries.

    This is the test that would have caught the original design: building a fresh
    env per rollout still passes every other test in this file (gradients flow,
    parameters move) while the agent silently re-plays the opening seconds of
    world 1-1 forever, never seeing the rest of the level.
    """
    built = []

    class SpyEnv(MarioLandEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            built.append(self)
            self.reset_calls = 0
            self.observed_step_counts = []

        def reset(self, **kwargs):
            self.reset_calls += 1
            return super().reset(**kwargs)

        def step(self, action):
            out = super().step(action)
            self.observed_step_counts.append(self._step_count)
            return out

    monkeypatch.setattr(train_module, "MarioLandEnv", SpyEnv)

    run_training(arm="baseline", rom_path=ROM_PATH, total_steps=32, n_envs=1,
                 rollout_len=8, checkpoint_every=1_000_000, checkpoint_dir=str(tmp_path))

    assert len(built) == 1, f"{len(built)} envs constructed; the env must outlive the rollouts"
    env = built[0]
    # Only the caller's own initial reset. 32 steps from the level start cannot
    # reach a game over (the episode survives individual deaths -- lives start at
    # 2+) nor the 3000-step truncation, so any further reset means a rollout
    # restarted the game.
    assert env.reset_calls == 1, f"env was reset {env.reset_calls} times, expected once"
    # The env's own step counter must run 1..32 unbroken THROUGH the four rollout
    # boundaries, rather than restarting at 1 every 8 steps.
    assert env.observed_step_counts == list(range(1, 33)), (
        f"env step count did not advance monotonically across rollouts: "
        f"{env.observed_step_counts}"
    )
    assert env.pyboy is None, "env was not closed at the end of the run"


def test_novelty_gate_scores_observations_not_logits(tmp_path, monkeypatch):
    """The curiosity signal must be built over the 12-dim observation, not the
    10-dim logits vector: it is part of the reward FUNCTION each arm optimises."""
    dims = []
    pushed_shapes = []

    class SpyGate(NoveltyGate):
        def __init__(self, dim, **kwargs):
            dims.append(dim)
            super().__init__(dim=dim, **kwargs)

        def push(self, state_vec):
            pushed_shapes.append(tuple(state_vec.shape))
            super().push(state_vec)

    monkeypatch.setattr(train_module, "NoveltyGate", SpyGate)
    run_training(arm="baseline", rom_path=ROM_PATH, total_steps=8, n_envs=1,
                 rollout_len=8, checkpoint_every=1_000_000, checkpoint_dir=str(tmp_path))

    assert dims == [OBS_DIM], f"novelty gate built over dim={dims}, expected [{OBS_DIM}]"
    assert pushed_shapes, "nothing was ever pushed into the novelty gate"
    assert set(pushed_shapes) == {(OBS_DIM,)}, (
        f"novelty vectors were {set(pushed_shapes)}, not observations"
    )


def test_final_checkpoint_is_saved_even_when_the_cadence_does_not_land(tmp_path):
    """checkpoint_every never fires here, so only an unconditional final save can
    put the run's actual trained weights on disk."""
    stats = run_training(arm="baseline", rom_path=ROM_PATH, total_steps=16, n_envs=1,
                         rollout_len=8, checkpoint_every=1_000_000,
                         checkpoint_dir=str(tmp_path))
    final = tmp_path / f"step_{stats['final_step']}.pt"
    assert final.exists(), f"no final checkpoint; directory holds {list(tmp_path.iterdir())}"
    model, optimizer = build_model("baseline")
    assert load_checkpoint(model, optimizer, str(final)) == stats["final_step"]
