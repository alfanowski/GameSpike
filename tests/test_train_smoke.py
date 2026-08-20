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
