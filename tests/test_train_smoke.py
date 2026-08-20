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
import json
import os

import pytest
import torch

import training.train as train_module
from envs.mario_land_env import MarioLandEnv, OBS_DIM
from training.novelty_gate import NoveltyGate
from training.train import (build_model, save_checkpoint, load_checkpoint, run_dir_for,
                            run_training)

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

    def spy_build_model(a, **kwargs):
        model, optimizer = real_build_model(a, **kwargs)
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
    run_dir = tmp_path / "baseline_seed0"
    final = run_dir / f"step_{stats['final_step']}.pt"
    assert final.exists(), f"no final checkpoint; directory holds {list(tmp_path.rglob('*'))}"
    model, optimizer = build_model("baseline")
    assert load_checkpoint(model, optimizer, str(final)) == stats["final_step"]


# --------------------------------------------------------------------------- #
# Seeding (finding I2): reproducibility, and symmetry across the two arms.
# --------------------------------------------------------------------------- #

def _short_run(tmp_path, arm, seed):
    return run_training(arm=arm, rom_path=ROM_PATH, total_steps=8, n_envs=1,
                        rollout_len=8, checkpoint_every=1_000_000,
                        checkpoint_dir=str(tmp_path), seed=seed)


def _trained_weights(run_dir):
    """The state dict a run actually left on disk -- the strongest available
    statement of 'these two runs came out the same'."""
    path = sorted(run_dir.glob("step_*.pt"))[-1]
    return torch.load(path, map_location="cpu", weights_only=True)["model"]


def test_same_seed_reproduces_the_same_training_run(tmp_path):
    """Two runs of the same arm at the same seed must land on bit-identical weights.

    Without an explicit `torch.manual_seed`, the trainable init (and the action
    sampling that follows it) came from whatever global RNG state the process
    happened to be in, so no training run was reproducible at all -- and a §5
    comparison across seeds needs runs that are identified by their seed.
    """
    a = _short_run(tmp_path / "a", "baseline", seed=5)
    torch.manual_seed(999)  # disturb the global RNG between the two runs
    torch.randn(64)
    b = _short_run(tmp_path / "b", "baseline", seed=5)

    assert a["mean_reward"] == b["mean_reward"]
    assert a["grad_norm"] == b["grad_norm"]
    wa = _trained_weights(tmp_path / "a" / "baseline_seed5")
    wb = _trained_weights(tmp_path / "b" / "baseline_seed5")
    assert wa.keys() == wb.keys()
    for name in wa:
        assert torch.equal(wa[name], wb[name]), f"{name} differs across identical seeds"


def test_different_seeds_produce_different_runs(tmp_path):
    """The counterpart: a seed that changes nothing is not a seed.

    The global RNG is put into the SAME state before each run, so `seed` is the only
    thing left that can make them differ. Without that, two runs would differ purely
    from RNG drift and this would stay green even if `seed` were ignored entirely.
    """
    torch.manual_seed(777)
    a = _short_run(tmp_path / "a", "baseline", seed=1)
    torch.manual_seed(777)
    b = _short_run(tmp_path / "b", "baseline", seed=2)
    wa = _trained_weights(tmp_path / "a" / "baseline_seed1")
    wb = _trained_weights(tmp_path / "b" / "baseline_seed2")
    assert any(not torch.equal(wa[name], wb[name]) for name in wa), (
        "two different training seeds produced identical weights"
    )


def test_reservoir_frozen_weights_vary_with_the_training_seed():
    """THE asymmetry test. build_model used to hardcode seed=0 for the reservoir, so
    across 'different' training seeds only the GRU arm's init actually varied while
    the reservoir arm was always the exact same frozen instance -- i.e. the multi-seed
    comparison §5 needs would have been sampling one arm's variation and not the
    other's."""
    a, _ = build_model("reservoir", seed=0)
    b, _ = build_model("reservoir", seed=1)
    assert not torch.equal(a.reservoir.W_in, b.reservoir.W_in), (
        "the frozen reservoir is identical at two different training seeds"
    )
    assert not torch.equal(a.reservoir.tt_core_0, b.reservoir.tt_core_0)
    # ...and the same seed must still reproduce it exactly.
    c, _ = build_model("reservoir", seed=1)
    assert torch.equal(b.reservoir.W_in, c.reservoir.W_in)
    assert torch.equal(b.reservoir.tt_core_0, c.reservoir.tt_core_0)


# --------------------------------------------------------------------------- #
# The §3 frozen-reservoir tripwire, in production (finding I3).
# --------------------------------------------------------------------------- #

def test_save_checkpoint_refuses_to_write_a_mutated_reservoir(tmp_path):
    """Spec §3 asks for a runtime tripwire 'at every checkpoint', not only in tests.

    Corrupt one frozen buffer in place -- exactly what an accidental fine-tune or a
    stray in-place write would do, and something the zero-nn.Parameter half of the
    invariant cannot see -- and save_checkpoint must refuse rather than persist a
    reservoir that is no longer the frozen one the experiment claims.
    """
    model, optimizer = build_model("reservoir")
    good = tmp_path / "good.pt"
    save_checkpoint(model, optimizer, step=0, path=str(good))  # clean model: writes
    assert good.exists()

    with torch.no_grad():
        model.reservoir.W_in[0, 0] += 1e-3
    bad = tmp_path / "bad.pt"
    with pytest.raises(AssertionError, match="bit-identical"):
        save_checkpoint(model, optimizer, step=1, path=str(bad))
    assert not bad.exists(), "a mutated reservoir reached disk anyway"


def test_baseline_checkpoints_are_not_subject_to_the_reservoir_tripwire(tmp_path):
    """The GRU arm has no frozen component; the tripwire must not be applied to it
    (every one of its weights is supposed to move)."""
    model, optimizer = build_model("baseline")
    with torch.no_grad():
        model.gru.weight_ih_l0 += 1.0
    save_checkpoint(model, optimizer, step=0, path=str(tmp_path / "ok.pt"))
    assert (tmp_path / "ok.pt").exists()


def test_a_full_reservoir_run_checkpoints_without_tripping(tmp_path):
    """The tripwire has to survive real training: a false positive here (e.g. from
    snntorch's transient `lif.mem` buffer, which every forward pass mutates) would
    make the reservoir arm unable to checkpoint at all."""
    stats = run_training(arm="reservoir", rom_path=ROM_PATH, total_steps=8, n_envs=1,
                         rollout_len=8, checkpoint_every=1_000_000,
                         checkpoint_dir=str(tmp_path), seed=3)
    assert (tmp_path / "reservoir_seed3" / f"step_{stats['final_step']}.pt").exists()


def test_loading_moves_the_tripwires_reference_onto_the_loaded_weights(tmp_path):
    """The tripwire's reference point has to follow `load_state_dict`.

    A checkpoint's frozen weights are persistent buffers, so loading overwrites the
    ones this process constructed. If the reference copy stayed on the constructed
    weights, any model built at a different seed from the checkpoint's would trip on
    its very next save despite nothing having been mutated -- a tripwire that cries
    wolf gets disabled, which is worse than not having one.
    """
    source, source_opt = build_model("reservoir", seed=8)
    path = tmp_path / "seed8.pt"
    save_checkpoint(source, source_opt, step=0, path=str(path))

    target, target_opt = build_model("reservoir", seed=9)  # different frozen weights
    assert not torch.equal(target.reservoir.W_in, source.reservoir.W_in)
    load_checkpoint(target, target_opt, str(path), expected_arm="reservoir")
    assert torch.equal(target.reservoir.W_in, source.reservoir.W_in)

    save_checkpoint(target, target_opt, step=1, path=str(tmp_path / "again.pt"))
    assert (tmp_path / "again.pt").exists()


def test_resuming_a_reservoir_run_does_not_trip_the_wire(tmp_path):
    """A resumed run's frozen weights are the ones that came off disk, so
    load_checkpoint has to move the tripwire's reference point onto them. Otherwise
    the very first post-resume checkpoint compares against the weights this process
    constructed and then immediately overwrote."""
    first = run_training(arm="reservoir", rom_path=ROM_PATH, total_steps=8, n_envs=1,
                         rollout_len=8, checkpoint_every=1_000_000,
                         checkpoint_dir=str(tmp_path), seed=4)
    resume_from = os.path.join(first["run_dir"], f"step_{first['final_step']}.pt")
    second = run_training(arm="reservoir", rom_path=ROM_PATH, total_steps=16, n_envs=1,
                          rollout_len=8, checkpoint_every=1_000_000,
                          checkpoint_dir=str(tmp_path), seed=4, resume_from=resume_from)
    assert second["final_step"] == 16


# --------------------------------------------------------------------------- #
# Checkpoint collision and arm mislabelling (finding I5).
# --------------------------------------------------------------------------- #

def test_the_two_arms_do_not_overwrite_each_others_checkpoints(tmp_path):
    """With a shared default --checkpoint-dir, both arms used to write
    `step_{step}.pt`, so running baseline then reservoir silently destroyed the
    baseline's checkpoints -- and the destroyed run is exactly the control §5 needs."""
    baseline = run_training(arm="baseline", rom_path=ROM_PATH, total_steps=8, n_envs=1,
                            rollout_len=8, checkpoint_every=1_000_000,
                            checkpoint_dir=str(tmp_path))
    reservoir = run_training(arm="reservoir", rom_path=ROM_PATH, total_steps=8, n_envs=1,
                             rollout_len=8, checkpoint_every=1_000_000,
                             checkpoint_dir=str(tmp_path))
    assert baseline["run_dir"] != reservoir["run_dir"]
    for stats in (baseline, reservoir):
        path = os.path.join(stats["run_dir"], f"step_{stats['final_step']}.pt")
        assert os.path.exists(path), f"{stats['arm']}'s checkpoint is gone: {path}"
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        assert ckpt["arm"] == stats["arm"], "checkpoint does not self-identify its arm"


def test_seeds_of_the_same_arm_do_not_overwrite_each_other(tmp_path):
    """§5 needs several independently-trained checkpoints PER ARM sitting on disk at
    once, so the seed has to be in the path too, not just the arm."""
    a = _short_run(tmp_path, "baseline", seed=0)
    b = _short_run(tmp_path, "baseline", seed=1)
    assert a["run_dir"] != b["run_dir"]
    assert run_dir_for(str(tmp_path), "baseline", 1) == b["run_dir"]
    for stats in (a, b):
        path = os.path.join(stats["run_dir"], f"step_{stats['final_step']}.pt")
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        assert ckpt["seed"] == stats["seed"]


def test_loading_a_checkpoint_into_the_wrong_arm_is_rejected_clearly(tmp_path):
    """A shape-mismatch traceback several frames inside torch does not tell anyone
    they mixed up --arm; this does."""
    model, optimizer = build_model("reservoir")
    path = tmp_path / "reservoir.pt"
    save_checkpoint(model, optimizer, step=0, path=str(path))

    wrong, wrong_opt = build_model("baseline")
    with pytest.raises(ValueError, match="arm mismatch"):
        load_checkpoint(wrong, wrong_opt, str(path), expected_arm="baseline")
    # ...and the matching arm still loads.
    right, right_opt = build_model("reservoir")
    assert load_checkpoint(right, right_opt, str(path), expected_arm="reservoir") == 0


def test_resuming_under_a_different_seed_is_rejected(tmp_path):
    """Resuming a seed-N run while labelling the process seed M would write
    checkpoints labelled M whose frozen reservoir is in fact N's."""
    stats = _short_run(tmp_path, "baseline", seed=11)
    resume_from = os.path.join(stats["run_dir"], f"step_{stats['final_step']}.pt")
    with pytest.raises(ValueError, match="seed mismatch"):
        run_training(arm="baseline", rom_path=ROM_PATH, total_steps=16, n_envs=1,
                     rollout_len=8, checkpoint_every=1_000_000,
                     checkpoint_dir=str(tmp_path), seed=12, resume_from=resume_from)


# --------------------------------------------------------------------------- #
# Per-update logging (finding I9).
# --------------------------------------------------------------------------- #

def test_every_update_is_logged_not_just_the_last_one(tmp_path):
    """`stats` is REASSIGNED per update, so the returned dict only ever describes
    the final one. A 100k-step run has ~780 updates; without this log there is no
    learning curve, no way to spot divergence mid-run, and nothing to plot."""
    stats = run_training(arm="baseline", rom_path=ROM_PATH, total_steps=32, n_envs=1,
                         rollout_len=8, checkpoint_every=1_000_000,
                         checkpoint_dir=str(tmp_path), seed=2)
    log_path = stats["log_path"]
    assert os.path.exists(log_path), f"no training log at {log_path}"
    records = [json.loads(line) for line in open(log_path, encoding="utf-8")]

    assert len(records) == stats["updates"] == 4, (
        f"{len(records)} log lines for {stats['updates']} updates"
    )
    assert [r["step"] for r in records] == [8, 16, 24, 32]
    assert [r["update"] for r in records] == [1, 2, 3, 4]
    for record in records:
        assert record["arm"] == "baseline" and record["seed"] == 2
        # Everything needed to plot a learning curve or spot a divergence.
        for key in ("mean_reward", "mean_extrinsic_reward", "policy_loss",
                    "value_loss", "entropy", "total_loss", "grad_norm"):
            assert isinstance(record[key], float), f"{key} missing from the log record"
    # The last log line and the returned summary describe the same update.
    assert records[-1]["grad_norm"] == stats["grad_norm"]
    assert records[-1]["step"] == stats["final_step"]


def test_the_log_is_appended_incrementally_not_written_at_the_end(tmp_path, monkeypatch):
    """A log that only materialises when the run finishes is useless for the case it
    exists for: watching (or post-mortem-ing) a long run that is still going, or
    that died."""
    seen = []
    real_collect = train_module.collect_rollout_with_model
    log_path = os.path.join(run_dir_for(str(tmp_path), "baseline", 0), "train_log.jsonl")

    def spy_collect(*args, **kwargs):
        # Sampled at the START of each rollout, i.e. strictly before the run ends.
        seen.append(sum(1 for _ in open(log_path, encoding="utf-8"))
                    if os.path.exists(log_path) else 0)
        return real_collect(*args, **kwargs)

    monkeypatch.setattr(train_module, "collect_rollout_with_model", spy_collect)
    run_training(arm="baseline", rom_path=ROM_PATH, total_steps=24, n_envs=1,
                 rollout_len=8, checkpoint_every=1_000_000, checkpoint_dir=str(tmp_path))

    assert seen == [0, 1, 2], (
        f"log line counts observed mid-run were {seen}; the log is not being flushed "
        "per update"
    )
