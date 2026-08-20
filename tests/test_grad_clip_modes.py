"""Gradient-clipping modes, run-tag isolation, and checkpoint backward compatibility.

WHY THIS FILE EXISTS. A diagnostic on `checkpoints/reservoir_seed0/step_500480.pt`
measured that the reservoir arm's trainable `embedding` (416 params, 0.3% of the
trainable budget) carries 100.0000% of the global gradient norm, because the
replay backpropagates through 128 sequential frozen-reservoir steps and the
gradient reaching the embedding grows ~1.22 per step (2.171 at L=1 -> 1.258e9 at
L=128) while the readout's own gradient grows only ~sqrt(L) (1.5 -> 8.9) and the
baseline GRU arm stays flat (29.97 / 51.71 / 48.18).

A SINGLE `clip_grad_norm_` over the whole parameter list then computes a clip
coefficient of 3.976e-10 from the exploding 0.3% and applies it to the 99.7% that
is not exploding, taking the readout's post-clip gradient norm to 3.52e-09. Adam
does not rescue it: Adam is invariant to a CONSTANT gradient rescaling but not to
a time-varying one, and the clip coefficient's max/median ratio over 1000 updates
is 2.63e5, so the readout's median |m_hat|/sqrt(v_hat) collapses to 7.475e-04
against the baseline's 1.346e-01. Measured counterfactual on one step with the
same gradients and the same restored Adam state: per-group clipping raises the
readout's median ||dp||/||p|| from 1.9034e-05 to 6.4186e-03, a factor of 337.
(Raising Adam's eps 1e-8 -> 1e-12 gives 1.11x, so the eps floor is not the
mechanism.)

`test_per_group_rescues_the_readout_from_the_exploding_embedding` is the
regression test for exactly that. The rest of the file guards the two things the
fix must not break: the default path must stay BIT-IDENTICAL (20 completed runs /
200 checkpoints have to remain exactly reproducible), and the 200 existing
checkpoints -- which predate both new keys -- must keep loading.
"""
import json
import os
import pathlib

import pytest
import torch
import torch.nn as nn

from training.train import (GRAD_CLIP_MODES, MAX_GRAD_NORM, apply_grad_clipping,
                            build_model, group_trainable_parameters, load_checkpoint,
                            run_dir_for, run_training)

ROM_PATH = os.environ.get("MARIO_LAND_ROM_PATH")
requires_rom = pytest.mark.skipif(
    not ROM_PATH or not os.path.exists(ROM_PATH),
    reason="MARIO_LAND_ROM_PATH not set or file missing; set it to your own legally-dumped ROM",
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Helpers. Deterministic and emulator-free: everything here is about what the
# clipping rule does to a given set of gradients, not about the game.
# --------------------------------------------------------------------------- #

def _synthetic_model():
    """A two-group stand-in with the SAME group shape as the real reservoir arm:
    one small `embedding` group and one large `readout` group. Named submodules,
    because `group_trainable_parameters` buckets on the first dot-separated
    component of `named_parameters()`."""
    torch.manual_seed(0)
    return nn.ModuleDict({
        "embedding": nn.Linear(12, 32),
        "readout": nn.Sequential(nn.Linear(32, 64), nn.Tanh(), nn.Linear(64, 10)),
    })


def _fill_grads(model, scale_by_group=None, seed=1234):
    """Give every trainable parameter a deterministic gradient, optionally scaled
    per group. Returns {param_name: grad_clone} so the exact same gradients can be
    replayed into a second model (or the same one) for an apples-to-apples
    comparison of the two clipping rules."""
    scale_by_group = scale_by_group or {}
    generator = torch.Generator().manual_seed(seed)
    stored = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        grad = torch.randn(param.shape, generator=generator, dtype=param.dtype)
        grad = grad * scale_by_group.get(name.split(".")[0], 1.0)
        param.grad = grad
        stored[name] = grad.clone()
    return stored


def _restore_grads(model, stored):
    for name, param in model.named_parameters():
        if name in stored:
            param.grad = stored[name].clone()


def _group_grad_norms(model):
    """Post-hoc {group: L2 norm of that group's gradients}, computed from whatever
    is currently in `.grad` -- i.e. the observable the whole bug is about."""
    return {
        group: float(torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(p.grad, 2.0) for p in params]), 2.0).item())
        for group, params in group_trainable_parameters(model).items()
    }


# --------------------------------------------------------------------------- #
# 1. The default path must not move. This is the test that protects 20 runs.
# --------------------------------------------------------------------------- #

def test_global_mode_is_bit_identical_to_the_historical_call():
    """`grad_clip_mode="global"` must be the ORIGINAL `clip_grad_norm_` over the
    trainable parameter list, to the bit.

    Not "close", not "equivalent up to float error": 200 checkpoints on disk and a
    results write-up depend on those runs being exactly reproducible, so the
    comparison is `torch.equal`. If this ever fails, the new flag has silently
    rewritten the history of every completed run.
    """
    model_a, model_b = _synthetic_model(), _synthetic_model()
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(pa, pb), "the two synthetic models are not identical to begin with"
    grads = _fill_grads(model_a, {"embedding": 1e9})
    _restore_grads(model_b, grads)

    new_norm, group_norms = apply_grad_clipping(model_a, grad_clip_mode="global")
    reference_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model_b.parameters() if p.requires_grad], MAX_GRAD_NORM)

    assert torch.equal(new_norm, reference_norm), (
        f"reported grad_norm changed: {new_norm} vs {reference_norm}"
    )
    for (name, pa), pb in zip(model_a.named_parameters(), model_b.parameters()):
        assert torch.equal(pa.grad, pb.grad), f"{name}: post-clip gradient is not bit-identical"
    # Explicitly None, not a computed dict: the default path is not allowed to run
    # a single extra tensor op just to make the log line richer.
    assert group_norms is None


def test_global_mode_still_reports_the_pre_clip_norm():
    """`grad_norm` has always meant the PRE-clip global norm, and old and new logs
    are compared on that field, so it must not quietly start meaning post-clip."""
    model = _synthetic_model()
    _fill_grads(model, {"embedding": 1e9})
    pre_clip = float(torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(p.grad, 2.0) for p in model.parameters()]),
        2.0).item())
    reported, _ = apply_grad_clipping(model, "global")
    assert float(reported.item()) == pytest.approx(pre_clip, rel=1e-6)
    assert float(reported.item()) > MAX_GRAD_NORM, "the case under test is not even clipping"


def test_per_group_reports_the_same_global_pre_clip_norm():
    """Both modes must report the SAME `grad_norm`, otherwise a per-group run's log
    cannot be plotted next to a completed run's."""
    model_a, model_b = _synthetic_model(), _synthetic_model()
    grads = _fill_grads(model_a, {"embedding": 1e9})
    _restore_grads(model_b, grads)
    global_reported, _ = apply_grad_clipping(model_a, "global")
    per_group_reported, group_norms = apply_grad_clipping(model_b, "per-group")
    assert torch.equal(global_reported, per_group_reported)
    assert set(group_norms) == {"embedding", "readout"}


# --------------------------------------------------------------------------- #
# 2. THE regression test: per-group clipping un-freezes the readout.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("arm", ["baseline", "reservoir"])
def test_per_group_bounds_every_group_separately(arm):
    """Every group's post-clip norm <= MAX_GRAD_NORM, on both arms.

    The rule is applied to BOTH arms deliberately: the baseline GRU's gradients do
    not explode and mostly pass through untouched, but a treatment given to one arm
    and not the other stops being a control, and the experiment's whole claim is
    that the arms differ only in architecture.
    """
    model, _ = build_model(arm, seed=0)
    _fill_grads(model, {"embedding": 1e9})
    _, reported_groups = apply_grad_clipping(model, "per-group")

    post = _group_grad_norms(model)
    assert set(post) == set(reported_groups)
    for group, norm in post.items():
        # 1e-6 tolerance: clip_grad_norm_ divides by (total_norm + 1e-6), so a
        # clipped group lands a hair UNDER the bound, never over it.
        assert norm <= MAX_GRAD_NORM + 1e-6, (
            f"[{arm}] group {group!r} post-clip norm {norm} exceeds {MAX_GRAD_NORM}"
        )


def test_per_group_rescues_the_readout_from_the_exploding_embedding():
    """THE regression test for the whole bug, on the real reservoir arm.

    Identical gradients, deliberately unbalanced the way the diagnostic measured
    them (embedding ~1e9, readout ~1). Under the global rule the readout's
    post-clip gradient is annihilated -- 3.52e-09 in the real measurement -- purely
    because 0.3% of the parameters exploded. Under per-group it is O(0.5), i.e. the
    readout is actually being trained.
    """
    model, _ = build_model("reservoir", seed=0)
    grads = _fill_grads(model, {"embedding": 1e9})

    apply_grad_clipping(model, "global")
    global_post = _group_grad_norms(model)

    _restore_grads(model, grads)
    apply_grad_clipping(model, "per-group")
    per_group_post = _group_grad_norms(model)

    print(f"readout post-clip norm: global={global_post['readout']:.6e} "
          f"per-group={per_group_post['readout']:.6e}")
    # O(0.5), not O(1e-9): the group is clipped on its own norm, so it lands ON the
    # bound instead of being scaled by someone else's clip coefficient.
    assert 0.4 <= per_group_post["readout"] <= MAX_GRAD_NORM + 1e-6, (
        f"readout post-clip norm under per-group is {per_group_post['readout']}, "
        f"expected O({MAX_GRAD_NORM})"
    )
    assert per_group_post["readout"] > 1e6 * global_post["readout"], (
        f"per-group barely changed the readout's gradient: "
        f"{per_group_post['readout']} vs {global_post['readout']} under global -- "
        "the coupling this flag exists to break is still there"
    )
    # ...and the exploding group is still clipped, in both modes. per-group is not
    # "turn clipping off".
    assert global_post["embedding"] <= MAX_GRAD_NORM + 1e-6
    assert per_group_post["embedding"] <= MAX_GRAD_NORM + 1e-6


def test_unknown_grad_clip_mode_is_rejected():
    """A typo'd mode must fail loudly, not silently fall through to the default and
    produce a run mislabelled in its own checkpoints."""
    model = _synthetic_model()
    _fill_grads(model)
    with pytest.raises(ValueError, match="unknown grad_clip_mode"):
        apply_grad_clipping(model, "pergroup")
    with pytest.raises(ValueError, match="unknown grad_clip_mode"):
        run_training(arm="baseline", rom_path="/nonexistent.gb", total_steps=8, n_envs=1,
                     rollout_len=8, checkpoint_every=1_000_000, checkpoint_dir="/nonexistent",
                     grad_clip_mode="per_group")


# --------------------------------------------------------------------------- #
# 3. Group discovery. A rename must fail HERE, loudly.
# --------------------------------------------------------------------------- #

def test_discovered_group_names_are_the_real_submodules():
    """Pin the group decomposition of both arms.

    If a refactor renames a submodule -- or nests everything under one -- per-group
    clipping quietly degenerates into global clipping (one group = one clip
    coefficient for everything) and the bug this flag fixes comes back invisibly.
    The names are asserted, not merely printed, for that reason.
    """
    baseline, _ = build_model("baseline", seed=0)
    reservoir, _ = build_model("reservoir", seed=0)

    baseline_groups = group_trainable_parameters(baseline)
    reservoir_groups = group_trainable_parameters(reservoir)
    print(f"baseline groups:  {sorted(baseline_groups)}")
    print(f"reservoir groups: {sorted(reservoir_groups)}")

    assert set(baseline_groups) == {"embedding", "gru", "actor_head", "critic_head"}
    assert set(reservoir_groups) == {"embedding", "readout"}
    # The split that matters: the reservoir arm's tiny exploding group versus the
    # large one it was suppressing (416 params vs 138,763 in the diagnostic).
    assert sum(p.numel() for p in reservoir_groups["embedding"]) == 416
    assert sum(p.numel() for p in reservoir_groups["readout"]) == 138_763
    assert len(reservoir_groups["readout"]) == 29, "the readout's tensor count changed"
    # Frozen weights can never be clipped: they are not trainable and hold no grad.
    assert "reservoir" not in reservoir_groups, (
        "the frozen reservoir turned up as a trainable group -- it must hold zero "
        "nn.Parameters"
    )


# --------------------------------------------------------------------------- #
# 4. Run-tag isolation: the completed matrix must be unreachable by a re-run.
# --------------------------------------------------------------------------- #

def test_run_dir_without_a_tag_is_byte_identical_to_the_historical_path():
    """Every existing run directory, every --resume-from path and every analysis
    script that globs `checkpoints/{arm}_seed{seed}` depends on this exact string."""
    assert run_dir_for("checkpoints", "reservoir", 0) == os.path.join(
        "checkpoints", "reservoir_seed0")
    assert run_dir_for("checkpoints", "reservoir", 0, None) == os.path.join(
        "checkpoints", "reservoir_seed0")
    # An empty tag is "no tag", not a trailing underscore.
    assert run_dir_for("checkpoints", "baseline", 7, "") == os.path.join(
        "checkpoints", "baseline_seed7")


def test_a_tagged_run_cannot_land_on_an_untagged_one():
    """The data-safety property. Without the tag, a corrected reservoir re-run at
    seed 0 writes `checkpoints/reservoir_seed0/step_N.pt` -- straight over the
    completed run's checkpoints, in place."""
    untagged = run_dir_for("checkpoints", "reservoir", 0)
    tagged = run_dir_for("checkpoints", "reservoir", 0, "per-group")
    assert tagged != untagged
    assert os.path.basename(tagged) == "reservoir_seed0_per-group"
    # ...and it is a sibling directory, not a subdirectory of the completed run.
    assert os.path.dirname(tagged) == os.path.dirname(untagged)
    assert not tagged.startswith(untagged + os.sep)


def test_two_different_tags_never_collide():
    dirs = {run_dir_for("checkpoints", arm, seed, tag)
            for arm in ("baseline", "reservoir")
            for seed in (0, 1)
            for tag in (None, "per-group", "eps1e-12")}
    assert len(dirs) == 2 * 2 * 3, f"run directories collided: {sorted(dirs)}"


# --------------------------------------------------------------------------- #
# 5. Backward compatibility with the 200 checkpoints already on disk.
# --------------------------------------------------------------------------- #

def _an_existing_checkpoint():
    """A real completed-run checkpoint, preferring reservoir_seed0 (the run the
    diagnostic was measured on). None if the repo has no runs -- checkpoints are
    gitignored, so a fresh clone legitimately has none."""
    preferred = REPO_ROOT / "checkpoints" / "reservoir_seed0"
    candidates = sorted(preferred.glob("step_*.pt")) if preferred.is_dir() else []
    if not candidates:
        candidates = sorted((REPO_ROOT / "checkpoints").glob("*/step_*.pt"))
    return candidates[0] if candidates else None


def test_an_existing_checkpoint_still_loads_and_defaults_the_new_keys(capsys):
    """The 200 checkpoints on disk predate `grad_clip_mode`/`run_tag` entirely.

    Every read of the new keys goes through `.get(...)`; a direct index would turn
    all 20 completed runs into unloadable files and take the results write-up with
    them. Skipped (not failed) when no run is present, so a fresh clone still gets
    a green suite.
    """
    path = _an_existing_checkpoint()
    if path is None:
        pytest.skip("no checkpoints on disk (they are gitignored); nothing to load")
    print(f"backward-compat checkpoint: {path}")

    raw = torch.load(str(path), map_location="cpu", weights_only=True)
    assert raw.get("grad_clip_mode", "global") in GRAD_CLIP_MODES
    assert raw.get("run_tag", None) is None or isinstance(raw.get("run_tag"), str)

    arm, seed = raw.get("arm"), raw.get("seed")
    model, optimizer = build_model(arm, seed=seed)
    # expected_grad_clip_mode deliberately disagrees with a legacy file's implicit
    # "global": that must WARN, never raise, or no existing run could be resumed
    # under the corrected rule at all.
    step = load_checkpoint(model, optimizer, str(path), expected_arm=arm,
                           expected_seed=seed, expected_grad_clip_mode="per-group")
    assert step == raw["step"]
    assert "WARNING" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# 6. End-to-end: run_training threads both new arguments through.
# --------------------------------------------------------------------------- #

@requires_rom
def test_run_training_threads_grad_clip_mode_and_run_tag(tmp_path):
    """A handful of steps on the real ROM, on the cheap arm: the flags have to
    reach the run directory, the JSONL log AND the checkpoint, or a corrected re-run
    is unidentifiable after the fact."""
    stats = run_training(arm="baseline", rom_path=ROM_PATH, total_steps=8, n_envs=1,
                         rollout_len=8, checkpoint_every=1_000_000,
                         checkpoint_dir=str(tmp_path), seed=0,
                         grad_clip_mode="per-group", run_tag="cliptest")

    assert stats["run_dir"] == str(tmp_path / "baseline_seed0_cliptest")
    assert stats["grad_clip_mode"] == "per-group" and stats["run_tag"] == "cliptest"
    assert not (tmp_path / "baseline_seed0").exists(), (
        "a tagged run also created the untagged directory -- the isolation is fake"
    )

    records = [json.loads(line) for line in open(stats["log_path"], encoding="utf-8")]
    assert records, "nothing was logged"
    for record in records:
        assert record["grad_clip_mode"] == "per-group"
        assert record["run_tag"] == "cliptest"
        assert set(record["grad_norm_groups"]) == {
            "embedding", "gru", "actor_head", "critic_head"}
        assert all(isinstance(v, float) for v in record["grad_norm_groups"].values())
        assert isinstance(record["grad_norm"], float)

    ckpt = torch.load(str(tmp_path / "baseline_seed0_cliptest" /
                          f"step_{stats['final_step']}.pt"),
                      map_location="cpu", weights_only=True)
    assert ckpt["grad_clip_mode"] == "per-group" and ckpt["run_tag"] == "cliptest"


@requires_rom
def test_default_run_is_untagged_global_and_logs_null_groups(tmp_path):
    """The defaults, end to end: unchanged directory, `grad_clip_mode="global"`
    recorded, and `grad_norm_groups` explicitly null rather than computed."""
    stats = run_training(arm="baseline", rom_path=ROM_PATH, total_steps=8, n_envs=1,
                         rollout_len=8, checkpoint_every=1_000_000,
                         checkpoint_dir=str(tmp_path), seed=0)

    assert stats["run_dir"] == str(tmp_path / "baseline_seed0")
    assert stats["grad_clip_mode"] == "global" and stats["run_tag"] is None
    records = [json.loads(line) for line in open(stats["log_path"], encoding="utf-8")]
    for record in records:
        assert record["grad_clip_mode"] == "global"
        assert record["run_tag"] is None
        assert record["grad_norm_groups"] is None
    ckpt = torch.load(str(tmp_path / "baseline_seed0" / f"step_{stats['final_step']}.pt"),
                      map_location="cpu", weights_only=True)
    assert ckpt["grad_clip_mode"] == "global" and ckpt["run_tag"] is None
