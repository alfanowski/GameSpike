"""`--neuron-model` end to end: train.py, evaluate.py and the checkpoint contract.

`models/spiking_reservoir.py` already holds the resonate-and-fire cell itself, and
`tests/test_resonate_and_fire.py` already pins its arithmetic (G0e-ii: the cell at
omega == 0 reproduces `snn.Leaky(beta=0.9)` bit for bit). THIS file pins the other
half of docs/EXPERIMENT_LOG.md §23.5's G0e -- the plumbing that decides whether the
pilot is a controlled comparison at all:

  * **G0e-i, the training half.** With `--neuron-model lif` (the default) and v2's
    flags, a short reservoir run must reproduce a committed
    `checkpoints_v2/reservoir_seed0/train_log.jsonl` prefix EXACTLY, float for
    float. The published v2 LIF arm is this pilot's experimental control (§23.9:
    "no new LIF or GRU runs are performed, which is legitimate only because G0e-i
    verifies the LIF path is bit-identical to the one that produced them"). If the
    neuron-model switch perturbed the LIF training path by one ULP, the control is
    no longer the thing that was published and the whole comparison is void.
  * **The checkpoint is the interface between training and evaluation.** An rf
    checkpoint carries five buffers a LIF model does not have
    (`reservoir.rf.{omega,cos_omega,sin_omega,beta,threshold}`), so
    `evaluate.py` cannot build the model before it knows which neuron model the
    file was written under -- `load_state_dict` at default strictness would refuse
    it. The round-trip is tested rather than the flag, because the flag being
    accepted proves nothing about the file being loadable.
  * **Backward compatibility.** The 400 checkpoints under `checkpoints/` and
    `checkpoints_v2/` predate all three new keys. A missing key must read back as
    `lif`/2.0/32.0, which is precisely what those files are, and evaluate exactly
    as they do today. Same rule, same reason, as
    `tests/test_grad_clip_modes.py`'s own backward-compat section.
  * **The baseline arm has no neuron model at all.** A GRU that silently ignored
    `--neuron-model rf` would let a run mislabelled in its own checkpoint reach
    disk and be tabulated as a resonate-and-fire result.
"""
import contextlib
import inspect
import json
import os
import pathlib
import subprocess

import pytest
import torch

from models.spiking_reservoir import NEURON_MODELS, SpikingReservoir
from training.evaluate import run_evaluation
from training.train import (NEURON_MODELS as TRAIN_NEURON_MODELS,
                            RF_PERIOD_MAX_DEFAULT, RF_PERIOD_MIN_DEFAULT,
                            build_model, load_checkpoint,
                            neuron_config_from_checkpoint, run_training,
                            save_checkpoint)

ROM_PATH = os.environ.get("MARIO_LAND_ROM_PATH")
requires_rom = pytest.mark.skipif(
    not ROM_PATH or not os.path.exists(ROM_PATH),
    reason="MARIO_LAND_ROM_PATH not set or file missing; set it to your own legally-dumped ROM",
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Short enough that an evaluation costs milliseconds; the point of every
# evaluation below is that the model was RECONSTRUCTED correctly, never what it
# scored.
EVAL_STEPS = 40

# The v2 run's own flags, verbatim from docs/RESULTS.md §23 (Reproduction (v2)).
# These are what `checkpoints_v2/reservoir_seed0/train_log.jsonl` was produced
# under, so G0e-i has to reproduce them exactly or it is comparing two different
# experiments and calling the difference a regression.
V2_ROLLOUT_LEN = 128
V2_FLAGS = dict(grad_clip_mode="per-group", embed_init_mode="centered", embed_scale=3.0)

# Every float `train_log.jsonl` carries per update. `grad_norm_groups` is compared
# separately (it is a dict, and under "per-group" it is the field that makes the
# gradient pathology visible), and the label fields are compared as labels.
LOG_FLOAT_FIELDS = ("mean_reward", "mean_extrinsic_reward", "policy_loss",
                    "value_loss", "entropy", "total_loss", "grad_norm")


def _v2_reservoir_seed0_log():
    """`checkpoints_v2/reservoir_seed0/train_log.jsonl`, or None.

    Checkpoints and training logs are gitignored and not distributed (docs/RESULTS.md
    §23), so a fresh clone legitimately has none and this test skips rather than
    fails -- same convention as `tests/test_grad_clip_modes.py::_an_existing_checkpoint`.

    The `--git-common-dir` fallback is load-bearing, not defensive padding: this
    repository's own documented workflow runs sessions from a `git worktree`
    (RESULTS.md §23 pins the v2 matrix to one, EXPERIMENT_LOG.md §17.1), and a
    worktree gets its own empty working tree -- the gitignored data directories
    exist only in the MAIN working tree. Looking only under `REPO_ROOT` would make
    G0e-i silently skip in exactly the situation it is written for.
    """
    roots = [REPO_ROOT]
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        if common:
            roots.append(pathlib.Path(common).parent)
    except (OSError, subprocess.CalledProcessError):
        pass
    for root in roots:
        path = root / "checkpoints_v2" / "reservoir_seed0" / "train_log.jsonl"
        if path.is_file():
            return path
    return None


def _an_existing_lif_checkpoint():
    """One real reservoir checkpoint from the v1 or v2 matrix -- i.e. a file written
    before `neuron_model` existed as a key at all. None if none are on disk."""
    roots = [REPO_ROOT]
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        if common:
            roots.append(pathlib.Path(common).parent)
    except (OSError, subprocess.CalledProcessError):
        pass
    for root in roots:
        for parent in ("checkpoints_v2", "checkpoints"):
            candidates = sorted((root / parent).glob("reservoir_seed*/step_*.pt")) \
                if (root / parent).is_dir() else []
            if candidates:
                return candidates[0]
    return None


@contextlib.contextmanager
def _single_threaded_torch():
    """Run torch's intra-op parallelism at ONE thread, restoring it afterwards.

    THIS IS PART OF THE EXPERIMENTAL CONDITION, not a performance tweak, and it
    took a failing G0e-i to find that out. `scripts/run_training_matrix.py::run_job`
    sets `OMP_NUM_THREADS=1`/`MKL_NUM_THREADS=1` on every child it spawns (so that N
    parallel jobs do not oversubscribe the machine), so every checkpoint under
    `checkpoints_v2/` was produced single-threaded. A float32 reduction splits
    differently across a different number of threads, and the two orderings agree
    only to the last bit or two.

    Measured, on this machine, over the first five updates of reservoir seed 0:
    multi-threaded, this tree reproduces the published log to within 1-3 ULP but
    NOT exactly (e.g. update 3's `policy_loss` -0.04000537469983101 against the
    published -0.04000537097454071); single-threaded it reproduces every field of
    every update exactly. Commit `3050d6c` -- the tree from before any
    resonate-and-fire code existed -- shows the identical multi-threaded offsets,
    which is what establishes that the offsets are the thread count and not the
    neuron-model change.

    So an exact-equality G0e-i run under the default thread count does not measure
    "has the LIF path drifted"; it measures "is this machine's thread count the one
    the matrix launcher imposes". Pinning it here is what makes the comparison
    one-variable.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _records(log_path, limit=None):
    with open(log_path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    return records if limit is None else records[:limit]


# --------------------------------------------------------------------------- #
# 1. G0e-i, the training half. THE test this whole pilot's validity rests on.
# --------------------------------------------------------------------------- #

@requires_rom
def test_lif_default_reproduces_the_v2_training_log_prefix_bit_exactly(tmp_path):
    """§23.5 G0e-i. Five updates of `--arm reservoir --seed 0` under v2's flags,
    with `--neuron-model` left at its default, must reproduce the committed v2 log
    prefix EXACTLY on every logged float.

    Exact equality (`==`), never `pytest.approx`: an approximate version of this
    test passes while the control drifts, which is the one failure mode that
    invalidates every number the pilot will produce. A tolerance here would be a
    tolerance on "is the published control still the control".

    Only the PREFIX is compared, because the v2 run went to 1,000,064 steps and
    this one goes to 640: training is a deterministic recurrence from a seeded
    start, so update N of a short run and update N of a long one are the same
    computation. `total_steps`/`checkpoint_every` enter nowhere except the loop
    bound and the save cadence, neither of which touches the RNG.

    The thread count is pinned for the same "one variable" reason the flags are --
    see `_single_threaded_torch`, without which this test measures the machine
    rather than the code.
    """
    v2_log = _v2_reservoir_seed0_log()
    if v2_log is None:
        pytest.skip("checkpoints_v2/reservoir_seed0/train_log.jsonl not on disk "
                    "(gitignored, not distributed) -- nothing to compare against")
    print(f"G0e-i reference log: {v2_log}")
    expected = _records(v2_log, limit=5)
    assert len(expected) == 5, f"{v2_log} has fewer than 5 records"

    with _single_threaded_torch():
        stats = run_training(arm="reservoir", rom_path=ROM_PATH,
                             total_steps=5 * V2_ROLLOUT_LEN, n_envs=1,
                             rollout_len=V2_ROLLOUT_LEN, checkpoint_every=10 ** 9,
                             checkpoint_dir=str(tmp_path), seed=0, **V2_FLAGS)
    got = _records(stats["log_path"], limit=5)
    assert len(got) == 5

    for i, (want, have) in enumerate(zip(expected, got), start=1):
        assert have["step"] == want["step"], f"update {i}: step diverged"
        for field in LOG_FLOAT_FIELDS:
            assert have[field] == want[field], (
                f"G0e-i FAILED at update {i}, field {field!r}: v2 published "
                f"{want[field]!r}, this tree produces {have[field]!r}. The LIF "
                "training path is no longer bit-identical to the one that produced "
                "checkpoints_v2/, so the published v2 arm is not this pilot's "
                "control and no rf-vs-lif comparison built on it is valid."
            )
        assert have["grad_norm_groups"] == want["grad_norm_groups"], (
            f"update {i}: per-group pre-clip norms diverged"
        )
        # The new label must be on the line, and it must say what was run.
        assert have["neuron_model"] == "lif"


@requires_rom
def test_the_default_neuron_model_is_lif_everywhere_it_is_recorded(tmp_path):
    """The default is not merely "lif" in one signature: it has to be lif in the
    checkpoint, in every log line and in the returned stats, or a v2-equivalent
    re-run would be labelled as something else on disk."""
    stats = run_training(arm="reservoir", rom_path=ROM_PATH, total_steps=8, n_envs=1,
                         rollout_len=8, checkpoint_every=10 ** 9,
                         checkpoint_dir=str(tmp_path), seed=0)
    assert stats["neuron_model"] == "lif"
    assert stats["rf_period_min"] == RF_PERIOD_MIN_DEFAULT
    assert stats["rf_period_max"] == RF_PERIOD_MAX_DEFAULT
    for record in _records(stats["log_path"]):
        assert record["neuron_model"] == "lif"
        assert record["rf_period_min"] == RF_PERIOD_MIN_DEFAULT
        assert record["rf_period_max"] == RF_PERIOD_MAX_DEFAULT
    raw = torch.load(os.path.join(stats["run_dir"], f"step_{stats['final_step']}.pt"),
                     map_location="cpu", weights_only=True)
    assert raw["neuron_model"] == "lif"
    assert raw["rf_period_min"] == RF_PERIOD_MIN_DEFAULT
    assert raw["rf_period_max"] == RF_PERIOD_MAX_DEFAULT
    assert "reservoir.rf.omega" not in raw["model"], (
        "a LIF checkpoint carries resonate-and-fire buffers -- the default path is "
        "not the historical construction"
    )


# --------------------------------------------------------------------------- #
# 2. rf round-trips through the checkpoint.
# --------------------------------------------------------------------------- #

@requires_rom
def test_an_rf_run_round_trips_through_its_own_checkpoint(tmp_path):
    """Train 2 updates under `--neuron-model rf`, then load the checkpoint back
    the way `evaluate.py` does.

    `load_state_dict` runs at DEFAULT strictness, which is the whole point: an rf
    checkpoint carries five buffers a LIF model does not have, so a loader that
    built the model before reading the file would fail here with torch's
    unexpected-key dump. Strict loading succeeding IS the proof that
    `reservoir.rf.omega` was both saved and matched.
    """
    stats = run_training(arm="reservoir", rom_path=ROM_PATH, total_steps=16, n_envs=1,
                         rollout_len=8, checkpoint_every=10 ** 9,
                         checkpoint_dir=str(tmp_path), seed=0,
                         neuron_model="rf", run_tag="rf")
    path = os.path.join(stats["run_dir"], f"step_{stats['final_step']}.pt")
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert raw["neuron_model"] == "rf"
    assert raw["rf_period_min"] == RF_PERIOD_MIN_DEFAULT
    assert raw["rf_period_max"] == RF_PERIOD_MAX_DEFAULT
    assert "reservoir.rf.omega" in raw["model"]

    # Exactly what evaluate.py now does: read the file's own labels FIRST, build
    # from them, then load.
    config = neuron_config_from_checkpoint(raw)
    assert config == {"neuron_model": "rf",
                      "rf_period_min": RF_PERIOD_MIN_DEFAULT,
                      "rf_period_max": RF_PERIOD_MAX_DEFAULT}
    model, optimizer = build_model("reservoir", **config)
    step = load_checkpoint(model, optimizer, path, expected_arm="reservoir",
                           expected_seed=0)
    assert step == stats["final_step"]
    assert model.neuron_model == "rf"
    assert torch.equal(model.reservoir.omega, raw["model"]["reservoir.rf.omega"])
    # ...and it is a real frequency draw, not a silently-LIF zero vector.
    assert float(model.reservoir.omega.abs().max()) > 0.0

    results = run_evaluation(arm="reservoir", checkpoint_path=path, rom_path=ROM_PATH,
                             n_episodes=1, max_steps_per_episode=EVAL_STEPS)
    assert results["neuron_model"] == "rf"
    assert isinstance(results["mean_extrinsic_return"], float)


@requires_rom
def test_an_rf_checkpoint_cannot_be_loaded_into_a_lif_model(tmp_path):
    """The failure this ordering exists to prevent, stated as a test: build first,
    read the labels second, and torch refuses the file several frames away from the
    actual mistake."""
    stats = run_training(arm="reservoir", rom_path=ROM_PATH, total_steps=8, n_envs=1,
                         rollout_len=8, checkpoint_every=10 ** 9,
                         checkpoint_dir=str(tmp_path), seed=0,
                         neuron_model="rf", run_tag="rf")
    path = os.path.join(stats["run_dir"], f"step_{stats['final_step']}.pt")
    lif_model, lif_optimizer = build_model("reservoir")
    with pytest.raises(Exception):
        load_checkpoint(lif_model, lif_optimizer, path, expected_arm="reservoir")


# --------------------------------------------------------------------------- #
# 3. Backward compatibility with every checkpoint written before this flag.
# --------------------------------------------------------------------------- #

def test_a_checkpoint_without_the_new_keys_reads_back_as_lif():
    """400 checkpoints under `checkpoints/` and `checkpoints_v2/` contain none of
    the three new keys. `.get(...)` with the historical default is the only read
    that keeps them loadable; a direct index turns every completed run into an
    unloadable file and takes the published results with it."""
    assert neuron_config_from_checkpoint({}) == {
        "neuron_model": "lif",
        "rf_period_min": RF_PERIOD_MIN_DEFAULT,
        "rf_period_max": RF_PERIOD_MAX_DEFAULT,
    }
    # A partially-labelled file (a key added later than another) still resolves.
    assert neuron_config_from_checkpoint({"neuron_model": "rf"})["rf_period_max"] == \
        RF_PERIOD_MAX_DEFAULT


@requires_rom
def test_a_stripped_checkpoint_evaluates_exactly_as_a_labelled_one(tmp_path):
    """The backward-compat claim, measured rather than asserted: the same weights
    with and without the `neuron_model` key must produce the IDENTICAL evaluation,
    per-episode return for per-episode return. If they differ at all, the new read
    path has moved numbers on files that predate it."""
    model, optimizer = build_model("reservoir")
    labelled = tmp_path / "labelled.pt"
    save_checkpoint(model, optimizer, step=0, path=str(labelled))

    raw = torch.load(str(labelled), map_location="cpu", weights_only=True)
    assert raw["neuron_model"] == "lif"
    for key in ("neuron_model", "rf_period_min", "rf_period_max"):
        del raw[key]
    stripped = tmp_path / "stripped.pt"
    torch.save(raw, str(stripped))

    common = dict(arm="reservoir", rom_path=ROM_PATH, n_episodes=2,
                  max_steps_per_episode=EVAL_STEPS, seed=3)
    from_labelled = run_evaluation(checkpoint_path=str(labelled), **common)
    from_stripped = run_evaluation(checkpoint_path=str(stripped), **common)

    assert from_stripped["neuron_model"] == "lif"
    for key in ("extrinsic_returns", "combined_returns", "episode_lengths",
                "episode_seeds"):
        assert from_stripped[key] == from_labelled[key], f"{key} moved"


@requires_rom
def test_a_real_pre_flag_checkpoint_still_evaluates(tmp_path):
    """The same guarantee against an actual file from the v1/v2 matrix rather than
    a synthesised one. Skipped (not failed) on a fresh clone, where checkpoints are
    legitimately absent."""
    path = _an_existing_lif_checkpoint()
    if path is None:
        pytest.skip("no reservoir checkpoints on disk (they are gitignored)")
    raw = torch.load(str(path), map_location="cpu", weights_only=True)
    print(f"pre-flag checkpoint: {path} (keys: {sorted(raw)})")
    assert "neuron_model" not in raw, (
        f"{path} already carries a neuron_model key -- pick an older file, this one "
        "does not test the backward-compat path"
    )
    results = run_evaluation(arm=raw["arm"], checkpoint_path=str(path),
                             rom_path=ROM_PATH, n_episodes=1,
                             max_steps_per_episode=EVAL_STEPS)
    assert results["neuron_model"] == "lif"


# --------------------------------------------------------------------------- #
# 4. The baseline arm has no neuron model.
# --------------------------------------------------------------------------- #

def test_the_baseline_arm_refuses_a_neuron_model():
    """A GRU has no neuron model. Silently ignoring the flag would let a run
    labelled `neuron_model="rf"` in its own checkpoint and log reach disk having
    trained an ordinary GRU -- a mislabelled experiment, which is worse than a
    crashed one because it is tabulated."""
    with pytest.raises(ValueError, match="neuron_model"):
        build_model("baseline", neuron_model="rf")
    with pytest.raises(ValueError, match="neuron_model"):
        run_training(arm="baseline", rom_path="/nonexistent.gb", total_steps=8,
                     n_envs=1, rollout_len=8, checkpoint_every=10 ** 9,
                     checkpoint_dir="/nonexistent", neuron_model="rf")
    # ...and the default is still accepted on the baseline arm, or the flag would
    # have made the baseline arm unbuildable.
    model, _ = build_model("baseline", neuron_model="lif")
    assert model is not None


def test_an_unknown_neuron_model_is_rejected_on_both_arms():
    for arm in ("baseline", "reservoir"):
        with pytest.raises(ValueError, match="neuron_model"):
            build_model(arm, neuron_model="resonate")


# --------------------------------------------------------------------------- #
# 5. Resuming across neuron models is refused, not warned about.
# --------------------------------------------------------------------------- #

def test_resuming_across_neuron_models_raises(tmp_path):
    """Unlike a `grad_clip_mode` mismatch, which WARNS because resuming under a
    different optimisation rule is a legitimate (if hazardous) deliberate act, a
    `neuron_model` mismatch is an ARCHITECTURE mismatch: the two models do not even
    have the same buffers, so there is no state to carry across and nothing the
    resulting checkpoint would be a valid instance of. It is the same class of
    error as an `arm` mismatch, and it raises for the same reason."""
    model, optimizer = build_model("reservoir", neuron_model="rf")
    path = tmp_path / "rf.pt"
    save_checkpoint(model, optimizer, step=0, path=str(path))

    fresh, fresh_optimizer = build_model("reservoir", neuron_model="rf")
    with pytest.raises(ValueError, match="neuron_model mismatch"):
        load_checkpoint(fresh, fresh_optimizer, str(path), expected_arm="reservoir",
                        expected_neuron_model="lif")
    # The matching case still loads.
    assert load_checkpoint(fresh, fresh_optimizer, str(path), expected_arm="reservoir",
                           expected_neuron_model="rf") == 0


def test_a_pre_flag_checkpoint_resumes_as_lif_without_raising(tmp_path):
    """The backward-compat corollary of the rule above: a file with no
    `neuron_model` key reads back as "lif", so resuming any of the 400 existing
    runs under the default is a match and never raises."""
    model, optimizer = build_model("reservoir")
    path = tmp_path / "legacy.pt"
    save_checkpoint(model, optimizer, step=0, path=str(path))
    raw = torch.load(str(path), map_location="cpu", weights_only=True)
    for key in ("neuron_model", "rf_period_min", "rf_period_max"):
        del raw[key]
    torch.save(raw, str(path))

    fresh, fresh_optimizer = build_model("reservoir")
    assert load_checkpoint(fresh, fresh_optimizer, str(path), expected_arm="reservoir",
                           expected_neuron_model="lif") == 0


# --------------------------------------------------------------------------- #
# 6. One rule, one place: the CLI's constants are the model's constants.
# --------------------------------------------------------------------------- #

def test_train_pys_neuron_constants_are_the_models_own():
    """`training/train.py` must not carry a second, drifting copy of either the
    valid-mode list or §23.2's pre-registered period bounds. The mode tuple is
    imported outright; the two bounds are defaults that also live in
    `SpikingReservoir.__init__`'s signature, so their agreement is asserted here
    against `inspect` rather than trusted."""
    assert TRAIN_NEURON_MODELS is NEURON_MODELS
    signature = inspect.signature(SpikingReservoir.__init__)
    assert signature.parameters["rf_period_min"].default == RF_PERIOD_MIN_DEFAULT
    assert signature.parameters["rf_period_max"].default == RF_PERIOD_MAX_DEFAULT
    # §23.2 fixes these numbers before measurement; they are not free parameters.
    assert (RF_PERIOD_MIN_DEFAULT, RF_PERIOD_MAX_DEFAULT) == (2.0, 32.0)


@requires_rom
def test_non_default_rf_period_bounds_are_recorded_and_honoured(tmp_path):
    """The bounds are part of a run's identity: two rf runs with different
    frequency ranges are different experiments, so the checkpoint and the log have
    to carry the numbers rather than merely the flag."""
    stats = run_training(arm="reservoir", rom_path=ROM_PATH, total_steps=8, n_envs=1,
                         rollout_len=8, checkpoint_every=10 ** 9,
                         checkpoint_dir=str(tmp_path), seed=0, neuron_model="rf",
                         rf_period_min=4.0, rf_period_max=8.0, run_tag="rf-narrow")
    assert stats["rf_period_min"] == 4.0 and stats["rf_period_max"] == 8.0
    for record in _records(stats["log_path"]):
        assert record["rf_period_min"] == 4.0 and record["rf_period_max"] == 8.0
    raw = torch.load(os.path.join(stats["run_dir"], f"step_{stats['final_step']}.pt"),
                     map_location="cpu", weights_only=True)
    assert raw["rf_period_min"] == 4.0 and raw["rf_period_max"] == 8.0
    # omega = 2*pi/T with T in [4, 8] -- the bounds actually reached the draw.
    omega = raw["model"]["reservoir.rf.omega"]
    import math
    assert float(omega.min()) >= 2 * math.pi / 8.0 - 1e-6
    assert float(omega.max()) <= 2 * math.pi / 4.0 + 1e-6
