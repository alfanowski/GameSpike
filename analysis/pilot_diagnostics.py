"""Go/no-go diagnostic on the interrupted 9-run v2 pilot, before ~4.5h of compute.

WHAT THIS DECIDES. Two construction defects were diagnosed in the reservoir arm
and fixed behind flags (`--grad-clip-mode per-group`, `--embed-init-mode centered
--embed-scale 3.0`; see `training/train.py`'s module docstring and
`docs/EXPERIMENT_LOG.md` §12). An interrupted pilot left 9 runs on disk -- 3 seeds
x 3 corner configurations, each stopped at update 2344 = step 300,032, i.e. 30% of
a full run. This module answers, from those runs plus the completed v1 runs, one
question: is `per-group` + `centered@3.0` safe to commit the full 10-seed x 2-arm
matrix to.

It is READ-ONLY with respect to `checkpoints/`, `checkpoints_init/` and
`results/`, it never trains, and it prints. It writes no files. Every number it
reports is recomputed from the on-disk artefacts each time it runs, so the report
is reproducible rather than transcribed.

THE THREE PRE-REGISTERED HYPOTHESES, and the exact falsification conditions they
were registered with (restated here so the code and the claim cannot drift apart):

  H1  the centred-init silent-unit suppression PERSISTS through training under
      per-group clipping. FALSIFIED IF the mean silent-unit fraction over seeds
      0-2, measured on `tests/data/real_obs_6000.npy` with the TRAINED embedding
      from `reservoir_seed{s}_clipemb/step_300032.pt`, exceeds 15%.
  H2  per-group clipping keeps the READOUT's effective optimizer step in a
      healthy range despite the embedding's exploding pre-clip norm. FALSIFIED IF
      the readout's median `||dp||/||p||` for one reconstructed optimizer step is
      < 1e-4. (v1's frozen-readout pathology: 1.9034e-05. Healthy baseline GRU:
      4.273e-04.)
  H3  DESCRIPTIVE ONLY, no pass/fail: the 2x2 factorial of the two fixes on
      `mean_extrinsic_reward` over updates 1876-2344. Three seeds at 30% of a run
      cannot support an arm claim and this module refuses to make one.

WHY THE SILENT-UNIT MEASUREMENT REUSES `tests/test_embedding_centering.py`'s
MACHINERY VERBATIM (`_silent_fraction`, and the committed 6,000-step real-
observation fixture). That test's docstring records that an i.i.d. Gaussian
surrogate matched to the same per-dimension means and stds does NOT reproduce the
effect and in fact points the wrong way (24.12% silent vs 1.66% on real data),
because real observations are strongly temporally correlated and a beta=0.9 LIF
membrane integrates low-frequency energy with gain up to 1/(1-beta)=10. The
fixture is a requirement, not a convenience. Re-deriving the measurement here
would have risked measuring something subtly different from the number the fix was
accepted on, so it is imported and reused unchanged.

WHICH ADAM STATISTIC H2 USES, AND WHY IT NEEDS NO GRADIENT AND NO EMULATOR. A
checkpoint stores `optimizer.state_dict()`, i.e. `exp_avg`, `exp_avg_sq` and the
step count, as they stood immediately AFTER the last `optimizer.step()`. torch's
Adam updates its moments and then applies

    dp = -(lr / bias_correction1) * exp_avg / (sqrt(exp_avg_sq)/sqrt(bias_correction2) + eps)

so those three stored quantities reconstruct the last REAL optimizer step of the
run EXACTLY -- the actual step that was taken on the actual clipped gradient, not
a proxy for it and not a synthetic replay. That is strictly stronger evidence than
re-running the environment to manufacture a fresh gradient would be, and it costs
no ROM, no rollout and no training. Both statistics the original diagnostic quoted
are reported:

  * per-tensor `||dp|| / ||p||`, median over the group's tensors -- the quantity
    H2's falsification condition is written in terms of;
  * elementwise `|m_hat| / sqrt(v_hat)`, median over the group's elements -- the
    lr- and eps-free version, which is what "the readout is effectively frozen"
    was originally shown with (7.475e-04 reservoir vs 1.346e-01 baseline GRU).

`||p||` is read from the same checkpoint, i.e. AFTER the step rather than before
it. The ratio is O(1e-3) at most, so post- and pre-step norms agree to a part in a
thousand; this is noted rather than corrected because correcting it would require
assuming the very update being measured.

THE 2x2 CELL LABELS. `clip` = per-group + legacy, `emb` = global + centered,
`clipemb` = per-group + centered, and the fourth cell (global + legacy, the
original v1 condition) has no pilot directory because it IS the completed v1 run
at `checkpoints/reservoir_seed{s}/`. v1 runs are 7,813 updates long, so they are
truncated to update 2344 here -- matched on UPDATE INDEX, not on step number,
because the two generations' checkpoint step numbering differs slightly
(step_300288 vs step_300032) while the update index means the same thing in both.
"""
import json
import math
import os
import statistics
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.embedding_init import EMBED_INIT_MODES  # noqa: E402
from training.train import (LEARNING_RATE, MAX_GRAD_NORM,  # noqa: E402
                            RF_PERIOD_MIN_DEFAULT, RF_PERIOD_MAX_DEFAULT,
                            build_model, group_trainable_parameters)

CHECKPOINTS = os.path.join(REPO_ROOT, "checkpoints")
REAL_OBS_PATH = os.path.join(REPO_ROOT, "tests", "data", "real_obs_6000.npy")

SEEDS = (0, 1, 2)
PILOT_UPDATE = 2344            # where the interrupted pilot stopped, on every run
WINDOW_START = 1876            # last 20% of the pilot: updates 1876..2344 inclusive
TREND_WINDOW = 500             # updates used for the instability/trend statistics

# Adam's own defaults, as recorded in every checkpoint's param_groups. Read from
# the checkpoint at use time rather than trusted from here; these are the expected
# values and a mismatch is reported instead of silently ignored.
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8

# The 2x2 factorial. `dir_suffix=None` means "the completed v1 run", which carries
# no tag because it predates the flags entirely.
CELLS = (
    ("global   + legacy  (v1)", None, "global", "legacy", 1.0),
    ("per-group+ legacy  (clip)", "clip", "per-group", "legacy", 1.0),
    ("global   + centered (emb)", "emb", "global", "centered", 3.0),
    ("per-group+ centered (clipemb)", "clipemb", "per-group", "centered", 3.0),
)


# --------------------------------------------------------------------------- #
# on-disk readers
# --------------------------------------------------------------------------- #

def run_dir(seed, suffix=None, arm="reservoir"):
    name = f"{arm}_seed{seed}" + (f"_{suffix}" if suffix else "")
    return os.path.join(CHECKPOINTS, name)


def read_log(seed, suffix=None, arm="reservoir"):
    """Every JSONL record of one run, in file order."""
    path = os.path.join(run_dir(seed, suffix, arm), "train_log.jsonl")
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_ckpt(path):
    # weights_only=True for the same reason training/train.py uses it: these files
    # hold only tensors and state dicts, and the default unpickles arbitrary code.
    return torch.load(path, map_location="cpu", weights_only=True)


def trainable_names(arm, seed):
    """`named_parameters()` order restricted to trainables -- which is exactly the
    index order Adam's `state` dict is keyed by, since `build_model` hands the
    optimizer `[p for p in model.parameters() if p.requires_grad]` as ONE group."""
    model, _ = build_model(arm, seed=seed)
    return [n for n, p in model.named_parameters() if p.requires_grad]


# --------------------------------------------------------------------------- #
# H1: silent units under the TRAINED embedding
# --------------------------------------------------------------------------- #

def silent_fraction(model, obs):
    """Verbatim `tests/test_embedding_centering.py::_silent_fraction`.

    Returns (silent, mean_spike_rate, saturated), where "silent" is the fraction
    of units that never fire once over the window and "saturated" the fraction
    that fire on every single step. "Never fired in 6,000 steps" is an UPPER BOUND
    on permanently silent, not the same thing -- a unit firing at true rate 1e-4
    reads as silent here. §12's limitations section says the same; it is repeated
    here so a number lifted out of this report carries its own caveat.

    THE STATE TUPLE IS FOUR-WIDE IN BOTH NEURON MODELS (§23.2), and threading all
    four is not optional even on the LIF arm. `imem` is the resonate-and-fire
    quadrature companion; under LIF it is an inert zeros tensor that the reservoir
    never reads, so passing it costs nothing and reproduces the pre-§23 numbers
    exactly, while NOT passing it makes `step` re-allocate a fresh zeros tensor
    per timestep and -- on the rf arm -- silently discard the quadrature state,
    which would turn a resonate-and-fire measurement into a plausible, wrong one.
    Unpacked positionally on purpose, so a future arity change fails loudly here
    rather than dropping a component; `tests/test_pilot_diagnostics.py` covers
    both modes for the same reason.
    """
    mem, imem, spk, _window = model.init_state(1, torch.device("cpu"))
    ever = torch.zeros(model.reservoir_size, dtype=torch.bool)
    always = torch.ones(model.reservoir_size, dtype=torch.bool)
    total_rate = 0.0
    with torch.no_grad():
        for t in range(obs.shape[0]):
            emb = model.embedding(obs[t:t + 1])
            spk, mem, imem = model.reservoir.step(emb, mem, spk, imem)
            fired = spk[0] > 0
            ever |= fired
            always &= fired
            total_rate += float(fired.float().mean())
    return (1.0 - ever.float().mean().item(),
            total_rate / obs.shape[0],
            always.float().mean().item())


def reservoir_at(seed, embed_init_mode, embed_scale, ckpt_path=None,
                 neuron_model="lif", rf_period_min=RF_PERIOD_MIN_DEFAULT,
                 rf_period_max=RF_PERIOD_MAX_DEFAULT):
    """The reservoir-arm model as this run had it, either at init or at a step.

    At init the construction sequence mirrors `run_training` exactly --
    `torch.manual_seed(seed)` then `build_model(..., seed=seed, ...)` -- because
    the trainable init is drawn from the GLOBAL RNG while the frozen reservoir
    takes its own `seed=` argument, and only doing both reproduces what
    `--seed s` actually produced.

    Loading a checkpoint also overwrites the frozen reservoir buffers, so this
    additionally checks them against the freshly-constructed ones: they must be
    bit-identical, which is an independent confirmation of spec §3's frozen
    invariant on the pilot's own files rather than on a fresh model.

    `neuron_model`/`rf_period_min`/`rf_period_max` (docs/EXPERIMENT_LOG.md §23)
    are FORWARDED VERBATIM to `build_model` and are not inferred from anything.
    The defaults are the historical path, bit-for-bit, so every pre-§23 caller is
    unaffected -- but unlike `embed_init_mode`/`embed_scale`, which the module
    docstring above shows are inert once a checkpoint is loaded, THESE ARE NOT
    INERT: an `rf` checkpoint carries five `reservoir.rf.*` buffers a LIF model
    does not have, so `load_state_dict` refuses it outright rather than
    overwriting. A caller that loads a checkpoint must therefore pass the
    construction arguments the checkpoint was WRITTEN under -- read them with
    `training.train.neuron_config_from_checkpoint`, which supplies the pre-§23
    defaults for the 400 committed files that predate the keys.
    """
    torch.manual_seed(seed)
    model, _ = build_model("reservoir", seed=seed, embed_init_mode=embed_init_mode,
                           embed_scale=embed_scale, neuron_model=neuron_model,
                           rf_period_min=rf_period_min, rf_period_max=rf_period_max)
    frozen_drift = None
    if ckpt_path is not None:
        reference = {k: v.clone() for k, v in model.state_dict().items()
                     if k.startswith("reservoir.")}
        model.load_state_dict(load_ckpt(ckpt_path)["model"])
        frozen_drift = max(
            (model.state_dict()[k] - v).abs().max().item() for k, v in reference.items()
        )
    model.eval()
    return model, frozen_drift


# --------------------------------------------------------------------------- #
# H2: what one real Adam step actually did
# --------------------------------------------------------------------------- #

def adam_step_stats(ckpt, names, group_filter):
    """Reconstruct the LAST optimizer step from the stored Adam state.

    `group_filter(name) -> bool` selects the parameter group (the first
    dot-separated component of the name, the same rule
    `group_trainable_parameters` clips by). Returns a dict of the two statistics
    plus the bookkeeping needed to trust them.
    """
    opt = ckpt["optimizer"]
    groups = opt["param_groups"]
    assert len(groups) == 1, f"expected one Adam param group, found {len(groups)}"
    g = groups[0]
    lr, (b1, b2), eps = g["lr"], g["betas"], g["eps"]
    assert (b1, b2) == (BETA1, BETA2) and eps == EPS and lr == LEARNING_RATE, (
        f"unexpected Adam hyperparameters in checkpoint: lr={lr}, betas={(b1, b2)}, eps={eps}"
    )
    model_sd = ckpt["model"]

    per_tensor, ratio_chunks, steps, n_params = [], [], set(), 0
    for idx, name in enumerate(names):
        if not group_filter(name):
            continue
        st = opt["state"][idx]
        m, v = st["exp_avg"].double(), st["exp_avg_sq"].double()
        t = float(st["step"])
        steps.add(t)
        p = model_sd[name].double()
        assert m.shape == p.shape, f"Adam state index {idx} does not match {name}"
        bc1, bc2 = 1.0 - b1 ** t, 1.0 - b2 ** t
        # The exact torch.optim.Adam update, single-tensor path.
        denom = v.sqrt() / math.sqrt(bc2) + eps
        dp = (lr / bc1) * m / denom
        per_tensor.append((name, float(dp.norm() / p.norm()), int(p.numel())))
        # The lr/eps-free version: |m_hat| / sqrt(v_hat), elementwise.
        ratio_chunks.append((m.abs() / bc1) / ((v / bc2).sqrt() + 1e-300))
        n_params += int(p.numel())

    ratios = torch.cat([c.flatten() for c in ratio_chunks])
    return {
        "adam_step_count": sorted(steps),
        "n_tensors": len(per_tensor),
        "n_params": n_params,
        "per_tensor": per_tensor,
        "median_dp_over_p": statistics.median(r for _, r, _ in per_tensor),
        "min_dp_over_p": min(r for _, r, _ in per_tensor),
        "max_dp_over_p": max(r for _, r, _ in per_tensor),
        "median_mhat_over_sqrt_vhat": float(ratios.median()),
    }


# --------------------------------------------------------------------------- #
# small numeric helpers (no scipy, matching analysis/aggregate_results.py's rule)
# --------------------------------------------------------------------------- #

def quantile(xs, q):
    """Linear-interpolated quantile of a list, sorted here so callers need not."""
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def ols_slope(xs, ys):
    """Least-squares slope and intercept of y on x. Returned as a pair so the
    caller can report the fitted change over the window rather than a bare slope,
    which is unreadable at 1e-6 per update."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, my - slope * mx


def finite_scan(obj, path=""):
    """Every non-finite leaf under a nested dict/list/tensor, as (path, value)."""
    bad = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            bad += finite_scan(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad += finite_scan(v, f"{path}[{i}]")
    elif isinstance(obj, torch.Tensor):
        if obj.is_floating_point() and not torch.isfinite(obj).all():
            bad.append((path, "tensor contains NaN/Inf"))
    elif isinstance(obj, float):
        if not math.isfinite(obj):
            bad.append((path, obj))
    return bad


# --------------------------------------------------------------------------- #
# report sections
# --------------------------------------------------------------------------- #

def section_config_sanity():
    print("\n" + "=" * 78)
    print("0. CONFIG SANITY -- did the pilot runs use the settings they claim?")
    print("=" * 78)
    for label, suffix, clip_mode, init_mode, scale in CELLS:
        for seed in SEEDS:
            recs = read_log(seed, suffix)
            got = {(r.get("grad_clip_mode", "global"), r.get("embed_init_mode", "legacy"),
                    float(r.get("embed_scale", 1.0)), r.get("run_tag"), r.get("arm"),
                    r.get("seed")) for r in recs}
            ok_log = got == {(clip_mode, init_mode, scale, suffix, "reservoir", seed)}
            updates = [r["update"] for r in recs]
            contiguous = updates == list(range(1, len(updates) + 1))
            steps_ok = all(r["step"] == r["update"] * 128 for r in recs)
            note = ""
            if suffix is not None:
                path = os.path.join(run_dir(seed, suffix), "step_300032.pt")
                c = load_ckpt(path)
                ok_ck = (c.get("grad_clip_mode", "global") == clip_mode
                         and c.get("embed_init_mode", "legacy") == init_mode
                         and float(c.get("embed_scale", 1.0)) == scale
                         and c.get("run_tag") == suffix and c.get("arm") == "reservoir"
                         and c.get("seed") == seed and c["step"] == 300032)
                note = f" ckpt={'OK' if ok_ck else 'MISMATCH ' + str(c.get('run_tag'))}"
            print(f"  {label:<30} seed{seed}  n={len(recs):<5} "
                  f"log={'OK' if ok_log else 'MISMATCH ' + str(sorted(got))} "
                  f"updates={'contiguous' if contiguous else 'GAPS'} "
                  f"step=update*128:{'OK' if steps_ok else 'NO'}{note}")
    assert all(m in EMBED_INIT_MODES for _, _, _, m, _ in CELLS)


def section_h1():
    print("\n" + "=" * 78)
    print("1. H1 -- does the centred-init silent-unit fix SURVIVE 300k steps of training?")
    print("    falsified if mean silent fraction over seeds 0-2 at step 300032 > 15%")
    print("=" * 78)
    obs = torch.as_tensor(np.load(REAL_OBS_PATH), dtype=torch.float32)
    print(f"  fixture: {os.path.relpath(REAL_OBS_PATH, REPO_ROOT)}  shape={tuple(obs.shape)}")
    print(f"\n  {'run':<28}{'stage':>12}{'silent':>11}{'spike rate':>13}"
          f"{'saturated':>11}{'frozen drift':>14}")
    print("  " + "-" * 87)
    finals = {}
    for suffix, init_mode, scale in (("clipemb", "centered", 3.0),
                                     ("emb", "centered", 3.0),
                                     ("clip", "legacy", 1.0)):
        for seed in SEEDS:
            row = []
            for stage, ckpt in (("init (step 0)", None),
                                ("step 100096", "step_100096.pt"),
                                ("step 200192", "step_200192.pt"),
                                ("step 300032", "step_300032.pt")):
                path = os.path.join(run_dir(seed, suffix), ckpt) if ckpt else None
                model, drift = reservoir_at(seed, init_mode, scale, path)
                s, rate, sat = silent_fraction(model, obs)
                row.append((stage, s, rate, sat, drift))
                if ckpt == "step_300032.pt":
                    finals.setdefault(suffix, []).append(s)
            for i, (stage, s, rate, sat, drift) in enumerate(row):
                name = f"reservoir_seed{seed}_{suffix}" if i == 0 else ""
                d = "n/a (built)" if drift is None else f"{drift:.1e}"
                print(f"  {name:<28}{stage:>12}{s:>10.4%}{rate:>13.6f}{sat:>10.4%}{d:>14}")
            print()
    print(f"  {'config':<32}{'mean silent @300032':>22}{'per seed':>34}")
    for suffix in ("clipemb", "emb", "clip"):
        vals = finals[suffix]
        print(f"  {suffix:<32}{sum(vals) / len(vals):>21.4%}   "
              f"{'  '.join(f'{v:.4%}' for v in vals):>31}")
    mean_clipemb = sum(finals["clipemb"]) / len(finals["clipemb"])
    verdict = "FALSIFIED" if mean_clipemb > 0.15 else "CONFIRMED"
    print(f"\n  H1 {verdict}: clipemb mean silent = {mean_clipemb:.4%} "
          f"vs the 15% falsification threshold")
    return mean_clipemb, verdict


def section_h2():
    print("\n" + "=" * 78)
    print("2. H2 -- did per-group clipping keep the READOUT's optimizer step healthy?")
    print("    falsified if the readout's median ||dp||/||p|| < 1e-4")
    print("    statistic: the LAST REAL Adam step, reconstructed exactly from the")
    print("    checkpoint's stored exp_avg / exp_avg_sq / step. No gradient re-run.")
    print("=" * 78)
    res_names = trainable_names("reservoir", 0)
    base_names = trainable_names("baseline", 0)
    is_readout = lambda n: n.split(".")[0] == "readout"       # noqa: E731
    is_embed = lambda n: n.split(".")[0] == "embedding"       # noqa: E731

    rows = []
    for seed in SEEDS:
        ck = load_ckpt(os.path.join(run_dir(seed, "clipemb"), "step_300032.pt"))
        rows.append((f"clipemb seed{seed} readout", adam_step_stats(ck, res_names, is_readout)))
        rows.append((f"clipemb seed{seed} embedding",
                     adam_step_stats(ck, res_names, is_embed)))
    for seed in SEEDS:
        ck = load_ckpt(os.path.join(run_dir(seed, None), "step_300288.pt"))
        rows.append((f"v1 global seed{seed} readout",
                     adam_step_stats(ck, res_names, is_readout)))
    for seed in SEEDS:
        ck = load_ckpt(os.path.join(run_dir(seed, None, "baseline"), "step_300288.pt"))
        rows.append((f"baseline GRU seed{seed} non-embed",
                     adam_step_stats(ck, base_names, lambda n: not is_embed(n))))
        rows.append((f"baseline GRU seed{seed} all",
                     adam_step_stats(ck, base_names, lambda n: True)))

    print(f"\n  {'checkpoint / group':<34}{'median':>12}{'min':>12}{'max':>12}"
          f"{'med |m|/sqrt(v)':>18}{'t':>7}")
    print("  " + "-" * 95)
    for label, st in rows:
        print(f"  {label:<34}{st['median_dp_over_p']:>12.4e}{st['min_dp_over_p']:>12.4e}"
              f"{st['max_dp_over_p']:>12.4e}{st['median_mhat_over_sqrt_vhat']:>18.4e}"
              f"{int(st['adam_step_count'][0]):>7}")
    print("  (median/min/max are per-tensor ||dp||/||p|| over the group's tensors;")
    print("   ||p|| is the post-step norm, which differs from the pre-step one by")
    print("   at most the ratio itself, i.e. <1e-3 here.)")

    clipemb_med = [st["median_dp_over_p"] for lbl, st in rows if "clipemb" in lbl
                   and "readout" in lbl]
    mean_med = sum(clipemb_med) / len(clipemb_med)
    verdict = "FALSIFIED" if mean_med < 1e-4 else "CONFIRMED"
    print(f"\n  H2 {verdict}: clipemb readout median ||dp||/||p|| per seed = "
          f"{', '.join(f'{v:.4e}' for v in clipemb_med)}  (mean {mean_med:.4e}) "
          f"vs the 1e-4 falsification threshold")
    return mean_med, verdict


def embedding_dc_drift(seed, suffix, init_mode, scale, obs, mu):
    """Decompose what the trainable embedding did to the centring invariant.

    `centered` initialises `b := -(W @ mu)`, i.e. it makes the embedding's output
    at the mean observation exactly zero. Both W and b are TRAINABLE, so that
    invariant is a starting point, not a constraint -- §12's limitations section
    says as much and explicitly records "whether it adapts was NOT verified" as an
    open question. This measures it.

    Reported per stage:
      |W|, |b|          -- have the tensors merely grown, or moved?
      |W@mu + b|        -- the RESIDUAL DC, zero at a centred init.
      |W@mu|            -- the DC an UNCENTRED layer with the same W would have,
                           so the ratio is "how much of the centring has decayed",
                           0.0 = still perfectly centred, 1.0 = as bad as legacy.
      membrane offset   -- the frozen per-unit steady-state offset that residual
                           DC produces: W_in @ (W@mu + b) / (1 - beta), the exact
                           quantity §12 measured at std 0.943583 for the legacy
                           init against a firing threshold of 1.0. This is a
                           LINEARISED input-driven estimate: it omits the frozen
                           recurrent TT term, so it explains the silent fraction's
                           direction and magnitude but does not equal it.
      AC std            -- std of W @ (obs - mu) over the fixture, i.e. the
                           INFORMATIVE part of the drive, for contrast with the DC.

    THE `/(1 - beta)` OFFSET FACTOR HERE IS LIF-SPECIFIC AND STAYS THAT WAY. This
    section reads the v1 `checkpoints/reservoir_seed{s}_{clip,clipemb}` runs by
    hardcoded path, and every one of them is a LIF run -- the resonate-and-fire
    neuron model postdates them by two generations. Under a rotated pole the
    standing offset is scaled by the REAL PART of the complex DC gain instead
    (docs/EXPERIMENT_LOG.md §23.10(b)); `analysis/reservoir_health.dc_offset_factor`
    is the generalised version, and it is the one to reach for if this measurement
    is ever pointed at an rf run.
    """
    torch.manual_seed(seed)
    model, _ = build_model("reservoir", seed=seed, embed_init_mode=init_mode,
                           embed_scale=scale)
    w_in = model.reservoir.W_in
    beta = float(model.reservoir.lif.beta)
    rows = []
    for stage, ckpt in (("init", None), ("100096", "step_100096.pt"),
                        ("200192", "step_200192.pt"), ("300032", "step_300032.pt")):
        if ckpt is None:
            weight = model.embedding.weight.detach().clone()
            bias = model.embedding.bias.detach().clone()
        else:
            sd = load_ckpt(os.path.join(run_dir(seed, suffix), ckpt))["model"]
            weight, bias = sd["embedding.weight"], sd["embedding.bias"]
        dc = weight @ mu + bias
        uncentred = (weight @ mu).norm().item()
        offset = (w_in @ dc) / (1.0 - beta)
        rows.append((stage, weight.norm().item(), bias.norm().item(), dc.norm().item(),
                     uncentred, dc.norm().item() / uncentred, offset.std().item(),
                     float((offset < -1).float().mean()),
                     ((obs - mu) @ weight.T).std().item()))
    return rows


def section_h1_mechanism():
    print("\n" + "=" * 78)
    print("1b. WHY H1 FAILED -- the centring invariant decays because W drifts and b")
    print("    does not follow it. (b := -(W @ mu) holds only at step 0.)")
    print("=" * 78)
    obs = torch.as_tensor(np.load(REAL_OBS_PATH), dtype=torch.float32)
    from envs.mario_land_env import OBS_MEAN
    mu = torch.tensor(OBS_MEAN, dtype=torch.float32)
    print(f"\n  {'run':<22}{'stage':>8}{'|W|':>8}{'|b|':>8}{'|W@mu+b|':>11}"
          f"{'uncentred':>11}{'decayed':>9}{'offset std':>12}{'off<-1':>9}{'AC std':>9}")
    print("  " + "-" * 97)
    for suffix, init_mode, scale in (("clipemb", "centered", 3.0), ("clip", "legacy", 1.0)):
        for seed in SEEDS:
            rows = embedding_dc_drift(seed, suffix, init_mode, scale, obs, mu)
            for i, r in enumerate(rows):
                name = f"{suffix} seed{seed}" if i == 0 else ""
                print(f"  {name:<22}{r[0]:>8}{r[1]:>8.4f}{r[2]:>8.4f}{r[3]:>11.4f}"
                      f"{r[4]:>11.4f}{r[5]:>9.3f}{r[6]:>12.4f}{r[7]:>8.2%}{r[8]:>9.4f}")
            print()
    print("  'decayed' 0.0 = still perfectly centred, 1.0 = as uncentred as legacy.")
    print("  The legacy init's own offset std at step 0 is 0.94-1.13 (threshold 1.0),")
    print("  so a centred run whose offset std passes ~1.1 has spent its whole")
    print("  advantage -- and at embed_scale 3.0 the recovered DC is ~3x legacy's.")


def section_full_run_reference():
    print("\n" + "=" * 78)
    print("7. FULL-RUN REFERENCE -- where does the drift END? (v1 seed0, legacy+global,")
    print("    the only condition for which a complete 1,000,064-step run exists)")
    print("=" * 78)
    obs = torch.as_tensor(np.load(REAL_OBS_PATH), dtype=torch.float32)
    from envs.mario_land_env import OBS_MEAN
    mu = torch.tensor(OBS_MEAN, dtype=torch.float32)
    torch.manual_seed(0)
    model, _ = build_model("reservoir", seed=0)
    w_in, beta = model.reservoir.W_in, float(model.reservoir.lif.beta)
    stages = [("init", None)] + [(str(s), f"step_{s}.pt") for s in
                                 (100096, 200192, 300288, 400384, 500480, 600576,
                                  700672, 800768, 900864, 1000064)]
    print(f"\n  {'stage':>9}{'silent':>11}{'spike rate':>13}{'saturated':>11}"
          f"{'|W|':>9}{'|W@mu+b|':>11}{'offset std':>12}")
    print("  " + "-" * 76)
    for stage, ckpt in stages:
        if ckpt is not None:
            sd = load_ckpt(os.path.join(run_dir(0, None), ckpt))["model"]
            with torch.no_grad():
                model.embedding.weight.copy_(sd["embedding.weight"])
                model.embedding.bias.copy_(sd["embedding.bias"])
        weight = model.embedding.weight.detach()
        bias = model.embedding.bias.detach()
        s, rate, sat = silent_fraction(model, obs)
        dc = weight @ mu + bias
        offset = (w_in @ dc) / (1.0 - beta)
        print(f"  {stage:>9}{s:>10.4%}{rate:>13.6f}{sat:>10.4%}{weight.norm():>9.4f}"
              f"{dc.norm():>11.4f}{offset.std():>12.4f}")
    print("\n  Under `legacy` the SILENT SET is pinned from step 0 (§12's A4a nesting")
    print("  result) and the drift shows up as a runaway SPIKE RATE instead: 10x the")
    print("  healthy ~2% band by the end of the run, with saturated units appearing.")
    print("  The embedding drifts without bound in every cell -- neither shipped fix")
    print("  addresses that, and it is the reason H1's invariant could not hold.")


def section_baseline_arm_check():
    print("\n" + "=" * 78)
    print("8. THE UNTESTED SURFACE -- the pilot ran NINE RESERVOIR RUNS AND ZERO")
    print("    BASELINE RUNS, yet the control applies both knobs to both arms. The")
    print("    GRU arm passes its embedding through tanh, so the risk the pilot")
    print("    never probed is tanh saturation at embed_scale 3.0.")
    print("=" * 78)
    obs = torch.as_tensor(np.load(REAL_OBS_PATH), dtype=torch.float32)
    print(f"\n  {'baseline GRU config':<28}{'pre-tanh std':>14}{'|mean|':>10}"
          f"{'|z|>2':>9}{'|z|>2.65':>11}{'post-tanh std':>15}")
    print("  " + "-" * 87)
    for label, kw in (("legacy, 1.0  (v1)", {}),
                      ("centered, 3.0 (v2 planned)",
                       dict(embed_init_mode="centered", embed_scale=3.0)),
                      ("legacy, 3.0", dict(embed_scale=3.0)),
                      ("centered, 1.0", dict(embed_init_mode="centered"))):
        torch.manual_seed(0)
        model, _ = build_model("baseline", seed=0, **kw)
        with torch.no_grad():
            z = model.embedding(obs)
        print(f"  {label:<28}{z.std():>14.4f}{z.mean().abs():>10.4f}"
              f"{float((z.abs() > 2).float().mean()):>8.3%}"
              f"{float((z.abs() > 2.65).float().mean()):>10.3%}"
              f"{torch.tanh(z).std():>15.4f}")
    print("\n  |z|>2.65 is where tanh' has fallen below 1% of its value at 0. Centring")
    print("  SHRINKS the input (||obs-mu|| < ||obs||), which offsets most of the 3x")
    print("  gain, so the v2 config lands close to v1 rather than in saturation.")


def window_mean(recs, lo, hi, field="mean_extrinsic_reward"):
    vals = [r[field] for r in recs if lo <= r["update"] <= hi]
    assert vals, f"no records in updates {lo}..{hi}"
    return sum(vals) / len(vals), len(vals)


def section_h3():
    print("\n" + "=" * 78)
    print(f"3. H3 -- 2x2 factorial, mean_extrinsic_reward over updates "
          f"{WINDOW_START}-{PILOT_UPDATE} (DESCRIPTIVE)")
    print("=" * 78)
    table = {}
    for label, suffix, _, _, _ in CELLS:
        per_seed = []
        for seed in SEEDS:
            recs = read_log(seed, suffix)
            m, n = window_mean(recs, WINDOW_START, PILOT_UPDATE)
            per_seed.append(m)
        table[label] = per_seed
    base = []
    for seed in SEEDS:
        recs = read_log(seed, None, "baseline")
        m, _ = window_mean(recs, WINDOW_START, PILOT_UPDATE)
        base.append(m)
    table["BASELINE GRU (v1, reference)"] = base

    print(f"\n  {'cell':<32}{'seed0':>12}{'seed1':>12}{'seed2':>12}{'mean':>12}{'sd':>12}")
    print("  " + "-" * 92)
    for label, vals in table.items():
        sd = statistics.stdev(vals)
        print(f"  {label:<32}" + "".join(f"{v:>12.6f}" for v in vals)
              + f"{sum(vals) / 3:>12.6f}{sd:>12.6f}")

    m = {lbl: sum(v) / 3 for lbl, v in table.items()}
    a = m["global   + legacy  (v1)"]
    clip_only = m["per-group+ legacy  (clip)"] - a
    emb_only = m["global   + centered (emb)"] - a
    both = m["per-group+ centered (clipemb)"] - a
    print(f"\n  main effect of per-group clipping alone : {clip_only:+.6f}")
    print(f"  main effect of centered init alone      : {emb_only:+.6f}")
    print(f"  effect of both together                 : {both:+.6f}")
    print(f"  additive prediction (sum of the two)    : {clip_only + emb_only:+.6f}")
    print(f"  interaction (both - additive prediction): {both - (clip_only + emb_only):+.6f}")
    print("\n  3 seeds at 30% of a run CANNOT support an arm claim. No p-value is")
    print("  computed here and none should be quoted: with n=3 the per-seed spread")
    print("  above is the honest summary, and it is printed for that reason.")
    return table, (clip_only, emb_only, both)


def section_instability():
    print("\n" + "=" * 78)
    print(f"4. GRADIENT INSTABILITY -- per-group PRE-clip norms, last {TREND_WINDOW} updates")
    print(f"    (MAX_GRAD_NORM = {MAX_GRAD_NORM}; these are the norms BEFORE clipping)")
    print("=" * 78)
    lo = PILOT_UPDATE - TREND_WINDOW + 1
    print(f"\n  {'run':<26}{'group':<11}{'median':>13}{'p95':>13}{'max':>13}"
          f"{'frac > 0.5':>12}")
    print("  " + "-" * 88)
    for suffix in ("clipemb", "clip"):
        for seed in SEEDS:
            recs = [r for r in read_log(seed, suffix) if r["update"] >= lo]
            for group in ("embedding", "readout"):
                vals = [r["grad_norm_groups"][group] for r in recs]
                frac = sum(1 for v in vals if v > MAX_GRAD_NORM) / len(vals)
                print(f"  {f'{suffix} seed{seed}':<26}{group:<11}{quantile(vals, 0.5):>13.4e}"
                      f"{quantile(vals, 0.95):>13.4e}{max(vals):>13.4e}{frac:>11.2%}")
            print()
    print("\n  THE CLIP COEFFICIENT the readout actually receives, over all "
          f"{PILOT_UPDATE} updates.")
    print("  This is the statistic H2 turns on. Adam is invariant to a CONSTANT")
    print("  rescaling of the gradient but NOT to a time-varying one: a coefficient")
    print("  that swings by orders of magnitude sets sqrt(v_hat) (beta2=0.999, long")
    print("  memory) from the rare large updates while m_hat (beta1=0.9) tracks the")
    print("  typical one, and the ratio collapses. A coefficient that is small but")
    print("  STEADY is harmless. Under `global` the readout shares the embedding's")
    print("  coefficient; under `per-group` it gets its own.")
    print(f"\n  {'run / rule':<26}{'median coeff':>15}{'min coeff':>13}"
          f"{'max/median':>13}{'median/min':>13}")
    print("  " + "-" * 80)
    for seed in SEEDS:
        for suffix in ("clipemb", "clip"):
            recs = read_log(seed, suffix)
            coeffs = [min(1.0, MAX_GRAD_NORM / (r["grad_norm_groups"]["readout"] + 1e-6))
                      for r in recs]
            med = statistics.median(coeffs)
            print(f"  {f'{suffix} seed{seed} per-group':<26}{med:>15.4e}{min(coeffs):>13.4e}"
                  f"{max(coeffs) / med:>13.4e}{med / min(coeffs):>13.4e}")
        recs = [r for r in read_log(seed, None) if r["update"] <= PILOT_UPDATE]
        coeffs = [min(1.0, MAX_GRAD_NORM / (r["grad_norm"] + 1e-6)) for r in recs]
        med = statistics.median(coeffs)
        print(f"  {f'v1 seed{seed} GLOBAL':<26}{med:>15.4e}{min(coeffs):>13.4e}"
              f"{max(coeffs) / med:>13.4e}{med / min(coeffs):>13.4e}\n")

    print(f"  Same statistics over ALL {PILOT_UPDATE} updates, clipemb only:")
    print(f"  {'run':<26}{'group':<11}{'median':>13}{'p95':>13}{'max':>13}"
          f"{'frac > 0.5':>12}")
    print("  " + "-" * 88)
    for seed in SEEDS:
        recs = read_log(seed, "clipemb")
        for group in ("embedding", "readout"):
            vals = [r["grad_norm_groups"][group] for r in recs]
            frac = sum(1 for v in vals if v > MAX_GRAD_NORM) / len(vals)
            print(f"  {f'clipemb seed{seed}':<26}{group:<11}{quantile(vals, 0.5):>13.4e}"
                  f"{quantile(vals, 0.95):>13.4e}{max(vals):>13.4e}{frac:>11.2%}")


def section_nan_inf():
    print("\n" + "=" * 78)
    print("5. NaN / Inf SCAN -- logged metrics and checkpointed tensors")
    print("=" * 78)
    numeric = ("mean_reward", "mean_extrinsic_reward", "policy_loss", "value_loss",
               "entropy", "total_loss", "grad_norm")
    clean = True
    for suffix in ("clipemb", "emb", "clip"):
        for seed in SEEDS:
            bad = []
            for r in read_log(seed, suffix):
                for k in numeric:
                    if not math.isfinite(float(r[k])):
                        bad.append((r["update"], k, r[k]))
                groups = r.get("grad_norm_groups")
                if groups:
                    for g, v in groups.items():
                        if not math.isfinite(float(v)):
                            bad.append((r["update"], f"grad_norm_groups.{g}", v))
            clean &= not bad
            print(f"  log  reservoir_seed{seed}_{suffix:<9} "
                  f"{'clean' if not bad else f'{len(bad)} NON-FINITE: {bad[:5]}'}")
    for seed in SEEDS:
        for ckpt in ("step_100096.pt", "step_200192.pt", "step_300032.pt"):
            path = os.path.join(run_dir(seed, "clipemb"), ckpt)
            c = load_ckpt(path)
            bad = finite_scan(c["model"], "model") + finite_scan(c["optimizer"], "optimizer")
            clean &= not bad
            print(f"  ckpt reservoir_seed{seed}_clipemb/{ckpt:<16} "
                  f"{'clean' if not bad else f'NON-FINITE: {bad[:5]}'}")
    print(f"\n  overall: {'no NaN/Inf anywhere' if clean else 'NON-FINITE VALUES PRESENT'}")
    return clean


def section_trend():
    print("\n" + "=" * 78)
    print(f"6. REWARD TREND -- OLS on mean_extrinsic_reward, last {TREND_WINDOW} updates")
    print("=" * 78)
    lo = PILOT_UPDATE - TREND_WINDOW + 1
    print(f"\n  {'run':<26}{'slope/update':>15}{'fitted change':>16}"
          f"{'first half':>13}{'second half':>13}")
    print("  " + "-" * 83)
    out = {}
    for label, suffix, _, _, _ in CELLS:
        for seed in SEEDS:
            recs = [r for r in read_log(seed, suffix) if lo <= r["update"] <= PILOT_UPDATE]
            xs = [r["update"] for r in recs]
            ys = [r["mean_extrinsic_reward"] for r in recs]
            slope, _ = ols_slope(xs, ys)
            half = len(ys) // 2
            first, second = sum(ys[:half]) / half, sum(ys[half:]) / (len(ys) - half)
            tag = (suffix or "v1") + f" seed{seed}"
            out[tag] = slope
            print(f"  {tag:<26}{slope:>15.3e}{slope * TREND_WINDOW:>16.6f}"
                  f"{first:>13.6f}{second:>13.6f}")
        print()
    return out


def main():
    print("=" * 78)
    print("GameSpike v2 pilot diagnostic -- 9 interrupted runs, 3 seeds x 3 configs")
    print(f"  repo      : {REPO_ROOT}")
    print(f"  torch     : {torch.__version__}")
    print(f"  pilot stop: update {PILOT_UPDATE} = step {PILOT_UPDATE * 128}")
    print("  READ-ONLY: this module writes nothing and trains nothing.")
    print("=" * 78)
    section_config_sanity()
    h1 = section_h1()
    section_h1_mechanism()
    h2 = section_h2()
    section_h3()
    section_instability()
    clean = section_nan_inf()
    section_trend()
    section_full_run_reference()
    section_baseline_arm_check()
    print("\n" + "=" * 78)
    print("SUMMARY OF THE TWO PASS/FAIL HYPOTHESES")
    print("=" * 78)
    print(f"  H1  {h1[1]:<10} mean silent fraction {h1[0]:.4%} (threshold: >15% falsifies)")
    print(f"  H2  {h2[1]:<10} readout median ||dp||/||p|| {h2[0]:.4e} "
          f"(threshold: <1e-4 falsifies)")
    print(f"  NaN/Inf: {'none' if clean else 'PRESENT -- investigate before committing compute'}")
    print("  H3 is descriptive; see section 3. It licenses no arm claim.")


if __name__ == "__main__":
    main()
