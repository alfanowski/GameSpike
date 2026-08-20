# Phase 1 Results — the mandatory-control comparison

**Version: v1. This document covers the Phase 1 comparison exactly as specified and
pre-registered — no more than that.** A corrected comparison (per-group gradient
clipping, and the centred embedding initialisation, applied identically to both arms)
is running at the time of writing and will be appended to this document as v2. Nothing
here is written as if the story were finished, because it is not: §6 documents an
optimizer/clipping interaction, discovered after the runs completed, that this
comparison cannot separate from the architectural question it was built to answer.

Date: 2026-08-20
Author: Andrea Alfano ("Alfanowski")
Scope: `docs/DESIGN.md` §5 / §5.1 (the mandatory control), executed under the protocol
pre-registered in `docs/EXPERIMENT_LOG.md` §2 and §7.

---

## 1. Headline

**Under the pre-registered protocol, the frozen spiking reservoir loses to the
matched-parameter trained GRU baseline on the declared scoreboard
(`mean_extrinsic_return`).** It loses in every trained condition measured — both
checkpoint-selection rules, both recurrent-state regimes — by 6.17 to 7.72 points of
mean episode return, with exact two-sided permutation p-values between 0.000433 and
0.001635 and Cohen's *d* between −1.27 and −1.91. Every bootstrap 95% confidence
interval on the difference of means excludes zero, on the losing side.

This is `DESIGN.md` §5's question and this is its answer as measured. `README.md`
committed in advance to reporting it as such: *"the mandatory-control design exists
specifically so a negative result — the reservoir failing to beat a conventional
trained recurrent policy at the same parameter budget — is scientifically informative
rather than a silently discarded run."*

Two things follow, and neither of them softens the sentence above.

- The result is **conservative, not inflated**. §5 shows the episode-return scoreboard
  actually *flatters* the reservoir: normalised per step of experience the baseline is
  about six times better, because the reservoir's episodes run roughly six times longer
  and accumulate dense progress reward over more steps. The scoreboard understates the
  baseline's advantage.
- The result is **confounded, and the confound is a limitation of this result rather
  than a rebuttal of it**. §6 documents a measured optimizer/clipping interaction that
  trained the reservoir arm's readout far more slowly than the baseline's. The
  as-specified comparison therefore does not cleanly separate *"the frozen reservoir is
  a weaker feature extractor"* from *"the frozen reservoir's readout was barely trained
  at all"*. A corrected comparison is required, and is running. Until it lands, the
  measured answer to §5 stands as stated, with the confound stated next to it.

---

## 2. Experimental setup

Stated exactly, so that every number below has a stated provenance.

### 2.1 Training

| | value |
|---|---|
| arms | 2 (`baseline` = trained GRU, `reservoir` = frozen spiking reservoir) |
| training seeds per arm | 10, independent (0–9) |
| env steps per run | 1,000,064 |
| PPO updates per run | 7,813 (`rollout_len=128`; 7,813 × 128 = 1,000,064) |
| rollout collection | single process, one PyBoy instance per run |
| optimizer | Adam, `lr=3e-4` |
| gradient clipping | global `clip_grad_norm_`, `MAX_GRAD_NORM=0.5` |
| game | Super Mario Land (Game Boy), world 1-1 |

`--seed N` drives both arms' trainable initialisation and action sampling **and** the
reservoir arm's frozen weights, so the seeds vary both arms symmetrically and the
reservoir arm's ten seeds are ten genuinely different frozen reservoirs (verified
bitwise, `EXPERIMENT_LOG.md` §4).

### 2.2 Parameter parity

| arm | trainable parameters |
|---|---|
| baseline (GRU) | 132,715 |
| reservoir | 139,179 |
| ratio | 1.0487 |

Inside the ±10% band `tests/test_parameter_parity.py` enforces against the arms
`training/train.py:build_model` actually constructs. `DESIGN.md` §5.1 records why
matched *trainable-parameter count*, not matched hidden width, is the binding
requirement: the control exists to neutralise trained capacity, and the sibling
project's load-bearing finding is that a frozen reservoir's storable capacity is
bounded by the trained readout's parameter count.

### 2.3 Evaluation

`training/evaluate.py`, 30 episodes per checkpoint, `max_steps_per_episode=3000`,
evaluation seed 0 (episode *i* draws its actions from a private generator seeded
`0 + i`). Actions are sampled from the policy, never argmaxed — the environment is
deterministic, so a greedy policy would replay one identical episode 30 times and
report a spread of exactly zero, which is a fake measurement rather than a weak one.

The full matrix is **2 recurrent-state regimes × 3 checkpoint selections × 2 arms ×
10 seeds = 120 evaluations. All 120 succeeded.**

- **Regimes.** `continuous` — recurrent state initialised once per episode and never
  reset, i.e. the policy runs for up to 3,000 steps, more than 20× the horizon it ever
  saw a gradient over. `reset128` — recurrent state reset every 128 steps, matching the
  truncated-BPTT regime training actually used. `evaluate.py`'s docstring is explicit
  that neither regime alone is sufficient: reporting both is what separates *"this arm
  is better"* from *"this arm degrades more slowly once played past the horizon it was
  trained on"*, and memory horizon is precisely the axis on which a frozen 8192-dim
  reservoir and a trained 192-dim GRU are most likely to differ.
- **Selections.** `final` — the highest-step checkpoint (`step_1000064.pt`). `best` —
  the checkpoint with the highest mean **training** reward in the window ending at it.
  `init` — the untrained (`--steps 0`) reference checkpoint for the same arm and seed.

### 2.4 Unit of analysis: the training seed, never the episode

Each of the 120 evaluations is reduced to **one number per training seed** before any
statistic sees it, so every test below is n=10 versus n=10.

This is not a stylistic choice. `training/evaluate.py`'s own "WHAT THIS HARNESS CANNOT
TELL YOU" section states the reasoning it is taken from: the spread the harness reports
is *policy-sampling variance and nothing else* — the emulator, the boot sequence and
`reset()` are all deterministic, so there is no environment variance, no start-state
variance and no opponent variance in the error bars at all. Training-run variance, which
in deep RL usually dominates, is invisible to it. Averaging more episodes shrinks the
wrong error bar: it makes a single checkpoint's number more precise without making the
arm comparison any more trustworthy. The harness is therefore the per-checkpoint
instrument for this experiment, not the experiment; the experiment is the comparison
*across* independently-trained seeds, which is what §3 reports.

`analysis/aggregate_results.py` enforces this by construction — it reduces every
checkpoint to one mean before any statistic is computed.

### 2.5 `best` is selected on training reward, never on evaluation reward

The `best` rule ranks checkpoints by mean **training** reward. Selecting on the
evaluation measure would pick the winner using the same data the comparison is then
scored on, which biases the reported gap upward — the selected checkpoint would be the
one whose evaluation noise happened to be favourable. `analysis/aggregate_results.py`
calls this out in its own section header as the one rule that must never be violated.
It is worth noting explicitly that this rule is what makes the `best` row below
informative: the reservoir arm loses under `best` as well as under `final`, so its loss
is not an artefact of having been scored at a moment its own training curve was down.

### 2.6 Statistics

Exact, not asymptotic:

- **Permutation test** — full enumeration of all C(20,10) = 184,756 label assignments,
  two-sided on the difference of means. No sampling, no normal approximation. The
  smallest p-value this design can return is 2/184,756 = 0.000011; a reported
  0.000011 means "at the floor", not "vanishingly small".
- **Mann-Whitney** — exact null distribution.
- **Bootstrap 95% CI on the difference of means** — percentile bootstrap, 20,000
  resamples, `numpy.random.default_rng(0)`, no distributional assumption.

At n=10 per group this matters. A t-test's p-value and a normal-approximation
Mann-Whitney's p-value are both statements about a limiting distribution that ten
observations do not reach, and the interesting p-values here are within about two
orders of magnitude of the design's own resolution floor. An exact test states the
resolution instead of extrapolating past it.

### 2.7 Provenance and hardware

- Evaluation was run from a `git worktree` pinned to commit **`64839a9`**, so the
  120 result files were produced by one fixed version of the harness while the main
  checkout continued to move.
- Hardware: **MacBook Air M4, 10 CPU cores, 16GB unified memory, no GPU.** **No cloud
  compute was used**, consistent with `DESIGN.md` §2's zero-budget constraint.
- Environment: Python 3.12.12, torch 2.13.0, pyboy 2.7.0, snntorch 1.0.0,
  gymnasium 1.3.0, numpy 2.5.2 (`EXPERIMENT_LOG.md` §1).
- `mean_extrinsic_return` is the scoreboard throughout. `mean_combined_return`
  (extrinsic + novelty subsidy) is diagnostic only and the arms are never ranked on it —
  curiosity is an exploration subsidy, not a score.

---

## 3. Primary result

Differences are **reservoir minus baseline**; a negative number favours the baseline.
Means and standard deviations are over the 10 training seeds of each arm.

| condition | reservoir mean ± SD (n=10) | baseline mean ± SD (n=10) | diff | exact permutation p | Cohen's d |
|---|---|---|---|---|---|
| final, continuous | 28.4169 ± 7.0593 | 36.1335 ± 3.4082 | −7.7167 | 0.000996 | −1.3922 |
| final, reset128 | 29.5721 ± 4.5113 | 35.7429 ± 2.5147 | −6.1708 | 0.001072 | −1.6897 |
| best, continuous | 29.2475 ± 7.5231 | 36.2665 ± 2.0513 | −7.0190 | 0.001635 | −1.2730 |
| best, reset128 | 28.8413 ± 5.0347 | 36.2360 ± 2.1198 | −7.3948 | 0.000433 | −1.9144 |
| init (untrained), continuous | 9.9625 ± 12.7567 | 8.1231 ± 2.3645 | +1.8394 | 0.659367 | +0.2005 |
| init (untrained), reset128 | 10.4835 ± 8.5046 | 7.8396 ± 2.7193 | +2.6440 | 0.363268 | +0.4188 |

Bootstrap 95% CIs on the difference of means (20,000 resamples, seed 0):

| condition | 95% CI on the difference |
|---|---|
| final, continuous | [−12.7665, −3.5839] |
| final, reset128 | [−9.2170, −3.1904] |
| best, continuous | [−12.1567, −2.9462] |
| best, reset128 | [−10.5504, −4.1608] |
| init, continuous | [−5.4257, 9.6017] |
| init, reset128 | [−2.5055, 7.9308] |

Four observations, in order of how much weight they carry.

1. **The direction is the same in all four trained conditions**, and every trained
   CI lies entirely below zero. The result does not depend on which regime is scored or
   which checkpoint-selection rule is applied.
2. **The reservoir arm's seed-to-seed spread is consistently larger** — SD 4.51 to 7.52
   against the baseline's 2.05 to 3.41. Whatever the reservoir arm learned, it learned
   it less reliably across seeds. That is visible in the CIs too: the reservoir-side
   conditions have the wider intervals.
3. **The `reset128` regime does not rescue the reservoir.** If the frozen 8192-dim
   reservoir's advantage were memory horizon, the matched-regime column is where it
   would show up least and the continuous column is where it would show up most; the
   gap is 7.72 in `continuous` and 6.17 in `reset128` at `final`, and 7.02 versus 7.39
   at `best`. There is no regime in this matrix where the reservoir wins.
4. **The untrained rows are the control, not a result** — see §4.1.

---

## 4. Two controls that make the result interpretable

A negative arm comparison is worth very little on its own. It could mean the arms
started unequal, or that the losing arm never learned anything at all and the comparison
is a comparison with a broken model. Both were measured, and neither is the case.

### 4.1 The untrained arms are statistically indistinguishable

The two `init` rows in §3 compare random-initialised, never-trained policies of the two
arms under exactly the same evaluation procedure: p = 0.659367 (continuous) and
p = 0.363268 (reset128), with Cohen's *d* of +0.2005 and +0.4188 and bootstrap CIs that
straddle zero in both regimes.

**The arms start equivalent.** The trained gap is therefore attributable to what was
learned during training, not to one arm having been handed a better starting point by
its initialisation. This is what makes §3's trained rows a statement about training
rather than about construction.

One honest caveat on these two rows: an absence of a detectable difference at n=10 is
not proof of equality. It bounds how large any initialisation asymmetry can plausibly
be, and the bound is loose — the CIs are [−5.4257, 9.6017] and [−2.5055, 7.9308],
which is wide next to a trained gap of about 7. What it rules out is a large
initialisation advantage in the *baseline's* favour that could have produced §3 on its
own; the untrained point estimates in fact lean slightly the other way.

### 4.2 Both arms learn significantly above their own initialisation

Comparing each arm's `final` checkpoints against its own `init` checkpoints, same seeds,
`continuous` regime:

| arm | trained − untrained | Cohen's d | exact permutation p |
|---|---|---|---|
| baseline | +28.010 | +9.550 | 0.000011 |
| reservoir | +18.454 | +1.790 | 0.001526 |

Both p-values are two-sided exact permutation tests over the same 184,756 splits;
0.000011 is the floor (2/184,756), i.e. the baseline's trained checkpoints beat its own
untrained ones under *every* one of the 184,756 relabellings but the two that reproduce
the observed split.

**The reservoir arm is a working arm that loses, not a broken one.** It improves on its
own initialisation by 18.45 points at p = 0.001526. It is not stuck at random, it is not
diverged, and it is not producing degenerate output. It learns, and then it loses to an
arm that learned more (+28.01) and far more consistently (*d* = +9.55 against +1.79 — a
difference driven as much by the reservoir arm's seed-to-seed spread as by its mean).

---

## 5. The decomposition — the scoreboard flatters the reservoir

This is the single most informative measurement in the document, and it does not
appear in the pre-registered protocol: it fell out of reading the episode-length column
next to the return column.

`final` checkpoints, `continuous` regime, all figures means over the 10 training seeds
of each arm. Reward per step is computed per seed as
`mean_extrinsic_return / mean_episode_length` and then averaged over seeds.

| | episode return | mean episode length | reward per step |
|---|---|---|---|
| baseline trained | 36.134 | 314.9 | 0.11455 |
| reservoir trained | 28.417 | 1917.0 | 0.01921 |
| baseline untrained | 8.123 | 2825.8 | 0.00287 |
| reservoir untrained | 9.963 | 2491.1 | 0.00527 |

- Baseline / reservoir **reward-per-step ratio: 5.96×**, exact permutation
  p = 0.000011 (the floor).
- **Episode-length difference: p = 0.000011** (the floor).
- The same pattern holds in the matched-regime column: per step **0.11322 vs 0.01937,
  ratio 5.85×**; lengths **315.4 vs 1906.6**. It also holds under `best` selection,
  where the reservoir's episodes average 1288.2 (continuous) and 1266.9 (reset128)
  steps against the baseline's 313.6 and 315.2.

**The two arms did not learn worse and better versions of the same strategy. They
learned qualitatively different strategies.**

- The **baseline learned to move right quickly and die quickly**: about 315-step
  episodes at 0.11455 reward per step. It converts the dense rightward-progress reward
  at a high rate and its episodes end early, presumably because moving right fast in
  world 1-1 is also how you run into things.
- The **reservoir learned to survive without progressing**: about 1917-step episodes at
  0.01921 reward per step. It stays alive roughly six times longer and earns roughly
  six times less per step of that life. Note that both untrained arms survive *longer
  still* (2825.8 and 2491.1 steps) while earning almost nothing per step, so long
  episodes are the untrained default here, not an achievement — the baseline moved away
  from it and the reservoir arm moved away from it much less.

**Consequence for §1's headline: the episode-return scoreboard is the metric most
favourable to the reservoir among the ones available here.** Episode return is an
integral over the episode, so an agent that survives ~6× longer accumulates dense
progress reward over ~6× more steps and closes part of a per-step gap it never actually
closed. Normalised per step of experience the baseline is about six times better —
which is the same order as the gap visible in training reward throughout both runs:
measured over the ten runs of each arm, the baseline's mean per-update extrinsic
training reward is **5.82× the reservoir's over all 7,813 updates** and **5.33× on the
final decile**. The evaluation scoreboard's 1.27× gap in return
(36.134 / 28.417) and the ~6× gap per step are the same underlying difference viewed
through two normalisations, and the scoreboard is the flattering one.

**The headline is therefore conservative rather than inflated.** Any reader who prefers
a per-step normalisation gets a larger reservoir deficit, not a smaller one. This is
stated here rather than left for a reader to discover, because a metric that favours the
arm the author is testing is exactly the kind of thing that should be disclosed by the
author.

---

## 6. The confound — global gradient clipping, and what it does to Adam

Reported here as a **limitation of this result**, not as an excuse that neutralises it.
The measurement in §3 is what the pre-registered protocol produced and it stands.
What follows is why the protocol, as specified, cannot cleanly attribute that
measurement to the architecture.

### 6.1 What was measured

A root-cause diagnostic — **measurement only, no training, no new runs** — on
`checkpoints/reservoir_seed0/step_500480.pt`, reproducing one PPO update exactly.
The full record lives in `training/train.py`'s module docstring, which is where it was
written down at the time; it is summarised here.

**Gradients reaching the trainable embedding explode exponentially with replay-chain
length.** The reservoir arm backpropagates through `rollout_len` = 128 sequential steps
of a frozen spiking reservoir, through surrogate spike gradients:

| replay-chain length L | global pre-clip gradient norm, reservoir arm | baseline arm |
|---|---|---|
| 1 | 2.171 | 29.97 |
| 32 | 52.19 | 51.71 |
| 64 | 1.111e4 | — |
| 96 | 3.988e6 | — |
| 128 | **1.258e9** | 48.18 |

That is a per-step multiplier of about 1.22. The readout's **own** gradient grows only
about √L over the same sweep (1.5 → 8.9), and the baseline arm is flat. **The explosion
is one path, not "the loss".**

**One tensor pair carries the entire global norm.** `embedding.weight` +
`embedding.bias` — **416 parameters, 0.3% of the trainable budget** — carry
**100.0000%** of the global gradient norm. The 29 readout/head tensors
(**138,763 parameters, 99.7% of the budget**) contribute about **5e-15%**.

**A single global clip therefore transfers the exploding group's coefficient onto the
group that is not exploding.** One `clip_grad_norm_` over the whole trainable list
computes a clip coefficient of **3.976e-10** from the exploding 0.3%, and applies it to
the 99.7% that is not exploding, taking the readout's post-clip gradient norm to
**3.52e-09**.

### 6.2 Why Adam does not absorb it

The obvious objection is that Adam is scale-invariant, so a rescaled gradient should not
matter. **Adam is invariant to a *constant* rescaling of the gradient, not to a
*time-varying* one**, and this rescaling is strongly time-varying: the clip
coefficient's **max/median ratio over 1,000 updates is 2.63e5**.

That gap interacts with Adam's two different memories. `sqrt(v_hat)` (beta2 = 0.999,
long memory) ends up set by the rare non-exploding updates, while `m_hat`
(beta1 = 0.9, short memory) tracks the typical, heavily-suppressed one. The measured
consequence: the readout's median `|m_hat| / sqrt(v_hat)` collapses to **7.475e-04**
against **1.346e-01** on the baseline GRU. In effect the readout is frozen.

Three checks were run to rule out the alternative explanations, and all three are
reported:

- **It is not an eps floor.** A single-step counterfactual with identical gradients and
  restored Adam state: raising Adam's eps from 1e-8 to 1e-12 gives only **1.11×**.
- **It is the shared clip coefficient.** The same counterfactual with per-group
  clipping raises the readout's median `||Δp|| / ||p||` from **1.9034e-05** to
  **6.4186e-03** — a factor of **337×**.
- **It is not a numerical failure.** Zero NaN and zero Inf anywhere, and the clipping
  itself is correctly implemented — the measured post-clip norm is
  **0.4999999713305232** against `MAX_GRAD_NORM = 0.5`.

Two more numbers put the effect at run scale rather than one-step scale:

- Median `||Δp|| / ||p||` across all trainable parameters: reservoir **1.911e-05** vs
  baseline **4.273e-04** — **the baseline's parameters move 22.35× more per update**.
- Per-arm median `grad_norm` over the full ten runs each: baseline **34.13 to 42.74**;
  reservoir **8.07e7 to 3.13e9**. This is not a seed-0 curiosity; every reservoir run
  on disk is in this regime.

### 6.3 What this does and does not license

**Stated plainly: the as-specified comparison does not cleanly separate "the frozen
reservoir is a weaker feature extractor" from "the frozen reservoir's readout was
trained far more slowly by an optimizer/clipping interaction".** Both are consistent
with §3. An arm comparison run under the global rule is partly measuring which arm's
optimiser survived its own clipping.

What it does **not** license:

- It is not a reason to withdraw §3. §3 is what the pre-registered protocol measured,
  under a clipping rule that was chosen before any result was seen and applied
  identically to both arms.
- It is not a prediction that the corrected comparison reverses the result. §6.4's
  pilot signal points the other way — the fix helps, and does not come close to closing
  the gap.
- It is not the reservoir arm's only handicap. §7 documents an independent construction
  defect in the reservoir's *input*, and §5 shows the two arms converged on different
  strategies rather than different quality of the same strategy.

`--grad-clip-mode per-group` now exists in `training/train.py` and clips each parameter
group to `MAX_GRAD_NORM` separately, so an exploding group cannot suppress a
non-exploding one. It is applied **identically to both arms** — the baseline GRU does
not need it, but a treatment only one arm receives is not a control. The default stays
`global` and bit-identical, because the 20 completed runs and 200 checkpoints this
document is built from have to remain exactly reproducible.

### 6.4 Preliminary paired pilot — labelled preliminary, and it means it

**This is not a result. It is 3 seeds at 425 PPO updates = 54,400 env steps, i.e. 5.4%
of a full run**, measured while the corrected runs are still executing. Mean training
extrinsic reward over the last 20% of those updates (the last 85 updates), averaged over
seeds 0–2:

| configuration | mean last-20% training extrinsic reward |
|---|---|
| reservoir, original (`global` clipping) | 0.0145 |
| reservoir, `per-group` clipping | 0.0308 |
| baseline reference, same window | 0.0847 |

**The fix is worth about 2.1×, and every one of the three seeds improved**
(0.0153 → 0.0343, 0.0226 → 0.0357, 0.0056 → 0.0224). Against a baseline reference of
0.0847 in the same window, per-group clipping recovers roughly half of the shortfall in
early training and does not close it.

Read that as a direction, not a magnitude. 5.4% of a run is the part of a learning curve
where the arms are furthest from their eventual behaviour, three seeds is below the ten
this project insists on for an arm claim, and the corrected full runs are what v2 will
report.

---

## 7. Reservoir construction findings

A prior and more uncomfortable question was pre-registered in `EXPERIMENT_LOG.md` §11
before any of it was measured: **is the frozen reservoir this experiment measures
actually a well-constructed reservoir at all?** If it is not, §3 is not a test of
"frozen spiking reservoir vs. trained GRU" but of "one particular badly-calibrated
frozen reservoir vs. trained GRU" — a much weaker claim, and one that has to be made
explicitly. The full ledger entry is `EXPERIMENT_LOG.md` §12; this section is the
summary, not a duplicate.

**The answer is that the reservoir as run in §3 is badly calibrated, and the defect was
in a place nobody had pre-registered: the input to the reservoir, not the reservoir.**

### 7.1 The observation is DC-dominated, and the LIF neuron amplifies the useless part

Measured over 6,000 real observation steps (3,000 under the trained policy, 3,000 under
a uniform-random policy, pooled):

- **77.70% of the observation's energy is its own mean** (‖E[obs]‖² = 1.331336 against
  E‖obs‖² = 1.713384). The real dimensions are mostly non-negative with large means —
  the level timer, lives, powerup state and the on-ground flag are near-constant or
  slowly drifting, and three slots are hardcoded zero.
- Consequently **76.11% of the reservoir's input-current variance is DC.**
- With `beta = 0.9`, a constant input integrates to steady state with gain
  **1/(1−β) = 10.0**, while a zero-mean fluctuating input accumulates only to
  **1/√(1−β²) = 2.2942** — a **4.3589× amplification favouring the component that
  carries no information**.
- Because `W_in` is frozen, the result is a **frozen per-unit membrane offset**: std
  **0.943583** across units, range **[−3.5080, +3.4847]**, against a firing threshold
  of 1.0. Measured directly: **14.93% of units sit permanently below −threshold
  (silent forever, whatever the input does)** and **14.50% sit permanently above it
  (saturated)**.

### 7.2 The fix costs zero parameters, and only works as a pair

The embedding is **linear**, so `W(obs − μ) ≡ W·obs + (−W·μ)` identically (verified
numerically to a maximum absolute difference of 5.96e-08). Centring the input is
therefore *exactly* the bias initialisation `embedding.bias := −(W·μ)`. It costs **zero
new parameters** (the bias already existed), changes no tensor shapes, and leaves the
bias trainable, so it is a starting point rather than a constraint. **Parameter parity
is unchanged**, which is what makes it admissible in this comparison at all.

Independently reproduced in `tests/test_embedding_centering.py` against the committed
6,000-step observation fixture, 8-seed means:

| init | silent-unit fraction | mean spike rate | saturated units |
|---|---|---|---|
| `legacy`, scale 1.0 | 45.5917% | 0.022551 | 0 |
| `legacy`, scale 3.0 | 30.3329% | 0.117302 | 0 |
| `centered`, scale 1.0 | 65.9454% | 0.000474 | 0 |
| **`centered`, scale 3.0** | **2.0523%** | **0.018482** | **0** |

**Centring alone is worse than doing nothing** — 65.9454% silent at scale 1.0, because
removing the DC drive without replacing it starves the reservoir. `legacy` at scale 3.0
buys firing by brute force and overshoots the healthy ~2% band five-fold. **The bias and
the gain are a fix only together**, which is why both knobs shipped
(`--embed-init-mode`, `--embed-scale`), both default to the historical behaviour, and
both are applied identically to the two arms — input centring is a generic
initialisation correction, not a reservoir-specific advantage, and a treatment only one
arm receives is not a control.

**A synthetic surrogate does not reproduce the effect.** An i.i.d. surrogate matching
the measured per-dimension means and standard deviations gives **24.1211% silent
against 1.6602% on real data**. Real observations are temporally correlated, and a leaky
integrator at β = 0.9 accumulates that low-frequency energy at gain up to 10, which an
i.i.d. surrogate has none of. **Calibrating a leaky integrator against i.i.d. synthetic
inputs is invalid in principle, not merely imprecise** — which is why the original
calibration's `# KNOWN GAP:` comment in `models/policy_value_reservoir.py` mattered
considerably more than it looked like it did.

### 7.3 Dead readout columns shrink monotonically and are perfectly nested

Columns of the readout's `in_proj.weight` whose Adam `exp_avg_sq` is exactly zero have
never received a gradient. Across the ten checkpoints of `reservoir_seed0`:

| step | dead columns | % of 8192 |
|---|---|---|
| 100,096 | 2,147 | 26.2085% |
| 500,480 | 1,329 | 16.2231% |
| 1,000,064 | **865** | **10.5591%** |

`dead(t+1) ⊂ dead(t)` at every one of the 9 checkpoint transitions, with
`newly_dead = 0` at all 9 — **no column ever dies after training starts; columns only
wake up.** At the final checkpoint the dead budget is 865 × 16 = **13,840 parameters,
9.9440% of the trainable budget**.

The predicted mechanism holds in the strong direction — **`dead \ silent` = 0, no
exceptions**: every dead column belongs to a unit that never fires, which rules out an
optimizer/clipping pathology as a *source* of dead columns. The predicted **set
equality does not hold**: Jaccard(dead, silent) is only **0.2325**, rising to **0.6600**
against units silent under all 11 sampled embeddings, because a unit only needs to have
fired once, ever, to have received gradient. The "one root cause" framing is right about
the mechanism and wrong about the arithmetic.

### 7.4 Three pre-registered hypotheses were disconfirmed

Reported as such, in the same detail as the one that held, per the ledger's own
p-hacking guard.

- **H4c (seed selection) — disconfirmed.** With the embedding fixed and only the
  reservoir seed varied, the silent fraction spans **2.56 percentage points**
  (28.27%–30.84%). Selecting among frozen reservoirs on firing-rate health would be
  selecting among candidates that differ by less than three points on the criterion, next
  to the ~44-point effect §7.1 identifies. `EXPERIMENT_LOG.md` §4's "seeds genuinely
  produce different frozen reservoirs" remains true bitwise and is **dynamically
  irrelevant**.
- **H5 (spectral radius) — disconfirmed, with a proof rather than a null result.**
  Normalized entanglement entropy is **exactly invariant** to `spectral_radius`. On the
  tensor-train path with `tt_core_std=None`, `spectral_radius` enters construction only
  through a single derived scalar that multiplies every i.i.d. Gaussian core by the same
  number, and normalized entanglement entropy is computed from the *normalized* Schmidt
  spectrum, in which a global rescaling cancels. **Multiplying all cores by 1000 changes
  S̄ by 2.8e-11.** The ledger's own mechanism note anticipated a weak dependence via a
  1/8 exponent; the truth is stronger — the dependence is zero. Separately, the silent
  fraction moves *opposite* to the hypothesis (53.11% at radius 0.7 against 43.96% at
  1.05), because the recurrent drive is part of what pushes hyperpolarised units back
  toward threshold.
- **H6 (TT-rank) — disconfirmed.** Sweeping `tt_rank` over {4, 8, 16, 32}, three seeds
  each, S̄ spans only **0.96221–0.99596**. There is no order-to-chaos transition. The
  reason is visible in the Schmidt spectrum, which is near-flat (0.16881 … 0.09403
  against 0.125 for a perfectly uniform 8-dimensional bond) — exactly what i.i.d.
  Gaussian cores generically produce.

**Conclusion, stated as strongly as the evidence supports: no construction that keeps
i.i.d. Gaussian cores can reach Sato et al.'s productive band S̄ ∈ [0.1, 0.5].**
Reaching it needs structured cores, which is a different construction rather than a knob
on this one.

### 7.5 The pre-registered falsification condition was untestable as formulated

`EXPERIMENT_LOG.md` §11's binding falsification condition reads: *"If entanglement
entropy CAN be moved into S̄ ∈ [0.1, 0.5] but downstream task performance does not
improve, that resolves the sibling project's open question NEGATIVELY."*

**Its antecedent never occurs.** §7.4 establishes that entanglement entropy cannot be
moved into that band by either pre-registered knob — provably so in H5's case. A
conditional whose antecedent is false is not evidence about its consequent.

**The sibling project's open question — whether the entanglement-entropy diagnostic
predicts task performance for a spiking substrate — therefore remains OPEN, resolved in
neither direction by this work.** This is recorded as a formulation failure of the
pre-registration: the condition was written assuming the manipulation would work, and a
pre-registration that only produces a verdict when its manipulation succeeds is not a
complete pre-registration. It is not being reframed as a tuning failure and it is not
being quietly dropped.

### 7.6 What §7 does and does not do to §3

It does **not** invalidate §3. §7's measurements are construction diagnostics; none of
them is a task-performance result, and the pre-registered condition that would have
connected the two was untestable (§7.5). Nothing in §7 licenses a claim about the arm
comparison.

It does **weaken the generality of §3's claim**, in exactly the way §11 of the ledger
warned it would. The reservoir evaluated in §3 had roughly 45% of its units silent under
real observations and about 15% of them permanently silent by construction. **§3 is a
measurement of one particular, measurably badly-calibrated frozen reservoir against a
trained GRU**, and it should be quoted that way.

---

## 8. Efficiency — a first-class result, reported in both directions

Measured with `/usr/bin/time`, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, one thread per
run, parallelism obtained across OS processes rather than within one (torch intra-op
threading bought only ~4% on the baseline and ~10% on the reservoir at this model size,
which is not worth the cross-process contention).

| | single-run | 4-way parallel (aggregate) |
|---|---|---|
| baseline | 918 env-steps/s | 3,030 env-steps/s |
| reservoir | 371 env-steps/s | 1,098 env-steps/s |

**The reservoir arm is about 2.5× slower per env step at matched trainable-parameter
count.** That is a cost of the architecture, and it holds regardless of which arm wins
the return comparison.

Checkpoint size, both including Adam optimizer state:

| arm | final checkpoint |
|---|---|
| baseline | 1.60 MB |
| reservoir | 2.83 MB (1.77×) |

The difference is the frozen buffers — chiefly `W_in`, plus the four TT cores — which
still have to be stored despite never receiving a gradient.

**What is not claimed here.** A tensor-train recurrent operator's compression ratio
against a dense 8192×8192 matrix was **not measured in this phase**, so **no compression
figure is claimed in this document**. The 1.77× above is a whole-checkpoint measurement
including optimizer state, and it is not a statement about TT compression.

---

## 9. Limitations

Specific, and none of them discovered after the fact except where said so.

**Scope of the measurement.**

- **One game, one level.** Super Mario Land, world 1-1. Nothing here generalises to
  another level of the same game without measurement, let alone another game.
- **One hyperparameter set, shared across both arms.** `lr = 3e-4`,
  `MAX_GRAD_NORM = 0.5`, `rollout_len = 128`, single-epoch PPO updates without advantage
  normalisation or a learning-rate schedule. The shared set is what makes the control
  fair; it is also a set that was chosen for the baseline's architecture and never tuned
  for the reservoir's. Ablation A1 in the ledger was pre-registered to test exactly this
  and has not been run.
- **One reservoir configuration** — `reservoir_size = 8192`, `tt_rank = 8`,
  `tt_n_cores = 4`, `beta = 0.9`, threshold 1.0, `legacy` embedding init — and §7 shows
  it is a badly calibrated one.
- **1,000,064 steps, and the two arms were not equally converged at that point.**
  Measured over the ten runs of each arm, mean per-update extrinsic training reward
  changes by **+0.58% from the fifth decile to the tenth on the baseline** and by
  **+13.33% on the reservoir**. The baseline had flattened well before the budget; the
  reservoir had not clearly flattened. A longer budget could narrow the gap, and this
  experiment does not bound by how much.

**Scope of the evaluation.**

- **The environment is deterministic.** PyBoy, the boot sequence and `reset()` are all
  deterministic and every episode starts from a bit-identical state, so evaluation
  carries **no environment variance at all** — no start-state variance, no opponent
  variance. The only stochasticity is the policy's own action sampling.
- **30 episodes per checkpoint therefore measure policy-sampling variance only.** They
  do not, and cannot, widen the error bar that matters. That is why the unit of analysis
  is the training seed (§2.4).
- **n = 10 seeds per arm.** Adequate for the observed effect sizes — *d* between −1.27
  and −1.91, with every trained CI clear of zero — and above the 3–5 seeds common in
  published deep-RL work, but small in absolute terms. The design's own resolution floor
  is p = 2/184,756.
- **Both arms were scored in the `continuous` regime, which neither was trained in.**
  This is why `reset128` is reported alongside it rather than instead of it. It is a
  real uncontrolled variable, disclosed rather than left implicit, and it sits directly
  on top of the memory-horizon axis where the two architectures most plausibly differ.

**Scope of the observation.**

- **The effective observation is narrower than its 12 dimensions suggest.** Three of the
  twelve slots (9–11) are documented reserved zeros, wired for enemy-relative features
  that a later plan adds. One more (`lives`) is near-constant under a trained policy.
  Both arms see the same observation, so this does not bias the comparison, but it does
  bound what either arm could have learned.

**Scope of the analysis.**

- **The gradient-clipping confound of §6**, which is the reason this document is v1 and
  not final.
- **Multiple comparisons, disclosed as pre-registered.** `EXPERIMENT_LOG.md` §8
  committed in advance to disclosing this rather than quietly ignoring it: six ablations
  were pre-registered against one 10-seed main comparison, and §3's six condition rows
  are themselves not independent — they are six views of the same 20 training runs.
  A per-row p-value is not a family-wise one. For what it is worth, and derived directly
  from the table: all four trained rows sit below 0.05/6 = 0.008333, so they survive a
  Bonferroni correction over the six rows; the two untrained rows do not come close to
  significance under any correction.
- **§5's decomposition was not pre-registered.** It was found by reading the
  episode-length column, after the numbers existed. It should be read with the discount
  a post-hoc finding deserves. What justifies its prominence is that it points *against*
  the author's own arm and makes the headline more conservative, not less.
- **The per-group-clipping group-count asymmetry.** `group_trainable_parameters` buckets
  by top-level submodule, which yields **2 groups on the reservoir arm** (`embedding`,
  `readout`) and **4 on the baseline** (`embedding`, `gru`, `actor_head`,
  `critic_head`). Clipping each group to `MAX_GRAD_NORM` separately therefore permits a
  larger total update norm on the arm with more groups. The rule is applied identically
  to both arms and the grouping is discovered from the model rather than hardcoded, but
  the resulting group counts differ, and that is a **disclosed minor asymmetry in the
  corrected protocol** rather than a perfectly symmetric treatment.

---

## 10. What this does and does not tell you

**What it tells you.** On Super Mario Land world 1-1, with one particular frozen
tensor-train spiking reservoir configuration, under one shared PPO hyperparameter set,
at matched trainable-parameter count (ratio 1.0487), across 10 independently-trained
seeds per arm and 1,000,064 env steps each, the frozen reservoir arm scores
significantly lower mean extrinsic return than the trained GRU baseline in every trained
condition measured, and roughly six times lower reward per step of experience. Both arms
learn significantly above their own untrained initialisation, and the untrained arms are
statistically indistinguishable from one another.

**What it does not tell you.**

- **It is not evidence about frozen reservoirs in general.** It is evidence about one
  configuration, which §7 shows was measurably badly calibrated, at one parameter budget,
  under one optimizer setup that §6 shows interacted badly with that architecture.
- **It is not evidence about the multi-game goal in `DESIGN.md` §1.1.** That roadmap's
  Phase 2 (multi-game generalization within Game Boy / Game Boy Color) and Phase 4
  (Pokémon-style RPG targets) are separate questions this experiment has no bearing on
  beyond removing the premise Phase 1 was supposed to supply.
- **It is not a claim that the reservoir arm cannot be made to win.** §6.4's pilot
  already shows one fix worth about 2.1× in early training. It does show that as
  specified, pre-registered, and run, it lost.
- **It resolves nothing about the entanglement-entropy diagnostic**, per §7.5.

**Consequence for the roadmap, which the project owner should weigh.** `DESIGN.md` §7's
build order makes the next ablation conditional on the reservoir arm winning: build-order
Phase 2 (the resonate-and-fire neuron swap) is written as *"once Phase 1 shows the
reservoir arm beating baseline"*, and build-order Phase 3 (DLIF, RSSR) is explicitly
*"not started until Phase 1–2 succeed"*. **Phase 1 as specified did not show that.**
Taken literally, the build order says stop.

Taken with §6 and §7, the more defensible reading is that the build order's precondition
has not yet been fairly tested — the arm that lost was handicapped by a clipping
interaction and a construction defect, both now measured and both now fixable at zero
parameter cost. That is what the corrected comparison is for. But it is a *reading*, and
the literal precondition is unmet; this document does not get to decide that question by
phrasing, and the decision belongs to the project owner, made on this evidence rather
than around it.

---

## 11. Reproduction

- **Evaluation harness:** `training/evaluate.py`, run from a worktree pinned to commit
  `64839a9`, 30 episodes, `--seed 0`, both regimes (`--state-reset-interval` unset and
  `128`).
- **Matrix driver:** `scripts/run_eval_matrix.py`.
- **Aggregation and statistics:** `analysis/aggregate_results.py` — permutation test by
  full enumeration, exact Mann-Whitney, percentile bootstrap
  (`numpy.random.default_rng(0)`, 20,000 resamples). The module reduces every checkpoint
  to one number per training seed before any statistic sees it.
- **Raw results:** `results/{final,best,init}/eval_{arm}_seed{N}_{regime}.json`, 120
  files. Each carries its own per-episode returns, per-episode lengths and episode seeds,
  so every aggregate in §3 and §5 can be recomputed from the files rather than trusted.
- **Checkpoints and training logs:** `checkpoints/{arm}_seed{N}/` — ten `step_*.pt`
  files and `train_log.jsonl` per run. Both `checkpoints/` and `checkpoints_init/` are
  gitignored and are not distributed; the runs are reproducible from
  `training/train.py --arm {baseline,reservoir} --seed N --steps 1000000`, which is
  deterministic given the seed.
- **Checkpoint filenames are not round numbers.** The final checkpoint is
  `step_1000064.pt`, not `step_1000000.pt`, because the step counter advances in
  increments of 128. Globbing for the round number matches nothing.
- **ROM.** Not distributed. Supply your own legally-dumped Super Mario Land ROM via
  `MARIO_LAND_ROM_PATH`.

---

## 12. What v2 will add

Appended beneath this document, never edited into it, on the same rule
`EXPERIMENT_LOG.md` §11 uses: if something stated here turns out to be wrong, the wrong
statement stays on the page with the correction beneath it.

- **The corrected comparison**: full-length runs under `--grad-clip-mode per-group` and
  `--embed-init-mode centered`, applied identically to both arms, at the same 10 seeds
  and the same evaluation matrix. §6.4's three-seed, 425-update pilot is the only signal
  currently available and is not a result.
- **Whether the corrected reservoir arm changes the verdict, in either direction**,
  reported the same way this document reports the uncorrected one.
- The reservoir arm's own convergence question from §9 — whether 1,000,064 steps was
  enough for an arm whose training reward was still rising at the end of it.

Until v2 exists, **§1 is the answer this repository has to `DESIGN.md` §5**, and §6 is
the reason it is v1.
