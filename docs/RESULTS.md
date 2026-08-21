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

---
---

# Phase 1 Results — v2, the corrected comparison

**Version: v2.** Appended beneath v1, never edited into it. **v1's numbers stand exactly
as written, including where v2 contradicts them** — the same append-only rule
`EXPERIMENT_LOG.md` §11 applies to itself. v1 §12 promised this section; it is superseded
by it, not replaced with it.

**Read v1 first.** v1's headline — the frozen spiking reservoir loses to the
matched-parameter trained GRU baseline under the pre-registered protocol — is the answer
this repository gave to `DESIGN.md` §5 *as specified*, and it is not withdrawn. v2 asks
one question: does that answer survive removing the optimizer/clipping confound v1 §6
documented? Neither the negative headline nor the confound cancels the other, and neither
should be quoted without the other.

Date: 2026-08-21
Scope: `docs/DESIGN.md` §5 / §5.1, re-run under the corrected configuration
pre-registered in `EXPERIMENT_LOG.md` §14.11. The structure of this section was fixed in
`EXPERIMENT_LOG.md` §17.9 **before any v2 number existed**, so it cannot have been chosen
to flatter them.

---

## 13. Headline (v2)

**Under the corrected protocol the frozen spiking reservoir still loses to the
matched-parameter trained GRU baseline on the declared scoreboard
(`mean_extrinsic_return`) — in all four trained conditions, by 7.26 to 8.97 points of
mean episode return, with exact two-sided permutation p-values between 0.000141 and
0.003497 and Cohen's *d* between −1.45 and −2.22.** Every trained bootstrap 95% CI
excludes zero, on the losing side. In the headline condition (`final`, `continuous`) the
gap is **−8.97 points, p = 0.000141, d = −2.2237**.

**The verdict is unchanged from v1. Its evidential status is not.** Three things changed,
and all three make v2 the more informative measurement:

1. **The confound v1 could not resolve is gone.** v1 §6.3 stated plainly that the
   as-specified comparison *"does not cleanly separate 'the frozen reservoir is a weaker
   feature extractor' from 'the frozen reservoir's readout was trained far more slowly by
   an optimizer/clipping interaction'."* Under per-group clipping the readout's effective
   optimizer step now sits **inside the healthy GRU's own band** (`EXPERIMENT_LOG.md`
   §15.2). That alternative explanation is no longer available, so a v2 loss is
   attributable to the architecture in a way a v1 loss was not.
2. **Both arms got substantially better, and the reservoir got better faster** — its mean
   per-update extrinsic *training* reward rose 0.018854 → 0.082267 (4.4×) while the
   baseline moved 0.109769 → 0.113186. The training-reward gap closed from **5.82× to
   1.38×**. The corrections worked, and they were not enough.
3. **The metric that flattered the reservoir in v1 no longer does** (§17 below). v1 §5
   found the reservoir surviving ~6× longer per episode and earning ~6× less per step, so
   episode return — an integral over the episode — closed part of a per-step gap it never
   actually closed. **In v2 the two arms' episode lengths are statistically
   indistinguishable** (353.1 vs 336.9 steps, p = 0.0725), so the return scoreboard and
   the per-step normalisation now agree: 1.26× and 1.32× respectively. **v2's headline is
   not conservative and not inflated; it is simply the number.**

`README.md`'s standing commitment applies unchanged: the mandatory-control design exists
so that a negative result is scientifically informative rather than a silently discarded
run. This is the second such result and it is reported the same way as the first.

---

## 14. What changed from v1, exactly

Three flags, **applied identically to both arms**, and nothing else:

| | v1 | v2 |
|---|---|---|
| gradient clipping | `--grad-clip-mode global` | `--grad-clip-mode per-group` |
| embedding initialisation | `--embed-init-mode legacy` | `--embed-init-mode centered` |
| embedding weight-init scale | `--embed-scale 1.0` | **`--embed-scale 3.0`** |
| untrained controls | `checkpoints_init/` (legacy init) | `checkpoints_v2_init/` (rebuilt, centred init) |

**`--embed-scale 3.0` is stated explicitly because v1 §12 omitted it**, and
`EXPERIMENT_LOG.md` §14.8 records why that omission would have been costly: followed
literally, v1 §12's recipe specifies `centered` at the default scale of 1.0, which is
**worse than doing nothing** — 65.9454% of reservoir units silent, against 45.5917% for
the v1 default — because centring removes the DC drive without replacing it and starves
the reservoir. The validated pairing is `centered` **with** scale 3.0. The bias and the
gain are a fix only together.

Everything else is held: 10 seeds per arm (0–9), 1,000,064 env steps per run, 7,813 PPO
updates, `lr = 3e-4`, `rollout_len = 128`, `MAX_GRAD_NORM = 0.5`, the same 12-dimensional
RAM observation, the same action set, the same novelty-gate curiosity signal on both arms,
the same evaluation protocol (30 episodes, eval seed 0, both recurrent-state regimes,
three checkpoint selections), and the same exact statistics.

**Why both arms receive both treatments.** The baseline GRU needs neither — its embedding
feeds a trainable recurrent core that can learn to absorb a DC offset, and its gradients
never explode. It gets them anyway, because **a treatment only one arm receives is not a
control**. The resulting asymmetry in benefit is a *result*, not a licence.

**Parameter parity is unchanged**: the centred initialisation writes into a bias tensor
that already existed, so it costs zero new parameters. The arms remain at 139,179
(reservoir) against 132,715 (baseline), ratio 1.0487.

---

## 15. Primary result (v2)

Differences are **reservoir minus baseline**; negative favours the baseline. Means and
standard deviations are over the 10 training seeds of each arm; n = 10 versus n = 10
throughout, one number per training seed (§2.4).

| condition | reservoir mean ± SD (n=10) | baseline mean ± SD (n=10) | diff | exact permutation p | Cohen's d |
|---|---|---|---|---|---|
| final, continuous | 34.2404 ± 2.7438 | 43.2060 ± 4.9982 | **−8.9656** | **0.000141** | **−2.2237** |
| final, reset128 | 34.3238 ± 3.2248 | 41.7829 ± 6.0647 | −7.4592 | 0.002327 | −1.5358 |
| best, continuous | 34.0465 ± 2.6659 | 41.4210 ± 5.8346 | −7.3746 | 0.001061 | −1.6258 |
| best, reset128 | 33.9450 ± 3.5979 | 41.2004 ± 6.1027 | −7.2554 | 0.003497 | −1.4484 |
| init (untrained), continuous | 11.6202 ± 13.9639 | 8.0019 ± 2.9485 | +3.6183 | 0.435223 | +0.3585 |
| init (untrained), reset128 | 10.0015 ± 10.4898 | 8.0115 ± 3.1401 | +1.9900 | 0.573546 | +0.2570 |

Exact Mann-Whitney U, same data: final/continuous U = 6.0 (p = 0.000325), final/reset128
U = 15.0 (p = 0.006841), best/continuous U = 11.0 (p = 0.002089), best/reset128 U = 12.0
(p = 0.002879), and both untrained rows U = 49.0 / 51.0 (p = 0.970512).

Bootstrap 95% CIs on the difference of means (20,000 resamples, `default_rng(0)`):

| condition | 95% CI on the difference |
|---|---|
| final, continuous | [−12.3127, −5.6187] |
| final, reset128 | [−11.6248, −3.4802] |
| best, continuous | [−11.2784, −3.6950] |
| best, reset128 | [−11.5210, −3.1817] |
| init, continuous | [−4.4561, 12.1346] |
| init, reset128 | [−4.1967, 8.6090] |

Four observations:

1. **The direction is the same in all four trained conditions**, under both
   checkpoint-selection rules and both recurrent-state regimes, with every trained CI
   entirely below zero. As in v1, the result does not depend on which regime is scored or
   which selection rule is applied.
2. **The seed-to-seed spread has reversed between the arms.** In v1 the reservoir was the
   noisier arm (SD 4.51–7.52 against the baseline's 2.05–3.41). In v2 the **reservoir is
   the more consistent arm** (SD 2.67–3.60 against the baseline's 5.00–6.10). The
   corrected reservoir arm learns something reliably; it is reliably worse. This is a real
   change and it is not one that favours the reservoir on the scoreboard.
3. **`reset128` still does not rescue the reservoir.** If the frozen 8192-dim reservoir's
   advantage were memory horizon, the matched-regime column is where the gap should
   narrow most. It narrows (−8.97 → −7.46 at `final`) but never closes, and under `best`
   the two regimes are within 0.12 points of each other. There is no regime in this matrix
   where the reservoir wins.
4. **The untrained rows are the control, not a result** — §16.1.

---

## 16. The two controls (v2)

### 16.1 The untrained arms are statistically indistinguishable

The `init` rows compare random-initialised, never-trained policies of the two arms under
the same evaluation procedure: p = 0.435223 (continuous) and p = 0.573546 (reset128),
Cohen's *d* of +0.3585 and +0.2570, and bootstrap CIs straddling zero in both regimes.
Exact Mann-Whitney gives p = 0.970512 in both.

**The arms start equivalent**, so §15's trained gap is attributable to what was learned,
not to one arm having been handed a better starting point. These are **fresh** controls
built under the *centred* initialisation (`checkpoints_v2_init/`, 20 runs at `--steps 0`)
— v1's controls were built under `legacy` and would have been the wrong control for v2
(`EXPERIMENT_LOG.md` §14.9).

The same caveat v1 §4.1 states applies here: absence of a detectable difference at n = 10
is not proof of equality, and the CIs ([−4.4561, 12.1346] and [−4.1967, 8.6090]) are wide
next to a trained gap of about 9. What it rules out is a large initialisation advantage
*in the baseline's favour* that could have produced §15 on its own; as in v1, the
untrained point estimates lean slightly the other way.

### 16.2 Both arms learn significantly above their own initialisation

`final` versus `init`, same arm, same seeds, `continuous` regime:

| arm | trained − untrained | Cohen's d | exact permutation p |
|---|---|---|---|
| baseline | +35.204 | +8.579 | 0.000011 |
| reservoir | +22.620 | +2.248 | 0.000043 |

0.000011 is the design's floor (2/184,756). **The reservoir arm is a working arm that
loses, not a broken one** — and it now improves on its own initialisation by 22.62 points
(against 18.45 in v1) at a much stronger *d* (+2.248 against +1.790). It learns more than
it did in v1, and still loses to an arm that learned more again.

---

## 17. The per-step decomposition — v1's most important caveat does NOT carry over

`EXPERIMENT_LOG.md` §17.9 required this to be **recomputed rather than assumed**, and
that instruction earned its keep: this is the largest qualitative change between the two
versions.

`final` checkpoints, means over the 10 training seeds. Reward per step is computed **per
seed** as `mean_extrinsic_return / mean_episode_length` and *then* averaged over seeds —
never as a ratio of the two mean columns (`analysis/per_step_decomposition.py` documents
why the orderings are not interchangeable, and its tests pin the distinction).

**v2, continuous:**

| | episode return | mean episode length | reward per step |
|---|---|---|---|
| baseline trained | 43.206 | 336.88 | 0.127974 |
| reservoir trained | 34.240 | 353.07 | 0.097147 |
| baseline untrained | 8.002 | 2779.06 | 0.002860 |
| reservoir untrained | 11.620 | 2600.90 | 0.005668 |

- Baseline / reservoir **reward-per-step ratio: 1.3173×**, exact permutation p = 0.000022.
- **Episode-length difference: +16.19 steps, p = 0.072517 — not significant.**
- `reset128` agrees: per step 0.125382 vs 0.096973, ratio **1.2930×** (p = 0.000054);
  lengths 331.93 vs 354.71 (p = 0.048735).

**The strategic divergence v1 found has disappeared.** v1 §5's central observation was
that the two arms *"did not learn worse and better versions of the same strategy — they
learned qualitatively different strategies"*: the baseline moved right fast and died fast
(~315-step episodes at 0.11455/step), while the reservoir survived without progressing
(~1917-step episodes at 0.01921/step). **In v2 the reservoir's episodes are 353 steps.**
It abandoned the survival strategy and now plays the same game the baseline plays, only
less well.

**Consequence for the headline, and it cuts against the reservoir.** v1's headline was
*conservative*, because a ~6× longer episode inflated the reservoir's integral return and
closed part of a per-step gap it never closed. **That correction no longer applies.** With
episode lengths equal, the return ratio (43.206 / 34.240 = **1.262×**) and the per-step
ratio (**1.317×**) are the same quantity viewed two ways, and they agree. v2's −8.97 is
neither flattered nor understated.

The untrained rows behave as in v1: both untrained arms survive far longer (2779 and 2601
steps) while earning almost nothing per step, confirming that **long episodes are the
untrained default here, not an achievement**. In v1 the trained reservoir had barely moved
away from that default; in v2 it has moved essentially as far as the baseline.

---

## 18. v1 versus v2, side by side

| condition | v1 diff | v2 diff | change |
|---|---|---|---|
| final, continuous | −7.7167 | **−8.9656** | wider by 1.25 |
| final, reset128 | −6.1708 | −7.4592 | wider by 1.29 |
| best, continuous | −7.0190 | −7.3746 | wider by 0.36 |
| best, reset128 | −7.3948 | −7.2554 | narrower by 0.14 |

**Both arms improved, and the baseline improved more in absolute terms** (`final`,
`continuous`): reservoir 28.4169 → 34.2404 (+5.82), baseline 36.1335 → 43.2060 (+7.07).
The gap therefore widened slightly rather than closing.

The training-log view, computed by the conventions pinned in `EXPERIMENT_LOG.md` §20.3 so
the two versions are comparable line for line:

| quantity | v1 | v2 |
|---|---|---|
| baseline/reservoir mean per-update extrinsic training reward, all 7,813 updates | 5.8220× | **1.3758×** |
| same, final decile | 5.3252× | **1.2400×** |
| convergence, 5th → 10th decile, baseline | +0.58% | +5.75% |
| convergence, 5th → 10th decile, reservoir | +13.33% | +12.54% |

**This is the sharpest single contrast in the two documents.** On *training* reward the
corrections closed most of the gap — from 5.82× to 1.38×. On the *evaluation* scoreboard
the gap did not close at all. The corrected reservoir arm optimises the training objective
far better than v1's did and converts that into episode return no better, which is a more
specific and more interesting negative result than v1's.

**The interpretive constraint, recorded in `EXPERIMENT_LOG.md` §17.6 before any v2 number
existed:** the v1 → v2 comparison changes **two things at once**, so no change here is
attributable to per-group clipping or to the centred initialisation alone. §15.3's 2×2
factorial, at three seeds and 30% of a run, attributes ~96% of the short-horizon effect to
per-group clipping (+0.061975 of a combined +0.064706), with centring worth ~9% alone and
~4% on top of clipping, and an interaction (−0.003143) smaller than the seed standard
deviation (0.0126) and therefore not interpretable. **A full-scale decomposition — a third
10-seed condition at `per-group` + `legacy` — was deliberately not run**, because the
mandate was the corrected two-arm comparison and a third arm would have put the primary
deliverable at risk for a secondary question. It is registered as the obvious next
ablation.

---

## 19. Reservoir health over a full run: A7 and A9

Both pre-registered before the v2 runs existed — A7 at `EXPERIMENT_LOG.md` §14.5, A9 at
§15.6 — with verdict bands fixed in advance and computed **in code** by
`analysis/reservoir_health.py`, so neither verdict is a judgement call made while looking
at the number. Per §17.11, **both versions are measured by the same instrument**, so the
v1 column here comes from that module and not from earlier prose.

| measurement | v1 (legacy + global) | v2 (centred@3.0 + per-group) | pre-registered band | verdict |
|---|---|---|---|---|
| **A7** mean final dead `in_proj` columns | 9.8010% (sd 3.5288) | **0.1636% (sd 0.0892)** | <2% confirms, ≥5% falsifies | **CONFIRMED** |
| **A9** mean final silent-unit fraction | 46.5222% (sd 0.8487) | **32.1606% (sd 2.0887)** | <40% confirms, ≥46% falsifies | **CONFIRMED** |

**A7 — the dead-gradient budget is essentially eliminated.** H7 predicted fewer than ~164
dead columns of 8192; the measured mean is **13.4 columns (0.1636%)**, a 60× reduction
against v1's 9.8010%. The nesting property holds for **all ten seeds** — `newly_dead = 0`
at every one of the 9 checkpoint transitions, so as in v1 no column ever dies after
training starts; columns only wake up. The architectural criticism that a tenth of the
trainable budget never received a gradient is **solved** by the input fix.

**A9 — the centred initialisation's advantage shrinks but does not vanish**, exactly as
H9 predicted. **It must not be read as the reservoir staying healthy.**
`EXPERIMENT_LOG.md` §15.4 fixed a binding constraint on this prose before the runs
launched, and it is honoured here verbatim: *centring holds the reservoir near-fully
active for roughly the first 100k steps and roughly halves the silent fraction at 300k,
after which the invariant decays because the trainable weight drifts and the bias does not
follow it.* The full trajectories confirm the decay continues to the end of the run —
silent fraction climbs from ~14% at 100k to ~36% at 1,000,064 on the worst seed.

**Two costs that A9 measures and that no shipped fix controls:**

- **The operating point still runs away.** Mean spike rate at the final checkpoint reaches
  **0.194** on seed 9 — roughly **10× the ~2% band** `models/spiking_reservoir.py`
  documents as healthy. v1's `legacy` runs ended at 0.200. The centred init changes *when*
  the operating point leaves the healthy band, not *whether* it does.
- **Saturated units reappear late** (0.54% by the final checkpoint on seed 9), having been
  zero at initialisation.

This is the successor problem `EXPERIMENT_LOG.md` §15.5 stated and A9 was pre-registered
to measure rather than extrapolate: **per-group clipping fixes the optimizer pathology and
centring fixes the *initial* operating point, and nothing in the current design regulates
where the operating point goes after that.**

**The frozen-reservoir invariant holds bit-for-bit**: max absolute difference **0.0e+00**
across all 100 v2 checkpoint loads (and 100 v1 loads), so every reservoir evaluated here
is bit-identical to the one its run was initialised with.

---

## 20. Efficiency (v2)

Measured with the machine **quiet** — no training, no evaluation, nothing else running —
which `EXPERIMENT_LOG.md` §17.9 requires explicitly, because §17.3's figures were taken
while the matrix was contending and may not be quoted as a §8 replacement. 50,048 env
steps per measurement, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, single run.

| flag set | baseline | reservoir | ratio |
|---|---|---|---|
| **v2** (`per-group`, `centered`, scale 3.0) | **1303.4 env-steps/s** | **439.4 env-steps/s** | **2.966×** |
| v1 (`global`, `legacy`, scale 1.0), same session | 1310.8 env-steps/s | 440.2 env-steps/s | 2.978× |

**The corrections are throughput-neutral.** Measuring both flag sets back to back on the
same quiet machine was done specifically to answer whether the corrected configuration
costs anything, and it does not — 2.966× against 2.978× is noise.

**This also resolves, honestly, a discrepancy against v1 §8**, which reports 918 and 371
env-steps/s for a ratio of 2.474×. Both arms measure ~40% faster here in absolute terms,
and the *ratio* differs too. Since the v1 flag set reproduces 2.978× in this session, **the
difference is a property of the measurement conditions, not of the flags** — machine
state, thermal condition and measurement window differ between the two sessions. The
defensible claim is the ratio measured within a single session; v1 §8's numbers are not
retracted, and this paragraph is why the two tables disagree.

Final checkpoint sizes, both including Adam optimizer state:

| arm | final checkpoint |
|---|---|
| baseline | 1,604,881 bytes (1.60 MB) |
| reservoir | 2,834,930 bytes (2.83 MB), **1.766×** |

Unchanged from v1 and for the same reason: the frozen buffers — chiefly `W_in`, plus the
four TT cores — must still be stored despite never receiving a gradient.

**The efficiency result is the one finding that is robust across both versions and both
verdicts: the reservoir arm costs about 3× the compute per env step and 1.77× the storage,
at matched trainable-parameter count.** As in v1, **no TT compression ratio against a
dense 8192×8192 matrix is claimed** — it was not measured.

---

## 21. Limitations (v2)

Everything still applicable from v1 §9 carries forward and is not repeated in full: one
game and one level; one shared hyperparameter set never tuned for the reservoir's
architecture; a deterministic environment, so 30 episodes per checkpoint measure
policy-sampling variance only and the unit of analysis is the training seed; n = 10 seeds
per arm; both arms scored in a `continuous` regime neither was trained in; three of the
twelve observation slots hardcoded to zero; and the pre-registered multiple-comparisons
disclosure (now covering eight pre-registered ablations against two 10-seed comparisons —
all four trained rows in §15 sit below 0.05/6 = 0.008333 and survive a Bonferroni
correction over the six rows of that table; the two untrained rows are nowhere near
significance under any correction).

**Added by v2, and specific to it:**

- **Two treatments changed at once** (§18, `EXPERIMENT_LOG.md` §17.6). No result here is
  attributable to per-group clipping or to the centred initialisation alone.
- **The embedding fix is transient, and was known to be before these runs launched.**
  H14a was falsified at 22.3877% against a 15% threshold *before* the matrix launched, and
  the launch proceeded under an explicitly recorded override of the pre-registered decision
  rule (`EXPERIMENT_LOG.md` §15.4) — because H14a is an *efficacy* hypothesis while the
  *validity* hypothesis H14b survived decisively. That override, and its reasoning, were
  written down before the matrix ran, not afterwards.
- **The per-group clipping group-count asymmetry**, disclosed in v1 §9 and load-bearing
  here: `group_trainable_parameters` buckets by top-level submodule, giving **2 groups on
  the reservoir arm** (`embedding`, `readout`) against **4 on the baseline** (`embedding`,
  `gru`, `actor_head`, `critic_head`) — verified directly during v2's baseline pre-flight.
  Clipping each group to `MAX_GRAD_NORM` separately therefore permits a larger total update
  norm on the arm with more groups. The rule is applied identically and the grouping is
  discovered from the model rather than hardcoded, but the counts differ, and **this
  asymmetry favours the baseline.** It is the most substantive uncontrolled variable in
  v2 and it is disclosed rather than buried.
- **The fixture caveat** (`EXPERIMENT_LOG.md` §14.13). Every silent-unit fraction in §19
  is measured against `tests/data/real_obs_6000.npy`, collected under **v1** policies.
  Holding the observation window fixed and varying only the embedding is the right
  controlled comparison for "did the trained embedding drift away from its centring", and
  it is **not** a measurement of what a v2 policy experiences in situ. The DC offset is
  policy-dependent, and a v2 agent's own observation distribution is a third distribution
  measured by none of these. **That in-situ measurement was not taken and remains a known
  gap.**
- **One reservoir configuration still**: `reservoir_size = 8192`, `tt_rank = 8`,
  `tt_n_cores = 4`, `beta = 0.9`, threshold 1.0. A8's structured-core construction
  (`tt_bond_decay`) is **not wired into training** and is not part of v2 at all.
- **The reservoir arm's training reward was still rising** (+12.54% from the 5th to the
  10th decile, against the baseline's +5.75%), so as in v1 a longer budget could narrow
  the gap and this experiment does not bound by how much. Note the baseline is now also
  rising, where in v1 it had flattened (+0.58%).
- **Provenance: the v2 reservoir arm was trained twice.** The first attempt reached ~95.7%
  and was destroyed by a machine-wide power loss at 23:38:53 on 2026-08-20. It was
  **restarted from step 0 rather than resumed**, so that both arms ran an identical
  uninterrupted protocol and so the published recipe reproduces the published numbers. The
  restarted runs were verified **bit-identical** to the destroyed ones over their full
  overlap — 523,236 values compared across ten seeds and seven float fields, **0
  mismatches** (`EXPERIMENT_LOG.md` §18). This is a note about provenance, not about the
  data.

---

## 22. What v2 does and does not tell you

**What it tells you.** On Super Mario Land world 1-1, with one particular frozen
tensor-train spiking reservoir configuration, under one shared PPO hyperparameter set, at
matched trainable-parameter count (ratio 1.0487), across 10 independently-trained seeds per
arm and 1,000,064 env steps each, **with the optimizer/clipping confound of v1 §6 removed
and the input-calibration defect of v1 §7 corrected at initialisation**, the frozen
reservoir arm still scores significantly lower mean extrinsic return than the trained GRU
baseline in every trained condition measured, and about 1.3× lower reward per step of
experience. Both arms learn significantly above their own untrained initialisation, the
untrained arms are statistically indistinguishable, and the two arms now converge on
strategies of the same *shape* (near-identical episode lengths) rather than the
qualitatively different ones v1 found.

**What it does not tell you.**

- **It is still not evidence about frozen reservoirs in general.** It is evidence about
  one configuration at one parameter budget under one hyperparameter set — now a
  *better-calibrated* instance of that configuration, but the same one. §19 shows its
  operating point still leaves the healthy band during training.
- **It is not a decomposition of the two fixes** (§18).
- **It is not evidence that the remaining gap is irreducible.** The reservoir's training
  reward was still rising at the budget's end, and the group-count asymmetry in §21 favours
  the baseline. What it shows is that removing the two diagnosed defects — which closed the
  *training-reward* gap from 5.82× to 1.38× — did not change the evaluation verdict.
- **It is not evidence about the multi-game goal in `DESIGN.md` §1.1.** Roadmap Phase 2
  (multi-game generalisation) and Phase 4 (Pokémon-style RPG targets) remain separate
  questions this experiment bears on only by supplying the premise Phase 1 was meant to
  establish.
- **It resolves nothing about the entanglement-entropy diagnostic.** A8 (`EXPERIMENT_LOG.md`
  §16) made that question *testable* for the first time by showing S̄ is tunable across
  essentially all of [0, 1] via a structured-core bond profile, but **A8 contains no
  training at all** and the knob is not wired into the training path. The sibling project's
  open question remains **open, in neither direction**.

**Consequence for the roadmap, which the project owner should weigh.** v1 §10 recorded
that `DESIGN.md` §7's build order makes the next ablation conditional on the reservoir arm
beating baseline, that Phase 1 as specified did not show that, and that taken literally
the build order says stop — while noting the more defensible reading was that the
precondition had not yet been *fairly* tested, because the losing arm was handicapped by a
clipping interaction and a construction defect.

**That reading has now been tested, and it did not hold.** Both handicaps were removed at
zero parameter cost, both were verified removed by pre-registered measurements (§19: A7
confirmed, A9 confirmed), the reservoir arm improved substantially in absolute terms, and
it still lost by a margin no smaller than before. The precondition for build-order Phase 2
(resonate-and-fire) and Phase 3 (DLIF, RSSR) is unmet under the fairest test this project
has been able to construct. **The decision remains the project owner's**, and it should be
made on this evidence rather than around it — but v2 removes the specific escape hatch v1
left open.

---

## 23. Reproduction (v2)

- **Training:** `scripts/run_training_matrix.py --arms {reservoir,baseline} --seeds 0-9
  --rom "$ROM" --steps 1000000 --checkpoint-every 100000 --checkpoint-dir checkpoints_v2
  --grad-clip-mode per-group --embed-init-mode centered --embed-scale 3.0 --jobs 10`,
  executed from a `git worktree` pinned to commit `dc966a3` (`EXPERIMENT_LOG.md` §17.1) so
  every import resolves from one fixed commit while the data lands in the main repository.
- **Untrained controls:** the same flags with `--steps 0 --checkpoint-dir
  checkpoints_v2_init`.
- **Evaluation:** `scripts/run_eval_matrix.py --rom "$ROM" --episodes 30 --eval-seed 0
  --jobs 8 --checkpoint-dir checkpoints_v2 --init-checkpoint-dir checkpoints_v2_init
  --results-dir results_v2`. `training/evaluate.py` and `scripts/run_eval_matrix.py` are
  **byte-identical** to commit `64839a9`, which v1's evaluation was pinned to, and
  re-running one v1 evaluation from the current tree reproduced its committed result file
  exactly — all 30 per-episode returns, lengths and episode seeds identical. **v1 and v2
  were scored by the same harness**, which is what makes §18's side-by-side legitimate
  rather than an artefact of harness drift.

  > **Correction of record, added 2026-08-21.** The byte-identity claim above was true when
  > written and is **no longer true of `training/evaluate.py`**: roadmap Phase 2a added an
  > optional `--task` flag to it (`docs/DESIGN_ROADMAP_PHASE2.md` §9). **`scripts/run_eval_matrix.py`
  > is untouched and remains byte-identical** — Phase 2a deliberately got its own driver,
  > `scripts/run_phase2a_eval.py`, rather than modifying it. What v1/v2's side-by-side
  > actually rests on is *behavioural* identity of the scoring path, and that is preserved by
  > construction: with `--task` unset the env is built exactly as before (`world_level=None`,
  > the power-on boot to world 1-1), which `tests/test_task_axis_rom.py` asserts directly.
  > Nothing in v1's or v2's numbers changes. Recorded here rather than silently, because a
  > reader checking the claim against the file would otherwise find it false and have no way
  > to tell whether the results had drifted.
- **Statistics:** `scripts/run_v2_analysis.sh`. **Do not use `EXPERIMENT_LOG.md` §14.11's
  Step 5 command** — pointing `--results-dir` at `results_v2` rather than
  `results_v2/{final,best,init}` silently compares nothing while still printing a
  healthy-looking training-log summary (§19.1 of that document).
- **Per-step decomposition:** `analysis/per_step_decomposition.py`, which reproduces v1
  §5's published figures digit-for-digit before being pointed at v2.
- **A7/A9:** `analysis/reservoir_health.py --checkpoint-dir checkpoints_v2 --arm reservoir
  --seeds 0-9`.
- **Raw results:** `results_v2/{final,best,init}/eval_{arm}_seed{N}_{regime}.json`, 120
  files, committed. Reports at `results_v2_report_{final,best,init}.{txt,json}`,
  `results_v2_health.txt` and `results_v1_health.txt`.
- **Checkpoints and training logs:** `checkpoints_v2/{arm}_seed{N}/` — ten `step_*.pt`
  files and `train_log.jsonl` per run, gitignored and not distributed, reproducible from
  the seed.
- **The final checkpoint is `step_1000064.pt`, not `step_1000000.pt`.** Globbing for the
  round number matches nothing.
