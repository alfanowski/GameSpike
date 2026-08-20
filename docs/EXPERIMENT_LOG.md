# Phase 1 Experiment Log

**Purpose.** This is a decision/status ledger, not a narrative writeup. Its job is
that a cold session — no chat history, no memory of how any of this was decided —
can reconstruct the full state of the Phase 1 experiment (design, why it was
designed that way, what has run, what hasn't, and what is and is not safe to quote)
from this file plus the repository alone. Entries record a decision, the reasoning
behind it, and the cost if that reasoning turns out to be wrong — update this file
rather than letting that context live only in a transcript. See `README.md` for the
one-paragraph project status and `docs/DESIGN.md` §5/§5.1 for the control's full
design rationale; this file is the operational companion, not a replacement.

**Standing rule for anyone reading this file: no number in the "Pilot observations"
section below is a Phase 1 result.** It is one seed, mid-run at the time it was
recorded. Treat every number there as diagnostic only, exactly as `README.md`'s
own status section and `training/evaluate.py`'s docstring insist for any
single-checkpoint number.

---

## 1. Reproduction environment

- venv: `.venv/bin/python`, Python 3.12.12, created with `uv` from the unmodified
  `requirements.txt`. **`.python-version` says 3.9 and is stale** — it predates
  `torch>=2.8` being added to `requirements.txt` and is incompatible with it. Do
  not let a tool that reads `.python-version` silently select 3.9 for this repo.
- Installed versions actually resolved (not just the `requirements.txt` floors):
  torch 2.13.0, pyboy 2.7.0, snntorch 1.0.0, gymnasium 1.3.0, numpy 2.5.2.
- ROM: `/Users/alfanowski/Desktop/Super Mario Land (World).gb` — quote it, the
  path contains a space and parentheses.
- Test suite, verified passing as of commit `990c5a1`:
  ```
  MARIO_LAND_ROM_PATH="/Users/alfanowski/Desktop/Super Mario Land (World).gb" \
    .venv/bin/python -m pytest tests/ -q
  ```
  → **169 passed, 0 skipped.**
- Hardware: MacBook Air M4, 10 CPU cores, 16GB unified memory, no CUDA, fanless.
  This is the only machine Phase 1 runs on (see `DESIGN.md` §2 — no rented GPU,
  zero-budget discipline carried over from the sibling projects).

---

## 2. Phase 1 experimental design (locked, currently executing)

- **2 arms** (`baseline` = trained GRU, `reservoir` = frozen spiking reservoir)
  **× 10 training seeds (0-9) × 1,000,000 env steps** each.
- `--rollout-len 128` (default) ⇒ exactly **7,813 PPO updates** per run
  (⌈1,000,000 / 128⌉).
- `--checkpoint-every 100000`. Because the step counter advances in increments of
  128 (not 1), the checkpoint-every-100k boundary is crossed *after* it, not on
  it — checkpoint filenames are **not round numbers**:
  ```
  step_100096.pt  step_200192.pt  step_300288.pt  step_400384.pt  step_500480.pt
  step_600576.pt  step_700672.pt  step_800768.pt  step_900864.pt
  step_1000064.pt   <- final, unconditional save (7813 * 128), NOT step_1000000.pt
  ```
  Ten checkpoints per run. **Globbing for `step_1000000.pt` matches nothing** —
  every prior aggregation attempt that assumed round step numbers will silently
  find zero files. Match on `step_*.pt` and parse the number, or match on
  `step_1000064.pt` specifically for the final one.
- Outputs: `checkpoints/{arm}_seed{seed}/`, containing the ten `step_*.pt` files
  above and `train_log.jsonl` (one JSON object per PPO update, appended live).
  Both `checkpoints/` and `checkpoints_init/` (below) are gitignored — see
  `.gitignore` lines 6-11 — never expect these to be in a fresh clone.
- **Untrained reference control:** `--steps 0` skips the training loop entirely
  but still writes a checkpoint, giving a random-initialized, never-trained
  policy. These live at `checkpoints_init/{arm}_seed{seed}/step_0.pt` — 20 of
  them (2 arms x 10 seeds) — and exist so the eventual writeup can test whether
  either arm beats its *own* initialization at all, not just whether it beats
  the other arm. Sizes: baseline 534,773 bytes, reservoir 1,693,621 bytes.
  (These are much smaller than a trained final checkpoint — e.g. the ~1.60MB /
  ~2.83MB figures in §4 below — because `save_checkpoint`, per
  `training/train.py:171-190`, always writes `optimizer.state_dict()` alongside
  the model, and Adam only allocates its per-parameter `exp_avg`/`exp_avg_sq`
  buffers lazily on the first real `.step()` call. At `--steps 0` that call never
  happens, so the optimizer state is near-empty. Do not compare init-checkpoint
  size to trained-checkpoint size as if they measured the same thing.)

### Decision: why 10 seeds x 1,000,000 steps

- **Reasoning.** `training/evaluate.py`'s own module docstring states plainly
  that one checkpoint per arm cannot support an arm comparison at all, because
  training-seed variance dominates policy-sampling variance. The unit of
  analysis is therefore **the training seed**, not the episode and not the
  checkpoint — each seed contributes exactly one number to the eventual
  comparison (see `analysis/aggregate_results.py`'s point 2, which enforces this
  by reducing every checkpoint to one mean before any statistic sees it).
  10 seeds per arm puts an exact two-sided permutation test's best achievable
  p-value at 2/C(20,10) = 2/184,756 — and is already above the 3-5 seeds common
  in published deep-RL work, which is the comparison this project is trying not
  to be weaker than.
  Training is deterministic given the seed, so a 1,000,000-step run's state at
  any intermediate checkpoint is bit-identical to what a shorter run would have
  produced at that same step. This is *why* 1,000,000 was chosen as the budget
  rather than something smaller: it locks in nothing, because any shorter
  reporting budget (100k, 300k, ...) can be pulled from the checkpointed prefix
  of the same run instead of requiring a second training pass.
- **Cost if wrong.** Too few seeds reproduces exactly the failure mode
  `evaluate.py`'s docstring warns about: a comparison with no statistical power,
  where an apparent arm difference is indistinguishable from training-seed noise.
  That failure would not surface as an error — it would surface as a p-value
  that looks fine until someone checks how many splits could have produced it.

---

## 3. Measured throughput (real, `/usr/bin/time`-measured — not estimated)

With `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`:

| | single-run | 4-way parallel (aggregate) |
|---|---|---|
| baseline | 918 env-steps/s | 3,030 env-steps/s |
| reservoir | 371 env-steps/s | 1,098 env-steps/s |

Under 10-way contention — the real experiment's actual condition, all 10 CPU
cores occupied — throughput per reservoir run drops to roughly **80-130
env-steps/s**.

Torch intra-op threading (`torch.set_num_threads(>1)`) bought only ~4%
(baseline) / ~10% (reservoir) at this model size — not worth it against the cost
of cross-process contention — so every run is pinned to 1 thread and parallelism
is obtained across OS processes, not within one.

**Efficiency finding, reportable regardless of the Phase 1 outcome:** the
reservoir arm is **~2.5x slower per env step** than the baseline at matched
trainable-parameter count (§5), and its checkpoints are **2.83MB vs 1.60MB
(1.77x larger)**, because the frozen buffers (`reservoir.W_in`, the four TT
cores) still have to be stored even though they never receive a gradient. This
is a real cost of the architecture, independent of whether it turns out to help.

---

## 4. Verified invariants

Recorded with the method used to check them, so either can be re-checked rather
than re-trusted:

- **Frozen weights are actually saved, not just held in memory.** The
  reservoir's frozen weights (`reservoir.W_in`, `reservoir.tt_core_0` through
  `tt_core_3`) are registered via `register_buffer` **without**
  `persistent=False`, so they ARE part of `state_dict()` and a reloaded
  checkpoint restores the exact frozen weights it was trained against — not a
  freshly re-initialized reservoir. (This is also what `assert_reservoir_frozen`
  checks before every reservoir-arm checkpoint write, per `DESIGN.md` §3.)
- **Seeds genuinely produce different frozen reservoirs.** Checked empirically:
  `torch.equal()` on `W_in` and all four TT cores between
  `checkpoints_init/reservoir_seed3` and `checkpoints_init/reservoir_seed7`
  returns `False` on every one of them. Had this come back `True`, all 10
  reservoir "seeds" would in fact share one frozen reservoir behind different
  labels, and the multi-seed design in §2 would have been an illusion — the
  training-seed-variance argument for 10 seeds only holds if the seeds are
  actually different reservoirs, which this confirms they are.

---

## 5. Pilot observations (seed 0 only, from `train_log.jsonl`) — NOT a result

One seed. Read every number below as "this is what happened once," not as
evidence about either arm.

- **baseline, seed 0, completed run** (all 7,813 updates): final
  `mean_extrinsic_reward` **+0.14794921875**, final entropy **0.919** (down from
  the uniform-random maximum of ln(10) ≈ 2.303), final grad_norm 18.1.
  Per-decile mean extrinsic reward rose from +0.066 to +0.118 and largely
  plateaued after roughly the first 40% of training.
- **reservoir, seed 0, partial run** (~167k of 1,000,000 steps at the time this
  was recorded): per-decile mean extrinsic reward rose from +0.0062 up to a
  peak of +0.0367 around deciles 5-6, then **regressed** to +0.0084. Entropy
  fell from 2.189 to 1.994, then rose back to 2.174 — i.e. drifted back toward
  uniform-random after having started to commit to a policy.
- **Gradient norms per decile:** baseline 15.8 → 85.1; reservoir 2.08e6 →
  **2.92e8**. `MAX_GRAD_NORM = 0.5` (`training/train.py:92`), so **both arms are
  clipped on essentially every update** — meaning both are, in effect, taking
  fixed-norm normalized-gradient steps regardless of the pre-clip magnitude, and
  the reservoir's pre-clip gradient magnitude grew roughly 140x over the course
  of the (partial) run. Whether that pre-clip growth is itself informative, or
  is fully absorbed by the clip and therefore harmless, is an open question this
  pilot cannot answer with one seed.

**Why this is being recorded at all, given it isn't a result:** the pattern
above (peak-then-regress reward, entropy drifting back toward uniform,
gradient-norm blowup) is exactly the shape the pre-registered ablation A1 (§7)
exists to distinguish — an optimization artifact vs. an architectural one. If a
future session sees this pattern hold up across more seeds once real results
land, this paragraph is where to look for whether it was anticipated.

---

## 6. Structural observation driving ablation A2

The reservoir arm has **139,179** trainable parameters against the baseline's
**132,715** (ratio 1.049 — inside the ±10% band `tests/test_parameter_parity.py`
enforces against whatever `training/train.py:build_model` actually constructs).
Of the reservoir arm's trainable parameters, **8192 x 16 = 131,072 (94.2%)** are
a single linear projection from the 8192-dim reservoir state down to
`d_model=16`. Per `DESIGN.md` §5.1, `d_model=16` is not an arbitrary choice —
it's forced by holding trainable-parameter count fixed against a baseline whose
upstream representation is 192-dim instead of 8192-dim (see §5.1's ~4.7x
parameter-budget blowup if head widths were matched instead).

So the control is fair by the metric §5.1 argues is the correct one to hold
fixed — and it *simultaneously* constrains the reservoir arm into a narrow
readout bottleneck almost entirely spent on one down-projection. Record this as
a nuance for interpreting whatever Phase 1 finds, not as an excuse in advance
for a negative result: if the reservoir arm underperforms, ablation A2 (§7)
exists specifically to check whether that's the reservoir's actual ceiling or
an artifact of this bottleneck.

---

## 7. Evaluation protocol (planned, not yet run)

- `training/evaluate.py`, 30 episodes per checkpoint, in **both** regimes:
  - default continuous playthrough (no recurrent-state reset), and
  - `--state-reset-interval 128` — the matched-regime counterpart, since
    training itself resets recurrent state at every rollout boundary
    (`--rollout-len 128`).
  `evaluate.py`'s own docstring explains why both get reported: it's what
  separates "this arm is better" from "this arm just degrades more slowly once
  played past the horizon it was actually trained on."
- **Two checkpoint-selection rules, applied identically to both arms** —
  implemented as `select_final_checkpoint` and `select_best_checkpoint` in
  `analysis/aggregate_results.py` §2.5, wired together via
  `build_eval_manifest(..., selection="final"|"best")`:
  - `final` — highest step in the run.
  - `best` — highest mean *training* reward in the window ending at that
    checkpoint.
  Selection is always on **training** reward, never on evaluation reward —
  selecting on the evaluation measure would test on the same data used to pick
  the winner and bias the reported comparison upward. This is called out
  explicitly in that module's own section header as "the one rule that must
  never be violated."
- Results land as `results/{final,best}/eval_{arm}_seed{trainseed}_{regime}.json`
  (regime ∈ {continuous, reset128}) — the `final`/`best` split is which
  `results_dir` a given evaluation pass writes into; the filename pattern itself
  (`^eval_(?P<arm>[A-Za-z0-9]+)_seed(?P<train_seed>\d+)_(?P<regime>continuous|reset128)\.json$`,
  `analysis/aggregate_results.py:476`) only encodes arm/seed/regime.
  `load_eval_results` refuses to guess arm/seed/regime from a filename that
  doesn't match this convention rather than silently mis-bucketing a result.

---

## 8. Pre-registered ablations

Declared here **before** any of them are run — this is the p-hacking guard: all
three get reported regardless of outcome, win or lose, exactly like the main
comparison itself (`README.md`'s "every result, positive or negative, will be
reported as such"). With three ablations run against one 10-seed main
comparison, there is a real multiple-comparisons problem. That will be
disclosed in the eventual writeup, not quietly ignored — noted here so it isn't
forgotten by the time there's a result worth misreporting.

- **A1 — learning-rate sensitivity.** Reservoir arm at lr 1e-4 and 3e-5 (default
  is `LEARNING_RATE = 3e-4`, `training/train.py:85`), 3 seeds, 300k steps.
  **Hypothesis:** the instability pattern in §5's pilot observation (reward
  peak-then-regress, gradient-norm blowup) is an optimization artifact of
  sharing the GRU arm's learning rate, not an architectural property of the
  frozen reservoir. Distinguishes "frozen reservoir is a poor feature
  extractor" from "frozen reservoir needs different optimization hyperparameters
  than a trained recurrent core does."
- **A2 — richer readout.** Larger `d_model` / more readout layers, deliberately
  **breaking** trainable-parameter parity — declared explicitly as testing a
  *different* question from the main comparison, not as a correction to it.
  **Hypothesis:** per §6's 94.2% figure, the arm is readout-bottlenecked rather
  than reservoir-limited.
- **A3 — resonate-and-fire neurons** (frozen, random natural frequencies), per
  `DESIGN.md` §7's build-order Phase 2 (not to be confused with Roadmap Phase 2
  in §1.1 — see §7's own naming note in that document). **Hypothesis:** Super
  Mario Land's enemy/obstacle timing is genuinely periodic, which is the
  mechanism resonate-and-fire was designed to exploit.

---

## 9. Operational hazards already hit (recorded so they are not repeated)

- **macOS ships bash 3.2, which has no `wait -n`.** A job-queue semaphore
  written against it busy-spins instead of blocking on a job slot, and spams
  its own log at roughly 1MB/min. Any future queue/runner script needs to be
  written against this constraint, not against a Linux bash 5 assumption.
- **A queue runner must have both an atomic single-instance lock (`mkdir`,
  which is atomic on a local filesystem) and a per-job "skip if the run
  directory already exists" guard.** Two runners were once live at the same
  time. Without the second guard, both would have launched the *same* run into
  one output directory — interleaving writes to a single `train_log.jsonl` and
  racing each other's checkpoint writes, corrupting both.
- **Do not `git checkout`/`git switch` in the live working directory while
  training processes are running.** They launch new Python from this checkout;
  switching branches under them is a footgun for whatever they read from disk
  next. Use a `git worktree` for any repo work done while training is live —
  as this ledger itself was written.

---

## 10. Status

**Phase 1 training matrix is executing, as of 2026-08-20.** No Phase 1 result
exists yet. No number in §5 (pilot observations) may be quoted as one — it is
one seed, and for the reservoir arm, partial.

This ledger will be updated as runs complete and as the evaluation protocol
(§7) and ablations (§8) actually run. Until then, treat this file's "Pilot
observations" and "Status" sections as the only parts expected to go stale
between reads — everything else (design, reasoning, invariants, hazards) is
expected to hold for the rest of Phase 1.

---

## 11. Pre-registered ablations A4-A6 (reservoir construction)

Extends §8. Same p-hacking guard, same rule: **every one of these is reported
regardless of outcome**, and the multiple-comparisons disclosure in §8 now
covers six ablations against one 10-seed main comparison, not three.

A1-A3 ask what to do with the reservoir arm *as built*. A4-A6 ask a prior and
more uncomfortable question: **is the frozen reservoir this experiment is
measuring actually a well-constructed reservoir at all?** If it is not, the
Phase 1 comparison is not a test of "frozen spiking reservoir vs. trained GRU",
it is a test of "one particular badly-calibrated frozen reservoir vs. trained
GRU" — a much weaker claim, and one the writeup would have to make explicitly.
That is the stake here.

**This section is written and committed BEFORE any of the measurements below
are taken.** The commit timestamp is the evidence; that is the whole point of
the ordering. Nothing in §11 may be edited after the fact except by appending
results beneath it — if a hypothesis stated here turns out wrong, the wrong
statement stays on the page.

### 11.0 Prior measurements being explained (diagnostics, not Phase 1 results)

Both come from `checkpoints/reservoir_seed0/step_1000064.pt` and are subject to
the same standing rule as §5: one seed, diagnostic only.

- **52.7%** of the 8192 reservoir units did not spike at all within a sampled
  128-step window.
- **1,329 columns (16.22%)** of the readout's `in_proj.weight` had Adam
  `exp_avg_sq` **exactly** 0 after 3,910 optimizer updates — i.e. **21,264
  parameters (15.28% of the 139,179 trainable budget, §6)** never received a
  single gradient in a million env steps of training.

### 11.1 Orientation commitment (declared before measuring, because getting it backwards invalidates A4a)

`ActorCriticReadout.in_proj` is `nn.Linear(reservoir_size, d_model)`
(`models/actor_critic_readout.py:21`), so `in_proj.weight` has shape
`(d_model, reservoir_size) = (16, 8192)`. **A reservoir unit therefore indexes a
COLUMN of `in_proj.weight` (dim 1), not a row.** Unit `j` owns the 16 entries
`in_proj.weight[:, j]`. This is consistent with the prior measurement above
reporting 1,329 *columns* and 1,329 × 16 = 21,264 parameters. Every mapping from
parameter index to unit index in A4a uses dim 1. Recorded here so that if it is
wrong, it is wrong on the record and before the result.

### A4 — silent units and dead gradients share one root cause

- **H4a (mechanism).** A reservoir unit that never fires contributes a
  structurally-zero input to its `in_proj` column, so that column's gradient is
  identically zero on every update, so Adam's `exp_avg_sq` for it stays exactly
  0. **Prediction: the set of dead-gradient `in_proj` columns EQUALS the set of
  PERMANENTLY silent units.** Note this is deliberately *not* the per-window
  silent set: a unit silent across one 128-step window may fire in another,
  which is precisely why 16.22% and 52.7% are different numbers and why
  comparing them directly would be a category error. **Test:** set equality plus
  Jaccard index, against a permanently-silent set measured over ≥5,000 real
  observation steps.
  **Cost if wrong.** If the sets do not coincide, then either some column dies
  for a reason other than silence (an optimizer/clipping pathology — see
  `training/train.py`'s docstring on the readout being effectively frozen under
  global clipping, which would be a competing explanation) or some silent unit
  still accrues gradient. Either way the "one root cause" framing is false and
  fixing the reservoir would not, on its own, recover the dead 15.28%.
- **H4b (calibration).** `PolicyValueReservoir`'s embedding init
  (`models/policy_value_reservoir.py:63`) was calibrated against **synthetic
  N(0,1) inputs** — the file carries an explicit `# KNOWN GAP:` comment saying
  the scalar has only been validated against synthetic input and must be
  re-measured against real observations before the spike rate is trusted. Real
  observation statistics are now obtainable. **Prediction: the real observation
  distribution is not N(0,1) — `envs/mario_land_env.py` clips every component
  into [-1,1] and three of the twelve are hardcoded 0.0 (slots 9-11, reserved
  enemy features) — the induced input current therefore misses the documented
  ~0.3 target, and recalibrating the embedding scale against real statistics
  reduces the silent-unit fraction.**
  **Cost if wrong.** If the silent fraction is insensitive to embedding scale,
  silence is a property of the recurrent regime rather than of the drive, and
  A4b is not the lever — A5/A6 would be.
- **H4c (selection).** **Prediction: selecting among candidate reservoirs
  (identical hyperparameters, different seeds) on firing-rate health yields a
  materially better dynamical regime than taking seed 0 as given.** This is the
  methodology the sibling `spiking-reservoir-lm` project used in its Task 3.3
  reservoir-selection probe. It is worth stating explicitly why this is not
  result-selection: the reservoir is **frozen and never trained**, and the
  selection criterion (firing-rate health under observation data) is computed
  **before any training happens** and never sees task reward. Selecting a frozen
  component on a training-free health statistic is construction; selecting a
  checkpoint on evaluation reward is the thing §7 forbids. These are different
  acts and the writeup will keep them distinguished.
  **Cost if wrong.** If seed-to-seed spread in silent fraction / spike rate is
  small, seed selection buys nothing and the 10-seed design's reservoir variance
  is not a construction lever — which would also mean §4's "seeds genuinely
  produce different frozen reservoirs" is true bitwise but irrelevant
  dynamically.

### A5 — the chaotic regime is a fixable knob

- **H5.** Sweeping `SpikingReservoir`'s `spectral_radius` over **{0.7, 0.85,
  0.95, 1.0, 1.05}** (default 1.0) moves normalized entanglement entropy from
  its measured **0.9918** — essentially the maximum the diagnostic can report,
  i.e. maximally chaotic — toward the productive band **S̄ ∈ [0.1, 0.5]**
  reported by Sato et al. 2025, **without collapsing the spike rate**. ≥3 seeds
  per setting.
- **Mechanism note, recorded in advance so the result is interpretable.** On the
  TT path `spectral_radius` does not rescale a materialized matrix; it enters
  only through the derived core std, `s = (spectral_radius² / (N·R_int))^(1/2d)`
  (`spiking_reservoir.py:_build_tt_cores`), with `tt_core_std=None`. Since d=4,
  the exponent is 1/8: a 0.7 vs 1.05 change in spectral radius is a ~1.05x
  change in core std. **If entropy turns out to be insensitive to spectral
  radius, that is the reason, and it will be reported as "the knob barely moves
  the construction", not dressed up as a null result about criticality.**
  Entanglement entropy is a function of the frozen cores alone and does not
  depend on observation data; spike rate and silent fraction do.
- **Cost if wrong.** If entropy cannot be moved out of ~0.99 by this knob, the
  reservoir cannot be tuned into the published productive band at all through
  spectral radius, and the Phase 1 reservoir is permanently outside the regime
  its own design doc cites as productive.

### A6 — TT-rank is a second regime knob

- **H6.** The tensor-train bond dimension drives an order-to-chaos transition
  analogous to spectral radius (the claim `spiking_reservoir.py`'s own docstring
  attributes to Sato et al. 2025). Sweep `tt_rank` over **{4, 8, 16, 32}**
  (default 8), other settings at defaults, ≥3 seeds each, reporting normalized
  entanglement entropy, % permanently silent units, and mean spike rate.
- **Confound, declared in advance.** With `tt_core_std=None` the core std is
  derived as a function of `R_int = tt_rank^(d-1)`, so raising `tt_rank`
  automatically shrinks the core std to hold the effective per-entry variance of
  `W_res` matched. The rank sweep is therefore **not** a pure variance sweep —
  and separately, the reported entropy is normalized by `log(r)` of the cut bond,
  so its denominator changes with rank too. Both effects will be stated with the
  numbers rather than left for a reader to discover.

### Explicit falsification condition (binding on all of A4-A6)

**If entanglement entropy CAN be moved into S̄ ∈ [0.1, 0.5] but downstream task
performance does not improve, that resolves the sibling project's open question
NEGATIVELY: the entanglement-entropy diagnostic does not predict task
performance for a spiking substrate, and it will be reported as such** — not
quietly dropped, and not reframed as a tuning failure. This is a live and
plausible outcome: `spiking_reservoir.py:entanglement_entropy`'s own docstring
already warns that Sato et al.'s band was validated on rate-based regression and
not on a spiking reservoir, which is why the method calls it "a diagnostic to
log, not a hard gate".

**Reporting rule for all three:** full swept curves, every candidate seed, every
setting — including the ones where the spike rate collapses. Never a single
favourable point. Any measurement whose method has a limitation gets the
limitation printed next to the number.
