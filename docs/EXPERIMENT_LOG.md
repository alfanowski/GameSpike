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

---

## 12. A4-A6 RESULTS

Appended beneath §11, never edited into it, per §11's own rule. **Three of the
five pre-registered hypotheses are disconfirmed and they are reported here in the
same detail as the one that held.** The headline is uncomfortable and is stated
first: neither of the two construction knobs §11 pre-registered (`spectral_radius`,
`tt_rank`) moves the reservoir's dynamical regime at all, and the real defect was
somewhere nobody had pre-registered — the *input* to the reservoir, not the
reservoir.

Unless stated otherwise every measurement is on `reservoir_seed0`-family models at
the production geometry (`reservoir_size=8192`, `tt_rank=8`, `tt_n_cores=4`,
`beta=0.9`, threshold 1.0) and every "silent" figure is measured against **6,000
real observation steps** — 3,000 collected under the trained policy
(`checkpoints/reservoir_seed0/step_1000064.pt`) and 3,000 under a uniform-random
policy, pooled.

### A4a — mechanism CONFIRMED, set equality DISCONFIRMED

The predicted mechanism holds exactly, in the strong direction:

- **`dead \ silent` = 0. No exceptions.** Every single dead-gradient `in_proj`
  column belongs to a unit that never fires. There is no column that died for an
  optimiser/clipping reason, which rules out the competing explanation §11 named
  (the readout being frozen out by global clipping) as a *source of dead columns*.

The predicted **set equality does not hold**, and the converse direction fails
badly:

| set | size | fraction of 8192 |
|---|---|---|
| dead `in_proj` columns (Adam `exp_avg_sq` exactly 0) | **865** | 10.5591% |
| units silent over the 6,000-step probe | **3,721** | 45.42% |
| Jaccard(dead, silent) | **0.2325** | — |
| Jaccard(dead, silent under **all 11** sampled embeddings) | **0.6600** | — |

So "silent ⇒ dead" is true and "dead ⇒ silent" is true, but "silent ⇒ dead" in the
*set* sense is false: most silent units are silent under the *final* embedding
while having fired at some earlier point in training, and a unit only needs to
have fired once, ever, to have received gradient. Restricting the silent set to
units silent under **all 11 sampled embedding snapshots** raises the Jaccard from
0.2325 to 0.6600, which is the direct evidence for that reading — but 0.6600 is
still not 1.0, and the honest conclusion is that **§11's "one root cause" framing
is right about the mechanism and wrong about the arithmetic**: fixing the silent
units would not, on its own, have recovered a 15.28% dead budget, because the dead
budget is not 15.28%.

**The dead set only ever SHRINKS, and it is perfectly nested.** Across the ten
checkpoints of `reservoir_seed0`, `dead(t+1) ⊂ dead(t)` at every one of the 9
transitions, with `newly_dead = 0` at all 9 — no column ever dies after training
starts; columns only wake up.

| step | dead columns | % of 8192 |
|---|---|---|
| 100096 | 2,147 | 26.2085% |
| 500480 | 1,329 | 16.2231% |
| 1000064 | **865** | **10.5591%** |

**Correction to a previously-recorded figure in this ledger.** §11.0 records
"1,329 columns (16.22%) … 21,264 parameters (15.28% of the 139,179 trainable
budget)" as a property of `checkpoints/reservoir_seed0/step_1000064.pt`. **That
attribution is wrong: 1,329 / 16.22% / 15.28% is the value at `step_500480`, not
at the final checkpoint.** The correct final-checkpoint figure is **865 columns
(10.5591%), 13,840 parameters, 9.9440% of the trainable budget.** The §11.0 text
is left standing as written, per the append-only rule; this paragraph is the
correction of record.

### A4b — CONFIRMED IN DIRECTION, PALLIATIVE IN MAGNITUDE

Recalibrating the embedding scale against real observation statistics does reduce
the silent fraction, as predicted — and it cannot fix the problem.

| embedding scale | silent fraction |
|---|---|
| 1.0 (default) | 42.38% |
| 2.33 (derived from real observation statistics) | 30.60% |
| … sweep continues … | … |
| 32.0 | hard floor near **20%** |

**A hard floor near 20% silent persists even at 32x the default scale.** Scale is
a multiplier: it multiplies the informative (AC) and uninformative (DC) parts of
the drive together, so it can never change their *ratio*, and it is the ratio that
strands units on the wrong side of threshold. §11's cost-if-wrong clause said an
insensitive silent fraction would mean "silence is a property of the recurrent
regime rather than of the drive" — that inference turns out to be a false
dichotomy, and A5/A6 below show the recurrent regime is not the lever either. It
is a third thing: the *offset* of the drive.

**The code's two stated calibration targets are mutually incompatible under real
observations.** `models/policy_value_reservoir.py` documented both "induced
input-current std ≈ 0.3" and "mean spike rate ≈ 2%". Under real observations the
induced input-current std at the default init is **0.128683**, not the ~0.3163 the
comment claimed — that figure was measured against synthetic N(0,1) inputs, which
the file itself flagged with an explicit `# KNOWN GAP:` comment. Driving the std
up to 0.3 pushes the spike rate far above the healthy band; holding the spike rate
at ~2% keeps the std near 0.13. **The spike-rate target is the one that
corresponds to healthy dynamics** and is the one now calibrated against; the 0.3
figure is retained in the source only as the historical record of how the scalar
was originally chosen.

### A4c — DISCONFIRMED

With the embedding held fixed and only the reservoir seed varied:

- silent fraction spans **2.56 percentage points** (28.27% – 30.84%);
- normalized entanglement entropy spans **0.98665 – 0.99417**.

**Reservoir seed selection is not worth the methodological complexity.** The
sibling project's Task 3.3 selection procedure would be selecting among candidates
that differ by less than three points on the criterion, which is noise next to the
44-point effect A4b/§12's root-cause section identifies. §11's cost-if-wrong
clause applies as written: §4's "seeds genuinely produce different frozen
reservoirs" remains true bitwise and is **dynamically irrelevant**.

### A5 — DISCONFIRMED, with a proof rather than a null result

Sweeping `spectral_radius` over {0.7, 0.85, 0.95, 1.0, 1.05}: **normalized
entanglement entropy is numerically identical to 5 decimal places at every
setting, for every seed.** Not "insensitive" — identical.

**This is provable, not empirical.** On the TT path with `tt_core_std=None`,
`spectral_radius` enters construction *only* through the derived scalar
`s = (spectral_radius² / (N·R_int))^(1/2d)`, which multiplies every i.i.d. Gaussian
core by the same number. Normalized entanglement entropy is computed from the
*normalized* Schmidt spectrum `p = σ²/Σσ²` of the mixed-canonical centre core, and
a global rescaling of all cores rescales every singular value by a common factor,
which cancels in that normalization. **Multiplying all cores by 1000 changes S̄ by
2.8e-11** — that is the check, and it is the whole explanation. §11's mechanism
note anticipated a *weak* dependence via the 1/8 exponent; the truth is stronger,
the dependence is exactly zero.

**Silent fraction moves OPPOSITE to the hypothesis.** H5 predicted that lowering
the spectral radius toward the ordered regime would improve firing health:

| spectral_radius | silent fraction |
|---|---|
| 0.7 | **53.11%** |
| 1.05 | **43.96%** |

Lowering the radius makes it *worse*, because the recurrent drive is part of what
pushes hyperpolarised units back toward threshold. §11's cost-if-wrong clause
therefore binds in full: **the Phase 1 reservoir cannot be tuned into the
published productive band through spectral radius at all**, and the reason is
structural, not a tuning failure.

### A6 — DISCONFIRMED

Sweeping `tt_rank` over {4, 8, 16, 32}, three seeds each: **S̄ spans only
0.96221 – 0.99596.** There is no order-to-chaos transition. The bond dimension
does not behave as a second criticality knob in this construction.

The reason is visible in the Schmidt spectrum itself. At the middle bond the
normalized spectrum is **near-flat — 0.16881 … 0.09403 against 0.125 for a
perfectly uniform 8-dimensional bond** — which is exactly what i.i.d. Gaussian
cores generically produce: a near-maximally-entangled cut, hence S̄ pinned near 1.

**Conclusion, stated as strongly as the evidence supports: no construction that
keeps i.i.d. Gaussian cores can reach the productive band S̄ ∈ [0.1, 0.5].**
Reaching it needs structured (correlated, or low-rank-biased) cores, which is a
different construction and not a knob on this one. The `tt_core_std=None` confound
§11 declared in advance is real but is not what produced this result — the result
survives it, because the spectrum shape, not its scale, is what pins the entropy.

### The falsification condition was UNTESTABLE AS FORMULATED

§11's binding falsification condition reads: *"If entanglement entropy CAN be
moved into S̄ ∈ [0.1, 0.5] but downstream task performance does not improve, that
resolves the sibling project's open question NEGATIVELY."*

**Its antecedent never occurs.** A5 and A6 together establish that entanglement
entropy cannot be moved into that band by either pre-registered knob — provably
so, in A5's case. A conditional whose antecedent is false is not evidence about
its consequent. **The sibling project's open question therefore remains OPEN,
resolved in neither direction.** It is recorded here as a formulation failure of
the pre-registration: the condition was written assuming the knobs worked, and a
pre-registration that only produces a verdict when its manipulation succeeds is
not a complete pre-registration. This is not being reframed as a tuning failure
and it is not being quietly dropped.

### Root cause — NOT PRE-REGISTERED, found post hoc

**Stated plainly: nothing below was pre-registered.** It was found while
investigating why A4b's scale sweep had a floor, it is a post-hoc explanation, and
it should be read with the discount that deserves. What justifies recording it at
this weight is that it is *measured*, it is *algebraically exact*, and it makes a
falsifiable prediction that was then checked across 8 seeds.

**The observation is DC-dominated.** Over the 6,000 real steps:

- `||E[obs]||² = 1.331336` against `E||obs||² = 1.713384` — **77.70% of the
  observation's energy is its own mean.** Real dimensions are mostly non-negative
  with large means (the level timer, lives, powerup state and the on-ground flag
  are near-constant or slowly-drifting; three slots are hardcoded zero).
- Consequently **76.11% of the reservoir's input-current variance is DC.**

**The LIF neuron amplifies exactly the useless component.** With `beta=0.9` a
constant input is integrated to steady state with gain `1/(1-beta) = 10.0`, while
a zero-mean fluctuating input accumulates only to `1/sqrt(1-beta²) = 2.2942` — a
**4.3589x amplification favouring DC** over AC.

**The result is a frozen per-unit membrane offset.** Because `W_in` is frozen, each
unit's DC offset is fixed for the entire run: std **0.943583** across units, range
**[-3.5080, +3.4847]**, against a firing threshold of 1.0. Measured directly:

- **1,223 units (14.93%) sit permanently below −threshold — silent forever**,
  whatever the input does;
- **1,188 units (14.50%) sit permanently above threshold — saturated.**

That is the floor A4b's sweep hit, and it explains why scale cannot clear it:
scale multiplies DC and AC together and leaves the offsets in place.

**The fix is a bias initialisation, and it is exact.** The embedding is LINEAR, so

    W @ (obs − μ) ≡ W @ obs + (−W @ μ)

identically — **verified numerically to a max absolute difference of 5.96e-08.**
Centring the input is therefore *precisely* the bias initialisation
`embedding.bias := −(W @ μ)`. It costs **zero new parameters** (the bias already
existed), changes no shapes (every existing `state_dict` still loads), and leaves
the bias **trainable**, so it is a starting point rather than a constraint.

**Measured effect, mean over 8 seeds:**

| | silent-unit fraction | mean spike rate | saturated units |
|---|---|---|---|
| default init | 44.7403% | 0.024351 | (see above) |
| centred init | **1.7532%** | **0.020912** | **zero** |

The spike rate lands inside the ~2% band `models/spiking_reservoir.py` documents as
healthy.

**Shipped as `--embed-init-mode {legacy,centered}` (default `legacy`) and
`--embed-scale` (default 1.0), applied IDENTICALLY TO BOTH ARMS.** The default
path is bit-identical to the code the 20 completed runs and 200 checkpoints were
produced under (verified: same `grad_norm` sequence, same trained `state_dict`,
same Adam state on both arms). Both settings are recorded in every checkpoint and
every JSONL log line, and `load_checkpoint` reads them with `.get(...)` defaults so
files that predate them still load.

Both arms get the treatment because **input centring is a generic initialisation
correction, not a reservoir-specific advantage**. It is expected to help the
reservoir arm more — the baseline's embedding feeds a **trainable GRU** that can
learn to absorb a constant DC offset, whereas the reservoir's feeds a **frozen
nonlinearity** that cannot — but that expected asymmetry is a *result*, not a
licence to apply the treatment asymmetrically. A treatment only one arm receives
is not a control.

**Independent reproduction in this repository** (`tests/test_embedding_centering.py`,
against the committed 6,000-step observation fixture, seed *s* used as both the
global and the frozen-reservoir seed, i.e. what `--seed s` actually produces):

| init | silent fraction (8-seed mean, range) | mean spike rate | saturated |
|---|---|---|---|
| `legacy`, scale 1.0 | 45.5917% (43.3594 – 47.8149) | 0.022551 | 0 |
| `legacy`, scale 2.33 | 32.2113% (28.8574 – 34.6191) | 0.086188 | 0 |
| `legacy`, scale 3.0 | 30.3329% (27.1729 – 32.5562) | 0.117302 | 0 |
| `centered`, scale 1.0 | 65.9454% (58.7769 – 73.3887) | 0.000474 | 0 |
| **`centered`, scale 3.0** | **2.0523% (1.2939 – 2.6489)** | **0.018482** | **0** |

Note the two rows that are *not* the fix: `legacy` at scale 3.0 buys firing by
brute force and overshoots the healthy band five-fold, and `centered` at scale 1.0
removes the DC drive without replacing it and starves the reservoir. **The bias and
the gain are only a fix together**, which is why both knobs shipped.

### Limitations, stated plainly

- **6,000 observation steps, not 1,000,064.** "Silent" throughout §12 means "did
  not fire once in 6,000 steps". A unit firing at a true rate of 1e-4 would read
  as silent here. Every silent fraction in this section is therefore an upper
  bound on the permanently-silent fraction, and the 14.93% figure derived from the
  DC offset (which is a statement about the offset, not about a firing count) is
  the only genuinely *permanent* number in the section.
- **11 embedding snapshots out of 7,813 optimizer updates.** A4a's "silent under
  all 11 sampled embeddings" set — the one that lifts the Jaccard to 0.6600 — is
  sampled at ~0.14% of the trajectory. The nesting result (`dead(t+1) ⊂ dead(t)`,
  `newly_dead = 0`) is likewise established only at the 9 checkpoint boundaries; a
  column could in principle die and revive between two of them.
- **The DC offset is policy-dependent, and a pooled bias is a compromise.** A bias
  fitted on the pooled data gives **1.37% silent on pooled data, 4.61% on
  trained-policy-only data, and 14.82% on random-policy-only data** — against
  **49.30%** and **53.99%** for the current default on those same two splits. The
  fix is a large win on every split and a *smaller* win on the split furthest from
  the data it was fitted on, which is exactly the shape one should expect and
  exactly the shape that should be watched. The observation distribution also
  shifts *during* training as the policy improves, so the offset a run needs at
  step 1,000,000 is not the one it needs at step 0.
- **The bias is trainable, and whether it adapts was NOT verified.** In principle
  the run can move the bias to track the policy-dependent shift above; no
  measurement in this section shows that it does. That is an open question, not a
  claim.
- **No task-performance result.** Everything in §12 is a *construction* diagnostic.
  Whether a healthier reservoir produces a better agent is unmeasured, and the
  pre-registered falsification condition that would have connected the two was
  untestable as formulated (above). Nothing here licenses a claim about the Phase 1
  arm comparison.

---

## 13. Development workflow change

Everything up to and including §12 was pushed directly to `main` via verified
clean fast-forwards. That was the workflow in force at the time; it is not
being revised retroactively and the history stands as intentional.

**From this point forward, changes go through a feature branch + pull request
against `main`** instead of direct pushes. Convention:

- **Substantive or headline-bearing changes** — anything that revises a
  published conclusion (e.g. a new revision of `docs/RESULTS.md`) — are left
  as an **open** PR for the repository owner to review and are not
  self-merged.
- **Mechanical infrastructure changes** (run drivers, test fixtures, ledger
  appends, tooling) still go through a branch and a PR for the audit trail,
  but may be merged once the test suite passes.

`gh` is authenticated as `alfanowski` with `repo` scope, so a future session
can use `gh pr create` directly rather than setting up auth from scratch.

---

## 14. Session handover, 2026-08-20 ~20:10 CEST — and what is on disk that §13 does not mention

Recorded at the start of a new orchestrator session, before any new measurement,
because the previous session ended without writing a closing entry and the gap
between "what the docs say" and "what is on disk" is itself a finding.

### 14.1 What happened to the previous session

The orchestrator session that produced everything up to §13 stopped writing at
**17:55:38 CEST** (last file write: `checkpoints/reservoir_seed1_clipemb/step_300032.pt`).
At 20:05 CEST the following was verified directly rather than assumed:

- zero training processes running (`ps aux`, no `python`/`train.py`/`pyboy` matches);
- no file write anywhere in the repository since 17:55:38;
- no commit on `main` since `133e09e` (17:26 CEST).

The working hypothesis for the stop is an unattended permission/approval prompt with
nobody present to answer it. **It is a hypothesis, not a diagnosis** — no log of that
session is available from inside the repository, which is exactly why this ledger
exists and why this entry is being written before any new work.

**Collision protocol adopted, in case that session ever resumes.** Before any write
to `checkpoints/` or `results/`, re-verify that no process other than this session's
is writing there. If one appears, do not share a path with it — use a distinct
directory name and record the collision here. This entry is the record that the
check was performed and was clean at 20:05.

### 14.2 Undocumented data on disk: the pilot ran 5.5x further than §6.4 of `RESULTS.md` reports

`docs/RESULTS.md` §6.4 describes the corrected-configuration pilot as **3 seeds at
425 PPO updates (54,400 env steps, 5.4% of a full run)** and labels it, correctly,
as not a result. That description was accurate when written. **It is now stale:**
the pilot kept running and reached **update 2344 = step 300,032 = 30.0% of a full
1,000,064-step run**, across **nine** run directories, not three:

| directory pattern | `--grad-clip-mode` | `--embed-init-mode` | `--embed-scale` |
|---|---|---|---|
| `checkpoints/reservoir_seed{0,1,2}_clip/` | `per-group` | `legacy` | 1.0 |
| `checkpoints/reservoir_seed{0,1,2}_emb/` | `global` | `centered` | 3.0 |
| `checkpoints/reservoir_seed{0,1,2}_clipemb/` | `per-group` | `centered` | 3.0 |

Each holds `train_log.jsonl` (2,344 lines) and checkpoints at `step_100096.pt`,
`step_200192.pt`, `step_300032.pt`. Config fields are recorded in every JSONL line,
so the table above is read off the data, not inferred from the directory name.

Two consequences, both stated so they are not re-derived later:

1. **This is a 2x2 factorial, not a one-armed pilot.** The fourth cell
   (`global` + `legacy`) is the v1 condition and is already on disk at
   `checkpoints/reservoir_seed{0..9}/`. Comparisons must be made at **matched update
   index** (update 2344) from the JSONL rather than at matched checkpoint filename,
   because the pilot's checkpoint boundaries (`step_300032`) and v1's
   (`step_300288`) do not coincide.
2. **`RESULTS.md` §6.4's "425 updates / 5.4%" figures are left standing** per the
   append-only rule. They are not wrong about what they measured; they are an
   incomplete description of what the pilot eventually produced. The correction of
   record is this subsection.

### 14.3 The instability that has to be cleared before the full matrix is launched

`checkpoints/reservoir_seed1_clipemb/train_log.jsonl` shows, on three consecutive
updates near the end of the pilot, `grad_norm_groups.embedding` moving
**474,381 -> 74,508,156,928 -> 3,571,024,640**. That field is the **pre-clip** norm,
so this is the §6.2 explosion still present under the configuration intended to
correct for it. The readout group's norm on those same lines stayed at roughly
10-25, which is consistent with per-group clipping doing its job — **but consistent
with is not the same as verified**, and committing 20 full-length runs (~4.5 h of
this machine) to a configuration whose containment has not been checked would be
exactly the kind of unforced error this ledger exists to prevent.

A blocking go/no-go diagnostic is therefore being run first, on data already on
disk, with no new training. Its hypotheses are pre-registered in §14.4 below.

### 14.4 Pre-registered: the go/no-go diagnostic (declared before measurement)

Same rule as §11: **reported regardless of outcome**, wrong statements stay on the
page with corrections beneath them.

- **H14a — the embedding fix survives training.** The centred initialisation's
  silent-unit suppression is not merely an initialisation property; the *trained*
  embedding at step 300,032 still yields a healthy reservoir.
  **Falsified if** the mean silent-unit fraction over seeds 0-2, measured on the
  committed `tests/data/real_obs_6000.npy` fixture using the trained `embedding`
  weights from `reservoir_seed{s}_clipemb/step_300032.pt`, **exceeds 15%**.
  This is the open question §12's limitations list explicitly declined to claim
  ("the bias is trainable, and whether it adapts was NOT verified"); the pilot
  checkpoints are the first data able to answer it.
- **H14b — per-group clipping contains the explosion.** The readout's effective
  optimizer step stays in a healthy range despite the embedding's exploding
  pre-clip norm.
  **Falsified if** the readout's median `|m_hat| / sqrt(v_hat)` (or, if
  reconstructible, median `||dp||/||p||` for one Adam step) under `clipemb` is
  **below 1e-4** — i.e. still within an order of magnitude of v1's frozen-readout
  pathology (1.9034e-05) rather than near the healthy baseline GRU (1.346e-01 /
  4.273e-04 respectively for the two statistics).
- **H14c — the 2x2 factorial is descriptive only.** Mean `mean_extrinsic_reward`
  over updates 1876-2344 for each of the four cells, three seeds each, reported
  per seed. **No p-value will be computed and no arm claim will be made from it**:
  three seeds at 30% of a run is below this project's own stated bar (§2), and
  saying so in advance is the point of writing it down here.

**Decision rule, fixed in advance:** the full 10-seed x 2-arm matrix is launched
under `--grad-clip-mode per-group --embed-init-mode centered --embed-scale 3.0`
**only if H14a and H14b both survive**. If either is falsified, the matrix is not
launched on that configuration and the reason is recorded here before anything
else is run.

### 14.5 Pre-registered: A7 — does the corrected input also fix the dead-gradient budget?

Declared **before the v2 runs exist**, which is the only moment at which declaring
it is worth anything.

§12/A4a established two things that together make a sharp prediction: every
dead-gradient `in_proj` column belongs to a unit that never fires
(`dead \ silent = 0`, no exceptions), and the dead set only ever shrinks
(`dead(t+1) ⊂ dead(t)`, `newly_dead = 0` at all 9 transitions of `reservoir_seed0`).
If silence is the *cause* of deadness, then an initialisation that removes most of
the silence should remove most of the deadness.

- **H7 (prediction).** Under `--embed-init-mode centered --embed-scale 3.0`, the
  dead-gradient `in_proj` column count at step 1,000,064, averaged over the ten v2
  reservoir seeds, is **below 2% of 8192 (i.e. fewer than ~164 columns)**, against
  the v1 measured value of **865 columns (10.5591%, 9.9440% of the trainable
  budget)** at the same step.
- **Falsified if** that mean is **at or above 5% of 8192 (~410 columns)** — a
  result which would mean silence is a *marker* of deadness rather than its cause,
  and that some second mechanism strands columns without gradient.
- **Ambiguous band, declared in advance so it cannot be spun either way:** a mean
  between 2% and 5% confirms the direction while falsifying the magnitude, and will
  be reported in exactly those words.
- **Test:** same procedure as A4a — Adam `exp_avg_sq` exactly 0 over dim 1 of
  `in_proj.weight`, read from the final checkpoint's stored optimizer state. Also
  report the per-seed spread, not only the mean, and the nesting property
  (`newly_dead = 0`) which should hold or fail independently of the magnitude.
- **Cost if wrong.** If H7 is falsified, the "one root cause" framing §11 advanced
  and §12/A4a already partially retracted is wrong a second time, and the honest
  conclusion becomes that the dead-parameter criticism is **not** solved by the
  input fix and remains an open structural defect of the architecture. That
  sentence will be written if that is what the numbers say.

### 14.6 The third architectural criticism is NOT expected to be solved, and that is stated up front

The three standing architectural criticisms of this design are (a) structurally
silent units, (b) a chunk of the trainable budget that never receives gradient,
(c) entanglement entropy indicating deep chaos in the reservoir dynamics.

- **(a) has a root cause and a fix** (§12, root-cause subsection); whether the fix
  survives training is H14a and whether it survives at task scale is v2.
- **(b) has a prediction under test** (H7 above), not yet an answer.
- **(c) is, on the present evidence, not fixable by any knob this construction
  exposes.** A5 proved `spectral_radius` enters only as a global rescaling of all
  TT cores, which cancels exactly in the normalised Schmidt spectrum (changing all
  cores by 1000x moves S-bar by 2.8e-11); A6 found `tt_rank` spans only
  0.96221-0.99596 with no order-to-chaos transition, because i.i.d. Gaussian cores
  generically produce a near-flat Schmidt spectrum. The conclusion recorded in §12
  — **no construction that keeps i.i.d. Gaussian cores can reach the productive
  band S-bar in [0.1, 0.5]** — stands, and it names the only remaining route:
  structured (correlated, or low-rank-biased) cores, which is a different
  construction rather than a setting on this one.

**This session does not expect to solve (c), and says so before trying.** If a
structured-core construction is attempted, it will be pre-registered as its own
ablation with its own falsification condition beneath this entry, and a negative
outcome will be reported as a negative outcome. The standing instruction on this
project is that a negative result reported honestly is the deliverable; a positive
result obtained by selection is not.

### 14.7 Pre-registered: A8 — the structured-core route to the entanglement-entropy criticism

Declared **before the construction is written and before any spectrum is
computed**, for the same reason as §11: the commit timestamp is the evidence.

§14.6 records why this ablation exists. A5 proved `spectral_radius` cannot move
normalised entanglement entropy at all on this construction (it enters only as a
global rescaling of every TT core, which cancels exactly in the normalised Schmidt
spectrum). A6 found `tt_rank` moves it across only 0.96221-0.99596. §12's stated
conclusion — **no construction that keeps i.i.d. Gaussian cores can reach the
productive band S-bar in [0.1, 0.5]** — is a claim about i.i.d. Gaussian cores
specifically, and it names its own escape route: structured cores. A8 is the test
of that escape route, and it is the difference between "we could not fix (c) with
the two knobs we happened to pre-register" and "we tried the construction our own
analysis pointed at, and here is what happened."

**Why a near-flat spectrum is the thing to attack.** S-bar is computed from the
normalised Schmidt spectrum `p = sigma^2 / sum(sigma^2)` at the middle bond. A6
measured that spectrum as near-flat (0.16881 … 0.09403 against 0.125 for a
perfectly uniform 8-dimensional bond), which is what i.i.d. Gaussian cores
generically produce, and a flat `p` is exactly what pins S-bar near 1. Lowering
S-bar therefore requires a *decaying* Schmidt spectrum, which requires breaking the
i.i.d. assumption along the bond index rather than rescaling it.

- **H8a (tunability).** Introducing a geometric bond profile — scaling bond index
  `r` of every TT core by `lambda^r` for a decay parameter `lambda` in (0, 1],
  which reduces to the existing i.i.d. Gaussian construction exactly at
  `lambda = 1.0` — makes normalised entanglement entropy tunable.
  **Prediction:** over a sweep `lambda` in {1.0, 0.9, 0.7, 0.5, 0.3, 0.1}, three
  reservoir seeds each, at fixed `tt_rank=8`, `tt_n_cores=4`, `reservoir_size=8192`,
  S-bar spans an absolute range **of at least 0.3** and **enters the band
  [0.1, 0.5] for at least one value of lambda**.
  **Falsified if** the sweep's range is below 0.3, **or** if it never enters
  [0.1, 0.5]. Either outcome means the structured-core route fails too, and
  criticism (c) is then reported as unsolved by two independent constructions
  rather than by one.
- **H8b (the fix must not silently undo the other fix).** A construction that
  buys a lower S-bar by destroying the reservoir's firing health is not a fix.
  **Prediction:** at whichever `lambda` first lands S-bar inside [0.1, 0.5], the
  silent-unit fraction on the committed `tests/data/real_obs_6000.npy` fixture,
  under `--embed-init-mode centered --embed-scale 3.0`, stays **below 10%** and the
  mean spike rate stays inside the documented healthy band (roughly 1-3%).
  **Falsified if** silent fraction is at or above 10%, or the spike rate leaves
  that band — in which case A8 is reported as "S-bar is movable, but not without
  cost", with the trade-off quantified rather than hidden.
- **Explicitly NOT claimed, whatever A8 returns.** A8 is a **construction**
  diagnostic. It cannot say whether a less chaotic reservoir produces a better
  agent, because that requires training runs A8 does not include. This is the exact
  mistake §12 already recorded once: the A4-A6 falsification condition was written
  as a conditional on task performance and became untestable when its antecedent
  failed. **A8's falsification conditions are therefore stated purely over
  construction quantities that A8 itself measures**, and any performance claim would
  need its own pre-registered training ablation, not yet declared.
- **Cost if wrong.** If H8a holds, this project has a genuinely new construction
  knob and the sibling project's open question becomes testable for the first time
  (it is currently OPEN in neither direction, per §12). If H8a is falsified, the
  honest headline is that the reservoir's dynamical regime is not reachable by any
  construction this project has been able to devise, which is a real and reportable
  limitation of the architecture rather than of the experiment.

### 14.8 Correction of record: `RESULTS.md` §12's v2 recipe is incomplete in a way that would have reproduced a DISCONFIRMED configuration

`docs/RESULTS.md` §12 ("What v2 will add") specifies the corrected runs as
*"full-length runs under `--grad-clip-mode per-group` and `--embed-init-mode
centered`, applied identically to both arms"*. **It does not mention
`--embed-scale`, whose default is 1.0.**

Followed literally, that recipe launches the configuration §12 of this ledger
already measured and rejected:

| configuration | silent fraction (8-seed mean) | mean spike rate |
|---|---|---|
| `legacy`, scale 1.0 (the v1 condition) | 45.5917% | 0.022551 |
| **`centered`, scale 1.0 (what §12's recipe literally specifies)** | **65.9454%** | **0.000474** |
| `centered`, scale 3.0 (the actual validated fix) | **2.0523%** | 0.018482 |

`centered` at scale 1.0 is not a weaker version of the fix — **it is worse than
doing nothing**, because centring removes the DC drive without replacing it and
starves the reservoir. This ledger's §12 states it plainly ("**the bias and the
gain are only a fix together**, which is why both knobs shipped"); `RESULTS.md`
§12 simply omits the second half.

**Resolution, and the evidence for it.** The nine pilot runs on disk record their
own configuration in every JSONL line, and the `clipemb` runs read
`"embed_init_mode": "centered", "embed_scale": 3.0`. So the previous session did
in fact run the validated pairing; only the write-up is incomplete. **v2 is
launched at `--embed-init-mode centered --embed-scale 3.0`**, and `RESULTS.md`
§12's sentence is corrected here rather than edited there, per the append-only
rule. The v2 section of `RESULTS.md` will state both knobs explicitly.

This is recorded at this weight because it is the exact failure mode a
reproduction attempt hits: a reader following the published recipe would have got
a starved reservoir, measured a worse-than-v1 result, and had no way to know the
recipe was missing a term.

### 14.9 Output layout for v2, and why it needs no code change

`scripts/run_eval_matrix.py` and `analysis/aggregate_results.py` have **no
`run_tag` support at all** (verified: zero occurrences of `run_tag`/`run-tag` in
either file or their tests). Three places hardcode the untagged shape:

- `analysis/aggregate_results.py:478` — `_CHECKPOINT_DIRNAME_RE` is anchored
  `^(?P<arm>[A-Za-z0-9]+)_seed(?P<seed>\d+)$`; a `_v2` suffix matches nothing, and
  the `[A-Za-z0-9]+` arm group cannot contain an underscore either.
- `analysis/aggregate_results.py:988` — `build_eval_manifest` composes
  `os.path.join(checkpoint_dir, f"{arm}_seed{seed}")` with no tag parameter.
- `scripts/run_eval_matrix.py:190` — `resolve_init_checkpoints` composes the same
  shape for `step_0.pt`.

Evaluation output filenames (`eval_{arm}_seed{seed}_{regime}.json`) are equally
tag-free, so v2 results written into `results/` would also **collide with v1's
files** rather than sit beside them.

**Decision: put the version coordinate in the PARENT directory, not in the run
directory name.** Every v2 run directory is then still named exactly
`{arm}_seed{n}`, every existing regex matches unchanged, and no code that the v1
results depend on for their reproducibility is touched:

| | v1 (untouched) | v2 |
|---|---|---|
| trained runs | `checkpoints/{arm}_seed{n}/` | `checkpoints_v2/{arm}_seed{n}/` |
| untrained controls | `checkpoints_init/{arm}_seed{n}/` | `checkpoints_v2_init/{arm}_seed{n}/` |
| evaluation output | `results/{sel}/` | `results_v2/{sel}/` |

`--run-tag` is deliberately NOT used for v2. It remains the right tool for a
one-off diagnostic re-run that never needs to reach the analysis path (which is
exactly what the nine pilot runs are), and the wrong tool for a full matrix that
does.

**Consequence that must not be forgotten: v2 needs its own untrained controls.**
`RESULTS.md` §2.3's `init` selection and §4.1's "the untrained arms are
statistically indistinguishable" control both rest on `--steps 0` checkpoints.
The existing ones in `checkpoints_init/` were built under the **legacy**
embedding init. Since the centred init is applied at construction time, a v2
comparison scored against legacy-init controls would be comparing against the
wrong control. Twenty fresh `--steps 0` runs are therefore part of the v2 matrix,
not an afterthought.

### 14.10 Launch-readiness verification (measured, 2026-08-20 ~20:20 CEST)

- **Test suite: 224 passed, 0 failed** (16.4s) with
  `MARIO_LAND_ROM_PATH="/Users/alfanowski/Desktop/Super Mario Land (World).gb"`
  set. Without it, 130 passed / 94 skipped — the ROM-dependent tests skip
  silently, so **a run that reports 130 passed has not actually exercised the
  environment**. §1 of this ledger records 169 as the count at commit `990c5a1`;
  224 is the current figure.
- **ROM verified present** at `/Users/alfanowski/Desktop/Super Mario Land (World).gb`,
  65,536 bytes.
- **Disk**: 148 GiB free. Measured per-run footprint is 18.41 MB (baseline) and
  30.70 MB (reservoir) including `train_log.jsonl`, so the 20 v2 runs cost ~491 MB.
  Not a constraint.
- **Memory is the one unverified risk.** No document in this repository records a
  per-process RSS for a PyBoy + torch training process, and the machine has 16 GB.
  Ten concurrent processes were run successfully during v1, so the configuration is
  empirically known to fit, but it was never measured. It is measured during the v2
  launch and recorded below rather than assumed.
- **No committed training launcher existed.** v1's parallelism was ad hoc, which is
  the direct cause of the §9 hazards. `scripts/run_training_matrix.py` is being
  written with tests for v2 so this is reproducible rather than remembered.

### 14.11 v2 execution checklist — written to be runnable by someone who was not here

The previous session stopped mid-experiment and left no closing entry, which cost
this session roughly an hour of reconstruction. This subsection exists so that does
not happen twice. **Every command below has been checked against the actual argparse
surface of the file it invokes, not against documentation.** Run from the repository
root with the venv python at `.venv/bin/python`.

Standing preamble for every command:

```
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export ROM="/Users/alfanowski/Desktop/Super Mario Land (World).gb"
```

**Step 0 — collision check (do this first, every time).** `ps aux | grep -i train`
must show nothing you did not start, and nothing under `checkpoints_v2/` may have an
mtime newer than your own last write. §14.1 explains why.

**Step 1 — untrained controls, both arms, centred init (20 fast runs, `--steps 0`).**
Required because `RESULTS.md` §2.3's `init` selection and §4.1's equivalence control
must be built under the SAME initialisation as the trained arms; the existing
`checkpoints_init/` was built under `legacy` and is the wrong control for v2 (§14.9).

```
for arm in baseline reservoir; do for s in $(seq 0 9); do
  .venv/bin/python -m training.train --arm $arm --rom "$ROM" --steps 0 --seed $s \
    --checkpoint-dir checkpoints_v2_init \
    --grad-clip-mode per-group --embed-init-mode centered --embed-scale 3.0
done; done
```

**Step 2 — the reservoir arm, 10 seeds, full length.** Long pole, ~2.1-3.5 h at 10-way
parallelism (§14.10).

```
.venv/bin/python scripts/run_training_matrix.py \
  --arms reservoir --seeds 0-9 --rom "$ROM" \
  --steps 1000000 --checkpoint-every 100000 --checkpoint-dir checkpoints_v2 \
  --grad-clip-mode per-group --embed-init-mode centered --embed-scale 3.0 --jobs 10
```

**Step 3 — the baseline arm, 10 seeds, full length.** Identical flags, `--arms baseline`.
The clipping and embedding treatments are applied to BOTH arms deliberately: a
treatment only one arm receives is not a control (§12, and `train.py`'s docstring).

**Fallback if `scripts/run_training_matrix.py` does not exist or is broken** — the
launcher is a convenience, not a dependency. One run is:

```
.venv/bin/python -m training.train --arm reservoir --rom "$ROM" \
  --steps 1000000 --checkpoint-every 100000 --checkpoint-dir checkpoints_v2 --seed 0 \
  --grad-clip-mode per-group --embed-init-mode centered --embed-scale 3.0
```

Backgrounding ten of these by hand is what v1 did and is what produced the §9
hazards; if you do it, at minimum check each run directory does not already exist
before launching into it.

**Step 4 — evaluation matrix**, 120 evaluations, same protocol as v1 (`RESULTS.md`
§2.3): 30 episodes, eval seed 0, both recurrent-state regimes, three selections.

```
.venv/bin/python scripts/run_eval_matrix.py --rom "$ROM" \
  --episodes 30 --eval-seed 0 --jobs 8 \
  --checkpoint-dir checkpoints_v2 --init-checkpoint-dir checkpoints_v2_init \
  --results-dir results_v2
```

It is resumable and its writes are atomic (`os.replace`), so it is safe to kill and
restart. Add `--dry-run` first to see the job list.

**Step 5 — statistics.**

```
.venv/bin/python -m analysis.aggregate_results \
  --results-dir results_v2 --checkpoint-dir checkpoints_v2
.venv/bin/python -m analysis.aggregate_results \
  --results-dir results_v2 --checkpoint-dir checkpoints_v2 --json > /tmp/v2_report.json
```

Exact permutation over all C(20,10) = 184,756 splits, exact Mann-Whitney, percentile
bootstrap at 20,000 resamples with `default_rng(0)` — identical to v1, so the two
versions are comparable line for line.

**Step 6 — write `RESULTS.md` v2 BENEATH v1.** Never edit v1's numbers. v2 must mirror
v1's table shapes exactly (§3 primary result, §4.1/§4.2 controls, §5 per-step
decomposition, §8 efficiency) so the two are readable side by side, and must state the
`--embed-scale 3.0` term that §12 of `RESULTS.md` omitted (§14.8).

**Step 7 — open a PR and DO NOT merge it.** §13: headline-bearing changes are left open
for the repository owner. `gh` is authenticated as `alfanowski` with `repo` scope.

**Things that will bite you, all of them already paid for once:**
- The final checkpoint is `step_1000064.pt`, not `step_1000000.pt` (§2). Globbing for a
  round number silently matches nothing.
- `--embed-init-mode centered` WITHOUT `--embed-scale 3.0` is worse than doing nothing
  (§14.8).
- Do not `git checkout`/`git switch` in this working directory while training is live
  (§9) — use a worktree.
- Running the test suite without `MARIO_LAND_ROM_PATH` set silently skips 94 tests and
  still prints a pass (§14.10).

### 14.12 Progress log for the v2 run (appended live, so an interrupted run is legible)

Times are CEST, 2026-08-20. This subsection is written **as the run proceeds**, not
afterwards, specifically so that a session picking this up cold can tell the
difference between "not started", "running" and "finished and unrecorded" — the
exact ambiguity that cost this session an hour at §14.1.

- **20:05** — handover checks (§14.1). No processes, no writes since 17:55, clean.
- **20:12** — pre-registration of the go/no-go diagnostic and A7 committed (`0c2c1bd`)
  before any measurement was taken.
- **20:13** — the 120 v1 evaluation JSON files, previously untracked, committed
  (`4a2e0ce`). Every number in `RESULTS.md` v1 had been reproducible only from this
  one machine's disk. They are 480 KB and they are the raw evidence behind a
  published conclusion, so they now live in the repository. Checkpoints stay
  gitignored: 551 MB, and regenerable from a seed. Evaluation outputs are not.
- **20:14** — A8 pre-registered and committed (`942a333`).
- **20:16** — launch-readiness verified (§14.10): 224 tests pass, ROM present, disk
  ample, eval driver confirmed to accept `--checkpoint-dir` / `--init-checkpoint-dir`
  / `--results-dir` independently, which is what makes the §14.9 layout work with no
  code change at all.
- **20:17** — pre-flight: one `--steps 0` reservoir run under the full v2 flag set,
  into a throwaway directory, verified to write a loadable checkpoint recording
  `grad_clip_mode='per-group'`, `embed_init_mode='centered'`, `embed_scale=3.0`.
  Throwaway deleted.
- **20:19** — **Step 1 complete: all 20 untrained controls built** under the centred
  init at `checkpoints_v2_init/{arm}_seed{0..9}/step_0.pt`, 21 MB total, every one
  spot-checked to record the intended config. These are v2's `init` selection and
  its §4.1 equivalence control.

### 14.13 A caveat on H14a that is a property of the fixture, not of the fix

Noted before the diagnostic's numbers arrive, so it cannot be read as a reaction
to them.

H14a measures the silent-unit fraction of a **v2-trained embedding** against
`tests/data/real_obs_6000.npy`, which was collected under **v1** policies (3,000
steps under a legacy-init trained policy, 3,000 under a uniform-random policy).
Holding the observation window fixed and varying only the embedding is the right
controlled comparison for the question H14a actually asks — *did the trained
embedding drift away from the centring it was initialised with* — because it
isolates the one thing under test.

It is **not** a measurement of the silent fraction a v2 policy actually experiences.
§12's limitations already established that the DC offset is policy-dependent: the
same pooled bias measures **1.37% silent on pooled data, 4.61% on trained-policy-only
data and 14.82% on random-policy-only data**. A v2 agent's own observation
distribution is a third distribution, measured by none of those.

**Consequence, stated in advance:** a passing H14a licenses "the centring is not
undone by training", and does **not** license "the v2 reservoir is 2% silent in
situ". The in-situ number requires collecting a fresh observation fixture under a
v2-trained policy, which cannot be done until the v2 runs exist. If the v2 matrix
completes, that measurement is cheap and should be taken; it is recorded here as a
known gap rather than discovered later as an objection.

### 14.14 The pilot is a BIT-IDENTICAL PREFIX of the v2 reservoir runs — verified, not assumed

**Measured at 20:21 CEST, before the matrix was launched.** A 12,800-step smoke run
was executed under the exact v2 flag set:

```
--arm reservoir --seed 0 --steps 12800 --checkpoint-every 6400 \
--checkpoint-dir /tmp/gs_smoke \
--grad-clip-mode per-group --embed-init-mode centered --embed-scale 3.0
```

and its `train_log.jsonl` compared update-for-update against
`checkpoints/reservoir_seed0_clipemb/train_log.jsonl` on seven float fields
(`mean_reward`, `mean_extrinsic_reward`, `policy_loss`, `value_loss`, `entropy`,
`total_loss`, `grad_norm`):

- **100 updates x 7 fields = 700 values compared.**
- **0 exact mismatches. Worst absolute difference: 0.0.**

The only field that differs is `run_tag` (`null` vs `"clipemb"`), which is the
output-path coordinate and by construction touches nothing in the training loop.

**Three things follow, and the third is the one that matters.**

1. Training is deterministic given the seed, end to end, across process
   invocations and across a `--checkpoint-dir` change. §2 asserted this as a design
   property; this is the first time it has been checked against an independently
   produced run rather than argued from the code.
2. `--run-tag` and `--checkpoint-dir` are confirmed inert with respect to training
   dynamics. Only the output path moves.
3. **The nine pilot runs are not merely "suggestive of" the v2 runs — for seeds
   0-2 they are literally the first 2,344 updates of them.** Whatever the go/no-go
   diagnostic measures on `reservoir_seed{0,1,2}_clipemb` at update 2344 is exactly
   what the v2 reservoir runs will pass through at update 2344. That upgrades the
   diagnostic from a pilot-based inference to a direct observation of the first 30%
   of three of the ten runs about to be launched.

**Why the v2 runs are nonetheless started FRESH rather than resumed from the pilot
checkpoints.** `--resume-from` restores the model and the optimizer state but not
the emulator state or the RNG stream position, so a resumed continuation is *not*
bit-identical to the corresponding suffix of a single full-length run. Mixing three
resumed seeds with seven fresh ones would put a protocol difference inside the arm,
which is precisely the kind of heterogeneity this project's own control discipline
forbids. **And it would buy nothing:** the ten runs execute in parallel and the batch
finishes when its slowest member does, so three seeds starting at 30% shortens the
batch by approximately zero. Fresh runs are both cleaner and free. The pilot's value
is as a diagnostic, which is how it is used.

## 15. Go/no-go diagnostic RESULTS — one hypothesis confirmed, one falsified, and a pre-registered decision rule that was wrong

Appended beneath §14, never edited into it. `analysis/pilot_diagnostics.py` is the
re-runnable artefact; it was verified to produce byte-identical output across three
independent runs, it trains nothing, and it writes nothing.

**Headline, stated before the reasoning: H14a is FALSIFIED. The centred embedding
initialisation does not survive training.** It is reported first because it is the
uncomfortable half.

### 15.1 H14a — FALSIFIED (22.3877% against a 15% threshold)

Silent-unit fraction on the committed 6,000-step fixture, using the **trained**
embedding from each `reservoir_seed{s}_clipemb` checkpoint:

| stage | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| init | 1.6602% | 1.8188% | 1.9531% | 1.8107% |
| step 100,096 | 4.7607% | 6.4087% | 4.7852% | 5.3182% |
| step 200,192 | 17.6147% | 20.1538% | 14.0381% | 17.2689% |
| **step 300,032** | **26.9165%** | **19.2749%** | **20.9717%** | **22.3877%** |

Zero saturated units at every stage. For reference at the same step: `emb` (global
clipping + centred) 22.6156%, `clip` (per-group + legacy) 45.8822%.

**The mechanism is measured, not inferred.** The centring identity is
`b = -(W @ mu)`. It decays because **`W` drifts and `b` does not follow it**. On
seed 0: `||W@mu + b||` goes 0.0000 -> 0.1690 -> 0.5350 -> 0.8945 while `||b||`
barely moves (0.9390 -> 0.9243) and `||W@mu||` grows (0.9390 -> 1.2129). The
fraction of the centring that has decayed reaches **0.737 by 30% of a run**. The
induced frozen membrane-offset std goes **0 -> 0.50 -> 1.60 -> 2.68** against a
firing threshold of 1.0. The informative AC drive is essentially flat over the same
window (0.0900 -> 0.0941, +4.6%), so **the drift is almost purely DC
re-accumulation** — the exact defect the centring was built to remove, growing back.

**This answers, negatively, the open question §12 recorded verbatim** ("The bias is
trainable, and whether it adapts was NOT verified"). It does not adapt. It decays.

### 15.2 H14b — CONFIRMED, decisively (3.6085e-04 against a 1e-4 threshold)

Measured from the **exact last real optimizer step** of each run rather than a
proxy: a checkpoint stores `exp_avg`, `exp_avg_sq` and the step count as of
immediately after the last `optimizer.step()`, and Adam's update is a closed form in
those three, so the step actually taken on the actual clipped gradient is
reconstructible with no ROM, no rollout and no synthetic gradient. This is stronger
evidence than the counterfactual §6.2 of `RESULTS.md` used, not weaker.

Readout median `||dp||/||p||`:

| condition | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| **v2 `clipemb` readout** | **2.6308e-04** | **4.6390e-04** | **3.5556e-04** |
| v1 `global` readout (the pathology) | 4.5535e-06 | 2.2807e-06 | 2.0369e-07 |
| baseline GRU, non-embedding (healthy reference) | 5.5672e-04 | 5.9212e-04 | 7.5568e-04 |

**The corrected readout now sits inside the healthy GRU's own band**, having been
one and a half to three orders of magnitude below it under the global rule.

The reason, and it is the crisp one: **Adam tolerates a small steady rescaling and
not a swinging one** (§6.2 of `RESULTS.md`). The clip coefficient the readout
actually receives, over all 2,344 updates:

| rule | median coefficient | max/median |
|---|---|---|
| per-group | 8.40e-02 – 1.03e-01 | **9.7 – 11.9** |
| global (v1) | 1.47e-09 – 6.50e-09 | **7.3e+05 – 1.15e+06** |

Five orders of magnitude of stabilisation on the coefficient's swing.

**The embedding gradient is still exploding, and it no longer matters.** Over the
last 500 updates the embedding group's pre-clip norm has median 7.0e8–1.6e9 and max
up to 1.3e11 — the `74,508,156,928` that triggered this whole diagnostic is roughly
the p95 of its own run's distribution, i.e. typical rather than anomalous. The
readout group's own norm over the same window has median 8–10 and p95 ~20. **Per-group
clipping is doing exactly the job it was added to do.**

**Zero NaN and zero Inf anywhere** — 9 log files across 7 numeric fields plus both
group norms, and 9 checkpoints scanned tensor-by-tensor across model and full Adam
state. **The frozen-reservoir invariant holds bit-for-bit**: each reservoir rebuilt
from its seed and diffed against the stored buffers across 27 checkpoint loads gives
max absolute difference **0.0e+00** in all 27.

### 15.3 H14c — the 2x2 factorial, descriptive only

Mean `mean_extrinsic_reward` over updates 1876–2344, three seeds:

| cell | seed 0 | seed 1 | seed 2 | mean | sd |
|---|---|---|---|---|---|
| global + legacy (v1) | 0.014272 | 0.011474 | 0.017130 | 0.014292 | 0.002828 |
| per-group + legacy (`clip`) | 0.071081 | 0.090592 | 0.067127 | 0.076267 | 0.012563 |
| global + centred (`emb`) | 0.017010 | 0.021098 | 0.022393 | 0.020167 | 0.002810 |
| **per-group + centred (`clipemb`)** | 0.070813 | 0.093462 | 0.072721 | **0.078998** | 0.012562 |
| *baseline GRU (v1), same window* | 0.114712 | 0.114499 | 0.099339 | *0.109517* | 0.008815 |

Clipping alone **+0.061975**; centring alone **+0.005875**; both **+0.064706**
against an additive prediction of +0.067850, i.e. an interaction of **-0.003143**
that is **smaller than the seed standard deviation (0.0126) and therefore not
interpretable**. Per-group clipping carries roughly 96% of the combined effect.

**No p-value is computed and no arm claim is made.** Three seeds at 30% of a run is
below this project's own bar (§2), which is why H14c was registered as descriptive.
Recorded plainly: at this point the corrected reservoir arm is still **~28% below**
the baseline GRU reference, and its last-500-update trend is **flat** (OLS slopes
-1.27e-05, -7.88e-08, +1.45e-05 per update on a level of ~0.079), not rising.

### 15.4 The pre-registered decision rule was WRONG, and this says so before acting on it

§14.4 fixed this rule in advance: *"the full matrix is launched … **only if H14a and
H14b both survive**. If either is falsified, the matrix is not launched on that
configuration and the reason is recorded here before anything else is run."*

**H14a is falsified. Read literally, that rule forbids the launch.** It is being
overridden, and the override is written down here, with its reasoning, **before the
matrix is launched** — not discovered afterwards as a justification.

**Why the rule was badly formulated.** It gated on two hypotheses that answer
different questions and carry different consequences:

- **H14b asks whether the experiment is VALID** — whether the corrected protocol
  actually removes the optimizer/clipping confound that made v1 uninterpretable.
  Had H14b failed, launching would have produced another confounded comparison and
  the rule would have been right to stop it. **H14b survived.**
- **H14a asks whether one of the two treatments is as EFFECTIVE as advertised.**
  Its falsification does not confound anything. The centred init is still applied
  **identically to both arms**, so it remains a valid control; it is simply a
  weaker treatment than §12 implied. A weaker-than-hoped treatment is a *result to
  report*, not a defect that invalidates a comparison.

Treating those two as interchangeable gates was the error. This is the same class of
mistake §12 already recorded once — *"a pre-registration that only produces a verdict
when its manipulation succeeds is not a complete pre-registration"* — and it is
logged here in the same spirit rather than quietly dropped.

**The corrected rule, stated explicitly:** *launch only if the validity hypothesis
(H14b) survives; a falsified efficacy hypothesis (H14a) is reported, not treated as
a launch gate.*

**Why the alternatives are worse, which is the substantive argument.**

- **Launching on `legacy` instead** would commit the headline comparison to a
  configuration with a known, diagnosed, *unfixed* defect — **45.8822% of the
  reservoir silent** against 22.3877% — in order to avoid a fix that turns out to be
  merely weaker than advertised. On every measured outcome centred is at least as
  good as legacy: silent fraction 22.39% vs 45.88%, reward 0.0790 vs 0.0763, zero
  saturated units in both. That trade is strictly negative.
- **Adding a drift-suppression knob now** (bias re-centring, per-group learning
  rates, a tighter embedding clip) would put an **unpiloted** mechanism inside the
  headline run. That is the highest-risk option of the three and it would need its
  own pre-registration and its own pilot.
- **Not launching at all** leaves `DESIGN.md` §5 answered only by a comparison whose
  own write-up says it cannot separate the architectural question from an optimizer
  artefact.

**Binding constraint on the write-up, recorded now so it cannot be softened later:
`RESULTS.md` v2 may NOT claim that the centred initialisation keeps the reservoir
healthy.** The defensible claim, and the only one, is: *centring holds the reservoir
near-fully active for roughly the first 100k steps and roughly halves the silent
fraction at 300k, after which the invariant decays because the trainable weight
drifts and the bias does not follow it.*

### 15.5 The finding neither fix addresses — the embedding drifts without bound

Not pre-registered, found while measuring H14a, and reported at this weight because
it is measured over a **complete** v1 run rather than extrapolated. Across all ten
checkpoints of `reservoir_seed0` (1,000,064 steps, `legacy` + `global`):

| quantity | step 100,096 | step 1,000,064 |
|---|---|---|
| `\|\|W\|\|` | 1.0490 | 1.6709 |
| `\|\|W@mu + b\|\|` (residual DC) | 0.3130 | 1.8594 (6x) |
| membrane-offset std | 0.9436 | **5.6270** |
| mean spike rate | 0.018749 | **0.200348** |

**The spike rate ends at 10x the ~2% band `models/spiking_reservoir.py` documents as
healthy**, and saturated units appear from step 600,576 onward, reaching 1.3428%.
Under `legacy` the silent set stays pinned near 46% (consistent with A4a's nesting
result) and the drift surfaces as runaway *firing* instead of silence.

**Neither shipped fix controls the reservoir's operating point over a full run.**
Per-group clipping fixes the optimizer pathology; centring fixes the *initial*
operating point. Nothing in the current design regulates where the operating point
goes after that. This is the successor question to A4, it is stated here as an open
problem rather than a solved one, and A9 below turns it from extrapolation into
measurement.

### 15.6 Pre-registered: A9 — where does the reservoir's operating point actually end up?

Declared **before the v2 matrix is launched**, so the endpoint is measured rather
than argued about afterwards. Cost is trivial (~1.4 s per checkpoint, ~5 minutes for
the whole matrix), which is why there is no excuse for extrapolating instead.

- **Measurement.** For every checkpoint of every v2 reservoir run (10 seeds x 10
  checkpoints), compute on the committed fixture: silent-unit fraction, saturated
  fraction, mean spike rate, `||W||`, residual DC `||W@mu + b||`, and membrane-offset
  std. Report the full trajectory, per seed, not just the endpoint.
- **H9 (prediction).** The centred init's advantage over legacy **shrinks but does
  not vanish** by step 1,000,064: mean silent-unit fraction over the ten v2 reservoir
  seeds at the final checkpoint is **below 40%**, against legacy's measured ~46%.
- **Falsified if** that mean is **at or above 46%**, i.e. the centred runs converge
  to the legacy operating point or worse — which would mean centring buys a
  healthier first third of a run and nothing at the end of it, and that the honest
  summary of the embedding fix is "transient".
- **Ambiguous band, declared in advance:** a mean between 40% and 46% confirms the
  direction while falsifying the magnitude, and will be reported in those words.
- **Explicitly not claimed:** A9 measures construction health, not task performance.
  Whether a healthier reservoir produces a better agent is what the v2 arm comparison
  measures, and A9 may not be substituted for it. (§12 records what happens when a
  falsification condition is written as a conditional on task performance.)

## 16. A8 RESULTS — the structured-core route WORKS, and the honest caveat is bigger than the result

Appended beneath §14.7's pre-registration, which is unrevised and was committed
(`942a333`) before any of these measurements existed.

**Both hypotheses are CONFIRMED.** §14.6 of this ledger said, in advance, that this
session did **not** expect to solve criticism (c). That expectation was wrong, and
being wrong in this direction is recorded with exactly the same prominence a
disconfirmation would have been given.

### 16.1 H8a — CONFIRMED. Entanglement entropy is tunable across essentially all of [0, 1]

Normalised entanglement entropy S-bar, production geometry (`reservoir_size=8192`,
`tt_rank=8`, `tt_n_cores=4`), three reservoir seeds per lambda:

| lambda | seed 0 | seed 1 | seed 2 | **mean** |
|---|---|---|---|---|
| 1.0 (= the existing construction) | 0.991824 | 0.994166 | 0.991603 | **0.992531** |
| 0.9 | 0.818772 | 0.792239 | 0.813486 | **0.808166** |
| **0.7** | 0.373856 | 0.410766 | 0.360506 | **0.381709** |
| **0.5** | 0.139848 | 0.193598 | 0.115370 | **0.149606** |
| 0.3 | 0.027806 | 0.043949 | 0.020546 | **0.030767** |
| 0.1 | 0.000612 | 0.001025 | 0.000435 | **0.000691** |

Absolute range **0.993731** against a pre-registered threshold of 0.3, and the
productive band **[0.1, 0.5] is entered at lambda = 0.7 and 0.5, on the mean and on
every individual seed**. Neither falsification clause is met.

**Anchor check:** lambda = 1.0 returns S-bar 0.991824, reproducing §11.0's
independently measured 0.9918. The sweep sits on exactly the construction A5/A6
measured, so this is a continuous extension of that work and not a different object.

**The entropy moved because the spectrum decayed**, which is the mechanism A6
identified as the only one available. Middle-bond normalised Schmidt spectrum, seed
0 (uniform reference 0.125):

- **lambda = 1.0:** 1.68814e-01 … 9.40263e-02 — near-flat, reproducing A6's
  0.16881 … 0.09403 exactly. max/min = **1.80**.
- **lambda = 0.7:** 7.30107e-01, 2.03600e-01, 5.17303e-02, 1.19159e-02, … —
  max/min = **2.3e4**.

### 16.2 The operator-norm confound was checked and does NOT explain the result

The obvious objection: the profile multiplies by factors <= 1, so it shrinks the
operator, and a near-trivial reservoir would have a low S-bar for an uninteresting
reason. **The shrinkage is real and it is reported: sigma_max falls 5.2x overall
(2.908942 -> 0.533385).** But it **saturates at lambda ~= 0.7 and is flat to within
4% below that**, while S-bar keeps falling by nearly three orders of magnitude
(0.374 -> 0.0006) over exactly that flat stretch. The two quantities are dissociated
precisely where the hypothesis is decided.

**The decisive control:** rescale the lambda = 0.7 cores by a single global factor —
which is exactly what the pre-existing `tt_core_std` knob does, and to which S-bar is
*provably* invariant (A5) — until `sigma_max` matches lambda = 1.0's to all printed
digits. **S-bar is unchanged bit-for-bit** (0.373856 / 0.410766 / 0.360506, seeds
0/1/2, before and after). A low S-bar at lambda = 0.7 is a genuine structural change,
not a shrunken reservoir.

### 16.3 H8b — CONFIRMED, and the cost is stated rather than buried

Firing health at lambda = 0.7 (the first in-band lambda on the pre-registered grid),
measured with the same `_silent_fraction` machinery on the same committed fixture,
under `centered` + `embed_scale=3.0`:

| lambda | mean silent | mean spike rate | saturated |
|---|---|---|---|
| 1.0 | 1.8107% | 0.018013 | 0% |
| **0.7** | **5.2734%** | **0.012857** | **0%** |
| 0.5 | 5.2612% | 0.012846 | 0% |

Silent 5.27% against a 10% threshold, spike rate 1.29% inside the 1-3% band, zero
saturated units. Neither falsification clause is met.

**Reported without softening: the profile still costs firing health.** Silent
fraction roughly **triples** (1.81% -> 5.27%) and the spike rate drops **29%**. Both
stay inside the pre-registered thresholds, so H8b holds *as written*, but this is a
real degradation and not a free lunch. The norm-matched control recovers about a
third of it (silent 4.15%), so roughly one third of the cost is operator shrinkage
and the remainder is the structural change itself.

### 16.4 Implementation, and the bit-identity guarantee that makes the sweep a controlled comparison

Shipped as `SpikingReservoir(tt_bond_decay=1.0)` — **default is the no-op**, the same
discipline `--grad-clip-mode` and `--embed-init-mode` were shipped under, because
published results depend on the existing construction being reproducible bit-for-bit.

- **Bond-index convention derived from the code, not assumed.** A core is
  `(r_{k-1}, m_k, n_k, r_k)`; axis 0 is the left bond, axes 1/2 the physical modes,
  axis 3 the right bond — read off `_tt_matvec`'s contraction `'bpans,amnc->bpmcs'`
  and `entanglement_entropy`'s `(rp, m*n, rk)` flattening. A test asserts the ratio
  `core_lambda / core_1` is constant over the physical axes and exactly
  `lambda^(a_left + a_right)` over the bond axes. **Putting the profile on a mode
  index would also have moved the entropy**, which is why it is checked against the
  axis structure rather than inferred from a favourable result.
- **Shared bonds are handled symmetrically** — every bond axis of every core gets the
  identical weight vector, so no core "owns" a bond. Consequence stated rather than
  buried: each internal bond index is suppressed twice, so effective suppression is
  `lambda^(2r)`. A one-sided convention is the same family reparametrised
  `lambda -> sqrt(lambda)`, not a different construction. Boundary bonds have
  `r_0 = r_d = 1`, so `lambda^0 = 1` — a no-op by construction, never a hidden edge
  case.
- **lambda = 1.0 is bit-identical, proven not asserted.** The multiplication is
  applied unconditionally (no short-circuit that would make the test vacuous), and
  full state dicts compare equal under `torch.equal` on every tensor. At *every*
  lambda the profile is applied after the `randn` draw and never consumes the
  generator, so `W_in` is bit-identical across lambdas and each core satisfies
  `torch.equal(C_lambda, C_1 * left * right)` **exactly**. The sweep is therefore one
  reservoir family under a deterministic per-entry factor, not six unrelated
  reservoirs.
- **The entropy estimator is the shipped A5/A6 one**, reused rather than
  reimplemented, and validated on cases where the answer is known independently:
  product state -> **0.0 exact**; maximally entangled dim-4 bond -> **1.0 to 1e-6**;
  a designed spectrum sigma = (1, 0.5, 0.25, 0.125) -> **0.522449 to 1e-6 against the
  analytic value**. The third case exists because the first two never exercise the
  `-sum p log p` sum on a non-degenerate spectrum.

### 16.5 What A8 does NOT show, stated at least as loudly as what it does

**A8 includes no training. It cannot say whether a lower S-bar produces a better
agent, and nothing in this section may be read as saying so.** This is the exact
mistake §12 already recorded once, when A4-A6's falsification condition was written
as a conditional on task performance and became untestable; §14.7 was deliberately
written to state its conditions purely over construction quantities, and that
scoping is what makes these verdicts clean.

Three further limits:

- **`entanglement_entropy`'s productive band [0.1, 0.5] comes from Sato et al., who
  validated it on rate-based regression, not on a spiking reservoir.** The band is
  imported, not established here.
- **The knob is not wired into training.** `tt_bond_decay` exists only on
  `SpikingReservoir`; `PolicyValueReservoir`, `build_model`, the CLI and the
  checkpoint dict do not carry it. Any training ablation needs that plumbing first,
  including recording it in checkpoints and log lines the way `embed_init_mode` and
  `embed_scale` are.
- **lambda pins S-bar to a band, not to a value.** At lambda = 0.5 the three seeds
  span 0.115-0.194, roughly +/-30% around the mean, and the relative spread grows as
  lambda falls. Any future design targeting a specific S-bar must measure it per seed
  rather than trust the lambda -> S-bar map.

### 16.6 Status of the three architectural criticisms, restated honestly

| criticism | status |
|---|---|
| (a) structurally silent units | **Root cause found and fixed at initialisation** (§12), but the fix **decays during training** (§15.1) — silent fraction 1.8% at init, 22.4% by 30% of a run. **Partially solved.** |
| (b) trainable budget receiving no gradient | **Mechanism confirmed, magnitude wrong** (A4a). A7 (§14.5) is pre-registered and its answer arrives with the v2 matrix. **Open, under test.** |
| (c) entanglement entropy / deep chaos | **Solved at construction** (A8): tunable across [0,1], productive band reachable at lambda 0.5-0.7, firing health preserved, confound controlled. **Whether it helps the agent is untested and remains OPEN.** |

§12's stated conclusion — *"no construction that keeps i.i.d. Gaussian cores can
reach the productive band"* — **needs no retraction**. It was scoped to i.i.d.
Gaussian cores, and that scope is precisely what A8 broke. The sibling project's open
question is now **testable for the first time**; it is still **open**.

### 16.7 Under-specifications in §14.7, reported rather than reinterpreted

Recorded because a pre-registration's weaknesses should be logged by the people who
hit them, not discovered by a reader:

1. *"spans an absolute range of at least 0.3"* does not say whether over individual
   seeds or per-lambda means. Both satisfy it here (0.9937 / 0.9918), so it is moot —
   but it should be tightened next time.
2. *"whichever lambda first lands S-bar inside [0.1, 0.5]"* does not say whether
   "first" means grid order or largest lambda. Both give 0.7 here.
3. The pre-registration does not say whether the profile should be normalised to
   preserve operator scale. It was implemented literally (unnormalised), and the
   scale shown to be separately recoverable through `tt_core_std`.

**None of these was used to bend a verdict**, and both verdicts hold under either
reading of every ambiguity.

## 17. v2 matrix run log

Continues §14.12's live log past the point where the matrix actually launched.
Appended as the run proceeds.

### 17.1 The training runs execute from a PINNED WORKTREE, not the working tree

The go/no-go diagnostic caught the working tree, at 20:21:10, in a state where the
reservoir arm **could not be constructed at all**: a concurrent edit had added
`tt_bond_decay` to `SpikingReservoir.__init__` and passed it to `_build_tt_cores`
before that function's signature was updated, so every `use_tensor_train=True`
construction raised `TypeError`. It self-healed seven seconds later. **Launching ten
2.5-hour runs inside such a window would have killed all ten at model
construction**, and nothing would have surfaced until the launcher reported ten
failures.

§9 of this ledger already said "do not `git checkout` while training is live", and
`RESULTS.md` §2.7 records that v1's *evaluation* was run from a worktree pinned to
`64839a9` for the same reason. **The v2 training matrix extends that discipline to
training**, which is the longer and more expensive half:

```
git worktree add .claude/worktrees/v2train dc966a3
```

- The runs execute with `cwd` inside that worktree, so `python -m training.train`
  resolves every import from the **pinned commit** — verified directly, not assumed
  (`models.spiking_reservoir.__file__` reported from inside the worktree).
- `--checkpoint-dir` is an **absolute path into the main repository**, so the data
  lands in `checkpoints_v2/` while the code stays frozen. The worktree is disposable;
  the data is not.
- The pinned commit `dc966a3` deliberately **excludes** the then-uncommitted A8
  changes to `models/spiking_reservoir.py`. The v2 runs therefore execute exactly the
  reservoir construction v1 used. A8's `tt_bond_decay=1.0` is a verified bit-identical
  no-op, so this changes nothing scientifically — it removes a class of accident, not
  a confound.
- **Verified before launching, not after:** a 1,280-step run from inside the worktree
  reproduced `reservoir_seed0_clipemb`'s training log with **0 mismatches** across
  10 updates x 7 float fields.

### 17.2 Launch, and the memory question finally answered with a number

**Reservoir arm launched 20:29 CEST**, 10 seeds, `--jobs 10`, under
`--grad-clip-mode per-group --embed-init-mode centered --embed-scale 3.0`.

Measured at 20:31, over a 120-second window across all ten live runs:

| quantity | measured |
|---|---|
| aggregate throughput | **1,225.6 env-steps/s** |
| per-run throughput | **122.6 env-steps/s** |
| projected wall clock | **~2.2 hours** |

Per-run throughput lands inside the **80-130 env-steps/s** band §3 predicted for the
10-way condition, which is the first time that figure has been checked against a real
ten-way matrix rather than extrapolated from a 4-way measurement.

**§14.10 flagged memory as the one unverified risk. It is now measured and it is not
a risk:**

| quantity | measured |
|---|---|
| RSS per training process | **~570 MB** |
| total across 10 processes | **5.70 GB** |
| system memory free | **76%** |

No document in this repository previously recorded a per-process memory figure. It is
recorded here so the next person sizing a run does not have to rediscover it.

### 17.3 Single-run throughput under the v2 configuration

Measured with `/usr/bin/time`, one run at a time on an otherwise-quiet machine,
25,600 steps, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, full v2 flag set:

| arm | wall clock | env-steps/s |
|---|---|---|
| baseline | 20.71 s | **1,236** |
| reservoir | 60.64 s | **422** |

**Reservoir/baseline ratio: 2.93x slower.** `RESULTS.md` §8 reports 2.5x under the v1
configuration (918 vs 371 steps/s). Two caveats before anyone quotes the difference
as an effect of the corrected configuration: these runs include process startup, ROM
boot and model construction inside a much shorter measurement window than §8's, and
the machine state differs. **The comparable claim is the ratio, not the absolute
numbers, and even the ratio should be re-measured on a quiet machine over a longer
window before it goes into a results table.** It is recorded here as a run-log
observation, not as a §8 replacement.

### 17.4 Operational hazard: a `pgrep -f` watcher matches its own command line

Added to §9's list because it cost nothing this time only because it was caught.

A completion watcher for the training matrix was written as:

```
until ! pgrep -f "run_training_matrix.py --arms reservoir"; do sleep 60; done
```

**This never fires.** `pgrep -f` matches against full command lines, and the watcher's
own shell command line *contains the pattern string*, so the watcher always finds at
least one match — itself — and loops forever. The failure is silent and looks
identical to "the job is still running", which is the worst possible failure mode for
an unattended run: the matrix would have finished and nothing would have noticed.

**Fix: watch the process ID, not a command-line pattern.**

```
while kill -0 "$LAUNCHER_PID" 2>/dev/null; do sleep 60; done
```

`kill -0` tests for existence without signalling, and a PID cannot match itself by
accident. The same trap applies to any `pgrep -f`/`grep` watcher written against a
string that appears in its own invocation — the `grep -v grep` idiom exists for
exactly this reason and does not help here, because the offending process is the
watcher rather than the grep.

### 17.5 Verified: all ten reservoir runs are distinct and the matrix is not double-running

Checked directly rather than inferred from a process count, because a raw
`ps | grep -c` had briefly reported 11 (it was matching the watcher shell, not an
eleventh trainer). Extracting `--seed N` from every live command line gives exactly
one process per seed 0-9, and exactly one launcher. No duplicate seed, no second
launcher, no orphan from a previous session.

### 17.6 Scope decision: v2 does NOT decompose the two fixes at full scale, and why

Recorded **before any v2 result exists**, so it is a scope decision rather than a
reaction to a number.

v1's reservoir arm is `global` + `legacy`. v2's is `per-group` + `centered@3.0`.
**The difference between them therefore confounds the two fixes**: any change in the
verdict cannot be attributed to one or the other from those two conditions alone.
Decomposing it at full scale would need a third full-length 10-seed condition
(`per-group` + `legacy`), i.e. another ~2.2 hours of this machine.

**That third arm is deliberately not being run**, for three reasons:

1. **The decomposition already has an answer at 30% of a run**, from §15.3's 2x2
   factorial: per-group clipping carries ~96% of the combined effect (+0.061975 of
   +0.064706), the centred init contributes ~9% alone and ~4% on top of clipping, and
   the interaction (-0.003143) is smaller than the seed standard deviation (0.0126)
   and therefore not interpretable. That is a real answer with stated limits, at
   three seeds and 30% length.
2. **The mandate is the corrected two-arm comparison.** Adding a third arm puts the
   primary deliverable at risk of not finishing for a secondary question.
3. **It would be reported as a decomposition it cannot support anyway.** Three seeds
   is below this project's own bar (§2); ten seeds on a third arm would fix that, but
   then honesty requires re-running the full evaluation matrix on it too.

**Stated as a limitation, not hidden:** `RESULTS.md` v2 must say that the v1 -> v2
comparison changes **two** things at once, that §15.3 attributes almost all of the
short-horizon effect to clipping, and that a full-scale decomposition was **not run**.
It is registered here as the obvious next ablation.

### 17.7 The baseline arm is chained to launch automatically, with a guard

The session is unattended, so the baseline arm is chained to start when the reservoir
launcher's PID exits rather than waiting for a human or an agent to notice.

**The chain is guarded**: it counts `checkpoints_v2/reservoir_seed*/step_1000064.pt`
and **refuses to launch the baseline arm unless all ten are present**, exiting with a
message instead. Chaining a second one-hour batch onto a failed first batch only
doubles the wreckage and destroys the evidence about why the first one failed. Both
arms run under the identical flag set, from the identical pinned worktree.

### 17.8 Live verification at 4% of the matrix — determinism, learning, and the whole downstream pipeline

Checked at 20:37 CEST against data being written by the live runs, rather than
waiting until 23:00 to discover a wiring fault.

**1. The v2 runs reproduce the pilot bit-for-bit, on live data.** Seeds 0-2 of
`checkpoints_v2/reservoir_seed{s}` compared against
`checkpoints/reservoir_seed{s}_clipemb` over the ~397 updates both had reached, on
seven float fields:

| seed | updates compared | mismatches |
|---|---|---|
| 0 | 397 | **0** |
| 1 | 395 | **0** |
| 2 | 397 | **0** |

This is the §14.14 result confirmed a third time, now on the actual matrix rather
than on a smoke run: the pilot is the prefix of these runs.

**2. All ten seeds are learning.** Mean extrinsic training reward, first 20 updates
vs last 20, at ~400 updates: every seed rises, from a 0.005 band to a 0.021-0.061
band. Seed 2 is the slowest (0.0066 -> 0.0209) and seed 9 the fastest
(-0.0019 -> 0.0610). No seed is flat, diverged, or degenerate.

**3. The entire downstream pipeline was dry-run against the v2 layout before the data
needed it**, which is the part that would otherwise be discovered at 01:00:

- `scripts/run_eval_matrix.py` resolves all **40/40** `init` jobs against
  `--checkpoint-dir checkpoints_v2 --init-checkpoint-dir checkpoints_v2_init
  --results-dir results_v2`, writing nothing.
- `analysis.aggregate_results.summarise_training_logs('checkpoints_v2')` discovers
  the reservoir runs and parses every line (`n_skipped_lines: 0`).
- `build_eval_manifest('checkpoints_v2', selection='final')` correctly reports the
  not-yet-existing baseline runs as `missing` with a reason string, rather than
  raising.

§14.9's claim that the parent-directory layout needs no code change is therefore
**verified end to end**, not merely argued from reading regexes.

### 17.9 Specification for `RESULTS.md` v2 — written before the numbers, so the shape cannot be chosen to flatter them

If this session stops before the write-up exists, **this subsection is the write-up's
specification** and a future session should execute it rather than reinvent it. It is
written now, deliberately, while the v2 numbers do not yet exist: a document whose
structure is fixed before its results cannot have had its structure chosen to suit
them.

**Placement.** Appended BENEATH `RESULTS.md` v1, never edited into it. v1's numbers
stay exactly as they are, including anything v2 contradicts. v1 §12 is the section
promising v2; it is superseded by, not replaced with, what follows.

**Mandatory sections, mirroring v1's shapes so the two read side by side:**

1. **Headline.** The v2 verdict on `DESIGN.md` §5, stated in one sentence, in the
   same form as v1 §1 — direction, effect size, exact permutation p, Cohen's d.
   **Whichever way it goes.** If the reservoir still loses, that is the headline. If
   it now wins, the headline says so *and* immediately states that two treatments
   changed at once (§17.6) so the win is not attributable to either alone.
2. **What changed from v1, exactly.** `--grad-clip-mode per-group`,
   `--embed-init-mode centered`, `--embed-scale 3.0`, applied identically to both
   arms; fresh untrained controls under the same init (§14.9); everything else held.
   **`--embed-scale 3.0` must be stated explicitly** — v1 §12 omitted it and that
   omission would have reproduced a disconfirmed configuration (§14.8).
3. **Primary result**, the same six-row table as v1 §3 (final/best x
   continuous/reset128, plus the two untrained rows), plus bootstrap CIs.
4. **Controls**, mirroring v1 §4.1 (untrained arms indistinguishable?) and §4.2 (does
   each arm beat its own initialisation?).
5. **Per-step decomposition**, mirroring v1 §5. v1 found the episode-return
   scoreboard *flatters* the reservoir by ~6x per step. **Recompute it; do not assume
   it carries over.**
6. **v1 vs v2, side by side** — did correcting the confound change the verdict, in
   which direction, and by how much.
7. **Reservoir health over a full run (A9)** and **the dead-gradient budget (A7)**,
   with the verdicts computed by `analysis/reservoir_health.py` against their
   pre-registered bands (§14.5, §15.6).
8. **Efficiency**, mirroring v1 §8, measured on a quiet machine rather than reusing
   §17.3's contended numbers.
9. **Limitations.** Must carry forward every still-applicable item from v1 §9 and add
   at minimum: the two-treatments-at-once confound (§17.6), H14a's falsification and
   what it means for the embedding fix (§15.1), the group-count asymmetry v1 §9
   already discloses (2 groups reservoir vs 4 baseline under per-group clipping), and
   the fixture caveat (§14.13).
10. **What v2 does and does not tell you**, mirroring v1 §10.

**Binding constraints on the prose, all of them already earned:**

- **May NOT claim the centred init keeps the reservoir healthy.** H14a is falsified
  (§15.4). The only defensible claim is the one §15.4 states verbatim.
- **May NOT present A8 as a Phase 1 result.** A8 is a construction diagnostic with no
  training in it (§16.5). It belongs in its own clearly-scoped section, and the honest
  Phase 1 headline comes first and is never buried under it.
- **May NOT quote §17.3's throughput numbers as a §8 replacement** — they were taken
  on a machine that was not quiet.
- **Reports the negative honestly if it is negative.** The standing instruction on
  this project is that the mandatory-control comparison's value rests entirely on it
  being genuinely honest; a result improved by selection is worth less than nothing.

**Delivery.** Feature branch + PR against `main`, **left OPEN, not self-merged** — it
revises a published conclusion, which §13 reserves for the repository owner. No AI
co-author trailer on any commit.

### 17.10 The pipeline is chained end-to-end and guarded at every step

Written because the previous session's failure mode was *stopping with work half-done
and nobody watching*. The v2 pipeline is therefore built so that the data and the
statistics complete **without requiring this session to still be alive**:

```
reservoir arm (10 runs)
    -> [guard: 10 final checkpoints?] -> baseline arm (10 runs)
        -> [guard: 10 + 10 final checkpoints?] -> evaluation matrix (120 evals)
            -> aggregation -> results_v2_report.txt / results_v2_report.json
```

**Every arrow is guarded, and every guard fails loudly rather than continuing.** The
evaluation guard matters most: evaluating a partial matrix would silently produce a
comparison with fewer seeds than it claims, which is a **wrong result rather than a
failed run** — exactly the failure mode `training/evaluate.py`'s docstring warns
about and the one this project's whole unit-of-analysis discipline (§2) exists to
prevent. The guards refuse and print, they never degrade gracefully.

**What is NOT automated, deliberately:** the `RESULTS.md` v2 write-up and the pull
request. Those require judgement about what the numbers mean, and §17.9 specifies
them in enough detail for a session that was not here to execute them. The A7/A9
measurements (`analysis/reservoir_health.py`) are likewise a manual step, because
their verdicts feed the write-up.

**Timeline projected from the measured rate at 20:39** (447 updates/min aggregate):

| stage | projected finish |
|---|---|
| reservoir arm | ~23:23 |
| baseline arm | ~00:23 |
| evaluation matrix | ~00:58 |

**If you are a later session reading this and those artefacts are missing:** check
`ps aux` first, then `checkpoints_v2_reservoir_launch.log`,
`checkpoints_v2_baseline_launch.log` and `results_v2_eval.log`, then each run's own
`checkpoints_v2/{arm}_seed{n}/launcher.log`. §14.11's checklist re-runs any stage by
hand; every stage is resumable and idempotent.

### 17.11 A7/A9 measurement tooling, and a 0.81-point inconsistency inside this ledger

`analysis/reservoir_health.py` (+ `tests/test_reservoir_health.py`, 38 tests; suite
now **321 passed, 0 failed**) implements both pre-registered measurements: A7's
dead-gradient `in_proj` column count (§14.5) and A9's operating-point trajectory
(§15.6). It is read-only, writes nothing, skips gracefully over runs that do not
exist yet, and computes each verdict against its pre-registered band **in code**, so
the verdict is not a judgement call made while looking at the number.

**It reproduces A4a exactly.** Run against the completed v1 `reservoir_seed0`:

```
865 dead cols, 10.5591%, 13840 params, 9.9440% of budget
newly_dead per transition: [0]   nesting_holds=True
```

**865 / 10.5591% / 13,840 / 9.9440% is a digit-for-digit match to §12/A4a's published
figures.** The frozen-reservoir invariant also holds at 0.0e+00 max abs diff. That is
the correctness check that matters: new tooling that reproduces an already-published
measurement is trustworthy in a way that new tooling producing new numbers is not.

**The inconsistency, recorded rather than smoothed over.** The same run's silent
fraction measures **46.2280%**, against the **45.42%** quoted in §12/A4a's prose
table — a gap of **0.81 percentage points**, i.e. roughly 66 of 8,192 units.

- The 46.2280% figure is a **bit-for-bit match to `analysis/pilot_diagnostics.py`'s
  own independent implementation** (every column agrees: spike rate 0.200348,
  saturated 1.3428%, `||W||` 1.6709, DC 1.8594, offset std 5.6270).
- So **two independently written measurement scripts agree with each other and both
  disagree with §12's prose.**
- **Nothing was adjusted to close the gap.** It is a pre-existing discrepancy between
  §12's prose and the tooling this project has since committed, not something the new
  script introduced. The likely cause is a difference in the probe definition behind
  A4a's 3,721-unit count, but that has **not been run down**, and saying "likely" is
  as far as the evidence goes.
- **Consequence for the write-up:** where a silent fraction for v1 is quoted next to
  a v2 number, it must come from `reservoir_health.py` so both sides of the
  comparison are measured by the same instrument. §12's 45.42% stays on the page per
  the append-only rule, with this paragraph as the correction of record.

**On the v1 smoke's "FALSIFIED" verdicts — they are not results.** Running the script
against v1 prints `A7 verdict: FALSIFIED` (10.5591% against a <2% prediction) and
`A9 verdict: FALSIFIED` (46.2280% against a <40% prediction). **v1 is the `legacy` +
`global` configuration; the predictions were registered about the v2 runs.** The
smoke output is the script exercising its verdict logic against the reference
condition, and what it actually shows is that **v1 sits far outside both predicted
bands** — which is precisely what makes the v2 predictions non-trivial. The real A7
and A9 verdicts are computed against `checkpoints_v2` when the matrix completes.

### 17.12 Incident at 21:37 — the chained automation was reaped, the guard caught it, nothing was corrupted

Recorded in full because the guard earning its keep is the whole argument for building
guards, and because two of the three causes are reusable traps.

**What happened.** The baseline and evaluation chains had been started as managed
background tasks. At ~21:37 the agent harness reaped those tasks, killing both chain
processes. The evaluation chain, which was waiting on the baseline chain's PID, saw
that PID vanish, concluded the baseline arm had "finished", and ran its guard:

```
both arms finished at 21:37: reservoir 0/10, baseline 0/10
ABORT: matrix incomplete (reservoir 0/10, baseline 0/10). Evaluation NOT run.
```

**The guard did exactly its job.** Without it, the pipeline would have run a 120-job
evaluation matrix against **zero** finished runs and then handed the aggregation an
empty result set — producing either a crash at 01:00 or, far worse, a "comparison"
computed over whatever partial data happened to exist. §17.10 argued that evaluating a
partial matrix is *a wrong result rather than a failed run*; this is that argument
being cashed in four hours later.

**Nothing was damaged.** The reservoir launcher (PID 30003) was started with `nohup`
from a wrapper that exited immediately, so it had already been orphaned and adopted by
`init` — it was never in the reaped task's process group. All ten training processes
ran through the incident untouched, and the matrix was at 42% when it happened.

**Three lessons, all reusable:**

1. **A process started as an agent-harness background *task* dies when that task is
   reaped; a process `nohup`-ed from a short-lived wrapper does not.** The difference
   is whether the long-running process is the task's own main process or an orphan it
   left behind. Long chains must be orphans.
2. **`setsid` does not exist on macOS.** `nohup cmd & disown` from a foreground call
   is the portable equivalent here.
3. **zsh aborts a command on an unmatched glob** rather than passing the pattern
   through. `ls checkpoints_v2/baseline_seed*/step_1000064.pt 2>/dev/null | wc -l`
   printed `no matches found` and yielded 0 **before `ls` ever ran**, so the `2>/dev/null`
   was useless. Guard counts must use `find ... | wc -l`, or `setopt NULL_GLOB`. This
   one is nastier than it looks: it fails *toward* zero, which for a completeness guard
   means it fails safe — but for any check written the other way round it would fail
   silently open.

**Remediation.** The three separate chain scripts were replaced by one
`/tmp/gs_pipeline.sh`, launched `nohup`-ed and disowned (PID 34500), which waits on the
reservoir launcher and then runs baseline -> evaluation -> aggregation -> A7/A9, with a
`find`-based guard on every arrow and a timestamped log at `pipeline_v2.log`.

## 18. Incident at 23:38:53 — the machine lost power, and the reservoir arm is being RESTARTED rather than resumed

A different and larger incident than §17.12's, four hours later and with a different
cause. Recorded in the same detail, and — because the recovery involves discarding
90% of a completed compute batch — **the decision is written here before the restart
is launched**, not afterwards as a justification.

### 18.1 What happened, with the evidence

**The whole machine rebooted at 23:38:53 CEST.** This is not inference:

```
sysctl kern.boottime  ->  { sec = 1787261933 } Thu Aug 20 23:38:53 2026
```

Verified at 23:46 with the system 8 minutes old, zero training processes alive, and
no `python`/`train.py`/`pyboy` in `ps aux`. Distinguishing features from §17.12's
harness reap, all checked rather than assumed:

- **Every process on the machine died, not just the chained automation.** The
  reservoir launcher (PID 30003), which §17.12 records as having survived the 21:37
  reap precisely because it was `nohup`-ed into an orphan, is gone. Orphaning does not
  survive a reboot and was never expected to; **checkpointing is what covers this
  failure mode, and it did.**
- **It was not a kernel panic.** `/Library/Logs/DiagnosticReports/` contains no
  `.panic` file, and the user DiagnosticReports directory's newest entry is from
  01:47, twenty-two hours earlier. A panic under the 10-way training load would have
  been a reason to reduce `--jobs` on the restart; the absence of one is why the
  restart runs at the same `--jobs 10`.
- **The agent session driving the run died with it.** No process, no resumable state,
  and `/tmp` was cleared — taking `/tmp/gs_pipeline.sh` (§17.12's remediation) with
  it. **That is the reusable lesson: a recovery script that lives in `/tmp` does not
  survive the incident class it exists to recover from.** It is replaced by a
  committed one (§18.4).
- No `crontab` and no relevant LaunchAgent existed, so nothing auto-resumed anything.

### 18.2 What was lost, precisely — and what was not

**Nothing was corrupted.** All ten `checkpoints_v2/reservoir_seed{0-9}/step_900864.pt`
were loaded with `torch.load(..., weights_only=True)`, matching `load_checkpoint`'s own
call. **All ten load cleanly**, and every one records the intended v2 configuration
(`arm=reservoir`, the right seed, `grad_clip_mode='per-group'`,
`embed_init_mode='centered'`, `embed_scale=3.0`, `step=900864`, 42 model tensors, 31
optimizer state entries). `save_checkpoint` is a bare `torch.save` with no
temp-file-then-`os.replace`, so a torn write was physically possible; it did not
happen. The last checkpoint mtimes cluster at 23:27:27–23:28:53, i.e. ten minutes
before the power cut, which is why.

**The runs were at ~95.7%, not 90%.** The checkpoints stop at step 900,864 but the
`train_log.jsonl` files run to **step 952,832–961,280 (updates 7,444–7,510 of 7,813)**.
`--checkpoint-every 100000` means the next checkpoint would have been the final one, so
**between 51,968 and 60,416 steps per seed of real, logged training exist as a learning
curve whose model state does not exist on disk.** That gap is what makes a resume
messier than it first looks (§18.3, point 2).

Untouched and verified: v1's 20 runs (all at `step_1000064.pt`), the 20 v2 untrained
controls at `checkpoints_v2_init/` (§14.12), and every committed artefact.

### 18.3 Decision: the reservoir arm is RESTARTED from step 0, all ten seeds

The cheap option was `--resume-from checkpoints_v2/reservoir_seed{n}/step_900864.pt`
for each seed — roughly 13 minutes of compute against ~2.2 hours for a restart. **It is
being rejected**, and the reasoning is recorded because rejecting a 10x-cheaper option
needs one.

1. **The published recipe must reproduce the published numbers.** §14.11's Step 2 is
   the documented command for the v2 reservoir arm. A resumed arm is not what that
   command produces, so the v2 numbers would be reproducible only by an undocumented
   bespoke recovery procedure. That is a reproducibility defect in the primary
   deliverable, traded for two hours of a machine that is otherwise idle.
2. **A resume would corrupt the arm's own learning curve.** `_append_log` opens
   `train_log.jsonl` in append mode, and a resumed run restarts its update counter at
   1. Resuming from step 900,864 would therefore append a second set of records for
   steps 900,992–~956,800 — **the same steps appearing twice with different values**,
   plus 775 lines whose `update` index collides with the run's own first 775. Repairing
   that means editing raw experimental data by hand, on the artefact the write-up's
   learning curves come from. Not worth doing to save two hours.
3. **§2's determinism property would break for this arm.** §2 states that a run's state
   at any intermediate checkpoint is bit-identical to what a shorter run would have
   produced. §14.14 and §17.8 both cash that property in (the pilot is a bit-identical
   *prefix* of the v2 reservoir runs, verified twice at 0 mismatches). `--resume-from`
   restores model and optimizer state but **not** the emulator state, the RNG stream
   position, or the `NoveltyGate` buffer — §14.14 already recorded exactly this, when it
   refused to resume the three pilot seeds into the v2 matrix for the same reason.
4. **It would put a protocol difference between the two arms of the mandatory control.**
   The baseline arm has not started and will run uninterrupted. A resumed reservoir arm
   differs from it by an `env.reset()` at 90%, a restarted RNG stream, and ~512 steps of
   elevated intrinsic reward while the novelty buffer refills. Each is individually
   negligible; the objection is that **the control exists to make the two arms differ in
   exactly one thing**, and §14.14 already rejected resume-induced heterogeneity when
   accepting it would have been free.
5. **Nothing unique is lost, and the restart is itself a measurement.** Determinism
   means the discarded prefix regenerates bit-identically. The crashed runs are
   therefore **preserved, not deleted** (moved to `checkpoints_v2_crashed/`, 292 MB,
   gitignored) so the fresh logs can be diffed against them over the ~7,038 overlapping
   updates. Zero mismatches would be a **third independent confirmation** of the
   determinism property, on data produced across a reboot and a different process
   generation — stronger evidence than either previous check.

**Cost if wrong.** ~2.2 hours, and greater exposure to a second interruption inside
that window. Both are mitigated rather than accepted: the crashed checkpoints are
preserved, so `--resume-from` remains available as a fallback if the schedule tightens,
and §18.4's measures reduce the exposure.

**What is NOT being changed:** the flag set, `--jobs 10`, the pinned worktree
(§17.1, commit `dc966a3`), and the untrained controls. The restart re-runs the same
experiment, not a revised one.

### 18.4 Resilience measures added, and the ones deliberately not added

Second real interruption in one night, so the question of a permanent fix is live. The
measures taken are the cheap ones that address the observed failures directly:

- **The pipeline script is committed to the repository** (`scripts/run_v2_pipeline.sh`)
  rather than written into `/tmp`. §17.12's remediation was lost to the very reboot it
  would have been useful for. A committed script survives, is legible in git, and is
  reviewable — the same argument §14.10 made for committing the training launcher.
- **The chain runs under `caffeinate -is`**, so an idle-sleep cannot suspend a
  multi-hour unattended batch.
- **The chain is `nohup`-ed and disowned from a short-lived wrapper**, per §17.12's
  lesson 1, so a harness reap cannot kill it. `setsid` still does not exist on macOS
  (lesson 2), and every completeness guard is `find`-based (lesson 3).

**Deliberately NOT added: a launchd watchdog that resumes training after a reboot.**
It would have saved nothing here — §18.3 explains why this interruption warranted a
restart rather than a resume, so an automatic resumer would have produced exactly the
artefact this entry rejects. It is also unattended automation that writes into the
experiment's output directories with no one watching, which is the hazard class §9 and
§17.12 are both about. The priority is a genuine v2 result, not infrastructure for its
own sake.

### 18.5 Verification that the restart lost nothing — measured, not argued

§18.3's fifth point claimed the discarded prefix would regenerate bit-identically. That
claim is now checked rather than trusted. The restarted runs' `train_log.jsonl` were
compared against the preserved `checkpoints_v2_crashed/` logs, update-for-update, on the
same seven float fields §14.14 and §17.8 used (`mean_reward`, `mean_extrinsic_reward`,
`policy_loss`, `value_loss`, `entropy`, `total_loss`, `grad_norm`), with the `step`
field asserted equal on every pair:

**3,535 values compared across all ten seeds. 0 mismatches. Worst absolute difference
0.0.**

This is the **third** independent confirmation of §2's determinism property, and the
strongest of the three: §14.14 compared a smoke run against a pilot within one machine
generation, §17.8 compared the live matrix against the same pilot, and this one compares
runs separated by **a power loss, a reboot, and a completely different process
generation**. Determinism holds across all of it, which is what makes discarding a
90%-complete batch a free operation rather than a costly one.

### 18.6 Pre-flight for the baseline arm, which had never been run under the v2 flags

The reservoir arm had a 30%-length pilot behind it (§14.14). **The baseline arm had
never once been executed under `--grad-clip-mode per-group --embed-init-mode centered
--embed-scale 3.0`** — it was gated behind the reservoir arm both times the matrix ran,
so a flag-handling fault on that arm would have surfaced at the moment stage 2 launched,
with nobody awake, after the reservoir arm's two hours were already spent.

A 1,280-step run into a throwaway directory was therefore executed first. It writes a
loadable checkpoint recording `grad_clip_mode='per-group'`, `embed_init_mode='centered'`,
`embed_scale=3.0`; every log line carries the same; and the per-group norms show the
**four** groups the baseline arm has (`embedding`, `gru`, `actor_head`, `critic_head`)
against the reservoir arm's two — the group-count asymmetry `RESULTS.md` v1 §9 already
discloses, here observed directly rather than inferred. Gradient norms are healthy
(global 0.254, no group above 0.20). Throwaway deleted.

## 19. Corrections of record found while the restarted matrix ran

Appended beneath §18 rather than edited into the sections they correct, per the
append-only rule. Both were found by executing the documented commands rather than
reading them.

### 19.1 §14.11's Step 5 aggregation command SILENTLY PRODUCES NO COMPARISON

§14.11 Step 5 gives the statistics command as:

```
.venv/bin/python -m analysis.aggregate_results \
  --results-dir results_v2 --checkpoint-dir checkpoints_v2
```

**Run exactly as written, this compares nothing.** `analysis/aggregate_results.py` reads
evaluation JSONs from the directory it is handed, but `scripts/run_eval_matrix.py` writes
them one level down, into `{results-dir}/{final,best,init}/` — which §7 of this ledger
describes correctly and §14.11 then forgets. Pointed at the parent directory the
aggregator finds zero eval files and prints:

```
--- regime=continuous: skipped (compare_arms: no eval results for arm(s)
    ['reservoir', 'baseline'] ... Arms found: []) ---
```

for both regimes, and then prints a complete, healthy-looking training-log summary
underneath. **That is the dangerous part: the output does not look like a failure.** It
looks like a successful aggregation with two skipped lines, and the section that would
have carried the headline is simply absent.

**Verified against v1, both directions.** `--results-dir results` compares nothing;
`--results-dir results/final` reproduces `RESULTS.md` v1 §3 digit-for-digit
(reservoir 28.4169 ± 7.0593, baseline 36.1335 ± 3.4082, diff −7.7167, exact permutation
p = 0.000996, Cohen's d = −1.3922, bootstrap CI [−12.7665, −3.5839]). `results/best` and
`results/init` likewise reproduce the remaining four rows of that table exactly.

**The corrected recipe needs THREE aggregator runs, not one**, because `final`, `best`
and `init` are three separate evaluation passes written into three directories, and v1's
six-row table is those three crossed with the two recurrent-state regimes each run
already reports internally:

```
for sel in final best init; do
  .venv/bin/python -m analysis.aggregate_results \
    --results-dir "results_v2/$sel" --checkpoint-dir checkpoints_v2
done
```

**`--selection` is not the flag that picks between them**, and reaching for it is the
natural wrong guess. It only affects `--manifest` output (verified in `main`); the report
is built from `--results-dir` and `--checkpoint-dir` alone. Passing `--selection best`
while pointing at the parent directory changes nothing and still yields an empty
comparison.

Shipped as `scripts/run_v2_analysis.sh`, committed, with a guard that refuses to
aggregate unless all 120 evaluation results are present.

### 19.2 `timeout` does not exist on macOS either

Added to §17.12's list of portability traps, which already records `setsid`. `timeout(1)`
is GNU coreutils and is **not** present on a stock macOS; the shell reports
`command not found: timeout` and, in a pipeline, the surrounding command may still appear
to proceed. Use `gtimeout` if coreutils is installed, or simply omit it and bound the
work by argument (e.g. `--steps 1280`) instead. Hit while writing §18.6's pre-flight.

### 19.3 A running bash script MUST NOT be edited in place

Recorded because it constrained this session's own remediation of §19.1. `bash` reads a
script incrementally, seeking by byte offset as it executes, so editing a script that is
currently running can make it resume at a shifted offset and execute garbage — a real
hazard for exactly the long-lived chained pipelines this project keeps building
(`scripts/run_v2_pipeline.sh` was mid-stage-1, blocked on a two-hour training launcher,
when §19.1 was discovered).

**The fix was therefore additive, not an edit:** the corrected aggregation shipped as a
new file (`scripts/run_v2_analysis.sh`) while the running script was left untouched on
disk. Its own stage 4 still runs the §14.11 command and still produces the empty
comparison described above; that output is superseded by the analysis script's, and is
harmless because every guard preceding it is unaffected. `run_v2_pipeline.sh` itself is
corrected only once it is no longer executing.
