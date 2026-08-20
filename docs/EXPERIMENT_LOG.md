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
