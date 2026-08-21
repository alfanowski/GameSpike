# Phase 2a Results — the specialist reference runs (SPEC-A / SPEC-B)

Date: 2026-08-21
Author: Andrea Alfano ("Alfanowski"), with research support from Claude (Opus 5)
Scope: `docs/DESIGN_ROADMAP_PHASE2.md` §15, executed exactly as pre-registered there
**before** any of these numbers existed.

**What this is:** the two single-task specialists and their untrained anchors — controls C1
and C2 of `DESIGN_ROADMAP_PHASE2.md` §2.2, and the denominator and zero of §8.1's normalized
score. **It is not a continual-learning result** and cannot be: no policy here was trained on
more than one task, so §8.3's forgetting measure `F` and backward transfer are undefined
(§15.5).

---

## 1. Headline

**Both tasks pass the pre-registered go/no-go, in both recurrent-state regimes.** The gate
declared in §15.3 before any run existed was: *p < 0.05 on an exact two-sided permutation
test AND Cohen's d ≥ 1.0, specialist against its own untrained anchor, n = 10 vs n = 10
training seeds.*

| regime | task | specialist | init anchor | denominator | p | Cohen's *d* | verdict |
|---|---|---|---|---|---|---|---|
| `continuous` | 1-1 | 39.837 | 7.979 | 31.858 | 1.08e-05 | +10.889 | **GO** |
| `continuous` | 2-1 | 17.139 | −15.341 | 32.480 | 1.08e-05 | +10.361 | **GO** |
| `reset128` | 1-1 | 38.828 | 8.195 | 30.633 | 1.08e-05 | +7.863 | **GO** |
| `reset128` | 2-1 | 15.997 | −15.345 | 31.342 | 1.08e-05 | +10.387 | **GO** |

Every bootstrap 95% CI on the difference excludes zero, on the winning side.

**Read the p-values correctly.** `1.08251e-05` is **2/184756**, the *resolution floor* of an
exact permutation test at n = 10 vs n = 10 — it means the two groups separate completely
(every specialist seed above every init seed), not that a tiny p was precisely estimated.
`EXPERIMENT_LOG.md` §2 already records that floor as this design's limit. It is the strongest
statement this test can make and it is also the only statement it can make once separation is
total.

**Consequence, and it is the point of running these:** §8.1's denominator `R_spec − R_init` is
≈31 return points on both tasks, comfortably large relative to seed-level spread. **The
normalized score is well-conditioned on both tasks**, which is exactly what §15.3 said the
gate was protecting against. Phase 2a's full matrix (§10) is not blocked by a degenerate
metric.

---

## 2. Setup

Unchanged from §15.1's pre-registration, restated so every number has a provenance:

| | value |
|---|---|
| arm | `baseline` (`models/policy_value_gru.py`, 132,715 trainable parameters) |
| tasks | Super Mario Land world **1-1** and world **2-1** |
| seeds per task | 10, independent (0–9) |
| env steps per run | 1,000,064 |
| optimizer / clipping / init | Phase 1 v2's, inherited unchanged: Adam `lr=3e-4`, per-group clipping, centred embedding, `--embed-scale 3.0` |
| centring constant | `OBS_MEAN_PHASE2A` (the {1-1, 2-1} mixture mean) |
| action set | Phase 1's ten |
| evaluation | 30 episodes, eval seed 0, both regimes, `final` and `init` selections |
| unit of analysis | **the training seed, never the episode** (`RESULTS.md` §2.4) |
| evaluations | 2 checkpoint-tasks × 2 eval-tasks × 10 seeds × 2 regimes × 2 selections = **160, all succeeded** |

**Nothing was tuned in response to any of these numbers**, as §15.1 committed in advance.

**Pre-launch invariant, verified:** §15.3.1 required the untrained anchors to be
task-independent, since at `--steps 0` the task cannot reach the weights. Checked in-pipeline
across all ten seeds before the 20M env steps were spent: `--task 1-1` and `--task 2-1`
produce **bit-identical** model tensors at the same seed, and different seeds differ.

---

## 3. The performance matrix

§8.2's `R[i, j]`, first row only — these are specialists, so there is no sequence yet. Cells
are §8.1's normalized score, seed-level, against anchors frozen from the ten-seed means.
By construction a specialist scores exactly 1.000 on its own task and an untrained policy
0.000.

**`reset128` (the regime training actually used):**

| trained on ↓ / evaluated on → | 1-1 | 2-1 |
|---|---|---|
| **1-1** | +1.000 | +0.653 |
| **2-1** | **+1.462** | +1.000 |

**`continuous`:**

| trained on ↓ / evaluated on → | 1-1 | 2-1 |
|---|---|---|
| **1-1** | +1.000 | +0.629 |
| **2-1** | **+1.329** | +1.000 |

Raw `mean_extrinsic_return` behind the same cells, `reset128`: 38.828 / 5.108 on the top row,
**52.992** / 15.997 on the bottom.

---

## 4. Forward transfer, and one finding nobody pre-registered

### 4.1 Both directions transfer above the untrained anchor

§15.4's descriptive measurement. Under §8.1's normalization the untrained reference is 0.0 by
construction, so forward transfer is just the zero-shot normalized score:

| regime | trained → evaluated | normalized | raw vs init | p | *d* |
|---|---|---|---|---|---|
| `reset128` | 1-1 → 2-1 | +0.653 | 5.108 vs −15.345 | 1.08e-05 | +13.041 |
| `reset128` | 2-1 → 1-1 | **+1.462** | 52.992 vs 8.195 | 1.08e-05 | +3.538 |
| `continuous` | 1-1 → 2-1 | +0.629 | 5.081 vs −15.341 | 1.08e-05 | +13.358 |
| `continuous` | 2-1 → 1-1 | **+1.329** | 50.333 vs 7.979 | 1.08e-05 | +3.385 |

**Neither task is a closed box.** A policy that has never seen the other level still plays it
far better than an untrained policy. That is a real, if unsurprising, precondition for the
continual-learning study: if transfer were zero in both directions there would be nothing for
interference or consolidation to act on.

### 4.2 Transfer is strongly ASYMMETRIC — and the harder task's specialist is the better 1-1 player

The number that stands out is `2-1 → 1-1 = +1.462`. **Above 1.0 means the 2-1 specialist
outscores the 1-1 specialist on 1-1 — the level the 1-1 specialist was trained on and the 2-1
specialist has never seen.**

Tested directly (post-hoc; see the caveat below):

| regime | evaluated on | foreign specialist | own specialist | difference | p | *d* |
|---|---|---|---|---|---|---|
| `reset128` | 1-1 | 52.992 | 38.828 | **+14.164** | 0.00736 | +1.092 |
| `continuous` | 1-1 | 50.333 | 39.837 | **+10.495** | 0.0210 | +0.827 |
| `reset128` | 2-1 | 5.108 | 15.997 | −10.889 | 1.08e-05 | −3.651 |
| `continuous` | 2-1 | 5.081 | 17.139 | −12.058 | 1.08e-05 | −3.863 |

The reverse direction does not hold at all: the 1-1 specialist is much *worse* on 2-1 than
2-1's own specialist. So the asymmetry is not symmetric noise — training on the harder level
produced a better player of the easier one, while the converse failed.

Mean episode lengths (`reset128`) are consistent with the obvious reading — that 2-1 forces
more careful play, which then generalises:

| trained → evaluated | mean episode length |
|---|---|
| 1-1 → 1-1 | 324.5 |
| 2-1 → 1-1 | **477.8** |
| 2-1 → 2-1 | 236.1 |
| 1-1 → 2-1 | 148.2 |

§14.5 already measured *why* 2-1 is harder: a naive hold-right policy dies there at frame 235
against 1-1's 336, burning both lives inside eight seconds.

**Three caveats, and they are load-bearing.**

1. **This was NOT pre-registered.** It was found by reading the off-diagonal after the
   numbers existed. It deserves the same discount `RESULTS.md` §9 applies to its own §5
   decomposition. What it has going for it is the same thing that decomposition had: it was
   not the result anyone was looking for.
2. **It does not survive a family-wise correction.** Four extra comparisons were run beyond
   the four pre-registered gates. At eight comparisons, Bonferroni asks for p < 0.00625;
   `reset128`'s 0.00736 and `continuous`'s 0.0210 both **fail** that. The effect is
   **suggestive, not established.** The two 2-1 rows do survive anything, but they are the
   unsurprising direction.
3. **Two tasks, one architecture, one hyperparameter set.** "Train on the harder level" is not
   a recommendation this document is entitled to make from n = 2 levels.

---

## 5. Cross-check against Phase 1

Phase 1 v2's baseline arm on 1-1 scored **43.206** (`continuous`) and **41.783** (`reset128`).
Phase 2a's 1-1 specialist scores **39.837** and **38.828** — the same family of numbers,
slightly lower.

That gap is expected and its sources are known and disclosed in advance (§15.5): Phase 2a
boots through PyBoy's wrapper rather than from power-on (observation slot 5 starts at 1.00
against 0.99), and it centres the embedding on the {1-1, 2-1} **mixture** mean rather than
1-1's own. **These are not the same instrument and their numbers should not be tabled side by
side as if they were.** The agreement is close enough to be a useful sanity check that the
task axis did not break the training path, and that is all it is being used for here.

---

## 6. Limitations

Everything from `RESULTS.md` §9 that is about the shared machinery still applies and is not
repeated in full: a deterministic environment, so 30 episodes per checkpoint measure
policy-sampling variance only and the unit of analysis is the training seed; n = 10 seeds;
one shared hyperparameter set never tuned for either task; three of twelve observation slots
hardcoded to zero; both arms scored in a `continuous` regime neither was trained in.

Specific to this document:

- **These are references, not a research result.** Their entire purpose is to be divided by.
- **Two levels of one game.** Nothing here generalises to another level, let alone another
  title. §14.7 already showed Kirby's progress signal is a harder problem than Mario's.
- **The exact-permutation floor is doing a lot of work.** Four of the headline rows sit at
  2/184756 because the groups separate completely. That is the strongest available statement
  and simultaneously an admission that the test cannot resolve *how* much better.
- **`d` values between +7.9 and +10.9 are not subtle effects and should not be quoted as
  though they were surprising.** Trained-beats-untrained at 1M steps is the expected outcome;
  the gate existed to catch its *failure*, not to celebrate its success.
- **The asymmetric-transfer finding of §4.2 does not survive multiple-comparison correction.**
  Stated again here because it is the most quotable number in this document and the most
  likely to be over-read.
- **No `best` checkpoint selection.** §15.1 dropped it deliberately: it is selected on
  training reward and adds nothing to a reference measurement.

---

## 7. What this does and does not license

**Licensed.** §8.1's normalized score is well-conditioned on both tasks, so the metrics
`DESIGN_ROADMAP_PHASE2.md` §8 pre-registers can now actually be computed. The pre-registered
precondition for proposing Phase 2a's full matrix (§10) is **met**.

**Not licensed.** Starting that matrix. §15.1 authorised SPEC-A, SPEC-B and INIT and nothing
else; INT, SEQ, the Q3 ablation and any mitigation arm remain unauthorised, and the decision
to spend that compute is the project owner's. This document is the evidence that decision
should rest on, not a substitute for it.

---

## 8. Reproduction

- **Everything:** `MARIO_LAND_ROM_PATH=… JOBS=8 bash scripts/run_phase2a_pipeline.sh`
  — INIT anchors, the §15.3.1 invariant check, both specialist matrices, then the
  cross-task evaluation. Ran unattended 14:47 → 17:01 CEST on a 10-core M4 (INIT + invariant
  to 14:48, SPEC 1-1 to 15:43, SPEC 2-1 to 16:42, 160 evaluations to 17:01).
- **Analysis:** `python -m analysis.phase2a_reference --results-dir results_p2a`, which
  reuses `analysis/aggregate_results.py`'s exact permutation test, Cohen's *d* and bootstrap
  CI — the same instrument Phase 1 used, per `EXPERIMENT_LOG.md` §17.11.
- **Raw results:** `results_p2a/{final,init}/eval_baseline_ckpt{W-L}_on{W-L}_seed{N}_{regime}.json`,
  160 files, committed. Report at `results_p2a_report.txt`.
- **Checkpoints:** `checkpoints_p2a/baseline_task{W-L}_seed{N}/` — gitignored and not
  distributed, reproducible from the seed.
- **The final checkpoint is `step_1000064.pt`**, not `step_1000000.pt`.
