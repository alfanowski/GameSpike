# GameSpike

A frozen spiking-reservoir reinforcement-learning agent for Super Mario Land
(Game Boy), evaluated against a matched-trainable-parameter GRU baseline under
a mandatory scientific control. Sibling project to
[`spiking-reservoir-lm`](https://github.com/alfanowski/spiking-reservoir-lm)
(frozen-reservoir byte-level text generation) and an unpublished biosignal
(ECG/EEG/EMG) design that share the same underlying reservoir-computing core —
reused here, unmodified, applied to real-time game control instead of language
or biosignal interpretation.

---

## Status, 2026-08-21: paused, with the question it was built to ask answered

**Active work on this repository is paused.** The repository owner chose to stop here after
completing Phase 1 and two bounded follow-up tracks. Nothing is abandoned and nothing is
withdrawn: every result is committed, every protocol is pre-registered in the ledger, and
every run is reproducible from its seed, so resuming later costs nothing structural.

**The headline, stated first and not buried under anything that came after: the frozen
spiking reservoir lost to the matched-parameter trained GRU baseline.** It lost under the
protocol as pre-registered (v1), and it lost again in a corrected rerun with two diagnosed
defects removed (v2) — the second time by a slightly *larger* margin, and without the
episode-length artefact that had flattered it the first time. Details and caveats in
[`docs/RESULTS.md`](docs/RESULTS.md) and in [Phase 1 below](#phase-1--the-result-the-frozen-reservoir-loses-twice).

That is a negative result, obtained rigorously, on a question that was posed so it could be
answered either way. **It is an answer, not a failure of the project's purpose** — the
mandatory-control design exists precisely so this outcome is scientifically informative
rather than a silently discarded run. What it does *not* license is described in
`RESULTS.md` §10 and §22, and is repeated in the sections below.

**What exists and runs.** All 12 tasks of the Phase 0 + Phase 1 implementation plan are
complete: the PyBoy environment wrapper with an empirically-confirmed Super Mario Land RAM
map, the discrete action space, both competing policy-value models at a verified-matched
trainable-parameter budget, the trajectory-novelty curiosity gate, the PPO core, rollout
collection, the training loop and the evaluation harness — plus, added since, a task axis for
starting at an arbitrary world-level, a selectable resonate-and-fire neuron model alongside
the default LIF, the results-aggregation and reservoir-health analysis modules, and the run
drivers for both published matrices. A 617-test suite covers it.

**Everything described below is merged into `main`.** Each of the three tracks was reviewed
and merged by the repository owner rather than self-merged by the session that produced it,
under `EXPERIMENT_LOG.md` §13's rule that anything revising or extending a published
conclusion is his to approve.

**What is open at the pause point**, listed so nothing reads as quietly settled:

- **Roadmap Phase 2's full experiment matrix is proposed, counted and unauthorised.** The
  validated testbed exists; the 160M-env-step matrix it would feed has not been started, and
  that decision is the owner's.
- **The reservoir's runaway operating point is unresolved.** Final spike rate 0.194 against
  a documented ~2% healthy band, regulated by neither shipped fix
  (`EXPERIMENT_LOG.md` §21.5). Track A diagnosed the mechanism behind it and demonstrated a
  structural attenuator for it, but could not place that attenuator at the control's
  operating point — so the question is better understood than it was and is still open. It
  remains the most concrete open architectural question this project has.

---

## Roadmap: the stated larger goal, and where each phase actually stands

**Stated explicitly, so it never reads as scope-narrowing by omission:** the actual goal of
this project is a general game-playing agent across Nintendo handheld titles — explicitly
including Game Boy Advance and **Pokémon Fire Red** — not a Mario-Land-only system. The
single-game scope of Phase 1 is a deliberate *sequencing* choice, the same "prove one case
scientifically before generalizing" discipline already used in the sibling
[`spiking-reservoir-lm`](https://github.com/alfanowski/spiking-reservoir-lm) project, not a
redefinition of the goal down to one game. Full version:
[`docs/DESIGN.md` §1.1](docs/DESIGN.md#11-roadmap-this-document-is-phase-1-of-a-stated-larger-goal--not-the-destination).

> **Two independent "Phase" numbering schemes exist in this project and conflating them is an
> error `DESIGN.md` §7 explicitly warns against.** The list below is the project-wide
> **roadmap** (`DESIGN.md` §1.1). `DESIGN.md` §7 separately defines a **build order internal
> to roadmap Phase 1** — its Phase 2 is the resonate-and-fire neuron swap, its Phase 3 is
> DLIF/RSSR. "Roadmap Phase 2" (multi-game) and "build-order Phase 2" (resonate-and-fire) are
> different things that happen to share a number.

- **Roadmap Phase 1 — CLOSED.** Super Mario Land, Game Boy: frozen reservoir vs.
  matched-parameter trained-GRU baseline, under a mandatory scientific control. Both the
  as-specified comparison (v1) and the corrected rerun (v2) are complete, published and
  merged. **The answer is negative for the reservoir arm**, in both. See
  [`docs/RESULTS.md`](docs/RESULTS.md).
- **Roadmap Phase 2 — designed; its cross-*level* testbed is validated; its cross-*title*
  experiment is scoped but not started.** Multi-game generalization *within* Game
  Boy / Game Boy Color — continual learning and catastrophic forgetting across titles, which
  is genuinely open research (see DeepMind's SIMA 1/2 and the continual-RL literature —
  Unicorn, DisCoRL, CORA — for the context it is scoped against). The design is
  [`docs/DESIGN_ROADMAP_PHASE2.md`](docs/DESIGN_ROADMAP_PHASE2.md), proposed on the trained-GRU
  architecture rather than the reservoir, since the reservoir was Phase 1's *question* rather
  than a premise of this roadmap. It splits into:
  - **Phase 2a (cross-level, inside Super Mario Land — precursor, not the roadmap goal).**
    Two single-task specialist references have been trained and published
    ([`docs/RESULTS_PHASE2A.md`](docs/RESULTS_PHASE2A.md)); **the full Phase 2a matrix — the
    interleaved and sequential continual-learning conditions that would actually measure
    forgetting — is proposed and unauthorised, and no continual-learning result exists.**
  - **Phase 2b (cross-*title* — the roadmap's actual Phase 2).** Unblocked in principle: the
    owner supplied a second cartridge, **Kirby's Dream Land (USA, Europe)**, after the design
    doc's original one-ROM constraint was written. **Not started**, and a viability probe
    found its RAM-map work is harder than the design estimated — see Track B below.
- **Roadmap Phase 3 (Game Boy Advance as a platform) — unchanged.** `pygba`/mGBA's direct
  Python bindings are dead on Apple Silicon — a hard native `SIGBUS`, root-caused to a cffi
  ABI-mode fragility on strict-alignment ARM64. A **Lua-scripting + local-socket bridge to
  `mgba-headless`** was built and verified end-to-end instead: 12/12 checks passed,
  deterministic byte-identical replays, ~3,062 fps single instance and ~15k fps aggregate
  across 8 parallel instances, with real gameplay driven through Super Mario Advance's World
  1-1. This is a **proven scratchpad spike, not yet a component of this repository** —
  formalizing it into a real `gymnasium.Env` is Phase 3's remaining work.
- **Roadmap Phase 4 (Pokémon Fire Red / RPG targets) — unchanged.** Needs Phase 3's platform
  work *plus* a hierarchical planning layer that does not exist yet in this architecture. The
  available Fire Red ROM is also the **Italian release** (cartridge code `BPRI`), not the US
  `BPRE` that published community RAM maps target, so its RAM addresses will need the same
  from-scratch empirical confirmation Super Mario Land's did.

---

## Phase 1 — the result: the frozen reservoir loses, twice

Reported in the order it happened, with v2 appended beneath v1 rather than replacing it.
Full write-up, including every control, confound and limitation:
[`docs/RESULTS.md`](docs/RESULTS.md).

### v1 — the comparison as pre-registered

2 arms × 10 independently-seeded training runs × 1,000,064 env steps each (7,813 PPO
updates), evaluated over 120 runs of the harness. Under the pre-registered protocol the
frozen spiking reservoir **loses** to the matched-parameter trained GRU baseline on the
declared scoreboard (`mean_extrinsic_return`), in every trained condition measured — both
checkpoint-selection rules, both recurrent-state regimes — **by 6.17 to 7.72 points of mean
episode return, exact two-sided permutation p between 0.000433 and 0.001635, Cohen's *d*
between −1.27 and −1.91.** Every bootstrap 95% CI on the difference excludes zero, on the
losing side. The two untrained arms are statistically indistinguishable (p = 0.66 and 0.36),
which is the control that makes the trained comparison readable.

**v1's own largest caveat pointed against the reservoir, not for it.** The reservoir's
episodes ran roughly **6× longer** than the baseline's (1,917.0 vs 314.9 steps, `final`,
`continuous`) at a far lower reward rate: the baseline earned **5.96×** more reward per step.
The reservoir had learned to survive without progressing, and the return-based scoreboard was
the *flattering* normalisation. Any reader preferring per-step reward gets a larger reservoir
deficit, not a smaller one.

### v2 — the corrected rerun, with two diagnosed defects removed

Two defects were root-caused after the v1 runs completed:

1. **A gradient-clipping confound.** A single *global* `clip_grad_norm_` let an exploding
   gradient in the reservoir arm's 416-parameter embedding dominate the clip coefficient and
   suppress the readout's effective gradient — so v1 could not cleanly separate "the frozen
   reservoir is a weaker feature extractor" from "its readout was barely trained". Fixed with
   per-group clipping.
2. **An input-calibration defect.** The observation is DC-dominated and the LIF neuron
   amplifies the useless part, leaving a large frozen per-unit membrane offset at
   initialisation under the `legacy` embedding init. Fixed with a centred embedding
   initialisation at `--embed-scale 3.0`.

Both were applied identically to both arms, and a second full matrix (2 arms × 10 seeds ×
1,000,064 steps, 120 fresh evaluations) was run.

**The frozen reservoir still loses, in all four trained conditions, by 7.26–8.97 points of
mean episode return (exact permutation p between 0.000141 and 0.003497, Cohen's *d* between
−1.45 and −2.22).** In the headline condition (`final`, `continuous`) the gap is **−8.97
points (34.2404 vs 43.2060), p = 0.000141, d = −2.2237**.

What changed, and what did not:

- **The corrections worked.** Both pre-registered construction hypotheses confirmed: the
  dead-gradient budget fell from 9.80% to **0.16%** of readout columns (A7), the silent-unit
  fraction from 46.52% to **32.16%** (A9). The reservoir measured in v2 is a genuinely
  better-calibrated one than the reservoir measured in v1.
- **The training-reward gap closed and the evaluation gap did not.** The baseline-to-reservoir
  ratio of mean per-update extrinsic *training* reward fell from **5.8220× to 1.3758×** over
  all 7,813 updates, while the evaluation gap **widened slightly** (`final`, `continuous`:
  −7.7167 → −8.9656).
- **v1's biggest caveat did not survive.** The ~6× episode-length asymmetry is gone: v2's two
  arms are statistically indistinguishable in episode length (353.07 vs 336.88 steps,
  **p = 0.072517, not significant**). The return ratio (1.262×) and the per-step ratio
  (1.317×) now agree. **v2's headline is neither conservative nor inflated; it is simply the
  number.** The reservoir abandoned the survival strategy and now plays the same game the
  baseline plays, less well.

**v1 is not withdrawn and is not superseded in substance.** It is what the pre-registered
protocol produced, and v2 is reported beneath it rather than in place of it. v2's own two
largest limitations are stated as loudly: **two treatments were changed at once** and are not
decomposed, and a **clipping group-count asymmetry favours the baseline**.

### What this does and does not license

- **It is not evidence about frozen reservoirs in general.** It is evidence about one
  configuration at one parameter budget under one hyperparameter set — a better-calibrated
  instance in v2, but the same configuration. Its operating point still leaves the healthy
  band during training.
- **It is not evidence that the remaining gap is irreducible.** The reservoir's training
  reward was still rising at the budget's end.
- **It is not evidence about the multi-game goal.** Roadmap Phases 2 and 4 are separate
  questions this experiment bears on only by supplying the premise they were sequenced behind.
- **One game, one level** — Super Mario Land, world 1-1. Nothing here generalises to another
  level without measurement, let alone another title.

---

## The two follow-up tracks run after Phase 1 closed

Both were bounded, both pre-registered their decision rules before producing numbers, and
neither produced a headline that displaces Phase 1's.

### Track A — build-order Phase 2 (resonate-and-fire): a confirmed mechanism, and a pre-registered stop with no result

Branch `feat/resonate-and-fire-pilot`, **PR #5, merged.**

`DESIGN.md` §7 gates the resonate-and-fire neuron swap on *"once Phase 1 shows the reservoir
arm beating baseline"*. **That precondition is not met.** This pilot ran anyway on a specific
mechanistic argument — that resonate-and-fire dynamics would structurally attenuate the DC
drive behind the reservoir's runaway operating point (`EXPERIMENT_LOG.md` §21.5) — with a
mechanism prediction, five validity gates and a stopping rule all committed **before any
implementation code existed**.

**The mechanism works, measured against its own advance prediction.** Mean DC gain **1.7873**
against **1.7846** predicted, DC/AC **0.7791** against **0.7779** predicted — a **÷5.60**
attenuation versus LIF — and it is **throughput-neutral** (439.7 vs 438.9 env-steps/s in the
same session). The LIF path was verified bit-for-bit unmoved, and ω ≡ 0 reproduces
`snn.Leaky(beta=0.9)` exactly.

**One validity gate did not pass, and the pilot stopped on its own rule.** Matching the LIF
control's *mean* firing rate leaves **30.31%** of the reservoir silent at initialisation
against a **15%** threshold and the LIF control's **1.81%**. Three refined grid points do
satisfy both gates — but reaching them costs **1.8×** the control's firing rate, which would
start the two arms 1.8× apart on *exactly the quantity the comparison exists to measure*.

That inconsistency is the finding: LIF's 8,192 units share one pole, so mean rate and silent
fraction move together; a frequency-dispersed bank cannot set both with one knob. **No
training was run, no comparison was produced, and no number from this track touches
`docs/RESULTS.md`'s verdicts.** The pilot also corrected its own committed prose about *why*
the fast units go silent — a spectral-density argument, not the band-power one originally
written — which reads against the author, not for him.

Two things it produced that are worth naming, both flagged at their real strength:

- **A correction of record against the already-published `RESULTS.md`**, now itself published
  as [`RESULTS.md` §24](docs/RESULTS.md) (fixture provenance: the committed 6,000-step fixture
  §19 measures against is not the collection `OBS_MEAN` was fitted to). It is
  **verdict-neutral** — every §19 verdict is taken at the final checkpoint, A7 and A9 both
  stay far from their band edges, the runaway is still ×14 and the spike rate still ends at
  0.194 — and it reads **in the reservoir's favour**, which is the direction worth disclosing
  loudest. It is appended beneath v2, **editing none of v1's or v2's numbers**, on the same
  rule by which v2 was appended beneath v1. No remedy is applied: each available one changes
  what a published number means, which is a decision rather than a correction.
- **An incidental, post-hoc, n = 1 observation** that resonate-and-fire appears to remove v1's
  exploding-embedding-gradient pathology outright (median embedding `grad_norm` **0.4317** vs
  LIF's **3.705e4**). The gradient effect spans five orders of magnitude; the reward
  comparison beside it is **not usable at n = 1 and no claim is made from it.**

### Track B — roadmap Phase 2 (multi-game): a design document and a validated cross-level testbed

Branches `design/roadmap-phase2-gru` (**PR #4, merged**) and `results/phase2a-specialists`
(**PR #6, merged**).

**The design** ([`docs/DESIGN_ROADMAP_PHASE2.md`](docs/DESIGN_ROADMAP_PHASE2.md)) scopes the
roadmap's actual multi-game goal on the GRU architecture, with pre-registered forgetting and
transfer metrics, the mandatory controls transposed from `DESIGN.md` §5, a counted compute
budget (90 training runs, ≈160M env steps) and its open decisions listed rather than quietly
made. It found the development machine held **exactly one Game Boy ROM**, and split the phase
accordingly: **2a** cross-*level* inside Super Mario Land (needs no second cartridge) and
**2b** cross-*title* (the roadmap goal). The owner has since supplied **Kirby's Dream Land
(USA, Europe)**, so 2b is unblocked — but the sequencing did not change, because 2a was never
merely a workaround for a missing cartridge.

**A viability probe produced one genuine negative worth reading before anyone starts 2b.**
Save-state determinism and arbitrary-level start both passed on Super Mario Land, and Kirby
binds with its published addresses holding — but **nothing traverses Kirby's first level**:
hold-right reaches `scroll_x` 13, right-plus-jump 30, right-plus-fly 71 before stalling, and
uniform random play 3–21. Mario's blind hold-right monotonic RAM-scan discovery method
therefore does not transfer to Kirby; its dense signals are score and survival, not spatial
progress. The probe established *that* traversal stalls and *where*, not *why*.

**The bounded first training step (SPEC-A / SPEC-B).** Two single-task GRU specialists —
Super Mario Land worlds **1-1** and **2-1**, 10 independent seeds each, 1,000,064 env steps
each (≈20M total), 160 evaluations, all succeeded — against untrained anchors, with a go/no-go
gate declared before any number existed: *permutation p < 0.05 **and** Cohen's d ≥ 1.0, on
both tasks.* Full write-up: [`docs/RESULTS_PHASE2A.md`](docs/RESULTS_PHASE2A.md).

**Result: GO on both tasks in both evaluation regimes** — p = 1.08e-05 (which is 2/184756, the
*resolution floor* of an exact permutation test at n = 10 vs n = 10, meaning the groups
separate completely), d between +7.86 and +10.89, with a normalisation denominator of ≈31
return points on both tasks.

**Read that correctly, because it is the easiest number in this repository to over-read.**
A trained policy beating an untrained one at 1M steps is the *expected* outcome, not a
discovery. The gate existed to catch its **failure** — a near-zero `R_spec − R_init` would
have made every later continual-learning metric unstable, a *mathematical* defect in the
measuring instrument. What was established is that **the evaluation methodology's denominator
is well-conditioned**, so the metrics Phase 2 pre-registers can now actually be computed.
This is infrastructure validation. **Nothing here is a continual-learning result** — no policy
was trained on more than one task, so forgetting and backward transfer are undefined — and
nothing here authorises starting the full matrix.

**One unregistered, exploratory finding, reported with its discount attached.** Transfer
between the two levels looks strongly **asymmetric**: the 2-1 specialist outscores the *1-1
specialist* on 1-1, a level it never trained on (52.992 vs 38.828 in `reset128`, +14.164,
p = 0.00736, d = +1.092; 50.333 vs 39.837 in `continuous`, p = 0.0210). The converse fails
badly, so it is not symmetric noise, and episode lengths point the same way (477.8 vs 324.5
steps). **But it was not pre-registered — it was found by reading the off-diagonal after the
numbers existed — and it does not survive a Bonferroni correction over the eight comparisons
run (0.00736 against a required 0.00625).** It is **suggestive, not established**, and n = 2
levels does not license "train on the harder level" as advice.

---

## Documents

- [`docs/RESULTS.md`](docs/RESULTS.md) — **Phase 1 results, v1 and v2**: the headline, the
  setup, the two controls, the return-vs-per-step decomposition, the gradient-clipping
  confound, the reservoir-construction findings and the limitations — then, appended beneath
  and never edited into v1, the corrected v2 comparison with the A7/A9 verdicts and a
  v1-vs-v2 side by side — and beneath *that*, §24's verdict-neutral correction of record on
  the observation fixture's provenance, appended on the same rule again.
- [`docs/RESULTS_PHASE2A.md`](docs/RESULTS_PHASE2A.md) — **the Phase 2a specialist
  references** (SPEC-A / SPEC-B): the pre-registered gate and its GO verdicts, the performance
  matrix, the unregistered asymmetric-transfer finding with its multiple-comparison discount,
  and an explicit statement of what it does and does not license. **It is a reference
  measurement, not a research result.**
- [`docs/DESIGN.md`](docs/DESIGN.md) — full design rationale: why a reactive platformer (not
  an RPG) was chosen as the first target, the architecture, the mandatory scientific control,
  the roadmap (§1.1), the build order (§7) and what is explicitly out of scope.
- [`docs/DESIGN_ROADMAP_PHASE2.md`](docs/DESIGN_ROADMAP_PHASE2.md) — **roadmap Phase 2's
  design**: multi-game generalization and catastrophic forgetting on the GRU architecture,
  with pre-registered metrics, transposed controls, a counted compute budget, the
  testbed-viability probe results and the SPEC-A/SPEC-B pre-registration. **The full matrix in
  it is a proposal and has not been started.**
- [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) — the operational ledger: pre-registered
  protocols and ablations, verified invariants, corrections of record, the hazards already hit
  (including a mid-run machine power loss and the determinism check that closed it), and the
  §13 rule that headline-bearing changes are the owner's to merge.
- [`docs/superpowers/plans/2026-08-19-mario-ppo-reservoir.md`](docs/superpowers/plans/2026-08-19-mario-ppo-reservoir.md) —
  the implementation plan (build-order Phase 0 + Phase 1 only).

## What this project is / is not

- **Is:** a test of whether a frozen, never-trained spiking reservoir —
  already shown (in the sibling LM project) to contribute nothing to storable
  knowledge, and therefore structurally unsuited to open-domain text
  generation — is nonetheless a useful real-time feature extractor for a
  bounded, reactive control task, where a frozen reservoir's actual strength
  (rich nonlinear dynamics obtained "for free," no gradient descent through the
  recurrent core) is a plausible structural fit. **That test has now been run, and the answer
  on this task is no.**
- **Is not, yet:** a claim of state-of-the-art game-playing performance, a general-purpose
  game-playing agent, or an RPG/strategy agent. Today this repository is a single-game
  experiment with a validated two-level testbed attached — that is what exists right now, not
  a permanent ceiling. Pokémon-style targets were explicitly considered and rejected as the
  *first* target (see `docs/DESIGN.md` §1) for the same structural reason the LM project could
  not compete on open-domain generation: this architecture has no mechanism for storing broad
  accumulated knowledge, only for reacting to a state stream — a genuinely permanent property
  of a frozen reservoir plus a small trained readout, not something more training steps fixes.
  That is exactly why the roadmap sequences an RPG target (Phase 4) behind a hierarchical
  planning layer that has to be designed and added on top.
- **Is not:** a continual-learning result. Roadmap Phase 2's forgetting metrics are defined
  and their instrument is validated; **no condition that could measure forgetting has been
  run.**
- **Every result, positive or negative, is reported as such.** The mandatory-control design
  (a matched-parameter trained GRU baseline) exists specifically so a negative result — the
  reservoir failing to beat a conventional trained recurrent policy at the same parameter
  budget — is scientifically informative rather than a silently discarded run. Phase 1 ended
  in a negative and Track A ended with no result at all; both are written up at the same
  length and with the same care a win would have received.

## Repository structure

```
GameSpike/
├── docs/
│   ├── DESIGN.md                  # design rationale, roadmap (§1.1), build order (§7)
│   ├── RESULTS.md                 # Phase 1 results write-up (v1, then v2 appended)
│   ├── RESULTS_PHASE2A.md         # Phase 2a specialist references (SPEC-A / SPEC-B)
│   ├── DESIGN_ROADMAP_PHASE2.md   # roadmap Phase 2 design — a proposal, not a result
│   ├── EXPERIMENT_LOG.md          # operational ledger + pre-registrations + corrections
│   └── superpowers/plans/         # implementation plan(s)
├── envs/                          # PyBoy wrapper, RAM-address map, action space, boot
├── models/                        # vendored frozen reservoir + policy-value models
├── training/                      # PPO, rollout collection, novelty gate, train/evaluate
├── analysis/                      # results aggregation, reservoir health, decompositions
├── scripts/                       # run drivers and pipelines (v2 matrix, Phase 2a, probes)
├── results/, results_v2/          # Phase 1 evaluation JSONs (v1, v2) — committed
├── results_p2a/                   # Phase 2a evaluation JSONs — committed
├── tests/                         # pytest suite (TDD — written alongside each task)
└── checkpoints*/                  # run outputs (gitignored, reproducible from seeds)
```

Every **task-performance** number quoted in a results document comes from a committed
evaluation JSON. The checkpoints those JSONs were produced from are gitignored and not
distributed, and are regenerable from the seed. Reservoir-health numbers are measured against
a committed observation fixture instead, whose provenance carries its own disclosed caveats.

## Setup

```bash
git clone https://github.com/alfanowski/GameSpike.git
cd GameSpike
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**You must supply your own legally-dumped Super Mario Land ROM** — it is never
committed to this repository (`.gitignore` excludes `*.gb`/`*.gbc`). Set
`MARIO_LAND_ROM_PATH` to its path before running the test suite or training
scripts; tests that require it skip cleanly when it is unset.

## Usage

### Training

```bash
python -m training.train --arm baseline  --rom "/path/to/Super Mario Land (World).gb" \
                         --steps 100000 --seed 0
python -m training.train --arm reservoir --rom "/path/to/Super Mario Land (World).gb" \
                         --steps 100000 --seed 0
```

Outputs go to `{--checkpoint-dir}/{arm}_seed{seed}/` (default `checkpoints/`), which
holds `step_{N}.pt` checkpoints and `train_log.jsonl` — one JSON object per PPO
update (step, mean reward, extrinsic-only mean reward, the three loss terms and the
pre-clip gradient norm), appended live, so a running job can be watched and plotted
as it goes. Both the arm and the seed are in the path *and* inside every checkpoint,
so the two arms and multiple seeds never overwrite each other and a checkpoint file
always self-identifies.

`--seed` drives both arms' trainable init and action sampling **and** the reservoir
arm's frozen weights, so the same seed reproduces a run exactly and different seeds
vary both arms symmetrically. **This matters for the comparison**: a real §5 verdict
needs several independently-trained seeds per arm, not one run each.

The Phase 1 v2 configuration — the one whose numbers are published — adds
`--grad-clip-mode per-group --embed-init-mode centered --embed-scale 3.0`, applied
identically to both arms.

Other flags: `--rollout-len` (truncated-BPTT window, default 128), `--checkpoint-every`
(default 10000 steps; a final checkpoint is always written regardless),
`--checkpoint-dir`, `--resume-from PATH`, `--task` (world-level, Phase 2a),
`--neuron-model` (default `lif`; `rf` selects the resonate-and-fire reservoir from Track A,
which **no published comparison uses** — see that track's caveats before running it),
`--n-envs` (accepted, currently unused — collection is single-process).

### Evaluation

```bash
python -m training.evaluate --arm reservoir \
       --checkpoint checkpoints/reservoir_seed0/step_100000.pt \
       --rom "/path/to/Super Mario Land (World).gb" --episodes 10 --seed 0
```

Reports `mean_extrinsic_return` (the scoreboard), `mean_combined_return` (the reward
the loss optimised — diagnostic only, never the scoreboard) and mean episode length,
each with a sample standard deviation, standard error and the raw per-episode values.
`--json` emits the raw results dict instead.

Read the caveats it prints, and `training/evaluate.py`'s "WHAT THIS HARNESS CANNOT
TELL YOU" docstring section, before quoting any number from it. In short: the spread
it reports is policy-sampling variance only, **one checkpoint per arm cannot support
an arm comparison at all** (training-seed variance dominates and this cannot see it),
and by default the policy is scored over a whole continuous episode while training reset
its recurrent state every `--rollout-len` steps — `--state-reset-interval 128` runs the
matched-regime counterpart.

### Reproducing the published matrices

```bash
# Phase 1 v2 — the corrected comparison
MARIO_LAND_ROM_PATH=… bash scripts/run_v2_pipeline.sh

# Phase 2a — the specialist references
MARIO_LAND_ROM_PATH=… JOBS=8 bash scripts/run_phase2a_pipeline.sh
```

Each results document's own "Reproduction" section is authoritative for exact
invocations, wall-clock times and the analysis commands that regenerate the reports.

## Running the test suite

```bash
python -m pytest tests/ -q
```

617 tests, as of `main` at the time of writing. Without `MARIO_LAND_ROM_PATH` set:
**496 passed, 121 skipped** — everything needing a real ROM skips cleanly. With it set:
**616 passed, 1 skipped**, the remaining skip requiring a Phase 2a checkpoint, which is
gitignored and regenerable from the seed.

## License

[Apache License 2.0](LICENSE).

## Citation

```bibtex
@misc{alfano_gamespike,
  author = {Andrea Alfano (Alfanowski)},
  title  = {GameSpike: A Frozen Tensor-Train Spiking Reservoir as a
            Real-Time Feature Extractor for Reinforcement Learning, Evaluated
            Against a Matched-Parameter Trained Baseline},
  year   = {2026},
  note   = {See docs/DESIGN.md for the full design rationale, docs/RESULTS.md
            (v1 and v2) for the Phase 1 results, and docs/RESULTS_PHASE2A.md
            for the roadmap Phase 2a specialist references.},
  howpublished = {\url{https://github.com/alfanowski/GameSpike}}
}
```
