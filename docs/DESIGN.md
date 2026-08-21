# Spiking Reservoir RL — Design Document

Status: design stable; Roadmap Phase 1's implementation is complete and the experiment
it describes has not yet been run (see README.md for the precise state of the code)
Date: 2026-08-19 (amended 2026-08-20: §1.1, §5.1, §9.1, §10)
Author: Andrea Alfano ("Alfanowski"), with research support from Claude (Sonnet 5)

## 1. Context and honest framing

This is the third project built on the reservoir-computing toolkit first developed in
`spiking-reservoir-lm` (frozen LIF spiking reservoir, native tensor-train / MPO
construction, entanglement-entropy diagnostic) and extended in
`biosignal-reservoir-verticals-design.md` (resonate-and-fire neurons, trajectory-novelty
write-gate, RSSR, DLIF, applied to ECG/EEG/EMG). This document does not re-derive that
toolkit — see those two documents for the underlying mechanisms and their individual
provenance/verification.

**Why this domain fits, and why the LM project's domain didn't.** The LM project's
central, load-bearing finding was that a frozen reservoir contributes zero to storable
knowledge — capacity is bounded by the trained readout's parameter count alone
(`spiking-reservoir-lm` PAPER.md §7.1, citing Allen-Zhu & Li's knowledge-capacity scaling
laws). Open-domain text generation needs exactly the thing a frozen reservoir cannot
provide: broad factual knowledge. **Playing a reactive video game does not.** It needs
fast, reliable processing of a continuous stream of game state into a small, bounded
action space — structurally the same shape as the biosignal verticals (bounded
classification/regression/control on real-time signals), not the shape of open-domain
generation. This is the same "match the architecture to the task" discipline that
produced the biosignal design, applied to a third domain.

**What is and is not claimed as novel.** Reservoir computing / Echo State Networks have
been applied to reinforcement learning before, but concentrated on memory/POMDP tasks,
navigation, and metalearning — not pixel/RAM-based retro game playing (verified by live
search at design time, not assumed). The specific combination proposed here — a frozen
spiking reservoir as a game-playing RL feature extractor, with the biosignal vertical's
trajectory-novelty write-gate repurposed as a zero-extra-training-cost curiosity/intrinsic-
reward signal — was not found in the published literature at design time. As with the
prior two projects: every individual component is established; the combination is the bet.

**Explicit target selection reasoning.** Pokémon (or any RPG) was considered and
rejected as the *first* target for the same structural reason the LM project failed at
open-domain generation: RPGs demand strategic planning, inventory/menu state, and
game-specific knowledge accumulated over long horizons — exactly what a frozen reservoir
with a small trained readout cannot represent well. **Super Mario Land (Game Boy, 1989)**
is the first target: a scrolling platformer with a naturally-shaped reward (rightward
progress), a small discrete action space, and enemy/obstacle patterns with real periodic
structure — a good match for resonate-and-fire's frequency decomposition, once that
mechanism is validated in isolation (§7).

### 1.1 Roadmap: this document is Phase 1 of a stated, larger goal — not the destination

**Stated explicitly, so it never reads as scope-narrowing by omission:** the actual goal
of this project is a general game-playing agent, not a Mario-Land-only system. This
document's own single-game, single-variable-ablation scope (§7) is a deliberate
*sequencing* decision — the same "prove one case scientifically before generalizing"
discipline already applied in `spiking-reservoir-lm` and the biosignal verticals project —
not a redefinition of the goal down to one game. The full, real roadmap:

- **Phase 1 (this document + implementation plan).** One game (Super Mario Land, Game
  Boy), frozen reservoir vs. matched-parameter trained-GRU baseline, mandatory scientific
  control (§5). Answers: does a frozen reservoir help at all, on real game control, at a
  real matched parameter budget? Without this answered honestly first, any later claim
  about generalizing across games would rest on an unverified premise.
- **Phase 2 (future, not started).** Multi-game generalization *within* Game Boy/Game Boy
  Color — the same architecture trained/tested across more than one title, addressing the
  actual hard problem discussed when this was scoped (continual learning, catastrophic
  forgetting between games — see the brainstorming transcript's review of DeepMind's SIMA
  1/2 and the continual-RL literature, e.g. Unicorn, DisCoRL, CORA). Genuinely open
  research, not a solved problem this project can assume away — not attempted until
  Phase 1 produces a real result.
- **Phase 3 (Game Boy Advance as a platform — UNBLOCKED 2026-08-20, not yet formalized).**
  `pygba`/mGBA's direct Python bindings are dead on this platform (hard native `SIGBUS`,
  root-caused to a documented cffi ABI-mode fragility on ARM64 — see §9.1). Instead, a
  dedicated spike proved a **Lua-scripting + local-socket bridge to `mgba-headless`**
  (mGBA's own official scripting engine, running inside mGBA's own process — no
  Python/C FFI boundary at all) works end-to-end: 12/12 checks passed, deterministic
  (byte-identical replays), fast (3,062 fps single instance; 15k fps aggregate across 8
  parallel instances), and real player-controlled gameplay was verified (drove Super
  Mario Advance through World 1-1 via a blind RAM scan plus live button input). Two real
  mGBA bugs were found and worked around along the way: its savestate-to-file API is
  completely broken (use the savestate-to-buffer API instead), and headless mode SIGSEGVs
  on any pixel read (fixed with a small additive patch restoring the missing
  `setVideoBuffer` call it never makes). The full recipe (Lua script, Python client, exact
  build steps) exists as a proven scratchpad spike, **not yet turned into a real component
  of this repository** — that formalization (a proper `gymnasium.Env`, the same
  empirical-RAM-confirmation discipline already applied to Super Mario Land) is Phase 3's
  actual remaining work, now unblocked rather than stuck on a platform bug.
- **Phase 4 (the actual named target).** Pokémon Fire Red / Pokémon-style RPGs on GBA.
  **Andrea's own Fire Red ROM is already on hand** (`~/Desktop/gba/Pokemon - Versione
  Rosso Fuoco.gba`) — but it is the **Italian release** (cartridge code `BPRI`), not the
  US release (`BPRE`) that published community RAM maps target, so its addresses will
  need the same from-scratch empirical confirmation Super Mario Land's did, not a
  drop-in reuse of an existing Fire Red RAM map. This phase is explicitly *not*
  abandoned — sequenced behind Phase 3's formalization and a real architectural
  prerequisite genuinely called out from the start: an RPG's strategic/inventory/long-
  horizon demands need a planning layer above the frozen reservoir (a hierarchical
  extension), which has no design yet and is not something this document's architecture
  provides on its own.

This ordering was chosen, and remains, for real technical reasons (an unresolved platform
bug, an unsolved research problem, a missing architectural layer) — not because the
smaller scope was more convenient. Each phase's own document should restate this roadmap
so it's never implicit.

> **Note added 2026-08-21.** Roadmap Phase 2 now has its own design document —
> [`docs/DESIGN_ROADMAP_PHASE2.md`](DESIGN_ROADMAP_PHASE2.md) — proposing it on the trained
> GRU architecture rather than the frozen reservoir, since `RESULTS.md` §22 records that
> Phase 1's result "is not evidence about the multi-game goal" and the reservoir was Phase
> 1's experimental question rather than a premise of this roadmap. **That document is a
> proposal under review: nothing in it is implemented and no run has been started.** It
> restates this roadmap as the paragraph above asks, and it flags one external blocker this
> section could not have known: only one Game Boy ROM exists on the development machine, so
> the cross-*title* experiment waits on a second cartridge while a cross-*level* precursor
> does not. Nothing in §1.1 is edited, per this document's append-only practice.

## 2. Target hardware and constraints

- Development and default training machine: same MacBook Air M4 as the LM project
  (10-core GPU, 16GB unified RAM, no CUDA).
- **Free burst compute, if and when actually needed:** Kaggle Notebooks (T4 x2 or P100,
  30 GPU-hours/week, visible quota, 12h session cap) — the backbone already established
  and load-bearing for the separate `llm-lab`/"surf" project. Credentials already exist
  locally (`~/.kaggle/kaggle.json`, mode 600). Google Colab free tier as unpredictable
  overflow only, same as the LM project's own prior decision. **No rented/paid GPU** —
  same zero-budget discipline as the sibling project.
- **Explicit expectation, stated up front so it isn't silently assumed later:** for this
  project's actual parameter budget (frozen reservoir + a readout on the order of the LM
  project's ~1.5M trained parameters), GPU compute is very unlikely to be the bottleneck.
  The dominant cost is expected to be **emulator rollout throughput** — PyBoy stepping
  through the game is inherently CPU-bound and sequential per instance; a GPU does not
  accelerate it. The engineering lever that matters is parallelizing multiple PyBoy
  instances across the M4's CPU cores, not chasing GPU quota. This expectation must be
  measured, not assumed, once a working rollout loop exists (mirrors the LM project's own
  discipline of measuring TT speedups empirically rather than assuming them).
- Emulator: **PyBoy** (Python-native Game Boy emulator with direct RAM read access and a
  documented headless mode) — the same tool used by the published Pokémon-RL projects
  surveyed during this project's feasibility research (Whidden's original work,
  `pokemonred_puffer`). Concrete RAM addresses for Super Mario Land (Mario's X/Y position,
  lives, coins, score, world/level, timer, game-over/level-complete flags) must be sourced
  from a public, verifiable disassembly/RAM map and confirmed empirically against known
  game states before use — not assumed or guessed at design time.

## 3. Core architecture

```
PyBoy (Super Mario Land ROM)
   │  RAM read → feature vector (position, velocity, enemy state, timer, lives, score)
   ▼
Small trainable embedding  ───────────────────┐
   │                                          │ surrogate gradient (same mechanism as
   ▼                                          │ the LM project's upstream embedding)
Frozen spiking reservoir                      │
(LIF, native tensor-train construction —      │
 the same `SpikingReservoir` class from       │
 spiking-reservoir-lm, reused as-is) ──────────┘
   │  spike history (bounded window)
   ▼
Windowed causal-attention readout
(same `AttentionReadout` structure, final projection replaced)
   │
   ├──► Actor head  → action logits (discrete button combinations)
   └──► Critic head → V(s) scalar value estimate

Trajectory-novelty write-gate (reused from the EMG vertical)
   │  novelty score on the reservoir's own state trajectory
   ▼
Intrinsic reward bonus, added to extrinsic reward before the PPO advantage computation
```

Only the embedding, the readout, and the actor/critic heads are trained (via PPO). The
reservoir's parameters remain `torch.Tensor` buffers with `requires_grad=False` — the
same invariant enforced in the LM project, verified the same way: zero
`nn.Parameter`s on the reservoir, the optimizer constructed only over trainable
components, and a runtime tripwire asserting the reservoir's weights are bit-identical to
their initialization at every checkpoint. All three are implemented:
`PolicyValueReservoir.assert_reservoir_frozen()` checks both halves (zero
`nn.Parameter`s, and every frozen buffer bit-identical to the snapshot taken at
construction), and `training.train.save_checkpoint` calls it on the reservoir arm
*before* every write, so a mutated reservoir cannot reach disk and be evaluated later
as if it had been frozen.

## 4. Components

- **Environment wrapper**: a Gymnasium-style wrapper around PyBoy, exposing `reset()` /
  `step(action)` / a fixed-size observation vector built from RAM reads. Episode
  termination on death, level timeout, or level completion. Extrinsic reward: dense
  reward on rightward position delta per step (avoids the sparse-reward problem that
  every published Pokémon-RL project has had to solve by hand), a level-completion bonus,
  and a death penalty.
- **Action space**: a small discrete set of button combinations (e.g. no-op, left, right,
  left+run, right+run, jump, left+jump, right+jump, left+run+jump, right+run+jump) —
  finalized once the actual control feel of Super Mario Land is verified in PyBoy, not
  assumed from the NES Mario convention.
- **Frozen reservoir**: the existing `SpikingReservoir` class, reused without
  modification. Its already-validated incremental `step()` method (bit-exact against a
  full re-run, 86×–106× faster in the LM project's own measurements) is a direct fit for
  a step-by-step RL rollout — no new incremental-state machinery needs to be built.
- **Readout / policy-value heads**: the existing windowed causal-attention
  `AttentionReadout`, with its final 256-way byte-logit projection replaced by two small
  heads (actor: action-space-sized logits; critic: scalar). The windowed-attention
  design's purpose carries over unchanged — a bounded-cost readout independent of episode
  length, avoiding the O(n²) trap the LM project specifically designed around.
- **Trajectory-novelty write-gate → intrinsic reward**: the mechanism designed for
  EMG abnormal-activation detection (`biosignal-reservoir-verticals-design.md` §2.3),
  repurposed here as a curiosity signal: reservoir states unlike recently-seen states
  produce a reward bonus, encouraging exploration without hand-authored reward shaping —
  the same problem every published Pokémon/Mario RL project solves with a bespoke
  exploration bonus, but here obtained as a byproduct of a mechanism already built for
  another domain, at effectively zero extra trained-parameter cost.

## 5. Mandatory scientific control

Same non-negotiable discipline as both prior projects. Baseline: the same PPO setup
(actor and critic heads, same RL algorithm, identical RAM-state observation, identical
curiosity signal) with the frozen reservoir replaced by a small trained recurrent
network (GRU) of matched trainable-parameter count. This isolates whether the frozen
reservoir contributes anything over a conventional, fully-trained RL feature extractor
at the same parameter budget — without this control, any result (positive or negative)
is not attributable to the reservoir specifically. If the reservoir does not beat this
baseline, that is reported as a negative result, not hidden or reframed — consistent
with both prior projects' practice.

### 5.1 What "matched" binds on (amendment, 2026-08-20)

This section originally read "same hidden sizes, same total trainable-parameter budget",
as if both could hold at once. **They cannot, and the binding requirement is matched
trainable-parameter COUNT** (within the 10% tolerance `tests/test_parameter_parity.py`
enforces against the arms `training/train.py:build_model` actually constructs), **not
matched hidden-layer sizes.** Recorded here rather than left as an unremarked gap
between the spec and the code:

- The two arms' readouts sit on top of architecturally different upstream
  representations: a 192-dim trained GRU hidden state on the baseline, an 8192-dim
  frozen reservoir state on the other. The reservoir readout's input projection alone
  therefore costs ~8.2k parameters per unit of its `d_model`, and dominates its budget.
- Fixing both arms' head width at the same number consequently blows the parameter
  budget by ~4.7x (measured: `d_model=64` gives 629,163 trainable parameters against
  the baseline's 132,715). Head hidden sizes end up at `hidden_dim=192` on the GRU arm
  and `d_model=16` on the reservoir arm — 139,179 vs 132,715 parameters, ratio 1.049.
- Parameter count is the correct thing to hold fixed, because it is the quantity the
  control exists to neutralize: the LM project's own load-bearing finding is that a
  frozen reservoir's storable capacity is bounded by the trained readout's parameter
  count, so "same trained capacity, different feature extractor" is precisely the
  comparison being made. Equal head widths at unequal parameter counts would compare
  two differently-sized models and attribute the difference to the reservoir.

The tradeoff is made explicit in code at `models/policy_value_reservoir.py`'s
`__init__` comment (why `d_model=16` and not `ActorCriticReadout`'s own default of 64,
including the measured ratios at 12/16/20) and enforced by
`tests/test_parameter_parity.py`. Retune `d_model`/`n_layers`/`hidden_dim` to stay in
band — never the tolerance.

## 6. Training procedure

- **Algorithm**: PPO (clipped surrogate objective), the standard, well-understood choice
  for continuous rollout-collection RL — chosen over the LM project's CMA-ES precedent
  because CMA-ES's per-candidate full-episode evaluation cost (already a known bottleneck
  even for the LM project's cheap forward passes) would be significantly worse here, where
  each candidate evaluation means playing through part of a Mario level.
- **Rollout collection**: multiple parallel PyBoy instances (multiprocessing, one
  emulator per process) feeding a shared rollout buffer — the concrete lever for using
  the M4's cores, per §2.
- **Checkpointing**: mandatory, same discipline as both prior projects (Colab/Kaggle
  session limits, but even local runs benefit from resumability) — checkpoint the
  trainable components on a fixed step interval, never assume an uninterrupted run.

## 7. Build order (phased, single-variable ablation — not a single big-bang build)

**Naming note:** "Phase 0-3" below are build-order stages *within* Roadmap Phase 1
(§1.1) — a separate, independently-numbered scheme for how this single game's
architecture gets built up mechanism-by-mechanism, not the same numbering as the
multi-game/multi-platform roadmap in §1.1 (whose Phase 1 is this entire document).
Kept as originally numbered (not renamed to avoid disrupting the implementation plan and
SDD ledger, which already reference "Phase 0"/"Phase 1" in this section's sense) — read
"Phase" in §7-§9 as this document's own internal build stages, and "Phase" in §1.1 as the
project-wide roadmap.

1. **Phase 0**: environment wrapper + RAM-state extraction, verified against known game
   states (manual play-through cross-check). No RL yet — pure plumbing correctness.
2. **Phase 1**: PPO baseline (GRU) vs. PPO + frozen reservoir, **both without**
   resonate-and-fire, DLIF, or RSSR — isolates the core claim ("does a frozen reservoir
   help here at all") before adding anything else. Trajectory-novelty-as-curiosity is
   included in this phase for both arms equally (it is not itself the variable under
   test here).
3. **Phase 2**: resonate-and-fire ablation — swap the reservoir's neuron model, holding
   everything else fixed, once Phase 1 shows the reservoir arm beating baseline.
4. **Phase 3 (stretch, explicitly deferred, not started until Phase 1–2 succeed)**: DLIF
   and RSSR, carried over from the biosignal design's own risk classification
   ("completely unproven," "never tested in any reservoir") — not part of the initial
   claim, evaluated only if the core pipeline is already validated.

> **Status note added 2026-08-21, after both Phase 1 comparisons ran.** The precondition
> stated in build-order Phase 2 above — *"once Phase 1 shows the reservoir arm beating
> baseline"* — **is not met.** The reservoir arm lost in `docs/RESULTS.md` v1 (as
> specified) and lost again in v2 (with the optimizer/clipping confound removed and the
> input-calibration defect corrected, both verified removed by the pre-registered A7 and
> A9 measurements). v1 §10 left open the reading that the precondition had not yet been
> *fairly* tested because the losing arm was handicapped; v2 tested that reading and it
> did not hold — the handicaps were removed at zero parameter cost, the arm improved
> substantially in absolute terms, and it still lost by a margin no smaller than before.
>
> **This note records the status; it does not revise the build order.** Whether to stop
> at the literal precondition, or to proceed on some other basis, is the project owner's
> decision and is deliberately left open here — see `docs/RESULTS.md` §22 for the
> evidence that decision should rest on. Nothing in §7 is edited, per this project's
> append-only practice for statements that turn out to be superseded by results.

## 8. Testing

Same discipline as the LM project's 152-test suite: reservoir-frozen invariant asserted
at runtime, RAM-parsing correctness tested against fixed known game states, PPO update
correctness (advantage estimation, ratio clipping, value loss) unit-tested independent of
the emulator, and the incremental-`step()` bit-exactness property re-verified in this
new call pattern (RL rollout, not autoregressive generation) before relying on its
measured speedup.

## 9. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| Frozen reservoir may simply not help vs. a trained-GRU baseline at matched parameter count | High (this is the actual experiment, per §5) | Mandatory control makes a negative result informative, not a silent failure |
| Emulator rollout throughput, not GPU compute, is the real bottleneck | Medium, expected | Parallelize PyBoy instances across CPU cores; measure before assuming Kaggle/Colab helps |
| Super Mario Land RAM addresses not yet verified | Medium, resolved by Phase 0 | Source from a public disassembly/RAM map, cross-check empirically against known states before any RL code depends on them |
| CMA-ES-style per-candidate full-episode cost ruled PPO in over evolutionary readout training | N/A (design decision, not open risk) | Documented here so it isn't re-litigated without new evidence |
| Trajectory-novelty-as-curiosity may need reward-weighting/annealing tuning (standard curiosity-bonus pitfall: agent "addicted" to novelty instead of task reward) | Medium | Standard mitigation: anneal the intrinsic-reward weight over training; monitor extrinsic-only return alongside combined return |
| DLIF/RSSR (Phase 3) may not transfer from biosignal domain at all | High, explicitly deferred | Not attempted until Phase 1–2 succeed, exactly as scoped in §7 |

## 9.1. Platform confirmation addendum (2026-08-20)

During implementation, a GBA pivot was proposed (the project owner had only GBA ROMs on
hand, no Game Boy dump) and investigated as a real feasibility spike rather than assumed
either way. Finding: `pygba`/`mgba` (the GBA-equivalent of PyBoy) has no Apple Silicon
macOS wheel on PyPI; building mGBA from source with Python bindings enabled succeeds
(core library, Qt frontend, and the bindings themselves all compile cleanly after adding
two undocumented build-time dependencies, `cffi` and `cached_property`), but loading any
real ROM through the built bindings crashes with a hard native `SIGBUS` /
`EXC_ARM_DA_ALIGN`, root-caused via the macOS crash reporter to `ffi_call_SYSV` inside
libffi, invoked through cffi's ABI-mode calling trampoline — a documented fragility of
that specific cffi mode on strict-alignment ARM64, external to this project and not
fixable by any packaging change. **Ruling: the platform stays Game Boy + PyBoy as
originally designed** (§1, §2) — Tasks 1-3's PyBoy-based pipeline, already built and
reviewed clean before this investigation, needs zero rework. The project owner supplies
an actual `.gb` ROM (Super Mario Land preferred per §1's rationale, but any reactive
Game Boy platformer is an acceptable substitute) rather than pursuing the GBA-emulation
path further. The full investigation trail (build log locations, exact crash signature,
dependency-fix sequence) is preserved in the implementation plan's SDD ledger
(`.superpowers/sdd/2026-08-19-mario-ppo-reservoir/progress.md`).

That ruling stands for this document — Phase 1 is a Game Boy experiment either way —
but the two escape hatches it named have since diverged, and this paragraph is the
one to read before repeating the claim that GBA is blocked:

- **Bridging over mGBA's own Lua scripting API via a local socket: BUILT AND VERIFIED
  WORKING (2026-08-20), so the `SIGBUS` above no longer blocks a GBA target at all.**
  It sidesteps the crash by construction — the script runs inside mGBA's own process,
  so there is no Python/C FFI boundary for libffi to misalign. 12/12 checks passed,
  deterministic, ~3k fps single-instance / ~15k fps across 8 parallel instances, with
  real player-controlled gameplay driven through Super Mario Advance's World 1-1.
  See §1.1's Phase 3 entry for the full result, including the two mGBA bugs worked
  around. It exists as a proven scratchpad spike and is **not yet a component of this
  repository**; turning it into a real `gymnasium.Env` (with the same empirical
  RAM-confirmation discipline applied to Super Mario Land here) is Phase 3's remaining
  work — formalization work, not a platform bug.
- **Patching mGBA to build cffi in API mode instead of ABI mode: still not attempted.**
  It would be the fix for the direct-Python-bindings path specifically. There is now
  little reason to spend that effort, since the socket bridge already delivers the
  capability, faster and without patching a third-party build system.

## 10. Out of scope for THIS document (not abandoned — see the §1.1 roadmap)

Everything below is deferred to a later roadmap phase (§1.1), not dropped. Restated here
so nobody reading only this section mistakes "not built yet" for "not planned":

- **Pokémon Fire Red / any RPG-genre target** — Roadmap Phase 4 (§1.1). Fire Red is a
  GBA title and this document's PyBoy-based pipeline cannot run GBA ROMs at all, so it
  needs Roadmap Phase 3's GBA support formalized into a real component first (the
  bridge itself works — §9.1 — but it is still a scratchpad spike). The binding
  constraint on this phase is now the other prerequisite: a hierarchical planning layer
  above the frozen reservoir, which has no design yet.
- **Multi-game generalization / continual learning across titles** — Roadmap Phase 2
  (§1.1). A separate, genuinely open research problem (see the brainstorming transcript's
  review of DeepMind's SIMA 1/2 and the continual-RL literature — Unicorn, DisCoRL, CORA);
  not attempted until this document's single-game pipeline produces a real result.
- **GBA as a platform generally** — Roadmap Phase 3 (§1.1). **No longer blocked** as of
  2026-08-20: the native `SIGBUS` in mGBA's Python bindings on Apple Silicon is real and
  unfixed, but a Lua-scripting + local-socket bridge to `mgba-headless` sidesteps it
  entirely and has been verified working end-to-end (§1.1, §9.1). Out of scope here for
  the ordinary reason instead — it is a later roadmap phase, and the bridge is still a
  scratchpad spike rather than a component of this repository.
- DLIF, RSSR (deferred to build-order Phase 3 within *this* document, §7 — a different,
  narrower deferral than the roadmap phases above; see §7's naming note).
- Any cloud GPU rental — zero-budget discipline carried over from the sibling `llm-lab`
  project.

## 11. Open decisions

- Exact discrete action-combination set — finalized during Phase 0 once PyBoy's actual
  input handling for Super Mario Land is confirmed.
- Reservoir size and TT-rank for this task — start from the LM project's validated
  TT@8192 configuration as a default, re-tune only if Phase 1 empirically motivates it.
- Whether rollout parallelism uses raw `multiprocessing` or a lighter-weight vectorized-env
  library — an implementation detail for the planning phase, not a design-level decision.

## References

Carried over from `spiking-reservoir-lm-design.md` and `biosignal-reservoir-verticals-design.md`
(reservoir computing, LIF dynamics, tensor-train construction, resonate-and-fire,
trajectory-novelty, DLIF, RSSR — full citations there, not repeated here). New to this
document:

- Whidden, P. — original Pokémon Red RL project (PPO on PyBoy, reward shaping)
- `pokemonred_puffer` (Rubinstein) — PyBoy-based RL environment precedent
- PPO: Schulman et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347
- Curiosity/intrinsic-reward literature survey (RND, ICM) — prior conversation's web
  research; full citations to be added to the implementation plan's bibliography
