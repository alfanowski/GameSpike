# Spiking Reservoir RL — Design Document

Status: brainstorm complete, ready for implementation planning
Date: 2026-08-19
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
- **Phase 3 (future, blocked on a concrete technical issue, not a preference).** Game Boy
  Advance as a platform. **Current status: blocked.** Investigated directly (2026-08-20,
  §9.1 addendum below): `pygba`/mGBA (the GBA-equivalent of PyBoy) has no Apple Silicon
  macOS wheel, and building it from source produces Python bindings that crash with a
  hard native `SIGBUS` (root-caused to a documented cffi ABI-mode fragility on ARM64) the
  moment a real ROM is loaded — a platform-level bug external to this project, confirmed
  by direct investigation, not assumed from documentation. Two possible unblocking paths
  were identified and neither has been attempted: patching mGBA's own build to use cffi's
  API mode instead of ABI mode, or bridging over mGBA's built-in Lua scripting API via a
  socket instead of direct Python bindings. Phase 3 starts only once one of those is
  proven to work.
- **Phase 4 (the actual named target).** Pokémon Fire Red / Pokémon-style RPGs on GBA,
  once Phase 3 unblocks the platform. This is explicitly *not* abandoned — it is
  sequenced behind a real technical blocker (Phase 3) and a real architectural
  prerequisite genuinely called out from the start: an RPG's strategic/inventory/long-
  horizon demands need a planning layer above the frozen reservoir (a hierarchical
  extension), which has no design yet and is not something this document's architecture
  provides on its own.

This ordering was chosen, and remains, for real technical reasons (an unresolved platform
bug, an unsolved research problem, a missing architectural layer) — not because the
smaller scope was more convenient. Each phase's own document should restate this roadmap
so it's never implicit.

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
their initialization at every checkpoint.

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

Same non-negotiable discipline as both prior projects. Baseline: identical PPO
architecture (actor/critic heads, same hidden sizes, same total trainable-parameter
budget), identical RAM-state observation, but with the frozen reservoir replaced by a
small trained recurrent network (GRU) of matched trainable-parameter count. This isolates
whether the frozen reservoir contributes anything over a conventional, fully-trained RL
feature extractor at the same parameter budget — without this control, any result
(positive or negative) is not attributable to the reservoir specifically. If the
reservoir does not beat this baseline, that is reported as a negative result, not
hidden or reframed — consistent with both prior projects' practice.

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
(`.superpowers/sdd/2026-08-19-mario-ppo-reservoir/progress.md`) for anyone who later
wants to revisit a GBA target with a different binding strategy (e.g. patching mGBA to
build cffi in API mode instead of ABI mode, or bridging over mGBA's own Lua scripting API
via a socket instead of direct Python bindings) — neither attempted here, both real,
nontrivial engineering efforts outside this document's scope.

## 10. Out of scope for THIS document (not abandoned — see the §1.1 roadmap)

Everything below is deferred to a later roadmap phase (§1.1), not dropped. Restated here
so nobody reading only this section mistakes "not built yet" for "not planned":

- **Pokémon Fire Red / any RPG-genre target** — Roadmap Phase 4 (§1.1). Requires both a
  hierarchical planning extension not designed here AND Roadmap Phase 3 (GBA platform
  support) to unblock, since Fire Red is a GBA title and this document's PyBoy-based
  pipeline cannot run GBA ROMs at all (§9.1).
- **Multi-game generalization / continual learning across titles** — Roadmap Phase 2
  (§1.1). A separate, genuinely open research problem (see the brainstorming transcript's
  review of DeepMind's SIMA 1/2 and the continual-RL literature — Unicorn, DisCoRL, CORA);
  not attempted until this document's single-game pipeline produces a real result.
- **GBA as a platform generally** — Roadmap Phase 3 (§1.1). Currently blocked on a
  confirmed native crash in mGBA's Python bindings on Apple Silicon (§9.1), not a
  scheduling choice.
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
