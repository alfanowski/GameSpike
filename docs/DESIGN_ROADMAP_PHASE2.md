# Roadmap Phase 2 — Multi-Game Generalization on the GRU Architecture

Status: **design proposal, nothing implemented, nothing run.** No training job has been
started under this document and none should be until it is reviewed.
Date: 2026-08-21
Author: Andrea Alfano ("Alfanowski"), with research support from Claude (Opus 5)
Scope: `docs/DESIGN.md` §1.1 **Roadmap Phase 2** — multi-game generalization within Game
Boy / Game Boy Color, and the continual-learning / catastrophic-forgetting problem it
opens.

---

## 0. How to read this document

**0.1 The numbering trap, restated because this project has already tripped over it.**
`docs/DESIGN.md` carries *two* independent "Phase" schemes and §7 says so explicitly.
`DESIGN.md` §1.1 is the project-wide **roadmap** (Phase 1 = Super Mario Land reservoir vs.
GRU; Phase 2 = multi-game; Phase 3 = GBA platform; Phase 4 = Pokémon Fire Red).
`DESIGN.md` §7 is a **build order internal to roadmap Phase 1** (its Phase 2 =
resonate-and-fire ablation; its Phase 3 = DLIF/RSSR). **This document is roadmap Phase 2
and has nothing to do with build-order Phase 2.** The filename says
`DESIGN_ROADMAP_PHASE2` rather than `DESIGN_PHASE2` for exactly that reason.

**0.2 What this document is.** A design document in the same register as `docs/DESIGN.md`:
it fixes the question, the controls, the metrics and the scope, and it names what it does
*not* decide. It is deliberately not an implementation plan. If it is accepted, the
implementation plan is a separate document under `docs/superpowers/plans/`, written the
way `2026-08-19-mario-ppo-reservoir.md` was — task by task, tests first.

**0.3 What this document is not.** It is not a result, not a claim, and not a licence to
spend compute. §12 proposes one bounded first step and explicitly leaves the decision to
start it with the project owner, on the same footing Phase 1's implementation had: a
design doc and a plan existed and were signed off before any run started.

**0.4 Open questions are marked.** Anything genuinely undetermined by the existing
documents is tagged **OPEN** and collected in §11. Anything the existing documents already
imply an answer to is decided here and the implication is cited, not re-litigated.

---

## 1. The roadmap, restated — and where Phase 1 actually left it

`DESIGN.md` §1.1 instructs that *"each phase's own document should restate this roadmap so
it's never implicit."* Restating it, unchanged in substance:

> The actual goal of this project is a general game-playing agent across Nintendo handheld
> titles — explicitly including Game Boy Advance and **Pokémon Fire Red** — not a
> Mario-Land-only system.

- **Roadmap Phase 1 — COMPLETE, and negative.** Super Mario Land, frozen spiking reservoir
  vs. matched-parameter trained GRU. Ran twice: as specified (v1) and corrected (v2). The
  reservoir lost both times, in v2 by 7.26–8.97 points of mean episode return
  (`docs/RESULTS.md` §13, §15).
- **Roadmap Phase 2 — this document.** Multi-game generalization within Game Boy / Game Boy
  Color.
- **Roadmap Phase 3 — GBA platform.** Unblocked 2026-08-20 by the verified Lua + local-socket
  bridge to `mgba-headless`; still a scratchpad spike, not a component of this repository
  (`DESIGN.md` §9.1).
- **Roadmap Phase 4 — Pokémon Fire Red / RPG targets.** Needs Phase 3 *plus* a hierarchical
  planning layer that has no design yet.

### 1.1 What Phase 1 settled, and what it did not

Phase 1 asked one question — *does a frozen reservoir help at all, on real game control, at
a matched trainable-parameter budget?* — and answered it: **no**, under the fairest test the
project was able to construct, with both diagnosed handicaps removed and both removals
verified by pre-registered measurement (`RESULTS.md` §19, `EXPERIMENT_LOG.md` §21.2).

`RESULTS.md` §22 states the consequence precisely and does not overstate it: *"It is not
evidence about the multi-game goal in `DESIGN.md` §1.1. Roadmap Phase 2 … remains a
separate question this experiment bears on only by supplying the premise Phase 1 was meant
to establish."*

That is the load-bearing sentence for this document. **The frozen-reservoir bet was Phase
1's experimental question, not a premise of the roadmap.** Everything Phase 1 built that is
*not* the reservoir is architecture-agnostic and carries forward untouched:

| Component | File(s) | Reused by Phase 2 |
|---|---|---|
| PyBoy env wrapper, Gymnasium 5-tuple API | `envs/mario_land_env.py` | yes, as the pattern and as task A itself |
| Empirically-confirmed Super Mario Land RAM map | `envs/ram_map.py` | yes, unchanged |
| Blind RAM-discovery tool | `envs/ram_scan_tool.py` | yes — this is how a second game's map gets built |
| PPO core (GAE, clipped surrogate, value loss, entropy) | `training/ppo.py` | yes, unchanged — zero game references |
| Rollout collection with recurrent state | `training/rollout.py` | yes, extended (§7.4) |
| Trajectory-novelty curiosity gate | `training/novelty_gate.py` | yes, but **one instance per task** (§7.5) |
| Evaluation harness + its stated blind spots | `training/evaluate.py` | yes, extended with a task axis |
| Seed-level statistics: exact permutation tests, bootstrap CIs | `analysis/aggregate_results.py` | yes, unchanged |
| Matrix drivers with resume + completeness guards | `scripts/run_*_matrix.py` | yes, extended with a task axis |

### 1.2 The architecture Phase 2 builds on, and why

**`models/policy_value_gru.py`** — 132,715 trainable parameters at
`obs_dim=12, embed_dim=32, hidden_dim=192, n_actions=10`; embedding → `tanh` → single-layer
GRU → linear actor and critic heads. It is the arm that won Phase 1 twice, and a read of
the file confirms it contains **no game-specific code at all**: `obs_dim` and `n_actions`
are constructor arguments and the module imports nothing from `envs/`.

Choosing it is not a demotion of the project's ambition. It is the same discipline Phase 1
used in the other direction: build on what has been measured to work, and put the
unmeasured thing in a controlled experiment rather than in the foundation. The reservoir
question is not closed by fiat — a separate, bounded pilot on
`feat/resonate-and-fire-pilot` is testing whether a reservoir *variant* closes Phase 1's
gap. **Phase 2 is deliberately independent of that pilot's outcome.** If the pilot wins,
its architecture drops into the slot `PolicyValueGRU` occupies here, because everything
below is specified against the *interface* (an `obs_dim`-in, `n_actions`-out recurrent
policy-value module), not against the GRU's internals.

---

## 2. The question Phase 2 asks, stated so it can fail

Phase 1's question was falsifiable and it was falsified. Phase 2's must be the same shape.

### 2.1 Primary question

> **Can one recurrent policy, at the trainable-parameter budget Phase 1 established, learn
> and retain more than one Game Boy task — and what does the order of training cost it?**

Decomposed into questions that can each be answered "no":

- **Q1 — capacity.** Trained on tasks {A, B} *interleaved*, does one policy reach a
  normalized score comparable to two independent single-task specialists? A failure here is
  a **capacity** result: 132,715 parameters do not hold two tasks, regardless of ordering.
- **Q2 — forgetting.** Trained *sequentially* A → B, how much of task A's performance
  survives? This is the catastrophic-forgetting measurement and it is the headline.
- **Q3 — conditioning.** Does telling the policy which task it is on (a one-hot task ID
  appended to the observation, §6.2) change the answer to Q1 or Q2?
- **Q4 — mitigation, conditional on Q2.** If sequential training does destroy task A, does
  the cheapest method the literature reports as *actually* effective recover it, and at what
  cost in steps and storage? The literature is unusually consistent on which method that is:
  **rehearsal**. CLEAR (Rolnick et al. 2019) — a replay buffer plus two behavioural-cloning
  auxiliary losses, **no extra network parameters and no need for task boundaries or task
  identity** — is reported to nearly eliminate forgetting and to equal or beat EWC and
  Progress & Compress; CORA (Powers et al. 2022) independently reproduced this, finding CLEAR
  "reliably outperforming every other method" among its baselines. Meanwhile EWC's own
  numbers in RL are poor in a specific, informative way: on Continual World's CW20 it reaches
  near-zero forgetting (0.02) but at **−0.17 forward transfer** — it stops the network
  forgetting by stopping it learning. **So Q4's first arm is rehearsal, not EWC**, and that
  ordering is a literature finding rather than a preference. **Not run unless Q2 shows real
  forgetting**, on the same "don't build on top of an unmeasured premise" rule that sequenced
  Phase 1 first.

### 2.2 The mandatory controls, transposed

`DESIGN.md` §5's discipline is non-negotiable and carries over. In Phase 1 the control was
a matched-parameter trained GRU. In Phase 2 there are four, and none is optional:

- **C1 — single-task specialists.** One policy per task, trained on that task alone, same
  architecture, same hyperparameters, same per-task step budget. This is both the upper
  reference and the denominator of the normalized score (§8.1). Without it, "the multi-task
  agent scored 41" means nothing.
- **C2 — untrained policies (`--steps 0`).** Already supported (`checkpoints_v2_init` in
  `RESULTS.md` §23) and already the pattern Phase 1 used for its `init` selection. This is
  the *lower* reference of the normalized score. Phase 1 used it to show the untrained arms
  were statistically indistinguishable; Phase 2 needs it as a scale anchor.
- **C3 — matched total experience.** A sequential A→B run sees 2 × *N* env steps. A
  specialist must therefore be reported at *N* (matched **per-task** experience) **and** a
  single-task run at 2*N* must exist (matched **total** experience), so "the continual agent
  is worse" is never confounded with "it got fewer steps on this task" or "more steps would
  have helped anyway".
- **C4 — matched trainable-parameter count across conditions.** Phase 1's parity discipline
  (`tests/test_parameter_parity.py`, ±10%) applied *between arms*. Phase 2 has one arm, so
  parity now binds *between conditions*: specialist, interleaved and sequential policies
  must have identical parameter counts, and any condition that deliberately differs (the
  task-conditioning ablation adds 64 parameters, §6.2) must state the delta rather than let
  it sit unremarked. A multi-task agent that quietly gets per-task heads is not answering
  Q1.

### 2.3 What a negative result looks like, pre-committed

Written before any number exists, in keeping with `EXPERIMENT_LOG.md` §17.9's rule that the
shape of the write-up is fixed before the numbers so the shape cannot be chosen to flatter
them. All three of these are **publishable outcomes of this phase**, not failures of it:

- **"One policy at this budget does not hold two tasks."** If the interleaved condition
  cannot approach specialist normalized score on both tasks simultaneously, that is a
  capacity finding, and it is directly relevant to the roadmap: it bounds what Phase 3/4
  can assume about a single shared trunk.
- **"Sequential training destroys the earlier task essentially completely."** The expected
  outcome by default — the continual-RL literature's baseline finding — but this project
  measures rather than assumes, and the *magnitude* on a Game Boy testbed at this parameter
  scale is not something anyone has published.
- **"Task conditioning makes no difference."** A disconfirmed hypothesis is a result; §7.4
  of `RESULTS.md` and `EXPERIMENT_LOG.md` §12 already contain four of them.

---

## 3. The binding constraint found while writing this document

**There is exactly one Game Boy ROM on this machine:
`/Users/alfanowski/Desktop/Super Mario Land (World).gb`.** No second `.gb`/`.gbc` file
exists on disk. (`~/Desktop/gba/` holds fourteen Game Boy *Advance* ROMs, which belong to
roadmap Phase 3 and are not usable by PyBoy — `DESIGN.md` §9.1.)

This is a hard, external blocker on the cross-*title* experiment and it is recorded here
rather than discovered halfway through an implementation plan. Two consequences:

1. **Roadmap Phase 2 proper cannot start until the project owner supplies a second
   legally-dumped Game Boy or Game Boy Color ROM.** The recommendation for which one is in
   §5.3; it is a decision only he can execute.
2. **Waiting is not the only option.** Everything Phase 2 needs that is *not* a second ROM —
   the task axis through the code, the shared observation schema, the normalized score, the
   performance matrix, the forgetting metric, the interleaved and sequential drivers, the
   per-task novelty gates — can be built and validated on a testbed that requires no new
   ROM at all. §4 is that testbed.

This is the same sequencing logic that produced Phase 1: prove the machinery on the cheap
case before spending the expensive one.

---

## 4. Phase 2a — cross-level continual learning inside Super Mario Land

### 4.1 The proposal

Treat **individual Super Mario Land levels as tasks**. Super Mario Land ships **twelve
levels across four worlds** (three per world, with a boss at the end of each third), and two
of them — **World 2-3 (the Marine Pop submarine) and World 4-3 (the Sky Pop biplane)** — are
not platformer levels at all: they are **forced-autoscroll shoot-'em-up stages** where Mario
pilots a vehicle, moves freely in two dimensions, and fires projectiles with B (torpedoes
and missiles respectively; 4-3 adds a maze section and the final boss).

One cartridge therefore contains both a *near*-transfer axis (1-1 → 2-1: same genre,
different layout, different enemy roster, water physics in world 2) and a genuine
*far*-transfer axis (1-1 → 2-3: different genre, different control semantics, same ROM,
same RAM map).

**Honest calibration of how big the near-transfer shift is.** No Super Mario Land–specific
generalization study exists. The closest published evidence is on NES Super Mario Bros.,
where within-game level transfer is reported as imperfect (Burda et al., "Large-Scale Study
of Curiosity-Driven Learning", arXiv:1808.04355). SML's per-world enemy rosters do differ,
which is confirmed; but physics, camera model and HUD stay constant across the platformer
levels, so cross-level shift is plausibly a **weaker** distribution shift than cross-title
shift. That is a reason to treat Phase 2a as a machinery validation and a *lower bound* on
the difficulty, not as a substitute for Phase 2b.

### 4.1.1 The autoscroll problem — a reward-design finding, not a detail

In the two vehicle stages **the camera scrolls on a fixed schedule, independent of what the
player does.** `read_level_progress()` is built from the camera's own position
(`ADDR_LEVEL_BLOCK × 16 + ADDR_MARIO_X`, `envs/ram_map.py:178-199`), and
`PROGRESS_REWARD_PER_PIXEL` pays the agent for increases in it.

**In an autoscroller that pays the agent for the passage of time.** An agent that presses
nothing collects nearly the full progress reward of an agent that plays well; the reward
becomes almost uncorrelated with skill. Phase 1's reward function, applied unmodified to
2-3 or 4-3, would produce a training signal that looks healthy and measures nothing — the
same failure class as a wrong RAM address, which `envs/ram_map.py:63-66` warns about
precisely because it "fails silently".

**Consequence, and it is a real one for §4.2's claim:** the far-transfer axis inside Super
Mario Land is *cheap*, but it is **not free**. It needs its own reward definition — survival
time plus score delta from destroying enemies, with progress removed or heavily discounted —
and that reward has to be designed and justified, not inherited. Any implementation plan
must treat "define and validate the vehicle-stage reward" as its own task with its own
acceptance test, not as a constant to tweak.

This also strengthens the case for **task-specific reward normalisation (§6.4)**: two tasks
whose reward functions have different *forms*, not just different scales, are exactly the
situation that normalisation exists for.

### 4.2 Why this is worth doing rather than a consolation prize

- **It costs no new ROM and no new game's RAM map.** `envs/ram_map.py` is game-wide, not
  level-wide: lives, score, timer, camera block and world/level byte are the same addresses
  in every level. That removes the single most expensive item on the refactor list (§9).
  It is **not** zero-cost: §4.1.1's autoscroll reward problem and §4.4's four unconfirmed
  semantics are real work, and the honest accounting is "one afternoon of empirical
  verification plus one new reward definition", not "free".
- **It isolates one variable.** Cross-title transfer changes the game, the RAM map, the
  reward function, the boot sequence and the observation semantics all at once. Cross-level
  transfer changes only the task distribution. Phase 1's entire methodology was built on
  single-variable ablation; this is the same move.
- **Every artefact it produces is exactly what Phase 2b needs.** The metric definitions, the
  performance matrix, the task axis in `train.py`/`evaluate.py`/the matrix drivers, the
  per-task novelty gates, the reward normalization — all of it transfers verbatim to
  cross-title. Nothing is thrown away.
- **It is honest about what it is.** Cross-level generalization is **not** roadmap Phase 2.
  `DESIGN.md` §1.1 says "across more than one title", and one ROM is one title. Phase 2a is
  a **precursor and a machinery validation**, and this document will not let it be reported
  as the multi-game result.

### 4.3 Starting at an arbitrary level — the mechanism exists and is better than expected

`envs/boot.py` boots from power-on, skips the title screen and always begins at 1-1. Three
mechanisms were investigated; the ranking is not what it looked like at the start.

1. **PyBoy's own Super Mario Land wrapper already does it.**
   `game_wrapper_super_mario_land.start_game(world_level=(W, L))` starts the game at an
   arbitrary world and level. It does so by **patching ROM bank 0 at offsets 0x450–0x461**
   (the Continue handler) — a real ROM patch applied at load time, not a live RAM poke, which
   is why it works where poking does not. This is the preferred mechanism and it costs this
   project nothing to adopt.
   **Caveat that shapes the design:** `reset_game()` replays only the state captured by the
   most recent `start_game()`, so switching target levels requires re-instantiating PyBoy.
   The workable flow is therefore: `start_game(world_level=…)` **once** per task, let the
   level intro finish, `save_state()` into an in-memory buffer, and `load_state()` that
   buffer on every `reset()`.
2. **PyBoy in-memory save states** — `pyboy.save_state(file_like)` / `load_state(...)`,
   confirmed present in the installed PyBoy (`pyboy/pyboy.py:957,993`). Not an alternative to
   (1) but its necessary companion, per the caveat above.
3. **Poking the world/level byte at `0xFFB4`** — **rejected.** `envs/ram_map.py`'s own
   confirmation note records that poking it changed what the HUD *drew*, which is evidence
   about the status bar, not about level loading. That PyBoy's maintainers reached for a ROM
   patch instead is corroborating evidence that the poke does not warp the game.

**What is verified and what is not, stated separately.** That `start_game(world_level=…)`
exists and patches those offsets is verified from PyBoy's own source. That
`world_level=(2,3)` and `(4,3)` **actually load the vehicle stages correctly** is *not*
verified by anyone, and is a required empirical check before the testbed depends on it.
Likewise: no open PyBoy determinism issue was found for Super Mario Land (the cartridge has
no RTC hardware, and the known input-carryover bug is closed with a documented workaround),
but **nobody has published a bit-identical-replay claim for save-state loading either.**
Phase 1's entire statistical design rests on the environment being deterministic
(`RESULTS.md` §9), so this project must run its own load / replay / hash check rather than
inherit an assumption — the same verify-don't-assume rule `envs/ram_map.py:63-66` states for
addresses.

### 4.4 What is not yet known about the vehicle levels, and is therefore a task, not a fact

The public documentation is thinner here than for Phase 1's addresses, and that is worth
saying plainly rather than discovering later:

- **`0xC0AB` (`ADDR_LEVEL_BLOCK`) is not documented in any public RAM map.** It exists in
  PyBoy's own reverse-engineered wrapper and in this project's independent empirical
  confirmation — and nowhere else. Its "monotonic camera counter" semantics almost certainly
  assume the normal camera-follows-player model, which **does not hold in the two vehicle
  stages** (§4.1.1). This is a real risk to `read_level_progress()`, not a formality.
- **`0xC202` (`ADDR_MARIO_X`) is hedged with a question mark even on DataCrystal.** This
  project's own by-hand confirmation is the stronger source; whether the byte holds the
  *vehicle's* X in 2-3 and 4-3 is unconfirmed.
- **The level timer (`0xDA00`–`0xDA02`) is level-agnostic** per DataCrystal, but whether it
  runs during the vehicle stages is unconfirmed.
- **The `on_ground` heuristic** (`ON_GROUND_STILL_FRAMES = 8`, itself documented as an
  inference rather than a confirmed ground-contact bit) almost certainly means nothing in a
  free-flight stage. The honest handling is to keep the slot and let its distribution shift —
  a task-agnostic agent has to cope with exactly that, and hiding the shift would be
  designing the difficulty away.

**Every item above is an empirical-verification checklist scoped to 2-3 and 4-3**, to be
worked with `envs/ram_scan_tool.py` under the same discipline that produced
`envs/ram_map.py`. It is the substance of §12's recommended first task.

---

## 5. Phase 2b — cross-title, the roadmap's actual Phase 2

### 5.1 What changes relative to Phase 2a

Everything in §4 stays; a second game adds, per title: an empirically-confirmed RAM map
(`envs/ram_map_<game>.py`), a boot-to-gameplay routine with its own measured frame counts
(`envs/boot_<game>.py`), an env module with its own reward and termination logic
(`envs/<game>_env.py`, realistically 250–300 lines), and a freshly measured `OBS_MEAN` for
that game's observation distribution. The coupling audit is unambiguous that this is the
expensive part and that no amount of software architecture makes it cheap — it is
reverse-engineering work.

### 5.2 One lever that materially reduces it

PyBoy ships **built-in game wrappers** for a small set of titles, auto-selected by cartridge
title. Verified directly in the installed package
(`.venv/lib/python3.12/site-packages/pyboy/plugins/`): **Super Mario Land, Kirby's Dream
Land, Tetris, Pokémon Gen 1, Pokémon Pinball**, plus two homebrew titles. These are not a
substitute for this project's own empirical confirmation discipline, but they are a
second, independent source to cross-check against — and the Super Mario Land wrapper reads
`0xDA15` (lives), `0xFFB4` (world/level), `0xC0AB` (level block) and `0xC202` (Mario X),
i.e. **it independently agrees with the addresses this project confirmed by hand**. That is
a free validation of `envs/ram_map.py` and a reason to trust the wrappers for other titles
as a *starting hypothesis* to confirm, rather than as gospel.

Concretely, `game_wrapper_kirby_dream_land.py` already exposes score (`0xD070`–`0xD073`),
health (`0xD086`), lives (`0xD089`), a game-over rule, and a menu-skipping `start_game()` —
which is most of a boot routine and most of a RAM map, handed over.

### 5.3 Which second title — a genuinely contested call, flagged as OPEN

Two research passes reached **opposite recommendations**, which is itself the useful signal:
this is not determined by the existing documents and should not be silently decided here. It
is **OPEN-2** in §11, and it is the one open item that costs the project owner something
real to execute.

**The case for Tetris (maximize genre distance).** Super Mario Land's own twelve levels
already supply a near-transfer axis (different worlds, different enemy rosters) *and*, via
the vehicle stages, a genre shift — so the marginal value of spending the one scarce ROM on
another platformer is low. Tetris maximises the distance instead: a different genre, no
avatar, no spatial progression, and a built-in PyBoy wrapper that already solves the
irritating part (reliable game-over detection, plus score/level/lines). Lowest integration
risk of any candidate.

**The case for Kirby's Dream Land (graded difficulty; near before far), which is what this
document recommends.** Four reasons, in decreasing order of weight:

1. **Roadmap Phase 2's question is cross-*title* transfer, and the graded path is near
   before far.** If one policy cannot hold two structurally *similar* titles, the
   cross-genre result is moot — and Phase 1's entire methodology is "prove the easy case
   before spending the hard one". Spending the scarce ROM on the hardest possible pairing
   risks a negative result that teaches nothing about *why*.
2. **Tetris would break §6.1's shared observation schema, which is a load-bearing design
   decision.** Six or more of the twelve slots (progress delta, player Y, both velocities,
   on-ground, timer) have no Tetris meaning and would be zero-filled, while the slots Tetris
   actually needs (board height, hole count, current and next piece) have no Mario meaning.
   The schema survives — that is what a union schema is *for* — but the resulting design
   question ("is a mostly-disjoint union schema still one shared representation?") is a
   second research question layered on top of the first. Kirby fills nearly every existing
   slot with a genuine analogue.
3. **Kirby has an external reference implementation and Tetris does not.**
   `lixado/PyBoy-RL` trained DDQN agents on Kirby *and* Mario, on this exact emulator — the
   same kind of external check `pokemonred_puffer` provided for Phase 1. No confirmed
   trained RL agent on Game Boy Tetris via PyBoy was found.
4. **The far-transfer axis is the one already covered in-ROM.** §4.1.1 is candid that the
   vehicle stages are cheap rather than free — but they are still far cheaper than a
   cartridge. The scarce resource should buy the axis that is *not* already partially
   covered, and cross-title *near* transfer is exactly that.

**If the project owner disagrees, the disagreement is legitimate** and the tradeoff above is
the whole of it: Tetris buys maximum scientific distance at the cost of a harder, more
confounded first measurement; Kirby buys an interpretable first measurement at the cost of a
smaller claim. Either way, the eventual full programme is both.

The evidence behind the Kirby recommendation follows; it is the only candidate with all
three legs at once:

- a public RAM map (DataCrystal) documenting X/Y (`0xD05C`/`0xD05D`), health, lives, score
  and scroll-X (`0xD051`) — so level progress composes the same way Mario's does, camera
  plus local X;
- a first-class built-in PyBoy wrapper (above);
- a **published prior RL implementation on this exact game and emulator** (`lixado/PyBoy-RL`,
  DDQN on both Kirby and Mario) to sanity-check against — the same kind of external
  reference `pokemonred_puffer` provided for Phase 1.

It is also structurally close enough to Mario to make "near transfer" mean something
(side-scrolling platformer, D-pad + A/B, rightward-biased progression) while differing in
ways that are real rather than cosmetic (Kirby floats and inhales; there is no timer; health
is a bar, not a binary powerup state; the wrapper's game-over rule is health *and* lives
exhausted, not a lives counter alone).

**Explicitly not recommended: Pokémon Red/Blue.** Best-documented Game Boy game in existence
(`pret/pokered`, byte-identical rebuild) and the evidence *reinforces* `DESIGN.md` §1's
rejection rather than undermining it: published PyBoy Pokémon RL work needs dozens of
hand-engineered reward terms and an auxiliary spatial-memory observation channel to be
tractable. That is a different architecture, not a different game.

---

## 6. Architecture: what changes in the policy, and what deliberately does not

### 6.1 Observation — one shared semantic schema, not per-task input adapters

**Decision: keep a single fixed-width observation vector shared by every task, whose slots
are defined semantically rather than per-game.** Each task fills the slots it has and
zero-fills the ones it does not — which is *already* the pattern in the codebase: slots
9–11 of `MarioLandEnv`'s 12-dim vector are documented reserved zeros, "zero-filled rather
than omitted so the observation's shape never changes under a downstream model"
(`envs/mario_land_env.py:147-149`).

The existing twelve slots generalise better than they look, because eight of them are
already about a *player in a scrolling game* rather than about Mario:

| slot | Phase 1 meaning | cross-task reading |
|---|---|---|
| 0 | in-level horizontal displacement this step | progress delta — every scrolling task has one |
| 1 | player screen Y | player screen Y |
| 2, 3 | screen-relative X/Y velocity | same |
| 4 | on-ground flag | contact/grounded state where it exists, else 0 |
| 5 | level timer / 400 | bounded time pressure where it exists, else 0 |
| 6 | lives / 9 | lives |
| 7 | powerup state / 4 | player condition (Kirby: health; vehicle stages: probably nothing) |
| 8 | score gained this step / 500 | score delta |
| 9–11 | reserved zeros | reserved / task ID (§6.2) |

**Rejected alternative: per-task input adapters** (a separate `nn.Linear` per game into a
shared trunk). It is what a lot of multi-task work does, and it is rejected here because it
puts per-task trainable parameters into the model, which makes Q1 unanswerable — a policy
with per-task adapters that "holds two tasks" has not demonstrated shared capacity, it has
demonstrated that two adapters fit in memory.

**One real consequence, flagged rather than buried.** `OBS_MEAN`
(`envs/mario_land_env.py:60-63`) is an empirically measured per-slot mean of *Mario's*
observation distribution, used by `embed_init_mode="centered"` — the correction that Phase 1
v2 shipped. Under a multi-task distribution the correct centring constant is the **mixture**
mean over the tasks in the study, and it must be re-measured, not reused. A per-task
centring constant would be per-task parameters by the back door and is rejected for the same
reason as adapters. This is a **pre-registerable measurement**, not a judgement call.

### 6.2 Task conditioning — the cheapest possible version, as an ablation

**Decision: append a *K*-dimensional one-hot task ID to the observation, making
`obs_dim = 12 + K`, and run *both* the conditioned and unconditioned variants as a
pre-registered ablation.**

Cost at K=2: `embedding` grows from 12×32+32 = 416 to 14×32+32 = 480 parameters. **+64
trainable parameters, +0.048%** against a 132,715-parameter budget — three orders of
magnitude inside `tests/test_parameter_parity.py`'s ±10% band, and small enough that the
comparison is not confounded by capacity.

Why an ablation rather than a decision, and why the *unconditioned* variant is the more
interesting one: the unconditioned setting is **task-agnostic continual RL**, formalised by
Caccia et al. (2022) as a POMDP in which task identity is a hidden latent the agent must
infer from *a stretch of trajectory rather than a single state* — which is exactly what a
recurrent policy is equipped to do, and this project's policy is recurrent. That paper's
`3RL` (replay + a recurrent network, **no explicit task ID**) matched or exceeded a
multi-task oracle, and its task-*aware* baselines **degraded** as the task count grew. So
"telling the policy which game it is on helps" is a genuinely falsifiable hypothesis here,
not a foregone conclusion — which is what makes Q3 worth a matrix slot. The conditioned
variant remains the standard multi-task setting and the fairer test of Q1 specifically.
Running one without the other answers half a question.

**Rejected alternatives**, with reasons: per-task actor heads (parameter isolation — makes
Q1 vacuous, and is really a *method*, so it belongs in Q4's mitigation family, not in the
baseline architecture); FiLM/hypernetwork conditioning (more machinery than a 64-parameter
one-hot buys at this scale, and adds an uncontrolled variable); learned task embeddings (a
learned embedding of a 2-element set *is* a one-hot with extra steps).

### 6.3 Action space — a shared union, and the honest cost of enlarging it

Phase 1's action set is ten combinations of `left/right/A/B` plus no-op
(`envs/mario_land_env.py:160-175`). It has no `up` and no `down`, because Super Mario Land's
platformer levels do not need them.

**The Marine Pop and Sky Pop stages do** — the vehicle moves freely in two dimensions.
Any Phase 2a testbed that includes them, and almost any cross-title testbed, therefore needs
a larger action set.

**Decision: define one shared union action space for Phase 2, a strict superset of Phase 1's
ten, applied identically to every task, unmasked.** Unmasked because Game Boy hardware
already gives the semantics for free — a game simply ignores a button it does not use — so
an action that is meaningless in task A is learnable-to-avoid rather than illegal. This
avoids building action-masking machinery (and the entropy/log-prob bookkeeping that comes
with it) for a benefit that at this scale is speculative.

**This is also the literature-consistent default, not a shortcut.** Multi-Game Decision
Transformer (Lee et al. 2022) plays 41 Atari games through a single **18-action union** —
the full joystick+button set, a strict superset of what any one game needs — rather than
per-game heads or masks; Gato (Reed et al. 2022) goes further and flattens every modality
into one token vocabulary. Per-game output heads are the standard *alternative* (Continual
World uses them), and they are rejected here for the same reason per-task input adapters
are: they are parameter isolation, which makes Q1 vacuous.

**The exact membership of that set is OPEN** and should be fixed the way `DESIGN.md` §11
fixed Phase 1's: *"finalized during Phase 0 once PyBoy's actual input handling … is
confirmed"* — i.e. by driving each task in PyBoy and observing which inputs do anything,
not by assuming. A reasonable starting proposal is the cross product
`{noop, left, right, up, down} × {—, A, B, A+B}` = 20 actions, pruned to what is
behaviourally distinguishable.

**The cost, stated up front.** Going from 10 to *N* actions changes `actor_head` from
192×10+10 = 1,930 parameters to 192×*N*+*N*, and it makes Phase 2's Super Mario Land numbers
**not comparable to Phase 1's published returns**. That is acceptable only because Phase 2
must re-run its own Mario specialist as the C1 reference anyway (§2.2) — but it must be said
out loud in the results, not discovered by a reader comparing two tables.

### 6.4 Reward scale across tasks

Two tasks with different reward scales fed to one PPO loss means the larger-scaled task
dominates the gradient, and the resulting "the agent prefers task B" is an artefact of
arithmetic, not of learning. The literature's standard answer is PopArt (Hessel et al.,
2019, "Multi-task Deep Reinforcement Learning with PopArt"), designed for exactly this on
DmLab-30 and Atari-57.

**Decision for the first pass: a fixed per-task reward scale constant, measured from the
untrained-policy return distribution and pre-registered before any training run.** Simple,
auditable, adds no adaptive machinery, and cannot be tuned after seeing results because it
is fixed before them. **PopArt is the named fallback**, not the default, because adaptive
return normalisation interacting with a 193-parameter critic is a second uncontrolled
variable, and Phase 1's whole lesson (`RESULTS.md` §21) is what happens when two treatments
change at once.

The scoreboard is reported in **both** native per-task units and normalized units (§8.1),
so the normalisation can never hide a result.

### 6.5 Critic

**Decision: a single shared critic head in the headline conditions**, with a **per-task
critic head as a pre-registered ablation**. The value function is genuinely task-specific
(it predicts returns whose scale and shape differ per task) while the *policy* is the thing
whose sharing is under test, so a per-task critic is a defensible concession — but it is
also parameter isolation, and unremarked parameter isolation is exactly what §2.2's C4
exists to prevent. Cost of a per-task critic at K=2: +193 parameters. Report both.

---

## 7. Training regimens

### 7.1 The conditions

Let *N* be the per-task step budget. **`N` = 1,000,064**, matching Phase 1 exactly — which
is not just internal continuity: **Continual World's CW10/CW20 protocol uses 1M env-steps
per task** and its authors describe that as a deliberately modest budget for modern deep RL,
CORA's CHORES suite uses ~1M-frame budgets, and the smallest per-task budget found anywhere
in the peer-reviewed continual-RL literature is Caccia et al.'s 500K "resource-constrained"
variant. **This project's existing budget sits at the standard low end of what the field
publishes, not below it** — which is a genuinely useful thing to know before committing
compute.

| condition | what it does | budget | answers |
|---|---|---|---|
| **SPEC-A**, **SPEC-B** | one policy per task, that task only | *N* each | C1, denominator of §8.1 |
| **SPEC-2N** | one policy, one task, double budget | 2*N* | C3 — separates "continual is worse" from "more steps helps" |
| **INT** | one policy, tasks alternating at rollout boundaries | 2*N* (*N* per task) | Q1 |
| **SEQ** | one policy, all of A then all of B | 2*N* (*N* per task) | Q2 — the headline |
| **INIT** | untrained (`--steps 0`) | 0 | C2, lower anchor |

Plus the Q3 ablation (each of INT and SEQ, with and without the one-hot task ID) and the
conditional Q4 mitigation arm.

### 7.2 Switching granularity for INT

**Decision: alternate tasks at rollout boundaries** — i.e. one full 128-step rollout, hence
one PPO update, comes from a single task, and the next comes from the other.

Rationale: 128 steps is already where recurrent state resets and where the PPO update
boundary sits (`training/rollout.py`, `training/train.py`'s `replay_rollout`), so switching
there requires no new state-management semantics and keeps the gradient replay bit-exact —
a property Phase 1 relied on and verified.

**Rejected: per-episode switching** (episode lengths differ across tasks, so task exposure
would be unequal and would drift with policy quality — the exposure ratio would become a
function of how well the agent plays, which is a confound). **Rejected: mixed minibatches
within one PPO update** (the rollout loop is single-env by construction; batching across
envs is a larger change than this phase needs).

**Disclosed limitation:** alternating updates is not the same as a mixed batch. IMPALA-style
multi-task learners mix tasks *within* a gradient step; this alternates between them. That is
a real difference from the literature's usual setup and belongs in the limitations section
of any write-up, not in a footnote.

### 7.3 Optimizer state across the task switch in SEQ

**Decision: carry Adam state across the switch (do not reset it), and pre-register that
choice.** It is the setting a naive continual learner would be in, which is what Q2 is
asking about. A reset-optimizer variant is a legitimate follow-up ablation; running both by
default doubles the matrix for a question nobody has asked yet.

### 7.4 Rollout collection with more than one live environment

`collect_rollout_with_model(env, obs, model, model_state_fns, n_steps, novelty_gate,
novelty_coef)` already takes the env, the current observation and the novelty gate **as
parameters**, and returns `final_obs` — the caller owns the lifecycle
(`training/rollout.py:83-95`). Multi-task collection is therefore a change to the *caller*,
not to the collector: the training loop holds a list of per-task
`(env, obs, novelty_gate)` tuples and selects one per rollout. That is a genuinely small
change and it is the single most encouraging finding of the coupling audit.

### 7.5 The novelty gate must be per task — this is a bug waiting to happen

`train.py` constructs **one** `NoveltyGate(dim=OBS_DIM, capacity=512, k=8)` for the entire
run, and `training/rollout.py:166-168` scores novelty on the **observation vector itself**.

With two tasks sharing one 512-entry buffer, every task switch produces a burst of
artificially high novelty — the agent would be paid an intrinsic reward simply for *changing
game*, which is a reward for something it does not control and cannot learn from. Left
unfixed it would silently corrupt the INT condition and could plausibly produce a
spectacular-looking, entirely spurious result.

**Decision: one `NoveltyGate` instance per task, never shared.** The gate has no trained
parameters, so this costs nothing but bookkeeping. Recorded here as a *design decision with
a stated failure mode*, so that if a future reader finds a shared gate in the code they know
it is a regression and not a choice.

---

## 8. Metrics, pre-registered

Phase 1's scoreboard was one number, `mean_extrinsic_return`, declared in advance and never
changed. Phase 2 needs more than one number, so the *set* is declared in advance instead,
and the headline is named so that a later reader cannot be told a different number was the
headline all along.

### 8.1 The cross-task currency: a specialist-normalized score

Raw episode return is not comparable across tasks. Mario's return is dense progress reward
plus a 50-point completion bonus and a −10 death penalty; another level, and certainly
another title, has a different scale entirely. Reporting "the continual agent scored 41"
across two tasks with different scales is meaningless.

The literature offers four conventions, and this document picks the one this project already
has both anchors for:

1. **Human-normalized score** (Atari/DQN lineage): needs a human baseline per game. Not
   available.
2. **Per-task max-observed-return normalization** (CORA, Powers et al. 2022): cheap and
   self-contained, but the denominator depends on what happened to be observed during the
   run, which makes it noisy at small scale and lets the denominator move.
3. **Bounded task-native metric** (Continual World, Wołczyk et al. 2021, uses success rate):
   sidesteps the problem entirely, but needs a per-task success criterion, which Meta-World
   supplies natively and Game Boy titles do not.
4. **Reference-run normalization**: normalize against a *single-task specialist of the same
   architecture*. This is Continual World's forward-transfer convention and the structural
   analogue of human-normalized Atari scores.

**Decision: (4), with the untrained policy as the lower anchor.** For task *j*, seed *s*:

> **`norm(j, s) = ( R(j, s) − R_init(j) ) / ( R_spec(j) − R_init(j) )`**

where `R(j, s)` is the mean extrinsic return of the policy under test on task *j* for seed
*s*; `R_init(j)` is the mean return of the **untrained** (`--steps 0`) policies on task *j*,
averaged over seeds (control C2); and `R_spec(j)` is the mean return of the **single-task
specialist** on task *j*, averaged over seeds (control C1). By construction the specialist
sits at 1.0 and an untrained policy at 0.0, per task, so scores are directly averageable
across tasks. Values above 1.0 are possible and meaningful (the multi-task policy beat the
specialist); values below 0.0 are possible and meaningful (it is worse than not training).

**Both anchors are means over ten seeds and are frozen before any Phase 2 condition is
scored**, so the denominator cannot drift with the thing being measured. `R_init(j)` and
`R_spec(j)` are published as raw numbers next to every normalized table.

**Secondary, reported alongside and never instead of it: the native per-task return**, in
the task's own units, exactly as Phase 1 reported it. A normalization can hide a result; a
raw number next to it cannot.

**Third, where a task has one: a bounded native progress fraction** — for Super Mario Land
levels, `read_level_progress()` at episode end divided by that level's measured total length
(1-1's is known: a real completion reached 2592, `envs/ram_map.py`). This is the closest
thing available to Continual World's bounded success rate and is worth having because it is
interpretable without any reference run at all. It is a *supporting* metric because it does
not exist for every conceivable task (Tetris has no "progress").

### 8.2 The performance matrix

For a task sequence of length *T*, define **`R[i, j]` = the normalized score (§8.1) on task
*j* after training stage *i* is complete**, with the convention `R[0, j]` = the untrained
policy = 0.0 by construction. `R[i, i]` is performance on a task immediately after learning
it; `R[T, j]` is performance on task *j* at the very end.

**Filling this matrix is the mechanical core of the phase**: every checkpoint must be
evaluated on *every* task, not only on the task it was trained on. That is the single most
consequential change to the evaluation matrix driver (§9, item 6).

**Continual evaluation, not just stage boundaries.** CORA's central methodological point is
that a single end-of-training snapshot hides transient forgetting and recovery. `train.py`
already writes a checkpoint every `--checkpoint-every` steps, so the *trace* — normalized
score on every task at every checkpoint — is available for free and should be plotted, not
just the four corners of a 2×2 matrix.

### 8.3 Forgetting, backward transfer, forward transfer — the exact definitions

Pre-registered, with the convention chosen explicitly rather than left to whoever writes the
results.

**Backward transfer** (Lopez-Paz & Ranzato, GEM, 2017):

> **`BWT = (1/(T−1)) · Σ_{i=1..T−1} ( R[T, i] − R[i, i] )`**

Negative means forgetting. This compares end-of-sequence performance to performance
*immediately after* that task was learned.

**Forgetting measure** (Chaudhry et al., A-GEM, 2018), the stricter convention:

> **`f_j = max_{l ∈ {1..T−1}} R[l, j] − R[T, j]`**,  **`F = (1/(T−1)) · Σ_{j=1..T−1} f_j`**

Peak-ever minus final, rather than post-training minus final. Strictly ≥ −BWT.

**Forward transfer** (GEM convention):

> **`FWT = (1/(T−1)) · Σ_{i=2..T} ( R[i−1, i] − b_i )`**

where `b_i` is the untrained reference on task *i*, which under §8.1's normalization is
**0.0 by construction** — so `FWT` reduces to the mean *zero-shot* normalized score on each
task just before it is trained. Clean, and it is a genuinely interesting number here: does a
policy trained only on Mario 1-1 do better than an untrained one on 2-1, before ever seeing
2-1?

**Average performance** (A-GEM convention): **`A_T = (1/T) · Σ_j R[T, j]`** — the single
number that answers "how good is the final agent, across everything it was supposed to have
learned".

**Note for the T = 2 case, stated so nobody re-derives it later.** With two tasks and
evaluation only at stage boundaries, `F = −BWT` exactly. They diverge only once the
continual-evaluation trace of §8.2 is used, because then a task's peak can occur strictly
between boundaries. **Pre-register both and compute both from the trace**, so the stricter
convention is available and the weaker one cannot be quietly substituted.

### 8.4 The declared headline

To make this unambiguous before any number exists, in keeping with `EXPERIMENT_LOG.md`
§17.9:

> **The Phase 2 headline is `F` (the A-GEM forgetting measure, §8.3) for the SEQ condition,
> and `A_T` (average final normalized performance, §8.3) for the INT condition, each with an
> exact permutation test and a bootstrap CI over ten training seeds.**

Everything else in §8 is supporting evidence for those two numbers.

### 8.5 Unit of analysis and statistics — unchanged from Phase 1

`RESULTS.md` §2.4's rule carries over verbatim and is not up for renegotiation: **the unit
of analysis is the training seed, never the episode.** The environment is deterministic, so
the per-checkpoint spread the evaluation harness reports is policy-sampling variance and
nothing else; averaging more episodes shrinks the wrong error bar. Every metric above is
reduced to one number per training seed before any statistic sees it, which
`analysis/aggregate_results.py` already enforces by construction.

- **Ten seeds per condition.** Continual World used 20; Caccia et al. used 8; deep-RL
  practice is commonly 3–5. Ten is what Phase 1 used, is affordable at Phase 2's budget
  (§10), and does not need to be re-argued.
- **Exact permutation tests and bootstrap CIs**, as in Phase 1 — already implemented.
- **Multiple comparisons disclosed, not corrected away.** Phase 1 pre-committed to
  disclosing this (`EXPERIMENT_LOG.md` §8) and reported the Bonferroni arithmetic honestly;
  Phase 2 has more conditions and therefore a larger family, and the disclosure obligation
  scales with it. The declared headline (§8.4) exists partly so that the family is small
  where it matters.
- **`rliable`-style aggregate statistics** (Agarwal et al. 2021, IQM + stratified bootstrap)
  are the field's recommended answer for compute-constrained labs. They are noted as
  available rather than adopted: Phase 1's exact permutation test over ten seeds is already
  a stronger instrument than the mean-with-error-bars that paper was written against, and
  switching statistical machinery between phases would make v1/v2/Phase 2 incomparable.

### 8.6 What is deliberately not a metric

- **Combined return** (extrinsic + novelty bonus). Phase 1 reported it as a diagnostic and
  never as the scoreboard; that stands.
- **Training reward.** Phase 1's most instructive single contrast was that the corrections
  closed the *training*-reward gap 5.82× → 1.38× and moved the evaluation gap not at all
  (`EXPERIMENT_LOG.md` §21.2). Training reward is a diagnostic here too.
- **Wall-clock or sample efficiency as a headline.** Reported (§10) because it is real, but
  the question is capacity and retention, not speed.

---

## 9. What has to change in the code

From the coupling audit, ranked cheapest to most expensive. This is scope information for
the implementation plan, not the plan itself.

1. **`models/` — nothing.** `PolicyValueGRU` already takes `obs_dim`/`n_actions`; it imports
   nothing from `envs/`.
2. **`training/ppo.py` — nothing.** Pure tensor math, zero game references.
3. **A task axis and a small env registry.** Today `MarioLandEnv`, `OBS_DIM`, `OBS_MEAN` are
   imported directly at three production sites (`training/train.py:142,605`,
   `training/rollout.py:25,33`, `training/evaluate.py:109,189`) and `N_ACTIONS` is computed
   at import time (`training/train.py:150`). Replace with a lookup keyed by a new `--task`
   flag. Mechanical, but it touches every one of those sites plus both matrix drivers.
4. **Checkpoint and output-path identity.** `save_checkpoint` records
   `arm/seed/grad_clip_mode/run_tag/embed_init_mode/embed_scale` and **no task**, and
   `run_dir_for` names directories `{arm}_seed{N}[_{tag}]`. Two tasks would silently collide
   on `baseline_seed0/`. A task coordinate has to be threaded through the checkpoint dict,
   the directory naming, and `analysis/aggregate_results.py`'s filename parsing. Small, but
   getting it wrong corrupts a matrix silently — precisely the failure class
   `EXPERIMENT_LOG.md` §19.4 already caught once.
5. **Per-task novelty gates and multi-env rollout ownership** in `train.py`'s loop (§7.4,
   §7.5).
6. **The evaluation matrix gains a task axis**, and — new, and the part that is not just a
   bigger loop — every checkpoint must be evaluated on **every** task, not only its own, to
   fill the performance matrix. That is the mechanical core of the forgetting measurement.
7. **Save-state-based level starts** for Phase 2a (§4.3), with determinism verified rather
   than assumed.
8. **A second game's RAM map, boot routine and env module** for Phase 2b (§5.1) — the
   genuinely expensive item, and reverse-engineering rather than software design.

---

## 10. Compute budget, measured rather than guessed

All figures from this project's own measurements, not estimates.

- **Single-run GRU throughput, quiet machine, v2 flags: 1,303.4 env-steps/s**
  (`RESULTS.md` §20).
- **Ten-way parallel, real matrix conditions:** Phase 1 v2's baseline arm ran 10 seeds ×
  1,000,064 steps between 03:12:01 and 03:57:32 — **45.5 minutes wall for ten 1M-step GRU
  runs**, i.e. ≈3,660 env-steps/s aggregate (`EXPERIMENT_LOG.md` §21.1).
- **Evaluation:** 120 evaluations in 21 minutes at `--jobs 8` (same source).

A Phase 2a matrix at *N* = 1,000,064 and 10 seeds per condition, with the conditions of §7.1
and the Q3 ablation, is on the order of **60–80 training runs totalling ~100M env steps**,
which at the measured 10-way aggregate is **roughly 7–8 hours of wall clock** — one
unattended overnight run, the same shape as Phase 1 v2. Evaluation, even at 3–4× v2's
matrix size because of the all-checkpoints-on-all-tasks requirement, stays well inside two
hours.

**This is affordable, and that is a design input, not a footnote.** It is why 10 seeds per
condition (Phase 1's *n*, and above the 3–5 common in published deep RL) is proposed rather
than 3, and why the answer to "can we afford the controls" is yes.

The reservoir arm's ~3× throughput penalty (`RESULTS.md` §20) does **not** apply — Phase 2
runs the GRU only.

---

## 11. Open decisions — for the project owner

Collected here so none of them is buried in prose. Each is genuinely undetermined by the
existing documents; everything not listed here is decided above with its rationale.

1. **Does Phase 2a (cross-level, §4) run at all, or does the phase wait for a second ROM?**
   Recommended: run it. It builds every artefact Phase 2b needs and it is the only path that
   does not block on an external dependency.
2. **Which second Game Boy / Game Boy Color ROM to obtain.** Recommended: **Kirby's Dream
   Land** (§5.3). Only he can execute this.
3. **Which levels are the tasks in Phase 2a.** A minimal, defensible set: **1-1** (the Phase 1
   task, so there is continuity), **2-1** (near transfer: platformer, different world, water),
   **2-3** (far transfer: the Marine Pop shooter). Whether to use two tasks or three changes
   the matrix size by ~50%.
4. **The exact union action set** (§6.3) — should be fixed empirically in the implementation
   plan's first task, the way Phase 1's was, not decided on paper.
5. **Whether the per-task critic head (§6.5) is a headline condition or an ablation.**
   Recommended: ablation.
6. **Whether Q4 (mitigation) is in scope for this phase at all**, or is deferred to a Phase
   2.5 the way build-order Phase 3 was deferred in `DESIGN.md` §7. Recommended: keep it
   conditional on Q2, and do not plan it in detail until Q2 has a number.

---

## 12. Recommended next step, bounded

**Not** "start training". The bounded next step, if this document is accepted, is:

> Write the Phase 2a implementation plan (`docs/superpowers/plans/`), and execute only its
> first task: **the save-state-based level-start mechanism (§4.3) plus the empirical
> confirmation of §4.4's four unknowns about the vehicle levels.**

That task involves no training, costs minutes of compute rather than hours, and is a strict
prerequisite for everything else in Phase 2a. It also answers a question that could
*invalidate the whole Phase 2a proposal* — if save states turn out to be nondeterministic
under PyBoy, or if `read_level_progress()` is meaningless in the vehicle stages, the testbed
design in §4 needs revising before anything is built on it. Finding that out for the price
of one afternoon is the cheapest risk reduction available.

The decision to spend compute on the actual matrix (§10) is the project owner's, and should
be made after that task reports, on the same footing Phase 1's implementation was: a design
document and an implementation plan existed and were reviewed first.

---

## 13. Out of scope for this document

- **Roadmap Phase 3 (GBA) and Phase 4 (Fire Red).** Not abandoned — `DESIGN.md` §1.1. The
  fourteen GBA ROMs on disk are not a shortcut to multi-game work: using them would change
  the platform *and* the games at once, and the mGBA bridge is still a scratchpad spike.
- **The frozen-reservoir architecture.** Phase 1 closed its question; a separate bounded
  pilot is testing a variant. Phase 2 is specified against the policy *interface* so that
  pilot's outcome changes nothing here (§1.2).
- **Pixel observations.** This project reads RAM. Adding vision would change the observation
  modality, the parameter budget and the throughput profile simultaneously.
- **Any cloud GPU rental.** Same zero-budget discipline as `DESIGN.md` §2 and §10.
- **More than two or three tasks.** A long task sequence is where continual-RL benchmarks
  live, and it is the right eventual target; it is not the right first measurement.

---

## References

Carried over from `docs/DESIGN.md` and not repeated. New to this document:

*(Bibliography completed alongside §8.)*
