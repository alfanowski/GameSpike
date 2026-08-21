# Roadmap Phase 2 — Multi-Game Generalization on the GRU Architecture

Status: **design proposal under review.** The testbed-viability probe of §13 has been
executed (§14). One bounded training step — the two single-task specialists and their
untrained anchors, ≈20M env steps — is **authorised and pre-registered in §15**; nothing
else in §7.1's table is. The full matrix of §10 remains a proposal and must not be started
without the project owner's decision.
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
spend compute. §13 proposes one bounded first step and explicitly leaves the decision to
start it with the project owner, on the same footing Phase 1's implementation had: a
design doc and a plan existed and were signed off before any run started.

**0.4 Open questions are marked.** Anything genuinely undetermined by the existing
documents is tagged **OPEN** and collected in §12. Anything the existing documents already
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

## 3. The binding constraint found while writing this document — since lifted

> **Note added 2026-08-21, hours after this section was written.** The constraint below is
> **no longer binding**: the project owner supplied **Kirby's Dream Land (USA/Europe)**
> (§12, OPEN-2). The section is kept unedited because its *reasoning* is what produced the
> Phase 2a / Phase 2b split, and that split survives the constraint being lifted — Phase 2a
> was never merely a workaround for a missing cartridge, it is the cheaper single-variable
> measurement that validates the machinery Phase 2b then reuses (§4.2). **The sequencing
> does not change.** What changes is that Phase 2b is now unblocked rather than waiting.

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

> **MEASURED 2026-08-21 (§14.3).** Confirmed, and worse than this section guessed: the camera
> term is *exactly* free — `level_block` advances by an identical +7 over 240 frames whether
> the player holds right or presses nothing. 78% of 2-3's progress reward and 50% of 4-3's
> accrues to a policy that does nothing at all.

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
`envs/ram_map.py`. It is the substance of §13's recommended first task.

> **RESOLVED 2026-08-21 (§14.1, §14.4).** All four checked. `0xC0AB` re-bases per level
> (12 platformer / 14 vehicle) but its monotonic-camera semantics are exactly what makes the
> autoscroll problem bite; `0xC202` does hold the craft's X; the timer does run in the
> vehicle stages. The on-ground heuristic is not merely meaningless in free flight — it is
> **actively misleading**, reporting "grounded" whenever the player stops pressing.

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

> **REVISED UPWARD 2026-08-21 (§14.7).** More expensive than this paragraph assumed. Kirby's
> first level is not traversable by hold-right, by right+jump, by right+fly, or by uniform
> random play, so the blind monotonic scan that discovered Mario's map **cannot discover
> Kirby's progress addresses**. That pass needs human-recorded input traces or save states
> captured deeper in the level. Kirby's health, lives, X, Y and score addresses *did* all
> confirm cleanly (§14.6) — it is specifically the progress signal that is hard.

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

### 5.3 Which second title — contested when written, resolved 2026-08-21

> **RESOLVED.** The project owner supplied **Kirby's Dream Land (USA/Europe)** — matching the
> recommendation this section reaches. See §12, OPEN-2, for the verified ROM details. The
> argument below is kept **unedited**, per this project's append-only practice: it is the
> reasoning the decision rests on, and a future reader revisiting the Tetris option should
> see the case for it as it was actually made, not a version rewritten after the fact.

Two research passes reached **opposite recommendations**, which is itself the useful signal:
this is not determined by the existing documents and should not be silently decided here. It
is **OPEN-2** in §12, and it is the one open item that costs the project owner something
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

The existing twelve slots generalise better than they look, because the nine that carry
anything are already about a *player in a scrolling game* rather than about Mario:

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
| 9–11 | reserved zeros | **still reserved** for the enemy-relative features a later plan wires — *not* reused for the task ID, which is appended instead (§6.2) |

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
trainable parameters, +0.048%** against a 132,715-parameter budget — more than two orders of
magnitude inside `tests/test_parameter_parity.py`'s ±10% band — the band is ~207× wider than
the change — and small enough that the comparison cannot be confounded by capacity.

**Appended, not squatted.** The task ID goes on the *end* of the vector, taking
`obs_dim` from 12 to 12 + *K*. It deliberately does **not** occupy the reserved slots 9–11,
which belong to the enemy-relative features `envs/mario_land_env.py:147-149` promises to a
later plan; quietly repurposing them would make that plan's observation silently incompatible
with every Phase 2 checkpoint.

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

> **REQUIRED, not optional, as of 2026-08-21 (§14.4, §14.6).** `up` and `down` do nothing in
> the platformer levels and *move the craft* in 2-3 and 4-3 (dy −59 holding up); Kirby needs
> them too (up = fly, down = duck/swallow). Phase 1's ten actions cannot play the vehicle
> stages at all.

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

Two clarifications the table cannot carry:

- **SPEC-2N is run for the *first* task of the sequence only.** Its job is to answer "would
  another *N* steps on task A have helped anyway?", which is only a confound for the task
  whose retention is being measured. Running it for every task would double a control that
  only one task needs.
- **INIT is per task**, because §8.1's lower anchor `R_init(j)` is a per-task quantity. It
  costs no env steps.

### 7.1.1 Task order is a variable, and running only one order is a weaker result

**Decision: run SEQ in both orders — A→B *and* B→A — and report forgetting for both.**

Order effects are real in continual RL, and a single ordering cannot separate *"training on
B destroys A"* from *"A is simply the more fragile task"*. The Phase 1 discipline of
symmetric treatment (`--seed` drives both arms symmetrically; the untrained arms were shown
statistically indistinguishable before any trained claim was made) points the same way here:
the two tasks should be treated symmetrically unless there is a reason not to.

The cost is 10 more runs at 2*N*, i.e. +20M env steps, ≈1.5 h at the measured aggregate
(§10). **If compute has to be cut, this is the first thing to cut** — and cutting it must
then be stated as a limitation, not omitted.

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
score on every task at every checkpoint — is available and should be plotted, not just the
four corners of a 2×2 matrix.

**It is available, but not free, so evaluation runs at two tiers.** Scoring every checkpoint
on every task at Phase 1's full protocol (30 episodes × two recurrent-state regimes) is on
the order of 1,800 evaluations and would cost more wall clock than the training it measures
(§10). The split:

- **Stage boundaries — the full Phase 1 protocol.** 30 episodes, both `continuous` and
  `reset128`. These fill `R[i, j]` and every statistic in §8.3 and §8.4 is computed from
  them. This tier is what the headline rests on.
- **The trace — a cheaper instrument, labelled as one.** 10 episodes, `reset128` only (the
  regime training actually used). Enough to see the *shape* of forgetting and recovery
  between boundaries; **not** enough to support an arm comparison, and it must never be used
  for one. This is the same distinction `training/evaluate.py`'s own "WHAT THIS HARNESS
  CANNOT TELL YOU" section already draws between a per-checkpoint instrument and the
  experiment.

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

**Note for the T = 2 case, stated so nobody re-derives it later.** Every statistic above is
computed from the **stage-boundary tier** of §8.2 — the full-protocol evaluations — and
never from the cheaper trace, so the instrument is the same one Phase 1 used. With two tasks
and boundary-only evaluation, **`F = −BWT` exactly**; they are not two independent findings
and must not be reported as if they were.

They can only diverge if a task's peak occurs strictly *between* boundaries. The trace exists
to reveal exactly that, and its role here is bounded: **if the trace shows an intermediate
peak above `R[i, i]`, that is reported as a caveat on `F` — never silently folded into it.**
Mixing a 10-episode instrument into a statistic computed from 30-episode measurements would
be precisely the kind of quiet instrument-switch `EXPERIMENT_LOG.md` §17.11 already had to
rule out once.

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

**The two-task Phase 2a matrix, counted rather than hand-waved:**

| condition | runs | steps/run | env steps |
|---|---|---|---|
| SPEC-A, SPEC-B | 2 × 10 | 1M | 20M |
| SPEC-2N (first task only, §7.1) | 10 | 2M | 20M |
| INT, unconditioned | 10 | 2M | 20M |
| SEQ A→B, unconditioned | 10 | 2M | 20M |
| SEQ B→A, unconditioned (§7.1.1) | 10 | 2M | 20M |
| INT + SEQ A→B + SEQ B→A, task-conditioned (Q3) | 30 | 2M | 60M |
| INIT | 2 × 10 | 0 | ~0 |
| **total** | **90 training runs** (+20 zero-step INIT) | | **160M** |

At the measured 10-way aggregate of ≈3,660 env-steps/s that is **≈12 hours of wall clock** —
one unattended overnight run, the same shape as Phase 1 v2's, which itself ran 23:55 → 04:22
including evaluation and analysis. **Two levers if that is too long**: dropping the reverse
order (§7.1.1) takes it to ≈9 h, and dropping the Q3 ablation takes it to ≈7.5 h. **A
three-task testbed (OPEN-3) is roughly 1.6× the full figure**, i.e. two nights rather than
one, which is the real argument for starting with two.

**Evaluation grows faster than training does**, because §8.2 requires every checkpoint to be
scored on every task. Counted at two tiers (§8.2): the stage-boundary matrix is ~360
evaluations at the full Phase 1 protocol (30 episodes, both regimes), and the
continual-evaluation trace is ~1,800 cheaper evaluations (10 episodes, `reset128` only).
Against v2's measured 120 full evaluations in 21 minutes at `--jobs 8`, that is **roughly 3
hours** — not the "under two" a first estimate suggested, and worth knowing before the
pipeline is chained end-to-end.

**This is affordable, and that is a design input, not a footnote.** It is why 10 seeds per
condition (Phase 1's *n*, above the 3–5 common in published deep RL, below Continual World's
20) is proposed rather than 3, and why the answer to "can we afford the controls" is yes.

The reservoir arm's ~3× throughput penalty (`RESULTS.md` §20) does **not** apply — Phase 2
runs the GRU only.

---

## 11. Risk register

Same format as `DESIGN.md` §9, and the same rule: a risk listed here with a mitigation is
not thereby solved.

| Risk | Severity | Mitigation |
|---|---|---|
| One 132,715-parameter policy simply cannot hold two tasks | High — **this is the experiment** (Q1) | The controls (§2.2) make a negative answer informative rather than a failed run, exactly as `DESIGN.md` §5's control did for Phase 1 |
| Progress-based reward is degenerate in the autoscroll vehicle stages (§4.1.1) | **High — MEASURED §14.3: the camera term is 100% free, 78%/50% of total** | Vehicle stages get their own reward definition (survival + score delta); "define and validate it" is its own plan task with its own acceptance test |
| Kirby's progress addresses cannot be found by the blind scan that found Mario's (§14.7) | **High — MEASURED**, no simple policy traverses level 1 | Discovery pass switches to human-recorded input traces or deep save states; Kirby's reward leans on score delta and survival, and slot 0 may be weak for Kirby — to be disclosed, not hidden |
| The on-ground slot reports "grounded" for "idle" in free-flight stages (§14.4) | Medium — silent | Keep the slot (task-agnostic agents must survive distribution shift) but never let a reward or termination rule read it |
| ~~`read_level_progress()` / `0xC0AB` semantics break in the vehicle stages~~ | ~~High~~ **CLOSED §14.1/§14.4** | Checked: `0xC0AB` re-bases per level and `0xC202` holds the craft's X. The semantics hold; the autoscroll row above is the real problem |
| ~~PyBoy save-state loading is not bit-identically deterministic~~ | ~~High~~ **CLOSED §14.2** | Measured: identical WRAM+HRAM digests across same-process reload, fresh-process reload and boot-from-power-on, on 1-1 and 2-3. `timer_div` randomisation does not affect it |
| A shared novelty buffer pays the agent for switching game (§7.5) | High if unnoticed, trivial to prevent | One `NoveltyGate` per task; recorded here so a shared gate reads as a regression |
| Two tasks' checkpoints collide on `{arm}_seed{N}/` (§9 item 4) | Medium — silent matrix corruption | A task coordinate threaded through the checkpoint dict, directory naming and the aggregation regex; this failure class was already caught once (`EXPERIMENT_LOG.md` §19.4) |
| Reward-scale imbalance makes one task dominate the gradient | Medium | Pre-registered fixed per-task reward scale (§6.4); PopArt named as the fallback |
| Enlarging the action space silently breaks comparability with Phase 1's published returns (§6.3) | Medium | Phase 2 re-runs its own Mario specialist as the C1 reference and says so in the results, rather than letting a reader compare two tables |
| Cross-level shift turns out too weak to measure anything | Medium | This is why Phase 2a is framed as machinery validation and a lower bound, and why it does not substitute for Phase 2b |
| The whole phase blocks on a ROM the project owner has to supply | Medium, external | Phase 2a needs no new ROM; the split exists precisely so nothing waits |
| Alternating-update interleaving is not a mixed minibatch (§7.2) | Low, disclosed | Stated as a limitation up front rather than discovered in review |

---

## 12. Open decisions — for the project owner

Collected here so none of them is buried in prose. Each is genuinely undetermined by the
existing documents; everything not listed here is decided above with its rationale.

- **OPEN-1 — Does Phase 2a (cross-level, §4) run at all, or does the phase wait for a second
  ROM?** *Recommended: run it.* It builds every artefact Phase 2b needs and it is the only
  path that does not block on an external dependency.
- **OPEN-2 — Which second Game Boy / Game Boy Color ROM to obtain. ~~OPEN~~ → RESOLVED
  2026-08-21: Kirby's Dream Land, supplied.** The project owner settled it directly and in
  favour of the recommendation, so §5.3's Kirby-vs-Tetris tradeoff stands as the reasoning
  and is not revisited. Verified on disk:
  `/Users/alfanowski/Desktop/Kirby's Dream Land (USA, Europe).gb`, **262,144 bytes**, header
  cartridge title **`KIRBY DREAM LAND`**, destination byte `0x01` (non-Japanese), cartridge
  type `0x01` (MBC1).
  **The release matters and is the good case.** It is the **USA/Europe** release, which is
  what the public DataCrystal RAM map and PyBoy's built-in wrapper both target — so unlike
  the Fire Red situation `DESIGN.md` §1.1 flags for roadmap Phase 4 (an Italian `BPRI`
  cartridge against community maps written for the US `BPRE`), there is **no
  release-mismatch tax** here. Published addresses are usable as *hypotheses to confirm*
  rather than as a from-scratch discovery problem.
  This unblocks Phase 2b's RAM-map work. It does **not** change the sequencing: Phase 2a
  still runs first, for the reasons in §4.2.
- **OPEN-3 — Which levels are the tasks in Phase 2a, and how many. ~~OPEN~~ → RESOLVED
  2026-08-21: the two-task set {1-1, 2-1}. 2-3 is DEFERRED, not dropped.**
  **1-1** carries continuity with Phase 1; **2-1** is the near-transfer task, and §14.5
  measured the shift as real rather than cosmetic — it kills a hold-right policy at frame 235
  against 1-1's 336, burning both lives inside eight seconds. That is sufficient for a
  meaningful near-transfer test on its own.
  **Why 2-3 is deferred rather than included.** It costs ~1.6× the compute *and* it requires
  the vehicle-stage reward redesign §14.3 showed is genuinely necessary — 78% of 2-3's
  progress reward and 50% of 4-3's accrues to a policy that does nothing, and the camera term
  is *exactly* free. That is a second, harder, not-yet-designed problem, not a scope
  increment. **Bundling an unsolved reward-design problem into the first training step this
  phase runs is the wrong order of operations.**
  **This is a deferral with a standing commitment.** Far transfer stays a real future step:
  2-3 gets its own reward-design pass, at the same rigor as everything else here, and is then
  added. Recorded this way so "deferred" is never later read as "quietly dropped" — the same
  distinction `DESIGN.md` §10 draws for its own out-of-scope list.
  **Consequence that removes a dependency.** At a two-task {1-1, 2-1} scope there is no
  vehicle stage, therefore no need to widen the action space (§6.3, §14.4), therefore **no
  `actor_head` re-run risk**. The specialists can run at Phase 1's ten-action set with no
  asterisk. When 2-3 is added later, the union action space arrives with it and that step
  re-runs its own references — which it would have to do anyway.
- **OPEN-4 — The exact union action set** (§6.3). **~~OPEN~~ → RESOLVED 2026-08-21 in favour
  of the recommendation: do not decide it on paper.** Fix it empirically, in the
  implementation plan's first task, the way `DESIGN.md` §11 fixed Phase 1's. Per OPEN-3's
  resolution the question is **not live for Phase 2a's two-task scope** — it becomes live when
  2-3 or Kirby enters, and §14.4/§14.6 have already supplied the measurements it will be
  decided on (`up`/`down` move the craft and are inert in the platformer levels; Kirby needs
  both).
- **OPEN-5 — Whether the per-task critic head (§6.5) is a headline condition or an
  ablation.** *Recommended: ablation.*
- **OPEN-6 — Whether Q4 (mitigation, §2.1) is in scope for this phase at all**, or is
  deferred the way build-order Phase 3 was deferred in `DESIGN.md` §7. *Recommended: keep it
  strictly conditional on Q2, and do not plan it in detail until Q2 has a number.*

---

## 13. Recommended next step, bounded

> **EXECUTED 2026-08-21 — results in §14.** The probe specified below was authorised and has
> been run: `scripts/phase2_viability_probe.py`, raw output at `results_phase2_probe.json`,
> ~6 seconds of emulation, no training and no checkpoints. The specification is kept here
> unedited so §14's results can be read against what was actually asked for.
>
> **Superseded as "the next step" by §15.** The probe came back viable, and one bounded
> training step — SPEC-A and SPEC-B, the specialist references — has since been authorised
> and pre-registered. It is a strict subset of §7.1, it is the denominator every later metric
> divides by, and it carries its own pre-registered go/no-go. Nothing beyond *it* has been
> started.

**Not** "start training". The bounded next step, if this document is accepted, is:

> Write the Phase 2a implementation plan (`docs/superpowers/plans/`), and execute only its
> first task — **a testbed-viability probe**. Super Mario Land side, four checks:
>
> 1. `start_game(world_level=…)` genuinely loads 2-1, 2-3 and 4-3 (not just their HUD).
> 2. A `save_state()` captured after the level intro, reloaded on every `reset()`, replays
>    **bit-identically** — verified by hashing a fixed-action replay, not asserted.
> 3. `read_level_progress()` is measured in 2-3 and 4-3 while pressing nothing, to quantify
>    how much of it is pure autoscroll (§4.1.1) — the number that decides the vehicle-stage
>    reward design.
> 4. The four unconfirmed RAM semantics of §4.4, checked with `envs/ram_scan_tool.py`.
>
> **Extended 2026-08-21, now that the Kirby ROM exists (§12, OPEN-2)** — three more checks,
> at the same cost bar, because they de-risk Phase 2b for the price of running them
> alongside:
>
> 5. PyBoy's Kirby wrapper actually binds to this cartridge and its `start_game()` reaches
>    live gameplay (the wrapper matches on `cartridge_title = "KIRBY DREAM LAN"`, 15
>    characters, against a header that reads `KIRBY DREAM LAND` — a truncation that should
>    match but has not been observed to).
> 6. The published addresses behave as documented — score `0xD070`–`0xD073`, health
>    `0xD086`, lives `0xD089` from the wrapper; player X/Y `0xD05C`/`0xD05D` and scroll
>    `0xD051` from DataCrystal — under the project's own rule that a published address is a
>    hypothesis, never a fact (`envs/ram_map.py:63-66`).
> 7. Whether a **level-progress signal composes** for Kirby the way it does for Mario
>    (unwrapped camera coarse part + local X fine part), since §6.1's shared observation
>    schema puts progress delta in slot 0 and Phase 2b's reward depends on it.

That task runs no training, costs minutes of compute rather than hours, and is a strict
prerequisite for everything else in Phase 2a. More importantly it can **invalidate the
Phase 2a proposal cheaply**: if save-state loading is not deterministic under PyBoy, or if
the camera-derived progress signal is meaningless in the vehicle stages, §4's testbed design
needs revising *before* anything is built on it. Finding that out for the price of an
afternoon is the cheapest risk reduction available in this phase.

**Nothing beyond those four checks should start without review.** The decision to spend
compute on the matrix of §10 is the project owner's, and belongs after that probe reports —
on exactly the footing Phase 1's implementation had: a design document and an implementation
plan existed and were reviewed before any run began.

---

## 14. Testbed-viability probe — RESULTS (run 2026-08-21)

The probe §13 proposed has been **run**. It is `scripts/phase2_viability_probe.py`, its raw
output is committed at `results_phase2_probe.json`, and it costs **~6 seconds** of emulation
end to end. No training was run and no checkpoint was written.

Every check is behavioural. "The status bar says 2-3" was not accepted as evidence that
level 2-3 loaded; "four submarine sprites are on screen and zero Mario sprites are" was —
using tile identifiers taken from PyBoy's own wrapper, which is a channel independent of
every RAM address in question.

**Headline: five checks pass cleanly, one passes with a number that changes a reward
design, and one returns a genuine negative that raises Phase 2b's cost.**

### 14.1 SML-1 — the levels really load. PASS

| requested | `0xFFB4` agrees | Mario sprites | submarine | plane | independent verdict | `level_block` at start |
|---|---|---|---|---|---|---|
| 1-1 | yes | 4 | 0 | 0 | **mario** | 12 |
| 2-1 | yes | 4 | 0 | 0 | **mario** | 12 |
| 2-3 | yes | 0 | **4** | 0 | **submarine** | 14 |
| 4-3 | yes | 0 | 0 | **4** | **plane** | 14 |

`start_game(world_level=…)` genuinely loads the level, vehicle stages included. §4.3's
ranking was right and §4.3's rejection of the `0xFFB4` poke needs no revisiting.

**This also closes §4.4's second unknown:** `0xC0AB` **re-bases per level** (12 on the
platformer levels, 14 on the vehicle ones), so `read_level_progress()` starts from a small
per-level baseline rather than accumulating across levels.

### 14.2 SML-2 — save states are bit-identically deterministic. PASS, 5/5, on two levels

Save states are 143,103 bytes. A 300-frame scripted replay from a state captured at the
level start produces **one identical WRAM+HRAM digest** across every condition tested, on
both 1-1 and 2-3:

| check | result |
|---|---|
| reload into the *same* emulator, ×3 | identical |
| reload into a *fresh* emulator, ×3 | identical |
| same-process digest == fresh-process digest | identical |
| boot from power-on, ×3 | identical |
| boot-from-power-on == save-state path | identical |

**And `timer_div` turns out not to matter, which was checked rather than assumed.** PyBoy
randomises the DIV register by default (`timer_div=None`); pinning it to 0 produces the
*same* digest, over 4 boots each. Pinning it is therefore free insurance, not a
requirement — recorded so nobody later cargo-cults the flag or, worse, omits it believing it
was load-bearing.

**Phase 1's determinism assumption survives into Phase 2a intact.** This was the check most
capable of invalidating §4 outright, and it did not.

### 14.3 SML-3 — the autoscroll problem is real, and worse than §4.1.1 guessed. CONFIRMED

240 frames, idle (pressing nothing) versus holding right:

| level | idle Δprogress | hold-right Δprogress | **free fraction** | idle Δ`level_block` | hold-right Δ`level_block` |
|---|---|---|---|---|---|
| 1-1 | 0 | +127 | **0.00** | 0 | +6 |
| 2-3 | +112 | +143 | **0.78** | **+7** | **+7** |
| 4-3 | +112 | +222 | **0.50** | **+7** | **+7** |

§4.1.1 predicted "mostly free". The measurement is sharper than that and the sharper version
is the one that matters:

> **The camera term of `read_level_progress()` is *exactly* free in the vehicle stages.**
> `level_block` advances by the same +7 whether the player holds right or presses nothing at
> all. The only player-responsive component left is `ADDR_MARIO_X`, the craft's *screen* X,
> which is bounded by the screen and saturates.

So it is not that a progress reward in 2-3 is noisy — it is that its unbounded component is
100% uncorrelated with the policy, and its correlated component cannot exceed one screen
width. **A vehicle-stage reward must be built on survival and score, not progress.** §4.1.1's
conclusion stands and is now quantified rather than argued.

### 14.4 SML-4 — the vehicle stages need `up`/`down`, measured. §6.3 CONFIRMED

Displacement after holding each direction for 60 frames:

| level | right | left | up | down |
|---|---|---|---|---|
| 1-1 | dx +31 | dx −13 | — | — |
| 2-1 | dx +31 | dx −36, dy +51 | — | — |
| 2-3 | dx +59 | dx −36 | **dy −59** | — |
| 4-3 | dx +59 | dx −36 | **dy −59** | **dy +14** |

`up` and `down` do **nothing** in the platformer levels and **move the craft** in the vehicle
stages. §6.3's union action space stops being a design preference and becomes a requirement:
Phase 1's ten actions cannot play 2-3 or 4-3 at all.

Two more §4.4 unknowns close here: **`0xC202` does hold the craft's X** (it responds to
left/right in both vehicle stages), and **the level timer does run in the vehicle stages**
(−6 over 240 frames, identical to the platformer levels).

The fourth is worse than "meaningless": **the on-ground heuristic is actively misleading in
free-flight.** `ON_GROUND_STILL_FRAMES` infers contact from Y holding still, and in a vehicle
stage Y holds still exactly when the player stops pressing — so slot 4 would report
"grounded" for "idle". Keep the slot (a task-agnostic agent must cope with distribution
shift) but never let a reward or termination rule read it.

### 14.5 SML-5 — how fast a naive rightward policy dies, and the artefact it explains

The first probe run reported *zero* progress for holding right in 2-1, which looked like a
broken level. It was a **death-and-reload artefact**: progress had collapsed back to the
level start by the time the window closed. Traced properly over 480 frames:

| level | first death (frame) | lives left after 480f | peak progress gain |
|---|---|---|---|
| 1-1 | 336 | 1 | +127 |
| 2-1 | **235** | **0** | +95 |
| 2-3 | none | 2 | +415 |
| 4-3 | 443 | 1 | +254 |

**2-1 punishes the rightward reflex harder than 1-1** — it kills a hold-right policy in
about two-thirds the time and burns both lives inside eight seconds. That is a *point in
2-1's favour* as the near-transfer task (§12, OPEN-3): the shift is real rather than
cosmetic, and a policy that merely learned "hold right" on 1-1 will visibly fail there.

### 14.6 KDL-1 and KDL-2 — Kirby binds and its published addresses hold. PASS

- **The wrapper binds.** `pyboy.game_wrapper` is `GameWrapperKirbyDreamLand` — the 15-character
  `cartridge_title` truncation §13 flagged is a non-issue. `start_game()` reaches live
  gameplay and the behavioural control gate passes (X 40 → 76 holding right → 36 holding left).
- **The published addresses behave as documented**, and are hereby hypotheses *confirmed by
  observation* rather than adopted from a wiki:

| address | source | observed |
|---|---|---|
| `0xD086` health | PyBoy wrapper | 6 at start, falls to 5/4 on damage; wrapper agrees |
| `0xD089` lives | PyBoy wrapper | raw **5**; wrapper reports **4** — **it subtracts one** |
| `0xD05C` X | DataCrystal | +36 holding right, −16 holding left |
| `0xD05D` Y | DataCrystal | −75 holding up (Kirby flies), +25 down, −30 on A (jump) |
| `0xD051` scroll | DataCrystal | ±2 with travel; behaves like a camera counter |
| `0xD070`–`0xD073` score | PyBoy wrapper | 0 at start, climbs with enemies defeated |

The lives off-by-one is recorded because it is exactly the kind of detail that silently
corrupts an observation slot: `envs/ram_map.py`'s Mario `read_lives` returns the raw BCD
value, so a Kirby env that copies the wrapper's convention and a Mario env that does not
would put two different quantities in the same shared slot 6 (§6.1).

**Kirby also needs `up`/`down`** — `up` is fly, `down` is duck/swallow — which is the same
conclusion §14.4 reached from the vehicle stages, arrived at independently.

### 14.7 KDL-3 — the genuine negative: Kirby's first level is not traversable by any simple policy

This is the finding that justifies having run a probe at all.

**The good half.** `0xD051` behaves like a real camera counter, and Kirby's X **pins at 76**
once the camera locks — structurally identical to Mario's X pinning at 81. So §6.1 slot 0's
two-part composition (unwrapped coarse camera term + fine local X) is the right *shape* for
Kirby.

**The bad half.** Nothing traverses the level:

| policy | max `scroll_x` reached |
|---|---|
| hold right (2,400 frames) | 13 |
| right + jump every 40 frames | 30 |
| right + fly (tap up) | **71**, then stalled at x=152 for a further 5,400 frames |
| uniform random over the 20-action union space, 2,000 agent steps × 4 seeds | 3, 11, 12, 21 |

Meanwhile **score climbs freely under random play** (0 → 1,000–2,000) and health falls, so
Kirby's *dense* signals are score and survival, not spatial progress.

Three consequences, none of which unseat Kirby as the choice:

1. **The RAM-discovery method does not transfer.** Mario's map was found with
   `envs/ram_scan_tool.py`'s blind hold-right monotonic scan, and that method only works if
   holding right makes the player travel. For Kirby it does not. The discovery pass needs
   **human-recorded input traces or save states captured deeper in the level** — a different
   and more manual technique. §9's item 8 gets more expensive.
2. **Kirby's reward should lean on score delta and survival**, not on a Mario-style progress
   term. Slot 0 of the shared schema may be weak or near-constant for Kirby, which is
   informative in itself and must be disclosed rather than papered over.
3. **`0xD051`'s wrap behaviour is undetermined.** It never exceeded 71 in 7,200 frames, so
   whether it wraps mod 256 (like Mario's `0xFFA4`) or is already coarse (like `0xC0AB`) is
   unknown. It cannot be assumed either way.

**Stated limitation.** The probe established *that* traversal stalls and *where* (x = 152,
`scroll_x` = 71); it did **not** establish *why*. Pillow is not installed in this virtualenv,
so no screenshot was captured and the obstacle was not identified visually. That is a known
gap, cheap to close, and not closed.

### 14.8 Verdict

**Phase 2a is viable and is not invalidated.** The check most capable of killing it —
save-state determinism — passed on every variant tested, and the level-start mechanism works
on all four candidate levels including the vehicle stages.

Three things the probe changed rather than confirmed, all folded in above: the vehicle-stage
reward must be survival-and-score (§14.3), the union action space is required rather than
preferred (§14.4), and **Phase 2b's Kirby RAM work is harder than §5.1 estimated** (§14.7).
None of them changes the sequencing in §4.2.

---

## 15. Pre-registration: SPEC-A / SPEC-B, the specialist reference runs

**Written 2026-08-21, before any Phase 2a training run existed and before any number was
produced.** This project pre-registers because `EXPERIMENT_LOG.md` §17.9 requires the shape
of a write-up to be fixed before the numbers, so the shape cannot be chosen to flatter them.

### 15.1 What is being run, and what is deliberately not

Authorised scope: **SPEC-A and SPEC-B only** — the two single-task specialists of §7.1 —
plus their zero-step `INIT` anchors. **≈20M env steps.** Nothing else in §7.1's table is
authorised: no INT, no SEQ, no Q3 ablation, no mitigation arm.

| | value |
|---|---|
| arm | `baseline` (the GRU, `models/policy_value_gru.py`) |
| tasks | `1-1` and `2-1` (§12, OPEN-3) |
| seeds per task | 10, independent (0–9) |
| env steps per run | 1,000,064 |
| INIT anchors | the same grid at `--steps 0` |
| optimizer / clipping / init | **inherited from Phase 1 v2 unchanged**: Adam `lr=3e-4`, `--grad-clip-mode per-group`, `--embed-init-mode centered`, `--embed-scale 3.0` |
| centring constant | `OBS_MEAN_PHASE2A`, the **mixture** mean over both tasks (§6.1) |
| action set | Phase 1's ten (§12, OPEN-3 — no vehicle stage, so no widening) |
| evaluation | 30 episodes, eval seed 0, both `continuous` and `reset128` |
| checkpoint selections | `final` and `init` only — `best` is selected on training reward and adds nothing to a reference measurement |
| unit of analysis | the training seed, never the episode (`RESULTS.md` §2.4) |

**No hyperparameter will be tuned in response to these results.** The shared set is what
makes the later comparisons fair, and it was chosen for Phase 1's architecture and never
tuned for 2-1 either. If 2-1 needs different hyperparameters, that is a finding to report,
not a knob to turn.

### 15.2 Why these runs exist

They are **control C1 and control C2** (§2.2), and they are the denominator and the zero of
§8.1's normalized score:

> `norm(j, s) = ( R(j, s) − R_init(j) ) / ( R_spec(j) − R_init(j) )`

Nothing else in Phase 2a is interpretable until `R_spec(j)` and `R_init(j)` exist for both
tasks. Running them first is not a warm-up; it is the measurement every later number divides
by.

### 15.3 Pre-registered go/no-go: is 2-1 learnable at this budget?

`R_spec − R_init` is a **denominator**. If a task's specialist barely beats its untrained
anchor, that denominator approaches zero and every normalized score built on it becomes
unstable — a *mathematical* failure of the metric, not merely a weak result. So the gate is
declared now, in advance, and it binds:

> **GO for task *j* requires BOTH:**
> 1. the specialist beats its own untrained anchor on `mean_extrinsic_return`, **exact
>    two-sided permutation test, p < 0.05**, n = 10 vs n = 10 seeds; **and**
> 2. the effect is large enough to divide by: **Cohen's *d* ≥ 1.0** between the specialist
>    and init seed-level means.
>
> **Phase 2a's full matrix (§10) proceeds only if BOTH tasks pass.**

Threshold 2 is set at *d* ≥ 1.0 because Phase 1's own measured effects sat at |*d*| between
1.27 and 2.22 (`RESULTS.md` §13), so 1.0 is a floor in a range this project has already
demonstrated it can resolve at n = 10 — not a number invented to be passable.

**What each outcome means, committed in advance:**

- **Both pass.** The testbed is viable; the §10 matrix is worth proposing. *Proposing, not
  starting.*
- **1-1 passes, 2-1 fails.** 2-1 is not learnable at 1M steps with this architecture and
  hyperparameter set. The cross-level testbed needs a different second task, or a longer
  budget — and either is a **new decision, not an adjustment**. Report and stop.
- **Both fail.** Almost certainly a plumbing fault rather than a scientific result, given
  Phase 1 trained successfully on 1-1. Debug before drawing any conclusion.

### 15.4 Secondary, descriptive, and explicitly not a gate

Every specialist is also scored on the **other** task, filling the off-diagonal of §8.2's
performance matrix. Under §8.1's normalization this is exactly the forward-transfer term
`FWT` of §8.3: does a policy trained only on 1-1 beat an untrained one on 2-1, having never
seen it?

This is **reported descriptively and gates nothing.** It costs no extra training — the same
checkpoints, evaluated twice — and it is the first genuinely novel number this phase
produces.

### 15.5 Three things these runs will NOT tell you

Stated now so they cannot be claimed later:

- **This is not a continual-learning result.** No policy here is trained on more than one
  task. Forgetting is not measured, and cannot be: §8.3's `F` and `BWT` are undefined
  without a sequential condition.
- **`R_spec(1-1)` is NOT comparable to Phase 1's published 1-1 returns.** Phase 2a boots
  every task through PyBoy's wrapper (`start_game(world_level=…)`, §14.1) so both tasks share
  one start mechanism, whereas Phase 1 booted from power-on through `envs/boot.py`. Both are
  valid; they are not the same instrument, and no table should place their numbers side by
  side without saying so.
- **It says nothing about Kirby or about cross-title transfer.** That is Phase 2b, and
  §14.7 already raised its cost.

---

## 16. Out of scope for this document

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

Carried over from `docs/DESIGN.md` and `docs/superpowers/plans/2026-08-19-mario-ppo-reservoir.md`
and not repeated (reservoir computing, LIF dynamics, tensor-train construction, PPO, PyBoy,
Whidden's Pokémon Red work, `pokemonred_puffer`). New to this document, grouped by what they
are load-bearing for:

**Continual-RL benchmarks and protocols**

- Powers, Xing, Kolve, Mottaghi, Gupta — *CORA: Benchmarks, Baselines, and Metrics as a
  Platform for Continual Reinforcement Learning Agents*, CoLLAs 2022, arXiv:2110.10067.
  Source of §8.1's per-task normalisation discussion and §8.2's continual-evaluation
  requirement; independently reproduces CLEAR's dominance over EWC-family methods.
- Wołczyk, Zając, Pascanu, Kuciński, Miłoś — *Continual World: A Robotic Benchmark for
  Continual Reinforcement Learning*, NeurIPS 2021, arXiv:2105.10919. Source of the
  reference-run normalisation convention (§8.1), the 1M-steps-per-task budget (§7.1), the
  CW20 numbers quoted for EWC and PackNet (§2.1), and the multi-head task-conditioning
  convention rejected in §6.2.
- Schwarz, Czarnecki, Luketina, Grabska-Barwinska, Teh, Pascanu, Hadsell — *Progress &
  Compress: A scalable framework for continual learning*, ICML 2018, arXiv:1805.06370.
  Source of the six-game Atari sequence CORA inherits, and of online-EWC's 49–57% of
  single-task performance.
- Fedus, Ghosh, Martin, Bellemare, Bengio, Larochelle — *On Catastrophic Interference in
  Atari 2600 Games*, arXiv:2002.12499. Context for why weight-space regularisers are weaker
  in RL than in classification.

**Metrics — the exact definitions §8.3 pre-registers**

- Lopez-Paz & Ranzato — *Gradient Episodic Memory for Continual Learning*, NeurIPS 2017,
  arXiv:1706.08840. BWT and FWT.
- Chaudhry, Ranzato, Rohrbach, Elhoseiny — *Efficient Lifelong Learning with A-GEM*,
  ICLR 2019, arXiv:1812.00420. The forgetting measure `F` (peak-minus-final), average
  accuracy `A_T`, and LCA.
- Agarwal, Schwarzer, Castro, Courville, Bellemare — *Deep Reinforcement Learning at the
  Edge of the Statistical Precipice*, NeurIPS 2021, arXiv:2108.13264. The `rliable`
  methodology noted in §8.5 as available rather than adopted.

**Methods — what §2.1's Q4 sequences and why**

- Rolnick, Ahuja, Schwarz, Lillicrap, Wayne — *Experience Replay for Continual Learning*
  (CLEAR), NeurIPS 2019. Replay plus behavioural-cloning auxiliary losses; no extra
  parameters, no task boundaries required.
- Kirkpatrick et al. — *Overcoming catastrophic forgetting in neural networks* (EWC),
  PNAS 2017, arXiv:1612.00796.
- Mallya & Lazebnik — *PackNet: Adding Multiple Tasks to a Single Network by Iterative
  Pruning*, CVPR 2018, arXiv:1711.05769. The strongest non-replay method in Continual
  World's table, at the cost of needing task IDs at evaluation.
- Rusu et al. — *Progressive Neural Networks*, arXiv:1606.04671. Parameter cost grows
  linearly in task count; noted in §6.2's rejected alternatives by family.
- Hessel, Soyer, Espeholt, Czarnecki, Schmitt, van Hasselt — *Multi-task Deep Reinforcement
  Learning with PopArt*, AAAI 2019, arXiv:1809.04474. §6.4's named fallback for cross-task
  reward-scale normalisation.

**Task conditioning and shared action/observation spaces**

- Caccia, Mueller, Kim, Charlin, Fakoor — *Task-Agnostic Continual Reinforcement Learning:
  Gaining Insights and Overcoming Challenges*, arXiv:2205.14495. The `3RL` result
  (replay + recurrence, no task ID, matching a multi-task oracle) that makes §6.2's
  unconditioned variant the interesting one rather than the lazy one; also the 500K
  steps-per-task "resource-constrained" protocol cited in §7.1.
- Lee, Nachum, Yang et al. — *Multi-Game Decision Transformers*, NeurIPS 2022,
  arXiv:2205.15241. The 18-action union action space across 41 Atari games that §6.3's
  unmasked-union decision follows.
- Reed et al. — *A Generalist Agent* (Gato), arXiv:2205.06175. Prompt-conditioning rather
  than task IDs; cited in §6.3 for the union-representation pattern.
- Espeholt et al. — *IMPALA*, ICML 2018, arXiv:1802.01561. The DMLab-30 multi-task setup
  PopArt was built for; also the origin of §7.2's disclosed difference between mixed
  minibatches and alternating updates.
- SIMA Team (DeepMind) — *Scaling Instructable Agents Across Many Simulated Worlds*, 2024,
  arXiv:2404.10179, and the SIMA 2 technical report, 2025. Cited by `DESIGN.md` §1.1 as the
  context this phase is scoped against, and cited here **as motivation only**: SIMA is
  pixel-based, natural-language-instruction-conditioned, 3D, and built on a large pretrained
  backbone. Its "one policy, many games" framing is a genuine precedent; essentially none of
  its mechanism transfers to a 132,715-parameter GRU reading twelve RAM-derived floats, and
  this document does not pretend otherwise.

**Environment and testbed**

- Burda, Edwards, Pathak, Storkey, Darrell, Efros — *Large-Scale Study of Curiosity-Driven
  Learning*, arXiv:1808.04355. The nearest published evidence on within-game level transfer
  (NES Super Mario Bros.), used in §4.1 to calibrate how large the cross-level shift
  plausibly is.
- PyBoy (Baekalfen) — built-in game wrappers, verified directly in the installed package at
  `.venv/lib/python3.12/site-packages/pyboy/plugins/`: Super Mario Land, Kirby's Dream Land,
  Tetris, Pokémon Gen 1, Pokémon Pinball. The Super Mario Land wrapper's
  `start_game(world_level=…)` ROM patch (§4.3) and its independent agreement with
  `envs/ram_map.py`'s hand-confirmed addresses (§5.2).
- `lixado/PyBoy-RL` — DDQN agents trained on both Super Mario Land and Kirby's Dream Land via
  PyBoy; the external reference implementation cited in §5.3.
- DataCrystal (tcrf.net) RAM maps for Kirby's Dream Land and Tetris (Game Boy), and
  `pret/pokered`, referenced in §5.2–§5.3. **Every address taken from any of these is a
  hypothesis to confirm empirically, never a fact to adopt** — `envs/ram_map.py:63-66`.
