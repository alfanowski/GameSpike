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

## Roadmap: Phase 1 of a larger, stated goal

**Stated explicitly, so it never reads as scope-narrowing by omission:** the actual goal of
this project is a general game-playing agent across Nintendo handheld titles — explicitly
including Game Boy Advance and **Pokémon Fire Red** — not a Mario-Land-only system. The
single-game scope below is a deliberate *sequencing* choice, the same "prove one case
scientifically before generalizing" discipline already used in the sibling
[`spiking-reservoir-lm`](https://github.com/alfanowski/spiking-reservoir-lm) project, not a
redefinition of the goal down to one game. Full version:
[`docs/DESIGN.md` §1.1](docs/DESIGN.md#11-roadmap-this-document-is-phase-1-of-a-stated-larger-goal--not-the-destination).

- **Phase 1 (this repository, currently running).** Super Mario Land, Game Boy: frozen
  reservoir vs. matched-parameter trained-GRU baseline, under a mandatory scientific
  control. Answers whether a frozen reservoir helps at all on real game control at a real
  matched parameter budget — without that answered honestly first, any later claim about
  generalizing across games would rest on an unverified premise.
- **Phase 2 (not started).** Multi-game generalization *within* Game Boy / Game Boy Color —
  the same architecture trained/tested across more than one title. This is genuinely open
  research — continual learning and catastrophic forgetting across titles — not a solved
  problem this project gets to assume away. See DeepMind's SIMA 1/2 and the continual-RL
  literature (Unicorn, DisCoRL, CORA) for the context this is scoped against.
- **Phase 3 (Game Boy Advance as a platform).** `pygba`/mGBA's direct Python bindings are
  dead on Apple Silicon — a hard native `SIGBUS`, root-caused to a cffi ABI-mode fragility
  on strict-alignment ARM64. A **Lua-scripting + local-socket bridge to `mgba-headless`**
  was built and verified end-to-end instead: 12/12 checks passed, deterministic
  byte-identical replays, ~3,062 fps single instance and ~15k fps aggregate across 8
  parallel instances, with real gameplay driven through Super Mario Advance's World 1-1.
  This is a **proven scratchpad spike, not yet a component of this repository** —
  formalizing it into a real `gymnasium.Env` is Phase 3's remaining work.
- **Phase 4 (Pokémon Fire Red / RPG targets).** Needs Phase 3's platform work *plus* a
  hierarchical planning layer that does not exist yet in this architecture — an RPG's
  strategic/inventory/long-horizon demands cannot be represented by a frozen reservoir plus
  a small trained readout alone, which was the original and still-correct reason RPGs were
  rejected as the Phase 1 target (see below). The available Fire Red ROM is also the
  **Italian release** (cartridge code `BPRI`), not the US `BPRE` that published community
  RAM maps target, so its RAM addresses will need the same from-scratch empirical
  confirmation Super Mario Land's did.

## Status: pipeline complete, Phase 1 experiment running (2026-08-20)

Stated as precisely as possible, in keeping with this project family's practice
of reporting real status rather than aspirational status (see
`spiking-reservoir-lm`'s own README/PAPER.md for that precedent).

**What exists.** All 12 tasks of the Phase 0 + Phase 1 implementation plan are
complete: the PyBoy environment wrapper with an empirically-confirmed Super Mario
Land RAM map, the discrete action space, both competing policy-value models at a
verified-matched trainable-parameter budget, the trajectory-novelty curiosity
gate, the PPO core, rollout collection, the training loop, and the evaluation
harness. Both entry points run end-to-end against a real ROM: `training/train.py`
collects rollouts, applies real gradient updates and writes checkpoints, and
`training/evaluate.py` loads a checkpoint, plays it and reports per-episode
statistics with spread. A 169-test suite covers it (including the results-aggregation
module in `analysis/`, added alongside this update).

**What does not exist yet.** No trained checkpoints, and no results. **The Phase 1
comparison (frozen reservoir vs. matched-parameter trained GRU) is now running**: 2 arms
× 10 independently-seeded training runs each × 1,000,000 env steps per run, per
`training/evaluate.py`'s own documented requirement that an honest comparison needs
several independently-trained seeds per arm, not one checkpoint each — plus a set of
untrained (`--steps 0`) reference checkpoints per arm/seed, kept as an experimental
control. Until it completes, this repository contains no evidence either way about the
question it exists to answer, and **no number in it should be quoted as one, in either
direction**. A results write-up (this project's equivalent of `PAPER.md`) will be added
once the comparison finishes and produces real numbers, not before.

- [`docs/DESIGN.md`](docs/DESIGN.md) — full design rationale: why a reactive
  platformer (not an RPG) was chosen as the first target, the architecture,
  the mandatory scientific control, and what is explicitly out of scope.
- [`docs/superpowers/plans/2026-08-19-mario-ppo-reservoir.md`](docs/superpowers/plans/2026-08-19-mario-ppo-reservoir.md) —
  the implementation plan (Phase 0 + Phase 1 only; later phases are separate,
  not-yet-written plans per the design doc's own phased build order).

## What this project is / is not

- **Is:** a test of whether a frozen, never-trained spiking reservoir —
  already shown (in the sibling LM project) to contribute nothing to storable
  knowledge, and therefore structurally unsuited to open-domain text
  generation — is nonetheless a useful real-time feature extractor for a
  bounded, reactive control task, where a frozen reservoir's actual strength
  (rich nonlinear dynamics obtained "for free," no gradient descent through the
  recurrent core) is a plausible structural fit.
- **Is not, yet:** a claim of state-of-the-art game-playing performance, a
  general-purpose game-playing agent, or an RPG/strategy agent. Today this repository is a
  single-game, single-architecture experiment — that is what exists right now, not a
  permanent ceiling. Pokémon-style targets were explicitly considered and rejected as the
  *first* target (see `docs/DESIGN.md` §1) for the same structural reason the LM project
  could not compete on open-domain generation: this architecture has no mechanism for
  storing broad accumulated knowledge, only for reacting to a state stream — a genuinely
  permanent property of a frozen reservoir plus a small trained readout, not something more
  training steps fixes. That is exactly why the roadmap above sequences an RPG target
  (Phase 4) behind a hierarchical planning layer that has to be designed and added on top,
  rather than assuming this architecture would get there alone.
- **Every result, positive or negative, will be reported as such.** The
  mandatory-control design (a matched-parameter trained GRU baseline) exists
  specifically so a negative result — the reservoir failing to beat a
  conventional trained recurrent policy at the same parameter budget — is
  scientifically informative rather than a silently discarded run.

## Repository structure

```
GameSpike/
├── docs/
│   ├── DESIGN.md                 # full design rationale
│   └── superpowers/plans/        # implementation plan(s)
├── envs/                         # PyBoy wrapper, RAM-address map, action space
├── models/                       # vendored frozen reservoir + policy-value models
├── training/                     # PPO, rollout collection, novelty gate, train/evaluate
├── tests/                        # pytest suite (TDD — written alongside each task)
└── checkpoints/                  # run outputs (gitignored, not distributed)
```

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
needs several independently-trained seeds per arm (see below), not one run each.

Other flags: `--rollout-len` (truncated-BPTT window, default 128), `--checkpoint-every`
(default 10000 steps; a final checkpoint is always written regardless),
`--checkpoint-dir`, `--resume-from PATH`, `--n-envs` (accepted, currently unused —
collection is single-process).

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
an arm comparison at all** (training-seed variance usually dominates and this cannot
see it), and by default the policy is scored over a whole continuous episode while
training reset its recurrent state every `--rollout-len` steps —
`--state-reset-interval 128` runs the matched-regime counterpart.

## Running the test suite

```bash
python -m pytest tests/ -q
```

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
  note   = {See docs/DESIGN.md for the full design rationale. A results
            write-up will be added once Phase 1 produces real data.},
  howpublished = {\url{https://github.com/alfanowski/GameSpike}}
}
```
