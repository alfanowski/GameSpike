# spiking-reservoir-rl

A frozen spiking-reservoir reinforcement-learning agent for Super Mario Land
(Game Boy), evaluated against a matched-trainable-parameter GRU baseline under
a mandatory scientific control. Sibling project to
[`spiking-reservoir-lm`](https://github.com/alfanowski/spiking-reservoir-lm)
(frozen-reservoir byte-level text generation) and an unpublished biosignal
(ECG/EEG/EMG) design that share the same underlying reservoir-computing core —
reused here, unmodified, applied to real-time game control instead of language
or biosignal interpretation.

---

## Status: pre-implementation (2026-08-19)

**This repository currently contains a design document and a detailed,
task-by-task implementation plan — no model or training code has landed yet.**
That is stated here directly rather than left implicit, in keeping with this
project family's practice of reporting real status rather than aspirational
status (see `spiking-reservoir-lm`'s own README/PAPER.md for that precedent).
A results write-up (this project's equivalent of `PAPER.md`) will be added once
Phase 1's core comparison — frozen reservoir vs. trained-GRU baseline — actually
runs and produces real numbers, not before.

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
- **Is not:** a claim of state-of-the-art game-playing performance, a
  general-purpose game-playing agent, or an RPG/strategy agent — Pokémon-style
  targets were explicitly considered and rejected for this phase (see
  `docs/DESIGN.md` §1) for the same structural reason the LM project could not
  compete on open-domain generation: this architecture has no mechanism for
  storing broad accumulated knowledge, only for reacting to a state stream.
- **Every result, positive or negative, will be reported as such.** The
  mandatory-control design (a matched-parameter trained GRU baseline) exists
  specifically so a negative result — the reservoir failing to beat a
  conventional trained recurrent policy at the same parameter budget — is
  scientifically informative rather than a silently discarded run.

## Repository structure (as specified by the implementation plan)

```
spiking-reservoir-rl/
├── docs/
│   ├── DESIGN.md                 # full design rationale
│   └── superpowers/plans/        # implementation plan(s)
├── envs/                         # PyBoy wrapper, RAM-address map, action space
├── models/                       # vendored frozen reservoir + policy-value models
├── training/                     # PPO, rollout collection, novelty gate, checkpointing
├── tests/                        # pytest suite (TDD — written alongside each task)
└── checkpoints/                  # trained checkpoints (gitignored, not distributed)
```

## Setup

```bash
git clone https://github.com/alfanowski/spiking-reservoir-rl.git
cd spiking-reservoir-rl
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**You must supply your own legally-dumped Super Mario Land ROM** — it is never
committed to this repository (`.gitignore` excludes `*.gb`/`*.gbc`). Set
`MARIO_LAND_ROM_PATH` to its path before running the test suite or training
scripts; tests that require it skip cleanly when it is unset.

## Running the test suite

```bash
python -m pytest tests/ -q
```

## License

[Apache License 2.0](LICENSE).

## Citation

```bibtex
@misc{alfano_spiking_reservoir_rl,
  author = {Andrea Alfano (Alfanowski)},
  title  = {spiking-reservoir-rl: A Frozen Tensor-Train Spiking Reservoir as a
            Real-Time Feature Extractor for Reinforcement Learning, Evaluated
            Against a Matched-Parameter Trained Baseline},
  year   = {2026},
  note   = {See docs/DESIGN.md for the full design rationale. A results
            write-up will be added once Phase 1 produces real data.},
  howpublished = {\url{https://github.com/alfanowski/spiking-reservoir-rl}}
}
```
