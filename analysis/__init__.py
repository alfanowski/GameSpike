"""Post-hoc statistical analysis of GameSpike evaluation results.

Turns the per-checkpoint JSON files `training/evaluate.py --json` produces into
the arm-vs-arm comparison design doc §5 (`docs/DESIGN.md`) demands: does the
frozen reservoir actually beat the matched-parameter trained GRU baseline, or
not. See `aggregate_results.py`'s module docstring for the full statistical
rationale (why stdlib+numpy only, why the training seed -- not the episode --
is the unit of analysis, why the CLI never prints a bare verdict).
"""
