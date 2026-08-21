"""Tests for `analysis/reservoir_health.py`'s pure logic.

Scope, deliberately narrow: this file tests the PURE functions -- dead-column
detection, the dim-1 orientation assertion (§11.1), the nesting/`newly_dead`
computation, the checkpoint-selection cap, the seed-spec parser, and the
verdict-band logic for both A7 and A9 -- on small hand-built synthetic
fixtures. None of it loads a real checkpoint: those are ~30 MB each, the
report-section functions that load them are exercised instead by the module's
own `--checkpoint-dir checkpoints --seeds 0-0 --max-checkpoints-per-run 2`
smoke run (see the task's return report, not this file), and a 10-process
training matrix is live on this machine while this suite runs -- loading real
weights here would cost CPU the constraint sheet asks this suite not to spend.

Every reference value below is hand-derivable from the fixture as constructed,
not merely "the code's own answer" -- the same standard
`tests/test_aggregate_results.py` holds itself to.
"""
import os

import pytest
import torch

from analysis.reservoir_health import (
    A7_CONFIRMED_BELOW,
    A7_FALSIFIED_AT_OR_ABOVE,
    A9_CONFIRMED_BELOW,
    A9_FALSIFIED_AT_OR_ABOVE,
    AMBIGUOUS_PHRASE,
    band_verdict,
    dead_mask_from_exp_avg_sq,
    find_run,
    nesting_and_newly_dead,
    parse_seed_spec,
    select_subset,
)


# --------------------------------------------------------------------------- #
# dead-column detection + dim-1 (column, not row) orientation
# --------------------------------------------------------------------------- #

class TestDeadMaskFromExpAvgSq:
    def test_known_zero_columns_are_dead(self):
        # d_model=4, reservoir_units=6. Columns 1 and 4 are ALL-ZERO across every
        # row -> dead. Every other column has at least one nonzero entry.
        exp_avg_sq = torch.tensor([
            [1.0, 0.0, 2.0, 3.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 9.0, 0.0, 0.0, 0.0],
        ])
        mask = dead_mask_from_exp_avg_sq(exp_avg_sq)
        assert mask.shape == (6,)
        assert mask.tolist() == [False, True, False, False, True, False]

    def test_all_zero_is_all_dead(self):
        exp_avg_sq = torch.zeros(3, 10)
        mask = dead_mask_from_exp_avg_sq(exp_avg_sq)
        assert mask.all()
        assert int(mask.sum()) == 10

    def test_no_zero_columns_is_none_dead(self):
        exp_avg_sq = torch.ones(3, 10)
        mask = dead_mask_from_exp_avg_sq(exp_avg_sq)
        assert not mask.any()

    def test_a_single_nonzero_entry_saves_the_whole_column(self):
        # Column 2 has exactly one nonzero entry (row 0) among 4 rows -- that is
        # enough for the column to be ALIVE, because "dead" requires EVERY entry
        # in the column to be exactly 0, not merely most of them.
        exp_avg_sq = torch.zeros(4, 5)
        exp_avg_sq[0, 2] = 1e-30
        mask = dead_mask_from_exp_avg_sq(exp_avg_sq)
        assert mask.tolist() == [True, True, False, True, True]

    def test_orientation_dim1_is_columns_not_rows(self):
        """The load-bearing case: a tensor with an entirely-zero ROW (dim 0) but
        NO entirely-zero COLUMN (dim 1). Reducing over the wrong axis (dim 1,
        treating dim 0 as the unit index) would report 1 dead unit here; the
        correct dim-1 reduction (§11.1: a unit indexes a COLUMN) must report 0,
        because every column has at least one nonzero entry from the other rows.
        """
        d_model, reservoir_units = 3, 5
        exp_avg_sq = torch.ones(d_model, reservoir_units)
        exp_avg_sq[1, :] = 0.0  # row 1 entirely zero; rows 0 and 2 are all-ones
        mask = dead_mask_from_exp_avg_sq(exp_avg_sq)
        assert mask.shape == (reservoir_units,)
        assert not mask.any(), (
            "a zeroed ROW must not be mistaken for a dead COLUMN -- this is "
            "exactly the backwards-axis bug §11.1 warns getting invalidates A7"
        )

    def test_backwards_shape_raises(self):
        # reservoir_units=8192, d_model=16 passed with axes swapped (8192, 16):
        # d_model is no longer « reservoir_units, so this must fail loudly
        # rather than silently compute a wrong-but-plausible number.
        exp_avg_sq = torch.zeros(8192, 16)
        with pytest.raises(AssertionError):
            dead_mask_from_exp_avg_sq(exp_avg_sq)

    def test_non_2d_raises(self):
        with pytest.raises(AssertionError):
            dead_mask_from_exp_avg_sq(torch.zeros(16, 8192, 1))


# --------------------------------------------------------------------------- #
# nesting / newly_dead
# --------------------------------------------------------------------------- #

class TestNestingAndNewlyDead:
    def _mask(self, dead_indices, n=6):
        m = torch.zeros(n, dtype=torch.bool)
        for i in dead_indices:
            m[i] = True
        return m

    def test_strictly_shrinking_sequence_holds_nesting(self):
        # dead(t) shrinks monotonically: {0,1,2,3} -> {0,1} -> {0} -> {} --
        # exactly A4a's "dead set only ever SHRINKS" result. newly_dead must be
        # 0 at every one of the 3 transitions.
        seq = [self._mask([0, 1, 2, 3]), self._mask([0, 1]),
              self._mask([0]), self._mask([])]
        newly_dead, holds = nesting_and_newly_dead(seq)
        assert newly_dead == [0, 0, 0]
        assert holds is True

    def test_constant_sequence_holds_nesting(self):
        seq = [self._mask([0, 2]), self._mask([0, 2]), self._mask([0, 2])]
        newly_dead, holds = nesting_and_newly_dead(seq)
        assert newly_dead == [0, 0]
        assert holds is True

    def test_a_column_dying_mid_run_violates_nesting(self):
        # {0} -> {0, 3}: column 3 died AFTER the first checkpoint, which A4a's
        # nesting result says never happens. newly_dead at that transition must
        # be exactly 1, and nesting must NOT hold.
        seq = [self._mask([0]), self._mask([0, 3]), self._mask([0, 3])]
        newly_dead, holds = nesting_and_newly_dead(seq)
        assert newly_dead == [1, 0]
        assert holds is False

    def test_newly_dead_count_is_exact_not_just_nonzero(self):
        seq = [self._mask([]), self._mask([1, 2, 4])]
        newly_dead, holds = nesting_and_newly_dead(seq)
        assert newly_dead == [3]
        assert holds is False

    def test_single_checkpoint_has_no_transitions(self):
        newly_dead, holds = nesting_and_newly_dead([self._mask([0, 1])])
        assert newly_dead == []
        assert holds is True  # vacuously -- all() of an empty list is True


# --------------------------------------------------------------------------- #
# verdict-band logic, both directions, both boundaries, and the ambiguous band
# --------------------------------------------------------------------------- #

class TestBandVerdictA7:
    """A7: confirmed < 2% (164/8192), falsified >= 5% (410/8192)."""

    def test_confirmed_well_below(self):
        assert band_verdict(0.01, A7_CONFIRMED_BELOW, A7_FALSIFIED_AT_OR_ABOVE) == "CONFIRMED"

    def test_confirmed_boundary_just_under_2pct(self):
        assert band_verdict(0.0199, A7_CONFIRMED_BELOW,
                            A7_FALSIFIED_AT_OR_ABOVE) == "CONFIRMED"

    def test_exactly_2pct_is_not_confirmed(self):
        # confirmed requires value < 2%, strictly -- 2% itself is ambiguous.
        assert band_verdict(0.02, A7_CONFIRMED_BELOW, A7_FALSIFIED_AT_OR_ABOVE) == "AMBIGUOUS"

    def test_ambiguous_midband(self):
        assert band_verdict(0.035, A7_CONFIRMED_BELOW, A7_FALSIFIED_AT_OR_ABOVE) == "AMBIGUOUS"

    def test_exactly_5pct_is_falsified(self):
        # falsified is "at or above" 5% -- the boundary itself falsifies.
        assert band_verdict(0.05, A7_CONFIRMED_BELOW, A7_FALSIFIED_AT_OR_ABOVE) == "FALSIFIED"

    def test_falsified_well_above(self):
        assert band_verdict(0.106, A7_CONFIRMED_BELOW, A7_FALSIFIED_AT_OR_ABOVE) == "FALSIFIED"


class TestBandVerdictA9:
    """A9: confirmed < 40%, falsified >= 46%."""

    def test_confirmed_below_40pct(self):
        assert band_verdict(0.35, A9_CONFIRMED_BELOW, A9_FALSIFIED_AT_OR_ABOVE) == "CONFIRMED"

    def test_exactly_40pct_is_ambiguous(self):
        assert band_verdict(0.40, A9_CONFIRMED_BELOW, A9_FALSIFIED_AT_OR_ABOVE) == "AMBIGUOUS"

    def test_ambiguous_midband(self):
        assert band_verdict(0.43, A9_CONFIRMED_BELOW, A9_FALSIFIED_AT_OR_ABOVE) == "AMBIGUOUS"

    def test_exactly_46pct_is_falsified(self):
        assert band_verdict(0.46, A9_CONFIRMED_BELOW, A9_FALSIFIED_AT_OR_ABOVE) == "FALSIFIED"

    def test_legacy_figure_sits_in_the_ambiguous_band(self):
        # 45.42% (§12's legacy silent fraction) sits inside [40%, 46%) --
        # ambiguous, not an outright falsification, even though it is legacy's
        # own measured value: the band exists precisely so a v2 result this
        # close to legacy is not spun either way.
        assert band_verdict(0.4542, A9_CONFIRMED_BELOW, A9_FALSIFIED_AT_OR_ABOVE) == "AMBIGUOUS"

    def test_ambiguous_phrase_is_the_pre_registered_wording(self):
        # Both pre-registrations require the ambiguous band be "reported in
        # exactly those words" -- pin the literal string so a rewording is caught.
        assert AMBIGUOUS_PHRASE == "confirms the direction while falsifying the magnitude"


# --------------------------------------------------------------------------- #
# checkpoint-selection cap (--max-checkpoints-per-run)
# --------------------------------------------------------------------------- #

class TestSelectSubset:
    def test_no_cap_returns_everything(self):
        items = list(range(10))
        assert select_subset(items, None) == items

    def test_cap_larger_than_list_returns_everything(self):
        items = list(range(5))
        assert select_subset(items, 100) == items

    def test_cap_of_one_keeps_only_last(self):
        items = list(range(10))
        assert select_subset(items, 1) == [9]

    def test_cap_of_two_keeps_first_and_last(self):
        items = list(range(10))
        assert select_subset(items, 2) == [0, 9]

    def test_last_item_always_present_under_any_cap(self):
        # cap=1 can only keep ONE item, and it must be the last (the "final
        # checkpoint" both A7 and A9's verdicts are computed from) -- "first
        # AND last" is only achievable once cap >= 2.
        items = list(range(17))
        for cap in range(1, len(items) + 1):
            selected = select_subset(items, cap)
            assert selected[-1] == items[-1]
            if cap >= 2:
                assert selected[0] == items[0]
            assert len(selected) <= cap


# --------------------------------------------------------------------------- #
# --seeds parsing
# --------------------------------------------------------------------------- #

class TestParseSeedSpec:
    def test_range(self):
        assert parse_seed_spec("0-9") == list(range(10))

    def test_single_range_of_one(self):
        assert parse_seed_spec("0-0") == [0]

    def test_comma_list(self):
        assert parse_seed_spec("0,2,5") == [0, 2, 5]

    def test_mixed_ranges_and_singles(self):
        assert parse_seed_spec("0,2-4,7") == [0, 2, 3, 4, 7]


# --------------------------------------------------------------------------- #
# graceful-skip behaviour on missing/incomplete run directories
# --------------------------------------------------------------------------- #

class TestFindRunGracefulSkip:
    def test_missing_directory_reports_status_not_exception(self, tmp_path):
        status, d, ckpts = find_run(str(tmp_path), "reservoir", 0)
        assert status == "missing_dir"
        assert ckpts == []
        assert not os.path.isdir(d)

    def test_directory_with_no_checkpoints_reports_status_not_exception(self, tmp_path):
        run_directory = tmp_path / "reservoir_seed0"
        run_directory.mkdir()
        (run_directory / "train_log.jsonl").write_text('{"update": 1}\n')
        (run_directory / "launcher.log").write_text("started\n")
        status, d, ckpts = find_run(str(tmp_path), "reservoir", 0)
        assert status == "no_checkpoints"
        assert ckpts == []
        assert os.path.isdir(d)

    def test_directory_with_checkpoints_is_ok_and_sorted_by_step(self, tmp_path):
        run_directory = tmp_path / "reservoir_seed0"
        run_directory.mkdir()
        for step in (300288, 100096, 200192):
            (run_directory / f"step_{step}.pt").write_bytes(b"")
        status, d, ckpts = find_run(str(tmp_path), "reservoir", 0)
        assert status == "ok"
        assert [s for s, _ in ckpts] == [100096, 200192, 300288]

    def test_non_matching_filenames_are_ignored(self, tmp_path):
        run_directory = tmp_path / "reservoir_seed0"
        run_directory.mkdir()
        (run_directory / "step_100096.pt").write_bytes(b"")
        (run_directory / "train_log.jsonl").write_text("")
        (run_directory / "launcher.log").write_text("")
        (run_directory / "not_a_checkpoint.pt").write_bytes(b"")
        status, d, ckpts = find_run(str(tmp_path), "reservoir", 0)
        assert status == "ok"
        assert len(ckpts) == 1
        assert ckpts[0][0] == 100096

    def test_different_seeds_and_arms_do_not_collide(self, tmp_path):
        (tmp_path / "reservoir_seed0").mkdir()
        (tmp_path / "reservoir_seed0" / "step_100096.pt").write_bytes(b"")
        # seed 1 has no directory at all; baseline_seed0 is a different arm.
        status1, _, _ = find_run(str(tmp_path), "reservoir", 1)
        status_baseline, _, _ = find_run(str(tmp_path), "baseline", 0)
        status0, _, ckpts0 = find_run(str(tmp_path), "reservoir", 0)
        assert status1 == "missing_dir"
        assert status_baseline == "missing_dir"
        assert status0 == "ok"
        assert len(ckpts0) == 1
