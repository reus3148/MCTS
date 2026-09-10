"""Tests for the GENIE BPC movetext helpers (v1.6)."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.genie.sequences import (  # noqa: E402
    annotate_regimens,
    primary_channel,
    regimen_channels,
    response_switch_table,
    sequence_lengths,
    switch_rates,
)


class RegimenChannelTests(unittest.TestCase):
    def test_trailing_spaces_in_release_names_are_tolerated(self):
        # "Tamoxifen " and "Docetaxel " really do carry a trailing space in the
        # v1.0-public release; a strict lookup would file both as "other".
        self.assertEqual(regimen_channels("Tamoxifen "), ("endocrine",))
        self.assertEqual(regimen_channels("Docetaxel "), ("chemotherapy",))

    def test_multi_drug_regimen_reports_every_channel_once(self):
        channels = regimen_channels("Letrozole, Palbociclib")
        self.assertEqual(channels, ("endocrine", "targeted_other"))

    def test_duplicate_drugs_collapse(self):
        self.assertEqual(
            regimen_channels("Investigational Drug, Investigational Drug"),
            ("investigational",))

    def test_unknown_drug_falls_to_other_rather_than_being_guessed(self):
        self.assertEqual(regimen_channels("Nonexistent Agent"), ("other",))

    def test_missing_regimen_drugs_is_empty(self):
        self.assertEqual(regimen_channels(float("nan")), ())
        self.assertEqual(regimen_channels("   "), ())


class PrimaryChannelTests(unittest.TestCase):
    def test_investigational_outranks_everything(self):
        # A blinded agent makes the regimen unclassifiable. If precedence went
        # the other way, "Investigational Drug, Letrozole" would be counted as
        # first-line endocrine therapy, which is exactly the mistake that would
        # corrupt the standard-treatment benefit estimate.
        self.assertEqual(
            primary_channel(("endocrine", "investigational")),
            "investigational")

    def test_chemo_outranks_endocrine_in_a_combined_regimen(self):
        self.assertEqual(
            primary_channel(("chemotherapy", "endocrine")), "chemotherapy")

    def test_empty_channels_are_labelled(self):
        self.assertEqual(primary_channel(()), "unclassified")


class AnnotateTests(unittest.TestCase):
    def test_endocrine_only_is_strict(self):
        frame = pd.DataFrame({"regimen_drugs": [
            "Letrozole", "Letrozole, Palbociclib", "Tamoxifen "]})
        annotated = annotate_regimens(frame)
        self.assertEqual(
            list(annotated["endocrine_only"]), [True, False, True])


class SequenceLengthTests(unittest.TestCase):
    def setUp(self):
        self.regimens = pd.DataFrame({
            "record_id": ["A", "A", "A", "B"],
            "ca_seq": [0, 0, 0, 0],
            "dx_reg_start_int": [10, 400, 3000, 40],
        })

    def test_counts_only_regimens_inside_the_horizon(self):
        lengths = sequence_lengths(self.regimens, horizon_years=5.0)
        self.assertEqual(lengths[("A", 0)], 2)
        self.assertEqual(lengths[("B", 0)], 1)

    def test_cancers_with_no_regimen_inside_the_horizon_are_absent(self):
        lengths = sequence_lengths(self.regimens, horizon_years=0.02)  # ~7 days
        self.assertNotIn(("A", 0), lengths.index)
        self.assertNotIn(("B", 0), lengths.index)

    def test_non_positive_horizon_is_rejected(self):
        with self.assertRaises(ValueError):
            sequence_lengths(self.regimens, horizon_years=0.0)


class ResponseSwitchTests(unittest.TestCase):
    def setUp(self):
        # Patient A: progressing scan on day 200, new regimen on day 240.
        # Patient B: stable scan on day 200, next regimen not until day 900.
        self.regimens = pd.DataFrame({
            "record_id": ["A", "A", "B", "B"],
            "dx_reg_start_int": [100.0, 240.0, 100.0, 900.0],
        })
        self.imaging = pd.DataFrame({
            "record_id": ["A", "B"],
            "scan_number": [1, 1],
            "dx_scan_days": [200.0, 200.0],
            "image_scan_type": ["CT", "CT"],
            "image_ca": ["Yes", "Yes"],
            "image_overall": [
                "Progressing/Worsening/Enlarging", "Stable/No change"],
        })
        self.followup = pd.Series({"A": 2000.0, "B": 2000.0})

    def test_forward_window_detects_the_switch(self):
        table = response_switch_table(
            self.imaging, self.regimens, self.followup, window_days=90)
        by_patient = table.set_index("record_id")
        self.assertTrue(by_patient.loc["A", "switch_after"])
        self.assertFalse(by_patient.loc["B", "switch_after"])

    def test_backward_control_is_not_triggered_by_the_forward_switch(self):
        table = response_switch_table(
            self.imaging, self.regimens, self.followup, window_days=90)
        self.assertFalse(table["switch_before"].any())

    def test_scan_before_first_regimen_is_dropped(self):
        imaging = self.imaging.copy()
        imaging.loc[0, "dx_scan_days"] = 50.0  # before A's first regimen
        table = response_switch_table(
            imaging, self.regimens, self.followup, window_days=90)
        self.assertNotIn("A", set(table["record_id"]))

    def test_scan_without_enough_followup_is_dropped(self):
        # Without this, a scan 10 days before the last recorded contact would
        # score as "no switch" for a purely administrative reason.
        followup = pd.Series({"A": 250.0, "B": 2000.0})
        table = response_switch_table(
            self.imaging, self.regimens, followup, window_days=90)
        self.assertNotIn("A", set(table["record_id"]))
        self.assertIn("B", set(table["record_id"]))

    def test_scan_with_no_overall_assessment_is_dropped(self):
        imaging = self.imaging.copy()
        imaging.loc[1, "image_overall"] = None
        table = response_switch_table(
            imaging, self.regimens, self.followup, window_days=90)
        self.assertNotIn("B", set(table["record_id"]))

    def test_other_assessments_are_kept_but_grouped_separately(self):
        imaging = self.imaging.copy()
        imaging.loc[1, "image_overall"] = "Mixed"
        table = response_switch_table(
            imaging, self.regimens, self.followup, window_days=90)
        self.assertEqual(
            table.set_index("record_id").loc["B", "group"], "other")

    def test_non_positive_window_is_rejected(self):
        with self.assertRaises(ValueError):
            response_switch_table(
                self.imaging, self.regimens, self.followup, window_days=0)

    def test_switch_rates_report_counts_alongside_rates(self):
        table = response_switch_table(
            self.imaging, self.regimens, self.followup, window_days=90)
        rates = switch_rates(table).set_index("group")
        self.assertEqual(int(rates.loc["progressing", "scans"]), 1)
        self.assertEqual(rates.loc["progressing", "switch_after"], 1.0)
        self.assertEqual(rates.loc["controlled", "switch_after"], 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
