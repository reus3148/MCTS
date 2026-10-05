"""Tests for the MSK-CHORD adapter (v2.2).

Every test builds its own tiny frame, so the suite runs without the release
present - it is CC BY-NC-ND 4.0 and git-ignored.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.msk.adaptation import (  # noqa: E402
    assessment_group,
    decision_points,
    ecog_encoding_check,
    ecog_trajectory,
    followup_days,
    opportunity_summary,
    switch_rates,
    switch_table,
)
from analysis.msk.axes import (  # noqa: E402
    AXIS_RESOLUTION,
    DAYS_PER_YEAR,
    axis_coverage,
    chemo_agent_counts,
    endocrine_class,
    endocrine_duration,
    timing_intervals,
    timing_table,
    treatment_channels,
    within_horizon,
)
from analysis.msk.loader import anchor_at_diagnosis, diagnosis_anchor  # noqa: E402


def treatment_frame(rows):
    """``(patient, start, stop, subtype, agent)`` -> a treatment timeline."""
    return pd.DataFrame(
        [{"PATIENT_ID": p, "dx_start_date": a, "dx_stop_date": b,
          "SUBTYPE": s, "AGENT": g} for p, a, b, s, g in rows])


def progression_frame(rows):
    """``(patient, day, call)`` -> a progression timeline."""
    return pd.DataFrame(
        [{"PATIENT_ID": p, "dx_start_date": d, "PROGRESSION": c,
          "PROCEDURE_TYPE": "CT"} for p, d, c in rows])


class AnchorTests(unittest.TestCase):
    def setUp(self):
        self.diagnosis = pd.DataFrame([
            {"PATIENT_ID": "A", "SUBTYPE": "Primary", "START_DATE": -400},
            {"PATIENT_ID": "A", "SUBTYPE": "Primary", "START_DATE": -100},
            {"PATIENT_ID": "B", "SUBTYPE": "Primary", "START_DATE": -50},
        ])

    def test_earliest_primary_wins(self):
        anchor = diagnosis_anchor(self.diagnosis)
        self.assertEqual(anchor["A"], -400)
        self.assertEqual(anchor["B"], -50)

    def test_non_primary_rows_are_ignored(self):
        mixed = pd.concat([self.diagnosis, pd.DataFrame([
            {"PATIENT_ID": "A", "SUBTYPE": "Metastasis", "START_DATE": -900}])])
        self.assertEqual(diagnosis_anchor(mixed)["A"], -400)

    def test_rebasing_shifts_both_date_columns(self):
        anchor = diagnosis_anchor(self.diagnosis)
        frame = pd.DataFrame([
            {"PATIENT_ID": "A", "START_DATE": 0, "STOP_DATE": 30}])
        out = anchor_at_diagnosis(frame, anchor)
        # Sequencing day 0 is 400 days after A's diagnosis.
        self.assertEqual(out["dx_start_date"].iloc[0], 400)
        self.assertEqual(out["dx_stop_date"].iloc[0], 430)

    def test_patients_without_an_anchor_are_dropped(self):
        anchor = diagnosis_anchor(self.diagnosis)
        frame = pd.DataFrame([
            {"PATIENT_ID": "A", "START_DATE": 0, "STOP_DATE": 1},
            {"PATIENT_ID": "Z", "START_DATE": 0, "STOP_DATE": 1}])
        out = anchor_at_diagnosis(frame, anchor)
        self.assertEqual(list(out["PATIENT_ID"]), ["A"])


class HorizonTests(unittest.TestCase):
    def test_negative_days_are_excluded(self):
        frame = treatment_frame([
            ("A", -10, 0, "Chemo", "X"), ("A", 10, 20, "Chemo", "X")])
        self.assertEqual(len(within_horizon(frame, 5.0)), 1)

    def test_upper_bound_is_inclusive(self):
        limit = 5.0 * DAYS_PER_YEAR
        frame = treatment_frame([("A", limit, limit, "Chemo", "X")])
        self.assertEqual(len(within_horizon(frame, 5.0)), 1)

    def test_rejects_non_positive_horizon(self):
        with self.assertRaises(ValueError):
            within_horizon(treatment_frame([("A", 1, 2, "Chemo", "X")]), 0)


class ChannelTests(unittest.TestCase):
    def test_supportive_care_is_not_a_treatment_channel(self):
        frame = treatment_frame([("A", 1, 2, "Bone Treatment", "ZOLEDRONIC ACID")])
        self.assertEqual(treatment_channels(frame)["channel"].iloc[0], "supportive")

    def test_unknown_subtype_falls_to_other(self):
        frame = treatment_frame([("A", 1, 2, "Nonsense", "X")])
        self.assertEqual(treatment_channels(frame)["channel"].iloc[0], "other")

    def test_endocrine_classes(self):
        self.assertEqual(endocrine_class("TAMOXIFEN"), "serm")
        self.assertEqual(endocrine_class("letrozole"), "aromatase_inhibitor")
        self.assertEqual(endocrine_class("FULVESTRANT"), "serd")
        self.assertEqual(endocrine_class(float("nan")), "unclassified")


class AxisCoverageTests(unittest.TestCase):
    def setUp(self):
        self.treatment = treatment_frame([
            ("A", 10, 100, "Chemo", "PACLITAXEL"),
            ("A", 200, 2000, "Hormone", "LETROZOLE"),
            ("B", 10, 100, "Hormone", "TAMOXIFEN"),
        ])
        self.surgery = pd.DataFrame([
            {"PATIENT_ID": "A", "dx_start_date": 150, "SUBTYPE": "PROCEDURE"},
            {"PATIENT_ID": "B", "dx_start_date": 20, "SUBTYPE": "SAMPLE"},
        ])
        self.radiation = pd.DataFrame([
            {"PATIENT_ID": "A", "dx_start_date": 300}])
        self.cohort = {"A", "B", "C"}

    def test_shares_use_the_whole_cohort_as_denominator(self):
        coverage = axis_coverage(
            self.treatment, self.surgery, self.radiation, self.cohort).set_index("axis")
        # C has no recorded treatment and must still be in the denominator.
        self.assertEqual(coverage.loc["chemo", "cohort"], 3)
        self.assertAlmostEqual(coverage.loc["chemo", "share"], 1 / 3)
        self.assertAlmostEqual(coverage.loc["endocrine", "share"], 2 / 3)

    def test_sample_rows_are_not_operations(self):
        coverage = axis_coverage(
            self.treatment, self.surgery, self.radiation, self.cohort).set_index("axis")
        self.assertEqual(coverage.loc["surgery", "patients"], 1)

    def test_timing_needs_both_dates(self):
        coverage = axis_coverage(
            self.treatment, self.surgery, self.radiation, self.cohort).set_index("axis")
        # B has chemo? No - B has hormone only, and a SAMPLE row. Only A qualifies.
        self.assertEqual(coverage.loc["timing", "patients"], 1)

    def test_every_axis_declares_its_level_resolution(self):
        self.assertEqual(
            set(AXIS_RESOLUTION),
            {"timing", "surgery", "chemo", "endocrine", "radiation"})
        for axis in AXIS_RESOLUTION.values():
            self.assertIn(axis["level"], (True, False, "partial"))
            self.assertTrue(axis["basis"])

    def test_surgery_and_radiation_levels_are_declared_unobservable(self):
        # The finding this report turns on: occurrence yes, rung no.
        self.assertFalse(AXIS_RESOLUTION["surgery"]["level"])
        self.assertFalse(AXIS_RESOLUTION["radiation"]["level"])
        self.assertTrue(AXIS_RESOLUTION["surgery"]["occurrence"])
        self.assertTrue(AXIS_RESOLUTION["radiation"]["occurrence"])


class TimingTests(unittest.TestCase):
    def test_chemo_before_surgery_is_neoadjuvant(self):
        treatment = treatment_frame([("A", 10, 100, "Chemo", "PACLITAXEL")])
        surgery = pd.DataFrame([
            {"PATIENT_ID": "A", "dx_start_date": 150, "SUBTYPE": "PROCEDURE"}])
        table = timing_table(treatment, surgery)
        self.assertEqual(table["timing"].iloc[0], "neoadjuvant")

    def test_same_day_counts_as_surgery_first(self):
        treatment = treatment_frame([("A", 50, 100, "Chemo", "PACLITAXEL")])
        surgery = pd.DataFrame([
            {"PATIENT_ID": "A", "dx_start_date": 50, "SUBTYPE": "PROCEDURE"}])
        self.assertEqual(timing_table(treatment, surgery)["timing"].iloc[0],
                         "surgery_first")

    def test_patients_missing_either_date_are_absent(self):
        treatment = treatment_frame([("A", 10, 100, "Chemo", "X")])
        surgery = pd.DataFrame(
            [{"PATIENT_ID": "B", "dx_start_date": 10, "SUBTYPE": "PROCEDURE"}])
        self.assertTrue(timing_table(treatment, surgery).empty)

    def test_plausible_neoadjuvant_needs_a_course_length_gap(self):
        # The post-hoc secondary reading: chemotherapy 3 years before an
        # operation is not a neoadjuvant course, however the dates order.
        treatment = treatment_frame([
            ("A", 10, 100, "Chemo", "X"),      # gap 120 days -> plausible
            ("B", 10, 100, "Chemo", "X"),      # gap 1000 days -> not
        ])
        surgery = pd.DataFrame([
            {"PATIENT_ID": "A", "dx_start_date": 130, "SUBTYPE": "PROCEDURE"},
            {"PATIENT_ID": "B", "dx_start_date": 1010, "SUBTYPE": "PROCEDURE"}])
        table = timing_table(treatment, surgery).set_index("PATIENT_ID")
        self.assertEqual(table.loc["A", "timing"], "neoadjuvant")
        self.assertEqual(table.loc["B", "timing"], "neoadjuvant")
        self.assertTrue(table.loc["A", "plausible_neoadjuvant"])
        self.assertFalse(table.loc["B", "plausible_neoadjuvant"])

    def test_adjuvant_patients_are_never_plausible_neoadjuvant(self):
        treatment = treatment_frame([("A", 200, 300, "Chemo", "X")])
        surgery = pd.DataFrame([
            {"PATIENT_ID": "A", "dx_start_date": 100, "SUBTYPE": "PROCEDURE"}])
        table = timing_table(treatment, surgery)
        self.assertEqual(table["timing"].iloc[0], "surgery_first")
        self.assertFalse(table["plausible_neoadjuvant"].iloc[0])

    def test_intervals_cover_every_patient_exactly_once(self):
        treatment = treatment_frame([
            ("A", 10, 100, "Chemo", "X"), ("B", 200, 300, "Chemo", "X"),
            ("C", 10, 100, "Chemo", "X")])
        surgery = pd.DataFrame([
            {"PATIENT_ID": "A", "dx_start_date": 130, "SUBTYPE": "PROCEDURE"},
            {"PATIENT_ID": "B", "dx_start_date": 100, "SUBTYPE": "PROCEDURE"},
            {"PATIENT_ID": "C", "dx_start_date": 2000, "SUBTYPE": "PROCEDURE"}])
        table = timing_table(treatment, surgery)
        intervals = timing_intervals(table)
        self.assertEqual(int(intervals["patients"].sum()), len(table))
        self.assertAlmostEqual(float(intervals["share"].sum()), 1.0, places=9)
        self.assertEqual(len(intervals), 9)
        self.assertTrue(intervals["reading"].str.len().gt(0).all())


class EndocrineDurationTests(unittest.TestCase):
    def test_overlapping_courses_are_merged_not_summed(self):
        # Tamoxifen and leuprolide given together for one year is one year.
        frame = treatment_frame([
            ("A", 0, 365, "Hormone", "TAMOXIFEN"),
            ("A", 0, 365, "Hormone", "LEUPROLIDE"),
        ])
        out = endocrine_duration(frame).set_index("PATIENT_ID")
        self.assertAlmostEqual(out.loc["A", "years"], 365 / DAYS_PER_YEAR, places=6)
        self.assertEqual(out.loc["A", "merged_spans"], 1)
        self.assertEqual(out.loc["A", "courses"], 2)

    def test_adjacent_courses_extend_the_span(self):
        frame = treatment_frame([
            ("A", 0, 365, "Hormone", "TAMOXIFEN"),
            ("A", 300, 1000, "Hormone", "LETROZOLE"),
        ])
        out = endocrine_duration(frame).set_index("PATIENT_ID")
        self.assertAlmostEqual(out.loc["A", "years"], 1000 / DAYS_PER_YEAR, places=6)
        self.assertEqual(out.loc["A", "merged_spans"], 1)

    def test_disjoint_courses_stay_separate(self):
        frame = treatment_frame([
            ("A", 0, 100, "Hormone", "TAMOXIFEN"),
            ("A", 500, 600, "Hormone", "TAMOXIFEN"),
        ])
        out = endocrine_duration(frame).set_index("PATIENT_ID")
        self.assertEqual(out.loc["A", "merged_spans"], 2)
        self.assertAlmostEqual(out.loc["A", "years"], 200 / DAYS_PER_YEAR, places=6)

    def test_five_years_is_the_extended_threshold(self):
        frame = treatment_frame([
            ("A", 0, 5 * DAYS_PER_YEAR + 1, "Hormone", "LETROZOLE"),
            ("B", 0, 3 * DAYS_PER_YEAR, "Hormone", "LETROZOLE"),
        ])
        out = endocrine_duration(frame).set_index("PATIENT_ID")
        self.assertTrue(out.loc["A", "extended"])
        self.assertFalse(out.loc["B", "extended"])

    def test_chemotherapy_rows_do_not_count(self):
        frame = treatment_frame([("A", 0, 365, "Chemo", "PACLITAXEL")])
        self.assertTrue(endocrine_duration(frame).empty)


class ChemoAgentTests(unittest.TestCase):
    def test_counts_distinct_agents_in_the_first_course(self):
        frame = treatment_frame([
            ("A", 0, 60, "Chemo", "DOXORUBICIN"),
            ("A", 0, 60, "Chemo", "CYCLOPHOSPHAMIDE"),
            ("A", 90, 180, "Chemo", "PACLITAXEL"),
            ("A", 900, 1000, "Chemo", "CAPECITABINE"),   # outside the course
        ])
        out = chemo_agent_counts(frame, course_days=180).set_index("PATIENT_ID")
        self.assertEqual(out.loc["A", "agents"], 3)
        self.assertEqual(out.loc["A", "start_days"], 3)

    def test_rejects_non_positive_course(self):
        with self.assertRaises(ValueError):
            chemo_agent_counts(treatment_frame([("A", 0, 1, "Chemo", "X")]), 0)


class DecisionPointTests(unittest.TestCase):
    def test_nearby_scans_collapse_to_one_opportunity(self):
        frame = progression_frame([
            ("A", 100, "N"), ("A", 103, "Y"), ("A", 200, "Y")])
        out = decision_points(frame, 5.0, dedup_days=30).set_index("PATIENT_ID")
        self.assertEqual(out.loc["A", "scans"], 3)
        self.assertEqual(out.loc["A", "opportunities"], 2)

    def test_dedup_zero_counts_every_scan(self):
        frame = progression_frame([("A", 100, "N"), ("A", 103, "Y")])
        out = decision_points(frame, 5.0, dedup_days=0).set_index("PATIENT_ID")
        self.assertEqual(out.loc["A", "opportunities"], 2)

    def test_pre_diagnosis_and_post_horizon_scans_are_dropped(self):
        limit = 5.0 * DAYS_PER_YEAR
        frame = progression_frame([
            ("A", -10, "Y"), ("A", 100, "Y"), ("A", limit + 1, "Y")])
        out = decision_points(frame, 5.0).set_index("PATIENT_ID")
        self.assertEqual(out.loc["A", "opportunities"], 1)

    def test_progressing_calls_are_counted_after_dedup(self):
        frame = progression_frame([
            ("A", 100, "Y"), ("A", 105, "N"), ("A", 200, "Y")])
        out = decision_points(frame, 5.0, dedup_days=30).set_index("PATIENT_ID")
        self.assertEqual(out.loc["A", "progressing"], 2)

    def test_rejects_negative_dedup(self):
        with self.assertRaises(ValueError):
            decision_points(progression_frame([("A", 1, "Y")]), 5.0, -1)

    def test_summary_denominator_is_the_cohort_not_the_table(self):
        frame = progression_frame([("A", 100, "Y"), ("B", 100, "N")])
        summary = opportunity_summary(decision_points(frame, 5.0), cohort_size=10)
        self.assertEqual(summary["patients"], 2)
        self.assertAlmostEqual(summary["share_of_cohort"], 0.2)

    def test_empty_input_summarises_to_zero(self):
        summary = opportunity_summary(pd.DataFrame(), cohort_size=10)
        self.assertEqual(summary["patients"], 0)


class AssessmentGroupTests(unittest.TestCase):
    def test_three_labels_plus_fallback(self):
        self.assertEqual(assessment_group("Y"), "progressing")
        self.assertEqual(assessment_group("N"), "controlled")
        self.assertEqual(assessment_group("Indeterminate"), "indeterminate")
        self.assertEqual(assessment_group("???"), "other")


class SwitchTableTests(unittest.TestCase):
    def setUp(self):
        self.treatment = treatment_frame([
            ("A", 0, 100, "Chemo", "PACLITAXEL"),
            ("A", 250, 400, "Chemo", "CAPECITABINE"),
            ("B", 0, 100, "Hormone", "LETROZOLE"),
        ])
        self.followup = pd.Series({"A": 2000.0, "B": 2000.0})

    def test_forward_window_sees_a_later_start(self):
        calls = progression_frame([("A", 200, "Y")])
        out = switch_table(calls, self.treatment, self.followup, 90)
        self.assertTrue(out["switch_after"].iloc[0])
        self.assertFalse(out["switch_before"].iloc[0])

    def test_backward_window_is_the_placebo(self):
        calls = progression_frame([("A", 300, "Y")])
        out = switch_table(calls, self.treatment, self.followup, 90)
        self.assertFalse(out["switch_after"].iloc[0])
        self.assertTrue(out["switch_before"].iloc[0])

    def test_calls_before_the_first_treatment_are_dropped(self):
        calls = progression_frame([("A", -5, "Y")])
        self.assertTrue(switch_table(calls, self.treatment, self.followup, 90).empty)

    def test_calls_without_a_full_window_of_followup_are_dropped(self):
        calls = progression_frame([("A", 200, "Y")])
        short = pd.Series({"A": 250.0, "B": 2000.0})
        self.assertTrue(switch_table(calls, self.treatment, short, 90).empty)

    def test_new_agent_is_stricter_than_any_start(self):
        # A re-start of an agent already received is a switch by the loose
        # definition and not by the strict one.
        treatment = treatment_frame([
            ("A", 0, 100, "Chemo", "PACLITAXEL"),
            ("A", 250, 300, "Chemo", "PACLITAXEL"),
        ])
        calls = progression_frame([("A", 200, "Y")])
        out = switch_table(calls, treatment, self.followup, 90)
        self.assertTrue(out["switch_after"].iloc[0])
        self.assertFalse(out["new_agent_after"].iloc[0])

    def test_supportive_care_is_not_a_switch(self):
        treatment = treatment_frame([
            ("A", 0, 100, "Chemo", "PACLITAXEL"),
            ("A", 250, 300, "Bone Treatment", "ZOLEDRONIC ACID"),
        ])
        calls = progression_frame([("A", 200, "Y")])
        out = switch_table(calls, treatment, self.followup, 90)
        self.assertFalse(out["switch_after"].iloc[0])

    def test_rejects_non_positive_window(self):
        with self.assertRaises(ValueError):
            switch_table(progression_frame([("A", 1, "Y")]),
                         self.treatment, self.followup, 0)

    def test_rates_carry_counts(self):
        calls = progression_frame([
            ("A", 200, "Y"), ("A", 300, "N"), ("B", 200, "N")])
        rates = switch_rates(
            switch_table(calls, self.treatment, self.followup, 90))
        self.assertIn("calls", rates.columns)
        self.assertIn("patients", rates.columns)
        self.assertEqual(int(rates["calls"].sum()), 3)


class FollowupTests(unittest.TestCase):
    def test_os_months_are_rebased_onto_the_diagnosis_axis(self):
        patients = pd.DataFrame([{"PATIENT_ID": "A", "OS_MONTHS": 12.0}])
        anchor = pd.Series({"A": -400.0})
        days = followup_days(patients, anchor)
        # 12 months after sequencing, which is itself 400 days after diagnosis.
        self.assertAlmostEqual(days["A"], 12.0 * 30.4375 + 400.0, places=6)


class EcogTests(unittest.TestCase):
    def test_a_constant_patient_is_not_moving(self):
        frame = pd.DataFrame([
            {"PATIENT_ID": "A", "dx_start_date": 10, "ECOG": 1},
            {"PATIENT_ID": "A", "dx_start_date": 200, "ECOG": 1}])
        out = ecog_trajectory(frame).set_index("PATIENT_ID")
        self.assertEqual(out.loc["A", "distinct"], 1)
        self.assertFalse(out.loc["A", "ever_worsened"])
        self.assertEqual(out.loc["A", "range"], 0)

    def test_worsening_and_improving_are_recorded_separately(self):
        frame = pd.DataFrame([
            {"PATIENT_ID": "A", "dx_start_date": 10, "ECOG": 0},
            {"PATIENT_ID": "A", "dx_start_date": 100, "ECOG": 2},
            {"PATIENT_ID": "A", "dx_start_date": 200, "ECOG": 1}])
        out = ecog_trajectory(frame).set_index("PATIENT_ID")
        self.assertTrue(out.loc["A", "ever_worsened"])
        self.assertTrue(out.loc["A", "ever_improved"])
        self.assertEqual(out.loc["A", "worst"], 2)
        self.assertEqual(out.loc["A", "first"], 0)
        self.assertEqual(out.loc["A", "last"], 1)

    def test_out_of_horizon_measurements_are_dropped(self):
        frame = pd.DataFrame([
            {"PATIENT_ID": "A", "dx_start_date": -5, "ECOG": 0},
            {"PATIENT_ID": "A", "dx_start_date": 100, "ECOG": 1}])
        out = ecog_trajectory(frame).set_index("PATIENT_ID")
        self.assertEqual(out.loc["A", "measurements"], 1)


class EcogEncodingTests(unittest.TestCase):
    """The check that stopped v2.2 reporting an encoding as a clinical fact.

    The release emits an ECOG row only when the value changes, which makes
    "does ECOG move within patients?" true by construction. These tests pin
    the detector in both directions so a future release that records every
    measurement is not mistaken for this one.
    """

    def test_change_point_encoding_is_detected(self):
        frame = pd.DataFrame([
            {"PATIENT_ID": "A", "dx_start_date": 10, "ECOG": 0},
            {"PATIENT_ID": "A", "dx_start_date": 50, "ECOG": 1},
            {"PATIENT_ID": "B", "dx_start_date": 10, "ECOG": 2},
            {"PATIENT_ID": "B", "dx_start_date": 90, "ECOG": 1}])
        check = ecog_encoding_check(frame)
        self.assertEqual(check["consecutive_pairs"], 2)
        self.assertEqual(check["equal_consecutive_pairs"], 0)
        self.assertTrue(check["change_point_encoded"])

    def test_a_repeated_value_clears_the_flag(self):
        frame = pd.DataFrame([
            {"PATIENT_ID": "A", "dx_start_date": 10, "ECOG": 1},
            {"PATIENT_ID": "A", "dx_start_date": 50, "ECOG": 1}])
        check = ecog_encoding_check(frame)
        self.assertEqual(check["equal_consecutive_pairs"], 1)
        self.assertFalse(check["change_point_encoded"])

    def test_same_day_pairs_are_counted_separately(self):
        frame = pd.DataFrame([
            {"PATIENT_ID": "A", "dx_start_date": 10, "ECOG": 0},
            {"PATIENT_ID": "A", "dx_start_date": 10, "ECOG": 1}])
        check = ecog_encoding_check(frame)
        self.assertEqual(check["same_day_pairs"], 1)
        self.assertEqual(check["equal_same_day_pairs"], 0)

    def test_a_single_row_cannot_establish_an_encoding(self):
        frame = pd.DataFrame([
            {"PATIENT_ID": "A", "dx_start_date": 10, "ECOG": 0}])
        check = ecog_encoding_check(frame)
        self.assertEqual(check["consecutive_pairs"], 0)
        self.assertFalse(check["change_point_encoded"])

    def test_the_check_ignores_the_horizon(self):
        # Deliberately unfiltered: the question is about the file, not a cohort.
        frame = pd.DataFrame([
            {"PATIENT_ID": "A", "dx_start_date": -500, "ECOG": 0},
            {"PATIENT_ID": "A", "dx_start_date": -400, "ECOG": 1}])
        self.assertEqual(ecog_encoding_check(frame)["consecutive_pairs"], 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
