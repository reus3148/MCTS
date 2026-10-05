"""Tests for the ECOG state-dependence analysis (v2.3)."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.msk.performance import (  # noqa: E402
    DAYS_PER_YEAR,
    chemo_share_by_ecog,
    ecog_at,
    ecog_changes,
    on_treatment,
    stop_rates,
    stop_rates_by_size,
    stop_table,
    transition_rates,
    treatment_spans,
)


def status_frame(rows):
    """``(patient, day, ecog)`` -> a performance-status timeline."""
    return pd.DataFrame(
        [{"PATIENT_ID": p, "dx_start_date": d, "ECOG": e} for p, d, e in rows])


def treatment_frame(rows):
    """``(patient, start, stop, subtype)`` -> a treatment timeline."""
    return pd.DataFrame(
        [{"PATIENT_ID": p, "dx_start_date": a, "dx_stop_date": b, "SUBTYPE": s}
         for p, a, b, s in rows])


def progression_frame(rows):
    return pd.DataFrame(
        [{"PATIENT_ID": p, "dx_start_date": d, "PROGRESSION": c}
         for p, d, c in rows])


class ChangeTests(unittest.TestCase):
    def test_first_row_is_not_a_change(self):
        frame = status_frame([("A", 10, 0), ("A", 100, 1), ("A", 200, 2)])
        changes = ecog_changes(frame)
        self.assertEqual(len(changes), 2)
        self.assertEqual(list(changes["day"]), [100, 200])

    def test_direction_and_size(self):
        frame = status_frame([("A", 10, 0), ("A", 100, 2), ("A", 200, 1)])
        changes = ecog_changes(frame).set_index("day")
        self.assertEqual(changes.loc[100, "direction"], "worsened")
        self.assertEqual(changes.loc[100, "size"], 2)
        self.assertEqual(changes.loc[200, "direction"], "improved")

    def test_a_single_record_yields_nothing(self):
        self.assertTrue(ecog_changes(status_frame([("A", 10, 1)])).empty)

    def test_out_of_horizon_rows_are_dropped(self):
        limit = 5.0 * DAYS_PER_YEAR
        frame = status_frame([("A", -5, 0), ("A", 10, 1), ("A", limit + 1, 2)])
        self.assertEqual(len(ecog_changes(frame, 5.0)), 0)


class SpanTests(unittest.TestCase):
    def test_overlapping_agents_are_one_span(self):
        frame = treatment_frame([
            ("A", 0, 100, "Hormone"), ("A", 50, 200, "Targeted")])
        spans = treatment_spans(frame)
        self.assertEqual(spans["A"], [(0.0, 200.0)])

    def test_disjoint_courses_stay_separate(self):
        frame = treatment_frame([
            ("A", 0, 100, "Chemo"), ("A", 300, 400, "Chemo")])
        self.assertEqual(len(treatment_spans(frame)["A"]), 2)

    def test_supportive_care_is_not_treatment(self):
        frame = treatment_frame([("A", 0, 100, "Bone Treatment")])
        self.assertNotIn("A", treatment_spans(frame))

    def test_on_treatment_is_inclusive_at_both_ends(self):
        spans = [(0.0, 100.0)]
        self.assertTrue(on_treatment(spans, 0))
        self.assertTrue(on_treatment(spans, 100))
        self.assertFalse(on_treatment(spans, 101))
        self.assertFalse(on_treatment(spans, -1))


class StopTableTests(unittest.TestCase):
    def setUp(self):
        self.followup = pd.Series({"A": 2000.0, "B": 2000.0})

    def test_a_stop_after_the_change_is_recorded(self):
        treatment = treatment_frame([("A", 0, 150, "Chemo")])
        changes = ecog_changes(status_frame([("A", 10, 0), ("A", 100, 2)]))
        table = stop_table(changes, treatment, self.followup, 90)
        self.assertEqual(len(table), 1)
        self.assertTrue(table["stopped_after"].iloc[0])

    def test_treatment_continuing_is_not_a_stop(self):
        treatment = treatment_frame([("A", 0, 900, "Hormone")])
        changes = ecog_changes(status_frame([("A", 10, 0), ("A", 100, 2)]))
        table = stop_table(changes, treatment, self.followup, 90)
        self.assertFalse(table["stopped_after"].iloc[0])

    def test_changes_off_treatment_are_dropped(self):
        # A stop is undefined for a patient who was not on treatment.
        treatment = treatment_frame([("A", 0, 50, "Chemo")])
        changes = ecog_changes(status_frame([("A", 10, 0), ("A", 300, 2)]))
        self.assertTrue(stop_table(changes, treatment, self.followup, 90).empty)

    def test_the_backward_window_is_the_mirror_question(self):
        # On treatment at the change, but not 90 days earlier - a start, which
        # is what the backward window must detect rather than a stop.
        treatment = treatment_frame([("A", 80, 900, "Hormone")])
        changes = ecog_changes(status_frame([("A", 10, 0), ("A", 100, 2)]))
        table = stop_table(changes, treatment, self.followup, 90)
        self.assertTrue(table["stopped_before"].iloc[0])
        self.assertFalse(table["stopped_after"].iloc[0])

    def test_changes_without_a_full_window_are_dropped(self):
        treatment = treatment_frame([("A", 0, 900, "Hormone")])
        changes = ecog_changes(status_frame([("A", 10, 0), ("A", 100, 2)]))
        short = pd.Series({"A": 150.0})
        self.assertTrue(stop_table(changes, treatment, short, 90).empty)

    def test_rejects_non_positive_window(self):
        changes = ecog_changes(status_frame([("A", 10, 0), ("A", 100, 2)]))
        with self.assertRaises(ValueError):
            stop_table(changes, treatment_frame([("A", 0, 900, "Chemo")]),
                       self.followup, 0)

    def test_rates_carry_counts_and_the_placebo_contrast(self):
        treatment = treatment_frame([("A", 0, 150, "Chemo"), ("B", 0, 900, "Hormone")])
        changes = ecog_changes(status_frame([
            ("A", 10, 0), ("A", 100, 2), ("B", 10, 2), ("B", 100, 1)]))
        rates = stop_rates(stop_table(changes, treatment, self.followup, 90))
        self.assertIn("changes", rates.columns)
        self.assertIn("forward_minus_backward", rates.columns)
        self.assertEqual(set(rates["direction"]), {"worsened", "improved"})

    def test_empty_input_gives_an_empty_summary(self):
        self.assertTrue(stop_rates(pd.DataFrame()).empty)


class StopBySizeTests(unittest.TestCase):
    """The dose-response check that decides whether the null survives.

    If stopping tracked performance status, the gap over the improvement
    baseline would widen with the size of the fall. These tests pin the
    arithmetic so a future run's flat profile cannot be blamed on the helper.
    """

    def setUp(self):
        self.table = pd.DataFrame([
            {"PATIENT_ID": "A", "direction": "improved",
             "ecog_from": 2, "ecog_to": 1, "stopped_after": False},
            {"PATIENT_ID": "B", "direction": "improved",
             "ecog_from": 2, "ecog_to": 0, "stopped_after": True},
            {"PATIENT_ID": "C", "direction": "worsened",
             "ecog_from": 0, "ecog_to": 1, "stopped_after": True},
            {"PATIENT_ID": "D", "direction": "worsened",
             "ecog_from": 0, "ecog_to": 1, "stopped_after": True},
            {"PATIENT_ID": "E", "direction": "worsened",
             "ecog_from": 0, "ecog_to": 3, "stopped_after": False},
        ])

    def test_baseline_is_the_improvement_rate(self):
        out = stop_rates_by_size(self.table)
        self.assertTrue((out["baseline_improved"] == 0.5).all())

    def test_gap_is_measured_against_that_baseline(self):
        out = stop_rates_by_size(self.table).set_index("size")
        self.assertAlmostEqual(out.loc[1, "stopped_after"], 1.0)
        self.assertAlmostEqual(out.loc[1, "gap"], 0.5)
        self.assertAlmostEqual(out.loc[3, "stopped_after"], 0.0)
        self.assertAlmostEqual(out.loc[3, "gap"], -0.5)

    def test_improvements_are_not_given_a_size_row(self):
        out = stop_rates_by_size(self.table)
        self.assertEqual(set(out["size"]), {1, 3})

    def test_counts_are_carried(self):
        out = stop_rates_by_size(self.table).set_index("size")
        self.assertEqual(out.loc[1, "changes"], 2)
        self.assertEqual(out.loc[1, "patients"], 2)

    def test_empty_input_is_handled(self):
        self.assertTrue(stop_rates_by_size(pd.DataFrame()).empty)


class EcogAtTests(unittest.TestCase):
    def test_carries_the_last_value_forward(self):
        records = [(10.0, 0), (100.0, 2)]
        self.assertEqual(ecog_at(records, 50), 0)
        self.assertEqual(ecog_at(records, 100), 2)
        self.assertEqual(ecog_at(records, 500), 2)

    def test_returns_none_before_the_first_record(self):
        self.assertIsNone(ecog_at([(10.0, 0)], 5))


class ChemoShareTests(unittest.TestCase):
    def test_share_is_per_start_not_per_patient(self):
        status = status_frame([("A", 0, 0)])
        treatment = treatment_frame([
            ("A", 10, 100, "Chemo"), ("A", 200, 300, "Chemo"),
            ("A", 400, 500, "Hormone")])
        out = chemo_share_by_ecog(status, treatment).set_index("ecog")
        self.assertEqual(out.loc[0, "starts"], 3)
        self.assertAlmostEqual(out.loc[0, "chemo_share"], 2 / 3)

    def test_the_level_in_force_at_the_start_is_used(self):
        status = status_frame([("A", 0, 0), ("A", 150, 2)])
        treatment = treatment_frame([
            ("A", 10, 100, "Chemo"), ("A", 200, 300, "Hormone")])
        out = chemo_share_by_ecog(status, treatment).set_index("ecog")
        self.assertEqual(out.loc[0, "chemo"], 1)
        self.assertEqual(out.loc[2, "chemo"], 0)

    def test_starts_before_the_first_ecog_record_are_dropped(self):
        # A baseline carried backwards would be invented, not observed.
        status = status_frame([("A", 500, 1)])
        treatment = treatment_frame([("A", 10, 100, "Chemo")])
        self.assertTrue(chemo_share_by_ecog(status, treatment).empty)

    def test_supportive_care_is_excluded(self):
        status = status_frame([("A", 0, 0)])
        treatment = treatment_frame([
            ("A", 10, 100, "Chemo"), ("A", 20, 60, "Bone Treatment")])
        out = chemo_share_by_ecog(status, treatment).set_index("ecog")
        self.assertEqual(out.loc[0, "starts"], 1)

    def test_strata_split_the_counts_without_losing_any(self):
        # The check that turned a surprising marginal reversal into a finding:
        # the same association has to be visible inside each stratum.
        status = status_frame([("A", 0, 0), ("B", 0, 2)])
        treatment = treatment_frame([
            ("A", 10, 100, "Chemo"), ("B", 10, 100, "Hormone")])
        strata = pd.Series({"A": "Stage 1-3", "B": "Stage 4"})
        out = chemo_share_by_ecog(status, treatment, strata=strata)
        self.assertIn("stratum", out.columns)
        self.assertEqual(int(out["starts"].sum()), 2)
        self.assertEqual(set(out["stratum"]), {"Stage 1-3", "Stage 4"})

    def test_unstratified_output_has_no_stratum_column(self):
        status = status_frame([("A", 0, 0)])
        treatment = treatment_frame([("A", 10, 100, "Chemo")])
        self.assertNotIn("stratum", chemo_share_by_ecog(status, treatment).columns)


class TransitionRateTests(unittest.TestCase):
    def test_person_time_starts_at_the_first_ecog_record(self):
        # One worsening over one year of observed time, no progression.
        status = status_frame([("A", 100, 0), ("A", 100 + 365.25, 1)])
        rates = transition_rates(status, progression_frame([])).set_index("period")
        self.assertAlmostEqual(rates.loc["before_first_progression", "person_years"],
                               1.0, places=6)
        self.assertEqual(rates.loc["before_first_progression", "worsening_events"], 1)
        self.assertAlmostEqual(
            rates.loc["before_first_progression", "worsenings_per_year"], 1.0, places=6)

    def test_events_are_split_at_the_first_progression(self):
        status = status_frame([
            ("A", 0, 0), ("A", 100, 1), ("A", 400, 2)])
        progression = progression_frame([("A", 200, "Y")])
        rates = transition_rates(status, progression).set_index("period")
        self.assertEqual(rates.loc["before_first_progression", "worsening_events"], 1)
        self.assertEqual(rates.loc["after_first_progression", "worsening_events"], 1)

    def test_improvements_are_not_worsenings(self):
        status = status_frame([("A", 0, 2), ("A", 200, 1)])
        rates = transition_rates(status, progression_frame([])).set_index("period")
        self.assertEqual(rates.loc["before_first_progression", "worsening_events"], 0)

    def test_indeterminate_calls_do_not_start_the_after_period(self):
        status = status_frame([("A", 0, 0), ("A", 400, 1)])
        progression = progression_frame([("A", 100, "Indeterminate")])
        rates = transition_rates(status, progression).set_index("period")
        self.assertEqual(rates.loc["after_first_progression", "person_years"], 0.0)
        self.assertEqual(rates.loc["before_first_progression", "worsening_events"], 1)

    def test_a_single_record_contributes_no_person_time(self):
        rates = transition_rates(status_frame([("A", 10, 1)]),
                                 progression_frame([])).set_index("period")
        self.assertEqual(rates.loc["before_first_progression", "person_years"], 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
