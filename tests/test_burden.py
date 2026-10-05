"""Tests for the disease-burden grading dial (v2.4)."""

from __future__ import annotations

from dataclasses import replace

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.dynamic.burden import (  # noqa: E402
    FULL_SEPARATION,
    GRADE_WEIGHTS,
    GRADES,
    declared_grades,
    grade_multipliers,
    graded_environment,
    with_burden_grade,
)
from analysis.dynamic.config import DynamicConfig  # noqa: E402
from analysis.dynamic.environment import DynamicBreastCancerEnvironment  # noqa: E402
from analysis.dynamic.schema import PatientProfile, RiskEstimate  # noqa: E402
from analysis.mcts.environment import all_plans  # noqa: E402

RAW_V07 = json.loads((ROOT / "configs" / "dynamic_v0_7.json").read_text(encoding="utf-8"))


def make_environment():
    patient = PatientProfile(
        patient_id="P", age=55.0, menopause="post", tumor_size_mm=25.0,
        lymph_pos=1, stage=2, grade=2, subtype="HR+/HER2-", er=1, pr=1, her2=0)
    table = {plan: RiskEstimate(five_year_os=0.85, five_year_rfs=0.78)
             for plan in all_plans()}
    return DynamicBreastCancerEnvironment(patient, table, DynamicConfig(**RAW_V07))


class WeightTests(unittest.TestCase):
    def test_weights_cover_every_grade_and_sum_to_one(self):
        self.assertEqual(set(GRADE_WEIGHTS), set(GRADES))
        self.assertAlmostEqual(sum(GRADE_WEIGHTS.values()), 1.0, places=12)

    def test_weights_match_the_msk_chord_counts(self):
        # Counts, not taste: reports/msk-chord-profile-v2.2 scored 32,977 calls.
        self.assertAlmostEqual(GRADE_WEIGHTS["controlled"], 18481 / 32977, places=12)
        self.assertAlmostEqual(GRADE_WEIGHTS["indeterminate"], 6620 / 32977, places=12)
        self.assertAlmostEqual(GRADE_WEIGHTS["progressing"], 7876 / 32977, places=12)

    def test_grades_are_ordered_worst_last(self):
        self.assertEqual(GRADES, ("controlled", "indeterminate", "progressing"))
        self.assertLess(FULL_SEPARATION["controlled"],
                        FULL_SEPARATION["indeterminate"])
        self.assertLess(FULL_SEPARATION["indeterminate"],
                        FULL_SEPARATION["progressing"])


class MultiplierTests(unittest.TestCase):
    def test_separation_zero_is_the_current_environment(self):
        # The identity the null control depends on.
        for value in grade_multipliers(0.0).values():
            self.assertAlmostEqual(value, 1.0, places=12)

    def test_mean_neutral_at_every_separation(self):
        # Without this the dial would change average prognosis as it turned,
        # and the map would read that instead of the grading (v0.5's lesson).
        for separation in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
            multipliers = grade_multipliers(separation)
            mean = sum(GRADE_WEIGHTS[grade] * multipliers[grade] for grade in GRADES)
            self.assertAlmostEqual(mean, 1.0, places=12)

    def test_ordering_is_preserved_and_strict_above_zero(self):
        for separation in (0.1, 0.5, 1.0):
            multipliers = grade_multipliers(separation)
            self.assertLess(multipliers["controlled"], multipliers["indeterminate"])
            self.assertLess(multipliers["indeterminate"], multipliers["progressing"])

    def test_spread_grows_monotonically_with_separation(self):
        previous = None
        for separation in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            multipliers = grade_multipliers(separation)
            spread = multipliers["progressing"] / multipliers["controlled"]
            if previous is not None:
                self.assertGreater(spread, previous)
            previous = spread

    def test_full_separation_reaches_the_declared_ratio(self):
        multipliers = grade_multipliers(1.0)
        declared = FULL_SEPARATION["progressing"] / FULL_SEPARATION["controlled"]
        self.assertAlmostEqual(
            multipliers["progressing"] / multipliers["controlled"], declared, places=12)

    def test_rejects_out_of_range_separation(self):
        for separation in (-0.01, 1.01):
            with self.assertRaises(ValueError):
                grade_multipliers(separation)

    def test_rejects_weights_that_do_not_sum_to_one(self):
        with self.assertRaises(ValueError):
            grade_multipliers(0.5, weights={
                "controlled": 0.5, "indeterminate": 0.2, "progressing": 0.2})


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = DynamicConfig(**RAW_V07)
        self.base = float(self.config.hazard_multipliers["death_after_recurrence"])

    def test_separation_zero_leaves_the_hazard_untouched(self):
        for grade in GRADES:
            out = with_burden_grade(self.config, grade, 0.0)
            self.assertAlmostEqual(
                float(out.hazard_multipliers["death_after_recurrence"]),
                self.base, places=12)

    def test_grades_move_the_hazard_in_the_right_direction(self):
        controlled = with_burden_grade(self.config, "controlled", 1.0)
        progressing = with_burden_grade(self.config, "progressing", 1.0)
        self.assertLess(
            float(controlled.hazard_multipliers["death_after_recurrence"]), self.base)
        self.assertGreater(
            float(progressing.hazard_multipliers["death_after_recurrence"]), self.base)

    def test_the_weighted_mean_hazard_is_preserved(self):
        mean = sum(
            GRADE_WEIGHTS[grade]
            * float(with_burden_grade(self.config, grade, 1.0)
                    .hazard_multipliers["death_after_recurrence"])
            for grade in GRADES)
        self.assertAlmostEqual(mean, self.base, places=12)

    def test_the_original_config_is_not_mutated(self):
        with_burden_grade(self.config, "progressing", 1.0)
        self.assertAlmostEqual(
            float(self.config.hazard_multipliers["death_after_recurrence"]),
            self.base, places=12)

    def test_rejects_an_unknown_grade(self):
        with self.assertRaises(ValueError):
            with_burden_grade(self.config, "stable", 0.5)

    def test_nothing_else_in_the_config_changes(self):
        out = with_burden_grade(self.config, "progressing", 1.0)
        for key in ("chemo", "endocrine", "radiation", "response", "timing"):
            self.assertEqual(out.hazard_multipliers[key],
                             self.config.hazard_multipliers[key])
        self.assertEqual(out.horizon_years, self.config.horizon_years)
        self.assertEqual(out.terminal_tail_years, self.config.terminal_tail_years)


def recurred_state(environment):
    """A follow-up state with a complete plan, one year in, already recurred.

    ``annual_event_probabilities`` reads the risk table through
    ``static_plan``, which raises on a partial plan - so every field has to be
    set before the hazards can be asked for.
    """
    return replace(
        environment.initial_state(), phase="followup", year=1, recurred=True,
        timing="surgery_first", surgery="BCS", chemo="standard",
        endocrine="standard", radiation="local")


class EnvironmentTests(unittest.TestCase):
    def test_graded_environment_keeps_the_patient_and_the_risks(self):
        environment = make_environment()
        graded = graded_environment(environment, "progressing", 1.0)
        self.assertIs(graded.patient, environment.patient)
        self.assertEqual(graded.risk_table, environment.risk_table)

    def test_a_worse_grade_raises_the_post_recurrence_death_probability(self):
        environment = make_environment()
        state = recurred_state(environment)
        probabilities = {
            grade: graded_environment(environment, grade, 1.0)
                   .annual_event_probabilities(state)[0]
            for grade in GRADES
        }
        self.assertLess(probabilities["controlled"], probabilities["indeterminate"])
        self.assertLess(probabilities["indeterminate"], probabilities["progressing"])

    def test_separation_zero_reproduces_the_ungraded_probabilities(self):
        environment = make_environment()
        state = recurred_state(environment)
        expected = environment.annual_event_probabilities(state)
        for grade in GRADES:
            graded = graded_environment(environment, grade, 0.0)
            self.assertEqual(graded.annual_event_probabilities(state), expected)

    def test_grading_does_not_touch_a_recurrence_free_patient(self):
        # The multiplier is on ``death_after_recurrence``, so a patient who has
        # not recurred must be identical across grades - otherwise the dial
        # would be changing baseline prognosis rather than grading recurrence.
        environment = make_environment()
        state = replace(recurred_state(environment), recurred=False)
        expected = environment.annual_event_probabilities(state)
        for grade in GRADES:
            graded = graded_environment(environment, grade, 1.0)
            self.assertEqual(graded.annual_event_probabilities(state), expected)


class DeclaredGradeTests(unittest.TestCase):
    def test_one_row_per_grade_with_the_declared_hazard(self):
        config = DynamicConfig(**RAW_V07)
        rows = declared_grades(config, 1.0)
        self.assertEqual(len(rows), 3)
        self.assertEqual({row["grade"] for row in rows}, set(GRADES))
        for row in rows:
            self.assertAlmostEqual(
                row["death_after_recurrence"],
                float(config.hazard_multipliers["death_after_recurrence"])
                * row["multiplier"], places=12)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
