"""Tests for the standard-treatment benefit dial (v2.1)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.dynamic.benefit import (  # noqa: E402
    FULL_BENEFIT,
    HIGHER_LEVEL,
    STANDARD_LEVEL,
    declared_ladder,
    guideline_plan,
    hazard_at_strength,
    plan_expected_utility,
    refusal_plan,
    with_standard_benefit,
)
from analysis.dynamic.config import DynamicConfig  # noqa: E402
from analysis.dynamic.environment import DynamicBreastCancerEnvironment  # noqa: E402
from analysis.dynamic.schema import PatientProfile, RiskEstimate  # noqa: E402
from analysis.mcts.environment import all_plans  # noqa: E402

RAW_V05 = json.loads((ROOT / "configs" / "dynamic_v0_5.json").read_text(encoding="utf-8"))


def make_environment(raw: dict, os=0.85, rfs=0.78):
    patient = PatientProfile(
        patient_id="P", age=55.0, menopause="post", tumor_size_mm=25.0,
        lymph_pos=1, stage=2, grade=2, subtype="HR+/HER2-", er=1, pr=1, her2=0)
    table = {plan: RiskEstimate(five_year_os=os, five_year_rfs=rfs) for plan in all_plans()}
    return DynamicBreastCancerEnvironment(patient, table, DynamicConfig(**raw))


class InterpolationTests(unittest.TestCase):
    def test_endpoints(self):
        self.assertEqual(hazard_at_strength(0.75, 0.0), 1.0)
        self.assertEqual(hazard_at_strength(0.75, 1.0), 0.75)

    def test_halfway_is_the_geometric_mean(self):
        self.assertAlmostEqual(hazard_at_strength(0.64, 0.5), 0.8, places=12)

    def test_rejects_out_of_range_strength(self):
        for strength in (-0.01, 1.01):
            with self.assertRaises(ValueError):
                hazard_at_strength(0.75, strength)

    def test_rejects_non_positive_hazard(self):
        with self.assertRaises(ValueError):
            hazard_at_strength(0.0, 0.5)


class SweepTests(unittest.TestCase):
    def test_strength_zero_reproduces_the_config_exactly(self):
        self.assertEqual(with_standard_benefit(RAW_V05, 0.0), RAW_V05)

    def test_strength_one_reaches_the_declared_endpoint(self):
        raised = with_standard_benefit(RAW_V05, 1.0)
        for channel, full in FULL_BENEFIT.items():
            level = STANDARD_LEVEL[channel]
            for outcome, value in full.items():
                self.assertAlmostEqual(
                    raised["hazard_multipliers"][channel][level][outcome], value, places=12)

    def test_the_ladder_keeps_its_increments(self):
        # Without this the sweep would leave intensified chemotherapy (0.95/0.90)
        # *worse* than a raised standard rung and silently measure that instead.
        for strength in (0.25, 0.5, 1.0):
            raised = with_standard_benefit(RAW_V05, strength)
            for channel in FULL_BENEFIT:
                standard, higher = STANDARD_LEVEL[channel], HIGHER_LEVEL[channel]
                for outcome in ("death", "recurrence"):
                    now = (RAW_V05["hazard_multipliers"][channel][higher][outcome]
                           / RAW_V05["hazard_multipliers"][channel][standard][outcome])
                    after = (raised["hazard_multipliers"][channel][higher][outcome]
                             / raised["hazard_multipliers"][channel][standard][outcome])
                    self.assertAlmostEqual(after, now, places=12)

    def test_the_higher_rung_is_never_worse_than_standard(self):
        for strength in (0.0, 0.3, 0.6, 1.0):
            raised = with_standard_benefit(RAW_V05, strength)
            for channel in FULL_BENEFIT:
                standard, higher = STANDARD_LEVEL[channel], HIGHER_LEVEL[channel]
                for outcome in ("death", "recurrence"):
                    self.assertLessEqual(
                        raised["hazard_multipliers"][channel][higher][outcome],
                        raised["hazard_multipliers"][channel][standard][outcome] + 1e-12)

    def test_no_treatment_stays_neutral(self):
        raised = with_standard_benefit(RAW_V05, 1.0)
        for channel in FULL_BENEFIT:
            for outcome in ("death", "recurrence"):
                self.assertEqual(raised["hazard_multipliers"][channel]["none"][outcome], 1.0)

    def test_hazards_fall_monotonically_with_strength(self):
        previous = None
        for strength in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            value = with_standard_benefit(
                RAW_V05, strength)["hazard_multipliers"]["endocrine"]["standard"]["recurrence"]
            if previous is not None:
                self.assertLess(value, previous)
            previous = value

    def test_the_original_config_is_not_mutated(self):
        with_standard_benefit(RAW_V05, 1.0)
        self.assertEqual(
            RAW_V05["hazard_multipliers"]["chemo"]["standard"]["death"], 1.0)

    def test_rejects_out_of_range_strength(self):
        with self.assertRaises(ValueError):
            with_standard_benefit(RAW_V05, 1.5)

    def test_declared_ladder_covers_every_level(self):
        rows = declared_ladder(RAW_V05)
        self.assertEqual(len(rows), 9)  # three channels x three levels
        pure_cost = [r for r in rows
                     if r["death_hazard"] == 1.0 and r["recurrence_hazard"] == 1.0
                     and r["acute_toxicity_probability"] > 0]
        # v1.5's finding, still true at strength 0: three pure-cost rungs.
        self.assertEqual(len(pure_cost), 3)


class PlanUtilityTests(unittest.TestCase):
    def setUp(self):
        self.environment = make_environment(RAW_V05)
        self.guideline = guideline_plan(self.environment)
        self.refusal = refusal_plan(self.environment, self.guideline)

    def test_refusal_keeps_surgery_and_drops_the_rest(self):
        self.assertEqual(self.refusal["surgery"], self.guideline["surgery"])
        for field in ("chemo", "endocrine", "radiation"):
            self.assertEqual(self.refusal[field], "none")

    def test_incomplete_plan_is_rejected(self):
        with self.assertRaises(ValueError):
            plan_expected_utility(self.environment, {"surgery": "MAST"})

    def test_at_strength_zero_refusing_beats_the_guideline(self):
        # v1.5's asymmetry, stated as arithmetic: standard treatment is pure
        # cost, so declining it is strictly better for this patient.
        guideline = plan_expected_utility(self.environment, self.guideline)
        refusal = plan_expected_utility(self.environment, self.refusal)
        self.assertGreater(refusal, guideline)

    def test_at_full_strength_the_guideline_beats_refusing(self):
        environment = make_environment(with_standard_benefit(RAW_V05, 1.0))
        guideline = plan_expected_utility(environment, self.guideline)
        refusal = plan_expected_utility(environment, self.refusal)
        self.assertGreater(guideline, refusal)

    def test_the_guideline_plan_improves_monotonically_with_strength(self):
        previous = None
        for strength in (0.0, 0.25, 0.5, 0.75, 1.0):
            environment = make_environment(with_standard_benefit(RAW_V05, strength))
            value = plan_expected_utility(environment, self.guideline)
            if previous is not None:
                self.assertGreater(value, previous)
            previous = value

    def test_the_refusal_plan_is_unaffected_by_strength(self):
        # It takes none of the three channels, so raising their benefit must
        # not move it at all - the control that makes the crossing readable.
        values = {round(plan_expected_utility(
            make_environment(with_standard_benefit(RAW_V05, strength)), self.refusal), 12)
            for strength in (0.0, 0.5, 1.0)}
        self.assertEqual(len(values), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
