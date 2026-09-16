"""Tests for the salvage indifference map (v2.0)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.dynamic.config import load_dynamic_config  # noqa: E402
from analysis.dynamic.environment import DynamicBreastCancerEnvironment  # noqa: E402
from analysis.dynamic.indifference import (  # noqa: E402
    expected_remaining_utility,
    indifference_map,
    indifference_ratio_per_patient,
    pick_indifference_ratio,
    salvage_advantage,
    salvage_cost,
    treat_share_by_ratio,
    with_salvage_ratio,
)
from analysis.dynamic.schema import PatientProfile, RiskEstimate  # noqa: E402
from analysis.mcts.environment import all_plans  # noqa: E402

CONFIG_V07 = load_dynamic_config(ROOT / "configs" / "dynamic_v0_7.json")


def make_patient(patient_id="P", stage=2):
    return PatientProfile(
        patient_id=patient_id, age=55.0, menopause="post", tumor_size_mm=25.0,
        lymph_pos=1, stage=stage, grade=2, subtype="HR+/HER2-", er=1, pr=1, her2=0)


def make_environment(os=0.85, rfs=0.78, patient_id="P"):
    table = {plan: RiskEstimate(five_year_os=os, five_year_rfs=rfs) for plan in all_plans()}
    return DynamicBreastCancerEnvironment(make_patient(patient_id), table, CONFIG_V07)


def followup(environment, year=1, **overrides):
    return replace(environment.initial_state(), phase="followup", timing="surgery_first",
                   surgery="MAST", chemo="standard", endocrine="standard",
                   radiation="none", year=year, **overrides)


class ExpectedUtilityTests(unittest.TestCase):
    def test_rejects_treatment_phase_states(self):
        environment = make_environment()
        with self.assertRaises(ValueError):
            expected_remaining_utility(environment, environment.initial_state())

    def test_immortal_patient_collects_every_remaining_year_plus_tail(self):
        environment = make_environment(os=0.999999, rfs=0.999999)
        state = followup(environment, year=2)
        per_year = CONFIG_V07.normalized(
            CONFIG_V07.reward["alive_year"] + CONFIG_V07.reward["recurrence_free_year"])
        expected = (CONFIG_V07.horizon_years - 2) * per_year + 10 * per_year
        self.assertAlmostEqual(expected_remaining_utility(environment, state), expected, places=3)

    def test_recurred_state_is_worth_less(self):
        environment = make_environment()
        free = expected_remaining_utility(environment, followup(environment))
        recurred = expected_remaining_utility(environment, followup(environment, recurred=True))
        self.assertLess(recurred, free)

    def test_later_state_has_less_remaining(self):
        environment = make_environment()
        early = expected_remaining_utility(environment, followup(environment, year=1))
        late = expected_remaining_utility(environment, followup(environment, year=4))
        self.assertLess(late, early)


class SalvageAdvantageTests(unittest.TestCase):
    def test_cost_matches_the_declared_blocks(self):
        environment = make_environment()
        self.assertEqual(salvage_cost(environment, "none"), 0.0)
        expected = CONFIG_V07.normalized(0.10 + 0.20 * 0.15)
        self.assertAlmostEqual(salvage_cost(environment, "systemic"), expected, places=12)

    def test_advantage_is_monotone_decreasing_in_the_hazard_ratio(self):
        environment = make_environment()
        state = replace(followup(environment, year=2), phase="salvage", recurred=True,
                        recurrence_year=2)
        values = [salvage_advantage(with_salvage_ratio(environment, r), state)
                  for r in (0.80, 0.90, 1.00)]
        self.assertGreater(values[0], values[1])
        self.assertGreater(values[1], values[2])

    def test_at_ratio_one_treating_is_pure_cost(self):
        environment = make_environment()
        state = replace(followup(environment, year=2), phase="salvage", recurred=True,
                        recurrence_year=2)
        advantage = salvage_advantage(with_salvage_ratio(environment, 1.0), state)
        self.assertAlmostEqual(advantage, -salvage_cost(environment, "systemic"), places=12)

    def test_higher_risk_patient_gains_more_from_the_same_ratio(self):
        # Multiplicative benefit, flat cost: the sicker patient is further from
        # indifference on the treating side.
        low = make_environment(os=0.95)
        high = make_environment(os=0.60)
        state_low = replace(followup(low, year=1), phase="salvage", recurred=True, recurrence_year=1)
        state_high = replace(followup(high, year=1), phase="salvage", recurred=True, recurrence_year=1)
        self.assertGreater(salvage_advantage(high, state_high), salvage_advantage(low, state_low))

    def test_needs_a_salvage_state(self):
        environment = make_environment()
        with self.assertRaises(ValueError):
            salvage_advantage(environment, followup(environment))


class MapTests(unittest.TestCase):
    def setUp(self):
        self.environments = {"low": make_environment(os=0.95, patient_id="low"),
                             "high": make_environment(os=0.60, patient_id="high")}
        self.plans = {key: followup(env) for key, env in self.environments.items()}
        self.ratios = np.round(np.arange(0.80, 1.001, 0.05), 2)

    def test_map_has_one_row_per_cell(self):
        frame = indifference_map(self.environments, self.plans, self.ratios, years=(1, 3))
        self.assertEqual(len(frame), 2 * 2 * len(self.ratios))

    def test_treat_share_falls_with_the_ratio(self):
        frame = indifference_map(self.environments, self.plans, self.ratios, years=(1, 3))
        share = treat_share_by_ratio(frame)
        self.assertTrue((np.diff(share["treat_share"]) <= 1e-12).all())

    def test_indifference_ratio_is_higher_for_the_sicker_patient(self):
        frame = indifference_map(self.environments, self.plans, self.ratios, years=(1, 3))
        crossing = indifference_ratio_per_patient(frame)
        self.assertGreaterEqual(crossing["high"], crossing["low"])

    def test_pick_returns_a_grid_value_closest_to_target(self):
        frame = indifference_map(self.environments, self.plans, self.ratios, years=(1, 3))
        share = treat_share_by_ratio(frame)
        chosen = pick_indifference_ratio(share, target=0.5)
        self.assertIn(chosen, set(share["hazard_ratio"]))

    def test_with_salvage_ratio_does_not_mutate_the_original(self):
        environment = self.environments["low"]
        with_salvage_ratio(environment, 0.5)
        self.assertEqual(environment.config.hazard_multipliers["salvage"]["systemic"]["death"], 0.85)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
