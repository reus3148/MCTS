"""Tests for the salvage-stakes dial (v2.5)."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.dynamic.config import DynamicConfig  # noqa: E402
from analysis.dynamic.environment import DynamicBreastCancerEnvironment  # noqa: E402
from analysis.dynamic.schema import PatientProfile, RiskEstimate  # noqa: E402
from analysis.dynamic.stakes import (  # noqa: E402
    BINARY_LINES,
    FULL_LADDER,
    LINES,
    current_config_scale,
    declared_ladder,
    scaled_line,
    staked_environment,
    with_salvage_stakes,
)
from analysis.mcts.environment import all_plans  # noqa: E402

RAW_V07 = json.loads((ROOT / "configs" / "dynamic_v0_7.json").read_text(encoding="utf-8"))


def make_environment():
    patient = PatientProfile(
        patient_id="P", age=55.0, menopause="post", tumor_size_mm=25.0,
        lymph_pos=1, stage=2, grade=2, subtype="HR+/HER2-", er=1, pr=1, her2=0)
    table = {plan: RiskEstimate(five_year_os=0.85, five_year_rfs=0.78)
             for plan in all_plans()}
    return DynamicBreastCancerEnvironment(patient, table, DynamicConfig(**RAW_V07))


class LadderShapeTests(unittest.TestCase):
    def test_lines_are_ordered_weakest_first_and_include_decline(self):
        self.assertEqual(LINES[0], "none")
        self.assertEqual(BINARY_LINES, ("none", "systemic"))
        hazards = [FULL_LADDER[line]["death"] for line in LINES]
        self.assertEqual(hazards, sorted(hazards, reverse=True))

    def test_stronger_lines_cost_more(self):
        costs = [FULL_LADDER[line]["burden"] for line in LINES]
        self.assertEqual(costs, sorted(costs))

    def test_declining_is_free_and_neutral(self):
        self.assertEqual(FULL_LADDER["none"],
                         {"death": 1.0, "recurrence": 1.0,
                          "burden": 0.0, "toxicity": 0.0})


class ScaledLineTests(unittest.TestCase):
    def test_scale_zero_is_indistinguishable_from_declining(self):
        # The identity the null control depends on.
        for line in LINES:
            rung = scaled_line(FULL_LADDER[line], 0.0)
            self.assertEqual(rung, {"death": 1.0, "recurrence": 1.0,
                                    "burden": 0.0, "toxicity": 0.0})

    def test_scale_one_reaches_the_declared_endpoint(self):
        rung = scaled_line(FULL_LADDER["intensive"], 1.0)
        self.assertAlmostEqual(rung["death"], 0.50, places=12)
        self.assertAlmostEqual(rung["burden"], 0.40, places=12)

    def test_hazards_interpolate_geometrically(self):
        rung = scaled_line({"death": 0.25, "recurrence": 1.0,
                            "burden": 0.0, "toxicity": 0.0}, 0.5)
        self.assertAlmostEqual(rung["death"], 0.5, places=12)

    def test_costs_interpolate_linearly(self):
        rung = scaled_line(FULL_LADDER["systemic"], 0.5)
        self.assertAlmostEqual(rung["burden"], 0.10, places=12)

    def test_cost_multiple_scales_only_the_costs(self):
        plain = scaled_line(FULL_LADDER["systemic"], 1.0)
        dearer = scaled_line(FULL_LADDER["systemic"], 1.0, cost_multiple=2.0)
        self.assertAlmostEqual(dearer["death"], plain["death"], places=12)
        self.assertAlmostEqual(dearer["burden"], plain["burden"] * 2, places=12)

    def test_toxicity_is_capped_at_one(self):
        rung = scaled_line(FULL_LADDER["intensive"], 1.0, cost_multiple=100.0)
        self.assertEqual(rung["toxicity"], 1.0)

    def test_rejects_out_of_range_scale(self):
        for scale in (-0.01, 1.01):
            with self.assertRaises(ValueError):
                scaled_line(FULL_LADDER["mild"], scale)

    def test_rejects_negative_cost_multiple(self):
        with self.assertRaises(ValueError):
            scaled_line(FULL_LADDER["mild"], 0.5, cost_multiple=-1.0)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = DynamicConfig(**RAW_V07)

    def test_the_arm_defines_which_lines_exist(self):
        binary = with_salvage_stakes(self.config, 1.0, BINARY_LINES)
        ladder = with_salvage_stakes(self.config, 1.0, LINES)
        self.assertEqual(set(binary.hazard_multipliers["salvage"]), {"none", "systemic"})
        self.assertEqual(set(ladder.hazard_multipliers["salvage"]), set(LINES))

    def test_decline_is_always_offered(self):
        out = with_salvage_stakes(self.config, 1.0, ("systemic",))
        self.assertIn("none", out.hazard_multipliers["salvage"])

    def test_burden_and_toxicity_follow_the_lines(self):
        out = with_salvage_stakes(self.config, 1.0, LINES)
        for block in (out.treatment_burden["salvage"],
                      out.acute_toxicity_probabilities["salvage"]):
            self.assertEqual(set(block), set(LINES))

    def test_rejects_a_line_outside_the_declared_ladder(self):
        with self.assertRaises(ValueError):
            with_salvage_stakes(self.config, 1.0, ("none", "miracle"))

    def test_the_original_config_is_not_mutated(self):
        before = dict(self.config.hazard_multipliers["salvage"])
        with_salvage_stakes(self.config, 1.0, LINES)
        self.assertEqual(self.config.hazard_multipliers["salvage"], before)

    def test_nothing_outside_the_salvage_channel_changes(self):
        out = with_salvage_stakes(self.config, 1.0, LINES)
        for key in ("chemo", "endocrine", "radiation", "response",
                    "death_after_recurrence"):
            self.assertEqual(out.hazard_multipliers[key],
                             self.config.hazard_multipliers[key])


class CurrentScaleTests(unittest.TestCase):
    def test_todays_rung_is_located_on_the_dial(self):
        config = DynamicConfig(**RAW_V07)
        scale = current_config_scale(config)
        # 0.70 ** scale must reproduce today's 0.85.
        self.assertAlmostEqual(
            FULL_LADDER["systemic"]["death"] ** scale,
            float(config.hazard_multipliers["salvage"]["systemic"]["death"]),
            places=12)
        self.assertTrue(0.0 < scale < 1.0)

    def test_rejects_an_endpoint_that_does_not_cut_the_hazard(self):
        config = DynamicConfig(**RAW_V07)
        with self.assertRaises(ValueError):
            current_config_scale(config, ladder={
                "systemic": {"death": 1.0, "recurrence": 1.0,
                             "burden": 0.0, "toxicity": 0.0}})


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.environment = make_environment()
        self.state = replace(
            self.environment.initial_state(), phase="followup", year=1,
            recurred=True, timing="surgery_first", surgery="BCS",
            chemo="standard", endocrine="standard", radiation="local")

    def test_at_scale_zero_every_line_matches_declining(self):
        staked = staked_environment(self.environment, 0.0, LINES)
        baseline = staked.annual_event_probabilities(
            replace(self.state, salvage="none"))
        for line in LINES:
            self.assertEqual(
                staked.annual_event_probabilities(replace(self.state, salvage=line)),
                baseline)

    def test_stronger_lines_lower_the_death_probability(self):
        staked = staked_environment(self.environment, 1.0, LINES)
        probabilities = [
            staked.annual_event_probabilities(replace(self.state, salvage=line))[0]
            for line in LINES]
        self.assertEqual(probabilities, sorted(probabilities, reverse=True))

    def test_the_patient_and_risk_table_are_carried_over(self):
        staked = staked_environment(self.environment, 1.0, LINES)
        self.assertIs(staked.patient, self.environment.patient)
        self.assertEqual(staked.risk_table, self.environment.risk_table)

    def test_the_binary_arm_offers_exactly_two_legal_salvage_actions(self):
        staked = staked_environment(self.environment, 1.0, BINARY_LINES)
        actions = staked.legal_actions(replace(self.state, phase="salvage"))
        self.assertEqual(set(actions), {"none", "systemic"})


class DeclaredLadderTests(unittest.TestCase):
    def test_rows_carry_the_raw_cost_and_the_dial_settings(self):
        config = DynamicConfig(**RAW_V07)
        rows = declared_ladder(config, 1.0, LINES, cost_multiple=2.0)
        self.assertEqual(len(rows), len(LINES))
        penalty = float(config.reward["acute_toxicity_penalty"])
        for row in rows:
            self.assertEqual(row["cost_multiple"], 2.0)
            self.assertAlmostEqual(
                row["raw_cost"],
                row["burden"] + row["toxicity_probability"] * penalty, places=12)

    def test_declining_costs_nothing_at_any_setting(self):
        config = DynamicConfig(**RAW_V07)
        for multiple in (1.0, 8.0):
            row = next(r for r in declared_ladder(config, 1.0, LINES,
                                                  cost_multiple=multiple)
                       if r["line"] == "none")
            self.assertEqual(row["raw_cost"], 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
