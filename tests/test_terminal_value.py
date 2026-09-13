"""Tests for the v0.7 terminal value (reports/horizon-v1.9)."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import random
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.dynamic.config import DynamicConfig, load_dynamic_config  # noqa: E402
from analysis.dynamic.environment import DynamicBreastCancerEnvironment  # noqa: E402
from analysis.dynamic.evaluation import simulate_episode  # noqa: E402
from analysis.dynamic.policies import DynamicNccnPolicy  # noqa: E402
from analysis.dynamic.salvage import without_salvage  # noqa: E402
from analysis.dynamic.schema import PatientProfile, RiskEstimate  # noqa: E402
from analysis.genie.survival import post_advanced_survival  # noqa: E402
from analysis.mcts.environment import all_plans  # noqa: E402

RAW_V07 = json.loads((ROOT / "configs" / "dynamic_v0_7.json").read_text(encoding="utf-8"))
CONFIG_V06 = load_dynamic_config(ROOT / "configs" / "dynamic_v0_6.json")
CONFIG_V07 = load_dynamic_config(ROOT / "configs" / "dynamic_v0_7.json")


def make_patient() -> PatientProfile:
    return PatientProfile(
        patient_id="TEST-T", age=55.0, menopause="post", tumor_size_mm=25.0,
        lymph_pos=1, stage=2, grade=2, subtype="HR+/HER2-", er=1, pr=1, her2=0,
    )


def make_environment(config: DynamicConfig, os=0.85, rfs=0.78):
    table = {plan: RiskEstimate(five_year_os=os, five_year_rfs=rfs)
             for plan in all_plans()}
    return DynamicBreastCancerEnvironment(make_patient(), table, config)


def final_state(environment, **overrides):
    return replace(
        environment.initial_state(), phase="terminal", timing="surgery_first",
        surgery="MAST", chemo="standard", endocrine="standard", radiation="none",
        year=environment.config.horizon_years, **overrides)


class ConfigTests(unittest.TestCase):
    def test_v07_differs_from_v06_only_by_the_tail(self):
        raw06 = json.loads((ROOT / "configs" / "dynamic_v0_6.json").read_text(encoding="utf-8"))
        raw07 = json.loads((ROOT / "configs" / "dynamic_v0_7.json").read_text(encoding="utf-8"))
        self.assertEqual(raw07.pop("terminal_tail_years"), 10)
        for key in ("label", "assumption_status"):
            raw06.pop(key)
            raw07.pop(key)
        self.assertEqual(raw06, raw07)

    def test_default_tail_is_zero_so_older_configs_are_unchanged(self):
        self.assertEqual(CONFIG_V06.terminal_tail_years, 0)

    def test_negative_tail_is_rejected(self):
        raw = dict(RAW_V07)
        raw["terminal_tail_years"] = -1
        path = ROOT / "reports" / "_tmp_negative_tail.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        try:
            with self.assertRaises(ValueError):
                load_dynamic_config(path)
        finally:
            path.unlink()

    def test_without_salvage_removes_the_three_blocks(self):
        stripped = without_salvage(RAW_V07)
        for block in ("hazard_multipliers", "acute_toxicity_probabilities",
                      "treatment_burden"):
            self.assertNotIn("salvage", stripped[block])
            self.assertIn("salvage", RAW_V07[block])  # original untouched
        self.assertEqual(stripped["terminal_tail_years"], 10)
        self.assertFalse(make_environment(DynamicConfig(**stripped)).salvage_enabled)


class TerminalValueTests(unittest.TestCase):
    def test_zero_tail_gives_zero(self):
        environment = make_environment(CONFIG_V06)
        self.assertEqual(environment.terminal_value(final_state(environment)), 0.0)

    def test_dead_patient_gets_nothing(self):
        environment = make_environment(CONFIG_V07)
        self.assertEqual(
            environment.terminal_value(final_state(environment, alive=False)), 0.0)

    def test_immortal_patient_gets_the_full_tail(self):
        # OS and RFS ~1 -> death and recurrence probabilities ~0 -> every tail
        # year pays alive_year + recurrence_free_year, normalised.
        environment = make_environment(CONFIG_V07, os=0.999999, rfs=0.999999)
        value = environment.terminal_value(final_state(environment))
        per_year = CONFIG_V07.normalized(
            CONFIG_V07.reward["alive_year"] + CONFIG_V07.reward["recurrence_free_year"])
        self.assertAlmostEqual(value, 10 * per_year, places=4)

    def test_recurred_patient_is_worth_less_than_a_recurrence_free_one(self):
        environment = make_environment(CONFIG_V07)
        free = environment.terminal_value(final_state(environment))
        recurred = environment.terminal_value(final_state(environment, recurred=True))
        self.assertLess(recurred, free)
        self.assertGreater(recurred, 0.0)

    def test_salvage_raises_the_terminal_value_of_a_recurred_patient(self):
        environment = make_environment(CONFIG_V07)
        untreated = environment.terminal_value(
            final_state(environment, recurred=True, salvage="none"))
        treated = environment.terminal_value(
            final_state(environment, recurred=True, salvage="systemic"))
        self.assertGreater(treated, untreated)

    def test_tail_is_deterministic_and_bounded(self):
        environment = make_environment(CONFIG_V07)
        state = final_state(environment)
        values = {environment.terminal_value(state) for _ in range(5)}
        self.assertEqual(len(values), 1)
        cap = 10 * CONFIG_V07.normalized(
            CONFIG_V07.reward["alive_year"] + CONFIG_V07.reward["recurrence_free_year"])
        self.assertLess(values.pop(), cap)

    def test_terminal_value_is_paid_once_on_reaching_the_horizon(self):
        environment = make_environment(CONFIG_V07, os=0.999999, rfs=0.999999)
        state = replace(final_state(environment), phase="followup",
                        year=CONFIG_V07.horizon_years - 1)
        next_state, reward, _ = environment.step(state, "advance_year", random.Random(1))
        self.assertEqual(next_state.phase, "terminal")
        year_reward = CONFIG_V07.normalized(
            CONFIG_V07.reward["alive_year"] + CONFIG_V07.reward["recurrence_free_year"])
        self.assertAlmostEqual(
            reward, year_reward + environment.terminal_value(next_state), places=10)

    def test_v06_episodes_differ_from_v07_only_in_the_final_reward(self):
        # With tail 0 the terminal-value branch adds exactly 0.0, so a v0.6
        # episode is what it was before v0.7 existed. Two environments that
        # differ only in the tail must therefore share every action and every
        # draw, and differ only in what a survivor is paid at the end.
        env06 = make_environment(CONFIG_V06)
        env07 = make_environment(CONFIG_V07)
        for seed in range(6):
            r06, t06 = simulate_episode(env06, DynamicNccnPolicy(env06), seed, include_trace=True)
            r07, t07 = simulate_episode(env07, DynamicNccnPolicy(env07), seed, include_trace=True)
            self.assertEqual([s["action"] for s in t06], [s["action"] for s in t07])
            self.assertEqual(r06["survived_5y"], r07["survived_5y"])
            if r07["survived_5y"]:
                self.assertGreater(r07["utility"], r06["utility"])
            else:
                self.assertAlmostEqual(r07["utility"], r06["utility"], places=12)


class SurvivalHelperTests(unittest.TestCase):
    def test_kaplan_meier_on_a_synthetic_cohort(self):
        frame = pd.DataFrame({
            "tt_os_adv_yrs": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, None],
            "os_adv_status": [1, 1, 0, 1, 1, 0, None],
        })
        summary = post_advanced_survival(frame, years=(1, 3, 5))
        self.assertEqual(summary["n"], 6)
        self.assertEqual(summary["events"], 4)
        self.assertLess(summary["survival"]["5"], summary["survival"]["1"])
        self.assertGreater(summary["median_years"], 1.0)

    def test_needs_at_least_one_advanced_row(self):
        with self.assertRaises(ValueError):
            post_advanced_survival(
                pd.DataFrame({"tt_os_adv_yrs": [None], "os_adv_status": [None]}))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
