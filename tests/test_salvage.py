"""Tests for the v0.6 salvage decision (reports/decision-points-v1.8)."""

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
from analysis.dynamic.policies import CachedMCTSPolicy, DynamicNccnPolicy  # noqa: E402
from analysis.dynamic.salvage import (  # noqa: E402
    FixedSalvageEnvironment,
    adaptation_opportunities,
    null_control,
    salvage_by_recurrence_year,
)
from analysis.dynamic.schema import DynamicState, PatientProfile, RiskEstimate  # noqa: E402
from analysis.mcts.environment import all_plans  # noqa: E402

CONFIG_V05 = load_dynamic_config(ROOT / "configs" / "dynamic_v0_5.json")
CONFIG_V06 = load_dynamic_config(ROOT / "configs" / "dynamic_v0_6.json")


def make_patient() -> PatientProfile:
    return PatientProfile(
        patient_id="TEST-S", age=55.0, menopause="post", tumor_size_mm=25.0,
        lymph_pos=1, stage=2, grade=2, subtype="HR+/HER2-", er=1, pr=1, her2=0,
    )


def make_environment(config: DynamicConfig, rfs: float = 0.40):
    # A low RFS so recurrences are common and the salvage phase gets exercised.
    table = {plan: RiskEstimate(five_year_os=0.85, five_year_rfs=rfs)
             for plan in all_plans()}
    return DynamicBreastCancerEnvironment(make_patient(), table, config)


def followup_state(environment, year: int = 1) -> DynamicState:
    return replace(
        environment.initial_state(), phase="followup", timing="surgery_first",
        surgery="MAST", chemo="standard", endocrine="standard",
        radiation="none", year=year)


class ConfigTests(unittest.TestCase):
    def test_v05_has_no_salvage_and_v06_has(self):
        self.assertFalse(make_environment(CONFIG_V05).salvage_enabled)
        self.assertTrue(make_environment(CONFIG_V06).salvage_enabled)

    def test_v06_differs_from_v05_only_by_the_salvage_blocks(self):
        raw05 = json.loads((ROOT / "configs" / "dynamic_v0_5.json").read_text(encoding="utf-8"))
        raw06 = json.loads((ROOT / "configs" / "dynamic_v0_6.json").read_text(encoding="utf-8"))
        for block in ("acute_toxicity_probabilities", "treatment_burden",
                      "hazard_multipliers"):
            raw06[block].pop("salvage")
        for key in ("label", "assumption_status"):
            raw05.pop(key); raw06.pop(key)
        self.assertEqual(raw05, raw06)

    def test_state_repr_hides_the_new_fields(self):
        # CachedMCTSPolicy seeds each search from repr(state); the new fields
        # must not change the repr of a pre-v0.6 state or v0.2-v1.7 stop
        # reproducing. Equality must still see them or the cache is wrong.
        plain = DynamicState(phase="followup", current_tumor_size_mm=25.0)
        decided = replace(plain, salvage="systemic", recurrence_year=2)
        self.assertEqual(repr(plain), repr(decided))
        self.assertNotEqual(plain, decided)


class PhaseTests(unittest.TestCase):
    def test_v05_never_enters_the_salvage_phase(self):
        environment = make_environment(CONFIG_V05, rfs=0.01)
        state = followup_state(environment)
        phases = set()
        for seed in range(200):
            next_state, _, info = environment.step(state, "advance_year", random.Random(seed))
            phases.add(next_state.phase)
        self.assertNotIn("salvage", phases)

    def test_recurrence_with_years_left_opens_the_salvage_phase(self):
        environment = make_environment(CONFIG_V06, rfs=0.01)
        state = followup_state(environment, year=1)
        opened = False
        for seed in range(200):
            next_state, _, info = environment.step(state, "advance_year", random.Random(seed))
            if info["event"] == "recurrence":
                self.assertEqual(next_state.phase, "salvage")
                self.assertEqual(next_state.recurrence_year, 2)
                opened = True
        self.assertTrue(opened, "no recurrence drawn in 200 seeds - lower rfs")

    def test_recurrence_in_the_final_year_stays_an_event(self):
        environment = make_environment(CONFIG_V06, rfs=0.01)
        state = followup_state(environment, year=CONFIG_V06.horizon_years - 1)
        for seed in range(200):
            next_state, _, info = environment.step(state, "advance_year", random.Random(seed))
            if info["event"] == "recurrence":
                self.assertEqual(next_state.phase, "terminal")

    def test_salvage_actions_are_the_declared_ones_for_both_policies(self):
        environment = make_environment(CONFIG_V06)
        state = replace(followup_state(environment, year=2), phase="salvage",
                        recurred=True, recurrence_year=2)
        self.assertEqual(environment.legal_actions(state), ("none", "systemic"))
        self.assertEqual(DynamicNccnPolicy(environment)(state), "systemic")

    def test_salvage_step_returns_to_followup_and_records_the_choice(self):
        environment = make_environment(CONFIG_V06)
        state = replace(followup_state(environment, year=2), phase="salvage",
                        recurred=True, recurrence_year=2)
        next_state, reward, info = environment.step(state, "systemic", random.Random(1))
        self.assertEqual(next_state.phase, "followup")
        self.assertEqual(next_state.salvage, "systemic")
        self.assertEqual(next_state.year, 2)
        self.assertLess(reward, 0.0)  # burden is charged
        self.assertEqual(info["event"], "salvage_selected")

    def test_systemic_salvage_lowers_post_recurrence_death_probability(self):
        environment = make_environment(CONFIG_V06)
        base = replace(followup_state(environment, year=2), recurred=True,
                       recurrence_year=2)
        untreated, _ = environment.annual_event_probabilities(replace(base, salvage="none"))
        treated, _ = environment.annual_event_probabilities(replace(base, salvage="systemic"))
        undecided, _ = environment.annual_event_probabilities(base)
        self.assertLess(treated, untreated)
        self.assertEqual(untreated, undecided)

    def test_full_episode_runs_under_v06_with_both_policies(self):
        environment = make_environment(CONFIG_V06, rfs=0.30)
        for policy in (DynamicNccnPolicy(environment),
                       CachedMCTSPolicy(environment, simulations=32, seed=4)):
            for seed in range(5):
                result, _ = simulate_episode(environment, policy, seed)
                self.assertIn("salvage", result)
                self.assertIn("recurrence_year", result)


class FixedSalvageTests(unittest.TestCase):
    def test_planning_environment_admits_only_the_fixed_action(self):
        environment = make_environment(CONFIG_V06)
        fixed = FixedSalvageEnvironment(environment)
        state = replace(followup_state(environment, year=2), phase="salvage",
                        recurred=True, recurrence_year=2)
        self.assertEqual(fixed.legal_actions(state), ("systemic",))
        self.assertEqual(environment.legal_actions(state), ("none", "systemic"))

    def test_everything_else_is_untouched(self):
        environment = make_environment(CONFIG_V06)
        fixed = FixedSalvageEnvironment(environment)
        for state in (environment.initial_state(), followup_state(environment)):
            self.assertEqual(fixed.legal_actions(state), environment.legal_actions(state))
            self.assertEqual(fixed.step(state, environment.legal_actions(state)[0], random.Random(3)),
                             environment.step(state, environment.legal_actions(state)[0], random.Random(3)))

    def test_mcts_on_the_fixed_environment_always_treats_at_salvage(self):
        environment = make_environment(CONFIG_V06)
        policy = CachedMCTSPolicy(FixedSalvageEnvironment(environment),
                                  simulations=16, seed=2)
        state = replace(followup_state(environment, year=2), phase="salvage",
                        recurred=True, recurrence_year=2)
        self.assertEqual(policy(state), "systemic")

    def test_undeclared_fixed_action_is_rejected(self):
        with self.assertRaises(ValueError):
            FixedSalvageEnvironment(make_environment(CONFIG_V06), action="heroic")


class NullControlTests(unittest.TestCase):
    def test_null_control_removes_benefit_and_cost_but_keeps_the_phase(self):
        raw = json.loads((ROOT / "configs" / "dynamic_v0_6.json").read_text(encoding="utf-8"))
        neutral = DynamicConfig(**null_control(raw))
        environment = make_environment(neutral)
        self.assertTrue(environment.salvage_enabled)
        for action in ("none", "systemic"):
            self.assertEqual(neutral.hazard_multipliers["salvage"][action]["death"], 1.0)
            self.assertEqual(neutral.acute_toxicity_probabilities["salvage"][action], 0.0)
            self.assertEqual(neutral.treatment_burden["salvage"][action], 0.0)
        # The original is not mutated.
        self.assertEqual(raw["hazard_multipliers"]["salvage"]["systemic"]["death"], 0.85)

    def test_null_control_needs_a_salvage_block(self):
        raw = json.loads((ROOT / "configs" / "dynamic_v0_5.json").read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            null_control(raw)


class DiagnosticTests(unittest.TestCase):
    def test_adaptation_opportunities_count_response_and_salvage(self):
        frame = pd.DataFrame({
            "response": ["not_applicable", "major", "not_applicable", "none"],
            "salvage": [None, None, "systemic", "none"],
        })
        self.assertEqual(list(adaptation_opportunities(frame)), [0, 1, 1, 2])

    def test_adaptation_opportunities_without_a_salvage_column(self):
        frame = pd.DataFrame({"response": ["major", "not_applicable"]})
        self.assertEqual(list(adaptation_opportunities(frame)), [1, 0])

    def test_salvage_by_recurrence_year_reports_decline_share(self):
        frame = pd.DataFrame({
            "salvage": ["none", "systemic", "none", None],
            "recurrence_year": [4, 2, 4, None],
        })
        table = salvage_by_recurrence_year(frame).set_index("recurrence_year")
        self.assertEqual(int(table.loc[4, "decisions"]), 2)
        self.assertEqual(table.loc[4, "declined_pct"], 100.0)
        self.assertEqual(table.loc[2, "declined_pct"], 0.0)

    def test_salvage_by_recurrence_year_handles_no_decisions(self):
        frame = pd.DataFrame({"salvage": [None], "recurrence_year": [None]})
        self.assertTrue(salvage_by_recurrence_year(frame).empty)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
