"""The published NCCN policy spec, pinned against the implementation.

``docs/nccn-policy-spec.md`` is the appendix a reviewer reads when they ask
"how did you implement NCCN?". Until now the rules lived only in
``analysis/nccn_policy.py``, and a documented rule set that nobody checks goes
stale the first time someone edits a threshold.

Every table row in that document has a test here. These are deliberately
written as **boundary cases** - the exact value on each side of every cut -
because a threshold that moves by one is precisely the change a prose
description would not catch.

The two asymmetry tests at the end are the ones that matter most for the
research question: they pin the fact that the guideline policy **cannot
reach** ``intensified``, ``extended`` or ``neoadjuvant``. v1.5 measured that
asymmetry at 41% of the utility gap, so if a future edit quietly lets the
baseline use those rungs, the headline changes and these tests say so.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.dynamic.config import DynamicConfig  # noqa: E402
from analysis.dynamic.environment import DynamicBreastCancerEnvironment  # noqa: E402
from analysis.dynamic.policies import DynamicNccnPolicy  # noqa: E402
from analysis.dynamic.schema import PatientProfile, RiskEstimate  # noqa: E402
from analysis.mcts.environment import all_plans  # noqa: E402
from analysis.nccn_policy import (  # noqa: E402
    nccn_chemo,
    nccn_hormone,
    nccn_plan,
    nccn_radio,
    nccn_surgery,
)

RAW_V05 = json.loads((ROOT / "configs" / "dynamic_v0_5.json").read_text(encoding="utf-8"))


def patient_row(**overrides) -> pd.Series:
    base = {
        "tumor_size_mm": 20.0, "lymph_pos": 0, "stage": 1, "grade": 1,
        "subtype": "HR+/HER2-", "er": 1, "pr": 1, "her2": 0,
    }
    base.update(overrides)
    return pd.Series(base)


# --- §1.1 수술 --------------------------------------------------------------

class SurgeryRuleTests(unittest.TestCase):
    """BCS iff size <= 30 mm AND stage <= 2."""

    def test_both_conditions_met_gives_bcs(self):
        self.assertEqual(nccn_surgery(patient_row(tumor_size_mm=30, stage=2)), "BCS")

    def test_one_millimetre_over_gives_mastectomy(self):
        self.assertEqual(nccn_surgery(patient_row(tumor_size_mm=31, stage=2)), "MAST")

    def test_one_stage_over_gives_mastectomy(self):
        self.assertEqual(nccn_surgery(patient_row(tumor_size_mm=30, stage=3)), "MAST")

    def test_missing_size_or_stage_gives_no_recommendation(self):
        self.assertIsNone(nccn_surgery(patient_row(tumor_size_mm=pd.NA)))
        self.assertIsNone(nccn_surgery(patient_row(stage=pd.NA)))


# --- §1.2 항암 --------------------------------------------------------------

class ChemoRuleTests(unittest.TestCase):
    def test_stage_below_one_declines_regardless_of_subtype(self):
        self.assertEqual(nccn_chemo(patient_row(stage=0, subtype="TNBC")), 0)

    def test_three_subtypes_always_receive_chemotherapy(self):
        for subtype in ("HR-/HER2+", "TNBC", "HR+/HER2+"):
            self.assertEqual(
                nccn_chemo(patient_row(subtype=subtype, stage=1,
                                       lymph_pos=0, tumor_size_mm=5, grade=1)),
                1, subtype)

    def test_hr_positive_her2_negative_needs_a_risk_factor(self):
        low = patient_row(subtype="HR+/HER2-", lymph_pos=0,
                          tumor_size_mm=20, grade=1)
        self.assertEqual(nccn_chemo(low), 0)

    def test_each_risk_factor_alone_is_enough(self):
        for override in ({"lymph_pos": 1}, {"tumor_size_mm": 21}, {"grade": 3}):
            fields = {"subtype": "HR+/HER2-", "lymph_pos": 0,
                      "tumor_size_mm": 20, "grade": 1, **override}
            self.assertEqual(nccn_chemo(patient_row(**fields)), 1, override)

    def test_the_size_cut_is_strictly_greater_than_twenty(self):
        self.assertEqual(nccn_chemo(patient_row(subtype="HR+/HER2-",
                                                tumor_size_mm=20, grade=1)), 0)
        self.assertEqual(nccn_chemo(patient_row(subtype="HR+/HER2-",
                                                tumor_size_mm=20.1, grade=1)), 1)

    def test_grade_two_is_not_a_risk_factor(self):
        self.assertEqual(nccn_chemo(patient_row(subtype="HR+/HER2-", grade=2,
                                                lymph_pos=0, tumor_size_mm=20)), 0)

    def test_unknown_subtype_gives_no_recommendation(self):
        self.assertIsNone(nccn_chemo(patient_row(subtype="그 외")))


# --- §1.3 내분비 ------------------------------------------------------------

class HormoneRuleTests(unittest.TestCase):
    def test_either_receptor_positive_is_enough(self):
        self.assertEqual(nccn_hormone(patient_row(er=1, pr=0)), 1)
        self.assertEqual(nccn_hormone(patient_row(er=0, pr=1)), 1)

    def test_both_negative_declines(self):
        self.assertEqual(nccn_hormone(patient_row(er=0, pr=0)), 0)

    def test_one_missing_still_decides_on_the_other(self):
        self.assertEqual(nccn_hormone(patient_row(er=pd.NA, pr=1)), 1)

    def test_both_missing_gives_no_recommendation(self):
        self.assertIsNone(nccn_hormone(patient_row(er=pd.NA, pr=pd.NA)))


# --- §1.4 방사선 ------------------------------------------------------------

class RadiationRuleTests(unittest.TestCase):
    def test_breast_conservation_always_gets_radiation(self):
        self.assertEqual(nccn_radio(patient_row(lymph_pos=0, stage=1), "BCS"), 1)

    def test_mastectomy_needs_four_nodes_or_stage_three(self):
        self.assertEqual(nccn_radio(patient_row(lymph_pos=3, stage=2), "MAST"), 0)
        self.assertEqual(nccn_radio(patient_row(lymph_pos=4, stage=2), "MAST"), 1)
        self.assertEqual(nccn_radio(patient_row(lymph_pos=0, stage=3), "MAST"), 1)

    def test_it_keys_on_the_recommended_surgery_not_the_actual_one(self):
        # Documented in the spec: the policy stays internally consistent by
        # reading its own surgery recommendation.
        row = patient_row(lymph_pos=0, stage=1)
        self.assertEqual(nccn_radio(row, "BCS"), 1)
        self.assertEqual(nccn_radio(row, "MAST"), 0)

    def test_no_surgery_recommendation_gives_no_radiation_recommendation(self):
        self.assertIsNone(nccn_radio(patient_row(), None))


# --- §1.5 완전한 계획 --------------------------------------------------------

class PlanCompletenessTests(unittest.TestCase):
    def test_any_missing_decision_voids_the_plan(self):
        for override in ({"stage": pd.NA}, {"tumor_size_mm": pd.NA},
                         {"subtype": pd.NA}):
            self.assertIsNone(nccn_plan(patient_row(**override)), override)

    def test_a_complete_row_gives_a_four_tuple(self):
        plan = nccn_plan(patient_row())
        self.assertIsInstance(plan, tuple)
        self.assertEqual(len(plan), 4)


# --- §2 환경으로의 사상 ------------------------------------------------------

def make_environment(subtype="HR+/HER2-", config_path="dynamic_v0_5.json",
                     **overrides):
    defaults = dict(patient_id="P", age=55.0, menopause="post",
                    tumor_size_mm=25.0, lymph_pos=1, stage=2, grade=2,
                    subtype=subtype, er=1, pr=1, her2=0)
    defaults.update(overrides)
    patient = PatientProfile(**defaults)
    table = {plan: RiskEstimate(five_year_os=0.85, five_year_rfs=0.78)
             for plan in all_plans()}
    raw = json.loads((ROOT / "configs" / config_path).read_text(encoding="utf-8"))
    return DynamicBreastCancerEnvironment(patient, table, DynamicConfig(**raw))


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.environment = make_environment()
        self.policy = DynamicNccnPolicy(self.environment)
        self.start = self.environment.initial_state()

    def test_timing_is_always_surgery_first(self):
        self.assertEqual(self.policy(replace(self.start, phase="timing")),
                         "surgery_first")

    def test_chemo_maps_to_standard_or_none(self):
        choice = self.policy(replace(self.start, phase="chemo"))
        self.assertIn(choice, ("standard", "none"))

    def test_radiation_follows_the_surgery_actually_taken(self):
        # Assert the premise rather than guarding on it: if this patient stops
        # being a radiation case the test must fail loudly, not pass silently.
        self.assertEqual(self.policy.plan[3], 1,
                         "fixture patient must be a radiation case")
        for surgery, expected in (("BCS", "local"), ("MAST", "regional")):
            state = replace(self.start, phase="radiation", surgery=surgery)
            self.assertEqual(self.policy(state), expected)

    def test_salvage_is_always_systemic(self):
        # v0.5 has no salvage block, so a test written against it would never
        # run its body - exactly the vacuous check v2.4 caught. Build v0.7,
        # which is the config the spec's salvage row describes.
        environment = make_environment(config_path="dynamic_v0_7.json")
        self.assertIn("salvage", environment.config.hazard_multipliers,
                      "v0.7 must offer the salvage channel")
        policy = DynamicNccnPolicy(environment)
        state = replace(environment.initial_state(), phase="salvage",
                        year=2, recurred=True)
        self.assertEqual(set(environment.legal_actions(state)),
                         {"none", "systemic"})
        self.assertEqual(policy(state), "systemic")

    def test_an_illegal_preference_falls_back_to_a_legal_action(self):
        # HR-negative patients have no endocrine option; the policy must not
        # return "standard" into a one-action phase.
        environment = make_environment(subtype="TNBC", er=0, pr=0)
        policy = DynamicNccnPolicy(environment)
        state = replace(environment.initial_state(), phase="endocrine")
        self.assertIn(policy(state), environment.legal_actions(state))


# --- §2.1 비대칭 — 연구 질문에 직접 걸리는 부분 -------------------------------

class AsymmetryTests(unittest.TestCase):
    """The guideline policy cannot reach the ladder's upper rungs.

    v1.5 measured this asymmetry at 41% of the utility gap. If a future edit
    lets the baseline use these levels, the headline changes - so it is pinned
    here rather than only described in prose.
    """

    UNREACHABLE = {
        "chemo": "intensified",
        "endocrine": "extended",
        "timing": "neoadjuvant",
    }

    SUBTYPES = (("HR+/HER2-", 1, 1), ("HR+/HER2+", 1, 1),
                ("TNBC", 0, 0), ("HR-/HER2+", 0, 0))

    def test_the_baseline_never_selects_an_upper_rung(self):
        for subtype, er, pr in self.SUBTYPES:
            environment = make_environment(subtype=subtype, er=er, pr=pr)
            policy = DynamicNccnPolicy(environment)
            start = environment.initial_state()
            for phase, forbidden in self.UNREACHABLE.items():
                state = replace(start, phase=phase)
                legal = environment.legal_actions(state)
                if phase == "endocrine" and not environment.patient.hr_positive:
                    # Receptor eligibility, not asymmetry: the environment
                    # offers HR-negative patients no endocrine option at all,
                    # and it denies it to *both* policies equally.
                    self.assertEqual(legal, ("none",), subtype)
                    continue
                # Otherwise assert the premise. Silently skipping an absent
                # level would make this pass for the wrong reason - the
                # asymmetry exists only while the searcher can reach what the
                # baseline cannot.
                self.assertIn(forbidden, legal, f"{subtype} / {phase}")
                self.assertNotEqual(policy(state), forbidden,
                                    f"{subtype} / {phase}")

    def test_hr_negative_endocrine_is_denied_to_both_policies(self):
        # The one place a level is missing for a reason that is *not* an
        # asymmetry. Pinned so it is never mistaken for one.
        for subtype, er, pr in self.SUBTYPES:
            environment = make_environment(subtype=subtype, er=er, pr=pr)
            state = replace(environment.initial_state(), phase="endocrine")
            legal = environment.legal_actions(state)
            if environment.patient.hr_positive:
                self.assertEqual(set(legal), {"none", "standard", "extended"}, subtype)
            else:
                self.assertEqual(legal, ("none",), subtype)

    def test_the_searcher_can_reach_them(self):
        # The other half of the asymmetry: these levels exist and are legal.
        environment = make_environment()
        start = environment.initial_state()
        self.assertIn("intensified",
                      environment.legal_actions(replace(start, phase="chemo")))
        self.assertIn("extended",
                      environment.legal_actions(replace(start, phase="endocrine")))
        self.assertIn("neoadjuvant",
                      environment.legal_actions(replace(start, phase="timing")))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
