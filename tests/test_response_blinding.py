"""Tests for the response-blinded planner (v1.7)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.dynamic.blinding import (  # noqa: E402
    BLINDING_MODES,
    UNOBSERVED,
    ResponseBlindEnvironment,
    ResponseBlindMCTSPolicy,
    blind_state,
    expected_post_chemo_size,
)
from analysis.dynamic.config import load_dynamic_config  # noqa: E402
from analysis.dynamic.environment import DynamicBreastCancerEnvironment  # noqa: E402
from analysis.dynamic.evaluation import simulate_episode  # noqa: E402
from analysis.dynamic.policies import CachedMCTSPolicy, DynamicNccnPolicy  # noqa: E402
from analysis.dynamic.schema import PatientProfile, RiskEstimate  # noqa: E402

CONFIG = load_dynamic_config(ROOT / "configs" / "dynamic_v0_5.json")


def make_patient(tumor_size_mm: float = 40.0, stage: int = 2) -> PatientProfile:
    return PatientProfile(
        patient_id="TEST-1", age=52.0, menopause="post",
        tumor_size_mm=tumor_size_mm, lymph_pos=1, stage=stage, grade=2,
        subtype="HR+/HER2-", er=1, pr=1, her2=0,
    )


def make_environment(patient: PatientProfile) -> DynamicBreastCancerEnvironment:
    # A flat risk table: every plan is equally good on prognosis, so anything
    # the searcher prefers comes from the declared channels, not from the Cox
    # model. Keeps these tests about blinding.
    from analysis.mcts.environment import all_plans

    table = {plan: RiskEstimate(five_year_os=0.85, five_year_rfs=0.78)
             for plan in all_plans()}
    return DynamicBreastCancerEnvironment(patient, table, CONFIG)


class ExpectedSizeTests(unittest.TestCase):
    def test_matches_the_configs_own_distribution(self):
        # standard: 0.25*0.5 + 0.50*0.75 + 0.25*1.0 = 0.75
        self.assertAlmostEqual(CONFIG.tumor_multiplier_mean("standard"), 0.75)
        # intensified: 0.35*0.5 + 0.45*0.75 + 0.20*1.0 = 0.7125
        self.assertAlmostEqual(CONFIG.tumor_multiplier_mean("intensified"), 0.7125)

    def test_expected_size_rounds_like_the_environment(self):
        self.assertEqual(expected_post_chemo_size(40.0, "standard", CONFIG), 30.0)
        self.assertEqual(
            expected_post_chemo_size(40.0, "intensified", CONFIG), 28.5)


class BlindStateTests(unittest.TestCase):
    def setUp(self):
        self.patient = make_patient()
        self.state = replace(
            make_environment(self.patient).initial_state(),
            phase="surgery", timing="neoadjuvant", chemo="standard",
            response="major", current_tumor_size_mm=20.0)

    def test_none_is_the_identity(self):
        self.assertEqual(
            blind_state(self.state, "none", self.patient, CONFIG), self.state)

    def test_label_hides_the_response_but_keeps_the_realised_size(self):
        blinded = blind_state(self.state, "label", self.patient, CONFIG)
        self.assertEqual(blinded.response, UNOBSERVED)
        self.assertEqual(blinded.current_tumor_size_mm, 20.0)

    def test_full_also_replaces_the_size_with_its_expectation(self):
        blinded = blind_state(self.state, "full", self.patient, CONFIG)
        self.assertEqual(blinded.response, UNOBSERVED)
        self.assertEqual(blinded.current_tumor_size_mm, 30.0)

    def test_every_mode_is_the_identity_when_nothing_was_observed(self):
        # A surgery-first patient never draws a response, so blinding one is a
        # no-op. If this ever fails, the arms stop being comparable on the
        # patients that carry no response at all.
        surgery_first = replace(
            self.state, timing="surgery_first", response=UNOBSERVED,
            current_tumor_size_mm=40.0)
        for mode in BLINDING_MODES:
            self.assertEqual(
                blind_state(surgery_first, mode, self.patient, CONFIG),
                surgery_first, mode)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            blind_state(self.state, "sideways", self.patient, CONFIG)

    def test_full_blinding_without_a_recorded_chemo_is_an_error(self):
        broken = replace(self.state, chemo=None)
        with self.assertRaises(ValueError):
            blind_state(broken, "full", self.patient, CONFIG)


class BlindEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.patient = make_patient()
        self.real = make_environment(self.patient)
        self.state = replace(
            self.real.initial_state(), phase="chemo", timing="neoadjuvant")

    def _step(self, environment, seed: int):
        return environment.step(self.state, "standard", random.Random(seed))

    def test_mode_none_reproduces_the_real_environment_exactly(self):
        blind = ResponseBlindEnvironment(self.real, "none")
        for seed in range(20):
            self.assertEqual(self._step(blind, seed), self._step(self.real, seed))

    def test_label_mode_keeps_the_realised_size(self):
        blind = ResponseBlindEnvironment(self.real, "label")
        for seed in range(20):
            real_state, _, _ = self._step(self.real, seed)
            blind_state_, _, _ = self._step(blind, seed)
            self.assertEqual(blind_state_.response, UNOBSERVED)
            self.assertEqual(blind_state_.current_tumor_size_mm,
                             real_state.current_tumor_size_mm)

    def test_full_mode_is_deterministic_in_the_size(self):
        blind = ResponseBlindEnvironment(self.real, "full")
        sizes = {self._step(blind, seed)[0].current_tumor_size_mm
                 for seed in range(50)}
        self.assertEqual(sizes, {30.0})

    def test_full_mode_still_consumes_the_same_rng_draws(self):
        # Common random numbers: the discarded draw keeps the rollout stream
        # aligned with the other modes, so between-arm differences are about
        # blinding rather than about which random numbers each arm happened to
        # reach.
        blind = ResponseBlindEnvironment(self.real, "full")
        real_rng, blind_rng = random.Random(7), random.Random(7)
        self.real.step(self.state, "standard", real_rng)
        blind.step(self.state, "standard", blind_rng)
        self.assertEqual(real_rng.random(), blind_rng.random())

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            ResponseBlindEnvironment(self.real, "sideways")


class BlindPolicyTests(unittest.TestCase):
    def setUp(self):
        self.patient = make_patient()
        self.environment = make_environment(self.patient)

    def test_mode_none_matches_the_unblinded_policy_episode_for_episode(self):
        # The blinded wrapper must be a pure pass-through at mode "none",
        # otherwise arm A stops being the v1.5 baseline and the whole
        # comparison loses its anchor.
        for seed in range(6):
            plain = CachedMCTSPolicy(self.environment, simulations=48, seed=11)
            wrapped = ResponseBlindMCTSPolicy(
                self.environment, mode="none", simulations=48, seed=11)
            expected, _ = simulate_episode(self.environment, plain, seed)
            actual, _ = simulate_episode(self.environment, wrapped, seed)
            self.assertEqual(actual, expected, seed)

    def test_blinded_decisions_are_counted_only_after_a_response(self):
        policy = ResponseBlindMCTSPolicy(
            self.environment, mode="full", simulations=48, seed=3)
        base = replace(
            self.environment.initial_state(), phase="endocrine",
            timing="neoadjuvant", chemo="standard", surgery="MAST")

        # No response drawn yet: nothing to hide, nothing to count.
        policy(replace(base, timing="surgery_first"))
        self.assertEqual(policy.blinded_decisions, 0)

        policy(replace(base, response="major", current_tumor_size_mm=20.0))
        self.assertEqual(policy.blinded_decisions, 1)

    def test_illegal_blinded_plan_falls_back_to_the_only_legal_action(self):
        # Baseline 40 mm, expected post-chemo size exactly 30 mm = the BCS cap,
        # so a blinded planner may choose BCS while a "none" responder is still
        # at 40 mm and can only have a mastectomy.
        policy = ResponseBlindMCTSPolicy(
            self.environment, mode="full", simulations=48, seed=5)
        state = replace(
            self.environment.initial_state(), phase="surgery",
            timing="neoadjuvant", chemo="standard", response="none",
            current_tumor_size_mm=40.0)
        self.assertEqual(self.environment.legal_actions(state), ("MAST",))
        self.assertEqual(policy(state), "MAST")

    def test_the_same_action_is_returned_for_indistinguishable_realisations(self):
        policy = ResponseBlindMCTSPolicy(
            self.environment, mode="full", simulations=64, seed=9)
        base = replace(
            self.environment.initial_state(), phase="endocrine",
            timing="neoadjuvant", chemo="standard", surgery="MAST")
        actions = {
            policy(replace(base, response=response,
                           current_tumor_size_mm=size))
            for response, size in
            (("major", 20.0), ("partial", 30.0), ("none", 40.0))
        }
        self.assertEqual(len(actions), 1)

    def test_nccn_never_reads_the_response(self):
        # The equality the run script asserts across arms rests on this.
        policy = DynamicNccnPolicy(self.environment)
        base = replace(
            self.environment.initial_state(), phase="endocrine",
            timing="neoadjuvant", chemo="standard", surgery="MAST")
        actions = {
            policy(replace(base, response=response))
            for response in ("major", "partial", "none", UNOBSERVED)
        }
        self.assertEqual(len(actions), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
