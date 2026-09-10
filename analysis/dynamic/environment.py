"""Stochastic, response-adaptive breast-cancer environment for PoC v0.2.

METABRIC supplies the patient baseline and fitted five-year OS/RFS risks.
Response, intensity, field, and toxicity differences come from the explicitly
synthetic assumptions in ``configs/dynamic_poc_v0_2.json``.
"""

from __future__ import annotations

from dataclasses import replace
import math
import random
from typing import Mapping

from analysis.mcts.environment import Plan
from .config import DynamicConfig
from .schema import DynamicState, PatientProfile, RiskEstimate

Action = str


class DynamicBreastCancerEnvironment:
    """Finite-horizon generative environment with patient-specific actions."""

    def __init__(
        self,
        patient: PatientProfile,
        risk_table: Mapping[Plan, RiskEstimate],
        config: DynamicConfig,
    ) -> None:
        self.patient = patient
        self.risk_table = dict(risk_table)
        self.config = config

    def initial_state(self) -> DynamicState:
        return DynamicState(
            phase="timing",
            current_tumor_size_mm=round(self.patient.tumor_size_mm, 2),
        )

    @staticmethod
    def is_terminal(state: DynamicState) -> bool:
        return state.phase == "terminal"

    def legal_actions(self, state: DynamicState) -> tuple[Action, ...]:
        if self.is_terminal(state):
            return ()
        if state.phase == "timing":
            actions = ["surgery_first"]
            high_risk_subtypes = set(
                self.config.eligibility["neoadjuvant_high_risk_subtypes"]
            )
            if (
                self.patient.stage
                >= int(self.config.eligibility["neoadjuvant_min_stage"])
                or self.patient.subtype in high_risk_subtypes
            ):
                actions.append("neoadjuvant")
            return tuple(actions)
        if state.phase == "surgery":
            actions = []
            if (
                state.current_tumor_size_mm
                <= float(self.config.eligibility["bcs_max_tumor_mm"])
                and self.patient.stage
                <= int(self.config.eligibility["bcs_max_stage"])
            ):
                actions.append("BCS")
            actions.append("MAST")
            return tuple(actions)
        if state.phase == "chemo":
            if state.timing == "neoadjuvant":
                return ("standard", "intensified")
            if self.patient.stage == 0:
                return ("none",)
            return ("none", "standard", "intensified")
        if state.phase == "endocrine":
            if not self.patient.hr_positive:
                return ("none",)
            actions = ["none", "standard", "extended"]
            if state.toxicity_count >= int(
                self.config.eligibility["toxicity_guardrail_threshold"]
            ):
                actions.remove("extended")
            return tuple(actions)
        if state.phase == "radiation":
            actions = ["none"]
            if state.surgery == "BCS":
                actions.append("local")
            actions.append("regional")
            if state.toxicity_count >= int(
                self.config.eligibility["toxicity_guardrail_threshold"]
            ) and "regional" in actions:
                actions.remove("regional")
            return tuple(actions)
        if state.phase == "followup":
            return ("advance_year",)
        raise ValueError(f"unknown phase: {state.phase!r}")

    def step(
        self,
        state: DynamicState,
        action: Action,
        rng: random.Random,
    ) -> tuple[DynamicState, float, dict[str, object]]:
        legal = self.legal_actions(state)
        if action not in legal:
            raise ValueError(
                f"illegal action {action!r} in {state.phase!r}; legal={legal!r}"
            )

        if state.phase == "timing":
            next_phase = "surgery" if action == "surgery_first" else "chemo"
            reward = self._burden_reward("timing", action)
            return (
                replace(state, phase=next_phase, timing=action),
                reward,
                {"event": "timing_selected", "toxicity": False},
            )

        if state.phase == "surgery":
            toxicity = self._sample_toxicity("surgery", action, rng)
            next_phase = "chemo" if state.timing == "surgery_first" else "endocrine"
            next_state = replace(
                state,
                phase=next_phase,
                surgery=action,
                toxicity_count=state.toxicity_count + int(toxicity),
            )
            return next_state, self._treatment_reward("surgery", action, toxicity), {
                "event": "surgery_completed",
                "toxicity": toxicity,
            }

        if state.phase == "chemo":
            toxicity = self._sample_toxicity("chemo", action, rng)
            response = state.response
            tumor_size = state.current_tumor_size_mm
            if state.timing == "neoadjuvant":
                response, tumor_size = self._chemo_response(state, action, rng)
                next_phase = "surgery"
            else:
                next_phase = "endocrine"
            next_state = replace(
                state,
                phase=next_phase,
                chemo=action,
                response=response,
                current_tumor_size_mm=tumor_size,
                toxicity_count=state.toxicity_count + int(toxicity),
            )
            return next_state, self._treatment_reward("chemo", action, toxicity), {
                "event": "chemo_completed",
                "response": response,
                "toxicity": toxicity,
                "tumor_size_mm": tumor_size,
            }

        if state.phase == "endocrine":
            toxicity = self._sample_toxicity("endocrine", action, rng)
            next_state = replace(
                state,
                phase="radiation",
                endocrine=action,
                toxicity_count=state.toxicity_count + int(toxicity),
            )
            return next_state, self._treatment_reward("endocrine", action, toxicity), {
                "event": "endocrine_selected",
                "toxicity": toxicity,
            }

        if state.phase == "radiation":
            toxicity = self._sample_toxicity("radiation", action, rng)
            next_state = replace(
                state,
                phase="followup",
                radiation=action,
                toxicity_count=state.toxicity_count + int(toxicity),
            )
            return next_state, self._treatment_reward("radiation", action, toxicity), {
                "event": "radiation_completed",
                "toxicity": toxicity,
            }

        if state.phase == "followup":
            return self._advance_followup(state, rng)

        raise ValueError(f"cannot step from phase {state.phase!r}")

    def static_plan(self, state: DynamicState) -> Plan:
        if None in (state.surgery, state.chemo, state.endocrine, state.radiation):
            raise ValueError("treatment plan is incomplete")
        return (
            state.surgery,
            int(state.chemo != "none"),
            int(state.endocrine != "none"),
            int(state.radiation != "none"),
        )

    def annual_event_probabilities(
        self,
        state: DynamicState,
    ) -> tuple[float, float]:
        risk = self.risk_table[self.static_plan(state)]
        horizon = float(self.config.horizon_years)
        os_survival = min(max(float(risk.five_year_os), 1e-6), 0.999999)
        rfs_survival = min(max(float(risk.five_year_rfs), 1e-6), 0.999999)
        death_hazard = -math.log(os_survival) / horizon
        total_rfs_hazard = -math.log(rfs_survival) / horizon
        recurrence_hazard = max(total_rfs_hazard - death_hazard, 0.0)

        for group, action in [
            ("chemo", state.chemo),
            ("endocrine", state.endocrine),
            ("radiation", state.radiation),
        ]:
            modifier = self.config.hazard_multipliers[group][action]
            death_hazard *= float(modifier["death"])
            recurrence_hazard *= float(modifier["recurrence"])

        # Timing is an explicit, declarable channel. Any real advantage of
        # neoadjuvant belongs here, where sensitivity analysis can see it -
        # not smuggled in through the response channel below.
        timing_table = self.config.hazard_multipliers.get("timing", {})
        timing_modifier = timing_table.get(
            state.timing, {"death": 1.0, "recurrence": 1.0}
        )
        death_hazard *= float(timing_modifier["death"])
        recurrence_hazard *= float(timing_modifier["recurrence"])

        # Only a neoadjuvant patient ever draws a response, so the response
        # channel is asymmetric by construction: surgery-first stays at
        # not_applicable = 1.0 forever. Dividing by the channel's expectation
        # under the response distribution that produced it keeps the *relative*
        # ordering (major beats none) while making the channel mean-neutral, so
        # choosing neoadjuvant no longer buys an undeclared hazard advantage.
        if state.response != "not_applicable":
            modifier = self.config.hazard_multipliers["response"][state.response]
            death_scale = recurrence_scale = 1.0
            if self.config.response_channel_neutralised:
                death_scale = self.config.response_multiplier_mean(
                    state.chemo, "death")
                recurrence_scale = self.config.response_multiplier_mean(
                    state.chemo, "recurrence")
            death_hazard *= float(modifier["death"]) / death_scale
            recurrence_hazard *= float(modifier["recurrence"]) / recurrence_scale

        if state.recurred:
            death_hazard *= float(
                self.config.hazard_multipliers["death_after_recurrence"]
            )
            recurrence_hazard = 0.0

        death_probability = min(1.0 - math.exp(-death_hazard), 0.95)
        recurrence_probability = min(
            1.0 - math.exp(-recurrence_hazard),
            0.95,
        )
        return death_probability, recurrence_probability

    def _advance_followup(
        self,
        state: DynamicState,
        rng: random.Random,
    ) -> tuple[DynamicState, float, dict[str, object]]:
        death_probability, recurrence_probability = (
            self.annual_event_probabilities(state)
        )
        year = state.year + 1
        if rng.random() < death_probability:
            next_state = replace(
                state,
                phase="terminal",
                year=year,
                alive=False,
            )
            return next_state, 0.0, {
                "event": "death",
                "death_probability": death_probability,
                "recurrence_probability": recurrence_probability,
            }

        recurrence_now = (
            not state.recurred and rng.random() < recurrence_probability
        )
        recurred = state.recurred or recurrence_now
        raw_reward = float(self.config.reward["alive_year"])
        if not recurred:
            raw_reward += float(self.config.reward["recurrence_free_year"])
        next_phase = (
            "terminal"
            if year >= int(self.config.horizon_years)
            else "followup"
        )
        next_state = replace(
            state,
            phase=next_phase,
            year=year,
            recurred=recurred,
        )
        discounted = self.config.normalized(raw_reward) * (
            self.config.discount_factor(year)
        )
        return next_state, discounted, {
            "event": "recurrence" if recurrence_now else "no_event",
            "death_probability": death_probability,
            "recurrence_probability": recurrence_probability,
        }

    def _chemo_response(
        self,
        state: DynamicState,
        action: Action,
        rng: random.Random,
    ) -> tuple[str, float]:
        """The response a neoadjuvant patient draws, and the size it leaves.

        Extracted as a hook so a *blinded* environment can hide one or both
        halves without touching :meth:`step`. The response reaches a planner by
        two separate routes - the label (a terminal hazard multiplier) and the
        tumour size (which gates BCS eligibility) - and
        ``analysis/dynamic/blinding.py`` closes them one at a time. See
        ``reports/closed-loop-value-v1.7``.
        """
        response = self._sample_response(action, rng)
        size = round(
            state.current_tumor_size_mm
            * float(self.config.tumor_size_multipliers[response]),
            2,
        )
        return response, size

    def _sample_response(self, action: Action, rng: random.Random) -> str:
        probabilities = self.config.response_probabilities[action]
        draw = rng.random()
        cumulative = 0.0
        for response, probability in probabilities.items():
            cumulative += float(probability)
            if draw <= cumulative:
                return response
        return next(reversed(probabilities))

    def _sample_toxicity(
        self,
        treatment: str,
        action: Action,
        rng: random.Random,
    ) -> bool:
        probability = float(
            self.config.acute_toxicity_probabilities[treatment][action]
        )
        return rng.random() < probability

    def _burden_reward(self, treatment: str, action: Action) -> float:
        burden = float(self.config.treatment_burden[treatment][action])
        return self.config.normalized(-burden)

    def _treatment_reward(
        self,
        treatment: str,
        action: Action,
        toxicity: bool,
    ) -> float:
        raw_reward = -float(self.config.treatment_burden[treatment][action])
        if toxicity:
            raw_reward -= float(self.config.reward["acute_toxicity_penalty"])
        return self.config.normalized(raw_reward)

