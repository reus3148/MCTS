"""Baseline and cached MCTS policies for the dynamic environment."""

from __future__ import annotations

import hashlib
import math

import pandas as pd

from analysis.nccn_policy import nccn_plan
from .environment import DynamicBreastCancerEnvironment
from .schema import DynamicState
from .search import StochasticSearchResult, stochastic_mcts_search


class DynamicNccnPolicy:
    """Map the existing simplified NCCN policy onto the expanded environment."""

    def __init__(self, environment: DynamicBreastCancerEnvironment) -> None:
        self.environment = environment
        patient = environment.patient
        row = pd.Series({
            "tumor_size_mm": patient.tumor_size_mm,
            "lymph_pos": patient.lymph_pos,
            "stage": patient.stage,
            "grade": patient.grade,
            "subtype": patient.subtype,
            "er": patient.er,
            "pr": patient.pr,
            "her2": patient.her2,
        })
        self.plan = nccn_plan(row)
        if self.plan is None:
            raise ValueError("NCCN baseline requires a complete simplified plan")

    def __call__(self, state: DynamicState) -> str:
        legal = self.environment.legal_actions(state)
        if len(legal) == 1:
            return legal[0]
        if state.phase == "timing":
            preferred = "surgery_first"
        elif state.phase == "surgery":
            preferred = str(self.plan[0])
        elif state.phase == "chemo":
            preferred = "standard" if int(self.plan[1]) else "none"
        elif state.phase == "endocrine":
            preferred = "standard" if int(self.plan[2]) else "none"
        elif state.phase == "radiation":
            if int(self.plan[3]):
                preferred = "local" if state.surgery == "BCS" else "regional"
            else:
                preferred = "none"
        elif state.phase == "salvage":
            # Guideline care after recurrence is systemic therapy for every
            # subtype (endocrine-based, HER2-directed or cytotoxic); the
            # simplified rule keeps only the decision to treat.
            preferred = "systemic"
        else:
            preferred = legal[0]
        return preferred if preferred in legal else legal[0]


class CachedMCTSPolicy:
    """Re-plan at each observed state and cache repeated state decisions."""

    def __init__(
        self,
        environment: DynamicBreastCancerEnvironment,
        simulations: int = 128,
        exploration_weight: float = math.sqrt(2.0),
        seed: int = 0,
    ) -> None:
        self.environment = environment
        self.simulations = simulations
        self.exploration_weight = exploration_weight
        self.seed = seed
        self.cache: dict[DynamicState, StochasticSearchResult] = {}

    def _state_seed(self, state: DynamicState) -> int:
        payload = (
            f"{self.seed}|{self.environment.patient.patient_id}|{state!r}"
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    def __call__(self, state: DynamicState) -> str:
        legal = self.environment.legal_actions(state)
        if len(legal) == 1:
            return legal[0]
        if state not in self.cache:
            self.cache[state] = stochastic_mcts_search(
                self.environment,
                state,
                simulations=self.simulations,
                exploration_weight=self.exploration_weight,
                seed=self._state_seed(state),
            )
        return self.cache[state].action

