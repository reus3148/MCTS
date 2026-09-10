"""Load and validate transparent assumptions for the dynamic PoC."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class DynamicConfig:
    label: str
    assumption_status: str
    horizon_years: int
    eligibility: Mapping[str, Any]
    response_probabilities: Mapping[str, Mapping[str, float]]
    tumor_size_multipliers: Mapping[str, float]
    acute_toxicity_probabilities: Mapping[str, Mapping[str, float]]
    treatment_burden: Mapping[str, Mapping[str, float]]
    hazard_multipliers: Mapping[str, Any]
    reward: Mapping[str, float]
    #: Annual discount rate applied to follow-up rewards. 0.0 reproduces the
    #: v0.2/v0.3/v0.4 behaviour, where a life-year in year 5 counts exactly as
    #: much as one in year 1. Health-economic convention is 0.03. Declared
    #: explicitly so it becomes a sensitivity parameter instead of a silent
    #: default.
    discount_rate_annual: float = 0.0
    #: Divide the response hazard channel by its own expectation. Only a
    #: neoadjuvant patient ever draws a response, so without this the channel
    #: hands that arm an undeclared hazard advantage. True is the correct
    #: default; v0.2-v0.4 ran with it off, and the impact of that is measured in
    #: ``reports/environment-fix-v0.5``.
    response_channel_neutralised: bool = True

    @property
    def max_followup_reward(self) -> float:
        return self.horizon_years * (
            float(self.reward["alive_year"])
            + float(self.reward["recurrence_free_year"])
        )

    def normalized(self, raw_reward: float) -> float:
        return raw_reward / self.max_followup_reward

    def discount_factor(self, year: int) -> float:
        """Present-value weight of a reward accrued at the end of ``year``."""
        if self.discount_rate_annual == 0.0:
            return 1.0
        return 1.0 / ((1.0 + float(self.discount_rate_annual)) ** int(year))

    def tumor_multiplier_mean(self, intensity: str) -> float:
        """E[tumour-size multiplier] of the response channel under ``intensity``.

        The size a blinded planner should expect after neoadjuvant chemotherapy,
        before it learns which response was drawn. Unlike the hazard channel
        this one is *not* neutralised - shrinking the tumour is the declared
        point of neoadjuvant treatment - so the mean is a real number below 1.0.
        """
        probabilities = self.response_probabilities[intensity]
        return sum(
            float(probability) * float(self.tumor_size_multipliers[response])
            for response, probability in probabilities.items()
        )

    def response_multiplier_mean(self, intensity: str, outcome: str) -> float:
        """E[hazard multiplier] of the response channel under ``intensity``.

        Only a patient who chose neoadjuvant ever draws a response, so only they
        ever pick up a response hazard multiplier; a surgery-first patient stays
        at ``not_applicable`` = 1.0. If this expectation is not 1.0, choosing
        neoadjuvant buys a hazard advantage that no assumption ever declared.
        The environment divides by this value to keep the channel neutral, which
        preserves the *relative* ordering (major better than none) while moving
        any real timing effect into an explicit, sensitivity-testable parameter.
        """
        probabilities = self.response_probabilities[intensity]
        table = self.hazard_multipliers["response"]
        return sum(
            float(probability) * float(table[response][outcome])
            for response, probability in probabilities.items()
        )


def _validate_probabilities(config: DynamicConfig) -> None:
    for intensity, probabilities in config.response_probabilities.items():
        total = sum(float(value) for value in probabilities.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"response probabilities for {intensity!r} sum to {total}"
            )
        if any(not 0 <= float(value) <= 1 for value in probabilities.values()):
            raise ValueError(f"invalid response probability for {intensity!r}")

    for treatment, actions in config.acute_toxicity_probabilities.items():
        if any(not 0 <= float(value) <= 1 for value in actions.values()):
            raise ValueError(f"invalid toxicity probability for {treatment!r}")


def load_dynamic_config(path: str | Path) -> DynamicConfig:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    config = DynamicConfig(**raw)
    if config.horizon_years < 1:
        raise ValueError("horizon_years must be positive")
    if config.max_followup_reward <= 0:
        raise ValueError("maximum follow-up reward must be positive")
    if not -1.0 < float(config.discount_rate_annual) < 1.0:
        raise ValueError("discount_rate_annual must be in (-1, 1)")
    _validate_probabilities(config)
    return config

