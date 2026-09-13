"""Giving the searching policy a second place to adapt (v0.6 / v1.8).

v1.7 found that the closed-loop share of the MCTS advantage is indistinguishable
from zero, and blamed the environment: a response is drawn only when the policy
chose neoadjuvant chemotherapy (20.8% of episodes), and after that nothing the
patient does or suffers ever opens another decision. v1.6 had just shown that
recorded care is the opposite - a "Progressing" scan is followed by a new
regimen 71.3% of the time.

The v0.6 environment opens one more decision at the event that opens one in
real care: **recurrence**. A recurrence with follow-up years still to play moves
the episode into a ``salvage`` phase with two actions, ``none`` and
``systemic``, before follow-up resumes. Both policies see the same two actions.
Guideline care after recurrence is systemic therapy, so ``DynamicNccnPolicy``
always treats; MCTS may decline.

This module holds the two pieces the v1.8 run needs beyond the environment:

``FixedSalvageEnvironment``
    A *planning* environment in which the salvage phase has exactly one legal
    action. A ``CachedMCTSPolicy`` built on it plans and acts as if the salvage
    decision were not its to make - the same trick v1.7 used to blind the
    planner, applied at the action level. The difference between an MCTS that
    decides salvage and one that defers it to the guideline rule is the value
    of the new decision point.

``null_control``
    A config in which the salvage phase exists but its actions have no effect:
    hazard 1.0, toxicity 0, burden 0. If the gap moves under that config, the
    *phase* is doing something the *treatment* is not, and nothing else in the
    run can be read.
"""

from __future__ import annotations

import copy

import pandas as pd

from .environment import Action, DynamicBreastCancerEnvironment
from .schema import DynamicState

#: The guideline's answer at the salvage decision (see DynamicNccnPolicy).
GUIDELINE_SALVAGE: Action = "systemic"


class FixedSalvageEnvironment(DynamicBreastCancerEnvironment):
    """Planning environment whose salvage phase admits one action only.

    Everything else - transitions, hazards, rewards, random draws - is the real
    environment's. Because the fixed action is legal in the real environment
    too, a policy planned here needs no fallback when it acts for real.
    """

    def __init__(
        self,
        environment: DynamicBreastCancerEnvironment,
        action: Action = GUIDELINE_SALVAGE,
    ) -> None:
        super().__init__(environment.patient, environment.risk_table,
                         environment.config)
        if self.salvage_enabled and action not in self.config.hazard_multipliers["salvage"]:
            raise ValueError(f"{action!r} is not a declared salvage action")
        self.fixed_action = action

    def legal_actions(self, state: DynamicState) -> tuple[Action, ...]:
        if state.phase == "salvage":
            return (self.fixed_action,)
        return super().legal_actions(state)


def null_control(config: dict) -> dict:
    """The salvage phase with its teeth removed.

    Returns a deep copy of ``config`` (a raw JSON dict) in which every salvage
    action has hazard 1.0 on death and recurrence, toxicity probability 0 and
    burden 0. The phase is still entered and still costs the planner a decision
    and one random draw, so this is a control for the *phase*, not for the
    treatment.
    """
    if "salvage" not in config["hazard_multipliers"]:
        raise ValueError("config declares no salvage block to neutralise")
    data = copy.deepcopy(config)
    for action in data["hazard_multipliers"]["salvage"]:
        data["hazard_multipliers"]["salvage"][action] = {
            "death": 1.0, "recurrence": 1.0}
        data["acute_toxicity_probabilities"]["salvage"][action] = 0.0
        data["treatment_burden"]["salvage"][action] = 0.0
    return data


def adaptation_opportunities(episodes: pd.DataFrame) -> pd.Series:
    """Number of places each episode gave the policy something to adapt to.

    A neoadjuvant response is one (v0.5's only one); a salvage decision is the
    other (v0.6). Counted from the episode summary ``run_policy_episodes``
    returns, so it works on frames from either environment.
    """
    response = (episodes["response"].astype(str) != "not_applicable").astype(int)
    salvage = episodes["salvage"].notna().astype(int) if "salvage" in episodes \
        else pd.Series(0, index=episodes.index)
    return response + salvage


def salvage_by_recurrence_year(episodes: pd.DataFrame) -> pd.DataFrame:
    """How the salvage choice depends on when the recurrence happened.

    The horizon-artefact diagnostic: a finite five-year horizon makes late
    salvage look worthless to any expected-utility planner, whether or not it
    would be worthless to the patient. If declines pile up in the last year the
    value of the decision point is partly an artefact of where the game ends.
    """
    decided = episodes.dropna(subset=["salvage"])
    if decided.empty:
        return pd.DataFrame(columns=["recurrence_year", "decisions", "declined_pct"])
    grouped = decided.groupby("recurrence_year")
    summary = pd.DataFrame({
        "decisions": grouped.size(),
        "declined_pct": grouped["salvage"].apply(
            lambda values: float((values == "none").mean() * 100)),
    })
    return summary.reset_index()
