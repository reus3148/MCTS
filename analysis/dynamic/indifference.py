"""Where does the salvage decision actually split? (v2.0)

v1.8 opened a decision at recurrence and the searcher declined it 85% of the
time; v1.9 credited life past the horizon and the same searcher accepted it
98.7% of the time. Neither is a decision. A decision point carries value only
where the two options are close to indifferent for a meaningful share of
patients *and* which side wins depends on something the policy can observe.

Before spending forty minutes of search per parameter setting, this module
asks the environment's own arithmetic. For a patient who has just recurred in
year *y* with a given plan, the expected remaining reward of treating versus
not treating is a deterministic sum over the years left in the horizon plus
the terminal value - no random draws, no search. Sweeping the declared salvage
hazard ratio then shows, for every patient and recurrence year, where treating
stops paying. That map says two things the experiment needs to know first:

* the hazard ratio at which roughly half of the (patient, year) cells favour
  treating - the indifference point - and
* whether that point *differs between patients*. If every patient flips at the
  same ratio, the decision is global and a searcher gains nothing from being
  able to make it; if it spreads, there is something to adapt to.

Chess players do this too: before calculating a line you evaluate the position
statically to see whether there is anything to calculate.
"""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pandas as pd

from .environment import DynamicBreastCancerEnvironment
from .schema import DynamicState


def expected_remaining_utility(
    environment: DynamicBreastCancerEnvironment,
    state: DynamicState,
) -> float:
    """Expected discounted reward from ``state`` to the end of the game.

    For a follow-up state only (the treatment phases are over). Runs the same
    annual event model the environment uses - death first, then recurrence
    among survivors, then the year's reward on the resulting status - as an
    expectation rather than a simulation, and adds the terminal value at the
    horizon. Salvage is never re-opened here: a patient who recurs inside this
    calculation carries the post-recurrence hazard with no salvage, exactly as
    in :meth:`DynamicBreastCancerEnvironment.terminal_value`.
    """
    if state.phase not in ("followup", "salvage"):
        raise ValueError("expected_remaining_utility is defined for follow-up states")
    config = environment.config
    horizon = int(config.horizon_years)
    alive_year = float(config.reward["alive_year"])
    free_year = float(config.reward["recurrence_free_year"])

    p_death, p_recurrence = environment.annual_event_probabilities(state)
    recurred_state = state if state.recurred else replace(
        state, recurred=True, salvage=None)
    p_death_recurred, _ = environment.annual_event_probabilities(recurred_state)

    alive_free = 0.0 if state.recurred else 1.0
    alive_recurred = 1.0 if state.recurred else 0.0
    total = 0.0
    for year in range(int(state.year) + 1, horizon + 1):
        survivors_free = alive_free * (1.0 - p_death)
        new_free = survivors_free * (1.0 - p_recurrence)
        new_recurred = (survivors_free * p_recurrence
                        + alive_recurred * (1.0 - p_death_recurred))
        raw = new_free * (alive_year + free_year) + new_recurred * alive_year
        total += config.normalized(raw) * config.discount_factor(year)
        alive_free, alive_recurred = new_free, new_recurred

    # Terminal value at the horizon, weighted by the probability of being in
    # each state when it arrives.
    end_free = replace(state, phase="terminal", year=horizon, recurred=False)
    end_recurred = replace(recurred_state, phase="terminal", year=horizon,
                           recurred=True)
    total += alive_free * environment.terminal_value(end_free)
    total += alive_recurred * environment.terminal_value(end_recurred)
    return total


def salvage_cost(environment: DynamicBreastCancerEnvironment, action: str) -> float:
    """Expected immediate cost of a salvage action, in normalised utility."""
    config = environment.config
    burden = float(config.treatment_burden["salvage"][action])
    toxicity = float(config.acute_toxicity_probabilities["salvage"][action])
    penalty = float(config.reward["acute_toxicity_penalty"])
    return config.normalized(burden + toxicity * penalty)


def salvage_advantage(
    environment: DynamicBreastCancerEnvironment,
    state: DynamicState,
    treat: str = "systemic",
    decline: str = "none",
) -> float:
    """E[utility | treat] minus E[utility | decline] at a salvage state.

    Positive means treating pays for this patient at this recurrence year
    under this environment's declared parameters.
    """
    if state.phase != "salvage":
        raise ValueError("salvage_advantage needs a salvage-phase state")
    after = replace(state, phase="followup")
    value_treat = (expected_remaining_utility(environment, replace(after, salvage=treat))
                   - salvage_cost(environment, treat))
    value_decline = (expected_remaining_utility(environment, replace(after, salvage=decline))
                     - salvage_cost(environment, decline))
    return value_treat - value_decline


def indifference_map(
    environments: dict[str, DynamicBreastCancerEnvironment],
    plans: dict[str, DynamicState],
    hazard_ratios: np.ndarray,
    years: tuple[int, ...],
) -> pd.DataFrame:
    """Salvage advantage for every patient x recurrence year x hazard ratio.

    ``environments`` map patient id -> environment built on the *base* config;
    the hazard ratio is swept by rebuilding the config's salvage block, so the
    caller supplies a factory rather than a ready environment when it needs
    the sweep - see :func:`sweep_environments`. ``plans`` map patient id -> a
    follow-up state carrying the plan the patient was treated with.
    """
    rows = []
    for patient_id, environment in environments.items():
        base = plans[patient_id]
        for year in years:
            state = replace(base, phase="salvage", year=int(year), recurred=True,
                            recurrence_year=int(year))
            for ratio in hazard_ratios:
                rows.append({
                    "patient_id": patient_id,
                    "recurrence_year": int(year),
                    "hazard_ratio": float(ratio),
                    "advantage": salvage_advantage(
                        with_salvage_ratio(environment, float(ratio)), state),
                })
    return pd.DataFrame(rows)


def with_salvage_ratio(
    environment: DynamicBreastCancerEnvironment, ratio: float,
) -> DynamicBreastCancerEnvironment:
    """A copy of the environment whose systemic salvage death multiplier is ``ratio``."""
    config = environment.config
    hazards = dict(config.hazard_multipliers)
    salvage = {key: dict(value) for key, value in hazards["salvage"].items()}
    salvage["systemic"] = {**salvage["systemic"], "death": float(ratio)}
    hazards["salvage"] = salvage
    new_config = replace(config, hazard_multipliers=hazards)
    return DynamicBreastCancerEnvironment(environment.patient, environment.risk_table,
                                          new_config)


def treat_share_by_ratio(map_frame: pd.DataFrame) -> pd.DataFrame:
    """Share of (patient, year) cells where treating pays, per hazard ratio."""
    grouped = map_frame.groupby("hazard_ratio")["advantage"]
    return pd.DataFrame({
        "hazard_ratio": grouped.size().index,
        "cells": grouped.size().to_numpy(),
        "treat_share": grouped.apply(lambda a: float((a > 0).mean())).to_numpy(),
        "mean_advantage": grouped.mean().to_numpy(),
    })


def indifference_ratio_per_patient(map_frame: pd.DataFrame,
                                   year: int | None = None) -> pd.Series:
    """The largest hazard ratio at which treating still pays, per patient.

    With multiplicative benefit and flat cost the advantage falls
    monotonically in the ratio, so the crossing is well defined. Patients for
    whom treating never pays get NaN; patients for whom it always pays get the
    top of the grid. ``year`` restricts to one recurrence year; default pools
    all years and takes each patient's median crossing.
    """
    frame = map_frame if year is None else map_frame[map_frame["recurrence_year"] == year]
    crossings = []
    for (patient_id, recurrence_year), block in frame.groupby(["patient_id", "recurrence_year"]):
        paying = block[block["advantage"] > 0]
        crossings.append({
            "patient_id": patient_id,
            "recurrence_year": recurrence_year,
            "indifference_ratio": float(paying["hazard_ratio"].max()) if len(paying) else math.nan,
        })
    table = pd.DataFrame(crossings)
    return table.groupby("patient_id")["indifference_ratio"].median()


def pick_indifference_ratio(share_table: pd.DataFrame, target: float = 0.5) -> float:
    """The grid ratio whose treat share is closest to ``target``."""
    if share_table.empty:
        raise ValueError("empty share table")
    index = (share_table["treat_share"] - target).abs().idxmin()
    return float(share_table.loc[index, "hazard_ratio"])
