"""Putting stakes into the adaptive channel, as a dial. (v2.5)

v2.4 ended four versions of work on the salvage decision with a number rather
than a hunch: grading the disease state produced disagreement between grades
but the information was worth **0.00070** against an action-value noise floor
of **0.02** - a ratio of 1/28. The channel's whole stakes are 0.0757 and the
grade-conditional part of that is 0.9%. The conclusion was that the binding
constraint is **the channel's stakes, not the state's resolution**.

That conclusion has an obvious next question and a non-obvious one.

The obvious one is *magnitude*: how much bigger would the channel have to be?
The non-obvious one is *structure*: our salvage channel offers exactly two
options, ``none`` and ``systemic``. Real care after progression is not
"treat or not" - 66.8% of progressing calls are followed by a **new systemic
treatment** (v2.2), which is a choice among lines, not a yes/no. A choice among
lines can be state-dependent in a way a yes/no cannot: a stronger, costlier
line pays only above a baseline-hazard threshold, so *which* line wins depends
on how sick the patient is, while *whether* to treat is a single crossing.

So this module sweeps both at once. ``scale`` turns the channel's stakes up and
down continuously; ``LINES`` versus ``BINARY_LINES`` switches the structure.
Comparing the two arms **at matched stakes** separates "the channel is too
small" from "the channel is the wrong shape", and those two findings imply
different fixes.

Construction, and why it is honest:

**Scale 0 is the null control, exactly.** At ``scale == 0`` every line has
hazard multiplier 1.0 and cost 0.0, so every option is identical to declining.
Stakes, state-dependence and information value must all be exactly zero there,
and a pre-registered check says so.

**Benefit and cost scale together.** Hazards interpolate geometrically
(``HR ** scale``) and costs linearly, which keeps the benefit-to-cost ratio
roughly fixed as the dial turns - so the sweep changes *how much is at stake*
without quietly changing *what kind of bargain* the channel offers.

**The ladder is declared, not estimated.** Nothing in METABRIC, GENIE BPC or
MSK-CHORD can tell us what a second-line regimen is worth in our utility units;
MSK-CHORD's left truncation alone rules out a survival contrast by line (v2.2).
So the endpoint is written down, the sweep is reported as a fraction of it, and
:func:`current_config_scale` says where today's config sits on the dial.
"""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Mapping, Sequence

from .config import DynamicConfig
from .environment import DynamicBreastCancerEnvironment

#: The graded ladder: declining, then three lines of increasing strength and
#: cost. Ordered weakest-first so ``LINES[0]`` is always the decline option.
LINES = ("none", "mild", "systemic", "intensive")

#: Today's structure, kept as the comparison arm. Two options is what v1.8
#: opened and what v2.0 and v2.4 found could not carry an adaptive decision.
BINARY_LINES = ("none", "systemic")

#: The ladder at ``scale == 1``. **Declared, not estimated.**
#:
#: Each rung buys more hazard reduction for more burden and more toxicity, and
#: the increments are chosen so that no rung dominates another on both axes -
#: otherwise the "choice among lines" would collapse back to a yes/no on the
#: single best rung and the structural comparison would be vacuous.
FULL_LADDER: dict[str, dict[str, float]] = {
    "none": {"death": 1.0, "recurrence": 1.0, "burden": 0.0, "toxicity": 0.0},
    "mild": {"death": 0.92, "recurrence": 1.0, "burden": 0.05, "toxicity": 0.08},
    "systemic": {"death": 0.70, "recurrence": 1.0, "burden": 0.20, "toxicity": 0.35},
    "intensive": {"death": 0.50, "recurrence": 1.0, "burden": 0.40, "toxicity": 0.55},
}


def scaled_line(full: Mapping[str, float], scale: float,
                cost_multiple: float = 1.0) -> dict[str, float]:
    """One rung at this stakes scale.

    Hazards geometrically (``HR ** scale``), costs linearly. At ``scale == 0``
    this returns a rung indistinguishable from declining, which is what makes
    the null control exact rather than approximate.

    ``cost_multiple`` is a **second, post-hoc dial** (v2.5). The ladder as first
    declared turned out to have a dominant rung - ``intensive`` beat every other
    option for every patient, year and grade, so four options behaved as one and
    the structural comparison had nothing to compare. Steepening the price of
    strength is what makes the rungs trade off, and sweeping it says how steep
    they would have to be. Toxicity is a probability, so it is capped at 1.
    """
    if not 0.0 <= float(scale) <= 1.0:
        raise ValueError("scale must lie in [0, 1]")
    if float(cost_multiple) < 0.0:
        raise ValueError("cost_multiple must not be negative")
    return {
        "death": float(full["death"]) ** float(scale),
        "recurrence": float(full["recurrence"]) ** float(scale),
        "burden": float(full["burden"]) * float(scale) * float(cost_multiple),
        "toxicity": min(
            float(full["toxicity"]) * float(scale) * float(cost_multiple), 1.0),
    }


def with_salvage_stakes(
    config: DynamicConfig,
    scale: float,
    lines: Sequence[str] = LINES,
    ladder: Mapping[str, Mapping[str, float]] = FULL_LADDER,
    cost_multiple: float = 1.0,
) -> DynamicConfig:
    """A config whose salvage channel is ``lines`` at this stakes scale.

    Replaces the salvage block wholesale rather than editing it, so the arm is
    defined by what it offers and not by what happened to be there before.
    ``none`` is always included: a channel with no decline option is not a
    decision.
    """
    chosen = tuple(dict.fromkeys(("none", *lines)))
    missing = [line for line in chosen if line not in ladder]
    if missing:
        raise ValueError(f"lines not in the declared ladder: {missing}")

    hazards = dict(config.hazard_multipliers)
    burdens = {key: dict(value) if isinstance(value, dict) else value
               for key, value in config.treatment_burden.items()}
    toxicities = {key: dict(value) if isinstance(value, dict) else value
                  for key, value in config.acute_toxicity_probabilities.items()}

    salvage_hazards: dict[str, dict[str, float]] = {}
    salvage_burden: dict[str, float] = {}
    salvage_toxicity: dict[str, float] = {}
    for line in chosen:
        rung = scaled_line(ladder[line], scale, cost_multiple)
        salvage_hazards[line] = {"death": rung["death"],
                                 "recurrence": rung["recurrence"]}
        salvage_burden[line] = rung["burden"]
        salvage_toxicity[line] = rung["toxicity"]

    hazards["salvage"] = salvage_hazards
    burdens["salvage"] = salvage_burden
    toxicities["salvage"] = salvage_toxicity
    return replace(config, hazard_multipliers=hazards,
                   treatment_burden=burdens,
                   acute_toxicity_probabilities=toxicities)


def staked_environment(
    environment: DynamicBreastCancerEnvironment,
    scale: float,
    lines: Sequence[str] = LINES,
    cost_multiple: float = 1.0,
) -> DynamicBreastCancerEnvironment:
    """A copy of ``environment`` whose salvage channel carries these stakes."""
    return DynamicBreastCancerEnvironment(
        environment.patient, environment.risk_table,
        with_salvage_stakes(environment.config, scale, lines,
                            cost_multiple=cost_multiple))


def current_config_scale(
    config: DynamicConfig,
    line: str = "systemic",
    ladder: Mapping[str, Mapping[str, float]] = FULL_LADDER,
) -> float:
    """Where today's declared salvage rung sits on this dial.

    Solved from the death hazard, because that is the parameter the channel's
    value actually turns on: ``ladder[line]["death"] ** scale == today's``.
    Reported so the sweep can be read as "today is here, and the answer is
    there" rather than as an abstract fraction.
    """
    today = float(config.hazard_multipliers["salvage"][line]["death"])
    full = float(ladder[line]["death"])
    if not 0.0 < full < 1.0:
        raise ValueError("the declared endpoint must cut the hazard")
    if not 0.0 < today <= 1.0:
        raise ValueError("today's hazard multiplier must lie in (0, 1]")
    return math.log(today) / math.log(full)


def declared_ladder(config: DynamicConfig, scale: float,
                    lines: Sequence[str] = LINES,
                    cost_multiple: float = 1.0) -> list[dict[str, float | str]]:
    """What the dial declares at this scale, as committable rows."""
    staked = with_salvage_stakes(config, scale, lines, cost_multiple=cost_multiple)
    penalty = float(config.reward["acute_toxicity_penalty"])
    rows = []
    for line in staked.hazard_multipliers["salvage"]:
        burden = float(staked.treatment_burden["salvage"][line])
        toxicity = float(staked.acute_toxicity_probabilities["salvage"][line])
        rows.append({
            "scale": float(scale),
            "cost_multiple": float(cost_multiple),
            "line": line,
            "death_hazard": float(staked.hazard_multipliers["salvage"][line]["death"]),
            "burden": burden,
            "toxicity_probability": toxicity,
            "raw_cost": burden + toxicity * penalty,
        })
    return rows
