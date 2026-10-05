"""Grading the disease state, as a dial rather than a decision. (v2.4)

Three versions in a row opened a decision point that turned out not to be one.
v2.0 explained why for the salvage channel, with arithmetic rather than
search: every patient's indifference hazard ratio sat between 0.95 and 0.96 -
a spread of **0.01** - because the declared benefit is multiplicative and the
declared cost is fixed, so the baseline hazard very nearly cancels out of the
crossing condition. Where nobody differs, there is nothing to adapt to.

v2.3 then found what the environment is actually missing, and it was not what
we expected. Performance status moves and tracks disease, but recorded care
does not visibly act on it. What recorded care *does* act on is the thing a
progression call measures: treatment changes 2.41x more often after
"progressing" than after "controlled" (v2.2). Our environment carries that as
``recurred``, a **binary** flag - so every recurred patient faces the same
post-recurrence hazard, which is exactly the condition that made v2.0's
spread 0.01.

This module grades it. The post-recurrence hazard is split across the three
levels a progression call actually distinguishes, and **how far apart they
sit is a dial**, not a chosen number - the pattern ``analysis/dynamic/benefit.py``
established in v2.1, applied to state instead of to benefit. At separation 0
the grades coincide and the environment is byte-for-byte the current one; at
separation 1 they span :data:`FULL_SEPARATION`.

Two things make the dial honest:

**Mean-neutrality.** Splitting one hazard into three must not change the
expected hazard, or the grading would smuggle in a prognosis change and every
later result would be measuring that instead. :func:`grade_multipliers`
renormalises so that the weighted mean multiplier is exactly 1 at every
separation - the same discipline v0.5 applied to the response channel.

**Weights from data, not from taste.** The grade distribution is MSK-CHORD's
(v2.2): of 32,977 scored progression calls in the breast cohort, 56.0% read
controlled, 20.1% indeterminate and 23.9% progressing. Those are counts, so
they are reported as counts and not swept.

What is *not* here: any change to the schema, the phases or the action space.
This module exists to answer "would a graded state create an indifference
region?" with arithmetic, before any of that is written. v2.0's lesson was
that a five-second calculation can replace a forty-minute search.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from .config import DynamicConfig
from .environment import DynamicBreastCancerEnvironment

#: The three levels a progression call distinguishes, worst last.
GRADES = ("controlled", "indeterminate", "progressing")

#: Share of scored progression calls at each level, from MSK-CHORD's breast
#: cohort (reports/msk-chord-profile-v2.2/tables/response_switch.csv:
#: 18,481 + 6,620 + 7,876 = 32,977 calls). Counts, so not swept.
GRADE_WEIGHTS: dict[str, float] = {
    "controlled": 18481 / 32977,
    "indeterminate": 6620 / 32977,
    "progressing": 7876 / 32977,
}

#: Relative death hazard at separation 1, before mean-normalisation.
#:
#: DECLARED, not estimated. A five-fold spread between controlled and
#: progressing disease is at the upper end of what the oncology literature
#: reports for progressive versus controlled disease, and it is the *endpoint
#: of a dial*: what this module reports is the separation at which the grades
#: stop agreeing, expressed both as a fraction of this endpoint and as the
#: hazard multipliers at that fraction - the way v2.1 reports its tipping
#: point. Moving the endpoint re-scales the axis and nothing else.
#:
#: MSK-CHORD cannot supply this number: its cohort is conditioned on surviving
#: from diagnosis to sequencing (median 441 days, p75 1,905), so a survival
#: contrast by progression status on it would be badly biased (v2.2).
FULL_SEPARATION: dict[str, float] = {
    "controlled": 0.5,
    "indeterminate": 1.0,
    "progressing": 2.5,
}


def grade_multipliers(
    separation: float,
    weights: Mapping[str, float] = GRADE_WEIGHTS,
    full: Mapping[str, float] = FULL_SEPARATION,
) -> dict[str, float]:
    """Mean-neutral death-hazard multipliers at this separation.

    Geometric interpolation from 1 (``full[grade] ** separation``), then
    divided by the weighted mean so that ``sum(weights[g] * out[g]) == 1``
    exactly. Without that division the dial would raise or lower the average
    post-recurrence hazard as it turned, and the map would be reading a
    prognosis change rather than a grading.

    At ``separation == 0`` every multiplier is exactly 1.0, which is the
    current environment - the identity that makes a null control possible.
    """
    if not 0.0 <= float(separation) <= 1.0:
        raise ValueError("separation must lie in [0, 1]")
    missing = set(GRADES) - set(weights) or set(GRADES) - set(full)
    if missing:
        raise ValueError(f"missing grades: {sorted(missing)}")
    total_weight = sum(float(weights[grade]) for grade in GRADES)
    if abs(total_weight - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to 1, got {total_weight}")

    raw = {grade: float(full[grade]) ** float(separation) for grade in GRADES}
    mean = sum(float(weights[grade]) * raw[grade] for grade in GRADES)
    if mean <= 0:
        raise ValueError("weighted mean multiplier must be positive")
    return {grade: raw[grade] / mean for grade in GRADES}


def with_burden_grade(
    config: DynamicConfig,
    grade: str,
    separation: float,
    weights: Mapping[str, float] = GRADE_WEIGHTS,
    full: Mapping[str, float] = FULL_SEPARATION,
) -> DynamicConfig:
    """A config whose post-recurrence death hazard is this grade's.

    Scales ``hazard_multipliers["death_after_recurrence"]`` rather than adding
    a field, so nothing in the schema, the phases or the action space changes.
    That is deliberate: this is the arithmetic preview, and a preview that
    required an environment rewrite would not be a preview.
    """
    if grade not in GRADES:
        raise ValueError(f"unknown grade {grade!r}; expected one of {GRADES}")
    multipliers = grade_multipliers(separation, weights, full)
    hazards = dict(config.hazard_multipliers)
    hazards["death_after_recurrence"] = (
        float(hazards["death_after_recurrence"]) * multipliers[grade])
    return replace(config, hazard_multipliers=hazards)


def graded_environment(
    environment: DynamicBreastCancerEnvironment,
    grade: str,
    separation: float,
) -> DynamicBreastCancerEnvironment:
    """A copy of ``environment`` carrying this grade's post-recurrence hazard."""
    return DynamicBreastCancerEnvironment(
        environment.patient, environment.risk_table,
        with_burden_grade(environment.config, grade, separation))


def declared_grades(
    config: DynamicConfig, separation: float,
) -> list[dict[str, float | str]]:
    """What the dial declares at this separation, as committable rows."""
    multipliers = grade_multipliers(separation)
    base = float(config.hazard_multipliers["death_after_recurrence"])
    return [{
        "separation": float(separation),
        "grade": grade,
        "weight": float(GRADE_WEIGHTS[grade]),
        "multiplier": multipliers[grade],
        "death_after_recurrence": base * multipliers[grade],
    } for grade in GRADES]
