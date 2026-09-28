"""Turning the standard-treatment benefit into a dial (v2.1).

v1.5 read the config and found that standard chemotherapy, standard endocrine
therapy and local radiotherapy are declared to have a survival benefit of
**exactly zero** while charging toxicity and burden - and that those three are
what the guideline prescribes. 41% of the original utility gap was MCTS
declining to pay for them. Every version since has repeated that the fix is the
highest-priority open item, and every version has left it open because picking
the numbers is a clinical judgement.

This module does not pick them. It makes the benefit a **dial** so the question
changes from "what is the number?" to "how much does the answer depend on it?"

``strength`` (lambda) runs from 0 to 1:

* 0 reproduces the current config exactly - standard treatment is pure cost.
* 1 gives standard treatment the hazard ratios in :data:`FULL_BENEFIT`.
* in between, hazard ratios interpolate geometrically -
  ``HR(lambda) = HR_full ** lambda`` - which is linear in log-hazard, the scale
  on which these effects are additive.

**The whole ladder moves together.** If only the standard rung were raised,
intensified chemotherapy (currently HR 0.95/0.90) would end up *worse* than
standard, and the sweep would silently invert the ladder and measure that
instead. So each higher rung keeps the increment it declares today:
``HR_intensified(lambda) = HR_standard(lambda) * HR_intensified_now``.
"""

from __future__ import annotations

import copy
from dataclasses import replace

from .environment import DynamicBreastCancerEnvironment
from .indifference import expected_remaining_utility
from .schema import DynamicState

#: Hazard ratios standard treatment is given at ``strength = 1``.
#:
#: These are order-of-magnitude figures for the randomised evidence on adjuvant
#: therapy (EBCTCG meta-analyses of polychemotherapy, five years of tamoxifen,
#: and radiotherapy after breast-conserving surgery), recorded here **from
#: memory and flagged for verification against the primary papers**. They are
#: deliberately not the point: the sweep reports where the conclusion turns as a
#: fraction of this endpoint *and* as the implied hazard ratio, so correcting
#: the endpoint rescales the axis without overturning the shape.
FULL_BENEFIT: dict[str, dict[str, float]] = {
    "chemo": {"death": 0.80, "recurrence": 0.75},
    "endocrine": {"death": 0.75, "recurrence": 0.60},
    "radiation": {"death": 0.85, "recurrence": 0.50},
}

#: Which rung of each ladder is "standard treatment" - the one the guideline
#: prescribes and the one currently declared to be worth nothing.
STANDARD_LEVEL = {"chemo": "standard", "endocrine": "standard", "radiation": "local"}

#: The rung above it, whose declared increment is preserved.
HIGHER_LEVEL = {"chemo": "intensified", "endocrine": "extended", "radiation": "regional"}


def hazard_at_strength(full: float, strength: float) -> float:
    """``full ** strength`` - geometric interpolation from 1.0 to ``full``."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if full <= 0.0:
        raise ValueError("a hazard ratio must be positive")
    return float(full) ** float(strength)


def with_standard_benefit(config: dict, strength: float) -> dict:
    """A raw config whose standard treatment carries ``strength`` of the benefit.

    ``strength = 0`` returns an exact copy of the input, so the sweep's first
    arm reproduces the committed environment rather than approximating it.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    data = copy.deepcopy(config)
    hazards = data["hazard_multipliers"]
    for channel, full in FULL_BENEFIT.items():
        standard = STANDARD_LEVEL[channel]
        higher = HIGHER_LEVEL[channel]
        current_increment = {
            outcome: float(hazards[channel][higher][outcome])
            / float(hazards[channel][standard][outcome])
            for outcome in ("death", "recurrence")
        }
        for outcome in ("death", "recurrence"):
            scaled = hazard_at_strength(full[outcome], strength)
            hazards[channel][standard][outcome] = scaled
            hazards[channel][higher][outcome] = scaled * current_increment[outcome]
    return data


def declared_ladder(config: dict) -> list[dict]:
    """The hazard ratios a config declares for the three treatment ladders."""
    rows = []
    for channel in FULL_BENEFIT:
        for level, hazards in config["hazard_multipliers"][channel].items():
            rows.append({
                "channel": channel,
                "level": level,
                "death_hazard": float(hazards["death"]),
                "recurrence_hazard": float(hazards["recurrence"]),
                "acute_toxicity_probability": float(
                    config["acute_toxicity_probabilities"][channel][level]),
                "treatment_burden": float(config["treatment_burden"][channel][level]),
            })
    return rows


#: The action fields a static plan has to fill, in the order they are played
#: when surgery comes first.
PLAN_FIELDS = ("surgery", "chemo", "endocrine", "radiation")


def plan_expected_utility(
    environment: DynamicBreastCancerEnvironment,
    actions: dict[str, str],
) -> float:
    """Expected utility of playing one fixed surgery-first plan, end to end.

    Deterministic: treatment burden is charged as declared, acute toxicity as
    its expectation, and follow-up is the environment's own annual event model
    solved forward rather than sampled. Used as a cheap preview of what a
    benefit setting does before spending search time on it.

    **Two simplifications, both stated rather than hidden.** The toxicity
    guardrail (which withdraws the higher rung once a toxicity has occurred) is
    ignored, because a plan is fixed here rather than chosen; and neoadjuvant
    timing is out of scope, so the response channel never fires. Both keep this
    a preview, not a replacement for the run.
    """
    missing = [field for field in PLAN_FIELDS if field not in actions]
    if missing:
        raise ValueError(f"plan is missing {missing}")
    config = environment.config
    total = config.normalized(-float(config.treatment_burden["timing"]["surgery_first"]))
    for field in PLAN_FIELDS:
        action = actions[field]
        burden = float(config.treatment_burden[field][action])
        toxicity = float(config.acute_toxicity_probabilities[field][action])
        penalty = float(config.reward["acute_toxicity_penalty"])
        total += config.normalized(-(burden + toxicity * penalty))
    state = replace(
        environment.initial_state(), phase="followup", timing="surgery_first",
        year=0, **{field: actions[field] for field in PLAN_FIELDS})
    return total + expected_remaining_utility(environment, state)


def refusal_plan(environment: DynamicBreastCancerEnvironment,
                 guideline: dict[str, str]) -> dict[str, str]:
    """The guideline plan with every declinable standard treatment refused.

    Surgery is kept - it is not a level any policy declines in this environment -
    and every other channel drops to ``none``. This is the plan a policy that
    treats standard therapy as pure cost would play, which is what MCTS does at
    ``strength = 0``.
    """
    plan = {field: "none" for field in PLAN_FIELDS}
    plan["surgery"] = guideline["surgery"]
    return plan


def guideline_plan(environment: DynamicBreastCancerEnvironment) -> dict[str, str]:
    """The simplified NCCN plan as a static surgery-first action dict."""
    from .policies import DynamicNccnPolicy

    plan = DynamicNccnPolicy(environment).plan
    surgery = str(plan[0])
    return {
        "surgery": surgery,
        "chemo": "standard" if int(plan[1]) else "none",
        "endocrine": "standard" if int(plan[2]) else "none",
        "radiation": ("local" if surgery == "BCS" else "regional") if int(plan[3]) else "none",
    }
