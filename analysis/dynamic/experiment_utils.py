"""Small, testable helpers shared by the robustness and sensitivity studies.

Keeping these here (rather than inside the numbered ``analysis/1x_*.py`` scripts,
whose module names start with a digit and cannot be imported) lets the unit
tests exercise them directly.
"""

from __future__ import annotations

from collections import Counter
import math
import statistics
from typing import Mapping, Sequence


def confidence_interval(values: Sequence[float], z: float = 1.96) -> dict[str, float]:
    """Mean and a normal-approximation confidence interval of the sample mean.

    Uses the standard error of the mean (sd / sqrt(n)); with the default
    ``z = 1.96`` this is a 95% interval. A single value yields a zero-width
    interval rather than an error.
    """
    data = [float(v) for v in values]
    n = len(data)
    if n == 0:
        raise ValueError("confidence_interval requires at least one value")
    mean = sum(data) / n
    if n > 1:
        variance = sum((v - mean) ** 2 for v in data) / (n - 1)
        sd = math.sqrt(variance)
    else:
        sd = 0.0
    stderr = sd / math.sqrt(n)
    half = z * stderr
    return {
        "mean": mean,
        "sd": sd,
        "stderr": stderr,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "n_seeds": n,
    }


def rescale_response_major(block: Mapping[str, float], major: float) -> dict[str, float]:
    """Return a response-probability block with ``major`` set to the given value.

    ``partial`` and ``none`` are rescaled proportionally so the three
    probabilities still sum to 1. Raises if ``major`` is outside [0, 1] or the
    remaining mass is degenerate.
    """
    if not 0.0 <= major <= 1.0:
        raise ValueError(f"major must be in [0, 1], got {major}")
    old_rest = float(block["partial"]) + float(block["none"])
    if old_rest <= 0.0:
        raise ValueError("cannot rescale when partial + none is zero")
    remainder = 1.0 - major
    return {
        "major": major,
        "partial": remainder * (float(block["partial"]) / old_rest),
        "none": remainder * (float(block["none"]) / old_rest),
    }


def summarize_decision_replicates(
    records: Sequence[tuple[str, Mapping[str, float]]],
) -> dict[str, object]:
    """Agreement, estimator noise and best-vs-runner-up gap over repeat searches.

    ``records`` is one ``(chosen_action, action_values)`` pair per independent
    replicate of the same search at the same state and budget. The three numbers
    that matter are:

    * ``agreement_pct`` - how often the replicates picked the same action;
    * ``value_noise_sd`` - the replicate-to-replicate spread of the top two
      actions' estimated values, i.e. pure Monte-Carlo noise, which must shrink
      as the budget grows;
    * ``value_gap`` - the distance between those two actions' mean estimates,
      a property of the environment that no amount of budget changes.

    ``separation`` divides the gap by the noise: below about 1 the ordering
    cannot be resolved from the replicate spread, which is the signature of
    genuine equipoise rather than an under-powered search.
    """
    if not records:
        raise ValueError("summarize_decision_replicates requires at least one replicate")

    counts = Counter(action for action, _ in records)
    modal, modal_count = min(
        counts.items(), key=lambda item: (-item[1], item[0])
    )

    actions = sorted({action for _, values in records for action in values})
    means: dict[str, float] = {}
    sds: dict[str, float] = {}
    for action in actions:
        series = [
            float(values[action]) for _, values in records if action in values
        ]
        means[action] = sum(series) / len(series)
        sds[action] = statistics.stdev(series) if len(series) > 1 else 0.0

    ranked = sorted(actions, key=lambda action: (-means[action], action))
    if len(ranked) > 1:
        gap = means[ranked[0]] - means[ranked[1]]
        noise = (sds[ranked[0]] + sds[ranked[1]]) / 2
    else:
        gap, noise = float("nan"), 0.0
    separation = gap / noise if noise > 0 else float("nan")

    return {
        "modal_action": modal,
        "agreement_pct": modal_count / len(records) * 100,
        "distinct_actions_chosen": len(counts),
        "n_legal_actions": len(actions),
        "best_action": ranked[0],
        "value_gap": gap,
        "value_noise_sd": noise,
        "separation": separation,
    }


def rank_parameters_by_influence(
    influence: Mapping[str, float],
) -> list[str]:
    """Parameter names ordered by ``|influence|``, largest first.

    Ties break alphabetically so a bootstrap replicate that lands on two equal
    magnitudes does not vote for an arbitrary order; without that, resampling
    noise would show up as rank churn that has nothing to do with the data.
    """
    if not influence:
        raise ValueError("rank_parameters_by_influence requires at least one parameter")
    return sorted(influence, key=lambda name: (-abs(float(influence[name])), name))


def patients_for_standard_error(
    per_patient_sd: float, target_standard_error: float
) -> int:
    """Patients needed for the between-patient SE of the mean to reach a target.

    Inverts ``se = sd / sqrt(n)``. This is the question a small simulated cohort
    has to answer honestly: an eight-patient sample does not become adequate
    because its point estimate looks stable, only because its spread is small
    enough for the differences being ranked.
    """
    if per_patient_sd < 0:
        raise ValueError("per_patient_sd must be non-negative")
    if target_standard_error <= 0:
        raise ValueError("target_standard_error must be positive")
    if per_patient_sd == 0:
        return 1
    return math.ceil((per_patient_sd / target_standard_error) ** 2)


#: Two-sided 5% critical value and the 80%-power quantile. Named rather than
#: inlined so a reader can see which convention a "minimum detectable" number
#: was computed under.
Z_ALPHA_TWO_SIDED_05 = 1.959963984540054
Z_POWER_80 = 0.8416212335729143


def minimum_detectable_difference(
    standard_error: float,
    z_alpha: float = Z_ALPHA_TWO_SIDED_05,
    z_power: float = Z_POWER_80,
) -> float:
    """Smallest true difference this design would catch 80% of the time.

    "No difference" and "could not detect one" are different claims, and a null
    result is only readable next to the size it could have found. Reported with
    every negative finding since ``reports/environment-fix-v0.5``; v1.7 needs it
    because its headline *is* a null.
    """
    if standard_error <= 0:
        raise ValueError("standard_error must be positive")
    return (float(z_alpha) + float(z_power)) * float(standard_error)
