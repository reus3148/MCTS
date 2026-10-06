"""Is the adaptive channel too small, or the wrong shape? (v2.5, arithmetic only)

This is the last experiment. v2.4 closed four versions of work on the salvage
decision with a number: grading the disease state made the grades disagree, but
knowing the grade was worth **0.00070** against an action-value noise floor of
**0.02** - a ratio of 1/28. Channel stakes 0.0757; grade-conditional share 0.9%.
The binding constraint was **the channel's stakes, not the state's resolution**.

That leaves two different fixes, and they are not the same fix.

**Magnitude.** The channel is simply too small, and anything that makes the
decision matter more - a longer horizon, a bigger declared effect - would
rescue it proportionally.

**Structure.** Our salvage channel offers exactly two options, ``none`` and
``systemic``. Real care after progression is not treat-or-not: 66.8% of
progressing calls are followed by a **new systemic treatment** (v2.2), which is
a choice among *lines*. That difference is not cosmetic. Whether to treat is a
single crossing in baseline hazard, so a sicker patient and a healthier one
differ only in how far they sit from one line. *Which* line to use has a
crossing of its own - a stronger, costlier line pays only above a hazard
threshold - so the **ranking itself** depends on state. A yes/no cannot be
state-dependent in that way; a ladder can.

So both are swept at once, and compared **at matched channel stakes**. That
comparison is the whole point: if the ladder carries more usable information
than the binary at the same stakes, the fix is shape and the environment needs
rewriting; if it does not, the fix is size and we can say how much.

Deterministic throughout, like v2.0 and v2.4 - expectations over the
environment's own annual event model, no search, no random draws, seconds
rather than hours. The disease state is graded at full separation
(``analysis/dynamic/burden.py``), because a state worth observing is the
precondition for asking what observing it is worth.

PRE-SPECIFIED PREDICTIONS, recorded before the run
--------------------------------------------------
v2.4's pre-registration covered stakes and state-dependence and **left out
resolution**, which is the leg that ended up carrying the conclusion. All three
are registered here, plus the null control.

1. **Null control.** At scale 0 every line is identical to declining, so
   channel stakes, state-dependence and information value are **exactly zero**
   (< 1e-12) in both arms. If not, the dial is not neutral and nothing else
   here can be read.
2. **Stakes.** At scale 1 the ladder arm's channel stakes exceed v2.4's
   **0.0757** by at least **2x**.
3. **State-dependence.** At scale 1 the ladder arm's best line differs across
   the three grades for at least **30%** of (patient, recurrence-year)
   decisions, while the binary arm's does so for under **10%**.
4. **Primary - resolution, structure versus magnitude.** **At matched channel
   stakes**, the ladder arm's expected value of knowing the grade is at least
   **3x** the binary arm's. If this fails, shape does not help and only size
   does.
5. **Crossing.** Some stakes scale <= 1 brings the ladder arm's per-decision
   information value up to the **0.02** noise floor.

Prediction 4 is the one that decides what the write-up says, and prediction 5
is the one most likely to fail while 4 passes: the ladder can be structurally
better and still not clear an absolute bar inside the declared range.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.dynamic.burden import GRADE_WEIGHTS, GRADES, graded_environment  # noqa: E402
from analysis.dynamic.cohort import (  # noqa: E402
    balanced_subtype_sample,
    build_reward_models,
    git_commit,
    input_manifest,
    make_risk_table,
)
from analysis.dynamic.config import load_dynamic_config  # noqa: E402
from analysis.dynamic.environment import DynamicBreastCancerEnvironment  # noqa: E402
from analysis.dynamic.indifference import (  # noqa: E402
    expected_remaining_utility,
    salvage_cost,
)
from analysis.dynamic.policies import DynamicNccnPolicy  # noqa: E402
from analysis.dynamic.schema import patient_from_row  # noqa: E402
from analysis.dynamic.stakes import (  # noqa: E402
    BINARY_LINES,
    FULL_LADDER,
    LINES,
    current_config_scale,
    declared_ladder,
    staked_environment,
)

REPORT_DIR = ROOT / "reports" / "stakes-map-v2.5"
TABLE_DIR = REPORT_DIR / "tables"
CONFIG_V07 = ROOT / "configs" / "dynamic_v0_7.json"
INPUT_CSV = ROOT / "data" / "processed" / "patients_with_nccn.csv"

RUN_DATE = "2026-10-06"
PER_SUBTYPE = 5                       # x4 subtypes x2 offsets = 40 patients
YEARS = (1, 2, 3, 4)
SCALES = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
SEPARATION = 1.0                      # disease grading at full separation (v2.4)

ARMS = {"binary": BINARY_LINES, "ladder": LINES}

#: POST-HOC second dial: how much more the strong lines would have to cost.
#: The pre-registered ladder has a dominant rung (see ``cost_sweep``).
COST_MULTIPLES = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 12.0)

#: v2.4's figures, the numbers this run is measured against.
V24_CHANNEL_STAKES = 0.0757
V24_INFORMATION_VALUE = 0.00070
ACTION_VALUE_NOISE = 0.02

NULL_TOLERANCE = 1e-12
STAKES_MULTIPLE_FLOOR = 2.0
LADDER_SWITCH_FLOOR = 0.30
BINARY_SWITCH_CEILING = 0.10
STRUCTURE_ADVANTAGE_FLOOR = 3.0

PRESPECIFIED_PREDICTION = {
    "null_control": (
        f"At scale 0, channel stakes, state-dependence and information value "
        f"are all < {NULL_TOLERANCE} in both arms."
    ),
    "stakes": (
        f"At scale 1 the ladder arm's channel stakes exceed v2.4's "
        f"{V24_CHANNEL_STAKES} by >= {STAKES_MULTIPLE_FLOOR}x."
    ),
    "state_dependence": (
        f"At scale 1 the ladder arm's best line differs across grades for >= "
        f"{LADDER_SWITCH_FLOOR:.0%} of decisions; the binary arm's for < "
        f"{BINARY_SWITCH_CEILING:.0%}."
    ),
    "primary_resolution_structure_vs_magnitude": (
        f"At matched channel stakes, the ladder arm's expected value of "
        f"knowing the grade is >= {STRUCTURE_ADVANTAGE_FLOOR}x the binary arm's."
    ),
    "crossing": (
        f"Some scale <= 1 brings the ladder arm's per-decision information "
        f"value to the {ACTION_VALUE_NOISE} noise floor."
    ),
    "why": (
        "v2.4 found the salvage channel cannot carry an adaptive decision and "
        "that stakes, not state resolution, were binding. Two different fixes "
        "follow - make the channel bigger, or make it a choice among lines "
        "rather than a yes/no - and they imply different rewrites. v2.4's "
        "pre-registration omitted resolution, the leg that carried its "
        "conclusion; all three legs are registered here."
    ),
}


def nccn_followup_state(environment: DynamicBreastCancerEnvironment):
    """A follow-up state carrying the simplified NCCN plan for this patient."""
    plan = DynamicNccnPolicy(environment).plan
    surgery = str(plan[0])
    return replace(
        environment.initial_state(), phase="followup", timing="surgery_first",
        surgery=surgery,
        chemo="standard" if int(plan[1]) else "none",
        endocrine="standard" if int(plan[2]) else "none",
        radiation=("local" if surgery == "BCS" else "regional") if int(plan[3]) else "none",
    )


def line_advantages(environment, state, lines) -> dict[str, float]:
    """Expected utility of each salvage line minus that of declining.

    ``none`` is 0 by construction, so a positive value means that line beats
    declining for this patient, year and grade under this environment.
    """
    after = replace(state, phase="followup")
    baseline = (expected_remaining_utility(environment, replace(after, salvage="none"))
                - salvage_cost(environment, "none"))
    out = {}
    for line in lines:
        value = (expected_remaining_utility(environment, replace(after, salvage=line))
                 - salvage_cost(environment, line))
        out[line] = value - baseline
    return out


def sweep(environments, plans) -> pd.DataFrame:
    """Advantage of every line, for every arm x scale x grade x patient x year."""
    rows = []
    for arm, lines in ARMS.items():
        for scale in SCALES:
            for patient_id, environment in environments.items():
                base = plans[patient_id]
                for grade in GRADES:
                    graded = graded_environment(environment, grade, SEPARATION)
                    staked = staked_environment(graded, scale, lines)
                    for year in YEARS:
                        state = replace(base, phase="salvage", year=int(year),
                                        recurred=True, recurrence_year=int(year))
                        advantages = line_advantages(staked, state, lines)
                        best = max(advantages, key=lambda line: advantages[line])
                        rows.append({
                            "arm": arm,
                            "scale": float(scale),
                            "patient_id": patient_id,
                            "grade": grade,
                            "recurrence_year": int(year),
                            "best_line": best,
                            "best_advantage": float(max(advantages[best], 0.0)),
                            **{f"adv_{line}": float(value)
                               for line, value in advantages.items()},
                        })
    return pd.DataFrame(rows)


def information_value(block: pd.DataFrame, lines) -> tuple[float, float]:
    """Expected value of knowing the grade, per decision and pooled.

    ``per_decision`` lets the planner know the patient and recurrence year and
    asks what the *grade* adds on top - the quantity a policy at this decision
    point would actually gain. ``pooled`` averages patients away first, which
    is the form v2.4 reported, kept so the two versions are comparable.

    Both take ``max(..., 0)`` nowhere: declining is already one of the options
    and carries advantage 0, so the maximum is never negative.
    """
    columns = [f"adv_{line}" for line in lines]
    weights = np.array([GRADE_WEIGHTS[grade] for grade in GRADES])

    per_decision = []
    for _, decision in block.groupby(["patient_id", "recurrence_year"]):
        indexed = decision.set_index("grade")
        if not set(GRADES) <= set(indexed.index):
            continue
        matrix = indexed.loc[list(GRADES), columns].to_numpy(dtype=float)
        informed = float(weights @ matrix.max(axis=1))
        blind = float((weights @ matrix).max())
        per_decision.append(informed - blind)

    by_grade = block.groupby("grade")[columns].mean()
    matrix = by_grade.loc[list(GRADES)].to_numpy(dtype=float)
    pooled = float(weights @ matrix.max(axis=1)) - float((weights @ matrix).max())
    return (float(np.mean(per_decision)) if per_decision else float("nan"), pooled)


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    """Stakes, state-dependence and information value for every arm x scale."""
    rows = []
    for (arm, scale), block in table.groupby(["arm", "scale"]):
        lines = ARMS[arm]
        switch = (block.groupby(["patient_id", "recurrence_year"])["best_line"]
                  .nunique().gt(1).mean())
        per_decision, pooled = information_value(block, lines)
        rows.append({
            "arm": arm,
            "scale": float(scale),
            "channel_stakes": float(block["best_advantage"].mean()),
            "max_advantage": float(block["best_advantage"].max()),
            "grade_switch_share": float(switch),
            "information_per_decision": per_decision,
            "information_pooled": pooled,
            "treat_share": float((block["best_line"] != "none").mean()),
            "lines_used": int(block["best_line"].nunique()),
        })
    return pd.DataFrame(rows).sort_values(["arm", "scale"]).reset_index(drop=True)


def at_matched_stakes(summary: pd.DataFrame, target: float) -> dict:
    """Each arm's information value at a common channel-stakes level.

    The arms reach the same stakes at different scales, so comparing them at a
    shared scale would compare different-sized channels and answer nothing.
    Both curves are interpolated onto ``target`` stakes instead; an arm that
    never reaches it is reported as not reaching it rather than extrapolated.
    """
    out: dict[str, float | bool] = {"target_stakes": float(target)}
    for arm in ARMS:
        block = summary[summary["arm"] == arm].sort_values("channel_stakes")
        stakes = block["channel_stakes"].to_numpy(dtype=float)
        if target > stakes.max() + 1e-12:
            out[f"{arm}_reaches_target"] = False
            out[f"{arm}_information"] = float("nan")
            continue
        out[f"{arm}_reaches_target"] = True
        out[f"{arm}_information"] = float(np.interp(
            target, stakes, block["information_per_decision"].to_numpy(dtype=float)))
        out[f"{arm}_scale"] = float(np.interp(
            target, stakes, block["scale"].to_numpy(dtype=float)))
    binary = out.get("binary_information")
    ladder = out.get("ladder_information")
    if binary and ladder and binary > 0:
        out["structure_multiple"] = float(ladder / binary)
    return out


def cost_sweep(environments, plans) -> pd.DataFrame:
    """POST-HOC: how steep must the price of strength be before rungs trade off?

    The pre-registered ladder has a **dominant rung**. At every stakes scale,
    every patient, every recurrence year and every grade picks ``intensive``:
    four options behave as one, so predictions 3 and 4 were testing a ladder
    that was a yes/no in disguise. The module docstring said the increments
    were chosen so no rung dominates; they were not, and that is a design
    error rather than a finding about medicine.

    What *is* a finding is how far off the declaration was. Multiplying every
    line's burden and toxicity by a factor and sweeping it says how much more
    the strong lines would have to cost before choosing between them became a
    real choice - and whether state-dependence appears at all once it does.

    Held at full stakes throughout, so the sweep isolates the price of
    strength from the size of the channel.
    """
    rows = []
    for multiple in COST_MULTIPLES:
        for patient_id, environment in environments.items():
            base = plans[patient_id]
            for grade in GRADES:
                graded = graded_environment(environment, grade, SEPARATION)
                staked = staked_environment(graded, 1.0, LINES,
                                            cost_multiple=multiple)
                for year in YEARS:
                    state = replace(base, phase="salvage", year=int(year),
                                    recurred=True, recurrence_year=int(year))
                    advantages = line_advantages(staked, state, LINES)
                    best = max(advantages, key=lambda line: advantages[line])
                    rows.append({
                        "arm": "ladder",
                        "cost_multiple": float(multiple),
                        "patient_id": patient_id,
                        "grade": grade,
                        "recurrence_year": int(year),
                        "best_line": best,
                        "best_advantage": float(max(advantages[best], 0.0)),
                        **{f"adv_{line}": float(value)
                           for line, value in advantages.items()},
                    })
    return pd.DataFrame(rows)


def summarise_costs(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for multiple, block in table.groupby("cost_multiple"):
        switch = (block.groupby(["patient_id", "recurrence_year"])["best_line"]
                  .nunique().gt(1).mean())
        per_decision, pooled = information_value(block, LINES)
        rows.append({
            "cost_multiple": float(multiple),
            "channel_stakes": float(block["best_advantage"].mean()),
            "grade_switch_share": float(switch),
            "information_per_decision": per_decision,
            "information_pooled": pooled,
            "treat_share": float((block["best_line"] != "none").mean()),
            "lines_used": int(block["best_line"].nunique()),
            "top_line_share": float(
                block["best_line"].value_counts(normalize=True).iloc[0]),
        })
    return pd.DataFrame(rows).sort_values("cost_multiple").reset_index(drop=True)


def crossing_scale(summary: pd.DataFrame, arm: str, floor: float) -> float:
    """Stakes scale at which this arm's per-decision information reaches ``floor``."""
    block = summary[summary["arm"] == arm].sort_values("scale")
    values = block["information_per_decision"].to_numpy(dtype=float)
    scales = block["scale"].to_numpy(dtype=float)
    if values.max() < floor:
        return float("nan")
    return float(np.interp(floor, values, scales))


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    base_config = load_dynamic_config(CONFIG_V07)
    today_scale = current_config_scale(base_config)

    raw = pd.read_csv(INPUT_CSV)
    os_model, rfs_model, os_test = build_reward_models(raw)
    neutral_model = os_model.neutralise_treatment_terms()
    first = balanced_subtype_sample(os_test, PER_SUBTYPE)
    second = balanced_subtype_sample(os_test, PER_SUBTYPE, offset=PER_SUBTYPE)
    if set(first["patient_id"]) & set(second["patient_id"]):
        raise SystemExit("cohorts A and B are not disjoint")
    sample = pd.concat([first, second], ignore_index=True)

    environments, plans = {}, {}
    for _, row in sample.iterrows():
        patient = patient_from_row(row)
        environment = DynamicBreastCancerEnvironment(
            patient, make_risk_table(row, neutral_model, rfs_model), base_config)
        environments[patient.patient_id] = environment
        plans[patient.patient_id] = nccn_followup_state(environment)

    print(f"환자 {len(environments)}명 · 팔 2개 · 판돈 {len(SCALES)}단계 "
          f"· 등급 {len(GRADES)}개 (탐색 없음)", flush=True)
    print(f"오늘의 config는 이 다이얼의 {today_scale:.3f} 지점에 있다 "
          f"(구제 사망 위험비 "
          f"{base_config.hazard_multipliers['salvage']['systemic']['death']})",
          flush=True)

    ladder_rows = [row for scale in SCALES
                   for row in declared_ladder(base_config, scale)]
    pd.DataFrame(ladder_rows).to_csv(TABLE_DIR / "declared_lines.csv", index=False)
    print("\n=== 다이얼이 선언하는 것 (사망 위험비) ===", flush=True)
    print(pd.DataFrame(ladder_rows).pivot(index="scale", columns="line",
                                          values="death_hazard").round(3).to_string(),
          flush=True)

    table = sweep(environments, plans)
    summary = summarise(table)
    summary.to_csv(TABLE_DIR / "stakes_summary.csv", index=False)
    choices = (table.groupby(["arm", "scale", "grade", "best_line"]).size()
               .rename("decisions").reset_index())
    choices.to_csv(TABLE_DIR / "line_choice_by_grade.csv", index=False)
    print("\n=== 팔별 요약 ===", flush=True)
    print(summary.round(5).to_string(index=False), flush=True)

    indexed = summary.set_index(["arm", "scale"])
    null_rows = summary[np.isclose(summary["scale"], 0.0)]
    null_ok = bool(
        (null_rows[["channel_stakes", "grade_switch_share",
                    "information_per_decision"]].abs() < NULL_TOLERANCE).all().all())

    ladder_full = indexed.loc[("ladder", 1.0)]
    binary_full = indexed.loc[("binary", 1.0)]
    matched = at_matched_stakes(summary, float(ladder_full["channel_stakes"]))
    crossing = crossing_scale(summary, "ladder", ACTION_VALUE_NOISE)

    costs = cost_sweep(environments, plans)
    cost_summary = summarise_costs(costs)
    cost_summary.to_csv(TABLE_DIR / "cost_sweep.csv", index=False)
    (costs.groupby(["cost_multiple", "grade", "best_line"]).size()
     .rename("decisions").reset_index()
     .to_csv(TABLE_DIR / "line_choice_by_cost.csv", index=False))
    print("")
    print("=== (사후) 강한 단의 값을 올리면 — 판돈은 1.0 고정 ===", flush=True)
    print(cost_summary.round(5).to_string(index=False), flush=True)
    traded = cost_summary[cost_summary["lines_used"] > 1]
    split = cost_summary[cost_summary["grade_switch_share"] > 0]
    first_trade = float(traded["cost_multiple"].min()) if len(traded) else float("nan")
    first_split = float(split["cost_multiple"].min()) if len(split) else float("nan")
    best_cost_info = float(cost_summary["information_per_decision"].max())
    print(f"  단이 갈리기 시작: 비용 배수 {first_trade} "
          f"· 등급이 갈리기 시작: {first_split} "
          f"· 최대 정보가치 {best_cost_info:.5f} (잡음 {ACTION_VALUE_NOISE})",
          flush=True)

    # The decisive comparison, and the one the pre-registered version could not
    # make: take the ladder at its best cost setting and ask what the binary arm
    # offers *at the same channel stakes*. If the ladder wins there, the fix is
    # shape rather than size - and it wins while carrying LESS stakes, which is
    # the opposite of what "make the channel bigger" would predict.
    peak = cost_summary.loc[cost_summary["information_per_decision"].idxmax()]
    peak_matched = at_matched_stakes(summary, float(peak["channel_stakes"]))
    peak_matched["ladder_information"] = float(peak["information_per_decision"])
    peak_matched["ladder_cost_multiple"] = float(peak["cost_multiple"])
    binary_at_peak = float(peak_matched.get("binary_information", float("nan")))
    grade_share = (float(peak["information_per_decision"])
                   / float(peak["channel_stakes"])
                   if peak["channel_stakes"] > 0 else float("nan"))
    print("")
    print("=== (사후) 봉우리에서, 같은 판돈의 이진 팔과 비교 ===", flush=True)
    print(f"  사다리 (비용 {peak['cost_multiple']:.1f}배): 판돈 "
          f"{peak['channel_stakes']:.4f} · 정보가치 "
          f"{peak['information_per_decision']:.5f} "
          f"· 판돈 대비 {grade_share:.1%}", flush=True)
    print(f"  이진 (같은 판돈): 정보가치 {binary_at_peak:.5f}", flush=True)
    print(f"  v2.4 (등급만, 이진): 판돈 {V24_CHANNEL_STAKES} · 정보가치 "
          f"{V24_INFORMATION_VALUE} · 판돈 대비 "
          f"{V24_INFORMATION_VALUE / V24_CHANNEL_STAKES:.1%}", flush=True)

    print("\n=== 같은 판돈에서 비교 ===", flush=True)
    print(f"  목표 판돈 {matched['target_stakes']:.4f}", flush=True)
    for arm in ("binary", "ladder"):
        if matched.get(f"{arm}_reaches_target"):
            print(f"  {arm:7s}: 정보가치 {matched[f'{arm}_information']:.5f} "
                  f"(판돈 척도 {matched.get(f'{arm}_scale', float('nan')):.3f})",
                  flush=True)
        else:
            print(f"  {arm:7s}: 그 판돈에 도달하지 못함", flush=True)
    if "structure_multiple" in matched:
        print(f"  구조 배수 {matched['structure_multiple']:.2f}배 "
              f"(하한 {STRUCTURE_ADVANTAGE_FLOOR})", flush=True)

    verdict = {
        "patients": int(len(environments)),
        "scales": list(SCALES),
        "separation": SEPARATION,
        "today_config_scale": today_scale,
        "v24_channel_stakes": V24_CHANNEL_STAKES,
        "v24_information_value": V24_INFORMATION_VALUE,
        "action_value_noise": ACTION_VALUE_NOISE,
        "ladder_stakes_at_full": float(ladder_full["channel_stakes"]),
        "binary_stakes_at_full": float(binary_full["channel_stakes"]),
        "ladder_stakes_multiple_of_v24": float(
            ladder_full["channel_stakes"] / V24_CHANNEL_STAKES),
        "ladder_switch_share_at_full": float(ladder_full["grade_switch_share"]),
        "binary_switch_share_at_full": float(binary_full["grade_switch_share"]),
        "ladder_information_at_full": float(ladder_full["information_per_decision"]),
        "binary_information_at_full": float(binary_full["information_per_decision"]),
        "ladder_information_pooled_at_full": float(ladder_full["information_pooled"]),
        "matched_stakes": matched,
        "crossing_scale_for_noise_floor": crossing,
        "cost_sweep_posthoc": {
            "multiples": list(COST_MULTIPLES),
            "first_multiple_with_two_lines": first_trade,
            "first_multiple_with_grade_split": first_split,
            "max_information_per_decision": best_cost_info,
            "clears_noise_floor": bool(best_cost_info >= ACTION_VALUE_NOISE),
            "share_of_noise_floor": float(best_cost_info / ACTION_VALUE_NOISE),
            "peak_cost_multiple": float(peak["cost_multiple"]),
            "peak_channel_stakes": float(peak["channel_stakes"]),
            "peak_grade_switch_share": float(peak["grade_switch_share"]),
            "peak_grade_conditional_share_of_stakes": grade_share,
            "binary_information_at_peak_stakes": binary_at_peak,
            "v24_grade_conditional_share_of_stakes": float(
                V24_INFORMATION_VALUE / V24_CHANNEL_STAKES),
            "information_multiple_over_v24": float(
                best_cost_info / V24_INFORMATION_VALUE),
        },
        "pre_registered_ladder_has_a_dominant_rung": bool(
            int(ladder_full["lines_used"]) == 1),
        "null_control_prediction_met": null_ok,
        "stakes_prediction_met": bool(
            ladder_full["channel_stakes"]
            >= STAKES_MULTIPLE_FLOOR * V24_CHANNEL_STAKES),
        "state_dependence_prediction_met": bool(
            ladder_full["grade_switch_share"] >= LADDER_SWITCH_FLOOR
            and binary_full["grade_switch_share"] < BINARY_SWITCH_CEILING),
        "primary_prediction_met": bool(
            matched.get("structure_multiple", 0.0) >= STRUCTURE_ADVANTAGE_FLOOR),
        "crossing_prediction_met": bool(not np.isnan(crossing)),
    }

    metrics = {
        "run_date": RUN_DATE,
        "analysis_label": "stakes-map-v2.5",
        "question": (
            "Is the adaptive channel unusable because it is too small, or "
            "because it is a yes/no rather than a choice among treatment lines?"
        ),
        "estimand": (
            "None. A deterministic preview of the environment's own arithmetic: "
            "the expected remaining utility of each salvage line at a graded "
            "disease state, swept over the channel's declared stakes and over "
            "two channel structures. No search, no random draws, no change to "
            "the committed environment."
        ),
        "scope_warning": (
            "Everything here is a property of our declared parameters, not of "
            "patients. The ladder's endpoint is declared and cannot be "
            "estimated from any dataset we hold - MSK-CHORD's left truncation "
            "alone rules out a survival contrast by line (v2.2)."
        ),
        "prespecified_prediction": PRESPECIFIED_PREDICTION,
        "design": {
            "config": "configs/dynamic_v0_7.json",
            "disease_grading": "burden.py at separation 1.0 (v2.4)",
            "arms": {arm: list(lines) for arm, lines in ARMS.items()},
            "full_ladder": {line: dict(values)
                            for line, values in FULL_LADDER.items()},
            "patients": int(len(environments)),
            "sampling": "balanced subtype, two disjoint cohorts of 20",
            "search": "none - deterministic expectation",
        },
        "verdict": verdict,
    }
    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = {
        "run_date": RUN_DATE,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit_before_run": git_commit(),
        "inputs": input_manifest({"model_input": INPUT_CSV, "config": CONFIG_V07}),
        "entry_point": "analysis/55_run_stakes_map.py",
        "base_seed": None,
        "seed_note": "no randomness - the map is a deterministic expectation",
    }
    (REPORT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== 사전 예측 채점 ===", flush=True)
    for key, met in (
        ("1 영대조 (판돈 0 = 채널 없음)", verdict["null_control_prediction_met"]),
        ("2 판돈", verdict["stakes_prediction_met"]),
        ("3 상태 의존", verdict["state_dependence_prediction_met"]),
        ("4 해상도 — 구조 대 크기 (주)", verdict["primary_prediction_met"]),
        ("5 잡음 바닥 통과", verdict["crossing_prediction_met"]),
    ):
        print(f"  {key}: {'통과' if met else '빗나감'}", flush=True)


if __name__ == "__main__":
    main()
