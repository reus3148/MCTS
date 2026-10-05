"""Would a graded disease state create a decision? (v2.4, arithmetic only)

v2.0 asked where the salvage decision splits and found that it does not: every
patient's indifference hazard ratio sat between 0.95 and 0.96, a spread of
**0.01**, because the declared benefit is multiplicative and the declared cost
is fixed. Where nobody differs there is nothing to adapt to, and the searcher
run at that point produced noise - it missed a known answer by 44 percentage
points.

v2.3 then located the missing ingredient, and it was not the one we expected.
Performance status moves and tracks disease, but recorded care does not
visibly act on it (stop-rate gap +1.3 points against a detectable minimum of
3.0, no dose-response). What recorded care *does* act on is the thing a
progression call measures: treatment changes **2.41x** more often after
"progressing" than after "controlled" (v2.2). Our environment carries that as
``recurred`` - a **binary** flag, so every recurred patient faces the same
post-recurrence hazard, which is precisely the condition that flattened v2.0's
map.

So this run asks, **without any search and without changing the environment**,
whether grading that flag would create an indifference region.
``analysis/dynamic/burden.py`` splits the post-recurrence death hazard across
the three levels a progression call distinguishes, weighted by MSK-CHORD's
observed mix (56.0% / 20.1% / 23.9%) and renormalised so the weighted mean
hazard is unchanged at every setting. How far apart the grades sit is a
**dial**, the pattern v2.1 established - applied to state rather than to
benefit.

The readout is v2.0's, so the two are directly comparable: for every patient,
recurrence year and grade, the salvage hazard ratio at which treating stops
paying. v2.0's answer was a spread of 0.01 across patients. The question here
is what the spread becomes **across grades**, and at what separation it
becomes large enough to matter.

Why this is worth a run at all: if the answer is "it stays flat", then grading
the state is not the fix either, and three versions of environment work are
saved. If the answer is "it splits", we have a target to build toward and a
number to build it to.

PRE-SPECIFIED PREDICTIONS, recorded before the run
--------------------------------------------------
1. **Known-answer check.** At separation 0 the across-grade spread of the
   indifference ratio is **< 0.005** and the per-patient spread reproduces
   v2.0's **0.01** to within 0.005. Separation 0 *is* the current environment
   by construction, so anything else means the dial is not neutral and every
   other number here is suspect.
2. **Primary - grading creates a region.** At separation 1.0 the across-grade
   spread of the indifference ratio is **>= 0.05**, five times v2.0's
   across-patient spread.
3. **Monotone.** The across-grade spread grows monotonically with separation.
4. **The grades disagree somewhere.** There exists a salvage hazard ratio at
   which, at separation 1.0, a majority of ``progressing`` cells favour
   treating while a majority of ``controlled`` cells favour declining. A
   spread that never produces disagreement would be a difference of degree
   with no decision in it.

Prediction 2 is the one that decides what v2.5 does. Prediction 4 is the one
most likely to fail while 2 passes: the grades can separate smoothly and still
never straddle a usable ratio.
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

from analysis.dynamic.burden import (  # noqa: E402
    FULL_SEPARATION,
    GRADE_WEIGHTS,
    GRADES,
    declared_grades,
    graded_environment,
)
from analysis.dynamic.cohort import (  # noqa: E402
    balanced_subtype_sample,
    build_reward_models,
    git_commit,
    input_manifest,
    make_risk_table,
)
from analysis.dynamic.config import load_dynamic_config  # noqa: E402
from analysis.dynamic.environment import DynamicBreastCancerEnvironment  # noqa: E402
from analysis.dynamic.indifference import salvage_advantage  # noqa: E402
from analysis.dynamic.policies import DynamicNccnPolicy  # noqa: E402
from analysis.dynamic.schema import patient_from_row  # noqa: E402

REPORT_DIR = ROOT / "reports" / "burden-map-v2.4"
TABLE_DIR = REPORT_DIR / "tables"

#: The full grid goes here, not into ``tables/``. It is 40 patients x 6
#: separations x 3 grades x 4 years x 21 ratios = 60,480 rows and 3.3 MB,
#: which is twenty times v2.0's committed map for a report whose figures need
#: only the aggregates. ``data/`` is git-ignored; everything in ``tables/`` is
#: small enough to read.
DERIVED_DIR = ROOT / "data" / "processed" / "burden_map_v2_4"
CONFIG_V07 = ROOT / "configs" / "dynamic_v0_7.json"
INPUT_CSV = ROOT / "data" / "processed" / "patients_with_nccn.csv"

RUN_DATE = "2026-10-05"
PER_SUBTYPE = 5                      # x4 subtypes x2 offsets = 40 patients
YEARS = (1, 2, 3, 4)                 # recurrence years that leave horizon
RATIOS = np.round(np.arange(0.80, 1.001, 0.01), 4)
SEPARATIONS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)

#: v2.0's headline, the number this run is measured against.
V20_PATIENT_SPREAD = 0.01

#: Action-value noise at budget 1024, from v2.0's known-answer check. The bar
#: any decision-relevant signal has to clear before a searcher can use it.
ACTION_VALUE_NOISE = 0.02

NULL_SPREAD_CEILING = 0.005
NULL_PATIENT_TOLERANCE = 0.005
GRADE_SPREAD_FLOOR = 0.05

PRESPECIFIED_PREDICTION = {
    "null_control": (
        f"At separation 0 the across-grade spread is < {NULL_SPREAD_CEILING} "
        f"and the per-patient spread reproduces v2.0's {V20_PATIENT_SPREAD} "
        f"to within {NULL_PATIENT_TOLERANCE}."
    ),
    "primary_grading_creates_a_region": (
        f"At separation 1.0 the across-grade spread of the indifference ratio "
        f"is >= {GRADE_SPREAD_FLOOR} - five times v2.0's across-patient spread."
    ),
    "monotone": (
        "The across-grade spread grows monotonically with separation."
    ),
    "grades_disagree": (
        "At separation 1.0 some salvage hazard ratio has a majority of "
        "progressing cells favouring treatment while a majority of controlled "
        "cells favour declining."
    ),
    "why": (
        "v2.0 found no indifference region because a binary recurrence flag "
        "gives every patient the same post-recurrence hazard. v2.2 showed "
        "recorded care acts on a graded progression signal (2.41x) and v2.3 "
        "showed it does not visibly act on performance status. Grading the "
        "flag is the remaining candidate, and arithmetic can test it before "
        "any environment code is written."
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


def with_salvage_ratio(environment: DynamicBreastCancerEnvironment, ratio: float):
    """A copy whose systemic-salvage death multiplier is ``ratio``."""
    config = environment.config
    hazards = dict(config.hazard_multipliers)
    salvage = {key: dict(value) for key, value in hazards["salvage"].items()}
    salvage["systemic"] = {**salvage["systemic"], "death": float(ratio)}
    hazards["salvage"] = salvage
    return DynamicBreastCancerEnvironment(
        environment.patient, environment.risk_table,
        replace(config, hazard_multipliers=hazards))


def burden_map(environments, plans, separations, ratios, years) -> pd.DataFrame:
    """Salvage advantage for patient x year x grade x separation x ratio.

    Deterministic throughout: :func:`salvage_advantage` is an expectation over
    the environment's own annual event model, so this is arithmetic rather
    than simulation. Five seconds instead of forty minutes per setting, which
    is the whole point of running it before building anything.
    """
    rows = []
    for patient_id, environment in environments.items():
        base = plans[patient_id]
        for separation in separations:
            for grade in GRADES:
                graded = graded_environment(environment, grade, separation)
                for year in years:
                    state = replace(base, phase="salvage", year=int(year),
                                    recurred=True, recurrence_year=int(year))
                    for ratio in ratios:
                        rows.append({
                            "patient_id": patient_id,
                            "separation": float(separation),
                            "grade": grade,
                            "recurrence_year": int(year),
                            "hazard_ratio": float(ratio),
                            "advantage": salvage_advantage(
                                with_salvage_ratio(graded, float(ratio)), state),
                        })
    return pd.DataFrame(rows)


def crossing_ratio(block: pd.DataFrame) -> float:
    """Highest hazard ratio at which treating still pays, by linear interpolation.

    ``advantage`` falls as the ratio rises (a weaker salvage buys less), so the
    crossing is where it last changes sign. Returns NaN when the sign never
    changes - treating pays everywhere, or nowhere.
    """
    ordered = block.sort_values("hazard_ratio")
    ratios = ordered["hazard_ratio"].to_numpy(dtype=float)
    values = ordered["advantage"].to_numpy(dtype=float)
    positive = values > 0
    if positive.all() or not positive.any():
        return float("nan")
    last = int(np.max(np.flatnonzero(positive)))
    if last + 1 >= len(values):
        return float("nan")
    low, high = ratios[last], ratios[last + 1]
    a, b = values[last], values[last + 1]
    return float(low + (high - low) * a / (a - b))


def crossings(table: pd.DataFrame) -> pd.DataFrame:
    """Indifference ratio per (separation, grade, patient, recurrence year)."""
    rows = []
    keys = ["separation", "grade", "patient_id", "recurrence_year"]
    for values, block in table.groupby(keys):
        rows.append(dict(zip(keys, values),
                         indifference_ratio=crossing_ratio(block)))
    return pd.DataFrame(rows)


def grade_spread(crossing: pd.DataFrame) -> pd.DataFrame:
    """Across-grade and across-patient spread of the indifference ratio."""
    rows = []
    for separation, block in crossing.groupby("separation"):
        by_grade = block.groupby("grade")["indifference_ratio"].median()
        by_patient = block.groupby("patient_id")["indifference_ratio"].median()
        record = {
            "separation": float(separation),
            "grade_spread": float(by_grade.max() - by_grade.min()),
            "patient_spread": float(by_patient.max() - by_patient.min()),
            "cells": int(len(block)),
            "undefined": int(block["indifference_ratio"].isna().sum()),
        }
        for grade in GRADES:
            record[f"ratio_{grade}"] = float(by_grade.get(grade, float("nan")))
        rows.append(record)
    return pd.DataFrame(rows).sort_values("separation").reset_index(drop=True)


def value_of_grade_information(table: pd.DataFrame, separation: float) -> pd.DataFrame:
    """What is it worth to *know* the grade before choosing? (Howard 1966)

    POST-HOC, and it should not have been. Our own triad is stakes x
    state-dependence x resolution (v2.0), and the four pre-specified
    predictions covered the first two and not the third. Spread and
    disagreement say the grades differ; neither says the difference is large
    enough for a searcher to act on, which is exactly what sank v2.0 - the
    option difference there was 0.002 against an action-value noise floor of
    roughly 0.02 at budget 1024, and the searcher missed a known answer by 44
    percentage points.

    ``informed`` is the expected advantage of a policy that sees the grade and
    may decline per grade; ``blind`` is the best single decision taken under
    the grade distribution alone. Their difference is the expected value of
    perfect information about the grade, in the same utility units as every
    other number in this project - so it can be compared with the noise floor
    directly.
    """
    block = table[np.isclose(table["separation"], separation)]
    rows = []
    for ratio, by_ratio in block.groupby("hazard_ratio"):
        means = by_ratio.groupby("grade")["advantage"].mean()
        informed = sum(GRADE_WEIGHTS[grade] * max(float(means.get(grade, 0.0)), 0.0)
                       for grade in GRADES)
        pooled = sum(GRADE_WEIGHTS[grade] * float(means.get(grade, 0.0))
                     for grade in GRADES)
        blind = max(pooled, 0.0)
        rows.append({
            "hazard_ratio": float(ratio),
            "informed": informed,
            "blind": blind,
            "value_of_knowing": informed - blind,
            "best_single_choice": "treat" if pooled > 0 else "decline",
        })
    return pd.DataFrame(rows).sort_values("hazard_ratio").reset_index(drop=True)


def disagreement(table: pd.DataFrame, separation: float) -> dict:
    """Is there a ratio where the grades give opposite majority answers?"""
    block = table[np.isclose(table["separation"], separation)]
    share = (block.assign(treat=block["advantage"] > 0)
             .groupby(["hazard_ratio", "grade"])["treat"].mean().unstack())
    if not {"controlled", "progressing"} <= set(share.columns):
        return {"exists": False}
    split = share[(share["progressing"] > 0.5) & (share["controlled"] < 0.5)]
    out = {"exists": bool(len(split)), "ratios": int(len(split))}
    if len(split):
        out["widest_ratio"] = float(
            (split["progressing"] - split["controlled"]).idxmax())
        row = split.loc[out["widest_ratio"]]
        out["progressing_treat_share"] = float(row["progressing"])
        out["controlled_treat_share"] = float(row["controlled"])
        out["indeterminate_treat_share"] = float(row.get("indeterminate", float("nan")))
        out["ratio_range"] = [float(split.index.min()), float(split.index.max())]
    return out


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    base_config = load_dynamic_config(CONFIG_V07)

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
    print(f"환자 {len(environments)}명 · 분리 {len(SEPARATIONS)}단계 "
          f"· 등급 {len(GRADES)}개 · 위험비 {len(RATIOS)}개 (탐색 없음)", flush=True)

    ladder = pd.DataFrame(
        [row for separation in SEPARATIONS
         for row in declared_grades(base_config, separation)])
    ladder.to_csv(TABLE_DIR / "declared_grades.csv", index=False)
    print("\n=== 다이얼이 선언하는 것 (재발 후 사망 위험 배수) ===", flush=True)
    print(ladder.pivot(index="separation", columns="grade",
                       values="death_after_recurrence").round(3).to_string(),
          flush=True)

    table = burden_map(environments, plans, SEPARATIONS, RATIOS, YEARS)
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(DERIVED_DIR / "burden_map.csv", index=False)
    # Panel C needs the treat share by ratio and grade at full separation, so
    # it ships as its own small table rather than as a reason to commit the grid.
    full_block = table[np.isclose(table["separation"], 1.0)]
    treat_share = (full_block.assign(treat=full_block["advantage"] > 0)
                   .groupby(["hazard_ratio", "grade"])["treat"]
                   .agg(["mean", "size"]).reset_index()
                   .rename(columns={"mean": "treat_share", "size": "cells"}))
    treat_share.to_csv(TABLE_DIR / "treat_share_by_grade.csv", index=False)

    crossing = crossings(table)
    crossing.to_csv(TABLE_DIR / "indifference_by_grade.csv", index=False)
    spread = grade_spread(crossing)
    spread.to_csv(TABLE_DIR / "spread_by_separation.csv", index=False)
    print("\n=== 무차별 위험비의 퍼짐 ===", flush=True)
    print(spread.round(4).to_string(index=False), flush=True)

    null_row = spread[np.isclose(spread["separation"], 0.0)].iloc[0]
    full_row = spread[np.isclose(spread["separation"], 1.0)].iloc[0]
    spreads = spread["grade_spread"].to_numpy(dtype=float)
    monotone = bool(np.all(np.diff(spreads) > -1e-12))
    split = disagreement(table, 1.0)
    information = value_of_grade_information(table, 1.0)
    information.to_csv(TABLE_DIR / "value_of_grade_information.csv", index=False)
    best = information.loc[information["value_of_knowing"].idxmax()]
    channel_stakes = float(
        table[np.isclose(table["separation"], 1.0)]
        .groupby("hazard_ratio")["advantage"].mean().max())
    print("\n=== 등급이 서로 다른 답을 내는 구간 (분리 1.0) ===", flush=True)
    if split["exists"]:
        print(f"  위험비 {split['ratio_range'][0]:.2f}~{split['ratio_range'][1]:.2f} "
              f"({split['ratios']}개 지점). 가장 벌어지는 "
              f"{split['widest_ratio']:.2f}에서 진행 "
              f"{split['progressing_treat_share']:.0%} · 판정보류 "
              f"{split['indeterminate_treat_share']:.0%} · 안정 "
              f"{split['controlled_treat_share']:.0%}", flush=True)
    else:
        print("  없다 - 등급이 벌어져도 같은 답을 낸다", flush=True)

    print("")
    print("=== 등급을 아는 것의 값 (완전정보, 분리 1.0) ===", flush=True)
    print(information[information["value_of_knowing"] > 1e-9].round(5)
          .to_string(index=False), flush=True)
    print(f"  최대 {best['value_of_knowing']:.5f} (위험비 "
          f"{best['hazard_ratio']:.2f}) · 예산 1024의 행동값 잡음 "
          f"{ACTION_VALUE_NOISE:.2f} 대비 "
          f"1/{ACTION_VALUE_NOISE / best['value_of_knowing']:.0f}", flush=True)
    print(f"  채널 전체 판돈(최강 구제에서의 평균 이점): {channel_stakes:.4f}",
          flush=True)

    verdict = {
        "patients": int(len(environments)),
        "recurrence_years": list(YEARS),
        "separations": list(SEPARATIONS),
        "hazard_ratios": [float(RATIOS.min()), float(RATIOS.max()), len(RATIOS)],
        "grade_weights": {grade: float(GRADE_WEIGHTS[grade]) for grade in GRADES},
        "full_separation": {grade: float(FULL_SEPARATION[grade]) for grade in GRADES},
        "v20_patient_spread": V20_PATIENT_SPREAD,
        "null_grade_spread": float(null_row["grade_spread"]),
        "null_patient_spread": float(null_row["patient_spread"]),
        "full_grade_spread": float(full_row["grade_spread"]),
        "full_patient_spread": float(full_row["patient_spread"]),
        "ratio_by_grade_at_full": {
            grade: float(full_row[f"ratio_{grade}"]) for grade in GRADES},
        "undefined_cells_at_full": int(full_row["undefined"]),
        "monotone": monotone,
        "disagreement_at_full": split,
        "action_value_noise": ACTION_VALUE_NOISE,
        "max_value_of_grade_information": float(best["value_of_knowing"]),
        "max_value_at_ratio": float(best["hazard_ratio"]),
        "noise_to_information_ratio": float(
            ACTION_VALUE_NOISE / best["value_of_knowing"]),
        "channel_stakes": channel_stakes,
        "resolution_clears_noise_posthoc": bool(
            best["value_of_knowing"] >= ACTION_VALUE_NOISE),
        "null_control_prediction_met": bool(
            null_row["grade_spread"] < NULL_SPREAD_CEILING
            and abs(null_row["patient_spread"] - V20_PATIENT_SPREAD)
            <= NULL_PATIENT_TOLERANCE),
        "primary_prediction_met": bool(
            full_row["grade_spread"] >= GRADE_SPREAD_FLOOR),
        "monotone_prediction_met": monotone,
        "disagreement_prediction_met": bool(split["exists"]),
    }

    metrics = {
        "run_date": RUN_DATE,
        "analysis_label": "burden-map-v2.4",
        "question": (
            "If the binary recurrence flag were graded into the three levels a "
            "progression call distinguishes, would an indifference region "
            "appear where v2.0 found none?"
        ),
        "estimand": (
            "None. A deterministic preview of the environment's own arithmetic: "
            "the expected remaining utility of treating versus declining at a "
            "salvage state, swept over the declared salvage hazard ratio, the "
            "grade of disease burden and the separation between grades. No "
            "search, no random draws, no change to the environment."
        ),
        "scope_warning": (
            "Everything here is a property of our declared parameters, not of "
            "patients. It says whether a graded state *could* carry a decision "
            "under the environment we have declared; it does not say that "
            "grading is clinically correct, and the separation endpoint is "
            "declared rather than estimated."
        ),
        "prespecified_prediction": PRESPECIFIED_PREDICTION,
        "design": {
            "config": "configs/dynamic_v0_7.json",
            "patients": int(len(environments)),
            "sampling": "balanced subtype, two disjoint cohorts of 20",
            "grade_weight_source": (
                "MSK-CHORD breast cohort, 32,977 scored progression calls "
                "(reports/msk-chord-profile-v2.2)"
            ),
            "mean_neutral": True,
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
        "inputs": input_manifest({
            "model_input": INPUT_CSV, "config": CONFIG_V07}),
        "entry_point": "analysis/53_run_burden_map.py",
        "derived_dir": str(DERIVED_DIR.relative_to(ROOT)).replace("\\", "/"),
        "derived_note": (
            "The full 60,480-row map is written there, not to tables/: the "
            "report's figures need only the aggregates, and data/ is "
            "git-ignored."
        ),
        "base_seed": None,
        "seed_note": "no randomness - the map is a deterministic expectation",
    }
    (REPORT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== 사전 예측 채점 ===", flush=True)
    for key, met in (
        ("1 영대조 (분리 0 = 현재 환경)", verdict["null_control_prediction_met"]),
        ("2 등급화가 구간을 만든다 (주)", verdict["primary_prediction_met"]),
        ("3 단조", verdict["monotone_prediction_met"]),
        ("4 등급이 서로 다른 답을 낸다", verdict["disagreement_prediction_met"]),
    ):
        print(f"  {key}: {'통과' if met else '빗나감'}", flush=True)
    print(f"  (사후, 사전등록했어야 함) 해상도: "
          f"{'통과' if verdict['resolution_clears_noise_posthoc'] else '빗나감'} "
          f"- 정보가치 {verdict['max_value_of_grade_information']:.5f} 대 "
          f"잡음 {ACTION_VALUE_NOISE:.2f}", flush=True)


if __name__ == "__main__":
    main()
