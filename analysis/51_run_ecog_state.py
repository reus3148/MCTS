"""Is performance status a state worth planning over? (v2.3 part 1)

v2.2 ended with a target: real care offers a median of 3 adaptation
opportunities over five years in 66% of patients, against our environment's one
in 20.8%. The obvious next move is to open more decision points. v1.8, v1.9 and
v2.0 say not to do that yet.

Those three versions opened a decision point each and each one failed a
different leg of the same triad - stakes, state-dependence, resolution. Look at
what our state actually carries: ``(phase, year, recurred, alive,
toxicity_count)`` plus the treatments already chosen. ``recurred`` is binary and
already opens salvage (v0.6). So a new mid-course decision would have **almost
nothing to condition on**, and would collapse to "always X" exactly as v1.8's
salvage did at 62% - 100% refusal and v1.9's did at 98.7% treatment.

The missing ingredient is state, not decisions. MSK-CHORD carries the obvious
candidate - ECOG, recorded for 73.6% of the cohort with changes documented in
45.2% - and our schema has no field for it. But "it moves" is not enough.
A variable earns a place in the state vector only if **the policy would act
differently on it**, and the cheapest way to find out is to ask whether
clinicians do.

So this run asks three questions, and deliberately writes no environment code
until they are answered:

    1. **Does treatment stop after performance status worsens?** (does anyone
       act on it)
    2. **Does the level gate the most toxic channel?** (is it an eligibility
       variable, which our environment models only through ``toxicity_count``)
    3. **Does it move faster once the disease progresses?** (is it tracking
       disease, or drifting with time - the state-dependence leg itself)

Descriptive throughout, like v1.6 and v2.2. Nothing here is adjusted or causal;
reading a confounded stop rate as an effect would repeat v1.3's mistake.

PRE-SPECIFIED PREDICTIONS, recorded before the run
--------------------------------------------------
None of these could be read off v2.2, which computed only the marginal ECOG
distribution, the share with a documented change, and the change-point
encoding. No association between ECOG and treatment has been computed at any
point in this project.

1. **Primary - treatment stops after worsening.** Among changes occurring while
   the patient is on systemic therapy, the share where treatment has stopped 90
   days later is **at least 10 percentage points higher** after a worsening
   than after an improvement. If false, ECOG is a prognostic covariate that
   nobody acts on, and adding it to the state vector would add noise rather
   than a decision variable.
2. **Placebo in time.** For worsenings, the forward-minus-backward gap exceeds
   the same gap for improvements. The backward window asks the mirror question
   (was the patient off treatment 90 days *before* the change), which the change
   cannot have caused.
3. **The level gates chemotherapy.** Among systemic treatment starts, the share
   that are chemotherapy is **at least 10 percentage points lower** when the
   ECOG in force is >= 2 than when it is 0 or 1. Our environment's only
   eligibility gate is ``toxicity_count``; if this holds, it is missing one that
   recorded care applies.
4. **State-dependence.** Recorded worsenings per person-year are **at least
   1.5x higher** after a patient's first progression call than before it. A
   variable that drifts at the same rate regardless of disease carries no
   information a policy could use.

Prediction 3 is the one most likely to fail, and informatively: ECOG in this
release is NLP-derived from notes, and a level recorded *because* a patient was
about to start chemotherapy would bias it toward the healthy end.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.dynamic.cohort import git_commit  # noqa: E402
from analysis.dynamic.experiment_utils import minimum_detectable_difference  # noqa: E402
from analysis.msk.adaptation import followup_days  # noqa: E402
from analysis.msk.loader import DATA_DIR, load  # noqa: E402
from analysis.msk.performance import (  # noqa: E402
    POOR_PERFORMANCE,
    chemo_share_by_ecog,
    ecog_changes,
    stop_rates,
    stop_rates_by_size,
    stop_table,
    transition_rates,
)

REPORT_DIR = ROOT / "reports" / "ecog-state-v2.3"
TABLE_DIR = REPORT_DIR / "tables"
DERIVED_DIR = ROOT / "data" / "external" / "msk_chord_2024_derived"

RUN_DATE = "2026-10-05"
HORIZON_YEARS = 5.0
WINDOW_DAYS = 90               # v1.6/v2.2's window, kept identical
WINDOW_SENSITIVITY = (60, 90, 180)

STOP_GAP_FLOOR = 0.10
CHEMO_GATE_FLOOR = 0.10
WORSENING_RATIO_FLOOR = 1.5

PRESPECIFIED_PREDICTION = {
    "primary_treatment_stops_after_worsening": (
        f"Among ECOG changes on treatment, the share stopped after "
        f"{WINDOW_DAYS} days is >= {STOP_GAP_FLOOR:.0%} points higher after a "
        "worsening than after an improvement."
    ),
    "placebo_in_time": (
        "For worsenings the forward-minus-backward gap exceeds the same gap "
        "for improvements."
    ),
    "level_gates_chemotherapy": (
        f"Among systemic starts, the chemotherapy share is >= "
        f"{CHEMO_GATE_FLOOR:.0%} points lower at ECOG >= {POOR_PERFORMANCE} "
        "than at ECOG 0-1."
    ),
    "state_dependence": (
        f"Recorded worsenings per person-year are >= {WORSENING_RATIO_FLOOR}x "
        "higher after the first progression call than before it."
    ),
    "why": (
        "v2.2 gave a target (3 adaptation points in 66% of patients) but our "
        "state carries almost nothing to condition on, so opening decisions "
        "first would repeat v1.8/v1.9/v2.0. A variable earns a place in the "
        "state vector only if the policy would act differently on it."
    ),
}


def two_proportion_z(p_a: float, n_a: int, p_b: float, n_b: int) -> tuple[float, float]:
    if min(n_a, n_b) == 0:
        return float("nan"), float("nan")
    pooled = (p_a * n_a + p_b * n_b) / (n_a + n_b)
    standard_error = float(np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b)))
    if standard_error == 0:
        return float("nan"), standard_error
    return float((p_a - p_b) / standard_error), standard_error


def clustered_stop_contrast(table: pd.DataFrame) -> dict:
    """Patient-clustered worsened-minus-improved contrast.

    Changes nest within patients, so the unclustered z overstates precision.
    Patients contributing both directions appear in both arms, so this is still
    approximate - and it is the larger of the two standard errors, which is the
    direction to err in.
    """
    per_patient = (table.groupby(["PATIENT_ID", "direction"])["stopped_after"]
                   .mean().rename("rate").reset_index())
    wide = per_patient.pivot(index="PATIENT_ID", columns="direction", values="rate")
    out: dict[str, float] = {}
    for direction in ("worsened", "improved"):
        if direction in wide:
            values = wide[direction].dropna()
            out[f"{direction}_patients"] = int(len(values))
            out[f"{direction}_mean"] = float(values.mean())
    if {"worsened", "improved"} <= set(wide.columns):
        a, b = wide["worsened"].dropna(), wide["improved"].dropna()
        difference = float(a.mean() - b.mean())
        standard_error = float(np.sqrt(
            a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
        out["difference"] = difference
        out["standard_error"] = standard_error
        out["z"] = difference / standard_error if standard_error else float("nan")
        out["minimum_detectable_difference"] = (
            minimum_detectable_difference(standard_error)
            if standard_error > 0 else float("nan"))
    return out


def window_sensitivity(changes: pd.DataFrame, treatment: pd.DataFrame,
                       followup: pd.Series) -> pd.DataFrame:
    rows = []
    for days in WINDOW_SENSITIVITY:
        rates = stop_rates(stop_table(changes, treatment, followup, days))
        if rates.empty:
            continue
        indexed = rates.set_index("direction")
        if not {"worsened", "improved"} <= set(indexed.index):
            continue
        worsened, improved = indexed.loc["worsened"], indexed.loc["improved"]
        rows.append({
            "window_days": days,
            "changes": int(indexed["changes"].sum()),
            "stopped_worsened": float(worsened["stopped_after"]),
            "stopped_improved": float(improved["stopped_after"]),
            "gap": float(worsened["stopped_after"] - improved["stopped_after"]),
            "placebo_worsened": float(worsened["forward_minus_backward"]),
            "placebo_improved": float(improved["forward_minus_backward"]),
        })
    return pd.DataFrame(rows)


def chemo_gate(shares: pd.DataFrame) -> dict:
    """Collapse the per-level chemotherapy shares onto the declared cut."""
    good = shares[shares["ecog"] < POOR_PERFORMANCE]
    poor = shares[shares["ecog"] >= POOR_PERFORMANCE]
    out: dict[str, float] = {}
    for label, block in (("good", good), ("poor", poor)):
        starts = int(block["starts"].sum())
        out[f"{label}_starts"] = starts
        out[f"{label}_chemo_share"] = (
            float(block["chemo"].sum() / starts) if starts else float("nan"))
    out["gap"] = out["good_chemo_share"] - out["poor_chemo_share"]
    z, _ = two_proportion_z(
        out["good_chemo_share"], out["good_starts"],
        out["poor_chemo_share"], out["poor_starts"])
    out["z_unclustered"] = z
    return out


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    release = load()
    patients = release.patients
    cohort_size = int(patients["PATIENT_ID"].nunique())
    status = release.timeline("performance_status")
    treatment = release.timeline("treatment")

    print(f"유방암 환자 {cohort_size}명 · ECOG 기록 "
          f"{status['PATIENT_ID'].nunique()}명", flush=True)

    changes = ecog_changes(status, HORIZON_YEARS)
    changes.to_csv(DERIVED_DIR / "ecog_changes.csv", index=False)
    direction_counts = (changes.groupby("direction").size().rename("changes")
                        .reset_index())
    direction_counts["patients"] = [
        int(changes.loc[changes["direction"] == row.direction,
                        "PATIENT_ID"].nunique())
        for row in direction_counts.itertuples()]
    direction_counts.to_csv(TABLE_DIR / "ecog_changes.csv", index=False)
    print(f"\n=== ECOG 변화 {len(changes)}건 "
          f"({changes['PATIENT_ID'].nunique()}명) ===", flush=True)
    print(direction_counts.to_string(index=False), flush=True)

    # ---- 1. does treatment stop after worsening? ---------------------------
    followup = followup_days(patients, release.anchor)
    table = stop_table(changes, treatment, followup, WINDOW_DAYS)
    table.to_csv(DERIVED_DIR / "ecog_stop_table.csv", index=False)
    rates = stop_rates(table)
    rates.to_csv(TABLE_DIR / "stop_rates.csv", index=False)
    windows = window_sensitivity(changes, treatment, followup)
    windows.to_csv(TABLE_DIR / "window_sensitivity.csv", index=False)
    print(f"\n=== 치료 중단 ({WINDOW_DAYS}일 창, 치료 중 발생한 변화 "
          f"{len(table)}건) ===", flush=True)
    print(rates.to_string(index=False), flush=True)

    indexed = rates.set_index("direction")
    worsened, improved = indexed.loc["worsened"], indexed.loc["improved"]
    stop_gap = float(worsened["stopped_after"] - improved["stopped_after"])
    z_unclustered, _ = two_proportion_z(
        float(worsened["stopped_after"]), int(worsened["changes"]),
        float(improved["stopped_after"]), int(improved["changes"]))
    clustered = clustered_stop_contrast(table)
    print(f"  격차 {stop_gap:+.4f} · 비군집 z {z_unclustered:.1f} "
          f"· 환자군집 z {clustered.get('z', float('nan')):.1f} "
          f"(MDE {clustered.get('minimum_detectable_difference', float('nan')):.4f})",
          flush=True)

    # Post-hoc, in the order the failures were chased.
    #
    # (a) DOSE-RESPONSE. If stopping tracked performance status, the gap would
    #     widen with the size of the fall. A flat or inverse profile means the
    #     null is not an artefact of a crude event definition.
    # (b) REVERSE CAUSALITY. ``stopped_before`` is also the indicator
    #     "treatment started in the last 90 days". If worsenings follow recent
    #     starts more often than improvements do, ECOG is partly measuring
    #     treatment toxicity, which our environment already carries as
    #     ``toxicity_count``.
    by_size = stop_rates_by_size(table)
    by_size.to_csv(TABLE_DIR / "stop_rates_by_size.csv", index=False)
    print("")
    print("  (사후 a) 악화 크기별 - 용량-반응이 있나", flush=True)
    print(by_size.to_string(index=False), flush=True)
    recent_start = {
        row.direction: float(row.stopped_before) for row in rates.itertuples()}
    print(f"  (사후 b) 변화 90일 내 치료 시작: 악화 "
          f"{recent_start['worsened']:.1%} 대 호전 {recent_start['improved']:.1%}",
          flush=True)

    # ---- 2. does the level gate chemotherapy? ------------------------------
    shares = chemo_share_by_ecog(status, treatment, HORIZON_YEARS)
    shares.to_csv(TABLE_DIR / "chemo_share_by_ecog.csv", index=False)
    gate = chemo_gate(shares)
    # (c) Is the reversal just stage mix? Advanced disease gets chemotherapy
    #     and also carries worse performance status, so the marginal
    #     association has to be checked inside stage before it is believed.
    stratified = chemo_share_by_ecog(
        status, treatment, HORIZON_YEARS,
        strata=patients.set_index("PATIENT_ID")["STAGE_HIGHEST_RECORDED"])
    stratified.to_csv(TABLE_DIR / "chemo_share_by_ecog_and_stage.csv", index=False)
    by_stage = {}
    for stratum, block in stratified.groupby("stratum"):
        good = block[block["ecog"] < POOR_PERFORMANCE]
        poor = block[block["ecog"] >= POOR_PERFORMANCE]
        if not len(good) or not len(poor) or not poor["starts"].sum():
            continue
        by_stage[str(stratum)] = {
            "good_chemo_share": float(good["chemo"].sum() / good["starts"].sum()),
            "poor_chemo_share": float(poor["chemo"].sum() / poor["starts"].sum()),
            "poor_starts": int(poor["starts"].sum()),
        }
        by_stage[str(stratum)]["gap"] = (
            by_stage[str(stratum)]["good_chemo_share"]
            - by_stage[str(stratum)]["poor_chemo_share"])
    print("\n=== ECOG별 항암 비중 (전신치료 시작 기준) ===", flush=True)
    print(shares.to_string(index=False), flush=True)
    print(f"  ECOG 0-1 {gate['good_chemo_share']:.1%} "
          f"(n={gate['good_starts']}) 대 ECOG>={POOR_PERFORMANCE} "
          f"{gate['poor_chemo_share']:.1%} (n={gate['poor_starts']}) "
          f"· 격차 {gate['gap']:+.4f} · z {gate['z_unclustered']:.1f}", flush=True)

    # ---- 3. does it track disease? -----------------------------------------
    transitions = transition_rates(
        status, release.timeline("progression"), HORIZON_YEARS)
    transitions.to_csv(TABLE_DIR / "transition_rates.csv", index=False)
    rate_before = float(transitions.loc[
        transitions["period"] == "before_first_progression",
        "worsenings_per_year"].iloc[0])
    rate_after = float(transitions.loc[
        transitions["period"] == "after_first_progression",
        "worsenings_per_year"].iloc[0])
    ratio = rate_after / rate_before if rate_before > 0 else float("nan")
    print("\n=== 악화율 (진행 판정 전후) ===", flush=True)
    print(transitions.to_string(index=False), flush=True)
    print(f"  비율 {ratio:.2f}배", flush=True)

    verdict = {
        "cohort": cohort_size,
        "horizon_years": HORIZON_YEARS,
        "window_days": WINDOW_DAYS,
        "ecog_changes": int(len(changes)),
        "ecog_change_patients": int(changes["PATIENT_ID"].nunique()),
        "changes_on_treatment": int(len(table)),
        "stop_rate_worsened": float(worsened["stopped_after"]),
        "stop_rate_improved": float(improved["stopped_after"]),
        "stop_gap": stop_gap,
        "z_unclustered": z_unclustered,
        "clustered": clustered,
        "forward_minus_backward_worsened": float(worsened["forward_minus_backward"]),
        "forward_minus_backward_improved": float(improved["forward_minus_backward"]),
        "chemo_gate": gate,
        "chemo_gate_by_stage_posthoc": by_stage,
        "stop_gap_by_size_posthoc": {
            str(int(row.size)): float(row.gap) for row in by_size.itertuples()},
        "recent_start_share_posthoc": recent_start,
        "poor_performance_cut": POOR_PERFORMANCE,
        "worsenings_per_year_before": rate_before,
        "worsenings_per_year_after": rate_after,
        "worsening_ratio": ratio,
        "primary_prediction_met": bool(stop_gap >= STOP_GAP_FLOOR),
        "placebo_in_time_prediction_met": bool(
            worsened["forward_minus_backward"]
            > improved["forward_minus_backward"]),
        "chemo_gate_prediction_met": bool(gate["gap"] >= CHEMO_GATE_FLOOR),
        "state_dependence_prediction_met": bool(ratio >= WORSENING_RATIO_FLOOR),
    }

    metrics = {
        "run_date": RUN_DATE,
        "analysis_label": "ecog-state-v2.3",
        "question": (
            "Does recorded care act on performance status - by stopping "
            "treatment after it worsens, by withholding chemotherapy at poor "
            "levels - and does it move faster once disease progresses?"
        ),
        "estimand": (
            "None. Descriptive associations in the breast slice of MSK-CHORD "
            "between recorded ECOG changes and subsequent systemic-treatment "
            "status, between the ECOG in force and the channel started, and "
            "between first progression and the rate of recorded worsening."
        ),
        "scope_warning": (
            "Descriptive only and confounded by indication in both directions: "
            "treatment stops because patients deteriorate, and patients "
            "deteriorate partly because of treatment. These say recorded care "
            "and performance status move together, not that either causes the "
            "other. The release records ECOG change points rather than "
            "measurements (v2.2), so counts are of documented changes."
        ),
        "licence": (
            "CC BY-NC-ND 4.0. Aggregates and per-file SHA-256 only; "
            "patient-level intermediates stay under data/ (git-ignored). "
            "Cite Jee et al., Nature 2024 (PMID 39506116)."
        ),
        "prespecified_prediction": PRESPECIFIED_PREDICTION,
        "design": {
            "release": "MSK-CHORD (MSK, Nature 2024), cBioPortal msk_chord_2024",
            "cancer_type": "Breast Cancer",
            "horizon_years": HORIZON_YEARS,
            "window_days": WINDOW_DAYS,
            "window_sensitivity": list(WINDOW_SENSITIVITY),
            "poor_performance_cut": POOR_PERFORMANCE,
            "treatment_definition": (
                "merged spans of Chemo/Hormone/Targeted/Biologic/Immuno; "
                "supportive and investigational excluded"
            ),
        },
        "verdict": verdict,
    }
    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    manifest = {
        "run_date": RUN_DATE,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit_before_run": git_commit(),
        "inputs": release.inputs,
        "data_dir": str(DATA_DIR.relative_to(ROOT)).replace("\\", "/"),
        "derived_dir": str(DERIVED_DIR.relative_to(ROOT)).replace("\\", "/"),
        "entry_point": "analysis/51_run_ecog_state.py",
        "base_seed": None,
        "seed_note": "no randomness - every number here is a count or a rate",
    }
    (REPORT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    print("\n=== 사전 예측 채점 ===", flush=True)
    for key, met in (
        ("1 악화 뒤 치료 중단 (주)", verdict["primary_prediction_met"]),
        ("2 시간 위약대조", verdict["placebo_in_time_prediction_met"]),
        ("3 ECOG가 항암을 가름", verdict["chemo_gate_prediction_met"]),
        ("4 상태 의존 (진행 후 악화 가속)", verdict["state_dependence_prediction_met"]),
    ):
        print(f"  {key}: {'통과' if met else '빗나감'}", flush=True)


if __name__ == "__main__":
    main()
