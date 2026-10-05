"""Does real care contain the adaptation our simulator has none of? (v2.2)

The founding analogy is chess: `지금 보드 -> 최선의 수 -> 결과` is the same object
as `지금 환자 상태 -> 최선의 치료 -> 결과`. v1.6 asked whether a real dataset
carries the *movetext* METABRIC lacked, and GENIE BPC answered yes - but only
for two of our five action axes, both systemic, with no surgery, no radiation
and no performance status.

v1.7 then found the deeper problem, and it was ours rather than the data's.
Blinding the response channel from the planner moved the utility gap by
-0.0004 (z = -0.29). The reason was arithmetic: our environment offers a
response signal **once**, in 20.8% of episodes. With one branch point a
sequential method is a static optimiser with extra steps, so v1.7 could not
distinguish "MCTS fails to adapt" from "there is nothing here to adapt to".
That limitation has stood for four versions, and no simulator change can
settle it - the question is how many adaptation opportunities *real* care has.

MSK-CHORD (MSK, Nature 2024) can answer it. Every CT, PET and MRI in the
release carries an NLP-derived progression call with a date; performance status
is recorded repeatedly; surgery and radiation have their own timelines. This
run asks three questions, in order of how much they change what we do next:

    1. **How many adaptation opportunities does a real patient's record
       contain**, against our environment's one?
    2. **Does treatment actually change after a progression call**, over and
       above the patient's own baseline switching rate?
    3. **Which of our five action axes can this release see**, and at what
       resolution?

As in v1.6 this is a data-adequacy assessment and deliberately descriptive.
Nothing is adjusted, weighted or causal. Reading a confounded switch rate as
an effect would repeat the mistake v1.3's negative control caught.

SCOPING DISCLOSURE (read before the predictions)
------------------------------------------------
A scoping pass over this release preceded the predictions below, and it saw:
cohort size, subtype and stage mix, the ``SUBTYPE`` counts and top-25 agents in
the treatment timeline, raw row counts and rows-per-patient medians for each
timeline, the marginal ECOG and ``PROGRESSION`` distributions, and the median
diagnosis-to-sequencing interval. Prediction 1 is therefore marked
**scoping-informed**: a raw rows-per-patient median was visible, though the
de-duplicated, diagnosis-anchored, horizon-limited count it predicts was not.
Predictions 2-6 could not be read off anything the scoping pass showed.

PRE-SPECIFIED PREDICTIONS, recorded before the run
--------------------------------------------------
1. **Adaptation opportunities** (*scoping-informed*) - the median number of
   de-duplicated progression calls within 5 years of diagnosis is **>= 5**,
   and **>= 60%** of the cohort has **>= 2**. Our environment offers one, in
   20.8% of episodes. If real care has many, the environment under-models
   adaptation and v1.7's null is a statement about our simulator, not about
   MCTS.
2. **Primary - treatment follows the signal** - a progressing call is followed
   by a systemic treatment start within 90 days at **>= 1.5x** the rate a
   controlled call is. The floor is set below GENIE BPC's 2.4x because the
   switch definition differs: MSK-CHORD records per-agent starts, so
   continuation rows inflate both arms.
3. **Placebo in time** - for progressing calls the forward-minus-backward
   switch-rate gap exceeds the same gap for controlled calls. A scan cannot
   cause a treatment that started before it, so a symmetric result would mean
   prediction 2 is reading "this patient switches a lot".
4. **Known-answer check on the label ordering** - the ``Indeterminate`` group's
   forward switch rate falls **between** the controlled and progressing rates.
   The three labels are an ordered signal or they are not usable as a response
   channel at all; this is the cheapest way to find out.
5. **Performance status is a state, not a covariate** - among patients with
   >= 2 ECOG measurements inside the horizon, **>= 50%** record **>= 2
   distinct** values. Our schema has no ECOG field. If ECOG does not move
   within patients, nothing is lost; if it moves, the state vector is missing
   a variable clinicians act on.
6. **The timing axis is readable** - the share of patients with both a surgery
   and a chemotherapy date whose chemotherapy came first is between **15% and
   45%**. This axis was wholly unobservable in GENIE BPC. A share outside that
   band would mean the surgery timeline is not recording what we think.

Prediction 4 is the one most likely to fail, and the most informative if it
does: the ``PROGRESSION`` field is NLP output, and an unordered label would
quietly invalidate any later use of it as a response channel.
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
from analysis.msk.adaptation import (  # noqa: E402
    DAYS_PER_MONTH,
    decision_points,
    ecog_encoding_check,
    ecog_trajectory,
    followup_days,
    opportunity_summary,
    switch_rates,
    switch_table,
)
from analysis.msk.axes import (  # noqa: E402
    AXIS_RESOLUTION,
    DAYS_PER_YEAR,
    NEOADJUVANT_INTERVAL_DAYS,
    axis_coverage,
    chemo_agent_counts,
    endocrine_duration,
    timing_intervals,
    timing_table,
)
from analysis.msk.loader import DATA_DIR, load  # noqa: E402

REPORT_DIR = ROOT / "reports" / "msk-chord-profile-v2.2"
TABLE_DIR = REPORT_DIR / "tables"

#: Patient-level intermediates go here, **not** into ``tables/``.
#:
#: MSK-CHORD is CC BY-NC-ND 4.0 - NonCommercial and **NoDerivatives** - so a
#: per-patient derived table carrying real MSK identifiers may not be
#: redistributed any more than the release itself may. ``tables/`` is committed
#: and published; this directory sits under ``data/``, which .gitignore
#: excludes. Everything in ``tables/`` is an aggregate.
DERIVED_DIR = ROOT / "data" / "external" / "msk_chord_2024_derived"

RUN_DATE = "2026-10-05"
HORIZON_YEARS = 5.0            # the simulator's horizon since v0.2
SWITCH_WINDOW_DAYS = 90        # v1.6's window, kept identical for comparability
DEDUP_DAYS = 30

#: What the environment offers, for the comparison prediction 1 is about.
ENVIRONMENT_OPPORTUNITIES = 1
ENVIRONMENT_OPPORTUNITY_SHARE = 0.208   # v1.7, reports/closed-loop-value-v1.7

OPPORTUNITY_MEDIAN_FLOOR = 5
OPPORTUNITY_TWO_SHARE_FLOOR = 0.60
SWITCH_RATIO_FLOOR = 1.5
ECOG_MOVING_SHARE_FLOOR = 0.50
NEOADJUVANT_BAND = (0.15, 0.45)

#: Design choices swept so the headline is not an artefact of one of them.
DEDUP_SENSITIVITY = (0, 30, 90)
WINDOW_SENSITIVITY = (60, 90, 180)

PRESPECIFIED_PREDICTION = {
    "adaptation_opportunities": (
        f"Median de-duplicated progression calls within {HORIZON_YEARS:.0f} "
        f"years of diagnosis >= {OPPORTUNITY_MEDIAN_FLOOR}, and >= "
        f"{OPPORTUNITY_TWO_SHARE_FLOOR:.0%} of the cohort has >= 2. "
        "SCOPING-INFORMED: a raw rows-per-patient median was visible before "
        "this was written."
    ),
    "primary_treatment_follows_signal": (
        f"A progressing call is followed by a systemic treatment start within "
        f"{SWITCH_WINDOW_DAYS} days at >= {SWITCH_RATIO_FLOOR}x the rate a "
        "controlled call is."
    ),
    "placebo_in_time": (
        "For progressing calls the forward-minus-backward switch-rate gap "
        "exceeds the same gap for controlled calls."
    ),
    "label_ordering_known_answer": (
        "The Indeterminate group's forward switch rate falls between the "
        "controlled and progressing rates."
    ),
    "ecog_is_a_state": (
        f"Among patients with >= 2 ECOG measurements inside the horizon, >= "
        f"{ECOG_MOVING_SHARE_FLOOR:.0%} record >= 2 distinct values."
    ),
    "timing_axis_readable": (
        f"Among patients with both a surgery and a chemotherapy date, the "
        f"neoadjuvant share lies in "
        f"[{NEOADJUVANT_BAND[0]:.0%}, {NEOADJUVANT_BAND[1]:.0%}]."
    ),
    "why": (
        "v1.7 blinded the response channel and the gap moved -0.0004 "
        "(z = -0.29) because our environment offers one response signal in "
        "20.8% of episodes. That limitation has stood since v1.7 and no "
        "simulator change can settle it: the question is how much adaptation "
        "real care contains."
    ),
}


def cohort_overview(patients: pd.DataFrame) -> pd.DataFrame:
    """Stage x receptor counts. The sampling frame for every later number."""
    frame = patients.copy()
    frame["receptor"] = [
        f"HR{'+' if hr == 'Yes' else '-'}/HER2{'+' if her2 == 'Yes' else '-'}"
        if hr in ("Yes", "No") and her2 in ("Yes", "No") else "(unrecorded)"
        for hr, her2 in zip(frame["HR"], frame["HER2"])
    ]
    counts = (frame.groupby(["STAGE_HIGHEST_RECORDED", "receptor"], dropna=False)
              .size().rename("patients").reset_index())
    return counts.sort_values(["STAGE_HIGHEST_RECORDED", "receptor"])


def truncation_table(anchor: pd.Series, patients: pd.DataFrame) -> pd.DataFrame:
    """How long each patient had to survive to enter the cohort.

    The release's central selection. Sequencing is day 0 and diagnosis is at
    ``anchor`` days (negative), so ``-anchor`` is guaranteed survival before
    the patient could be observed at all. Reported as a table because a naive
    Kaplan-Meier from diagnosis on this cohort would be wrong by exactly this
    much, and v1.9 taught us to write the scale problem down next to the
    number rather than at the end of the document.
    """
    waited = (-anchor).rename("days_dx_to_sequencing")
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    rows = [{
        "statistic": f"p{int(q * 100)}",
        "days": float(waited.quantile(q)),
        "years": float(waited.quantile(q) / DAYS_PER_YEAR),
    } for q in quantiles]
    rows.append({"statistic": "mean", "days": float(waited.mean()),
                 "years": float(waited.mean() / DAYS_PER_YEAR)})
    rows.append({"statistic": "share_over_1y",
                 "days": float((waited > DAYS_PER_YEAR).mean()),
                 "years": float("nan")})
    # The release spells this field out rather than using Yes/No, and a naive
    # ``== "Yes"`` silently reports 0% prior treatment - which would have made
    # the cohort look far cleaner than it is.
    prior = patients["PRIOR_MED_TO_MSK"].astype(str).str.strip()
    for label, value in (
        ("share_prior_treatment_elsewhere", "Prior medications to MSK"),
        ("share_no_prior_treatment", "No prior medications"),
        ("share_prior_treatment_unknown", "Unknown"),
    ):
        rows.append({"statistic": label,
                     "days": float((prior == value).mean()),
                     "years": float("nan")})
    return pd.DataFrame(rows)


def two_proportion_z(p_a: float, n_a: int, p_b: float, n_b: int) -> tuple[float, float]:
    """Unclustered two-proportion z and its pooled standard error."""
    if min(n_a, n_b) == 0:
        return float("nan"), float("nan")
    pooled = (p_a * n_a + p_b * n_b) / (n_a + n_b)
    standard_error = float(np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b)))
    if standard_error == 0:
        return float("nan"), standard_error
    return float((p_a - p_b) / standard_error), standard_error


def clustered_contrast(table: pd.DataFrame, column: str = "switch_after") -> dict:
    """Patient-clustered contrast between progressing and controlled calls.

    Calls are nested within patients, so the unclustered z that v1.6 reported
    overstates precision. This averages within patient first and contrasts the
    patient-level means, which is the comparison the sentence "treatment
    changes after a progression call" actually makes. Patients contributing
    both kinds of call appear in both arms, so the two arms are not
    independent; the standard error is therefore still approximate, and it is
    the *larger* of the two estimates, which is the direction to err in.
    """
    per_patient = (table.groupby(["PATIENT_ID", "group"])[column]
                   .mean().rename("rate").reset_index())
    wide = per_patient.pivot(index="PATIENT_ID", columns="group", values="rate")
    out: dict[str, float] = {}
    for group in ("progressing", "controlled", "indeterminate"):
        if group in wide:
            values = wide[group].dropna()
            out[f"{group}_patients"] = int(len(values))
            out[f"{group}_mean"] = float(values.mean())
            out[f"{group}_se"] = float(values.std(ddof=1) / np.sqrt(len(values))) \
                if len(values) > 1 else float("nan")
    if {"progressing", "controlled"} <= set(wide.columns):
        a, b = wide["progressing"].dropna(), wide["controlled"].dropna()
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


def dedup_sensitivity(progression: pd.DataFrame, cohort_size: int) -> pd.DataFrame:
    """Opportunity count at each de-duplication window."""
    rows = []
    for days in DEDUP_SENSITIVITY:
        points = decision_points(progression, HORIZON_YEARS, days)
        summary = opportunity_summary(points, cohort_size)
        rows.append({
            "dedup_days": days,
            "patients": summary["patients"],
            "median": summary["median"],
            "p90": summary["p90"],
            "at_least_two": summary["at_least_two"],
            "at_least_five": summary["at_least_five"],
        })
    return pd.DataFrame(rows)


def opportunity_by_entry(points: pd.DataFrame, anchor: pd.Series,
                         patients: pd.DataFrame) -> pd.DataFrame:
    """Opportunity count split by how late the patient entered the record.

    POST-HOC, and the most important caveat on the headline. Sequencing is when
    a patient becomes visible to MSK's record, and the cohort's p75 diagnosis-
    to-sequencing interval is 1,905 days. For a quarter of these patients the
    whole five-year window we are counting in closed *before* they entered,
    so their opportunity count is near zero for an administrative reason
    rather than a clinical one.

    Splitting by entry therefore says whether the headline median understates
    or overstates adaptation in care that was actually observed.
    """
    waited = (-anchor).rename("days")
    edges = [-float("inf"), 90, 365, 1095, float("inf")]
    labels = ["<= 90d", "91-365d", "1-3y", "> 3y"]
    bucket = pd.cut(waited, bins=edges, labels=labels)
    frame = pd.DataFrame({"bucket": bucket})
    joined = frame.join(
        points.set_index("PATIENT_ID")["opportunities"], how="left")
    joined["opportunities"] = joined["opportunities"].fillna(0)
    rows = []
    for label in labels:
        block = joined[joined["bucket"] == label]
        if block.empty:
            continue
        rows.append({
            "entry_bucket": label,
            "patients": int(len(block)),
            "share_of_cohort": float(len(block) / len(patients)),
            "median": float(block["opportunities"].median()),
            "p75": float(block["opportunities"].quantile(0.75)),
            "p90": float(block["opportunities"].quantile(0.90)),
            "at_least_two": float((block["opportunities"] >= 2).mean()),
            "zero": float((block["opportunities"] == 0).mean()),
        })
    return pd.DataFrame(rows)


def genie_cross_check() -> pd.DataFrame | None:
    """The same opportunity count, run on GENIE BPC - which we have had since v1.6.

    Added **after** the first run of this report, in response to a direct
    question: was this release necessary to reach the headline?

    The honest answer is no. GENIE BPC carries an imaging table with a
    radiologist assessment and a diagnosis-relative day on every scan, so the
    adaptation-opportunity count was computable from data already in the
    repository six weeks earlier. v1.6 used that table for switch rates and
    never counted opportunities per patient. The gap was in the question we
    asked, not in the data we held.

    Reporting it changes what this result *is*: not a finding that required a
    new release, but a finding **replicated in two independent cohorts** -
    different institutions, different curation, different assessment
    vocabularies, same median. That is the stronger claim, and it is only
    available because the check was run.

    Returns ``None`` if the GENIE release is not present, so this report still
    runs on a machine that holds only one of the two.
    """
    try:
        from analysis.genie.loader import load as load_genie
        release = load_genie()
    except (FileNotFoundError, ImportError):
        return None

    limit = HORIZON_YEARS * DAYS_PER_YEAR
    scans = release.imaging.dropna(subset=["dx_scan_days", "image_overall"])
    scans = scans[(scans["dx_scan_days"] >= 0) & (scans["dx_scan_days"] <= limit)]

    counts = []
    for _, block in scans.groupby("record_id"):
        kept: list[float] = []
        for day in sorted(float(value) for value in block["dx_scan_days"]):
            if kept and day - kept[-1] < DEDUP_DAYS:
                continue
            kept.append(day)
        counts.append(len(kept))
    if not counts:
        return None

    series = pd.Series(counts)
    cohort = int(release.cancers["record_id"].nunique())
    return pd.DataFrame([{
        "release": "GENIE BPC Breast v1.0-public",
        "cohort": cohort,
        "patients_with_an_opportunity": int(len(series)),
        "share_of_cohort": float(len(series) / cohort),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "p90": float(series.quantile(0.90)),
        "max": int(series.max()),
        "at_least_two": float((series >= 2).mean()),
        "at_least_five": float((series >= 5).mean()),
    }])


def window_sensitivity(progression: pd.DataFrame, treatment: pd.DataFrame,
                       followup: pd.Series) -> pd.DataFrame:
    """Switch-rate contrast at each forward/backward window length."""
    rows = []
    for days in WINDOW_SENSITIVITY:
        table = switch_table(progression, treatment, followup, days, HORIZON_YEARS)
        rates = switch_rates(table).set_index("group")
        if not {"progressing", "controlled"} <= set(rates.index):
            continue
        progressing, controlled = rates.loc["progressing"], rates.loc["controlled"]
        rows.append({
            "window_days": days,
            "calls": int(len(table)),
            "switch_progressing": float(progressing["switch_after"]),
            "switch_controlled": float(controlled["switch_after"]),
            "ratio": float(progressing["switch_after"] / controlled["switch_after"]),
            "placebo_progressing": float(progressing["forward_minus_backward"]),
            "placebo_controlled": float(controlled["forward_minus_backward"]),
        })
    return pd.DataFrame(rows)


def distribution(values: pd.Series, name: str) -> pd.DataFrame:
    """Value counts as a committable aggregate - no identifiers, no rows."""
    counts = values.value_counts().sort_index()
    frame = counts.reset_index()
    frame.columns = [name, "patients"]
    frame["share"] = frame["patients"] / counts.sum()
    return frame


def quantile_summary(values: pd.Series, name: str) -> pd.DataFrame:
    rows = [{"statistic": "n", name: float(len(values))}]
    for quantile in (0.1, 0.25, 0.5, 0.75, 0.9):
        rows.append({"statistic": f"p{int(quantile * 100)}",
                     name: float(values.quantile(quantile))})
    rows.append({"statistic": "mean", name: float(values.mean())})
    rows.append({"statistic": "max", name: float(values.max())})
    return pd.DataFrame(rows)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    release = load()
    patients = release.patients
    cohort = set(patients["PATIENT_ID"])
    cohort_size = len(cohort)

    print(f"유방암 환자 {cohort_size}명 · 진단일 확인 {len(release.anchor)}명",
          flush=True)

    overview = cohort_overview(patients)
    overview.to_csv(TABLE_DIR / "cohort_overview.csv", index=False)
    print("\n=== 코호트 ===", flush=True)
    print(overview.to_string(index=False), flush=True)

    truncation = truncation_table(release.anchor, patients)
    truncation.to_csv(TABLE_DIR / "left_truncation.csv", index=False)
    waited_median = float(truncation.loc[
        truncation["statistic"] == "p50", "days"].iloc[0])
    prior_share = float(truncation.loc[
        truncation["statistic"] == "share_prior_treatment_elsewhere",
        "days"].iloc[0])
    print(f"\n진단->시퀀싱 중앙값 {waited_median:.0f}일 "
          f"· MSK 외부 선행치료 {prior_share:.1%}", flush=True)

    # ---- 1. action-axis coverage -------------------------------------------
    coverage = axis_coverage(
        release.timeline("treatment"), release.timeline("surgery"),
        release.timeline("radiation"), cohort, HORIZON_YEARS)
    coverage.to_csv(TABLE_DIR / "action_space_coverage.csv", index=False)
    print("\n=== 행동축 관측 가능성 ===", flush=True)
    print(coverage[["axis", "patients", "share", "level_resolution"]]
          .to_string(index=False), flush=True)

    timing = timing_table(release.timeline("treatment"),
                          release.timeline("surgery"), HORIZON_YEARS)
    timing.to_csv(DERIVED_DIR / "timing.csv", index=False)
    timing_counts = (timing.groupby("timing").size().rename("patients")
                     .reset_index())
    timing_counts["share"] = timing_counts["patients"] / len(timing)
    timing_counts.to_csv(TABLE_DIR / "timing.csv", index=False)
    intervals = timing_intervals(timing)
    intervals.to_csv(TABLE_DIR / "timing_intervals.csv", index=False)
    neoadjuvant_share = float(
        timing_counts.loc[timing_counts["timing"] == "neoadjuvant",
                          "share"].sum())
    plausible_share = float(timing["plausible_neoadjuvant"].mean())
    print(f"\n선행치료 비율 {neoadjuvant_share:.1%} (n={len(timing)}) "
          f"· 간격 {NEOADJUVANT_INTERVAL_DAYS[0]}-{NEOADJUVANT_INTERVAL_DAYS[1]}일 "
          f"제한(사후) {plausible_share:.1%}", flush=True)
    print(intervals.to_string(index=False), flush=True)

    endocrine = endocrine_duration(release.timeline("treatment"))
    endocrine.to_csv(DERIVED_DIR / "endocrine_duration.csv", index=False)
    quantile_summary(endocrine["years"], "years").to_csv(
        TABLE_DIR / "endocrine_duration_summary.csv", index=False)
    chemo = chemo_agent_counts(release.timeline("treatment"))
    chemo.to_csv(DERIVED_DIR / "chemo_agents.csv", index=False)
    distribution(chemo["agents"], "first_course_agents").to_csv(
        TABLE_DIR / "chemo_agent_distribution.csv", index=False)
    print(f"내분비 기간 중앙값 {endocrine['years'].median():.2f}년 "
          f"· 5년 이상 {endocrine['extended'].mean():.1%} (n={len(endocrine)})",
          flush=True)
    print(f"1차 항암 약제 수 중앙값 {chemo['agents'].median():.0f} "
          f"(n={len(chemo)})", flush=True)

    # ---- 2. adaptation opportunities ---------------------------------------
    points = decision_points(
        release.timeline("progression"), HORIZON_YEARS, DEDUP_DAYS)
    points.to_csv(DERIVED_DIR / "decision_points.csv", index=False)
    distribution(points["opportunities"], "opportunities").to_csv(
        TABLE_DIR / "decision_point_distribution.csv", index=False)
    opportunities = opportunity_summary(points, cohort_size)
    sensitivity = dedup_sensitivity(release.timeline("progression"), cohort_size)
    sensitivity.to_csv(TABLE_DIR / "dedup_sensitivity.csv", index=False)
    print("\n=== 적응 기회 ===", flush=True)
    print(f"  중앙값 {opportunities['median']:.0f} · p90 "
          f"{opportunities['p90']:.0f} · 최대 {opportunities['max']} "
          f"· 2회 이상 {opportunities['at_least_two']:.1%} "
          f"· 코호트 포함 {opportunities['share_of_cohort']:.1%}", flush=True)
    print(f"  환경: {ENVIRONMENT_OPPORTUNITIES}회, 에피소드의 "
          f"{ENVIRONMENT_OPPORTUNITY_SHARE:.1%}", flush=True)
    print(sensitivity.to_string(index=False), flush=True)

    cross = genie_cross_check()
    if cross is not None:
        cross.to_csv(TABLE_DIR / "genie_cross_check.csv", index=False)
        row = cross.iloc[0]
        print("")
        print(f"  (교차 확인) GENIE BPC 같은 규칙: 중앙값 {row['median']:.0f} "
              f"· 2회 이상 {row['at_least_two']:.1%} "
              f"· 코호트 포함 {row['share_of_cohort']:.1%} (n={row['cohort']})",
              flush=True)
        print("  -> 이 헤드라인은 v1.6부터 보유한 자료로도 나왔다. "
              "새 자료의 기여는 '발견'이 아니라 '독립 재현'이다.", flush=True)

    entry = opportunity_by_entry(points, release.anchor, patients)
    entry.to_csv(TABLE_DIR / "opportunity_by_entry.csv", index=False)
    print("\n  (사후) 기록 진입 시점별 - 전체를 0으로 채운 분모", flush=True)
    print(entry.to_string(index=False), flush=True)
    # The apples-to-apples comparison with the environment: patients whose
    # record starts within 90 days of diagnosis, so the five-year window we
    # count in is a window we could actually observe. A weighted mean of
    # bucket medians would not be a median, so the bucket is read directly.
    near = entry[entry["entry_bucket"] == "<= 90d"]
    observed_median = float(near["median"].iloc[0]) if len(near) else float("nan")
    observed_two = float(near["at_least_two"].iloc[0]) if len(near) else float("nan")
    observed_patients = int(near["patients"].iloc[0]) if len(near) else 0
    print(f"  -> 진단 90일 내 진입군 (n={observed_patients}): 중앙값 "
          f"{observed_median:.0f} · 2회 이상 {observed_two:.1%} "
          f"(환경 1회 · {ENVIRONMENT_OPPORTUNITY_SHARE:.1%})", flush=True)

    # ---- 3. does treatment follow the signal? ------------------------------
    followup = followup_days(patients, release.anchor)
    table = switch_table(
        release.timeline("progression"), release.timeline("treatment"),
        followup, SWITCH_WINDOW_DAYS, HORIZON_YEARS)
    rates = switch_rates(table)
    rates.to_csv(TABLE_DIR / "response_switch.csv", index=False)
    windows = window_sensitivity(
        release.timeline("progression"), release.timeline("treatment"), followup)
    windows.to_csv(TABLE_DIR / "window_sensitivity.csv", index=False)
    print(f"\n=== 판독 {len(table)}건 ({SWITCH_WINDOW_DAYS}일 창) ===", flush=True)
    print(rates.to_string(index=False), flush=True)

    indexed = rates.set_index("group")
    progressing, controlled = indexed.loc["progressing"], indexed.loc["controlled"]
    ratio = float(progressing["switch_after"] / controlled["switch_after"])
    z_unclustered, _ = two_proportion_z(
        float(progressing["switch_after"]), int(progressing["calls"]),
        float(controlled["switch_after"]), int(controlled["calls"]))
    clustered = clustered_contrast(table)
    print(f"  비율 {ratio:.2f}x · 비군집 z {z_unclustered:.1f} "
          f"· 환자군집 z {clustered.get('z', float('nan')):.1f} "
          f"(MDE {clustered.get('minimum_detectable_difference', float('nan')):.4f})",
          flush=True)

    indeterminate_rate = (float(indexed.loc["indeterminate", "switch_after"])
                          if "indeterminate" in indexed.index else float("nan"))
    ordered = bool(
        float(controlled["switch_after"]) < indeterminate_rate
        < float(progressing["switch_after"]))

    # ---- 4. is performance status a state? ---------------------------------
    # The encoding check comes first on purpose. Prediction 5 asks whether ECOG
    # moves within patients, and the answer is 100.0% - because the release
    # records change points, not measurements. The check is what turns that
    # from a finding into a known artefact.
    encoding = ecog_encoding_check(release.timeline("performance_status"))
    ecog = ecog_trajectory(release.timeline("performance_status"), HORIZON_YEARS)
    ecog.to_csv(DERIVED_DIR / "ecog_trajectory.csv", index=False)
    distribution(ecog["first"], "first_ecog").to_csv(
        TABLE_DIR / "ecog_first_value.csv", index=False)
    distribution(ecog["measurements"], "rows_recorded").to_csv(
        TABLE_DIR / "ecog_row_counts.csv", index=False)
    repeated = ecog[ecog["measurements"] >= 2]
    moving_share = float((repeated["distinct"] >= 2).mean()) if len(repeated) else 0.0
    # Post-hoc replacement for the vacuous prediction: the share of the whole
    # cohort whose performance status is *known* to have moved.
    documented_change_share = float(len(repeated) / cohort_size)
    print("\n=== ECOG ===", flush=True)
    print(f"  인코딩 검사: 연속 쌍 {encoding['consecutive_pairs']}개 중 같은 값 "
          f"{encoding['equal_consecutive_pairs']}개 "
          f"(같은 날 쌍 {encoding['same_day_pairs']}개 중 "
          f"{encoding['equal_same_day_pairs']}개) "
          f"-> 변화점만 기록: {encoding['change_point_encoded']}", flush=True)
    print(f"  기록 {len(ecog)}명 ({len(ecog) / cohort_size:.1%}) · 변화 문서화 "
          f"{len(repeated)}명 (코호트의 {documented_change_share:.1%}) "
          f"· 악화 경험 {repeated['ever_worsened'].mean():.1%}", flush=True)
    if encoding["change_point_encoded"]:
        print(f"  주의: '값이 변한 비율 {moving_share:.1%}'는 구조상 1이다 - "
              "예측 5는 실패할 수 없었으므로 근거가 아니다.", flush=True)

    # Post-hoc diagnostics for prediction 6, in the order they were tried.
    #
    # Hypothesis A, REJECTED: a sequenced cohort with 1,099 stage IV patients
    # records systemic therapy before operations that were never adjuvant, so
    # the stage mix inflates the share. Restricting to stage 1-3 moves it by
    # under a point, so stage mix explains almost none of the miss. Kept in the
    # output because CLAUDE.md requires rejected hypotheses to be recorded.
    #
    # Hypothesis B, SUPPORTED: the surgery timeline's PROCEDURE rows are every
    # operation a patient ever had, not the definitive breast operation, so
    # "chemotherapy before the first recorded operation" is a wider set than a
    # neoadjuvant course. See NEOADJUVANT_INTERVAL_DAYS.
    early = set(patients.loc[
        patients["STAGE_HIGHEST_RECORDED"] == "Stage 1-3", "PATIENT_ID"])
    early_timing = timing[timing["PATIENT_ID"].isin(early)]
    neoadjuvant_share_early = float(
        (early_timing["timing"] == "neoadjuvant").mean()) if len(early_timing) else float("nan")
    print(f"\n  (사후 A, 기각) Stage 1-3만: 선행치료 {neoadjuvant_share_early:.1%} "
          f"(n={len(early_timing)}) - 전체 {neoadjuvant_share:.1%}와 거의 같다",
          flush=True)
    print(f"  (사후 B, 지지) 간격 제한: {plausible_share:.1%} - 사전 구간 "
          f"[{NEOADJUVANT_BAND[0]:.0%}, {NEOADJUVANT_BAND[1]:.0%}] 안으로 들어온다",
          flush=True)

    # ---- verdict ------------------------------------------------------------
    verdict = {
        "cohort": cohort_size,
        "anchored": int(len(release.anchor)),
        "horizon_years": HORIZON_YEARS,
        "days_dx_to_sequencing_median": waited_median,
        "share_prior_treatment_elsewhere": prior_share,
        "action_axes_total": int(len(coverage)),
        "action_axes_with_occurrence": int(
            sum(1 for axis in AXIS_RESOLUTION.values() if axis["occurrence"])),
        "action_axes_with_level": int(
            sum(1 for axis in AXIS_RESOLUTION.values() if axis["level"] is True)),
        "action_axes_with_partial_level": int(
            sum(1 for axis in AXIS_RESOLUTION.values()
                if axis["level"] == "partial")),
        "axis_coverage_share": {
            row.axis: float(row.share) for row in coverage.itertuples()},
        "neoadjuvant_share": neoadjuvant_share,
        "neoadjuvant_share_interval_restricted_posthoc": plausible_share,
        "neoadjuvant_interval_days": list(NEOADJUVANT_INTERVAL_DAYS),
        "neoadjuvant_band": list(NEOADJUVANT_BAND),
        "timing_patients": int(len(timing)),
        "endocrine_years_median": float(endocrine["years"].median()),
        "endocrine_extended_share": float(endocrine["extended"].mean()),
        "chemo_agents_median": float(chemo["agents"].median()),
        "opportunities": opportunities,
        "opportunity_entry_within_90d_posthoc": {
            "patients": observed_patients,
            "median": observed_median,
            "at_least_two": observed_two,
        },
        "genie_cross_check": (
            cross.iloc[0].to_dict() if cross is not None else None),
        "headline_was_reachable_from_genie_bpc": bool(cross is not None),
        "environment_opportunities": ENVIRONMENT_OPPORTUNITIES,
        "environment_opportunity_share": ENVIRONMENT_OPPORTUNITY_SHARE,
        "dedup_days": DEDUP_DAYS,
        "switch_window_days": SWITCH_WINDOW_DAYS,
        "calls_scored": int(len(table)),
        "switch_rate_progressing": float(progressing["switch_after"]),
        "switch_rate_controlled": float(controlled["switch_after"]),
        "switch_rate_indeterminate": indeterminate_rate,
        "switch_rate_ratio": ratio,
        "new_agent_progressing": float(progressing["new_agent_after"]),
        "new_agent_controlled": float(controlled["new_agent_after"]),
        "z_unclustered": z_unclustered,
        "clustered": clustered,
        "forward_minus_backward_progressing": float(
            progressing["forward_minus_backward"]),
        "forward_minus_backward_controlled": float(
            controlled["forward_minus_backward"]),
        "ecog_patients": int(len(ecog)),
        "ecog_share_of_cohort": float(len(ecog) / cohort_size),
        "ecog_repeated_patients": int(len(repeated)),
        "ecog_moving_share": moving_share,
        "ecog_documented_change_share": documented_change_share,
        "ecog_ever_worsened_share": float(repeated["ever_worsened"].mean())
            if len(repeated) else float("nan"),
        "ecog_encoding": encoding,
        "ecog_first_value_counts": {
            str(value): int(count) for value, count
            in ecog["first"].value_counts().sort_index().items()},
        "neoadjuvant_share_stage_1_3_posthoc": neoadjuvant_share_early,
        "opportunity_prediction_met": bool(
            opportunities["median"] >= OPPORTUNITY_MEDIAN_FLOOR
            and opportunities["at_least_two"] >= OPPORTUNITY_TWO_SHARE_FLOOR),
        "primary_prediction_met": bool(ratio >= SWITCH_RATIO_FLOOR),
        "placebo_in_time_prediction_met": bool(
            progressing["forward_minus_backward"]
            > controlled["forward_minus_backward"]),
        "label_ordering_prediction_met": ordered,
        # Scored as VACUOUS, not passed. The encoding check shows the test
        # could not fail, so the nominal pass carries no evidence. CLAUDE.md:
        # "부호만 맞은 결과를 '통과'로 적지 않는다" - and this is weaker than
        # sign-only, because the comparison had no power at all.
        "ecog_is_a_state_prediction_met": None,
        "ecog_is_a_state_prediction_vacuous": bool(
            encoding["change_point_encoded"]),
        "ecog_is_a_state_nominal_value": moving_share,
        "ecog_moving_share_floor": ECOG_MOVING_SHARE_FLOOR,
        "timing_axis_prediction_met": bool(
            NEOADJUVANT_BAND[0] <= neoadjuvant_share <= NEOADJUVANT_BAND[1]),
    }

    metrics = {
        "run_date": RUN_DATE,
        "analysis_label": "msk-chord-profile-v2.2",
        "question": (
            "How many adaptation opportunities does real breast-cancer care "
            "contain, does treatment change after a progression call, and "
            "which of the environment's five action axes can MSK-CHORD see?"
        ),
        "estimand": (
            "None. This is a descriptive data-adequacy assessment of the "
            "breast-cancer slice of MSK-CHORD: counts of de-duplicated "
            "progression calls per patient, the unadjusted association "
            "between a progression call and a subsequent systemic treatment "
            "start, within-patient ECOG movement, and coverage of the "
            "environment's action axes."
        ),
        "scope_warning": (
            "Descriptive only. Switch rates are unadjusted and confounded by "
            "indication; they say recorded care contains adaptation, not that "
            "adaptation caused anything. The cohort is conditioned on "
            "surviving from diagnosis to sequencing, so OS on this cohort "
            "must not be fed to Kaplan-Meier without left-truncation handling."
        ),
        "licence": (
            "CC BY-NC-ND 4.0. NonCommercial and NoDerivatives: neither raw "
            "nor patient-level derived tables may be redistributed. This "
            "report carries aggregate counts and per-file SHA-256 only. Cite "
            "Jee et al., Nature 2024 (PMID 39506116)."
        ),
        "prespecified_prediction": PRESPECIFIED_PREDICTION,
        "design": {
            "release": "MSK-CHORD (MSK, Nature 2024), cBioPortal msk_chord_2024",
            "cancer_type": "Breast Cancer",
            "time_origin": (
                "re-anchored at earliest recorded primary diagnosis; the "
                "release's own origin is the sequencing date"
            ),
            "horizon_years": HORIZON_YEARS,
            "dedup_days": DEDUP_DAYS,
            "switch_window_days": SWITCH_WINDOW_DAYS,
            "dedup_sensitivity": list(DEDUP_SENSITIVITY),
            "window_sensitivity": list(WINDOW_SENSITIVITY),
            "os_months_origin": "sequencing date (verified by residual check)",
            "days_per_month": DAYS_PER_MONTH,
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
        "entry_point": "analysis/49_run_msk_chord_profile.py",
        "derived_dir": str(DERIVED_DIR.relative_to(ROOT)).replace("\\", "/"),
        "derived_note": (
            "Patient-level intermediates are written there, not to tables/. "
            "MSK-CHORD is CC BY-NC-ND 4.0 (NoDerivatives), so per-patient "
            "derived tables are not redistributable; tables/ holds aggregates "
            "only."
        ),
        "base_seed": None,
        "seed_note": "no randomness - every number here is a count or a rate",
    }
    (REPORT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    print("\n=== 사전 예측 채점 ===", flush=True)
    for key, met in (
        ("1 적응 기회 (스코핑 참고)", verdict["opportunity_prediction_met"]),
        ("2 신호 뒤 치료 변경 (주)", verdict["primary_prediction_met"]),
        ("3 시간 위약대조", verdict["placebo_in_time_prediction_met"]),
        ("4 라벨 순서 기지정답", verdict["label_ordering_prediction_met"]),
        ("5 ECOG는 상태", verdict["ecog_is_a_state_prediction_met"]),
        ("6 timing 축 판독", verdict["timing_axis_prediction_met"]),
    ):
        label = "무효(실패 불가)" if met is None else ("통과" if met else "빗나감")
        print(f"  {key}: {label}", flush=True)


if __name__ == "__main__":
    main()
