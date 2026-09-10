"""Does a real dataset contain the movetext our simulator has been inventing? (v1.6)

The project's founding analogy is chess: `지금 보드 -> 최선의 수 -> 결과` is the
same object as `지금 환자 상태 -> 최선의 치료 -> 결과`. What made the analogy hard
to honour was that METABRIC records only the *result* of each game plus the
opening position. There is no movetext - no ordered list of treatments with the
dates they were played, and no evaluation of the position between moves.

Everything from v0.2 onward worked around that by declaring the moves and their
values in ``configs/dynamic_v0_5.json``. v1.5 showed what that costs: 81% of the
original +0.0332 utility gap came from two asymmetries we had written into the
environment ourselves, and the single largest was that standard chemotherapy,
standard endocrine therapy and local radiotherapy are declared to have *zero*
survival benefit while charging toxicity and burden. Filling those in was put on
the human-review list because no data we had could speak to it.

GENIE BPC Breast Cancer v1.0-public is a real curated cohort with movetext:
regimen-level drugs with start and end days from diagnosis, radiologist
assessments on every imaging report, PFS, OS and time-to-next-treatment. This
run does not estimate a treatment effect. It asks a prior question:

    **Which of the parameters our environment currently declares could this
    dataset supply, and which could it not?**

That is a data-adequacy assessment, and it is deliberately descriptive. Nothing
here is adjusted, weighted or causal. Reading a confounded switch rate as an
effect would repeat exactly the mistake v1.3's negative control caught.

PRE-SPECIFIED PREDICTIONS, recorded before the run
--------------------------------------------------
Written after seeing only the cohort's size, subtype mix, top-20 drug list and
the marginal counts of ``image_overall`` - none of which the four below can be
read off.

1. **Game length** - the median number of regimens started within 5 years of
   diagnosis is **3 or fewer**. Our environment plays a 5-year game with five
   action axes; if real games are much longer, the horizon is wrong.
2. **Primary - closed loop exists** - a "Progressing" scan is followed by a new
   regimen within 90 days at **at least twice** the rate a "Stable" or
   "Improving" scan is. This is the empirical counterpart of the response
   channel v0.5 mean-neutralised and v1.5 could not separate from noise. If it
   is false, the recorded games contain no visible adaptation and the
   closed-loop premise loses its observational support.
3. **Placebo in time** - for progressing scans, the forward-window switch rate
   exceeds the backward-window rate by more than it does for controlled scans.
   A scan cannot cause a regimen that started before it, so a symmetric result
   would mean prediction 2 is picking up "this patient switches a lot" rather
   than "this scan triggered a switch".
4. **Standard endocrine therapy is estimable** - at least **150** HR-positive
   index cancers have a first-line regimen that is endocrine therapy and
   nothing else. Below that there is no point asking this dataset what standard
   endocrine therapy is worth.

Prediction 1 is the one most likely to be wrong: this is a tertiary-centre,
sequencing-referred cohort with 241 stage IV cancers, so games may be long.
Prediction 2 is the one that matters most for what we do next.
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
from analysis.genie.loader import DATA_DIR, load  # noqa: E402
from analysis.genie.sequences import (  # noqa: E402
    DAYS_PER_YEAR,
    annotate_regimens,
    response_switch_table,
    sequence_lengths,
    switch_rates,
)

REPORT_DIR = ROOT / "reports" / "genie-bpc-profile-v1.6"
TABLE_DIR = REPORT_DIR / "tables"

RUN_DATE = "2026-09-10"
HORIZON_YEARS = 5.0            # the simulator's horizon since v0.2
SWITCH_WINDOW_DAYS = 90
MEDIAN_SEQUENCE_CEILING = 3
SWITCH_RATIO_FLOOR = 2.0
FIRST_LINE_ENDOCRINE_FLOOR = 150

#: The environment's action axes (v1.5's ``ACTION_FIELDS``) against what this
#: release can observe. Free to write down, and the reason the assessment
#: exists, so it ships as a table rather than as a claim in prose.
ACTION_COVERAGE = (
    ("timing", "neoadjuvant vs adjuvant", False,
     "regimen start days exist but no surgery date, so before/after is undefined"),
    ("surgery", "lumpectomy vs mastectomy", False,
     "not curated - GENIE BPC records systemic regimens only"),
    ("chemo", "none / standard / intensified", True,
     "regimen drugs and start-end days present for every line"),
    ("endocrine", "none / standard / extended", True,
     "regimen drugs and start-end days present; duration gives standard vs extended"),
    ("radiation", "none / local / regional", False,
     "not curated"),
)

PRESPECIFIED_PREDICTION = {
    "game_length": (
        f"Median regimens started within {HORIZON_YEARS:.0f} years of diagnosis "
        f"is <= {MEDIAN_SEQUENCE_CEILING}."
    ),
    "primary_closed_loop": (
        f"A progressing scan is followed by a new regimen within "
        f"{SWITCH_WINDOW_DAYS} days at >= {SWITCH_RATIO_FLOOR:.0f}x the rate a "
        "stable or improving scan is."
    ),
    "placebo_in_time": (
        "For progressing scans the forward-minus-backward switch-rate gap "
        "exceeds the same gap for controlled scans - a scan cannot cause a "
        "regimen that started before it."
    ),
    "endocrine_estimable": (
        f"At least {FIRST_LINE_ENDOCRINE_FLOOR} HR-positive index cancers have "
        "a first-line regimen that is endocrine therapy and nothing else."
    ),
    "why": (
        "v1.5 found that 81% of the original utility gap came from asymmetries "
        "declared in our own config, the largest being zero declared benefit "
        "for standard treatment. Before that can be fixed we have to know "
        "whether any dataset we hold can speak to it."
    ),
}

HR_POSITIVE_SUBTYPES = ("HR+, HER2-", "HR+, HER2+")


def cohort_overview(cancers: pd.DataFrame) -> pd.DataFrame:
    counts = (cancers.groupby(["institution", "bca_subtype"], dropna=False)
              .size().rename("cancers").reset_index())
    counts["bca_subtype"] = counts["bca_subtype"].fillna("(unrecorded)")
    return counts.sort_values(["institution", "bca_subtype"])


def sequence_table(lengths: pd.Series, cancers: pd.DataFrame) -> pd.DataFrame:
    joined = cancers.set_index(["record_id", "ca_seq"]).join(
        lengths, how="left")
    joined["n_regimens"] = joined["n_regimens"].fillna(0).astype(int)
    rows = []
    for stage, group in joined.groupby("stage_dx", dropna=False):
        played = group[group["n_regimens"] > 0]["n_regimens"]
        rows.append({
            "stage_dx": stage,
            "cancers": int(len(group)),
            "cancers_with_a_regimen": int(len(played)),
            "median_regimens": float(played.median()) if len(played) else np.nan,
            "p90_regimens": float(played.quantile(0.9)) if len(played) else np.nan,
            "max_regimens": int(played.max()) if len(played) else 0,
        })
    overall = joined[joined["n_regimens"] > 0]["n_regimens"]
    rows.append({
        "stage_dx": "ALL",
        "cancers": int(len(joined)),
        "cancers_with_a_regimen": int(len(overall)),
        "median_regimens": float(overall.median()),
        "p90_regimens": float(overall.quantile(0.9)),
        "max_regimens": int(overall.max()),
    })
    return pd.DataFrame(rows)


def channel_by_line(regimens: pd.DataFrame, lines: int = 5) -> pd.DataFrame:
    inside = regimens[
        regimens["regimen_number_within_cancer"].between(1, lines)]
    counts = (inside.groupby(
        ["regimen_number_within_cancer", "primary_channel"])
        .size().rename("regimens").reset_index())
    totals = counts.groupby("regimen_number_within_cancer")["regimens"].transform("sum")
    counts["percent"] = (counts["regimens"] / totals * 100).round(2)
    return counts.sort_values(
        ["regimen_number_within_cancer", "regimens"], ascending=[True, False])


def first_line_endocrine(regimens: pd.DataFrame,
                         cancers: pd.DataFrame) -> pd.DataFrame:
    first = regimens[regimens["regimen_number_within_cancer"] == 1]
    joined = first.merge(
        cancers[["record_id", "ca_seq", "bca_subtype", "stage_dx"]],
        on=["record_id", "ca_seq"], how="left")
    hr_positive = joined[joined["bca_subtype"].isin(HR_POSITIVE_SUBTYPES)]
    rows = []
    for (subtype, stage), group in hr_positive.groupby(
            ["bca_subtype", "stage_dx"], dropna=False):
        rows.append({
            "bca_subtype": subtype,
            "stage_dx": stage,
            "first_line_regimens": int(len(group)),
            "endocrine_only": int(group["endocrine_only"].sum()),
        })
    frame = pd.DataFrame(rows).sort_values(
        ["bca_subtype", "stage_dx"]).reset_index(drop=True)
    frame.loc[len(frame)] = {
        "bca_subtype": "ALL HR+",
        "stage_dx": "ALL",
        "first_line_regimens": int(len(hr_positive)),
        "endocrine_only": int(hr_positive["endocrine_only"].sum()),
    }
    return frame


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    release = load()
    cancers = release.cancers
    regimens = annotate_regimens(release.regimens)

    print(f"환자 {release.n_patients}명 · index 암 {len(cancers)}건 · "
          f"index 레지멘 {len(regimens)}건", flush=True)

    overview = cohort_overview(cancers)
    overview.to_csv(TABLE_DIR / "cohort_overview.csv", index=False)

    coverage = pd.DataFrame(
        [{"action_axis": axis, "environment_levels": levels,
          "observable_in_genie_bpc": observable, "note": note}
         for axis, levels, observable, note in ACTION_COVERAGE])
    coverage.to_csv(TABLE_DIR / "action_space_coverage.csv", index=False)
    observable = int(coverage["observable_in_genie_bpc"].sum())
    print(f"\n환경 행동축 {len(coverage)}개 중 관측 가능 {observable}개", flush=True)
    print(coverage.to_string(index=False), flush=True)

    lengths = sequence_lengths(regimens, HORIZON_YEARS)
    sequences = sequence_table(lengths, cancers)
    sequences.to_csv(TABLE_DIR / "sequence_lengths.csv", index=False)
    median_length = float(sequences.loc[
        sequences["stage_dx"] == "ALL", "median_regimens"].iloc[0])
    print(f"\n5년 내 레지멘 수 중앙값 {median_length:.1f}", flush=True)
    print(sequences.to_string(index=False), flush=True)

    lines = channel_by_line(regimens)
    lines.to_csv(TABLE_DIR / "channel_by_line.csv", index=False)

    endocrine = first_line_endocrine(regimens, cancers)
    endocrine.to_csv(TABLE_DIR / "first_line_endocrine.csv", index=False)
    endocrine_only = int(endocrine.loc[
        endocrine["bca_subtype"] == "ALL HR+", "endocrine_only"].iloc[0])
    print(f"\nHR+ 1차 단독 내분비요법 {endocrine_only}건", flush=True)

    followup = (cancers.assign(
        days=cancers["tt_os_dx_yrs"] * DAYS_PER_YEAR)
        .groupby("record_id")["days"].max())
    scans = response_switch_table(
        release.imaging, regimens, followup, SWITCH_WINDOW_DAYS)
    rates = switch_rates(scans)
    rates.to_csv(TABLE_DIR / "response_switch.csv", index=False)
    print(f"\n판독 {len(scans)}건 ({SWITCH_WINDOW_DAYS}일 창)", flush=True)
    print(rates.to_string(index=False), flush=True)

    indexed = rates.set_index("group")
    progressing = indexed.loc["progressing"]
    controlled = indexed.loc["controlled"]
    ratio = float(progressing["switch_after"] / controlled["switch_after"])

    # Two-proportion z for the forward-window difference. Descriptive: scans are
    # clustered within patients, so this understates the standard error. It is
    # reported to show the comparison is not a handful of scans, not as a test.
    n_p = int(progressing["scans"])
    n_c = int(controlled["scans"])
    p_p = float(progressing["switch_after"])
    p_c = float(controlled["switch_after"])
    pooled = (p_p * n_p + p_c * n_c) / (n_p + n_c)
    standard_error = float(np.sqrt(pooled * (1 - pooled) * (1 / n_p + 1 / n_c)))
    z_unclustered = float((p_p - p_c) / standard_error) if standard_error else np.nan

    verdict = {
        "patients": release.n_patients,
        "index_cancers": int(len(cancers)),
        "index_regimens": int(len(regimens)),
        "median_regimens_within_horizon": median_length,
        "horizon_years": HORIZON_YEARS,
        "game_length_prediction_met": bool(
            median_length <= MEDIAN_SEQUENCE_CEILING),
        "switch_window_days": SWITCH_WINDOW_DAYS,
        "scans_scored": int(len(scans)),
        "switch_rate_progressing": p_p,
        "switch_rate_controlled": p_c,
        "switch_rate_ratio": ratio,
        "primary_prediction_met": bool(ratio >= SWITCH_RATIO_FLOOR),
        "z_unclustered": z_unclustered,
        "forward_minus_backward_progressing": float(
            progressing["forward_minus_backward"]),
        "forward_minus_backward_controlled": float(
            controlled["forward_minus_backward"]),
        "placebo_in_time_prediction_met": bool(
            progressing["forward_minus_backward"]
            > controlled["forward_minus_backward"]),
        "hr_positive_first_line_endocrine_only": endocrine_only,
        "endocrine_estimable_prediction_met": bool(
            endocrine_only >= FIRST_LINE_ENDOCRINE_FLOOR),
        "action_axes_total": int(len(coverage)),
        "action_axes_observable": observable,
        "action_axes_missing": coverage.loc[
            ~coverage["observable_in_genie_bpc"], "action_axis"].tolist(),
    }

    metrics = {
        "run_date": RUN_DATE,
        "analysis_label": "genie-bpc-profile-v1.6",
        "question": (
            "Which parameters that configs/dynamic_v0_5.json currently declares "
            "could GENIE BPC Breast Cancer v1.0-public supply, and which not?"
        ),
        "estimand": (
            "None. This is a descriptive data-adequacy assessment of the "
            "release: sequence lengths, channel mix by line, the association "
            "between a radiologist's assessment and a subsequent regimen "
            "start, and coverage of the environment's action axes."
        ),
        "scope_warning": (
            "Descriptive only. The switch rates are unadjusted and confounded "
            "by indication; they say the recorded games contain adaptation, "
            "not that adaptation caused anything."
        ),
        "prespecified_prediction": PRESPECIFIED_PREDICTION,
        "design": {
            "release": "GENIE BPC Breast Cancer v1.0-public",
            "index_cancers_only": True,
            "horizon_years": HORIZON_YEARS,
            "switch_window_days": SWITCH_WINDOW_DAYS,
            "hr_positive_subtypes": list(HR_POSITIVE_SUBTYPES),
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
        "entry_point": "analysis/37_run_genie_bpc_profile.py",
        "base_seed": None,
        "seed_note": "no randomness - every number here is a count or a rate",
    }
    (REPORT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    print("\n=== 사전 예측 채점 ===", flush=True)
    for key, met in (
        ("1 기보 길이", verdict["game_length_prediction_met"]),
        ("2 폐루프 (주)", verdict["primary_prediction_met"]),
        ("3 시간 위약대조", verdict["placebo_in_time_prediction_met"]),
        ("4 내분비 추정가능", verdict["endocrine_estimable_prediction_met"]),
    ):
        print(f"  {key}: {'통과' if met else '빗나감'}", flush=True)


if __name__ == "__main__":
    main()
