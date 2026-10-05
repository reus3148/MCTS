"""Is performance status a state clinicians act on, or just a number they record?

v2.2 established that MSK-CHORD records ECOG for 73.6% of the breast cohort and
that 45.2% have a documented change. That makes it a *time-varying* variable.
It does not make it a **state** in the sense the simulator needs.

The distinction matters because of what v2.0 found. A decision point is worth
opening only when it has stakes, state-dependence and resolution; v1.8, v1.9
and v2.0 each opened one that was missing a leg, and each collapsed to "always
X" or to noise. Our environment's state is currently ``(phase, year, recurred,
alive, toxicity_count)`` plus the treatments chosen - with recurrence binary and
near-absorbing, there is almost nothing for a mid-course decision to condition
on. Adding decision points before adding state would repeat v1.8 exactly.

So before writing any environment change, this module asks the data the
question the simulator cannot answer:

:func:`treatment_state` / :func:`stop_table`
    **Do clinicians stop or change systemic therapy after performance status
    worsens?** Same forward/backward placebo design as v2.2's switch analysis,
    which is what makes the two comparable. If treatment does not respond to
    ECOG, then an ECOG field would be a prognostic covariate and not a state
    the policy should plan over.

:func:`chemo_share_by_ecog`
    **Does the level gate the most toxic channel?** Real guidelines withhold
    chemotherapy at poor performance status; our environment's only eligibility
    gate is ``toxicity_count``. If the gate is visible in recorded care, the
    environment is missing it.

:func:`transition_rates`
    **How fast does it move, and does the rate depend on disease state?** These
    are the numbers an environment would be calibrated against, and the
    dependence on progression is the state-dependence leg itself: a variable
    that drifts with age alone adds no decision-relevant information.

Everything is a pure function over data frames, so the tests run without the
release present.
"""

from __future__ import annotations

import pandas as pd

DAYS_PER_YEAR = 365.25

#: Systemic treatment classes that count as "on treatment". Matches
#: ``adaptation.SWITCH_SUBTYPES`` so the two analyses describe the same object;
#: bone-directed and investigational agents are excluded for the same reasons.
TREATMENT_SUBTYPES = ("Chemo", "Hormone", "Targeted", "Biologic", "Immuno")

#: ECOG at or above which guidelines withhold cytotoxic chemotherapy. Declared
#: here rather than discovered: 2 is the conventional cut, and reading it off
#: the data would make the gate unfalsifiable.
POOR_PERFORMANCE = 2


def ecog_changes(performance_status: pd.DataFrame,
                 horizon_years: float = 5.0) -> pd.DataFrame:
    """One row per recorded ECOG change, with its direction.

    The release stores change points rather than measurements
    (``adaptation.ecog_encoding_check``), so every row after a patient's first
    *is* a change: ``direction`` is therefore always non-zero, and the first
    row of each patient is dropped because nothing preceded it.
    """
    limit = horizon_years * DAYS_PER_YEAR
    frame = performance_status.dropna(subset=["dx_start_date", "ECOG"])
    frame = frame[(frame["dx_start_date"] >= 0) & (frame["dx_start_date"] <= limit)]

    rows = []
    for patient, block in frame.groupby("PATIENT_ID"):
        ordered = block.sort_values("dx_start_date")
        values = [int(value) for value in ordered["ECOG"]]
        days = [float(day) for day in ordered["dx_start_date"]]
        for index in range(1, len(values)):
            rows.append({
                "PATIENT_ID": patient,
                "day": days[index],
                "previous_day": days[index - 1],
                "ecog_from": values[index - 1],
                "ecog_to": values[index],
                "direction": "worsened" if values[index] > values[index - 1]
                             else "improved",
                "size": abs(values[index] - values[index - 1]),
            })
    return pd.DataFrame(rows)


def treatment_spans(treatment: pd.DataFrame) -> dict[str, list[tuple[float, float]]]:
    """Merged on-treatment intervals per patient, in diagnosis-relative days.

    Merged rather than listed because "on systemic therapy" is one condition,
    not one per agent: a patient on letrozole and palbociclib is on treatment
    once. Without merging, a stop would be recorded every time one agent of a
    combination ended.
    """
    systemic = treatment[treatment["SUBTYPE"].isin(TREATMENT_SUBTYPES)].dropna(
        subset=["dx_start_date", "dx_stop_date"])
    spans: dict[str, list[tuple[float, float]]] = {}
    for patient, block in systemic.groupby("PATIENT_ID"):
        ordered = sorted((float(row.dx_start_date), float(row.dx_stop_date))
                         for row in block.itertuples())
        merged: list[list[float]] = []
        for start, stop in ordered:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], stop)
            else:
                merged.append([start, stop])
        spans[patient] = [(start, stop) for start, stop in merged]
    return spans


def on_treatment(spans: list[tuple[float, float]], day: float) -> bool:
    """Was the patient on systemic therapy on ``day``?"""
    return any(start <= day <= stop for start, stop in spans)


def stop_table(
    changes: pd.DataFrame,
    treatment: pd.DataFrame,
    followup: pd.Series,
    window_days: int = 90,
) -> pd.DataFrame:
    """Did systemic therapy stop around this ECOG change?

    One row per change that (a) happens while the patient is on systemic
    therapy - a stop is undefined otherwise - and (b) has ``window_days`` of
    follow-up left, so "still on treatment" is not an artefact of the record
    ending.

    ``stopped_after``
        On treatment at the change, off treatment ``window_days`` later.
    ``stopped_before``
        The placebo-in-time control, asked of the window the change cannot have
        caused: off treatment ``window_days`` *before* the change, on treatment
        at it. This is the mirror image rather than the identical question,
        because "stopping" is directional - the backward analogue of a stop is
        a start.

        Which is why it carries a second reading, and on this cohort the more
        informative one: it is exactly the indicator "**treatment started
        within the last** ``window_days``". If worsenings follow recent starts
        more often than improvements do, then ECOG is partly measuring
        treatment toxicity rather than only disease - the arrow runs from
        treatment to performance status, not only the other way.

    Reading them together is what separates "ECOG worsening precedes stopping"
    from "this patient's treatment is intermittent anyway".
    """
    if window_days <= 0:
        raise ValueError("window_days must be positive")
    spans = treatment_spans(treatment)
    if not spans:
        return pd.DataFrame(columns=[
            "PATIENT_ID", "day", "direction", "stopped_after", "stopped_before"])

    usable = changes[changes["PATIENT_ID"].isin(spans)].copy()
    usable["followup"] = usable["PATIENT_ID"].map(followup)
    usable = usable[usable["day"] + window_days <= usable["followup"]]

    rows = []
    for row in usable.itertuples():
        patient_spans = spans[row.PATIENT_ID]
        day = float(row.day)
        if not on_treatment(patient_spans, day):
            continue
        rows.append({
            "PATIENT_ID": row.PATIENT_ID,
            "day": day,
            "direction": row.direction,
            "ecog_from": row.ecog_from,
            "ecog_to": row.ecog_to,
            "stopped_after": not on_treatment(patient_spans, day + window_days),
            "stopped_before": not on_treatment(patient_spans, day - window_days),
        })
    return pd.DataFrame(rows)


def stop_rates(table: pd.DataFrame) -> pd.DataFrame:
    """Stop rates by direction of the ECOG change, with counts."""
    if table.empty:
        return pd.DataFrame(columns=["direction", "changes", "patients",
                                     "stopped_after", "stopped_before"])
    grouped = table.groupby("direction")
    summary = pd.DataFrame({
        "changes": grouped.size(),
        "patients": grouped["PATIENT_ID"].nunique(),
        "stopped_after": grouped["stopped_after"].mean(),
        "stopped_before": grouped["stopped_before"].mean(),
    })
    summary["forward_minus_backward"] = (
        summary["stopped_after"] - summary["stopped_before"])
    return summary.reset_index()


def stop_rates_by_size(table: pd.DataFrame) -> pd.DataFrame:
    """Stop rate by how far ECOG fell, against the improvement baseline.

    The dose-response check, and the one that decides whether a null result on
    :func:`stop_rates` is a real null or an artefact of a crude definition. A
    one-step change from 0 to 1 is a different clinical event from a three-step
    change to bed-bound; if stopping tracked performance status, the gap would
    widen with the size of the fall. If it does not widen, "clinicians act on
    ECOG" is not rescued by a better definition of the event.
    """
    if table.empty:
        return pd.DataFrame(columns=["size", "changes", "stopped_after", "gap"])
    improved = table[table["direction"] == "improved"]
    baseline = float(improved["stopped_after"].mean()) if len(improved) else float("nan")
    worsened = table[table["direction"] == "worsened"].copy()
    worsened["size"] = worsened["ecog_to"] - worsened["ecog_from"]
    grouped = worsened.groupby("size")
    summary = pd.DataFrame({
        "changes": grouped.size(),
        "patients": grouped["PATIENT_ID"].nunique(),
        "stopped_after": grouped["stopped_after"].mean(),
    })
    summary["baseline_improved"] = baseline
    summary["gap"] = summary["stopped_after"] - baseline
    return summary.reset_index()


def ecog_at(records: list[tuple[float, int]], day: float) -> int | None:
    """Most recent ECOG recorded on or before ``day``, or ``None``."""
    seen = [value for recorded, value in records if recorded <= day]
    return seen[-1] if seen else None


def chemo_share_by_ecog(
    performance_status: pd.DataFrame,
    treatment: pd.DataFrame,
    horizon_years: float = 5.0,
    strata: pd.Series | None = None,
) -> pd.DataFrame:
    """Share of systemic starts that are chemotherapy, by ECOG at the start.

    The denominator is *systemic starts*, not patients: the question is whether
    the level gates which channel is chosen once a decision to treat has been
    made. Using patients as the denominator would blend that with the separate
    question of whether poor performance status means no treatment at all.

    Starts before the patient's first ECOG record are dropped rather than
    assigned a level, since a baseline carried backwards would be invented.

    ``strata``, when given, maps patient to a stratum (stage, say) and adds it
    as a grouping column. Stratifying is how a reversal of the expected
    direction gets checked against the obvious confounder before it is
    believed: if poor performance status marks advanced disease, and advanced
    disease gets chemotherapy, the marginal association says nothing about
    fitness.
    """
    limit = horizon_years * DAYS_PER_YEAR
    status = performance_status.dropna(subset=["dx_start_date", "ECOG"])
    records: dict[str, list[tuple[float, int]]] = {}
    for patient, block in status.groupby("PATIENT_ID"):
        ordered = block.sort_values("dx_start_date")
        records[patient] = [(float(day), int(value)) for day, value
                            in zip(ordered["dx_start_date"], ordered["ECOG"])]

    systemic = treatment[treatment["SUBTYPE"].isin(TREATMENT_SUBTYPES)].dropna(
        subset=["dx_start_date"])
    systemic = systemic[(systemic["dx_start_date"] >= 0)
                        & (systemic["dx_start_date"] <= limit)]

    rows = []
    for row in systemic.itertuples():
        patient_records = records.get(row.PATIENT_ID)
        if not patient_records:
            continue
        level = ecog_at(patient_records, float(row.dx_start_date))
        if level is None:
            continue
        record = {"ecog": level, "is_chemo": row.SUBTYPE == "Chemo"}
        if strata is not None:
            record["stratum"] = strata.get(row.PATIENT_ID)
        rows.append(record)
    columns = ["ecog", "starts", "chemo", "chemo_share"]
    if not rows:
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(rows)
    keys = ["stratum", "ecog"] if strata is not None else ["ecog"]
    grouped = frame.groupby(keys, dropna=False)
    summary = pd.DataFrame({
        "starts": grouped.size(),
        "chemo": grouped["is_chemo"].sum(),
    })
    summary["chemo_share"] = summary["chemo"] / summary["starts"]
    return summary.reset_index()


def transition_rates(
    performance_status: pd.DataFrame,
    progression: pd.DataFrame,
    horizon_years: float = 5.0,
) -> pd.DataFrame:
    """Annual rate of recorded ECOG worsening, before and after first progression.

    The rate an environment would be calibrated against, split by the one piece
    of disease state our simulator already carries. If worsening is no commoner
    after progression, performance status is drifting with time rather than
    tracking disease, and a policy would gain nothing by observing it.

    Person-time is counted from each patient's **first ECOG record**, not from
    diagnosis: before that record the release cannot show a change, so counting
    that time would dilute both rates by the same arbitrary amount.
    """
    limit = horizon_years * DAYS_PER_YEAR
    status = performance_status.dropna(subset=["dx_start_date", "ECOG"])
    status = status[(status["dx_start_date"] >= 0) & (status["dx_start_date"] <= limit)]

    # A cohort with no recorded progression is a real case (and the one every
    # unit test builds), so an empty frame has to mean "nobody progressed"
    # rather than raising on a column that was never created.
    if progression.empty or "PROGRESSION" not in progression:
        first_progression = pd.Series(dtype=float)
    else:
        first_progression = (
            progression[progression["PROGRESSION"] == "Y"]
            .dropna(subset=["dx_start_date"])
            .groupby("PATIENT_ID")["dx_start_date"].min())

    exposure = {"before": 0.0, "after": 0.0}
    events = {"before": 0, "after": 0}
    for patient, block in status.groupby("PATIENT_ID"):
        ordered = block.sort_values("dx_start_date")
        days = [float(day) for day in ordered["dx_start_date"]]
        values = [int(value) for value in ordered["ECOG"]]
        start, end = days[0], min(limit, max(days))
        if end <= start:
            continue
        progressed = float(first_progression.get(patient, float("inf")))
        split = min(max(progressed, start), end)
        exposure["before"] += (split - start) / DAYS_PER_YEAR
        exposure["after"] += (end - split) / DAYS_PER_YEAR
        for index in range(1, len(values)):
            if values[index] <= values[index - 1]:
                continue
            key = "after" if days[index] >= progressed else "before"
            events[key] += 1

    rows = []
    for key in ("before", "after"):
        years = exposure[key]
        rows.append({
            "period": f"{key}_first_progression",
            "worsening_events": events[key],
            "person_years": years,
            "worsenings_per_year": events[key] / years if years > 0 else float("nan"),
        })
    return pd.DataFrame(rows)
