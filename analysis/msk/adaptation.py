"""How many places real breast-cancer care has to adapt, and whether it does.

This module exists because of one number from v1.7. Blinding the response
channel from the planner moved the utility gap by -0.0004 (z = -0.29), and the
reason was arithmetic rather than subtle: our environment offers a response
signal **once**, in 20.8% of episodes. A sequential method with one branch
point is a static optimiser with extra steps, so v1.7 could not tell whether
MCTS was failing to adapt or had nothing to adapt to.

MSK-CHORD can answer the question the simulator could not. Every CT, PET and
MRI in the release carries an NLP-derived progression call and a date, so the
number of moments at which a clinician held a fresh read of the board is a
countable quantity rather than a design choice.

Three measurements, in the order they should be read:

:func:`decision_points`
    How many adaptation opportunities a patient's record contains. Nearby
    scans collapse into one opportunity - a CT and a PET three days apart are
    one look at the board, and counting them twice would inflate the headline
    we are trying to hold the simulator to.

:func:`switch_table` / :func:`switch_rates`
    Whether treatment actually changes after a progression call, with the
    backward window as the placebo-in-time control. v1.6 established this
    design on GENIE BPC; keeping it identical is what makes the two releases
    comparable.

:func:`ecog_trajectory`
    Whether performance status is a *state* or a baseline covariate. If ECOG
    never moves within a patient, the simulator loses nothing by holding it
    fixed; if it moves, our state vector is missing a variable that clinicians
    demonstrably act on.
"""

from __future__ import annotations

import pandas as pd

DAYS_PER_YEAR = 365.25
DAYS_PER_MONTH = 30.4375

#: Progression calls in the release. ``Indeterminate`` is kept as its own
#: group rather than merged into either side: it is 20% of breast assessments,
#: and a clinician reading "indeterminate" is in a different decision
#: situation from one reading "no progression".
PROGRESSING = "Y"
CONTROLLED = "N"
INDETERMINATE = "Indeterminate"

#: Treatment subtypes that count as a systemic move. ``Bone Treatment`` is
#: supportive care and ``Investigational`` is unclassifiable, so neither counts
#: as a switch of the plan our action space describes.
SWITCH_SUBTYPES = ("Chemo", "Hormone", "Targeted", "Biologic", "Immuno")


def followup_days(patients: pd.DataFrame, anchor: pd.Series) -> pd.Series:
    """Days from diagnosis to death or last contact, per patient.

    ``OS_MONTHS`` in MSK-CHORD is measured from the **sequencing** date, not
    from diagnosis: residual checks on this cohort put 0% of recorded events
    after the endpoint on that reading and 66% after it on the diagnosis
    reading. Converting therefore means adding the diagnosis-to-sequencing
    interval back in, which ``anchor`` holds as a negative number of days.

    That interval is also the release's central selection: a patient had to
    survive from diagnosis to sequencing to be in the cohort at all (median
    441 days here). Follow-up computed this way is correct for placing events
    on a diagnosis-relative axis, and is **not** a survival time that may be
    fed to Kaplan-Meier without left-truncation handling.
    """
    indexed = patients.set_index("PATIENT_ID")
    end = indexed["OS_MONTHS"] * DAYS_PER_MONTH
    return (end - indexed.index.map(anchor)).rename("followup_days")


def assessment_group(value: str | float) -> str:
    if value == PROGRESSING:
        return "progressing"
    if value == CONTROLLED:
        return "controlled"
    if value == INDETERMINATE:
        return "indeterminate"
    return "other"


def decision_points(
    progression: pd.DataFrame,
    horizon_years: float = 5.0,
    dedup_days: int = 30,
) -> pd.DataFrame:
    """Distinct adaptation opportunities per patient inside the horizon.

    An opportunity is a day on which a fresh progression call existed. Calls
    within ``dedup_days`` of the previous retained one are dropped as the same
    clinical read; ``dedup_days=0`` would count every scan, which overstates
    the figure by roughly a third on this cohort.

    Returns one row per patient with a recorded call, plus the share of
    opportunities that actually said "progressing" - the subset at which the
    board had genuinely changed.
    """
    if dedup_days < 0:
        raise ValueError("dedup_days must not be negative")
    limit = horizon_years * DAYS_PER_YEAR
    inside = progression.dropna(subset=["dx_start_date", "PROGRESSION"])
    inside = inside[(inside["dx_start_date"] >= 0) & (inside["dx_start_date"] <= limit)]

    rows = []
    for patient, block in inside.groupby("PATIENT_ID"):
        ordered = block.sort_values("dx_start_date")
        kept_days: list[float] = []
        kept_calls: list[str] = []
        for row in ordered.itertuples():
            day = float(row.dx_start_date)
            if kept_days and day - kept_days[-1] < dedup_days:
                continue
            kept_days.append(day)
            kept_calls.append(assessment_group(row.PROGRESSION))
        progressing = sum(1 for call in kept_calls if call == "progressing")
        rows.append({
            "PATIENT_ID": patient,
            "scans": len(ordered),
            "opportunities": len(kept_days),
            "progressing": progressing,
            "first_day": kept_days[0],
            "last_day": kept_days[-1],
        })
    return pd.DataFrame(rows)


def switch_table(
    progression: pd.DataFrame,
    treatment: pd.DataFrame,
    followup: pd.Series,
    window_days: int = 90,
    horizon_years: float = 5.0,
) -> pd.DataFrame:
    """Did systemic treatment change soon after this call - and just before?

    One row per progression call that (a) falls inside the horizon, (b) happens
    on or after the patient's first systemic treatment start, and (c) has
    ``window_days`` of follow-up left. Condition (c) is administrative
    censoring: without it, calls near the end of follow-up would count as "no
    switch" because nothing more was recorded.

    Two readings of "changed", reported side by side because they can diverge:

    ``switch_after``
        Any systemic start day in the forward window. Directly comparable with
        the GENIE BPC figure from v1.6, which counted regimen starts.
    ``new_agent_after``
        A start in the window for an agent the patient had never received
        before the call. Stricter, and immune to the re-dosing and
        continuation rows that a per-agent release records.

    ``switch_before`` / ``new_agent_before`` ask the same two questions of the
    window *preceding* the call, which the call cannot have caused.
    """
    if window_days <= 0:
        raise ValueError("window_days must be positive")
    limit = horizon_years * DAYS_PER_YEAR

    systemic = treatment[treatment["SUBTYPE"].isin(SWITCH_SUBTYPES)].dropna(
        subset=["dx_start_date", "AGENT"])
    if systemic.empty:
        return pd.DataFrame(columns=[
            "PATIENT_ID", "dx_start_date", "PROGRESSION", "group",
            "switch_after", "switch_before", "new_agent_after", "new_agent_before"])

    starts = {
        patient: sorted((float(row.dx_start_date), str(row.AGENT))
                        for row in block.itertuples())
        for patient, block in systemic.groupby("PATIENT_ID")
    }
    first_start = {patient: events[0][0] for patient, events in starts.items()}

    calls = progression.dropna(subset=["dx_start_date", "PROGRESSION"]).copy()
    calls = calls[calls["PATIENT_ID"].isin(starts)]
    calls["first_start"] = calls["PATIENT_ID"].map(first_start)
    calls["followup"] = calls["PATIENT_ID"].map(followup)
    calls = calls[
        (calls["dx_start_date"] >= calls["first_start"])
        & (calls["dx_start_date"] <= limit)
        & (calls["dx_start_date"] + window_days <= calls["followup"])
    ].copy()

    def _window(patient: str, low: float, high: float) -> list[tuple[float, str]]:
        """Starts in the half-open interval ``(low, high]``."""
        return [event for event in starts[patient] if low < event[0] <= high]

    after, before, new_after, new_before = [], [], [], []
    for row in calls.itertuples():
        day = float(row.dx_start_date)
        seen = {agent for start, agent in starts[row.PATIENT_ID] if start <= day}
        forward = _window(row.PATIENT_ID, day, day + window_days)
        backward = _window(row.PATIENT_ID, day - window_days, day)
        # For the backward window, "new" is judged against what the patient had
        # received before that window opened, not before the call.
        earlier = {agent for start, agent in starts[row.PATIENT_ID]
                   if start <= day - window_days}
        after.append(bool(forward))
        before.append(bool(backward))
        new_after.append(any(agent not in seen for _, agent in forward))
        new_before.append(any(agent not in earlier for _, agent in backward))

    calls["switch_after"] = after
    calls["switch_before"] = before
    calls["new_agent_after"] = new_after
    calls["new_agent_before"] = new_before
    calls["group"] = calls["PROGRESSION"].map(assessment_group)
    return calls[[
        "PATIENT_ID", "dx_start_date", "PROCEDURE_TYPE", "PROGRESSION", "group",
        "switch_after", "switch_before", "new_agent_after", "new_agent_before",
    ]]


def switch_rates(table: pd.DataFrame) -> pd.DataFrame:
    """Switch rates by call group, with counts so n is never hidden."""
    grouped = table.groupby("group")
    summary = pd.DataFrame({
        "calls": grouped.size(),
        "patients": grouped["PATIENT_ID"].nunique(),
        "switch_after": grouped["switch_after"].mean(),
        "switch_before": grouped["switch_before"].mean(),
        "new_agent_after": grouped["new_agent_after"].mean(),
        "new_agent_before": grouped["new_agent_before"].mean(),
    })
    summary["forward_minus_backward"] = (
        summary["switch_after"] - summary["switch_before"])
    summary["new_forward_minus_backward"] = (
        summary["new_agent_after"] - summary["new_agent_before"])
    return summary.reset_index()


def ecog_encoding_check(performance_status: pd.DataFrame) -> dict:
    """Does the release record every ECOG measurement, or only the changes?

    This check exists because the obvious reading of the ECOG timeline is
    wrong. Asking "what share of patients with two or more measurements show
    two or more distinct values?" returns **100.0%** on this cohort, which
    looks like a finding about patients and is a fact about the file: the
    release emits a row only when the value changes, so two rows cannot carry
    one value. On the breast cohort, 0 of 27,692 consecutive pairs are equal,
    including 0 of 1,790 pairs recorded on the same day.

    Run on the **whole** timeline, before any horizon filter, because the
    question is about the encoding rather than about a cohort. If
    ``equal_consecutive_pairs`` is zero then any test of the form "does ECOG
    move within patients?" cannot fail, and its result is not evidence.
    """
    frame = performance_status.dropna(subset=["dx_start_date", "ECOG"])
    pairs = equal = same_day = same_day_equal = 0
    for _, block in frame.groupby("PATIENT_ID"):
        ordered = block.sort_values("dx_start_date")
        values = list(ordered["ECOG"])
        days = list(ordered["dx_start_date"])
        for index in range(1, len(values)):
            pairs += 1
            if values[index] == values[index - 1]:
                equal += 1
            if days[index] == days[index - 1]:
                same_day += 1
                if values[index] == values[index - 1]:
                    same_day_equal += 1
    return {
        "consecutive_pairs": pairs,
        "equal_consecutive_pairs": equal,
        "same_day_pairs": same_day,
        "equal_same_day_pairs": same_day_equal,
        "change_point_encoded": bool(pairs > 0 and equal == 0),
    }


def ecog_trajectory(
    performance_status: pd.DataFrame,
    horizon_years: float = 5.0,
) -> pd.DataFrame:
    """Within-patient ECOG movement inside the horizon.

    The question is whether performance status belongs in the state vector.
    Our schema holds no ECOG field, and v1.6 recorded that GENIE BPC could not
    have supplied one.

    Read the output through :func:`ecog_encoding_check` first. Because the
    release stores change points rather than measurements, ``distinct`` and
    ``ever_worsened`` are conditional on a change having been recorded at all,
    and ``measurements`` counts **documented changes plus one**, not clinic
    visits. The quantity that carries information is therefore the share of the
    *cohort* reaching two rows - patients whose performance status is known to
    have moved - and not the share of movers among patients with two rows,
    which is 1 by construction.
    """
    limit = horizon_years * DAYS_PER_YEAR
    inside = performance_status.dropna(subset=["dx_start_date", "ECOG"])
    inside = inside[(inside["dx_start_date"] >= 0) & (inside["dx_start_date"] <= limit)]

    rows = []
    for patient, block in inside.groupby("PATIENT_ID"):
        ordered = block.sort_values("dx_start_date")
        values = [int(value) for value in ordered["ECOG"]]
        worsened = any(later > earlier
                       for earlier, later in zip(values, values[1:]))
        improved = any(later < earlier
                       for earlier, later in zip(values, values[1:]))
        rows.append({
            "PATIENT_ID": patient,
            "measurements": len(values),
            "distinct": len(set(values)),
            "first": values[0],
            "last": values[-1],
            "worst": max(values),
            "range": max(values) - min(values),
            "ever_worsened": worsened,
            "ever_improved": improved,
        })
    return pd.DataFrame(rows)


def opportunity_summary(points: pd.DataFrame, cohort_size: int) -> dict:
    """Headline numbers for the adaptation-opportunity count.

    ``share_of_cohort`` uses the whole cohort as the denominator, so a patient
    with no recorded imaging counts as zero opportunities rather than being
    dropped - the simulator's 20.8% was computed the same way.
    """
    if points.empty:
        return {"patients": 0, "share_of_cohort": 0.0}
    opportunities = points["opportunities"]
    return {
        "patients": int(len(points)),
        "cohort": int(cohort_size),
        "share_of_cohort": float(len(points) / cohort_size),
        "median": float(opportunities.median()),
        "mean": float(opportunities.mean()),
        "p25": float(opportunities.quantile(0.25)),
        "p75": float(opportunities.quantile(0.75)),
        "p90": float(opportunities.quantile(0.90)),
        "max": int(opportunities.max()),
        "at_least_two": float((opportunities >= 2).mean()),
        "at_least_five": float((opportunities >= 5).mean()),
        "progressing_median": float(points["progressing"].median()),
        "progressing_at_least_one": float((points["progressing"] >= 1).mean()),
    }
