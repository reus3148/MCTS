"""How much of our action space MSK-CHORD can actually see.

The simulator from v0.5 onward decides on five axes - ``timing``, ``surgery``,
``chemo``, ``endocrine``, ``radiation`` - and v1.6 found that GENIE BPC
observed only two of them, both systemic. That verdict set the ceiling on
every comparison we can make against recorded care, so it is worth re-asking
of a release that carries surgery and radiation timelines.

Two separate questions, kept separate on purpose:

**Occurrence** - was the move played at all, and when?
    Answerable for all five axes here, which is the upgrade over GENIE.

**Level** - *which* rung of the ladder, out of ``none``/``standard``/
    ``intensified``?
    Answerable only where the release records something that distinguishes
    rungs. Agent identity distinguishes chemotherapy regimens; cumulative
    duration distinguishes standard from extended endocrine therapy. Nothing
    in the release distinguishes breast-conserving surgery from mastectomy, or
    local from regional radiation - ``data_timeline_radiation.txt`` carries a
    single ``SUBTYPE`` value for all 8,864 breast rows.

Reporting a partially observable axis as observable would let a later analysis
claim it had validated a rung that was never in the data, so each axis carries
its resolution explicitly in :data:`AXIS_RESOLUTION`.
"""

from __future__ import annotations

import pandas as pd

DAYS_PER_YEAR = 365.25

#: Systemic treatment classes in the release's ``SUBTYPE`` column, mapped onto
#: the channels the simulator reasons about. ``Bone Treatment`` (zoledronic
#: acid, denosumab) and ``Investigational`` are deliberately *not* folded into
#: a channel: the first is supportive care our action space has no rung for,
#: and the second is unclassifiable by construction.
SUBTYPE_CHANNELS = {
    "Chemo": "chemo",
    "Hormone": "endocrine",
    "Targeted": "targeted",
    "Biologic": "biologic",
    "Immuno": "immunotherapy",
    "Bone Treatment": "supportive",
    "Investigational": "investigational",
    "Other": "other",
}

#: Endocrine agents by class. Used only to report what resolution exists, not
#: to assign a rung: the simulator's ``extended`` rung is about *duration*,
#: and duration is what the release records well.
ENDOCRINE_CLASSES = {
    "serm": ("TAMOXIFEN", "TOREMIFENE", "RALOXIFENE"),
    "aromatase_inhibitor": ("LETROZOLE", "ANASTROZOLE", "EXEMESTANE"),
    "serd": ("FULVESTRANT", "ELACESTRANT"),
    "ovarian_suppression": ("LEUPROLIDE", "GOSERELIN", "DEGARELIX"),
    "other": ("MEGESTROL", "BICALUTAMIDE", "ENZALUTAMIDE"),
}

#: Years of endocrine therapy above which the simulator would call the plan
#: ``extended`` rather than ``standard``. Five years is the trial convention
#: (ATLAS, MA.17) and is declared here rather than discovered in the data.
EXTENDED_ENDOCRINE_YEARS = 5.0

#: Chemotherapy-to-surgery gap that a *neoadjuvant course* plausibly spans.
#:
#: POST-HOC (v2.2). Reading the timing axis off first dates alone classified
#: 48.3% of patients as neoadjuvant, outside the pre-specified 15-45% band.
#: The cause is in ``data_timeline_surgery.txt``: its ``PROCEDURE`` rows are
#: every operation a patient ever had - median 2 per patient, 90th-percentile
#: date 3,795 days after diagnosis - not the definitive breast operation. So
#: "chemotherapy before the first recorded operation" also catches systemic
#: therapy for advanced disease followed much later by an unrelated procedure:
#: 27.8% of the neoadjuvant group has a gap over 180 days. This window is the
#: standard four-to-six-cycle interval and is applied as a *secondary*,
#: explicitly post-hoc reading alongside the raw one.
NEOADJUVANT_INTERVAL_DAYS = (30, 180)

#: What each axis of the action space can be read at. ``occurrence`` means the
#: move and its date are recorded; ``level`` means the rung is distinguishable.
AXIS_RESOLUTION = {
    "timing": {"occurrence": True, "level": True,
               "basis": "surgery and chemotherapy start dates, both anchored at diagnosis"},
    "surgery": {"occurrence": True, "level": False,
                "basis": "SUBTYPE is PROCEDURE/SAMPLE only - no BCS vs mastectomy"},
    "chemo": {"occurrence": True, "level": "partial",
              "basis": "agent identity and regimen count observable; dose density is not"},
    "endocrine": {"occurrence": True, "level": "partial",
                  "basis": "cumulative duration separates standard from extended; adherence is not observed"},
    "radiation": {"occurrence": True, "level": False,
                  "basis": "one SUBTYPE value for every row - no site, dose or fractionation"},
}


def within_horizon(frame: pd.DataFrame, horizon_years: float,
                   column: str = "dx_start_date") -> pd.DataFrame:
    """Rows starting in ``[0, horizon_years]`` days from diagnosis.

    The lower bound matters as much as the upper. Treatment recorded at
    negative diagnosis-relative time is either a data-entry artefact or care
    for an earlier cancer; either way it is not a move in this game.
    """
    if horizon_years <= 0:
        raise ValueError("horizon_years must be positive")
    limit = horizon_years * DAYS_PER_YEAR
    return frame[(frame[column] >= 0) & (frame[column] <= limit)]


def treatment_channels(treatment: pd.DataFrame) -> pd.DataFrame:
    """Add a ``channel`` column mapping ``SUBTYPE`` onto simulator channels."""
    annotated = treatment.copy()
    annotated["channel"] = annotated["SUBTYPE"].map(
        lambda value: SUBTYPE_CHANNELS.get(value, "other"))
    return annotated


def endocrine_class(agent: str | float) -> str:
    """Class of one endocrine agent, or ``unclassified``."""
    if not isinstance(agent, str):
        return "unclassified"
    name = agent.strip().upper()
    for label, agents in ENDOCRINE_CLASSES.items():
        if name in agents:
            return label
    return "unclassified"


def axis_coverage(
    treatment: pd.DataFrame,
    surgery: pd.DataFrame,
    radiation: pd.DataFrame,
    cohort: set[str],
    horizon_years: float = 5.0,
) -> pd.DataFrame:
    """Share of ``cohort`` with each axis recorded inside the horizon.

    One row per axis, with the count of patients whose move is visible. The
    denominator is the whole cohort, including patients with no recorded
    treatment at all - a patient who was never treated at MSK is a patient
    whose game we cannot read, and hiding them in the denominator would make
    coverage look like completeness.
    """
    channels = treatment_channels(within_horizon(treatment, horizon_years))
    rows = []

    def _add(axis: str, patients: set[str]) -> None:
        seen = len(patients & cohort)
        rows.append({
            "axis": axis,
            "patients": seen,
            "cohort": len(cohort),
            "share": seen / len(cohort) if cohort else 0.0,
            "level_resolution": str(AXIS_RESOLUTION[axis]["level"]),
            "basis": AXIS_RESOLUTION[axis]["basis"],
        })

    surgical = set(within_horizon(surgery, horizon_years)
                   .loc[lambda f: f["SUBTYPE"] == "PROCEDURE", "PATIENT_ID"])
    chemo = set(channels.loc[channels["channel"] == "chemo", "PATIENT_ID"])
    endocrine = set(channels.loc[channels["channel"] == "endocrine", "PATIENT_ID"])
    radio = set(within_horizon(radiation, horizon_years)["PATIENT_ID"])

    # ``timing`` needs *both* a surgery date and a chemotherapy date: with only
    # one of them there is no ordering to read.
    _add("timing", surgical & chemo)
    _add("surgery", surgical)
    _add("chemo", chemo)
    _add("endocrine", endocrine)
    _add("radiation", radio)
    return pd.DataFrame(rows)


def timing_table(
    treatment: pd.DataFrame,
    surgery: pd.DataFrame,
    horizon_years: float = 5.0,
) -> pd.DataFrame:
    """Neoadjuvant vs surgery-first, per patient, from recorded dates.

    This axis is the one GENIE BPC could not read at all: it has no surgery
    table, so the simulator's ``timing`` decision had no counterpart in
    recorded care. Here both dates exist, so the ordering is a fact rather
    than an assumption.

    Ties (chemotherapy starting the same day as surgery) are counted as
    ``surgery_first``: the release records dates, not times, and same-day
    systemic therapy after an operation is the far commoner event.

    ``plausible_neoadjuvant`` is the post-hoc secondary reading described at
    :data:`NEOADJUVANT_INTERVAL_DAYS`. The raw ``timing`` column stays as the
    pre-specified one so the two can be reported side by side.
    """
    channels = treatment_channels(within_horizon(treatment, horizon_years))
    chemo = (channels[channels["channel"] == "chemo"]
             .groupby("PATIENT_ID")["dx_start_date"].min().rename("first_chemo"))
    procedures = within_horizon(surgery, horizon_years)
    operations = (procedures[procedures["SUBTYPE"] == "PROCEDURE"]
                  .groupby("PATIENT_ID")["dx_start_date"].min().rename("first_surgery"))
    both = pd.concat([chemo, operations], axis=1).dropna()
    both["timing"] = [
        "neoadjuvant" if row.first_chemo < row.first_surgery else "surgery_first"
        for row in both.itertuples()
    ]
    both["days_between"] = both["first_surgery"] - both["first_chemo"]
    low, high = NEOADJUVANT_INTERVAL_DAYS
    both["plausible_neoadjuvant"] = (
        (both["timing"] == "neoadjuvant")
        & both["days_between"].between(low, high))
    return both.reset_index()


def timing_intervals(timing: pd.DataFrame) -> pd.DataFrame:
    """Chemotherapy-to-surgery gaps in bins, with counts.

    The bimodality is the point: one mode 1-90 days *before* chemotherapy
    (adjuvant) and one 90-180 days *after* it (neoadjuvant), with a long tail
    on both sides that belongs to neither. Shipped as a table because it is
    the evidence for the post-hoc interval window, and CLAUDE.md requires the
    figures in a figure to come from a committed table.
    """
    edges = [-float("inf"), -365, -90, -1, 0, 30, 90, 180, 365, float("inf")]
    labels = ["< -365", "-365..-90", "-90..-1", "0", "1..30",
              "31..90", "91..180", "181..365", "> 365"]
    binned = pd.cut(timing["days_between"], bins=edges, labels=labels)
    counts = binned.value_counts().reindex(labels).rename("patients")
    frame = counts.reset_index()
    frame.columns = ["days_surgery_minus_chemo", "patients"]
    frame["share"] = frame["patients"] / len(timing)
    frame["reading"] = [
        "chemo long after surgery", "chemo after surgery",
        "adjuvant (surgery first)", "same day",
        "very short gap", "neoadjuvant (short)", "neoadjuvant (typical)",
        "surgery long after chemo", "surgery much later - not a course",
    ]
    return frame


def endocrine_duration(
    treatment: pd.DataFrame, horizon_years: float = 10.0,
) -> pd.DataFrame:
    """Cumulative endocrine exposure per patient, in years.

    Summed over the **union** of recorded courses rather than over rows:
    tamoxifen and leuprolide given together are one period of endocrine
    therapy, and adding their durations would report ten years of treatment
    for five years of care. The horizon defaults to ten years because the
    question this answers - standard or extended - is about year six onward.
    """
    channels = treatment_channels(within_horizon(treatment, horizon_years))
    endocrine = channels[channels["channel"] == "endocrine"].dropna(
        subset=["dx_start_date", "dx_stop_date"])

    rows = []
    for patient, block in endocrine.groupby("PATIENT_ID"):
        spans = sorted(
            (float(row.dx_start_date), float(row.dx_stop_date))
            for row in block.itertuples())
        merged: list[list[float]] = []
        for start, stop in spans:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], stop)
            else:
                merged.append([start, stop])
        days = sum(stop - start for start, stop in merged)
        classes = {endocrine_class(agent) for agent in block["AGENT"]}
        rows.append({
            "PATIENT_ID": patient,
            "courses": len(block),
            "merged_spans": len(merged),
            "years": days / DAYS_PER_YEAR,
            "extended": days / DAYS_PER_YEAR >= EXTENDED_ENDOCRINE_YEARS,
            "classes": ",".join(sorted(classes)),
        })
    return pd.DataFrame(rows)


def chemo_agent_counts(
    treatment: pd.DataFrame, course_days: int = 180, horizon_years: float = 5.0,
) -> pd.DataFrame:
    """Distinct chemotherapy agents in the first ``course_days`` of treatment.

    A proxy for the simulator's ``standard`` / ``intensified`` rungs, and
    reported as a proxy. Three agents starting together is an AC-T-shaped
    regimen; one is single-agent therapy. What the release cannot show is dose
    density, which is the axis along which "intensified" is actually defined in
    the trials - so this supports a two-rung read at best.
    """
    if course_days <= 0:
        raise ValueError("course_days must be positive")
    channels = treatment_channels(within_horizon(treatment, horizon_years))
    chemo = channels[channels["channel"] == "chemo"]
    first = chemo.groupby("PATIENT_ID")["dx_start_date"].min()
    chemo = chemo.assign(first_start=chemo["PATIENT_ID"].map(first))
    course = chemo[chemo["dx_start_date"] <= chemo["first_start"] + course_days]
    counts = course.groupby("PATIENT_ID")["AGENT"].nunique().rename("agents")
    regimens = chemo.groupby("PATIENT_ID")["dx_start_date"].nunique().rename("start_days")
    return pd.concat([counts, regimens], axis=1).reset_index()
