"""Turning GENIE BPC rows into something the chess framing can read.

The project started from one observation: a chess position leading to a move
leading to a new position is structurally the same object as a patient state
leading to a treatment leading to a new state. METABRIC gave us only the
*result* of each game and the pieces on the board at move one. It had no
movetext, which is why the environment from v0.2 onward had to invent the
moves and score them from declared parameters.

This module reads the movetext. Two pieces:

``regimen_channels`` / ``primary_channel``
    What kind of move was played. Drug names collapse onto the channels the
    simulator already reasons about (chemo / endocrine / HER2-directed / ...),
    so a real sequence can be compared with a simulated one.

``response_switch_table``
    Whether the player looked at the board before moving. Every imaging report
    carries a radiologist's overall assessment, and every regimen carries a
    start day. If "Progressing" is followed by a new regimen far more often
    than "Stable" is, then the recorded games contain closed-loop adaptation -
    the thing a sequential policy is supposed to be good at, and the thing
    v1.5 could not separate from noise in the simulator.

Everything here is a pure function over data frames so it can be tested
without the release present.
"""

from __future__ import annotations

import pandas as pd

DAYS_PER_YEAR = 365.25

#: Drug -> channel. Built from the 83 distinct agents in the v1.0-public
#: release; anything unrecognised falls to ``other`` rather than being guessed.
DRUG_CHANNELS: dict[str, str] = {}


def _register(channel: str, *drugs: str) -> None:
    for drug in drugs:
        DRUG_CHANNELS[drug.strip().lower()] = channel


_register(
    "endocrine",
    "Tamoxifen", "Toremifene", "Letrozole", "Anastrozole", "Exemestane",
    "Fulvestrant", "Leuprolide", "Goserlin Acetate", "Megestrol Acetate",
    "Bicalutamide", "Other hormone",
)
_register(
    "chemotherapy",
    "Paclitaxel", "Nabpaclitaxel", "Docetaxel", "Ixabepilone",
    "Cyclophosphamide", "Ifosfamide", "Doxorubicin HCL", "Epirubicin HCL",
    "Pegylated Liposomal Doxorubicin", "Capecitabine", "Fluorouracil",
    "Leucovorin", "Methotrexate", "Gemcitabine HCL", "Carboplatin",
    "Cisplatin", "Oxaliplatin", "Eribulin Mesylate", "Vinorelbine Tartrate",
    "Irinotecan HCL", "Etoposide", "Dacarbazine", "Temozolomide",
    "Bendamustine",
)
_register(
    "her2_targeted",
    "Trastuzumab", "Trastuzumab/Hyaluronidase-oysk", "Pertuzumab",
    "Trastuzumab Emtansine", "Trastuzumab Deruxtecan", "Lapatinib Ditosylate",
    "Neratinib", "Tucatinib",
)
_register(
    "targeted_other",
    "Palbociclib", "Abemaciclib", "Ribociclib", "Everolimus", "Alpelisib",
    "Taselisib", "Olaparib", "Talazoparib", "Bevacizumab", "Ramucirumab",
    "Pazopanib HCL", "Sorafenib Tosylate", "Lenvatinib Mesylate", "Apatinib",
    "Ponatinib HCL", "Olaratumab", "Sacituzumab Govitecan",
)
_register(
    "immunotherapy",
    "Pembrolizumab", "Nivolumab", "Atezolizumab", "Ipilimumab", "BCG Solution",
)
_register("investigational", "Investigational Drug")
_register(
    "other",
    "Other NOS", "Other antineoplastic", "Rituximab",
    "Yttrium Y90 Ibritumomab Tiuxetan",
)

#: Precedence when a regimen mixes channels. ``investigational`` outranks
#: everything because a blinded agent makes the whole regimen unclassifiable,
#: and pretending otherwise would let unknown drugs be counted as endocrine.
CHANNEL_PRECEDENCE = (
    "investigational", "her2_targeted", "chemotherapy", "targeted_other",
    "immunotherapy", "endocrine", "other",
)

#: Radiologist assessments, grouped for the response question.
PROGRESSING = "Progressing/Worsening/Enlarging"
CONTROLLED = ("Stable/No change", "Improving/Responding")


def regimen_channels(regimen_drugs: str | float) -> tuple[str, ...]:
    """Channels present in one regimen, sorted, without duplicates.

    ``regimen_drugs`` is GENIE's comma-joined drug list. Names carry stray
    trailing spaces in the release ("Tamoxifen ", "Docetaxel "), so they are
    stripped and lower-cased before lookup.
    """
    if not isinstance(regimen_drugs, str) or not regimen_drugs.strip():
        return ()
    found = {
        DRUG_CHANNELS.get(part.strip().lower(), "other")
        for part in regimen_drugs.split(",") if part.strip()
    }
    return tuple(sorted(found))


def primary_channel(channels: tuple[str, ...]) -> str:
    """Collapse a regimen's channels to one label using :data:`CHANNEL_PRECEDENCE`."""
    for channel in CHANNEL_PRECEDENCE:
        if channel in channels:
            return channel
    return "unclassified"


def annotate_regimens(regimens: pd.DataFrame) -> pd.DataFrame:
    """Add ``channels``, ``primary_channel`` and ``endocrine_only`` columns."""
    annotated = regimens.copy()
    annotated["channels"] = annotated["regimen_drugs"].map(regimen_channels)
    annotated["primary_channel"] = annotated["channels"].map(primary_channel)
    annotated["endocrine_only"] = annotated["channels"].map(
        lambda channels: channels == ("endocrine",))
    return annotated


def sequence_lengths(regimens: pd.DataFrame, horizon_years: float) -> pd.Series:
    """Regimens started within ``horizon_years`` of diagnosis, per index cancer.

    Cancers with no regimen inside the horizon are absent from the result, not
    zero: this counts the length of the games that were played, and a policy
    comparison that never gets a move is a different question.
    """
    if horizon_years <= 0:
        raise ValueError("horizon_years must be positive")
    limit = horizon_years * DAYS_PER_YEAR
    inside = regimens[regimens["dx_reg_start_int"] <= limit]
    return inside.groupby(["record_id", "ca_seq"]).size().rename("n_regimens")


def response_switch_table(
    imaging: pd.DataFrame,
    regimens: pd.DataFrame,
    followup_days: pd.Series,
    window_days: int = 90,
) -> pd.DataFrame:
    """Did a new regimen start soon after this scan - and had one just before?

    One row per scan that (a) carries an overall assessment, (b) happens on or
    after the patient's first regimen start, and (c) has ``window_days`` of
    follow-up left. Condition (c) is administrative censoring keyed on
    ``followup_days`` (days from diagnosis to death or last contact); without
    it, scans near the end of follow-up would count as "no switch" for the
    trivial reason that nothing more was recorded.

    ``switch_after`` is the closed-loop signal. ``switch_before`` is the
    placebo-in-time control: the same question asked of the window *preceding*
    the scan, which the scan cannot have caused.
    """
    if window_days <= 0:
        raise ValueError("window_days must be positive")

    starts = (regimens.dropna(subset=["dx_reg_start_int"])
              .groupby("record_id")["dx_reg_start_int"]
              .apply(lambda values: sorted(float(value) for value in values)))
    first_start = starts.map(lambda values: values[0])

    scans = imaging.dropna(subset=["image_overall", "dx_scan_days"]).copy()
    scans = scans[scans["record_id"].isin(starts.index)]
    scans["first_start"] = scans["record_id"].map(first_start)
    scans["followup_days"] = scans["record_id"].map(followup_days)
    scans = scans[
        (scans["dx_scan_days"] >= scans["first_start"])
        & (scans["dx_scan_days"] + window_days <= scans["followup_days"])
    ].copy()

    def _any_start(record_id: str, low: float, high: float) -> bool:
        """Regimen starting in the half-open interval ``(low, high]``."""
        return any(low < value <= high for value in starts[record_id])

    scans["switch_after"] = [
        _any_start(row.record_id, row.dx_scan_days,
                   row.dx_scan_days + window_days)
        for row in scans.itertuples()
    ]
    scans["switch_before"] = [
        _any_start(row.record_id, row.dx_scan_days - window_days,
                   row.dx_scan_days)
        for row in scans.itertuples()
    ]
    scans["assessment"] = scans["image_overall"].where(
        scans["image_overall"].isin([PROGRESSING, *CONTROLLED]), "other")
    scans["group"] = scans["assessment"].map(
        lambda value: "progressing" if value == PROGRESSING
        else ("controlled" if value in CONTROLLED else "other"))
    return scans[[
        "record_id", "scan_number", "dx_scan_days", "image_scan_type",
        "image_overall", "assessment", "group", "switch_after",
        "switch_before",
    ]]


def switch_rates(table: pd.DataFrame) -> pd.DataFrame:
    """Switch rates by assessment group, with counts so n is never hidden."""
    grouped = table.groupby("group")
    summary = pd.DataFrame({
        "scans": grouped.size(),
        "switch_after": grouped["switch_after"].mean(),
        "switch_before": grouped["switch_before"].mean(),
    })
    summary["forward_minus_backward"] = (
        summary["switch_after"] - summary["switch_before"])
    return summary.reset_index()
