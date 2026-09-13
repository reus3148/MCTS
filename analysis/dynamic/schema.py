"""Dataset-neutral patient and environment state schema.

Only the adapter knows METABRIC column names. A future K-CURE adapter can map
its columns to the same ``PatientProfile`` without changing the environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd


STANDARD_PATIENT_FIELDS = (
    "patient_id",
    "age",
    "menopause",
    "tumor_size_mm",
    "lymph_pos",
    "stage",
    "grade",
    "subtype",
    "er",
    "pr",
    "her2",
)

METABRIC_COLUMN_MAP = {field: field for field in STANDARD_PATIENT_FIELDS}


@dataclass(frozen=True)
class PatientProfile:
    patient_id: str
    age: float
    menopause: str
    tumor_size_mm: float
    lymph_pos: int
    stage: int
    grade: int
    subtype: str
    er: int
    pr: int
    her2: int

    @property
    def hr_positive(self) -> bool:
        return self.er == 1 or self.pr == 1

    @property
    def her2_positive(self) -> bool:
        return self.her2 == 1


def patient_from_row(
    row: pd.Series,
    column_map: Mapping[str, str] = METABRIC_COLUMN_MAP,
) -> PatientProfile:
    missing_mappings = set(STANDARD_PATIENT_FIELDS) - set(column_map)
    if missing_mappings:
        raise ValueError(f"missing patient field mappings: {missing_mappings}")
    values = {field: row[column_map[field]] for field in STANDARD_PATIENT_FIELDS}
    missing_values = [field for field, value in values.items() if pd.isna(value)]
    if missing_values:
        raise ValueError(f"patient has missing required values: {missing_values}")
    return PatientProfile(
        patient_id=str(values["patient_id"]),
        age=float(values["age"]),
        menopause=str(values["menopause"]),
        tumor_size_mm=float(values["tumor_size_mm"]),
        lymph_pos=int(values["lymph_pos"]),
        stage=int(values["stage"]),
        grade=int(values["grade"]),
        subtype=str(values["subtype"]),
        er=int(values["er"]),
        pr=int(values["pr"]),
        her2=int(values["her2"]),
    )


@dataclass(frozen=True)
class RiskEstimate:
    five_year_os: float
    five_year_rfs: float


@dataclass(frozen=True)
class DynamicState:
    phase: str
    current_tumor_size_mm: float
    timing: str | None = None
    surgery: str | None = None
    chemo: str | None = None
    endocrine: str | None = None
    radiation: str | None = None
    response: str = "not_applicable"
    toxicity_count: int = 0
    year: int = 0
    alive: bool = True
    recurred: bool = False
    #: v0.6 (reports/decision-points-v1.8): the treatment chosen at the salvage
    #: decision a recurrence opens, and the year the recurrence happened.
    #: ``repr=False`` on purpose - ``CachedMCTSPolicy`` seeds each search from
    #: ``repr(state)``, so hiding the new fields from the repr keeps every
    #: pre-v0.6 state's seed byte-identical and v0.2-v1.7 reproducible. Equality
    #: and hashing still include them, so the decision cache stays correct.
    salvage: str | None = field(default=None, repr=False)
    recurrence_year: int | None = field(default=None, repr=False)

