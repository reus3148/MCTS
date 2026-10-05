"""Reading the MSK-CHORD (MSK, Nature 2024) cBioPortal release.

Four things this module hides from the analysis scripts.

1. **Two file formats in one directory.** The ``data_clinical_*`` tables carry
   four comment lines (display name, description, datatype, priority) before
   the real header; the ``data_timeline_*`` tables are plain TSV. Getting this
   wrong does not raise - it silently makes the display-name row the header.
2. **Where the files are.** The release is CC BY-NC-ND 4.0, so it lives under
   ``data/external`` (git-ignored) and is located by one constant.
3. **The time origin.** Every timeline date is *days relative to the sequencing
   date*, not to diagnosis. Our simulator and the GENIE adapter are both
   anchored at diagnosis, so :func:`anchor_at_diagnosis` re-bases them. The
   median breast-cancer diagnosis sits 424 days *before* sequencing, so
   skipping this step would place adjuvant treatment at negative time and
   drop it from every horizon.
4. **Which columns matter.** The sample table is 24 columns and the patient
   table 26; naming the handful we use keeps the manifest meaningful.

Encoding is handled by :mod:`analysis.genie.loader`'s sniffing reader, reused
here rather than reimplemented - MSK-CHORD is UTF-8, but the convention is
that every adapter in this project goes through the same door (CLAUDE.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from analysis.genie.loader import ENCODINGS, sha256  # noqa: F401  (re-exported)

ROOT = Path(__file__).resolve().parent.parent.parent

#: Release directory. Git-ignored; see ``reports/msk-chord-profile-v2.2``.
DATA_DIR = ROOT / "data" / "external" / "msk_chord_2024"

#: Comment lines before the header in the ``data_clinical_*`` tables.
CLINICAL_HEADER_OFFSET = 4

#: The cancer type we keep. MSK-CHORD pools six; ours is breast.
BREAST = "Breast Cancer"

PATIENT_FILE = "data_clinical_patient.txt"
SAMPLE_FILE = "data_clinical_sample.txt"

#: ``data_timeline_<key>.txt`` files this project reads. The remaining
#: timelines in the release (PSA, Gleason, CEA, CA 19-9, MMR, PD-L1, specimen,
#: prior_meds, tumor_sites, cna) are either other cancers' markers or not used
#: by the eligibility assessment; adding one here is all it takes to load it.
TIMELINES = (
    "diagnosis",
    "treatment",
    "surgery",
    "radiation",
    "performance_status",
    "progression",
    "cancer_presence",
)

PATIENT_COLUMNS = (
    "PATIENT_ID", "GENDER", "CURRENT_AGE_DEID",
    "STAGE_HIGHEST_RECORDED", "HR", "HER2",
    "OS_MONTHS", "OS_STATUS", "PRIOR_MED_TO_MSK",
)

SAMPLE_COLUMNS = (
    "SAMPLE_ID", "PATIENT_ID", "CANCER_TYPE", "CANCER_TYPE_DETAILED",
    "SAMPLE_TYPE", "METASTATIC_SITE", "PRIMARY_SITE",
)


def read_clinical(path: Path, usecols=None) -> pd.DataFrame:
    """Read a ``data_clinical_*`` table, skipping its four comment lines."""
    return _read(path, usecols=usecols, skiprows=CLINICAL_HEADER_OFFSET)


def read_timeline(path: Path, usecols=None) -> pd.DataFrame:
    """Read a ``data_timeline_*`` table (plain TSV, header on line 1)."""
    return _read(path, usecols=usecols, skiprows=0)


def _read(path: Path, usecols, skiprows: int) -> pd.DataFrame:
    last: UnicodeDecodeError | None = None
    for encoding in ENCODINGS:
        try:
            return pd.read_csv(
                path, sep="\t", skiprows=skiprows,
                usecols=list(usecols) if usecols else None,
                encoding=encoding, low_memory=False)
        except UnicodeDecodeError as error:  # pragma: no cover - order-dependent
            last = error
    raise last  # type: ignore[misc]


def diagnosis_anchor(diagnosis: pd.DataFrame) -> pd.Series:
    """Day of the earliest recorded primary diagnosis, per patient.

    MSK-CHORD records 5,424 primary-diagnosis rows for 5,366 breast patients:
    a handful carry two tumour-registry entries. We take the **earliest**,
    matching the GENIE adapter's index-cancer rule - the first game, not the
    second primary spliced onto it.
    """
    primary = diagnosis[diagnosis["SUBTYPE"] == "Primary"]
    return (primary.dropna(subset=["START_DATE"])
            .groupby("PATIENT_ID")["START_DATE"].min().rename("dx_day"))


def anchor_at_diagnosis(
    frame: pd.DataFrame, anchor: pd.Series,
    columns: tuple[str, ...] = ("START_DATE", "STOP_DATE"),
) -> pd.DataFrame:
    """Re-base ``columns`` from days-since-sequencing to days-since-diagnosis.

    Rows for patients without an anchor are dropped rather than kept at an
    unknown offset: a move with no known time cannot be placed in a horizon,
    and silently treating sequencing day 0 as diagnosis would shift every such
    patient's whole sequence by over a year.
    """
    known = frame[frame["PATIENT_ID"].isin(anchor.index)].copy()
    offset = known["PATIENT_ID"].map(anchor)
    for column in columns:
        if column in known:
            known[f"dx_{column.lower()}"] = known[column] - offset
    return known


@dataclass(frozen=True)
class MskChordRelease:
    """The breast-cancer slice of the release, plus input hashes.

    ``timelines`` are already re-anchored at diagnosis: each carries
    ``dx_start_date`` (and ``dx_stop_date`` where the source has one) in days
    from diagnosis, alongside the original sequencing-relative columns.
    """

    patients: pd.DataFrame
    samples: pd.DataFrame
    timelines: dict[str, pd.DataFrame]
    anchor: pd.Series
    inputs: dict

    @property
    def n_patients(self) -> int:
        return int(self.patients["PATIENT_ID"].nunique())

    def timeline(self, key: str) -> pd.DataFrame:
        return self.timelines[key]


def load(data_dir: Path = DATA_DIR, cancer_type: str = BREAST) -> MskChordRelease:
    """Load the ``cancer_type`` slice, re-anchored at diagnosis.

    The cohort is defined by the *sample* table's ``CANCER_TYPE``, because that
    is where MSK-CHORD records the OncoTree assignment. Every patient in the
    release has at least one sequenced sample, which is the release's central
    selection: see the report's identification section.
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"MSK-CHORD release not found at {data_dir}. It is CC BY-NC-ND 4.0 "
            "and not redistributable - see reports/msk-chord-profile-v2.2/README.md.")

    inputs: dict[str, dict] = {}

    def _record(key: str, path: Path) -> None:
        inputs[key] = {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(path),
        }

    sample_path = data_dir / SAMPLE_FILE
    patient_path = data_dir / PATIENT_FILE
    samples = read_clinical(sample_path, SAMPLE_COLUMNS)
    patients = read_clinical(patient_path, PATIENT_COLUMNS)
    _record("samples", sample_path)
    _record("patients", patient_path)

    samples = samples[samples["CANCER_TYPE"] == cancer_type].copy()
    cohort = set(samples["PATIENT_ID"])
    patients = patients[patients["PATIENT_ID"].isin(cohort)].copy()

    raw: dict[str, pd.DataFrame] = {}
    for key in TIMELINES:
        path = data_dir / f"data_timeline_{key}.txt"
        frame = read_timeline(path)
        _record(key, path)
        raw[key] = frame[frame["PATIENT_ID"].isin(cohort)].copy()

    anchor = diagnosis_anchor(raw["diagnosis"])
    timelines = {key: anchor_at_diagnosis(frame, anchor)
                 for key, frame in raw.items()}
    return MskChordRelease(
        patients=patients, samples=samples, timelines=timelines,
        anchor=anchor, inputs=inputs)
