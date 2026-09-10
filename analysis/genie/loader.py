"""Reading the GENIE BPC Breast Cancer v1.0-public release.

Three things this module exists to hide from the analysis scripts:

1. **Where the files are.** The release is not redistributable, so it lives
   under ``data/`` (git-ignored) and is located by a single constant here.
2. **Encoding.** The GENIE files are UTF-8, but the K-CURE free-box CSVs that
   land in the same directory tree are a mix of ``utf-8-sig`` and ``cp949``.
   :func:`read_csv` sniffs rather than assuming, so the same helper survives
   the move to the Korean tables.
3. **Which columns matter.** The cancer-level table alone is 185 columns wide.
   Naming the handful we use keeps the analysis honest about its inputs and
   makes the manifest meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

#: Release directory. Git-ignored; see ``reports/genie-bpc-profile-v1.6``.
DATA_DIR = ROOT / "data" / "external" / "MCTS_data" / "clinical_data"

#: Encodings tried in order. UTF-8 first so a genuine UTF-8 file is never
#: silently mis-read as cp949 (which decodes almost any byte string).
ENCODINGS = ("utf-8", "utf-8-sig", "cp949")

PATIENT_FILE = "patient_level_dataset.csv"
CANCER_FILE = "cancer_level_dataset_index.csv"
REGIMEN_FILE = "regimen_cancer_level_dataset.csv"
IMAGING_FILE = "imaging_level_dataset.csv"

#: Columns kept from the index-cancer table.
CANCER_COLUMNS = (
    "record_id", "ca_seq", "institution",
    "age_dx", "stage_dx", "bca_subtype",
    "ca_bca_er", "ca_bca_pr", "ca_bca_her_summ",
    "ca_n_regimens",
    "os_dx_status", "tt_os_dx_yrs",
)

#: Columns kept from the regimen table.
REGIMEN_COLUMNS = (
    "record_id", "ca_seq", "regimen_number_within_cancer", "redcap_ca_index",
    "regimen_drugs", "drugs_ct_yn",
    "dx_reg_start_int", "dx_reg_end_any_int",
    "os_g_status", "tt_os_g_yrs",
    "pfs_i_or_m_g_status", "tt_pfs_i_or_m_g_yrs",
)

#: Columns kept from the imaging table.
IMAGING_COLUMNS = (
    "record_id", "scan_number", "dx_scan_days",
    "image_scan_type", "image_ca", "image_overall",
)


def read_csv(path: Path, usecols=None) -> pd.DataFrame:
    """Read a CSV, trying :data:`ENCODINGS` in order.

    Raises the *last* decoding error if every encoding fails, so the message
    names a real problem rather than "cp949 could not decode byte", which is
    what a UTF-8 file would report.
    """
    last: UnicodeDecodeError | None = None
    for encoding in ENCODINGS:
        try:
            return pd.read_csv(
                path, usecols=list(usecols) if usecols else None,
                encoding=encoding, low_memory=False)
        except UnicodeDecodeError as error:  # pragma: no cover - order-dependent
            last = error
    raise last  # type: ignore[misc]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class GenieRelease:
    """The four tables this project reads, plus their hashes."""

    cancers: pd.DataFrame
    regimens: pd.DataFrame
    imaging: pd.DataFrame
    patients: pd.DataFrame
    inputs: dict

    @property
    def n_patients(self) -> int:
        return int(self.cancers["record_id"].nunique())


def load(data_dir: Path = DATA_DIR) -> GenieRelease:
    """Load the release, keeping only the columns named above.

    Index cancers only. ``redcap_ca_index == "No"`` regimens belong to a
    *non-index* second primary; folding them into a patient's treatment
    sequence would splice two different games' movetext together.
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"GENIE BPC release not found at {data_dir}. It is not "
            "redistributable - see reports/genie-bpc-profile-v1.6/README.md.")

    files = {
        "cancers": (data_dir / CANCER_FILE, CANCER_COLUMNS),
        "regimens": (data_dir / REGIMEN_FILE, REGIMEN_COLUMNS),
        "imaging": (data_dir / IMAGING_FILE, IMAGING_COLUMNS),
        "patients": (data_dir / PATIENT_FILE, None),
    }
    frames, inputs = {}, {}
    for key, (path, columns) in files.items():
        frames[key] = read_csv(path, columns)
        inputs[key] = {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(path),
        }

    regimens = frames["regimens"]
    frames["regimens"] = regimens[regimens["redcap_ca_index"] == "Yes"].copy()
    return GenieRelease(inputs=inputs, **frames)
