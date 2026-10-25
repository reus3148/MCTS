"""Why the v0.7 gap never crosses zero (v2.6, post hoc, no search).

v2.6's run found no tipping point in the terminal-value game. This script asks,
by arithmetic alone, which part of the action space keeps the gap open. For
every patient and dial setting it scores three static surgery-first plans with
``plan_expected_utility``:

* ``guideline``  - the simplified NCCN plan;
* ``restricted`` - the best plan using only levels NCCN ever plays
  (chemo none/standard, endocrine none/standard, radiation any);
* ``full``       - the best plan over every legal level, which adds the two
  rungs NCCN never plays: intensified chemotherapy and extended endocrine.

``restricted - guideline`` is what a planner gains by choosing *whether* to
treat; ``full - restricted`` is what it gains from rungs only it can reach -
v1.5's asymmetry. Both environments (v0.5 and v0.7) are scored so the two can
be compared on the relative scale.

**Post hoc**: written after the run, on seeing the result. Static plans ignore
neoadjuvant timing, the toxicity guardrail and salvage choice, so the levels
are a preview of the run, not a replacement for it.

Run: ``py analysis/60_static_decomposition_v07.py``
"""

from __future__ import annotations

from itertools import product
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.dynamic.benefit import (  # noqa: E402
    guideline_plan,
    plan_expected_utility,
    with_standard_benefit,
)
from analysis.dynamic.cohort import (  # noqa: E402
    balanced_subtype_sample,
    build_reward_models,
    make_risk_table,
)
from analysis.dynamic.config import DynamicConfig  # noqa: E402
from analysis.dynamic.environment import DynamicBreastCancerEnvironment  # noqa: E402
from analysis.dynamic.schema import patient_from_row  # noqa: E402

INPUT_CSV = ROOT / "data" / "processed" / "patients_with_nccn.csv"
CONFIGS = {"v0.5": ROOT / "configs" / "dynamic_v0_5.json",
           "v0.7": ROOT / "configs" / "dynamic_v0_7.json"}
TABLE_DIR = ROOT / "reports" / "tipping-v07-v2.6" / "tables"
STRENGTHS = (0.0, 0.15, 0.30, 0.445, 0.60, 0.80, 1.00)
PER_SUBTYPE = 5
#: Levels the simplified NCCN policy ever plays (docs/nccn-policy-spec.md).
NCCN_LEVELS = {"chemo": {"none", "standard"}, "endocrine": {"none", "standard"},
               "radiation": {"none", "local", "regional"}}


def legal_levels(environment: DynamicBreastCancerEnvironment, surgery: str) -> dict:
    patient = environment.patient
    chemo = ("none",) if patient.stage == 0 else ("none", "standard", "intensified")
    endocrine = ("none", "standard", "extended") if patient.hr_positive else ("none",)
    radiation = ("none", "local", "regional") if surgery == "BCS" else ("none", "regional")
    return {"chemo": chemo, "endocrine": endocrine, "radiation": radiation}


def best_plan(environment, surgery: str, allowed: dict) -> tuple[float, dict]:
    best_value, best = float("-inf"), None
    for chemo, endocrine, radiation in product(
            allowed["chemo"], allowed["endocrine"], allowed["radiation"]):
        plan = {"surgery": surgery, "chemo": chemo,
                "endocrine": endocrine, "radiation": radiation}
        value = plan_expected_utility(environment, plan)
        if value > best_value:
            best_value, best = value, plan
    return best_value, best


def main() -> None:
    raw = pd.read_csv(INPUT_CSV)
    os_model, rfs_model, os_test = build_reward_models(raw)
    neutral = os_model.neutralise_treatment_terms()
    sample = pd.concat([balanced_subtype_sample(os_test, PER_SUBTYPE),
                        balanced_subtype_sample(os_test, PER_SUBTYPE, offset=PER_SUBTYPE)],
                       ignore_index=True)

    rows = []
    for label, path in CONFIGS.items():
        base = json.loads(path.read_text(encoding="utf-8"))
        for strength in STRENGTHS:
            config = DynamicConfig(**with_standard_benefit(base, strength))
            for _, row in sample.iterrows():
                patient = patient_from_row(row)
                environment = DynamicBreastCancerEnvironment(
                    patient, make_risk_table(row, neutral, rfs_model), config)
                guide = guideline_plan(environment)
                legal = legal_levels(environment, guide["surgery"])
                restricted = {f: tuple(a for a in legal[f] if a in NCCN_LEVELS[f])
                              for f in legal}
                g = plan_expected_utility(environment, guide)
                r, _ = best_plan(environment, guide["surgery"], restricted)
                full, plan = best_plan(environment, guide["surgery"], legal)
                rows.append({
                    "environment": label, "strength": strength,
                    "guideline": g, "restricted": r, "full": full,
                    "uses_intensified": plan["chemo"] == "intensified",
                    "uses_extended": plan["endocrine"] == "extended",
                })

    frame = pd.DataFrame(rows)
    summary = (frame.groupby(["environment", "strength"])
               .agg(guideline=("guideline", "mean"), restricted=("restricted", "mean"),
                    full=("full", "mean"),
                    intensified_pct=("uses_intensified", "mean"),
                    extended_pct=("uses_extended", "mean"))
               .reset_index())
    summary["choice_gain"] = summary["restricted"] - summary["guideline"]
    summary["exclusive_rung_gain"] = summary["full"] - summary["restricted"]
    summary["static_gap"] = summary["full"] - summary["guideline"]
    for column in ("choice_gain", "exclusive_rung_gain", "static_gap"):
        summary[f"{column}_rel"] = summary[column] / summary["guideline"]
    summary["intensified_pct"] *= 100
    summary["extended_pct"] *= 100
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLE_DIR / "static_decomposition.csv"
    summary.round(6).to_csv(out, index=False)
    print(summary[["environment", "strength", "choice_gain_rel",
                   "exclusive_rung_gain_rel", "static_gap_rel",
                   "intensified_pct", "extended_pct"]].round(4).to_string(index=False))
    print(out)


if __name__ == "__main__":
    main()
