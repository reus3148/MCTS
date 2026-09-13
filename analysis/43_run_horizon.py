"""Does crediting life after the horizon fix what v1.8 found? (v1.9)

v1.8 opened a decision at recurrence and watched it collapse: with a five-year
horizon that counts nothing after year five, the searcher declined salvage 62%
of the time in year 1 and 100% in year 4, and the guideline policy - which must
always treat - lost utility for it (-0.0010, z = -13.7). The declared salvage
parameters were net-negative *inside the horizon*, and v1.8 argued that no
reasonable parameters could escape that as long as a late recurrence has almost
no years left to benefit. The fix has to be to the horizon, not to the numbers.

GENIE BPC says how much the cliff was cutting off: overall survival from the
first advanced-disease diagnosis has a median of 3.9 years, with 40% alive at
five years and 18% at ten (894 index cancers). A recurrence at year 4 of our
game leaves, in expectation, most of its life outside the game.

v0.7 credits **ten years of passive follow-up** at the end of the decision
horizon as a terminal value: the expected discounted reward of continuing under
the state's own hazards with no further decisions. Same function for both
policies; no random draws; zero when the tail is zero, so v0.2-v0.6 are
untouched. **Nothing else changes** - the salvage parameters are exactly v0.6's,
so anything that moves is the tail's doing.

Four arms on the same 40 patients, 12 seeds and treatment-neutral reward model
as v1.5, v1.7 and v1.8:

=======  ===================  ======================================
arm      environment          MCTS at the salvage decision
=======  ===================  ======================================
A        v0.7, no salvage     (phase does not exist)
B        v0.7 + inert salvage decides, but the actions do nothing
C        v0.7 + salvage       decides
D        v0.7 + salvage       defers to the guideline rule (systemic)
=======  ===================  ======================================

Arm A of v1.8 (v0.5, no tail) is read from its committed metrics for the
seed-paired comparison in prediction 5; the seeds and patients are identical.

PRE-SPECIFIED PREDICTIONS, recorded before the run
--------------------------------------------------
1. **Null control** - |B - A| < 0.005 (seed-paired), as in v1.8.
2. **The horizon artefact goes away** - MCTS's salvage decline rate in year 4
   exceeds its year-1 rate by less than 10 percentage points (v1.8: +37.8).
   With ten years credited past the horizon, a year-4 recurrence has as much
   to gain from treatment as a year-1 one.
3. **The guideline stops losing on the new channel** - NCCN's utility in C is
   not below its utility in B by more than 0.0005 (v1.8: -0.0010, z = -13.7).
   Salvage's 15% hazard reduction now runs for eleven years, not one to four.
4. **Value of the decision point** - C - D > 0. Direction only; the size is
   the measurement, reported with its minimum detectable difference.
5. **The tail widens the baseline gap** - A (v0.7) minus v1.8's arm A (v0.5)
   is positive. The channels that carry a declared benefit - intensified chemo,
   extended endocrine, regional radiation - are used only by MCTS (v1.5), and
   the tail extends their benefit past year five. This is a fairness probe: if
   it holds, the terminal value amplifies an asymmetry we already know about,
   and the gap under v0.7 must not be quoted as if it were the same number.

Prediction 4 is the one most likely to fail: with salvage net-positive, both
policies may simply always treat, leaving the decision worth little. Prediction
3 is the one that says whether the horizon was the problem.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.dynamic.cohort import (  # noqa: E402
    BASE_SEED,
    balanced_subtype_sample,
    build_reward_models,
    git_commit,
    input_manifest,
    make_risk_table,
)
from analysis.dynamic.config import DynamicConfig, _validate_probabilities  # noqa: E402
from analysis.dynamic.environment import DynamicBreastCancerEnvironment  # noqa: E402
from analysis.dynamic.evaluation import run_policy_episodes  # noqa: E402
from analysis.dynamic.experiment_utils import (  # noqa: E402
    minimum_detectable_difference,
)
from analysis.dynamic.policies import CachedMCTSPolicy, DynamicNccnPolicy  # noqa: E402
from analysis.dynamic.salvage import (  # noqa: E402
    FixedSalvageEnvironment,
    adaptation_opportunities,
    null_control,
    salvage_by_recurrence_year,
    without_salvage,
)
from analysis.dynamic.schema import patient_from_row  # noqa: E402
from analysis.genie.loader import DATA_DIR as GENIE_DIR, load as load_genie  # noqa: E402
from analysis.genie.survival import post_advanced_survival  # noqa: E402

INPUT_CSV = ROOT / "data" / "processed" / "patients_with_nccn.csv"
CONFIG_V07 = ROOT / "configs" / "dynamic_v0_7.json"
REPORT_DIR = ROOT / "reports" / "horizon-v1.9"
TABLE_DIR = REPORT_DIR / "tables"
PRIOR_METRICS = ROOT / "reports" / "decision-points-v1.8" / "metrics.json"

RUN_DATE = "2026-09-13"
PER_SUBTYPE = 5                 # v1.2's cohorts A and B: 40 patients
N_SEEDS = 12
SIMULATIONS = 1024
EPISODES_PER_POLICY = 40
EXPLORATION_WEIGHT = math.sqrt(2.0)
NULL_CONTROL_TOLERANCE = 0.005
ARTEFACT_TOLERANCE_PP = 10.0    # year-4 minus year-1 decline rate, percentage points
NCCN_LOSS_TOLERANCE = 0.0005
LAST_DECIDABLE_YEAR = 4

ARMS = (
    ("tail", "A. v0.7 말기 가치만 (구제 결정 없음)"),
    ("inert", "B. v0.7 + 영대조 구제 (효과 없음)"),
    ("decide", "C. v0.7 + 구제 — MCTS가 결정"),
    ("defer", "D. v0.7 + 구제 — 규칙에 위임"),
)

PRESPECIFIED_PREDICTION = {
    "null_control": f"|B - A| < {NULL_CONTROL_TOLERANCE}, seed-paired, as in v1.8.",
    "artefact_gone": (
        f"MCTS's salvage decline rate in year {LAST_DECIDABLE_YEAR} exceeds its "
        f"year-1 rate by less than {ARTEFACT_TOLERANCE_PP:.0f} percentage points "
        "(v1.8: +37.8)."
    ),
    "nccn_no_longer_loses": (
        f"NCCN utility in C is not below B by more than {NCCN_LOSS_TOLERANCE} "
        "(v1.8: -0.0010, z = -13.7)."
    ),
    "decision_point": "C - D > 0, direction only; size reported with its MDE.",
    "baseline_widens": (
        "A (v0.7) minus v1.8's arm A (v0.5) is positive: the tail extends the "
        "benefit of channels only MCTS uses. A fairness probe."
    ),
    "why": (
        "v1.8 found that a five-year cliff makes late treatment worthless to any "
        "expected-utility planner. v0.7 credits ten passive years past the horizon, "
        "sized against GENIE BPC survival after advanced disease, and changes "
        "nothing else."
    ),
}

ACTION_FIELDS = ("timing", "surgery", "chemo", "endocrine", "radiation", "salvage")


def config_from_dict(data: dict) -> DynamicConfig:
    config = DynamicConfig(**data)
    _validate_probabilities(config)
    return config


def evaluate(sample, os_model, rfs_model, config, defer: bool) -> dict:
    gaps, mcts_means, nccn_means = [], [], []
    frames = {"MCTS": [], "NCCN": []}
    for seed_index in range(N_SEEDS):
        seed = BASE_SEED + seed_index * 1_000
        mcts_util, nccn_util = [], []
        for _, row in sample.iterrows():
            patient = patient_from_row(row)
            environment = DynamicBreastCancerEnvironment(
                patient, make_risk_table(row, os_model, rfs_model), config)
            planner = FixedSalvageEnvironment(environment) if defer else environment
            offset = int(hashlib.sha256(
                patient.patient_id.encode()).hexdigest()[:8], 16) % 100_000
            patient_seed = seed + offset
            policy = CachedMCTSPolicy(
                planner, simulations=SIMULATIONS,
                exploration_weight=EXPLORATION_WEIGHT, seed=patient_seed)
            md = pd.DataFrame(run_policy_episodes(
                environment, policy, EPISODES_PER_POLICY, patient_seed + 10_000))
            nd = pd.DataFrame(run_policy_episodes(
                environment, DynamicNccnPolicy(environment),
                EPISODES_PER_POLICY, patient_seed + 10_000))
            mcts_util.append(md["utility"].mean())
            nccn_util.append(nd["utility"].mean())
            frames["MCTS"].append(md)
            frames["NCCN"].append(nd)
        gaps.append(float(np.mean(mcts_util) - np.mean(nccn_util)))
        mcts_means.append(float(np.mean(mcts_util)))
        nccn_means.append(float(np.mean(nccn_util)))

    joined = {name: pd.concat(parts, ignore_index=True) for name, parts in frames.items()}
    mix = {
        name: {
            field: frame[field].astype(str).value_counts(normalize=True)
            .mul(100).round(3).to_dict()
            for field in ACTION_FIELDS if field in frame.columns
        }
        for name, frame in joined.items()
    }
    opportunities = adaptation_opportunities(joined["MCTS"])
    return {
        "per_seed_gap": gaps,
        "per_seed_mcts": mcts_means,
        "per_seed_nccn": nccn_means,
        "utility_gap": float(np.mean(gaps)),
        "standard_error": float(np.std(gaps, ddof=1) / math.sqrt(N_SEEDS)),
        "mcts_utility": float(np.mean(mcts_means)),
        "nccn_utility": float(np.mean(nccn_means)),
        "mcts_recurrence_pct": float(joined["MCTS"]["recurred_by_5y"].mean() * 100),
        "mcts_salvage_decisions": int(joined["MCTS"]["salvage"].notna().sum()),
        "mcts_episodes_with_adaptation_pct": float((opportunities > 0).mean() * 100),
        "action_mix": mix,
        "salvage_by_year": salvage_by_recurrence_year(joined["MCTS"]),
    }


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    raw07 = json.loads(CONFIG_V07.read_text(encoding="utf-8"))
    configs = {
        "tail": config_from_dict(without_salvage(raw07)),
        "inert": config_from_dict(null_control(raw07)),
        "decide": config_from_dict(raw07),
        "defer": config_from_dict(raw07),
    }

    # Grounding for the tail length: how long a recurrence leaves in real care.
    genie = load_genie()
    survival = post_advanced_survival(genie.cancers)
    pd.DataFrame({
        "years": list(survival["survival"].keys()),
        "survival": list(survival["survival"].values()),
    }).to_csv(TABLE_DIR / "genie_post_advanced_survival.csv", index=False)
    print(f"GENIE BPC 진행 후 생존: n={survival['n']} · 중앙 {survival['median_years']:.2f}년 · "
          + " · ".join(f"S({y})={s:.2f}" for y, s in survival["survival"].items()),
          flush=True)

    raw = pd.read_csv(INPUT_CSV)
    os_model, rfs_model, os_test = build_reward_models(raw)
    neutral_model = os_model.neutralise_treatment_terms()

    first = balanced_subtype_sample(os_test, PER_SUBTYPE)
    second = balanced_subtype_sample(os_test, PER_SUBTYPE, offset=PER_SUBTYPE)
    if set(first["patient_id"]) & set(second["patient_id"]):
        raise SystemExit("cohorts A and B are not disjoint")
    sample = pd.concat([first, second], ignore_index=True)

    results = {}
    started = time.perf_counter()
    for key, label in ARMS:
        print(f"\n[{key}] running...", flush=True)
        result = evaluate(sample, neutral_model, rfs_model, configs[key],
                          defer=(key == "defer"))
        result["label"] = label
        results[key] = result
        print(f"  gap {result['utility_gap']:+.4f} (SE {result['standard_error']:.4f})  "
              f"MCTS {result['mcts_utility']:.4f} vs NCCN {result['nccn_utility']:.4f}  "
              f"재발 {result['mcts_recurrence_pct']:.1f}% · 구제 결정 "
              f"{result['mcts_salvage_decisions']} · 적응 기회 "
              f"{result['mcts_episodes_with_adaptation_pct']:.1f}%  "
              f"({time.perf_counter() - started:.0f}s)", flush=True)

    def paired_series(a: np.ndarray, b: np.ndarray) -> dict:
        difference = np.asarray(b) - np.asarray(a)
        stderr = float(difference.std(ddof=1) / math.sqrt(len(difference)))
        return {
            "difference": float(difference.mean()),
            "standard_error": stderr,
            "z": float(difference.mean() / stderr) if stderr > 0 else float("nan"),
            "seeds_positive": int((difference > 0).sum()),
        }

    def paired(a: str, b: str, key: str = "per_seed_gap") -> dict:
        return paired_series(results[a][key], results[b][key])

    prior = json.loads(PRIOR_METRICS.read_text(encoding="utf-8"))
    v05 = prior["arms"]["v05"]
    v18 = prior["verdict"]

    by_year = results["decide"]["salvage_by_year"]
    by_year.to_csv(TABLE_DIR / "salvage_by_recurrence_year.csv", index=False)
    indexed = by_year.set_index("recurrence_year")["declined_pct"] if len(by_year) else pd.Series(dtype=float)
    decline_y1 = float(indexed.get(1.0, float("nan")))
    decline_y4 = float(indexed.get(float(LAST_DECIDABLE_YEAR), float("nan")))

    arm_table = pd.DataFrame([
        {
            "arm": key,
            "label": results[key]["label"],
            "utility_gap": results[key]["utility_gap"],
            "standard_error": results[key]["standard_error"],
            "mcts_utility": results[key]["mcts_utility"],
            "nccn_utility": results[key]["nccn_utility"],
            "mcts_recurrence_pct": results[key]["mcts_recurrence_pct"],
            "mcts_salvage_decisions": results[key]["mcts_salvage_decisions"],
            "mcts_episodes_with_adaptation_pct":
                results[key]["mcts_episodes_with_adaptation_pct"],
        }
        for key, _ in ARMS
    ])
    arm_table.to_csv(TABLE_DIR / "arms.csv", index=False)
    print("\n" + arm_table.to_string(index=False), flush=True)
    print("\n구제 결정, 재발 연도별 (C팔):", flush=True)
    print(by_year.to_string(index=False), flush=True)

    seed_rows = []
    for key, _ in ARMS:
        for index in range(N_SEEDS):
            seed_rows.append({
                "arm": key, "seed_index": index,
                "utility_gap": results[key]["per_seed_gap"][index],
                "mcts_utility": results[key]["per_seed_mcts"][index],
                "nccn_utility": results[key]["per_seed_nccn"][index],
            })
    pd.DataFrame(seed_rows).to_csv(TABLE_DIR / "per_seed.csv", index=False)

    mix_rows = []
    for key, _ in ARMS:
        for policy_name, fields in results[key]["action_mix"].items():
            for field, counts in fields.items():
                for action, percent in counts.items():
                    mix_rows.append({"arm": key, "policy": policy_name,
                                     "field": field, "action": action,
                                     "percent": percent})
    pd.DataFrame(mix_rows).to_csv(TABLE_DIR / "action_mix.csv", index=False)

    decide_vs_defer = paired("defer", "decide")
    nccn_forced = paired("inert", "decide", key="per_seed_nccn")
    tail_vs_v05 = paired_series(v05["per_seed_gap"], results["tail"]["per_seed_gap"])
    verdict = {
        "gap_tail": results["tail"]["utility_gap"],
        "gap_inert": results["inert"]["utility_gap"],
        "gap_decide": results["decide"]["utility_gap"],
        "gap_defer": results["defer"]["utility_gap"],
        "gap_v05_from_v1_8": v05["utility_gap"],
        "inert_vs_tail": paired("tail", "inert"),
        "decide_vs_defer": decide_vs_defer,
        "value_of_decision_point": decide_vs_defer["difference"],
        "minimum_detectable_difference": minimum_detectable_difference(
            decide_vs_defer["standard_error"]),
        "nccn_forced_salvage_cost": nccn_forced,
        "nccn_forced_salvage_cost_v1_8": v18["nccn_forced_salvage_cost"],
        "mcts_decide_vs_inert": paired("inert", "decide", key="per_seed_mcts"),
        "tail_vs_v05": tail_vs_v05,
        "nccn_tail_vs_v05": paired_series(v05["per_seed_nccn"], results["tail"]["per_seed_nccn"]),
        "mcts_tail_vs_v05": paired_series(v05["per_seed_mcts"], results["tail"]["per_seed_mcts"]),
        "salvage_decline_pct_year1": decline_y1,
        "salvage_decline_pct_year4": decline_y4,
        "salvage_decline_pct_year1_v1_8": 62.155388,
        "salvage_decline_pct_year4_v1_8": 100.0,
        "mcts_salvage_mix_decide": results["decide"]["action_mix"]["MCTS"].get("salvage", {}),
        "adaptation_pct_decide": results["decide"]["mcts_episodes_with_adaptation_pct"],
        "genie_post_advanced_survival": survival,
        "null_control_passed": bool(
            abs(results["inert"]["utility_gap"] - results["tail"]["utility_gap"]) < NULL_CONTROL_TOLERANCE),
        "artefact_gone_prediction_met": bool((decline_y4 - decline_y1) < ARTEFACT_TOLERANCE_PP)
        if not (math.isnan(decline_y1) or math.isnan(decline_y4)) else None,
        "nccn_no_longer_loses_prediction_met": bool(nccn_forced["difference"] > -NCCN_LOSS_TOLERANCE),
        "decision_point_prediction_met": bool(decide_vs_defer["difference"] > 0),
        "baseline_widens_prediction_met": bool(tail_vs_v05["difference"] > 0),
        "nccn_identical_decide_vs_defer": bool(
            abs(results["decide"]["nccn_utility"] - results["defer"]["nccn_utility"]) < 1e-12),
    }
    if not verdict["nccn_identical_decide_vs_defer"]:
        raise SystemExit("NCCN utility differs between arms C and D; it must not")

    metrics = {
        "run_date": RUN_DATE,
        "analysis_label": "horizon-v1.9",
        "question": (
            "Does crediting life after the decision horizon remove the artefact "
            "that made v1.8's salvage decision collapse, and does a decision "
            "point then carry a detectable closed-loop share?"
        ),
        "estimand": (
            "MCTS-minus-NCCN utility gap over the same 40 patients and 12 seeds "
            "as v1.5-v1.8 under the v0.7 environment (ten passive years credited "
            "past the horizon), in four arms: no salvage, inert salvage, MCTS "
            "deciding salvage, MCTS deferring salvage. C minus D is the value of "
            "the decision point; A minus v1.8's v0.5 arm is what the tail alone does."
        ),
        "scope_warning": (
            "The tail length and the salvage parameters are declared synthetic "
            "assumptions. A property of our simulator, not a clinical effect."
        ),
        "prespecified_prediction": PRESPECIFIED_PREDICTION,
        "design": {
            "patients": int(len(sample)),
            "seeds": N_SEEDS,
            "simulations": SIMULATIONS,
            "episodes_per_policy": EPISODES_PER_POLICY,
            "reward_model": "treatment-neutral (v1.4)",
            "config": "configs/dynamic_v0_7.json",
            "terminal_tail_years": raw07["terminal_tail_years"],
            "declared_salvage": {
                "hazard_multipliers": raw07["hazard_multipliers"]["salvage"],
                "acute_toxicity_probabilities": raw07["acute_toxicity_probabilities"]["salvage"],
                "treatment_burden": raw07["treatment_burden"]["salvage"],
            },
            "arms": [{"arm": key, "label": label} for key, label in ARMS],
        },
        "arms": {
            key: {k: v for k, v in results[key].items()
                  if k not in ("action_mix", "salvage_by_year")}
            for key, _ in ARMS
        },
        "action_mix": {key: results[key]["action_mix"] for key, _ in ARMS},
        "verdict": verdict,
    }
    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = {
        "run_date": RUN_DATE,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit_before_run": git_commit(),
        "inputs": {
            **input_manifest({"data": INPUT_CSV, "assumptions_v07": CONFIG_V07}),
            "genie_cancers": genie.inputs["cancers"],
        },
        "genie_data_dir": str(GENIE_DIR.relative_to(ROOT)).replace("\\", "/"),
        "entry_point": "analysis/43_run_horizon.py",
        "base_seed": BASE_SEED,
    }
    (REPORT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== 사전 예측 채점 ===", flush=True)
    for name, met in (
        ("1 영대조 (|B-A| < 0.005)", verdict["null_control_passed"]),
        ("2 지평 인공물 소멸 (4년차-1년차 거절 < 10%p)", verdict["artefact_gone_prediction_met"]),
        ("3 NCCN이 더 이상 손해 보지 않음 (> -0.0005)", verdict["nccn_no_longer_loses_prediction_met"]),
        ("4 결정 지점의 값 (C-D > 0)", verdict["decision_point_prediction_met"]),
        ("5 말기 가치가 기준선 격차를 넓힘 (A - v0.5 > 0)", verdict["baseline_widens_prediction_met"]),
    ):
        state = "통과" if met else ("빗나감" if met is not None else "판정 불가")
        print(f"  {name}: {state}", flush=True)
    print(f"\n결정 지점의 값 (C-D): {verdict['value_of_decision_point']:+.4f} "
          f"(z = {decide_vs_defer['z']:.2f}, MDE {verdict['minimum_detectable_difference']:.4f})",
          flush=True)
    print(f"NCCN 강제 구제의 대가: {nccn_forced['difference']:+.4f} (z = {nccn_forced['z']:.2f}); "
          f"v1.8: {v18['nccn_forced_salvage_cost']['difference']:+.4f}", flush=True)
    print(f"거절률 1년차 {decline_y1:.1f}% → 4년차 {decline_y4:.1f}% (v1.8: 62.2% → 100%)",
          flush=True)
    print(f"말기 가치만으로 격차 변화: {tail_vs_v05['difference']:+.4f} (z = {tail_vs_v05['z']:.2f})",
          flush=True)


if __name__ == "__main__":
    main()
