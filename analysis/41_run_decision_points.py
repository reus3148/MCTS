"""Does giving the policy a second place to adapt create a closed-loop share? (v1.8)

v1.7 took the response channel away from the searching policy and the gap did
not move: -0.0004, z = -0.29, with a minimum detectable difference of 0.0036
(18% of the gap). Its explanation was structural. A response is drawn only when
the policy chose neoadjuvant chemotherapy - 20.8% of episodes - and after that
nothing the patient does or suffers ever opens another decision. v1.6 had just
shown that recorded care is the opposite: a "Progressing" scan is followed by a
new regimen 71.3% of the time, and a 5-year game runs 2 to 9 moves.

So v0.6 opens one more decision, at the event that opens one in real care: a
**recurrence** with follow-up years still to play moves the episode into a
``salvage`` phase with two actions, ``none`` and ``systemic``. Both policies see
both. Guideline care after recurrence is systemic therapy, so NCCN always
treats; MCTS may decline. Declared parameters (synthetic, flagged for clinical
review): systemic salvage multiplies post-recurrence death hazard by 0.85, with
toxicity probability 0.20 and burden 0.10.

Those magnitudes were chosen so the decision is *genuinely patient-dependent*:
the cost (about 0.017 in normalised utility) is of the same order as the benefit
of a 15% hazard reduction over the one to three years a recurrence leaves inside
the horizon. A decision that is always right or always wrong would not be a
place to adapt.

Four arms on the same 40 patients, 12 seeds and treatment-neutral reward model
as v1.5 and v1.7:

=======  ==========  ======================================
arm      environment MCTS at the salvage decision
=======  ==========  ======================================
A        v0.5        (phase does not exist)
B        v0.6 null   decides, but the actions do nothing
C        v0.6        decides
D        v0.6        defers to the guideline rule (systemic)
=======  ==========  ======================================

Arm D plans on ``FixedSalvageEnvironment`` - the salvage phase has one legal
action - so the planner never assumes a choice it will not get. C minus D is
the value of the new decision point to a searching policy.

Two equalities are asserted rather than hoped for: arm A reproduces v1.5's and
v1.7's baseline gap exactly, and NCCN's utility is identical in arms C and D
(same real environment, same seeds; only MCTS's planner changes).

PRE-SPECIFIED PREDICTIONS, recorded before the run
--------------------------------------------------
1. **Null control** - |B - A| < 0.005 (seed-paired). The phase with its teeth
   removed still costs the planner a decision and one random draw per
   recurrence; if that alone moves the gap, nothing else here can be read.
2. **Primary** - C - D > 0. A policy that may decline salvage where it does not
   pay cannot do worse in expectation than one that must always accept it.
3. **Magnitude** - C - D >= 0.0036, v1.7's minimum detectable difference. This
   is the claim that matters: that a decision point where real care adapts
   turns the closed-loop share from undetectable into detectable.
4. **Horizon artefact** - MCTS declines salvage more often when the recurrence
   falls in the last decidable year (year 4) than earlier. A five-year horizon
   makes late treatment look worthless to any expected-utility planner whether
   or not it would be worthless to the patient. This prediction is expected to
   hold, and if it does, part of whatever prediction 3 finds is an artefact of
   where the game ends rather than closed-loop skill.

Prediction 3 is the one most likely to fail: with the chosen magnitudes most
recurrences may still make treating the obvious choice, leaving little for a
searcher to gain by deciding. Prediction 1 is the one that has to hold.
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
)
from analysis.dynamic.schema import patient_from_row  # noqa: E402

INPUT_CSV = ROOT / "data" / "processed" / "patients_with_nccn.csv"
CONFIG_V05 = ROOT / "configs" / "dynamic_v0_5.json"
CONFIG_V06 = ROOT / "configs" / "dynamic_v0_6.json"
REPORT_DIR = ROOT / "reports" / "decision-points-v1.8"
TABLE_DIR = REPORT_DIR / "tables"
PRIOR_METRICS = ROOT / "reports" / "closed-loop-value-v1.7" / "metrics.json"

RUN_DATE = "2026-09-13"
PER_SUBTYPE = 5                 # v1.2's cohorts A and B: 40 patients
N_SEEDS = 12
SIMULATIONS = 1024
EPISODES_PER_POLICY = 40
EXPLORATION_WEIGHT = math.sqrt(2.0)
NULL_CONTROL_TOLERANCE = 0.005
LAST_DECIDABLE_YEAR = 4         # a recurrence in year 5 has nothing left to treat

ARMS = (
    ("v05", "A. v0.5 환경 (구제 결정 없음)"),
    ("inert", "B. v0.6 영대조 (구제 단계는 있으나 효과 없음)"),
    ("decide", "C. v0.6 — MCTS가 구제를 결정"),
    ("defer", "D. v0.6 — MCTS가 구제를 가이드라인 규칙에 위임"),
)

PRESPECIFIED_PREDICTION = {
    "null_control": (
        f"|B - A| < {NULL_CONTROL_TOLERANCE}. The phase with no effect still "
        "costs a decision and a random draw; if that alone moves the gap, "
        "nothing else can be read."
    ),
    "primary": (
        "C - D > 0: a policy that may decline salvage where it does not pay "
        "cannot do worse in expectation than one that must always accept it."
    ),
    "magnitude": (
        "C - D >= 0.0036, v1.7's minimum detectable difference - a decision "
        "point where real care adapts turns the closed-loop share from "
        "undetectable into detectable."
    ),
    "horizon_artefact": (
        f"MCTS declines salvage more often for recurrences in year "
        f"{LAST_DECIDABLE_YEAR} than earlier. Expected to hold; if it does, part "
        "of prediction 3 is an artefact of where the game ends."
    ),
    "why": (
        "v1.7 found no closed-loop share and blamed an environment that offers "
        "one chance to adapt in 20.8% of episodes. v1.6 showed real care adapts "
        "at progression. v0.6 opens a decision at recurrence to test whether the "
        "share appears once there is somewhere to adapt."
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
        "mcts_mean_adaptation_opportunities": float(opportunities.mean()),
        "action_mix": mix,
        "salvage_by_year": salvage_by_recurrence_year(joined["MCTS"]),
    }


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    raw05 = json.loads(CONFIG_V05.read_text(encoding="utf-8"))
    raw06 = json.loads(CONFIG_V06.read_text(encoding="utf-8"))
    configs = {
        "v05": config_from_dict(raw05),
        # "inert", not "null": pandas reads the string "null" back as NaN.
        "inert": config_from_dict(null_control(raw06)),
        "decide": config_from_dict(raw06),
        "defer": config_from_dict(raw06),
    }

    raw = pd.read_csv(INPUT_CSV)
    os_model, rfs_model, os_test = build_reward_models(raw)
    # v1.4: every arm runs on the treatment-neutral reward model.
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
              f"{result['mcts_salvage_decisions']} · 적응 기회 있음 "
              f"{result['mcts_episodes_with_adaptation_pct']:.1f}%  "
              f"({time.perf_counter() - started:.0f}s)", flush=True)

    def paired(a: str, b: str, key: str = "per_seed_gap") -> dict:
        difference = np.array(results[b][key]) - np.array(results[a][key])
        stderr = float(difference.std(ddof=1) / math.sqrt(N_SEEDS))
        return {
            "difference": float(difference.mean()),
            "standard_error": stderr,
            "z": float(difference.mean() / stderr) if stderr > 0 else float("nan"),
            "seeds_positive": int((difference > 0).sum()),
        }

    prior = json.loads(PRIOR_METRICS.read_text(encoding="utf-8"))
    v17_baseline = prior["verdict"]["gap_none"]
    v17_mde = prior["verdict"]["minimum_detectable_difference"]
    baseline_gap = results["v05"]["utility_gap"]

    by_year = results["decide"]["salvage_by_year"]
    by_year.to_csv(TABLE_DIR / "salvage_by_recurrence_year.csv", index=False)
    late = by_year[by_year["recurrence_year"] == LAST_DECIDABLE_YEAR]
    early = by_year[by_year["recurrence_year"] < LAST_DECIDABLE_YEAR]
    late_decline = float(late["declined_pct"].iloc[0]) if len(late) else float("nan")
    early_decline = float(
        (early["declined_pct"] * early["decisions"]).sum() / early["decisions"].sum()
    ) if len(early) and early["decisions"].sum() else float("nan")

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
            "mcts_mean_adaptation_opportunities":
                results[key]["mcts_mean_adaptation_opportunities"],
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
    verdict = {
        "gap_v05": baseline_gap,
        "gap_inert": results["inert"]["utility_gap"],
        "gap_decide": results["decide"]["utility_gap"],
        "gap_defer": results["defer"]["utility_gap"],
        "inert_vs_v05": paired("v05", "inert"),
        "decide_vs_v05": paired("v05", "decide"),
        "defer_vs_v05": paired("v05", "defer"),
        "decide_vs_defer": decide_vs_defer,
        # What always treating costs the guideline policy: NCCN in C minus NCCN
        # in B, seed-paired. Negative means the declared salvage is a net loss
        # inside the horizon - the v1.5 asymmetry in a new channel.
        "nccn_forced_salvage_cost": paired("inert", "decide", key="per_seed_nccn"),
        "nccn_phase_only_effect": paired("v05", "inert", key="per_seed_nccn"),
        "mcts_decide_vs_inert": paired("inert", "decide", key="per_seed_mcts"),
        "value_of_decision_point": decide_vs_defer["difference"],
        "minimum_detectable_difference": minimum_detectable_difference(
            decide_vs_defer["standard_error"]),
        "null_control_passed": bool(
            abs(results["inert"]["utility_gap"] - baseline_gap) < NULL_CONTROL_TOLERANCE),
        "null_control_tolerance": NULL_CONTROL_TOLERANCE,
        "primary_prediction_met": bool(decide_vs_defer["difference"] > 0),
        "magnitude_prediction_met": bool(decide_vs_defer["difference"] >= v17_mde),
        "magnitude_threshold_from_v1_7": v17_mde,
        "salvage_decline_pct_last_year": late_decline,
        "salvage_decline_pct_earlier_years": early_decline,
        "horizon_artefact_prediction_met": bool(late_decline > early_decline)
        if not (math.isnan(late_decline) or math.isnan(early_decline)) else None,
        "adaptation_pct_v05": results["v05"]["mcts_episodes_with_adaptation_pct"],
        "adaptation_pct_decide": results["decide"]["mcts_episodes_with_adaptation_pct"],
        "mcts_salvage_mix_decide": results["decide"]["action_mix"]["MCTS"].get("salvage", {}),
        "nccn_salvage_mix_decide": results["decide"]["action_mix"]["NCCN"].get("salvage", {}),
        "reproduces_v1_7_baseline": bool(abs(baseline_gap - v17_baseline) < 1e-9),
        "v1_7_baseline_gap": v17_baseline,
        "nccn_identical_decide_vs_defer": bool(
            abs(results["decide"]["nccn_utility"] - results["defer"]["nccn_utility"]) < 1e-12),
    }

    if not verdict["reproduces_v1_7_baseline"]:
        raise SystemExit(
            f"arm A gap {baseline_gap} does not reproduce v1.7's {v17_baseline}; "
            "the v0.6 changes altered v0.5 behaviour")
    if not verdict["nccn_identical_decide_vs_defer"]:
        raise SystemExit("NCCN utility differs between arms C and D; it must not")

    metrics = {
        "run_date": RUN_DATE,
        "analysis_label": "decision-points-v1.8",
        "question": (
            "When the environment opens a second decision - at recurrence - does "
            "a searching policy gain a detectable closed-loop share from it?"
        ),
        "estimand": (
            "MCTS-minus-NCCN utility gap over the same 40 patients and 12 seeds "
            "as v1.5/v1.7 under four arms: the v0.5 environment, the v0.6 "
            "environment with an inert salvage phase, and v0.6 with MCTS "
            "deciding versus deferring the salvage action. C minus D is the "
            "value of the decision point to the searcher."
        ),
        "scope_warning": (
            "Salvage benefit, toxicity and burden are declared synthetic "
            "parameters pending clinical review. A property of our simulator, "
            "not a clinical effect."
        ),
        "prespecified_prediction": PRESPECIFIED_PREDICTION,
        "design": {
            "patients": int(len(sample)),
            "seeds": N_SEEDS,
            "simulations": SIMULATIONS,
            "episodes_per_policy": EPISODES_PER_POLICY,
            "reward_model": "treatment-neutral (v1.4)",
            "configs": {"v05": "configs/dynamic_v0_5.json",
                        "v06": "configs/dynamic_v0_6.json"},
            "declared_salvage": {
                "hazard_multipliers": raw06["hazard_multipliers"]["salvage"],
                "acute_toxicity_probabilities": raw06["acute_toxicity_probabilities"]["salvage"],
                "treatment_burden": raw06["treatment_burden"]["salvage"],
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
        "inputs": input_manifest({"data": INPUT_CSV, "assumptions_v05": CONFIG_V05,
                                  "assumptions_v06": CONFIG_V06}),
        "entry_point": "analysis/41_run_decision_points.py",
        "base_seed": BASE_SEED,
    }
    (REPORT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== 사전 예측 채점 ===", flush=True)
    for name, met in (
        ("1 영대조 (|B-A| < 0.005)", verdict["null_control_passed"]),
        ("2 주 예측 (C-D > 0)", verdict["primary_prediction_met"]),
        ("3 크기 (C-D >= v1.7 MDE 0.0036)", verdict["magnitude_prediction_met"]),
        ("4 지평 인공물 (마지막 해 거절이 더 많다)", verdict["horizon_artefact_prediction_met"]),
    ):
        print(f"  {name}: {'통과' if met else '빗나감' if met is not None else '판정 불가'}",
              flush=True)
    print(f"\n결정 지점의 값 (C-D): {verdict['value_of_decision_point']:+.4f} "
          f"(z = {decide_vs_defer['z']:.2f}, MDE {verdict['minimum_detectable_difference']:.4f})",
          flush=True)


if __name__ == "__main__":
    main()
