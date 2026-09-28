"""How good does standard treatment have to be before the guideline wins? (v2.1)

v1.5 found that `configs/dynamic_v0_5.json` declares standard chemotherapy,
standard endocrine therapy and local radiotherapy to have a survival benefit of
**exactly zero** while charging toxicity and burden - and that those three are
what NCCN prescribes and MCTS declines. 41% of the original utility gap was that
asymmetry. Every version since has put "declare the standard-treatment benefit"
at the top of the human-review list, and left it there, because choosing the
numbers is a clinical judgement none of us can make from the data we hold.

This run does not choose them. It turns the benefit into a **dial** and asks the
question that does not need the answer: **how much does the conclusion depend on
it?** ``strength`` runs 0 (today's config, standard treatment is pure cost) to 1
(the randomised-evidence endpoint in ``analysis/dynamic/benefit.py``), with
hazard ratios interpolating geometrically and every ladder keeping the increment
it declares today, so intensified never becomes worse than standard.

A cheap arithmetic preview came first (v2.0's practice): for each patient, the
expected utility of the guideline plan against the same plan with every
declinable standard treatment refused. Averaged over the 40 patients it crosses
at strength ~0.50, and - unlike the salvage channel, where every patient flipped
at the same hazard ratio (v2.0) - **the crossing spreads from 0.20 to 1.00 across
patients**. That is the effect heterogeneity v2.0 said a real decision needs, and
it is already in this channel.

The search is run at seven settings of the dial so the answer is a curve, not a
point. Arm ``s000`` must reproduce v1.5/v1.7/v1.8's baseline gap exactly.

PRE-SPECIFIED PREDICTIONS, recorded after the arithmetic preview and before any
search
--------------------------------------------------
1. **Monotone** - the MCTS-minus-NCCN gap falls with strength at every adjacent
   pair. NCCN prescribes the three standard treatments and MCTS declines them,
   so paying the guideline for them can only close the gap.
2. **Tipping point in [0.40, 0.75]** - the gap crosses zero somewhere in that
   band. The preview's plan-level crossing is 0.50; a searcher should need a
   little *more* benefit than a fixed refusal plan does, because it can keep
   taking whichever channels pay while declining the rest.
3. **NCCN moves more** - from strength 0 to 1, NCCN's utility rises by more than
   MCTS's. The same pattern as v1.4 and v1.5: the gap closes because the
   guideline stops being penalised, not because the searcher gets worse.
4. **The searcher stays flexible** - at strength 1 the gap is negative but
   smaller in magnitude than the preview's mean plan-level advantage (+0.0273),
   because MCTS can accept the channels that pay for a given patient and decline
   the others, which a fixed plan cannot.

Prediction 2 is the one most likely to be wrong and the one that matters: it is
the answer to "how much of the standard-treatment benefit would have to be real
before our headline reverses".
"""

from __future__ import annotations

from dataclasses import replace
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

from analysis.dynamic.benefit import (  # noqa: E402
    FULL_BENEFIT,
    declared_ladder,
    guideline_plan,
    plan_expected_utility,
    refusal_plan,
    with_standard_benefit,
)
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
from analysis.dynamic.experiment_utils import minimum_detectable_difference  # noqa: E402
from analysis.dynamic.policies import CachedMCTSPolicy, DynamicNccnPolicy  # noqa: E402
from analysis.dynamic.schema import patient_from_row  # noqa: E402

INPUT_CSV = ROOT / "data" / "processed" / "patients_with_nccn.csv"
CONFIG_V05 = ROOT / "configs" / "dynamic_v0_5.json"
REPORT_DIR = ROOT / "reports" / "benefit-tipping-v2.1"
TABLE_DIR = REPORT_DIR / "tables"
PRIOR_METRICS = ROOT / "reports" / "channel-decomposition-v1.5" / "metrics.json"

RUN_DATE = "2026-09-26"
PER_SUBTYPE = 5
N_SEEDS = 12
SIMULATIONS = 1024
EPISODES_PER_POLICY = 40
EXPLORATION_WEIGHT = math.sqrt(2.0)
STRENGTHS = (0.0, 0.25, 0.40, 0.50, 0.60, 0.75, 1.00)
PREVIEW_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
TIPPING_BAND = (0.40, 0.75)

ACTION_FIELDS = ("timing", "surgery", "chemo", "endocrine", "radiation")

PRESPECIFIED_PREDICTION = {
    "monotone": (
        "The gap falls with strength at every adjacent pair: NCCN prescribes the "
        "three standard treatments and MCTS declines them."
    ),
    "tipping_point": (
        f"The gap crosses zero in [{TIPPING_BAND[0]}, {TIPPING_BAND[1]}]. The "
        "preview's plan-level crossing is 0.50; a searcher should need a little "
        "more, because it can keep whichever channels pay."
    ),
    "nccn_moves_more": (
        "From strength 0 to 1, NCCN's utility rises by more than MCTS's - the gap "
        "closes because the guideline stops being penalised (v1.4, v1.5 pattern)."
    ),
    "searcher_stays_flexible": (
        "At strength 1 the gap is negative but smaller in magnitude than the "
        "preview's mean plan-level advantage, because MCTS can accept the "
        "channels that pay and decline the rest."
    ),
    "why": (
        "v1.5 found standard treatment declared as pure cost and 41% of the gap "
        "coming from MCTS declining it. Choosing the real numbers is a clinical "
        "judgement; this run asks instead how much the conclusion depends on them."
    ),
}


def config_from_dict(data: dict) -> DynamicConfig:
    config = DynamicConfig(**data)
    _validate_probabilities(config)
    return config


def evaluate(sample, os_model, rfs_model, config) -> dict:
    gaps, mcts_means, nccn_means = [], [], []
    frames = {"MCTS": [], "NCCN": []}
    for seed_index in range(N_SEEDS):
        seed = BASE_SEED + seed_index * 1_000
        mcts_util, nccn_util = [], []
        for _, row in sample.iterrows():
            patient = patient_from_row(row)
            environment = DynamicBreastCancerEnvironment(
                patient, make_risk_table(row, os_model, rfs_model), config)
            offset = int(hashlib.sha256(
                patient.patient_id.encode()).hexdigest()[:8], 16) % 100_000
            patient_seed = seed + offset
            policy = CachedMCTSPolicy(
                environment, simulations=SIMULATIONS,
                exploration_weight=EXPLORATION_WEIGHT, seed=patient_seed)
            md = pd.DataFrame(run_policy_episodes(
                environment, policy, EPISODES_PER_POLICY, patient_seed + 10_000))
            nd = pd.DataFrame(run_policy_episodes(
                environment, DynamicNccnPolicy(environment),
                EPISODES_PER_POLICY, patient_seed + 10_000))
            md["patient_id"] = patient.patient_id
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
    # How often does the searcher now accept each standard treatment?
    mcts = joined["MCTS"]
    accepted = {
        "chemo": float((mcts["chemo"].astype(str) != "none").mean() * 100),
        "endocrine": float((mcts["endocrine"].astype(str) != "none").mean() * 100),
        "radiation": float((mcts["radiation"].astype(str) != "none").mean() * 100),
    }
    return {
        "per_seed_gap": gaps,
        "per_seed_mcts": mcts_means,
        "per_seed_nccn": nccn_means,
        "utility_gap": float(np.mean(gaps)),
        "standard_error": float(np.std(gaps, ddof=1) / math.sqrt(N_SEEDS)),
        "mcts_utility": float(np.mean(mcts_means)),
        "nccn_utility": float(np.mean(nccn_means)),
        "mcts_accept_pct": accepted,
        "action_mix": mix,
    }


def crossing_strength(strengths, gaps) -> float:
    """Linear interpolation of where the gap curve crosses zero."""
    for lower, upper, gap_low, gap_high in zip(strengths, strengths[1:], gaps, gaps[1:]):
        if gap_low > 0.0 >= gap_high:
            span = gap_low - gap_high
            return float(lower + (upper - lower) * (gap_low / span)) if span else float(lower)
    return float("nan")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    raw05 = json.loads(CONFIG_V05.read_text(encoding="utf-8"))

    raw = pd.read_csv(INPUT_CSV)
    os_model, rfs_model, os_test = build_reward_models(raw)
    neutral_model = os_model.neutralise_treatment_terms()
    first = balanced_subtype_sample(os_test, PER_SUBTYPE)
    second = balanced_subtype_sample(os_test, PER_SUBTYPE, offset=PER_SUBTYPE)
    if set(first["patient_id"]) & set(second["patient_id"]):
        raise SystemExit("cohorts A and B are not disjoint")
    sample = pd.concat([first, second], ignore_index=True)

    patients, profiles = {}, {}
    for _, row in sample.iterrows():
        patient = patient_from_row(row)
        risk_table = make_risk_table(row, neutral_model, rfs_model)
        patients[patient.patient_id] = (patient, risk_table)
        # Recorded once at strength 0 so the crossing can be read against what
        # the guideline actually prescribes for this patient.
        environment = DynamicBreastCancerEnvironment(
            patient, risk_table, config_from_dict(raw05))
        plan = guideline_plan(environment)
        treated = [field for field in ("chemo", "endocrine", "radiation")
                   if plan[field] != "none"]
        state = replace(
            environment.initial_state(), phase="followup", timing="surgery_first",
            surgery=plan["surgery"], chemo=plan["chemo"],
            endocrine=plan["endocrine"], radiation=plan["radiation"])
        risk = environment.risk_table[environment.static_plan(state)]
        profiles[patient.patient_id] = {
            "subtype": patient.subtype,
            "stage": int(patient.stage),
            "nccn_plan": "+".join(field[:3] for field in treated) or "none",
            "n_treatments": len(treated),
            "five_year_os": float(risk.five_year_os),
            "five_year_rfs": float(risk.five_year_rfs),
        }

    # --- Part 1: the arithmetic preview -------------------------------------
    preview_rows = []
    for strength in PREVIEW_GRID:
        config = config_from_dict(with_standard_benefit(raw05, float(strength)))
        for patient_id, (patient, risk_table) in patients.items():
            environment = DynamicBreastCancerEnvironment(patient, risk_table, config)
            plan = guideline_plan(environment)
            preview_rows.append({
                "strength": float(strength),
                "patient_id": patient_id,
                "guideline": plan_expected_utility(environment, plan),
                "refusal": plan_expected_utility(environment, refusal_plan(environment, plan)),
            })
    preview = pd.DataFrame(preview_rows)
    preview["advantage"] = preview["guideline"] - preview["refusal"]
    preview.to_csv(TABLE_DIR / "preview_map.csv", index=False)
    preview_summary = preview.groupby("strength").agg(
        mean_advantage=("advantage", "mean"),
        guideline_wins=("advantage", lambda a: float((a > 0).mean()))).reset_index()
    preview_summary.to_csv(TABLE_DIR / "preview_summary.csv", index=False)

    per_patient = []
    for patient_id, block in preview.groupby("patient_id"):
        winning = block[block["advantage"] > 0]
        per_patient.append({
            "patient_id": patient_id,
            "crossing_strength": float(winning["strength"].min()) if len(winning) else math.nan,
        })
    per_patient = pd.DataFrame(per_patient)
    per_patient = per_patient.join(
        pd.DataFrame(profiles).T.rename_axis("patient_id"), on="patient_id")
    per_patient.to_csv(TABLE_DIR / "preview_crossing_by_patient.csv", index=False)
    mechanism = (per_patient.dropna(subset=["crossing_strength"])
                 .groupby("nccn_plan")
                 .agg(patients=("patient_id", "size"),
                      median_crossing=("crossing_strength", "median"),
                      min_crossing=("crossing_strength", "min"),
                      max_crossing=("crossing_strength", "max"),
                      median_five_year_rfs=("five_year_rfs", "median"))
                 .reset_index().sort_values("median_crossing"))
    mechanism.to_csv(TABLE_DIR / "preview_crossing_by_plan.csv", index=False)
    by_subtype = (per_patient.dropna(subset=["crossing_strength"])
                  .groupby("subtype")
                  .agg(patients=("patient_id", "size"),
                       median_crossing=("crossing_strength", "median"))
                  .reset_index().sort_values("median_crossing"))
    by_subtype.to_csv(TABLE_DIR / "preview_crossing_by_subtype.csv", index=False)
    print("\n처방 조합별 교차 강도:", flush=True)
    print(mechanism.to_string(index=False), flush=True)
    print("\n아형별 교차 강도:", flush=True)
    print(by_subtype.to_string(index=False), flush=True)
    crossings = per_patient["crossing_strength"].dropna()
    preview_mean_advantage_at_one = float(
        preview_summary.loc[preview_summary["strength"] == 1.0, "mean_advantage"].iloc[0])

    print(preview_summary.round(4).to_string(index=False), flush=True)
    print(f"\n환자별 교차 강도: 최소 {crossings.min():.2f} · 중앙 {crossings.median():.2f} · "
          f"최대 {crossings.max():.2f} · 퍼짐 {crossings.max() - crossings.min():.2f} "
          f"(교차 없음 {int(per_patient['crossing_strength'].isna().sum())}명)", flush=True)

    ladder_rows = []
    for strength in STRENGTHS:
        for row in declared_ladder(with_standard_benefit(raw05, strength)):
            ladder_rows.append({"strength": strength, **row})
    pd.DataFrame(ladder_rows).to_csv(TABLE_DIR / "declared_ladder.csv", index=False)

    if "--preview-only" in sys.argv:
        # Part 1 is deterministic, so it can be regenerated on its own without
        # re-running an hour of search. Leaves metrics.json and the search
        # tables from the full run untouched.
        print("\n(preview only - Part 2 skipped)", flush=True)
        return

    # --- Part 2: the search at seven settings -------------------------------
    results = {}
    started = time.perf_counter()
    for strength in STRENGTHS:
        key = f"s{int(round(strength * 1000)):03d}"
        config = config_from_dict(with_standard_benefit(raw05, strength))
        print(f"\n[{key}] strength {strength:.2f} running...", flush=True)
        result = evaluate(sample, neutral_model, rfs_model, config)
        result["strength"] = float(strength)
        results[key] = result
        accepted = result["mcts_accept_pct"]
        print(f"  gap {result['utility_gap']:+.4f} (SE {result['standard_error']:.4f})  "
              f"MCTS {result['mcts_utility']:.4f} vs NCCN {result['nccn_utility']:.4f}  "
              f"수용률 항암 {accepted['chemo']:.0f}% · 내분비 {accepted['endocrine']:.0f}% · "
              f"방사선 {accepted['radiation']:.0f}%  ({time.perf_counter() - started:.0f}s)",
              flush=True)

    keys = [f"s{int(round(s * 1000)):03d}" for s in STRENGTHS]
    gaps = [results[key]["utility_gap"] for key in keys]

    def paired(a: str, b: str, field: str = "per_seed_gap") -> dict:
        difference = np.array(results[b][field]) - np.array(results[a][field])
        stderr = float(difference.std(ddof=1) / math.sqrt(N_SEEDS))
        return {
            "difference": float(difference.mean()),
            "standard_error": stderr,
            "z": float(difference.mean() / stderr) if stderr > 0 else float("nan"),
            "seeds_positive": int((difference > 0).sum()),
        }

    arm_table = pd.DataFrame([
        {"arm": key, "strength": results[key]["strength"],
         "utility_gap": results[key]["utility_gap"],
         "standard_error": results[key]["standard_error"],
         "mcts_utility": results[key]["mcts_utility"],
         "nccn_utility": results[key]["nccn_utility"],
         "mcts_accept_chemo_pct": results[key]["mcts_accept_pct"]["chemo"],
         "mcts_accept_endocrine_pct": results[key]["mcts_accept_pct"]["endocrine"],
         "mcts_accept_radiation_pct": results[key]["mcts_accept_pct"]["radiation"]}
        for key in keys])
    arm_table.to_csv(TABLE_DIR / "arms.csv", index=False)
    print("\n" + arm_table.to_string(index=False), flush=True)

    seed_rows = []
    for key in keys:
        for index in range(N_SEEDS):
            seed_rows.append({"arm": key, "strength": results[key]["strength"],
                              "seed_index": index,
                              "utility_gap": results[key]["per_seed_gap"][index],
                              "mcts_utility": results[key]["per_seed_mcts"][index],
                              "nccn_utility": results[key]["per_seed_nccn"][index]})
    pd.DataFrame(seed_rows).to_csv(TABLE_DIR / "per_seed.csv", index=False)

    mix_rows = []
    for key in keys:
        for policy_name, fields in results[key]["action_mix"].items():
            for field, counts in fields.items():
                for action, percent in counts.items():
                    mix_rows.append({"arm": key, "strength": results[key]["strength"],
                                     "policy": policy_name, "field": field,
                                     "action": action, "percent": percent})
    pd.DataFrame(mix_rows).to_csv(TABLE_DIR / "action_mix.csv", index=False)

    prior = json.loads(PRIOR_METRICS.read_text(encoding="utf-8"))
    baseline = prior["verdict"]["gap_baseline"]
    tipping = crossing_strength(list(STRENGTHS), gaps)
    zero_to_one = {
        "mcts": results[keys[-1]]["mcts_utility"] - results[keys[0]]["mcts_utility"],
        "nccn": results[keys[-1]]["nccn_utility"] - results[keys[0]]["nccn_utility"],
    }
    differences = np.diff(gaps)

    verdict = {
        "strengths": list(STRENGTHS),
        "gaps": gaps,
        "gap_at_zero": gaps[0],
        "gap_at_one": gaps[-1],
        "reproduces_v1_5_baseline": bool(abs(gaps[0] - baseline) < 1e-9),
        "v1_5_baseline_gap": baseline,
        "tipping_strength": tipping,
        "tipping_band": list(TIPPING_BAND),
        "implied_hazard_ratios_at_tipping": {
            channel: {outcome: float(value) ** tipping for outcome, value in full.items()}
            for channel, full in FULL_BENEFIT.items()
        } if not math.isnan(tipping) else None,
        "monotone_prediction_met": bool((differences <= 0).all()),
        "max_increase_between_adjacent": float(differences.max()),
        "tipping_prediction_met": bool(
            TIPPING_BAND[0] <= tipping <= TIPPING_BAND[1]) if not math.isnan(tipping) else False,
        "utility_change_zero_to_one": zero_to_one,
        "nccn_moves_more_prediction_met": bool(zero_to_one["nccn"] > zero_to_one["mcts"]),
        "preview_mean_advantage_at_one": preview_mean_advantage_at_one,
        "searcher_flexible_prediction_met": bool(
            gaps[-1] < 0 and abs(gaps[-1]) < preview_mean_advantage_at_one),
        "preview_crossing": {
            "min": float(crossings.min()), "median": float(crossings.median()),
            "max": float(crossings.max()), "spread": float(crossings.max() - crossings.min()),
            "patients_without_crossing": int(per_patient["crossing_strength"].isna().sum()),
            "correlation_with_n_treatments": float(
                per_patient["crossing_strength"].corr(
                    per_patient["n_treatments"].astype(float))),
            "correlation_with_five_year_rfs": float(
                per_patient["crossing_strength"].corr(
                    per_patient["five_year_rfs"].astype(float))),
            "by_plan": mechanism.to_dict(orient="records"),
            "by_subtype": by_subtype.to_dict(orient="records"),
        },
        "mcts_accept_pct_at_zero": results[keys[0]]["mcts_accept_pct"],
        "mcts_accept_pct_at_one": results[keys[-1]]["mcts_accept_pct"],
        "minimum_detectable_difference": minimum_detectable_difference(
            paired(keys[0], keys[-1])["standard_error"]),
        "zero_vs_one_paired": paired(keys[0], keys[-1]),
    }
    if not verdict["reproduces_v1_5_baseline"]:
        raise SystemExit(
            f"strength 0 gap {gaps[0]} does not reproduce v1.5's {baseline}")

    metrics = {
        "run_date": RUN_DATE,
        "analysis_label": "benefit-tipping-v2.1",
        "question": (
            "How much of the randomised standard-treatment benefit would have to be "
            "declared before the MCTS-minus-NCCN gap reverses?"
        ),
        "estimand": (
            "MCTS-minus-NCCN utility gap over the same 40 patients and 12 seeds as "
            "v1.5, under the v0.5 environment with the declared benefit of standard "
            "chemotherapy, standard endocrine therapy and local radiotherapy raised "
            "to a fraction 'strength' of a randomised-evidence endpoint, the ladder "
            "keeping its declared increments. Reported as a curve in strength."
        ),
        "scope_warning": (
            "The endpoint hazard ratios are recorded from memory and flagged for "
            "verification against the primary meta-analyses. The tipping point is "
            "reported both as a fraction of that endpoint and as the implied hazard "
            "ratio, so a corrected endpoint rescales the axis rather than "
            "overturning the shape. Not a clinical effect estimate."
        ),
        "prespecified_prediction": PRESPECIFIED_PREDICTION,
        "design": {
            "patients": int(len(sample)), "seeds": N_SEEDS, "simulations": SIMULATIONS,
            "episodes_per_policy": EPISODES_PER_POLICY,
            "reward_model": "treatment-neutral (v1.4)",
            "config": "configs/dynamic_v0_5.json with the standard-treatment ladders scaled",
            "strengths": list(STRENGTHS),
            "full_benefit_endpoint": FULL_BENEFIT,
            "preview_grid": [float(value) for value in PREVIEW_GRID],
        },
        "arms": {key: {k: v for k, v in results[key].items() if k != "action_mix"}
                 for key in keys},
        "action_mix": {key: results[key]["action_mix"] for key in keys},
        "verdict": verdict,
    }
    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = {
        "run_date": RUN_DATE,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit_before_run": git_commit(),
        "inputs": input_manifest({"data": INPUT_CSV, "assumptions_v05": CONFIG_V05}),
        "entry_point": "analysis/47_run_benefit_tipping.py",
        "base_seed": BASE_SEED,
    }
    (REPORT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== 사전 예측 채점 ===", flush=True)
    for name, met in (
        ("1 단조성", verdict["monotone_prediction_met"]),
        (f"2 전환점이 [{TIPPING_BAND[0]}, {TIPPING_BAND[1]}] 안", verdict["tipping_prediction_met"]),
        ("3 NCCN이 더 많이 오른다", verdict["nccn_moves_more_prediction_met"]),
        ("4 탐색기가 유연하다", verdict["searcher_flexible_prediction_met"]),
    ):
        print(f"  {name}: {'통과' if met else '빗나감'}", flush=True)
    print(f"\n전환점 강도 = {tipping:.3f}", flush=True)
    if verdict["implied_hazard_ratios_at_tipping"]:
        for channel, values in verdict["implied_hazard_ratios_at_tipping"].items():
            print(f"  {channel}: 사망 HR {values['death']:.3f} · 재발 HR {values['recurrence']:.3f}",
                  flush=True)


if __name__ == "__main__":
    main()
