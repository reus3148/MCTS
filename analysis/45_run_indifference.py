"""Is there anywhere for the salvage decision to split - and does the searcher find it? (v2.0)

v1.8: with a five-year cliff the searcher declined salvage 85% of the time.
v1.9: with ten years of terminal value it accepted 98.7% of the time. A decision
everyone answers the same way is not a decision, and the value of a decision
point cannot be measured where there is nothing to decide.

Before spending search time, this run asks the environment's own arithmetic
(``analysis/dynamic/indifference.py``): for every one of the 40 patients, every
recurrence year, and every salvage hazard ratio from 0.80 to 1.00, is treating
worth more than declining in expectation? That map is deterministic and instant,
and it answers two questions the experiment depends on:

1. **Where is the indifference point?** The ratio at which about half the
   (patient, year) cells favour treating.
2. **Does it differ between patients?** If every patient flips at the same
   ratio the decision is global - the searcher gains nothing from being able
   to make it, whatever the ratio.

Then the searcher is run at that ratio, with two purposes: a known-answer check
(does MCTS's accept rate by recurrence year track the map's?) and the value of
the decision point C - D, now at the one place it could be non-trivial.

Arms (v0.7 environment, salvage death ratio set to the indifference point):

=======  =====================================  ===========
arm      MCTS at the salvage decision           seeds
=======  =====================================  ===========
C        decides                                24
D        defers to the guideline rule           24
=======  =====================================  ===========

Twenty-four seeds rather than twelve: v1.9's tail doubled the noise floor and
its minimum detectable difference was 27% of the gap. The first twelve seeds are
the same as every run since v1.5, so v1.9's arms A (tail only) and B (inert
salvage) pair with them for the fairness diagnostic.

PRE-SPECIFIED PREDICTIONS, recorded before the run
--------------------------------------------------
Written after computing the analytic map (it *is* the design input) but before
running any search.

1. **Known answer** - at the indifference ratio, MCTS's salvage accept rate in
   each recurrence year is within 15 percentage points of the map's treat share
   for that year. The searcher, with 1024 simulations, should recover what the
   expectation says.
2. **Year-dependence** - MCTS accepts salvage less often for year-4 recurrences
   than for year-1 recurrences, by at least 20 percentage points. At the
   indifference ratio the map says which side wins depends on the years left,
   and that is a state the policy can see.
3. **The decision point is worth little even here** - C - D is positive but
   below 10% of arm C's gap. At an indifference point the two options are, by
   construction, nearly equal, so choosing correctly is worth little. This is
   the claim that matters: a decision point carries value only where the
   *stakes* are large and *which option wins* varies with observable state.
4. **Near-neutral for the guideline** - NCCN's utility in C differs from v1.9's
   inert arm by less than 5% of the gap (seed-paired over the first twelve
   seeds). Forced treatment near indifference should cost or gain the guideline
   almost nothing.

Prediction 1 is the one most likely to fail - the searcher is noisy at exactly
the states where the options are closest. Prediction 3 is the one that decides
what to build next.
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
from analysis.dynamic.indifference import (  # noqa: E402
    indifference_map,
    indifference_ratio_per_patient,
    pick_indifference_ratio,
    treat_share_by_ratio,
)
from analysis.dynamic.policies import CachedMCTSPolicy, DynamicNccnPolicy  # noqa: E402
from analysis.dynamic.salvage import (  # noqa: E402
    FixedSalvageEnvironment,
    adaptation_opportunities,
    salvage_by_recurrence_year,
)
from analysis.dynamic.schema import patient_from_row  # noqa: E402

INPUT_CSV = ROOT / "data" / "processed" / "patients_with_nccn.csv"
CONFIG_V07 = ROOT / "configs" / "dynamic_v0_7.json"
REPORT_DIR = ROOT / "reports" / "indifference-v2.0"
TABLE_DIR = REPORT_DIR / "tables"
PRIOR_METRICS = ROOT / "reports" / "horizon-v1.9" / "metrics.json"

RUN_DATE = "2026-09-16"
PER_SUBTYPE = 5
N_SEEDS = 24
PRIOR_SEEDS = 12
SIMULATIONS = 1024
EPISODES_PER_POLICY = 40
EXPLORATION_WEIGHT = math.sqrt(2.0)
RATIOS = np.round(np.arange(0.80, 1.0001, 0.01), 2)
YEARS = (1, 2, 3, 4)
KNOWN_ANSWER_TOLERANCE_PP = 15.0
YEAR_DEPENDENCE_FLOOR_PP = 20.0
DECISION_VALUE_CEILING = 0.10        # relative to arm C's gap
NCCN_NEUTRAL_TOLERANCE = 0.05        # relative to arm C's gap

ARMS = (
    ("decide", "C. v0.7 @ 무차별 위험비 — MCTS가 결정"),
    ("defer", "D. v0.7 @ 무차별 위험비 — 규칙에 위임"),
)

PRESPECIFIED_PREDICTION = {
    "known_answer": (
        f"At the indifference ratio, MCTS's accept rate per recurrence year is within "
        f"{KNOWN_ANSWER_TOLERANCE_PP:.0f} pp of the analytic treat share for that year."
    ),
    "year_dependence": (
        f"MCTS accepts salvage at least {YEAR_DEPENDENCE_FLOOR_PP:.0f} pp less often for "
        "year-4 than for year-1 recurrences."
    ),
    "decision_value_small": (
        f"C - D > 0 but below {DECISION_VALUE_CEILING:.0%} of arm C's gap: at an "
        "indifference point the options are nearly equal by construction."
    ),
    "nccn_neutral": (
        f"|NCCN(C) - NCCN(v1.9 inert)| < {NCCN_NEUTRAL_TOLERANCE:.0%} of arm C's gap, "
        "seed-paired over the first twelve seeds."
    ),
    "why": (
        "v1.8 and v1.9 found the salvage decision collapsing to one answer in "
        "opposite directions. The analytic map locates the only ratio where it "
        "splits; the search is run there to check the searcher against a known "
        "answer and to measure what a decision point is worth at indifference."
    ),
}

ACTION_FIELDS = ("timing", "surgery", "chemo", "endocrine", "radiation", "salvage")


def config_from_dict(data: dict) -> DynamicConfig:
    config = DynamicConfig(**data)
    _validate_probabilities(config)
    return config


def nccn_followup_state(environment: DynamicBreastCancerEnvironment):
    """A follow-up state carrying the simplified NCCN plan for this patient."""
    plan = DynamicNccnPolicy(environment).plan
    surgery = str(plan[0])
    return replace(
        environment.initial_state(), phase="followup", timing="surgery_first",
        surgery=surgery,
        chemo="standard" if int(plan[1]) else "none",
        endocrine="standard" if int(plan[2]) else "none",
        radiation=("local" if surgery == "BCS" else "regional") if int(plan[3]) else "none",
    )


def evaluate(sample, os_model, rfs_model, config, defer: bool, seeds: int) -> dict:
    gaps, mcts_means, nccn_means = [], [], []
    frames = {"MCTS": [], "NCCN": []}
    per_patient = []
    for seed_index in range(seeds):
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
    decided = joined["MCTS"].dropna(subset=["salvage"])
    by_patient = (decided.groupby("patient_id")["salvage"]
                  .agg(decisions="size",
                       accept_pct=lambda s: float((s == "systemic").mean() * 100))
                  .reset_index())
    opportunities = adaptation_opportunities(joined["MCTS"])
    return {
        "per_seed_gap": gaps,
        "per_seed_mcts": mcts_means,
        "per_seed_nccn": nccn_means,
        "utility_gap": float(np.mean(gaps)),
        "standard_error": float(np.std(gaps, ddof=1) / math.sqrt(seeds)),
        "mcts_utility": float(np.mean(mcts_means)),
        "nccn_utility": float(np.mean(nccn_means)),
        "mcts_recurrence_pct": float(joined["MCTS"]["recurred_by_5y"].mean() * 100),
        "mcts_salvage_decisions": int(len(decided)),
        "mcts_episodes_with_adaptation_pct": float((opportunities > 0).mean() * 100),
        "action_mix": mix,
        "salvage_by_year": salvage_by_recurrence_year(joined["MCTS"]),
        "salvage_by_patient": by_patient,
    }


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    raw07 = json.loads(CONFIG_V07.read_text(encoding="utf-8"))
    base_config = config_from_dict(raw07)

    raw = pd.read_csv(INPUT_CSV)
    os_model, rfs_model, os_test = build_reward_models(raw)
    neutral_model = os_model.neutralise_treatment_terms()
    first = balanced_subtype_sample(os_test, PER_SUBTYPE)
    second = balanced_subtype_sample(os_test, PER_SUBTYPE, offset=PER_SUBTYPE)
    if set(first["patient_id"]) & set(second["patient_id"]):
        raise SystemExit("cohorts A and B are not disjoint")
    sample = pd.concat([first, second], ignore_index=True)

    # --- Part 1: the analytic map ------------------------------------------
    environments, plans = {}, {}
    for _, row in sample.iterrows():
        patient = patient_from_row(row)
        environment = DynamicBreastCancerEnvironment(
            patient, make_risk_table(row, neutral_model, rfs_model), base_config)
        environments[patient.patient_id] = environment
        plans[patient.patient_id] = nccn_followup_state(environment)
    table = indifference_map(environments, plans, RATIOS, YEARS)
    table.to_csv(TABLE_DIR / "indifference_map.csv", index=False)
    share = treat_share_by_ratio(table)
    share.to_csv(TABLE_DIR / "treat_share_by_ratio.csv", index=False)
    crossing = indifference_ratio_per_patient(table)
    crossing.rename("indifference_ratio").reset_index().to_csv(
        TABLE_DIR / "indifference_ratio_by_patient.csv", index=False)
    ratio_star = pick_indifference_ratio(share)
    oracle_by_year = (table[table["hazard_ratio"] == ratio_star]
                      .groupby("recurrence_year")["advantage"]
                      .apply(lambda a: float((a > 0).mean() * 100)))
    oracle_by_year.rename("oracle_treat_pct").reset_index().to_csv(
        TABLE_DIR / "oracle_treat_share_by_year.csv", index=False)

    print(share.to_string(index=False), flush=True)
    print(f"\n환자별 무차별 위험비: 최소 {crossing.min():.2f} · 중앙 {crossing.median():.2f} · "
          f"최대 {crossing.max():.2f}  (전환 폭 {crossing.max() - crossing.min():.2f})", flush=True)
    print(f"무차별 위험비 HR* = {ratio_star:.2f}", flush=True)
    print("HR*에서 연도별 오라클 치료 비율:", oracle_by_year.round(1).to_dict(), flush=True)

    # --- Part 2: the searcher at the indifference ratio ---------------------
    raw_star = json.loads(json.dumps(raw07))
    raw_star["hazard_multipliers"]["salvage"]["systemic"]["death"] = float(ratio_star)
    raw_star["label"] = f"metabric-dynamic-v0.7-salvage-hr{ratio_star:.2f}"
    config_star = config_from_dict(raw_star)

    results = {}
    started = time.perf_counter()
    for key, label in ARMS:
        print(f"\n[{key}] running ({N_SEEDS} seeds)...", flush=True)
        result = evaluate(sample, neutral_model, rfs_model, config_star,
                          defer=(key == "defer"), seeds=N_SEEDS)
        result["label"] = label
        results[key] = result
        print(f"  gap {result['utility_gap']:+.4f} (SE {result['standard_error']:.4f})  "
              f"MCTS {result['mcts_utility']:.4f} vs NCCN {result['nccn_utility']:.4f}  "
              f"재발 {result['mcts_recurrence_pct']:.1f}% · 구제 결정 "
              f"{result['mcts_salvage_decisions']}  ({time.perf_counter() - started:.0f}s)",
              flush=True)

    def paired_series(a, b) -> dict:
        difference = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
        stderr = float(difference.std(ddof=1) / math.sqrt(len(difference)))
        return {
            "difference": float(difference.mean()),
            "standard_error": stderr,
            "z": float(difference.mean() / stderr) if stderr > 0 else float("nan"),
            "seeds_positive": int((difference > 0).sum()),
            "seeds": int(len(difference)),
        }

    prior = json.loads(PRIOR_METRICS.read_text(encoding="utf-8"))
    inert = prior["arms"]["inert"]

    by_year = results["decide"]["salvage_by_year"]
    by_year["accept_pct"] = 100.0 - by_year["declined_pct"]
    by_year = by_year.merge(oracle_by_year.rename("oracle_treat_pct").reset_index(),
                            on="recurrence_year", how="left")
    by_year["gap_pp"] = by_year["accept_pct"] - by_year["oracle_treat_pct"]
    by_year.to_csv(TABLE_DIR / "salvage_by_recurrence_year.csv", index=False)
    results["decide"]["salvage_by_patient"].to_csv(
        TABLE_DIR / "salvage_by_patient.csv", index=False)
    print("\n연도별: MCTS 수용률 vs 오라클", flush=True)
    print(by_year.to_string(index=False), flush=True)

    arm_table = pd.DataFrame([
        {"arm": key, "label": results[key]["label"],
         "utility_gap": results[key]["utility_gap"],
         "standard_error": results[key]["standard_error"],
         "mcts_utility": results[key]["mcts_utility"],
         "nccn_utility": results[key]["nccn_utility"],
         "mcts_recurrence_pct": results[key]["mcts_recurrence_pct"],
         "mcts_salvage_decisions": results[key]["mcts_salvage_decisions"],
         "mcts_episodes_with_adaptation_pct": results[key]["mcts_episodes_with_adaptation_pct"]}
        for key, _ in ARMS])
    arm_table.to_csv(TABLE_DIR / "arms.csv", index=False)

    seed_rows = []
    for key, _ in ARMS:
        for index in range(N_SEEDS):
            seed_rows.append({"arm": key, "seed_index": index,
                              "utility_gap": results[key]["per_seed_gap"][index],
                              "mcts_utility": results[key]["per_seed_mcts"][index],
                              "nccn_utility": results[key]["per_seed_nccn"][index]})
    pd.DataFrame(seed_rows).to_csv(TABLE_DIR / "per_seed.csv", index=False)

    mix_rows = []
    for key, _ in ARMS:
        for policy_name, fields in results[key]["action_mix"].items():
            for field, counts in fields.items():
                for action, percent in counts.items():
                    mix_rows.append({"arm": key, "policy": policy_name, "field": field,
                                     "action": action, "percent": percent})
    pd.DataFrame(mix_rows).to_csv(TABLE_DIR / "action_mix.csv", index=False)

    decide_vs_defer = paired_series(results["defer"]["per_seed_gap"],
                                    results["decide"]["per_seed_gap"])
    nccn_vs_inert = paired_series(inert["per_seed_nccn"],
                                  results["decide"]["per_seed_nccn"][:PRIOR_SEEDS])
    gap_c = results["decide"]["utility_gap"]
    accept = by_year.set_index("recurrence_year")["accept_pct"]
    known_answer_met = bool((by_year["gap_pp"].abs() <= KNOWN_ANSWER_TOLERANCE_PP).all())
    year_dependence = float(accept.get(1, float("nan")) - accept.get(4, float("nan")))
    patient_accept = results["decide"]["salvage_by_patient"]
    eligible = patient_accept[patient_accept["decisions"] >= 10]

    verdict = {
        "indifference_ratio": float(ratio_star),
        "indifference_ratio_by_patient": {
            "min": float(crossing.min()), "median": float(crossing.median()),
            "max": float(crossing.max()), "spread": float(crossing.max() - crossing.min()),
        },
        "treat_share_at_ratio_star": float(
            share.loc[share["hazard_ratio"] == ratio_star, "treat_share"].iloc[0]),
        "transition_band": {
            "last_ratio_all_treat": float(share.loc[share["treat_share"] >= 0.999, "hazard_ratio"].max()),
            "first_ratio_none_treat": float(share.loc[share["treat_share"] <= 0.001, "hazard_ratio"].min()),
        },
        "oracle_treat_pct_by_year": oracle_by_year.round(3).to_dict(),
        "mcts_accept_pct_by_year": accept.round(3).to_dict(),
        "known_answer_max_gap_pp": float(by_year["gap_pp"].abs().max()),
        "known_answer_prediction_met": known_answer_met,
        "year_dependence_pp": year_dependence,
        "year_dependence_prediction_met": bool(year_dependence >= YEAR_DEPENDENCE_FLOOR_PP),
        "gap_decide": gap_c,
        "gap_defer": results["defer"]["utility_gap"],
        "decide_vs_defer": decide_vs_defer,
        "value_of_decision_point": decide_vs_defer["difference"],
        "value_of_decision_point_relative": float(decide_vs_defer["difference"] / gap_c)
        if gap_c else float("nan"),
        "minimum_detectable_difference": minimum_detectable_difference(
            decide_vs_defer["standard_error"]),
        "minimum_detectable_difference_relative": float(
            minimum_detectable_difference(decide_vs_defer["standard_error"]) / gap_c)
        if gap_c else float("nan"),
        "decision_value_prediction_met": bool(
            0.0 < decide_vs_defer["difference"] < DECISION_VALUE_CEILING * gap_c),
        "nccn_vs_v1_9_inert": nccn_vs_inert,
        "nccn_neutral_prediction_met": bool(
            abs(nccn_vs_inert["difference"]) < NCCN_NEUTRAL_TOLERANCE * gap_c),
        "patient_accept_sd_pct": float(eligible["accept_pct"].std(ddof=1))
        if len(eligible) > 1 else float("nan"),
        "patients_with_10plus_decisions": int(len(eligible)),
        "mcts_salvage_mix_decide": results["decide"]["action_mix"]["MCTS"].get("salvage", {}),
        "nccn_identical_decide_vs_defer": bool(
            abs(results["decide"]["nccn_utility"] - results["defer"]["nccn_utility"]) < 1e-12),
    }
    if not verdict["nccn_identical_decide_vs_defer"]:
        raise SystemExit("NCCN utility differs between arms C and D; it must not")

    metrics = {
        "run_date": RUN_DATE,
        "analysis_label": "indifference-v2.0",
        "question": (
            "Where, if anywhere, does the salvage decision split between treating and "
            "declining, does it split between patients or only across recurrence years, "
            "and what is the decision point worth to a searcher at that ratio?"
        ),
        "estimand": (
            "Part 1: the expected-utility advantage of systemic over no salvage for each of "
            "the 40 patients x 4 recurrence years x 21 hazard ratios under the v0.7 "
            "environment and the patient's NCCN plan (deterministic). Part 2: the "
            "MCTS-minus-NCCN utility gap over the same 40 patients and 24 seeds with the "
            "salvage ratio at the indifference point, MCTS deciding versus deferring."
        ),
        "scope_warning": (
            "Declared synthetic parameters throughout. A property of our simulator, not a "
            "clinical effect."
        ),
        "prespecified_prediction": PRESPECIFIED_PREDICTION,
        "design": {
            "patients": int(len(sample)), "seeds": N_SEEDS, "simulations": SIMULATIONS,
            "episodes_per_policy": EPISODES_PER_POLICY,
            "reward_model": "treatment-neutral (v1.4)",
            "config": "configs/dynamic_v0_7.json with salvage systemic death ratio = HR*",
            "ratio_grid": [float(r) for r in RATIOS],
            "recurrence_years": list(YEARS),
            "map_plan": "each patient's simplified NCCN plan",
            "arms": [{"arm": key, "label": label} for key, label in ARMS],
        },
        "arms": {
            key: {k: v for k, v in results[key].items()
                  if k not in ("action_mix", "salvage_by_year", "salvage_by_patient")}
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
        "inputs": input_manifest({"data": INPUT_CSV, "assumptions_v07": CONFIG_V07}),
        "entry_point": "analysis/45_run_indifference.py",
        "base_seed": BASE_SEED,
        "salvage_death_ratio_used": float(ratio_star),
    }
    (REPORT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== 사전 예측 채점 ===", flush=True)
    for name, met in (
        ("1 정답 재현 (연도별 ±15%p)", verdict["known_answer_prediction_met"]),
        ("2 연도 의존 (1년차-4년차 ≥ 20%p)", verdict["year_dependence_prediction_met"]),
        ("3 결정 지점의 값 작음 (0 < C-D < 10% of gap)", verdict["decision_value_prediction_met"]),
        ("4 가이드라인에 중립 (|ΔNCCN| < 5% of gap)", verdict["nccn_neutral_prediction_met"]),
    ):
        print(f"  {name}: {'통과' if met else '빗나감'}", flush=True)
    print(f"\nC-D = {verdict['value_of_decision_point']:+.4f} "
          f"({verdict['value_of_decision_point_relative']:+.1%} of gap, z = {decide_vs_defer['z']:.2f}, "
          f"MDE {verdict['minimum_detectable_difference_relative']:.1%})", flush=True)
    print(f"환자별 수용률 SD: {verdict['patient_accept_sd_pct']:.1f}%p "
          f"(결정 10건 이상 환자 {verdict['patients_with_10plus_decisions']}명)", flush=True)


if __name__ == "__main__":
    main()
