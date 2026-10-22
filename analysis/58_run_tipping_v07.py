"""Does the 44.5% tipping point survive a longer game? (v2.6)

v2.1 turned the one parameter that controls the headline into a dial. Standard
chemotherapy, standard endocrine therapy and local radiotherapy are declared in
``configs/dynamic_v0_5.json`` to buy **no** survival while charging toxicity and
burden; v2.1 raised that benefit from 0 to the randomised-trial endpoint and
found the utility gap crossing zero at **44.5%** of the endpoint. That number
is now the most quoted figure in the project (``docs/final-claims.md`` B2: "give
standard treatment less than half of its literature benefit and the conclusion
flips").

It was measured in a **five-year game**. A reviewer's first question writes
itself: *standard treatment pays off over decades and costs you now - did the
five-year horizon do the work?* v1.8 already showed what a horizon can do to a
decision (every late salvage decision declined, for no clinical reason), and
v1.9 built the fix: ``configs/dynamic_v0_7.json`` credits ten years of
decision-free follow-up beyond the horizon as a terminal value.

This run repeats v2.1's sweep, unchanged, on v0.7. It asks one thing: **where is
the tipping point when benefit has fifteen years to accrue instead of five?**

What has to be held fixed for the answer to mean anything:

- **Same cohort, seeds, budget and reward model** as v1.9's v0.7 arm, so that
  strength 0 here *is* that arm - which makes the null control exact rather
  than approximate (prediction 1).
- **Same dial** (``analysis/dynamic/benefit.py``). ``tests/test_benefit.py``
  pins that the dial leaves v0.7 untouched at strength 0 and never reaches into
  the salvage block or the tail.
- **Relative gaps** beside every absolute one. The terminal value roughly
  triples the utility scale (v1.9), so an absolute gap here cannot sit beside
  one from v0.5. The tipping *strength* needs no correction - it is where the
  gap is zero on either scale - which is exactly why it is the right quantity
  to compare.

What is *not* held fixed, and must be read as part of the answer: v0.7 also
opens the salvage decision, and v1.9 found the terminal value amplifies the
MCTS-only intensified-chemotherapy rung (12% -> 39% of episodes). Both act on
the gap. This run measures the tipping point of the environment we would
actually defend, not of v0.5 with a longer tail bolted on.

PRE-SPECIFIED PREDICTIONS, recorded and committed before the run
----------------------------------------------------------------
Prediction 3's direction was already written down on 2026-09-26, in the v2.1
report's limitations ("in v0.7 the benefit works over 15 years, so the tipping
point should fall") and in the v2.1 minutes. It is carried here unchanged.

1. **Known-answer null control.** Strength 0 reproduces v1.9's v0.7 arm
   (``gap_decide`` = +0.054202) to within 1e-9. Same config, cohort, seeds and
   model; anything else means the sweep is not measuring what it claims.
2. **Primary - B2 survives.** The v0.7 tipping strength is **below 0.50**: the
   claim "less than half of the literature benefit flips the conclusion" holds
   in the longer game. If not, B2's wording changes everywhere it is quoted.
3. **Direction.** The v0.7 tipping strength is **below 0.445** - benefit that
   accrues for fifteen years needs less of it to catch up with a cost paid once.
   Counted as met only if the gap at strength 0.445 is negative; reported as
   "direction only" unless that gap is also distinguishable from zero
   (|z| >= 2), per CLAUDE.md's rule against counting a sign as a pass.
4. **The guideline moves more.** From strength 0 to 1, NCCN's utility rises
   more than MCTS's - as in v0.5 (+0.0609 vs +0.0350): the guideline stops
   paying a penalty, the searcher already avoided it.
5. **Monotone within noise.** No adjacent pair of settings shows the gap
   *rising* by more than twice that difference's paired standard error. v2.1
   registered strict monotonicity and failed it formally on noise (+0.0004,
   z = +0.27); the tolerance is registered relative to the noise this time.

The one most likely to fail is 3: the amplified intensified rung improves with
the dial too (the ladder keeps its increments), so the searcher also gains from
a longer game and the crossing could move either way.
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

from analysis.dynamic.benefit import (  # noqa: E402
    FULL_BENEFIT,
    crossing_strength,
    relative_gap,
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
CONFIG_V07 = ROOT / "configs" / "dynamic_v0_7.json"
REPORT_DIR = ROOT / "reports" / "tipping-v07-v2.6"
TABLE_DIR = REPORT_DIR / "tables"
V19_METRICS = ROOT / "reports" / "horizon-v1.9" / "metrics.json"
V21_METRICS = ROOT / "reports" / "benefit-tipping-v2.1" / "metrics.json"

RUN_DATE = "2026-10-22"
PER_SUBTYPE = 5                 # v1.2's cohorts A and B: 40 patients
N_SEEDS = 12
SIMULATIONS = 1024
EPISODES_PER_POLICY = 40
EXPLORATION_WEIGHT = math.sqrt(2.0)
#: v2.1's grid shifted to resolve the low end, with v2.1's tipping point itself
#: as an arm so prediction 3 is a direct reading rather than an interpolation.
STRENGTHS = (0.0, 0.15, 0.30, 0.445, 0.60, 0.80, 1.00)
V21_TIPPING = 0.445
B2_CEILING = 0.50
NULL_TOLERANCE = 1e-9
NOISE_MULTIPLE = 2.0

ACTION_FIELDS = ("timing", "surgery", "chemo", "endocrine", "radiation", "salvage")

PRESPECIFIED_PREDICTION = {
    "known_answer_null_control": (
        f"Strength 0 reproduces v1.9's v0.7 arm (gap_decide) to within {NULL_TOLERANCE}."),
    "primary_b2_survives": (
        f"The v0.7 tipping strength is below {B2_CEILING}: less than half of the "
        "literature benefit still flips the conclusion."),
    "direction": (
        f"The v0.7 tipping strength is below v2.1's {V21_TIPPING}; met only if the "
        f"gap at strength {V21_TIPPING} is negative, and reported as direction "
        "only unless that gap is also distinguishable from zero (|z| >= 2). "
        "Direction first written in the v2.1 report on 2026-09-26."),
    "guideline_moves_more": (
        "From strength 0 to 1, NCCN's utility rises more than MCTS's."),
    "monotone_within_noise": (
        f"No adjacent pair shows the gap rising by more than {NOISE_MULTIPLE}x that "
        "difference's paired standard error."),
    "why": (
        "B2 (44.5%) is the project's most quoted number and was measured in a "
        "five-year game. Standard treatment pays off over decades and costs now; "
        "a reviewer will ask whether the horizon did the work. v0.7 credits ten "
        "more years as a terminal value (v1.9)."),
}


def config_from_dict(data: dict) -> DynamicConfig:
    config = DynamicConfig(**data)
    _validate_probabilities(config)
    return config


def evaluate(sample, os_model, rfs_model, config) -> dict:
    """v2.1's loop, unchanged in every respect that touches the random stream.

    Identical seed derivation to v1.9's ``evaluate`` (base seed + 1,000 per
    seed index + a hash offset per patient), which is what lets strength 0
    reproduce v1.9's v0.7 arm exactly.
    """
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
    mcts = joined["MCTS"]
    accepted = {
        "chemo": float((mcts["chemo"].astype(str) != "none").mean() * 100),
        "chemo_intensified": float((mcts["chemo"].astype(str) == "intensified").mean() * 100),
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


def paired(results: dict, a: str, b: str, field: str = "per_seed_gap") -> dict:
    """b minus a, paired by seed."""
    difference = np.array(results[b][field]) - np.array(results[a][field])
    stderr = float(difference.std(ddof=1) / math.sqrt(N_SEEDS))
    return {
        "difference": float(difference.mean()),
        "standard_error": stderr,
        "z": float(difference.mean() / stderr) if stderr > 0 else float("nan"),
        "seeds_positive": int((difference > 0).sum()),
    }


def key_for(strength: float) -> str:
    return f"s{int(round(strength * 1000)):04d}"


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    raw07 = json.loads(CONFIG_V07.read_text(encoding="utf-8"))
    v19 = json.loads(V19_METRICS.read_text(encoding="utf-8"))["verdict"]
    v21 = json.loads(V21_METRICS.read_text(encoding="utf-8"))["verdict"]

    raw = pd.read_csv(INPUT_CSV)
    os_model, rfs_model, os_test = build_reward_models(raw)
    neutral_model = os_model.neutralise_treatment_terms()
    first = balanced_subtype_sample(os_test, PER_SUBTYPE)
    second = balanced_subtype_sample(os_test, PER_SUBTYPE, offset=PER_SUBTYPE)
    if set(first["patient_id"]) & set(second["patient_id"]):
        raise SystemExit("cohorts A and B are not disjoint")
    sample = pd.concat([first, second], ignore_index=True)
    print(f"환자 {len(sample)}명 · 시드 {N_SEEDS} · 예산 {SIMULATIONS} · "
          f"강도 {len(STRENGTHS)}단계 · v0.7 (말기 가치 {raw07['terminal_tail_years']}년)",
          flush=True)

    results = {}
    started = time.perf_counter()
    for strength in STRENGTHS:
        key = key_for(strength)
        config = config_from_dict(with_standard_benefit(raw07, strength))
        print(f"\n[{key}] strength {strength:.3f} running...", flush=True)
        result = evaluate(sample, neutral_model, rfs_model, config)
        result["strength"] = float(strength)
        result["relative_gap"] = relative_gap(result["utility_gap"], result["nccn_utility"])
        results[key] = result
        accepted = result["mcts_accept_pct"]
        print(f"  gap {result['utility_gap']:+.5f} (SE {result['standard_error']:.5f}) "
              f"· 상대 {result['relative_gap']:+.3%} · MCTS {result['mcts_utility']:.4f} "
              f"vs NCCN {result['nccn_utility']:.4f} · 항암 {accepted['chemo']:.0f}% "
              f"(강화 {accepted['chemo_intensified']:.0f}%) · 내분비 "
              f"{accepted['endocrine']:.0f}% · 방사선 {accepted['radiation']:.0f}%  "
              f"({time.perf_counter() - started:.0f}s)", flush=True)

    keys = [key_for(s) for s in STRENGTHS]
    gaps = [results[k]["utility_gap"] for k in keys]

    arm_table = pd.DataFrame([{
        "arm": k, "strength": results[k]["strength"],
        "utility_gap": results[k]["utility_gap"],
        "standard_error": results[k]["standard_error"],
        "relative_gap": results[k]["relative_gap"],
        "mcts_utility": results[k]["mcts_utility"],
        "nccn_utility": results[k]["nccn_utility"],
        "mcts_accept_chemo_pct": results[k]["mcts_accept_pct"]["chemo"],
        "mcts_intensified_chemo_pct": results[k]["mcts_accept_pct"]["chemo_intensified"],
        "mcts_accept_endocrine_pct": results[k]["mcts_accept_pct"]["endocrine"],
        "mcts_accept_radiation_pct": results[k]["mcts_accept_pct"]["radiation"],
    } for k in keys])
    arm_table.to_csv(TABLE_DIR / "arms.csv", index=False)

    pd.DataFrame([{
        "arm": k, "strength": results[k]["strength"], "seed_index": i,
        "utility_gap": results[k]["per_seed_gap"][i],
        "mcts_utility": results[k]["per_seed_mcts"][i],
        "nccn_utility": results[k]["per_seed_nccn"][i],
    } for k in keys for i in range(N_SEEDS)]).to_csv(TABLE_DIR / "per_seed.csv", index=False)

    pd.DataFrame([{
        "arm": k, "strength": results[k]["strength"], "policy": policy,
        "field": field, "action": action, "percent": percent,
    } for k in keys for policy, fields in results[k]["action_mix"].items()
        for field, counts in fields.items() for action, percent in counts.items()
    ]).to_csv(TABLE_DIR / "action_mix.csv", index=False)

    # Side-by-side with v2.1 on the relative scale: the comparison the whole
    # run exists for, committed as a table so the figure reads it.
    v21_arms = pd.read_csv(ROOT / "reports" / "benefit-tipping-v2.1" / "tables" / "arms.csv")
    v21_arms = v21_arms.assign(
        relative_gap=v21_arms["utility_gap"] / v21_arms["nccn_utility"], environment="v0.5")
    v26_arms = arm_table.assign(environment="v0.7")
    pd.concat([
        v21_arms[["environment", "strength", "utility_gap", "standard_error",
                  "relative_gap", "mcts_utility", "nccn_utility"]],
        v26_arms[["environment", "strength", "utility_gap", "standard_error",
                  "relative_gap", "mcts_utility", "nccn_utility"]],
    ], ignore_index=True).to_csv(TABLE_DIR / "compare_v05_v07.csv", index=False)

    tipping = crossing_strength(list(STRENGTHS), gaps)
    at_v21 = results[key_for(V21_TIPPING)]
    z_at_v21 = (at_v21["utility_gap"] / at_v21["standard_error"]
                if at_v21["standard_error"] > 0 else float("nan"))
    adjacent = [paired(results, a, b) for a, b in zip(keys, keys[1:])]
    worst_rise = max(adjacent, key=lambda d: d["difference"] - NOISE_MULTIPLE * d["standard_error"])
    monotone_ok = all(d["difference"] <= NOISE_MULTIPLE * d["standard_error"] for d in adjacent)
    zero_to_one = {
        "mcts": results[keys[-1]]["mcts_utility"] - results[keys[0]]["mcts_utility"],
        "nccn": results[keys[-1]]["nccn_utility"] - results[keys[0]]["nccn_utility"],
    }
    null_gap = results[keys[0]]["utility_gap"]
    v21_relative_at_zero = v21["gap_at_zero"] / (
        v21_arms.loc[v21_arms["strength"] == 0.0, "nccn_utility"].iloc[0])

    verdict = {
        "strengths": list(STRENGTHS),
        "gaps": gaps,
        "relative_gaps": [results[k]["relative_gap"] for k in keys],
        "gap_at_zero": gaps[0],
        "gap_at_one": gaps[-1],
        "relative_gap_at_zero": results[keys[0]]["relative_gap"],
        "relative_gap_at_one": results[keys[-1]]["relative_gap"],
        "v21_relative_gap_at_zero": float(v21_relative_at_zero),
        "v19_gap_decide": v19["gap_decide"],
        "null_difference": float(null_gap - v19["gap_decide"]),
        "tipping_strength": tipping,
        "v21_tipping_strength": v21["tipping_strength"],
        "tipping_shift": (float(tipping - v21["tipping_strength"])
                          if not math.isnan(tipping) else float("nan")),
        "implied_hazard_ratios_at_tipping": {
            channel: {outcome: float(value) ** tipping for outcome, value in full.items()}
            for channel, full in FULL_BENEFIT.items()
        } if not math.isnan(tipping) else None,
        "gap_at_v21_tipping": at_v21["utility_gap"],
        "gap_at_v21_tipping_se": at_v21["standard_error"],
        "gap_at_v21_tipping_z": float(z_at_v21),
        "gap_at_v21_tipping_mde": minimum_detectable_difference(at_v21["standard_error"]),
        "direction_distinguishable": bool(abs(z_at_v21) >= 2.0),
        "adjacent_paired": adjacent,
        "worst_adjacent_rise": worst_rise,
        "utility_change_zero_to_one": zero_to_one,
        "intensified_chemo_pct_at_zero": results[keys[0]]["mcts_accept_pct"]["chemo_intensified"],
        "intensified_chemo_pct_at_one": results[keys[-1]]["mcts_accept_pct"]["chemo_intensified"],
        "null_control_prediction_met": bool(abs(null_gap - v19["gap_decide"]) < NULL_TOLERANCE),
        "primary_prediction_met": bool(not math.isnan(tipping) and tipping < B2_CEILING),
        "direction_prediction_met": bool(at_v21["utility_gap"] < 0),
        "guideline_moves_more_prediction_met": bool(zero_to_one["nccn"] > zero_to_one["mcts"]),
        "monotone_prediction_met": bool(monotone_ok),
    }

    metrics = {
        "run_date": RUN_DATE,
        "analysis_label": "tipping-v07-v2.6",
        "question": (
            "Where does the utility gap cross zero, as the declared benefit of "
            "standard treatment rises from 0 to the trial endpoint, when the game "
            "credits ten years beyond the five-year horizon (v0.7)?"),
        "estimand": (
            "The strength of declared standard-treatment benefit (fraction of the "
            "randomised-trial endpoint hazard ratios, geometric interpolation) at "
            "which the mean MCTS-minus-NCCN utility gap crosses zero, in the v0.7 "
            "environment, for 40 subtype-balanced METABRIC patients, 12 seeds, "
            "budget 1024, treatment-neutral reward model."),
        "scope_warning": (
            "A property of our declared parameters, not of patients. The endpoint "
            "hazard ratios are declared from EBCTCG orders of magnitude and await "
            "checking against the source; the tipping point is reported both as a "
            "fraction and as the hazard ratios at that fraction."),
        "prespecified_prediction": PRESPECIFIED_PREDICTION,
        "design": {
            "config": "configs/dynamic_v0_7.json",
            "dial": "analysis/dynamic/benefit.py (unchanged from v2.1)",
            "strengths": list(STRENGTHS),
            "patients": int(len(sample)),
            "seeds": N_SEEDS,
            "budget": SIMULATIONS,
            "episodes_per_policy": EPISODES_PER_POLICY,
            "reward_model": "treatment-neutral (v1.4)",
            "null_control_reference": "reports/horizon-v1.9 gap_decide",
        },
        "arms": {k: {f: v for f, v in results[k].items() if f != "action_mix"} for k in keys},
        "verdict": verdict,
    }
    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = {
        "run_date": RUN_DATE,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit_before_run": git_commit(),
        "inputs": input_manifest({"data": INPUT_CSV, "assumptions_v07": CONFIG_V07}),
        "entry_point": "analysis/58_run_tipping_v07.py",
        "base_seed": BASE_SEED,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    (REPORT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n" + arm_table.round(5).to_string(index=False), flush=True)
    print(f"\n전환점: v0.7 {tipping:.3f}  (v0.5 {v21['tipping_strength']:.3f}, "
          f"이동 {verdict['tipping_shift']:+.3f})", flush=True)
    print(f"강도 {V21_TIPPING}에서 격차 {at_v21['utility_gap']:+.5f} "
          f"(SE {at_v21['standard_error']:.5f}, z {z_at_v21:+.2f})", flush=True)
    print(f"상대 격차 강도 0: v0.7 {verdict['relative_gap_at_zero']:+.3%} vs "
          f"v0.5 {v21_relative_at_zero:+.3%}", flush=True)
    print("\n=== 사전 예측 채점 ===", flush=True)
    for name, met in (
        ("1 영대조 (강도 0 = v1.9 v0.7 팔)", verdict["null_control_prediction_met"]),
        ("2 B2가 버틴다 (전환점 < 0.50)", verdict["primary_prediction_met"]),
        ("3 방향 (전환점 < 0.445)", verdict["direction_prediction_met"]),
        ("4 NCCN이 더 많이 오른다", verdict["guideline_moves_more_prediction_met"]),
        ("5 잡음 안에서 단조", verdict["monotone_prediction_met"]),
    ):
        print(f"  {name}: {'통과' if met else '빗나감'}", flush=True)
    if verdict["direction_prediction_met"] and not verdict["direction_distinguishable"]:
        print("  (3은 방향만 맞음 — 강도 0.445의 격차가 0과 구분되지 않는다)", flush=True)


if __name__ == "__main__":
    main()
