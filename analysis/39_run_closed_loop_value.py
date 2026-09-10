"""How much of the MCTS advantage is closed-loop adaptation? (v1.7)

v1.5 decomposed the gap and could name 81% of it: the reward model's confounded
treatment coefficients (v1.4, 40%) and the declared cost of treatment that MCTS
declines to pay (v1.5, 41%). The remainder - 4%, +0.0013 with a standard error
of 0.0017 - was labelled "closed-loop adaptation included" and left there,
because nothing in that design could separate it from noise.

v1.6 then established that the phenomenon is real outside our simulator: in
GENIE BPC a "Progressing" scan is followed by a new regimen at 2.41x the rate a
"Stable" scan is, and the placebo-in-time control holds (forward minus backward
+37.6%p for progressing, -1.2%p for controlled). So the question stops being
"is there such a thing" and becomes **how much of our simulated advantage is
made of it.**

This run takes the response channel away from the searching policy and measures
what the policy loses. The environment is untouched: every arm simulates
episodes in the real environment with the same seeds, so every arm faces the
same draws and the same outcomes. Only what the *planner* is shown changes.

The response reaches a planner by two separate routes, and they are closed one
at a time:

===========  =======================  ================================
arm          label the planner sees   size the planner sees
===========  =======================  ================================
A ``none``   the drawn response       the realised size
B ``label``  ``not_applicable``       the realised size
C ``full``   ``not_applicable``       E[size] under the chosen intensity
===========  =======================  ================================

The label multiplies the terminal hazard (mean-neutralised since v0.5, so its
expectation is exactly 1.0 but "major" still beats "none"). The tumour size
gates breast-conserving surgery, so that route changes the **legal action set**,
not merely a value estimate. See ``analysis/dynamic/blinding.py``.

Two equalities are asserted rather than hoped for:

* Arm A must reproduce v1.5's baseline gap **exactly**. The blinded wrapper is
  a pass-through at mode ``none``, so anything else means the v1.6 refactor of
  ``_chemo_response`` changed behaviour.
* NCCN utility must be **identical in all three arms**. ``DynamicNccnPolicy``
  never reads ``state.response``, and the environment and seeds do not change.

PRE-SPECIFIED PREDICTIONS, recorded before the run
--------------------------------------------------
1. **Primary** - arm C's MCTS utility is *lower* than arm A's. Information has
   no negative value to a policy that may ignore it, so blinding should cost
   something. If C is higher, the searcher was being *hurt* by the response
   channel, which would be a finding about our search, not about adaptation.
2. **Magnitude** - the utility gap falls by less than **0.005** from A to C.
   v1.5 put everything unexplained at +0.0013 (SE 0.0017), and closed-loop
   adaptation has to live inside that. A larger drop would mean v1.5's
   decomposition was leaking.
3. **Monotone** - arm B sits between A and C. Closing one of the two routes
   should cost no more than closing both.
4. **Option value** - MCTS chooses neoadjuvant *less often* in arm C. A blinded
   planner cannot hope that a major response will win back breast-conserving
   surgery, so the option value of going first with chemotherapy disappears.

Prediction 4 is the one most likely to be wrong: neoadjuvant still shrinks the
tumour by its expectation under full blinding, so the mean benefit survives even
though the option value does not. Prediction 1 is the one that has to hold for
the rest to be interpretable.
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

from analysis.dynamic.blinding import ResponseBlindMCTSPolicy  # noqa: E402
from analysis.dynamic.cohort import (  # noqa: E402
    BASE_SEED,
    balanced_subtype_sample,
    build_reward_models,
    git_commit,
    input_manifest,
    make_risk_table,
)
from analysis.dynamic.config import load_dynamic_config  # noqa: E402
from analysis.dynamic.environment import DynamicBreastCancerEnvironment  # noqa: E402
from analysis.dynamic.experiment_utils import (  # noqa: E402
    minimum_detectable_difference,
)
from analysis.dynamic.evaluation import run_policy_episodes  # noqa: E402
from analysis.dynamic.policies import DynamicNccnPolicy  # noqa: E402
from analysis.dynamic.schema import patient_from_row  # noqa: E402

INPUT_CSV = ROOT / "data" / "processed" / "patients_with_nccn.csv"
CONFIG_PATH = ROOT / "configs" / "dynamic_v0_5.json"
REPORT_DIR = ROOT / "reports" / "closed-loop-value-v1.7"
TABLE_DIR = REPORT_DIR / "tables"
PRIOR_METRICS = ROOT / "reports" / "channel-decomposition-v1.5" / "metrics.json"

RUN_DATE = "2026-09-10"
PER_SUBTYPE = 5                 # v1.2's cohorts A and B: 40 patients
N_SEEDS = 12
SIMULATIONS = 1024
EPISODES_PER_POLICY = 40
EXPLORATION_WEIGHT = math.sqrt(2.0)
MAGNITUDE_CEILING = 0.005

ARMS = (
    ("none", "A. 그대로 (응답 라벨 + 실제 종양 크기)"),
    ("label", "B. 라벨만 가림 (종양 크기는 그대로)"),
    ("full", "C. 완전 블라인드 (라벨 + 크기 둘 다)"),
)

PRESPECIFIED_PREDICTION = {
    "primary": (
        "Arm C's MCTS utility is lower than arm A's - information has no "
        "negative value to a policy that may ignore it."
    ),
    "magnitude": (
        f"The utility gap falls by less than {MAGNITUDE_CEILING} from A to C. "
        "v1.5 put everything unexplained at +0.0013 (SE 0.0017) and closed-loop "
        "adaptation has to live inside that."
    ),
    "monotone": (
        "Arm B sits between A and C: closing one of the two routes should cost "
        "no more than closing both."
    ),
    "option_value": (
        "MCTS chooses neoadjuvant less often in arm C - a blinded planner "
        "cannot hope a major response will win back breast-conserving surgery."
    ),
    "why": (
        "v1.5 could not separate closed-loop adaptation from noise. v1.6 showed "
        "the phenomenon is real in recorded care, so the remaining question is "
        "how much of the simulated advantage is made of it."
    ),
}

ACTION_FIELDS = ("timing", "surgery", "chemo", "endocrine", "radiation")


def evaluate(sample, os_model, rfs_model, config, mode: str) -> dict:
    gaps, mcts_means, nccn_means = [], [], []
    actions = {"MCTS": [], "NCCN": []}
    blinded_decisions = fallbacks = 0
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
            policy = ResponseBlindMCTSPolicy(
                environment, mode=mode, simulations=SIMULATIONS,
                exploration_weight=EXPLORATION_WEIGHT, seed=patient_seed)
            md = pd.DataFrame(run_policy_episodes(
                environment, policy, EPISODES_PER_POLICY, patient_seed + 10_000))
            nd = pd.DataFrame(run_policy_episodes(
                environment, DynamicNccnPolicy(environment),
                EPISODES_PER_POLICY, patient_seed + 10_000))
            blinded_decisions += policy.blinded_decisions
            fallbacks += policy.fallbacks
            mcts_util.append(md["utility"].mean())
            nccn_util.append(nd["utility"].mean())
            actions["MCTS"].append(md)
            actions["NCCN"].append(nd)
        gaps.append(float(np.mean(mcts_util) - np.mean(nccn_util)))
        mcts_means.append(float(np.mean(mcts_util)))
        nccn_means.append(float(np.mean(nccn_util)))

    mix = {}
    for policy_name, frames in actions.items():
        joined = pd.concat(frames, ignore_index=True)
        mix[policy_name] = {
            field: joined[field].astype(str).value_counts(normalize=True)
            .mul(100).round(3).to_dict()
            for field in ACTION_FIELDS if field in joined.columns
        }
    return {
        "per_seed_gap": gaps,
        "per_seed_mcts": mcts_means,
        "utility_gap": float(np.mean(gaps)),
        "standard_error": float(np.std(gaps, ddof=1) / math.sqrt(N_SEEDS)),
        "mcts_utility": float(np.mean(mcts_means)),
        "nccn_utility": float(np.mean(nccn_means)),
        "blinded_decisions": int(blinded_decisions),
        "fallbacks": int(fallbacks),
        "action_mix": mix,
    }


def neoadjuvant_share(result: dict) -> float:
    return float(result["action_mix"]["MCTS"].get("timing", {})
                 .get("neoadjuvant", 0.0))


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    config = load_dynamic_config(CONFIG_PATH)
    raw = pd.read_csv(INPUT_CSV)
    os_model, rfs_model, os_test = build_reward_models(raw)
    # v1.4: the reward model carries confounded treatment coefficients, so every
    # arm here runs on the neutralised model, exactly as v1.5 did.
    neutral_model = os_model.neutralise_treatment_terms()

    first = balanced_subtype_sample(os_test, PER_SUBTYPE)
    second = balanced_subtype_sample(os_test, PER_SUBTYPE, offset=PER_SUBTYPE)
    if set(first["patient_id"]) & set(second["patient_id"]):
        raise SystemExit("cohorts A and B are not disjoint")
    sample = pd.concat([first, second], ignore_index=True)

    results = {}
    started = time.perf_counter()
    for mode, label in ARMS:
        print(f"\n[{mode}] running...", flush=True)
        result = evaluate(sample, neutral_model, rfs_model, config, mode)
        result["label"] = label
        results[mode] = result
        print(f"  gap {result['utility_gap']:+.4f} "
              f"(SE {result['standard_error']:.4f})  "
              f"MCTS {result['mcts_utility']:.4f} vs "
              f"NCCN {result['nccn_utility']:.4f}  "
              f"블라인드 결정 {result['blinded_decisions']} · "
              f"대체 {result['fallbacks']}  "
              f"({time.perf_counter() - started:.0f}s)", flush=True)

    def paired(a: str, b: str, key: str = "per_seed_gap") -> dict:
        difference = np.array(results[b][key]) - np.array(results[a][key])
        stderr = float(difference.std(ddof=1) / math.sqrt(N_SEEDS))
        return {
            "difference": float(difference.mean()),
            "standard_error": stderr,
            "z": float(difference.mean() / stderr) if stderr > 0 else float("nan"),
        }

    prior = json.loads(PRIOR_METRICS.read_text(encoding="utf-8"))
    baseline_gap = results["none"]["utility_gap"]
    v15_baseline = prior["verdict"]["gap_baseline"]
    nccn_values = [results[mode]["nccn_utility"] for mode, _ in ARMS]
    nccn_spread = float(max(nccn_values) - min(nccn_values))

    arm_table = pd.DataFrame([
        {
            "arm": mode,
            "label": results[mode]["label"],
            "utility_gap": results[mode]["utility_gap"],
            "standard_error": results[mode]["standard_error"],
            "mcts_utility": results[mode]["mcts_utility"],
            "nccn_utility": results[mode]["nccn_utility"],
            "mcts_neoadjuvant_pct": neoadjuvant_share(results[mode]),
            "blinded_decisions": results[mode]["blinded_decisions"],
            "fallbacks": results[mode]["fallbacks"],
        }
        for mode, _ in ARMS
    ])
    arm_table.to_csv(TABLE_DIR / "arms.csv", index=False)
    print("\n" + arm_table.to_string(index=False), flush=True)

    seed_rows = []
    for mode, _ in ARMS:
        for index, (gap, mcts) in enumerate(zip(
                results[mode]["per_seed_gap"], results[mode]["per_seed_mcts"])):
            seed_rows.append({"arm": mode, "seed_index": index,
                              "utility_gap": gap, "mcts_utility": mcts})
    pd.DataFrame(seed_rows).to_csv(TABLE_DIR / "per_seed.csv", index=False)

    mix_rows = []
    for mode, _ in ARMS:
        for policy_name, fields in results[mode]["action_mix"].items():
            for field, counts in fields.items():
                for action, percent in counts.items():
                    mix_rows.append({"arm": mode, "policy": policy_name,
                                     "field": field, "action": action,
                                     "percent": percent})
    pd.DataFrame(mix_rows).to_csv(TABLE_DIR / "action_mix.csv", index=False)

    verdict = {
        "gap_none": baseline_gap,
        "gap_label": results["label"]["utility_gap"],
        "gap_full": results["full"]["utility_gap"],
        "mcts_none": results["none"]["mcts_utility"],
        "mcts_label": results["label"]["mcts_utility"],
        "mcts_full": results["full"]["mcts_utility"],
        "label_vs_none": paired("none", "label"),
        "full_vs_none": paired("none", "full"),
        "full_vs_label": paired("label", "full"),
        "closed_loop_value": baseline_gap - results["full"]["utility_gap"],
        # A null is only readable next to the size it could have found.
        "minimum_detectable_difference": minimum_detectable_difference(
            paired("none", "full")["standard_error"]),
        "seeds_with_full_below_none": int(sum(
            1 for x, y in zip(results["none"]["per_seed_gap"],
                              results["full"]["per_seed_gap"]) if y < x)),
        "seeds_with_label_below_none": int(sum(
            1 for x, y in zip(results["none"]["per_seed_gap"],
                              results["label"]["per_seed_gap"]) if y < x)),
        "share_of_v15_baseline": float(
            (baseline_gap - results["full"]["utility_gap"]) / baseline_gap)
        if baseline_gap else float("nan"),
        "primary_prediction_met": bool(
            results["full"]["mcts_utility"] < results["none"]["mcts_utility"]),
        "magnitude_prediction_met": bool(
            abs(baseline_gap - results["full"]["utility_gap"])
            < MAGNITUDE_CEILING),
        "magnitude_ceiling": MAGNITUDE_CEILING,
        "monotone_prediction_met": bool(
            min(results["none"]["utility_gap"], results["full"]["utility_gap"])
            <= results["label"]["utility_gap"]
            <= max(results["none"]["utility_gap"],
                   results["full"]["utility_gap"])),
        "neoadjuvant_pct_none": neoadjuvant_share(results["none"]),
        "neoadjuvant_pct_label": neoadjuvant_share(results["label"]),
        "neoadjuvant_pct_full": neoadjuvant_share(results["full"]),
        "option_value_prediction_met": bool(
            neoadjuvant_share(results["full"])
            < neoadjuvant_share(results["none"])),
        "blinded_decisions_full": results["full"]["blinded_decisions"],
        "fallbacks_full": results["full"]["fallbacks"],
        "reproduces_v1_5_baseline": bool(abs(baseline_gap - v15_baseline) < 1e-9),
        "v1_5_baseline_gap": v15_baseline,
        "nccn_utility_spread_across_arms": nccn_spread,
        "nccn_identical_across_arms": bool(nccn_spread < 1e-12),
    }

    if not verdict["reproduces_v1_5_baseline"]:
        raise SystemExit(
            f"arm A gap {baseline_gap} does not reproduce v1.5's "
            f"{v15_baseline}; the blinding wrapper is not a pass-through")
    if not verdict["nccn_identical_across_arms"]:
        raise SystemExit(
            f"NCCN utility moved by {nccn_spread} across arms; it must not, "
            "because DynamicNccnPolicy never reads state.response")

    metrics = {
        "run_date": RUN_DATE,
        "analysis_label": "closed-loop-value-v1.7",
        "question": (
            "How much of the MCTS-minus-NCCN utility gap comes from the "
            "searching policy being able to observe the response it drew?"
        ),
        "estimand": (
            "MCTS-minus-NCCN utility gap over the same 40 patients and 12 seeds "
            "as v1.5, under three planners that differ only in what they are "
            "shown about the neoadjuvant response. Episodes are simulated in "
            "the unmodified environment in every arm."
        ),
        "scope_warning": (
            "A property of our own simulator's information structure. Not a "
            "clinical effect, and not a claim about real closed-loop care."
        ),
        "prespecified_prediction": PRESPECIFIED_PREDICTION,
        "design": {
            "patients": int(len(sample)),
            "seeds": N_SEEDS,
            "simulations": SIMULATIONS,
            "episodes_per_policy": EPISODES_PER_POLICY,
            "reward_model": "treatment-neutral (v1.4)",
            "config": "configs/dynamic_v0_5.json",
            "arms": [{"arm": mode, "label": label} for mode, label in ARMS],
        },
        "arms": {
            mode: {key: value for key, value in results[mode].items()
                   if key != "action_mix"}
            for mode, _ in ARMS
        },
        "action_mix": {mode: results[mode]["action_mix"] for mode, _ in ARMS},
        "verdict": verdict,
    }
    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    manifest = {
        "run_date": RUN_DATE,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit_before_run": git_commit(),
        "inputs": input_manifest(
            {"data": INPUT_CSV, "assumptions": CONFIG_PATH}),
        "entry_point": "analysis/39_run_closed_loop_value.py",
        "base_seed": BASE_SEED,
    }
    (REPORT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    print("\n=== 사전 예측 채점 ===", flush=True)
    for name, met in (
        ("1 주 예측 (블라인드가 손해)", verdict["primary_prediction_met"]),
        ("2 크기 (< 0.005)", verdict["magnitude_prediction_met"]),
        ("3 단조성 (B가 A와 C 사이)", verdict["monotone_prediction_met"]),
        ("4 옵션 가치 (선행치료 감소)", verdict["option_value_prediction_met"]),
    ):
        print(f"  {name}: {'통과' if met else '빗나감'}", flush=True)
    print(f"\n폐루프 적응의 값: {verdict['closed_loop_value']:+.4f} "
          f"(v1.5 기준선 {v15_baseline:+.4f}의 "
          f"{verdict['share_of_v15_baseline'] * 100:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
