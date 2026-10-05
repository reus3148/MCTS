"""Figure for the disease-burden grading map (fig50).

Four panels that walk the v2.0 triad in order and end where it fails: what the
dial declares, how far the grades' indifference ratios spread, whether they
ever give opposite answers, and - the panel the argument turns on - what
knowing the grade is actually worth against the searcher's noise floor.

Every number is read from the committed ``metrics.json`` and ``tables/``.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "reports" / "burden-map-v2.4"
TABLE_DIR = REPORT_DIR / "tables"
FIGURE_DIR = REPORT_DIR / "figures"
PUBLIC_DIR = ROOT / "public" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams.update({
    "font.family": "Malgun Gothic",
    "axes.unicode_minus": False,
    "figure.facecolor": "#fafaf9",
    "axes.facecolor": "#fafaf9",
    "axes.edgecolor": "#292524",
    "axes.labelcolor": "#292524",
    "axes.titlecolor": "#1c1917",
    "xtick.color": "#57534e",
    "ytick.color": "#57534e",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e7e5e4",
    "grid.linewidth": 0.6,
})

INK = "#1c1917"
BLUE = "#2563eb"
GREEN = "#15803d"
RED = "#dc2626"
AMBER = "#d97706"
GRAY = "#78716c"

METRICS = json.loads((REPORT_DIR / "metrics.json").read_text(encoding="utf-8"))

GRADE_LABEL = {"controlled": "안정", "indeterminate": "판정 보류",
               "progressing": "진행"}
GRADE_COLOUR = {"controlled": GREEN, "indeterminate": AMBER, "progressing": RED}
GRADES = ("controlled", "indeterminate", "progressing")


def save(fig: plt.Figure, filename: str) -> None:
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(path, PUBLIC_DIR / filename)
    print(path)


def fig50_burden_map() -> None:
    verdict = METRICS["verdict"]
    ladder = pd.read_csv(TABLE_DIR / "declared_grades.csv")
    spread = pd.read_csv(TABLE_DIR / "spread_by_separation.csv")
    treat_share = pd.read_csv(TABLE_DIR / "treat_share_by_grade.csv")
    information = pd.read_csv(TABLE_DIR / "value_of_grade_information.csv")
    noise = verdict["action_value_noise"]

    fig, axes = plt.subplots(1, 4, figsize=(20.5, 5.3),
                             gridspec_kw={"wspace": 0.32,
                                          "width_ratios": [1.0, 1.05, 1.1, 1.1]})

    # --- A: what the dial declares -------------------------------------------
    ax = axes[0]
    for grade in GRADES:
        block = ladder[ladder["grade"] == grade].sort_values("separation")
        ax.plot(block["separation"], block["death_after_recurrence"], marker="o",
                markersize=4.5, color=GRADE_COLOUR[grade], linewidth=1.9,
                label=GRADE_LABEL[grade])
    base = float(ladder.loc[ladder["separation"] == 0,
                            "death_after_recurrence"].iloc[0])
    ax.axhline(base, color=GRAY, linestyle=":", linewidth=1.1)
    ax.text(0.5, base + 0.12, f"현재 환경 ×{base:.1f} (가중평균 고정)",
            fontsize=8.5, color=GRAY, ha="center")
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("등급 분리 강도 (0 = 현재 환경)")
    ax.set_ylabel("재발 후 사망 위험 배수")
    ax.set_title("A. 다이얼이 선언하는 것", fontsize=11.5, loc="left", pad=10)
    ax.legend(fontsize=9, frameon=False, loc="upper left")

    # --- B: how far the indifference ratios spread ---------------------------
    ax = axes[1]
    ax.plot(spread["separation"], spread["grade_spread"], marker="o",
            markersize=5, color=BLUE, linewidth=2, label="등급 간")
    ax.plot(spread["separation"], spread["patient_spread"], marker="s",
            markersize=4.5, color=GRAY, linewidth=1.6, linestyle="--",
            label="환자 간")
    floor = 0.05
    ax.axhline(floor, color=RED, linestyle=":", linewidth=1.4)
    ax.text(0.5, floor + 0.0022, "사전 예측 하한 0.05", fontsize=9, color=RED,
            ha="center", fontweight="bold")
    full = float(verdict["full_grade_spread"])
    ax.annotate(f"{full:.4f}", xy=(1.0, full), xytext=(0.74, full + 0.011),
                fontsize=10, fontweight="bold", color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.1))
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0, floor * 1.32)
    ax.set_xlabel("등급 분리 강도")
    ax.set_ylabel("무차별 위험비의 퍼짐")
    ax.set_title("B. 벌어지긴 하는데, 예측만큼은 아니다",
                 fontsize=11.5, loc="left", pad=10)
    ax.legend(fontsize=9, frameon=False, loc="upper left")

    # --- C: do the grades ever disagree? -------------------------------------
    ax = axes[2]
    share = treat_share.pivot(index="hazard_ratio", columns="grade",
                              values="treat_share")
    for grade in GRADES:
        ax.plot(share.index, share[grade] * 100, marker="o", markersize=3.6,
                color=GRADE_COLOUR[grade], linewidth=1.9, label=GRADE_LABEL[grade])
    ax.axhline(50, color=GRAY, linestyle=":", linewidth=1.0)
    split = verdict["disagreement_at_full"]
    if split.get("exists"):
        low, high = split["ratio_range"]
        ax.axvspan(low - 0.004, high + 0.004, color="#fde68a", alpha=0.6)
        ax.text((low + high) / 2, 86,
                f"답이 갈리는 구간\n{low:.2f}~{high:.2f}", ha="center",
                fontsize=9.5, color=AMBER, fontweight="bold", linespacing=1.4)
    ax.set_xlim(float(share.index.min()) - 0.005, float(share.index.max()) + 0.005)
    ax.set_ylim(-4, 104)
    ax.set_xlabel("선언된 구제치료 사망 위험비")
    ax.set_ylabel("치료가 유리한 (환자×연도) 칸 (%)")
    ax.set_title("C. 갈리기는 한다 (분리 1.0)", fontsize=11.5, loc="left", pad=10)
    ax.legend(fontsize=9, frameon=False, loc="lower left")

    # --- D: but what is knowing worth? ---------------------------------------
    ax = axes[3]
    ax.bar(information["hazard_ratio"], information["value_of_knowing"],
           width=0.007, color=BLUE, alpha=0.9)
    ax.axhline(noise, color=RED, linewidth=2.2)
    ax.text(float(information["hazard_ratio"].min()) + 0.004, noise * 0.90,
            f"예산 1024의 행동값 잡음 {noise:.2f}", fontsize=10, color=RED,
            fontweight="bold", va="top")
    best = float(verdict["max_value_of_grade_information"])
    at = float(verdict["max_value_at_ratio"])
    ax.annotate(f"최대 {best:.5f}\n잡음의 1/{verdict['noise_to_information_ratio']:.0f}",
                xy=(at, best), xytext=(at - 0.055, noise * 0.42),
                fontsize=10, fontweight="bold", color=INK, linespacing=1.4,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.set_xlim(float(information["hazard_ratio"].min()) - 0.005,
                float(information["hazard_ratio"].max()) + 0.005)
    ax.set_ylim(0, noise * 1.25)
    ax.set_xlabel("선언된 구제치료 사망 위험비")
    ax.set_ylabel("등급을 아는 것의 값 (완전정보)")
    ax.set_title("D. 그런데 알아도 쓸 수가 없다", fontsize=11.5, loc="left", pad=10)

    fig.suptitle(
        f"재발 플래그를 등급으로 쪼개면 결정이 생기나 — 답이 갈리는 구간은 생기지만"
        f"(안정 {split.get('controlled_treat_share', 0) * 100:.0f}% 대 진행 "
        f"{split.get('progressing_treat_share', 0) * 100:.0f}%), 등급을 아는 값이 "
        f"{best:.5f}로 탐색 잡음 {noise:.2f}의 1/"
        f"{verdict['noise_to_information_ratio']:.0f}이다 "
        f"(환자 40명 · v0.7 환경 · 탐색 없음)",
        fontsize=13, y=1.04, color=INK)
    fig.text(0.5, -0.08,
             "전부 우리가 선언한 파라미터의 성질이지 환자의 성질이 아니다. 등급 가중치는 "
             "MSK-CHORD 유방암 32,977건의 판정 분포(56.0/20.1/23.9%)이고, 분리 끝점은 "
             "선언값이다. 가중평균 위험은 모든 분리에서 고정되어 있다(평균 중립). "
             "분리 0은 현재 환경과 동일하다 - 영대조가 그것을 확인한다.",
             ha="center", fontsize=9.5, color=GRAY)
    save(fig, "fig50_burden_map.png")


if __name__ == "__main__":
    fig50_burden_map()
