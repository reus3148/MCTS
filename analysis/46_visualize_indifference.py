"""Figure for the indifference experiment (fig46).

Four panels: the analytic map's S-curve with the (narrow) band where patients
flip, the known-answer check of the searcher against the map by recurrence
year, how much the searcher's accept rate varies between patients, and the
value of the decision point at the indifference ratio.

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

REPORT_DIR = ROOT / "reports" / "indifference-v2.0"
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


def save(fig: plt.Figure, filename: str) -> None:
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(path, PUBLIC_DIR / filename)
    print(path)


def fig46_indifference() -> None:
    verdict = METRICS["verdict"]
    share = pd.read_csv(TABLE_DIR / "treat_share_by_ratio.csv")
    by_year = pd.read_csv(TABLE_DIR / "salvage_by_recurrence_year.csv")
    by_patient = pd.read_csv(TABLE_DIR / "salvage_by_patient.csv")
    arms = pd.read_csv(TABLE_DIR / "arms.csv").set_index("arm")
    ratio_star = verdict["indifference_ratio"]
    band = verdict["indifference_ratio_by_patient"]

    fig, axes = plt.subplots(1, 4, figsize=(20.5, 5.3),
                             gridspec_kw={"wspace": 0.36,
                                          "width_ratios": [1.15, 1.0, 1.0, 0.9]})

    # --- A: the map ----------------------------------------------------------
    ax = axes[0]
    ax.axvspan(band["min"] - 0.005, band["max"] + 0.005, color="#fde68a", alpha=0.7,
               label=f"환자별 무차별 구간 {band['min']:.2f}~{band['max']:.2f}")
    ax.plot(share["hazard_ratio"], share["treat_share"] * 100, marker="o", color=INK,
            linewidth=1.8, markersize=4)
    ax.axvline(ratio_star, color=RED, linestyle="--", linewidth=1.3)
    ax.text(ratio_star - 0.006, 12, f"HR* = {ratio_star:.2f}", color=RED, fontsize=9.5,
            ha="right")
    ax.set_xlim(0.795, 1.005)
    ax.set_ylim(-3, 108)
    ax.set_xlabel("선언된 구제 사망 위험비")
    ax.set_ylabel("치료가 이득인 (환자 × 재발연도) 칸 (%)")
    ax.set_title("A. 환경의 산술이 그린 지도 — 전환 폭 0.01",
                 fontsize=11.5, loc="left", pad=10)
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")

    # --- B: known answer by year --------------------------------------------
    ax = axes[1]
    positions = np.arange(len(by_year))
    width = 0.38
    ax.bar(positions - width / 2, by_year["oracle_treat_pct"], width, color="#a8a29e",
           label="오라클 (기댓값 산술)")
    ax.bar(positions + width / 2, by_year["accept_pct"], width, color=BLUE,
           label="MCTS (예산 1024)")
    for position, row in zip(positions, by_year.itertuples()):
        ax.text(position - width / 2, row.oracle_treat_pct + 2, f"{row.oracle_treat_pct:.0f}",
                ha="center", fontsize=9, color=GRAY)
        ax.text(position + width / 2, row.accept_pct + 2, f"{row.accept_pct:.0f}",
                ha="center", fontsize=9, color=INK, fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{int(y)}년차\n(n={int(n):,})" for y, n in
                        zip(by_year["recurrence_year"], by_year["decisions"])], fontsize=9)
    ax.set_ylim(0, 122)
    ax.set_ylabel("구제 수용률 (%)")
    ax.set_title(f"B. 정답을 아는 문제 — 최대 차이 {verdict['known_answer_max_gap_pp']:.0f}%p",
                 fontsize=11.5, loc="left", pad=10)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")

    # --- C: between-patient spread -------------------------------------------
    ax = axes[2]
    eligible = by_patient[by_patient["decisions"] >= 10].sort_values("accept_pct")
    positions = np.arange(len(eligible))
    ax.bar(positions, eligible["accept_pct"], 0.8, color=BLUE)
    ax.axhline(float(eligible["accept_pct"].mean()), color=INK, linestyle="--", linewidth=1.0)
    ax.set_xticks([])
    ax.set_xlabel(f"환자 (결정 10건 이상, {len(eligible)}명, 수용률 순)")
    ax.set_ylabel("MCTS 구제 수용률 (%)")
    ax.set_ylim(0, 108)
    ax.set_title(f"C. 환자 간 퍼짐 — SD {verdict['patient_accept_sd_pct']:.0f}%p",
                 fontsize=11.5, loc="left", pad=10)

    # --- D: the decision point's value --------------------------------------
    ax = axes[3]
    labels = ["C MCTS 결정", "D 규칙 위임"]
    gaps = [arms.loc["decide", "utility_gap"], arms.loc["defer", "utility_gap"]]
    errors = [arms.loc["decide", "standard_error"], arms.loc["defer", "standard_error"]]
    ax.bar([0, 1], gaps, 0.55, color=[BLUE, "#93c5fd"], yerr=errors, capsize=5, ecolor=INK)
    top = max(g + e for g, e in zip(gaps, errors))
    for position, (gap, error) in enumerate(zip(gaps, errors)):
        ax.text(position, gap + error + top * 0.02, f"{gap:+.4f}", ha="center",
                fontsize=10, color=INK, fontweight="bold")
    bracket_y = top * 1.2
    ax.plot([0, 0, 1, 1], [bracket_y - top * 0.03, bracket_y, bracket_y, bracket_y - top * 0.03],
            color=INK, linewidth=1.1)
    value = verdict["value_of_decision_point"]
    z = verdict["decide_vs_defer"]["z"]
    ax.text(0.5, bracket_y + top * 0.03,
            f"C - D = {value:+.4f}\n({verdict['value_of_decision_point_relative']:+.1%} of gap, z = {z:.2f})",
            ha="center", fontsize=9.5, color=INK, fontweight="bold", linespacing=1.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("MCTS 빼기 NCCN 효용 격차")
    ax.set_ylim(0, top * 1.5)
    ax.set_title(f"D. 무차별 지점에서의 결정 지점의 값 (시드 24)",
                 fontsize=11.5, loc="left", pad=10)

    fig.suptitle(
        f"구제 결정이 갈리는 곳은 어디인가 — 위험비 {ratio_star:.2f}에서만, 그리고 환자가 아니라 재발 연도로 "
        "(v0.7 환경 · 환자 40명 · 예산 1024 · 치료 중립 보상모형)",
        fontsize=13, y=1.03, color=INK)
    fig.text(0.5, -0.05,
             "지도는 각 환자의 NCCN 계획 위에서 환경의 연간 사건 모형을 기댓값으로 푼 것이다. "
             "탐색은 지도가 고른 위험비 한 곳에서만 돌렸다.",
             ha="center", fontsize=9.5, color=GRAY)
    save(fig, "fig46_indifference.png")


if __name__ == "__main__":
    fig46_indifference()
