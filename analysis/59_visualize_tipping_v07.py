"""Figure for the tipping-point robustness check (fig53).

Four panels: the two gap curves on the scale they can share (relative to the
guideline's utility), the two tipping points against the line the headline
claim B2 draws at one half, how each policy's utility moves as the dial turns
in the longer game, and what the searcher does with the standard treatments -
including the intensified rung v1.9 found the terminal value amplifies.

A second figure (fig54) splits the v0.7 gap that never closes, post hoc and
by arithmetic, into what a planner gains by choosing which standard treatments
to give and what it gains from rungs the guideline never plays.

Every number is read from the committed ``metrics.json`` and ``tables/``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import sys

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "reports" / "tipping-v07-v2.6"
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


def fig53_tipping_v07() -> None:
    verdict = METRICS["verdict"]
    compare = pd.read_csv(TABLE_DIR / "compare_v05_v07.csv")
    arms = pd.read_csv(TABLE_DIR / "arms.csv")
    v05 = compare[compare["environment"] == "v0.5"].sort_values("strength")
    v07 = compare[compare["environment"] == "v0.7"].sort_values("strength")
    t05 = float(verdict["v21_tipping_strength"])
    t07 = float(verdict["tipping_strength"])
    has_t07 = not math.isnan(t07)

    fig, axes = plt.subplots(1, 4, figsize=(20.5, 5.3),
                             gridspec_kw={"wspace": 0.33,
                                          "width_ratios": [1.25, 0.9, 1.05, 1.05]})

    # --- A: the two curves on a shared scale -------------------------------
    ax = axes[0]
    ax.plot(v05["strength"], v05["relative_gap"] * 100, marker="o", markersize=5,
            color=GRAY, linewidth=2, label="v0.5 (5년 게임, v2.1)")
    ax.errorbar(v07["strength"], v07["relative_gap"] * 100,
                yerr=v07["standard_error"] / v07["nccn_utility"] * 100,
                marker="o", markersize=6, color=BLUE, linewidth=2.2, capsize=4,
                ecolor=BLUE, label="v0.7 (말기 가치 10년)")
    ax.axhline(0, color=INK, linewidth=1.1)
    ax.axvline(t05, color=GRAY, linestyle="--", linewidth=1.2)
    if has_t07:
        ax.axvline(t07, color=BLUE, linestyle="--", linewidth=1.4)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("표준치료에 선언한 이득 (문헌 끝점의 분율)")
    ax.set_ylabel("효용 격차 ÷ NCCN 효용 (%)")
    ax.set_title("A. 두 환경의 격차 곡선 (상대 척도)", fontsize=11.5, loc="left", pad=10)
    ax.legend(fontsize=9, frameon=False, loc="upper right")

    # --- B: the two tipping points against B2's line ------------------------
    ax = axes[1]
    labels = ["v0.5\n(5년)", "v0.7\n(15년)"]
    values = [t05, t07 if has_t07 else float("nan")]
    colours = [GRAY, BLUE]
    for index, (value, colour) in enumerate(zip(values, colours)):
        if math.isnan(value):
            ax.text(index, 0.25, "0~1 범위에서\n교차 없음", ha="center", va="center",
                    fontsize=10, color=RED, fontweight="bold")
            continue
        ax.bar(index, value, width=0.55, color=colour, alpha=0.9)
        ax.text(index, value + 0.025, f"{value:.3f}", ha="center", fontsize=12,
                fontweight="bold", color=INK)
    ax.axhline(0.5, color=RED, linestyle=":", linewidth=1.6)
    ax.text(-0.55, 0.515, "절반 (B2의 선)", fontsize=9, color=RED, ha="left",
            va="bottom", fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("전환점 (문헌 끝점의 분율)")
    ax.set_title("B. 전환점", fontsize=11.5, loc="left", pad=10)

    # --- C: how each policy moves in the longer game ------------------------
    ax = axes[2]
    ax.plot(arms["strength"], arms["nccn_utility"], marker="s", markersize=5,
            color=AMBER, linewidth=2, label="NCCN")
    ax.plot(arms["strength"], arms["mcts_utility"], marker="o", markersize=5,
            color=BLUE, linewidth=2, label="MCTS")
    change = verdict["utility_change_zero_to_one"]
    ax.text(0.03, 0.06,
            f"0→1 상승: NCCN {change['nccn']:+.4f} · MCTS {change['mcts']:+.4f}",
            transform=ax.transAxes, fontsize=9.5, color=INK, fontweight="bold")
    if has_t07:
        ax.axvline(t07, color=BLUE, linestyle="--", linewidth=1.2)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("선언 강도")
    ax.set_ylabel("평균 효용 (v0.7 척도)")
    ax.set_title("C. 가이드라인이 따라잡는 방식", fontsize=11.5, loc="left", pad=10)
    ax.legend(fontsize=9, frameon=False, loc="upper left")

    # --- D: what the searcher takes -----------------------------------------
    ax = axes[3]
    for column, colour, label, style in (
        ("mcts_accept_chemo_pct", RED, "항암 (모든 단)", "-"),
        ("mcts_intensified_chemo_pct", RED, "강화 항암만", "--"),
        ("mcts_accept_endocrine_pct", BLUE, "내분비", "-"),
        ("mcts_accept_radiation_pct", GREEN, "방사선", "-"),
    ):
        ax.plot(arms["strength"], arms[column], marker="o", markersize=4,
                color=colour, linewidth=1.8, linestyle=style, label=label)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0, 105)
    ax.set_xlabel("선언 강도")
    ax.set_ylabel("MCTS가 그 치료를 받은 에피소드 (%)")
    ax.set_title("D. 탐색기가 받아들이는 것 (v0.7)", fontsize=11.5, loc="left", pad=10)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")

    shift = verdict["tipping_shift"]
    headline = (f"전환점 {t05:.3f} → {t07:.3f} ({shift:+.3f})" if has_t07
                else f"전환점 {t05:.3f} → v0.7에서는 범위 안에서 교차 없음")
    fig.suptitle(
        f"표준치료 이득이 15년에 걸쳐 작동하면 — {headline} "
        f"(환자 40명 · 시드 12 · 예산 1024 · 치료 중립 보상모형)",
        fontsize=13, y=1.03, color=INK)
    fig.text(0.5, -0.05,
             "v0.7은 효용 척도가 v0.5의 약 세 배라 절대 격차를 나란히 두지 않는다(패널 A는 NCCN "
             "효용 대비). 전환점 자체는 척도와 무관하다. 끝점 위험비는 원문 대조 대기 중인 선언값이다.",
             ha="center", fontsize=9.5, color=GRAY)
    save(fig, "fig53_tipping_v07.png")


def fig54_static_decomposition() -> None:
    static = pd.read_csv(TABLE_DIR / "static_decomposition.csv")
    compare = pd.read_csv(TABLE_DIR / "compare_v05_v07.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    for ax, label, title in ((axes[0], "v0.5", "A. v0.5 (5년 게임)"),
                             (axes[1], "v0.7", "B. v0.7 (말기 가치 10년)")):
        block = static[static["environment"] == label].sort_values("strength")
        run = compare[compare["environment"] == label].sort_values("strength")
        x = block["strength"].to_numpy()
        choice = block["choice_gain_rel"].to_numpy() * 100
        rung = block["exclusive_rung_gain_rel"].to_numpy() * 100
        width = 0.075
        ax.bar(x, choice, width=width, color=GREEN, alpha=0.85,
               label="줄 치료를 고르는 몫 (NCCN이 쓰는 단만)")
        ax.bar(x, rung, width=width, bottom=choice, color=RED, alpha=0.85,
               label="NCCN이 쓰지 않는 단의 몫 (강화 항암·연장 내분비)")
        ax.plot(run["strength"], run["relative_gap"] * 100, marker="o", color=INK,
                linewidth=2, markersize=5, label="실제 탐색 격차 (MCTS - NCCN)")
        ax.axhline(0, color=INK, linewidth=1.0)
        ax.set_xlim(-0.06, 1.06)
        ax.set_xlabel("표준치료에 선언한 이득 (문헌 끝점의 분율)")
        ax.set_title(title, fontsize=11.5, loc="left", pad=10)
    axes[0].set_ylabel("NCCN 효용 대비 (%)")
    axes[1].legend(fontsize=9, frameon=False, loc="upper right")
    fig.suptitle(
        "v0.7에서 격차가 닫히지 않는 이유 — 정적 계획 산술 분해 (사후 분석, 탐색 없음, 환자 40명)",
        fontsize=13, y=1.02, color=INK)
    fig.text(0.5, -0.05,
             "막대 = 환자마다 가장 좋은 고정 계획이 NCCN 계획보다 얼마나 나은가(탐색이 도달할 수 있는 "
             "상한의 근사). 선 = v2.1·v2.6의 실제 탐색 결과. 고정 계획은 선행 항암·독성 가드레일·구제 "
             "선택을 무시한다.",
             ha="center", fontsize=9.5, color=GRAY)
    save(fig, "fig54_static_decomposition_v07.png")


if __name__ == "__main__":
    fig53_tipping_v07()
    fig54_static_decomposition()
