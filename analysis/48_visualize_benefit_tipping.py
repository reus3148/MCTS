"""Figure for the standard-treatment benefit sweep (fig47).

Four panels: what the config declares at each setting of the dial, the cheap
arithmetic preview that located the crossing before any search, the searched
gap curve with its tipping point, and what the searcher does with the three
standard treatments as their declared benefit rises.

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

REPORT_DIR = ROOT / "reports" / "benefit-tipping-v2.1"
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

CHANNEL_LABEL = {"chemo": "표준 항암", "endocrine": "표준 내분비", "radiation": "국소 방사선"}
CHANNEL_COLOR = {"chemo": RED, "endocrine": BLUE, "radiation": GREEN}


def save(fig: plt.Figure, filename: str) -> None:
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(path, PUBLIC_DIR / filename)
    print(path)


def fig47_benefit_tipping() -> None:
    verdict = METRICS["verdict"]
    arms = pd.read_csv(TABLE_DIR / "arms.csv")
    preview = pd.read_csv(TABLE_DIR / "preview_summary.csv")
    crossings = pd.read_csv(TABLE_DIR / "preview_crossing_by_patient.csv")
    ladder = pd.read_csv(TABLE_DIR / "declared_ladder.csv")
    tipping = verdict["tipping_strength"]

    fig, axes = plt.subplots(1, 4, figsize=(20.5, 5.3),
                             gridspec_kw={"wspace": 0.34,
                                          "width_ratios": [1.0, 1.05, 1.25, 1.05]})

    # --- A: what the dial declares -------------------------------------------
    ax = axes[0]
    standard = {"chemo": "standard", "endocrine": "standard", "radiation": "local"}
    for channel, level in standard.items():
        block = ladder[(ladder["channel"] == channel) & (ladder["level"] == level)]
        block = block.sort_values("strength")
        ax.plot(block["strength"], block["recurrence_hazard"], marker="o", markersize=4,
                color=CHANNEL_COLOR[channel], linewidth=1.8, label=CHANNEL_LABEL[channel])
    ax.axhline(1.0, color=GRAY, linestyle=":", linewidth=1.0)
    ax.axvline(tipping, color=INK, linestyle="--", linewidth=1.2)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("선언 강도 (0 = 현재 config)")
    ax.set_ylabel("선언된 재발 위험비")
    ax.set_title("A. 다이얼이 선언하는 것", fontsize=11.5, loc="left", pad=10)
    ax.legend(fontsize=9, frameon=False, loc="lower left")

    # --- B: the arithmetic preview ------------------------------------------
    ax = axes[1]
    ax.plot(preview["strength"], preview["mean_advantage"], marker="o", markersize=3.5,
            color=INK, linewidth=1.8)
    ax.axhline(0.0, color=GRAY, linewidth=1.0)
    ax.fill_between(preview["strength"], 0, preview["mean_advantage"],
                    where=preview["mean_advantage"] >= 0, color="#bbf7d0", alpha=0.8)
    ax.fill_between(preview["strength"], 0, preview["mean_advantage"],
                    where=preview["mean_advantage"] < 0, color="#fecaca", alpha=0.8)
    spread = verdict["preview_crossing"]
    ax.axvspan(spread["min"], spread["max"], color="#fde68a", alpha=0.5,
               label=f"환자별 교차 {spread['min']:.2f}~{spread['max']:.2f}")
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("선언 강도")
    ax.set_ylabel("NCCN 계획 빼기 전면 거절 계획 (기댓값)")
    ax.set_title("B. 탐색 없는 산술 미리보기", fontsize=11.5, loc="left", pad=10)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")

    # --- C: the searched gap curve -------------------------------------------
    ax = axes[2]
    ax.errorbar(arms["strength"], arms["utility_gap"], yerr=arms["standard_error"],
                marker="o", markersize=6, color=BLUE, linewidth=2, capsize=5, ecolor=INK)
    ax.axhline(0.0, color=INK, linewidth=1.1)
    for row in arms.itertuples():
        offset = row.standard_error + (max(arms["utility_gap"]) - min(arms["utility_gap"])) * 0.05
        ax.text(row.strength, row.utility_gap + offset, f"{row.utility_gap:+.4f}",
                ha="center", fontsize=9, color=INK, fontweight="bold")
    ax.axvline(tipping, color=RED, linestyle="--", linewidth=1.4)
    ax.text(tipping + 0.02, max(arms["utility_gap"]) * 0.55,
            f"전환점\n강도 {tipping:.2f}", color=RED, fontsize=10, fontweight="bold",
            linespacing=1.4)
    ax.set_xlim(-0.05, 1.05)
    span = float(arms["utility_gap"].max() - arms["utility_gap"].min())
    ax.set_ylim(float(arms["utility_gap"].min()) - span * 0.22,
                float(arms["utility_gap"].max()) + span * 0.22)
    ax.set_xlabel("선언 강도")
    ax.set_ylabel("MCTS 빼기 NCCN 효용 격차")
    ax.set_title("C. 표준치료 이득을 얼마나 주면 뒤집히나",
                 fontsize=11.5, loc="left", pad=10)

    # --- D: what the searcher accepts ---------------------------------------
    ax = axes[3]
    for channel in ("chemo", "endocrine", "radiation"):
        ax.plot(arms["strength"], arms[f"mcts_accept_{channel}_pct"], marker="o",
                markersize=4, color=CHANNEL_COLOR[channel], linewidth=1.8,
                label=CHANNEL_LABEL[channel])
    ax.axvline(tipping, color=INK, linestyle="--", linewidth=1.2)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0, 105)
    ax.set_xlabel("선언 강도")
    ax.set_ylabel("MCTS가 그 치료를 받은 에피소드 (%)")
    ax.set_title("D. 탐색기는 이득이 붙자 받기 시작한다",
                 fontsize=11.5, loc="left", pad=10)
    ax.legend(fontsize=9, frameon=False, loc="upper left")

    gap0, gap1 = verdict["gap_at_zero"], verdict["gap_at_one"]
    fig.suptitle(
        f"표준치료에 이득을 선언하면 무엇이 달라지나 — 격차 {gap0:+.4f} → {gap1:+.4f}, "
        f"전환점 강도 {tipping:.2f} (환자 40명 · 시드 12 · 예산 1024 · 치료 중립 보상모형)",
        fontsize=13, y=1.03, color=INK)
    fig.text(0.5, -0.05,
             "끝점 위험비는 무작위 배정 메타분석의 자릿수로 적은 선언값이고 원문 대조가 필요하다. "
             "전환점은 끝점의 분율과 그때의 위험비로 함께 보고한다.",
             ha="center", fontsize=9.5, color=GRAY)
    save(fig, "fig47_benefit_tipping.png")


if __name__ == "__main__":
    fig47_benefit_tipping()
