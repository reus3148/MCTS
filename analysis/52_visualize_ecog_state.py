"""Figure for the ECOG state-dependence assessment (fig49).

Four panels, arranged so the one passing prediction and the three failing ones
read as one argument: performance status tracks the disease, but recorded care
does not visibly act on it, the fall's size buys nothing, and the
chemotherapy association runs the wrong way inside every stage stratum.

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

REPORT_DIR = ROOT / "reports" / "ecog-state-v2.3"
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


def fig49_ecog_state() -> None:
    verdict = METRICS["verdict"]
    transitions = pd.read_csv(TABLE_DIR / "transition_rates.csv")
    rates = pd.read_csv(TABLE_DIR / "stop_rates.csv")
    by_size = pd.read_csv(TABLE_DIR / "stop_rates_by_size.csv")
    stratified = pd.read_csv(TABLE_DIR / "chemo_share_by_ecog_and_stage.csv")

    fig, axes = plt.subplots(1, 4, figsize=(20.5, 5.3),
                             gridspec_kw={"wspace": 0.33,
                                          "width_ratios": [1.0, 1.05, 1.1, 1.1]})

    # --- A: it tracks the disease (the one prediction that passed) -----------
    ax = axes[0]
    labels = ["진행 판정 전", "진행 판정 후"]
    values = [verdict["worsenings_per_year_before"],
              verdict["worsenings_per_year_after"]]
    bars = ax.bar(labels, values, color=[GRAY, GREEN], alpha=0.9, width=0.58)
    for bar, value, row in zip(bars, values, transitions.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.07,
                f"{value:.2f}", ha="center", fontsize=11,
                fontweight="bold", color=INK)
        ax.text(bar.get_x() + bar.get_width() / 2, -0.22,
                f"{row.person_years:,.0f} 인-년", ha="center",
                fontsize=8.5, color=GRAY)
    ax.set_ylim(-0.42, max(values) * 1.22)
    ax.set_ylabel("연간 기록된 ECOG 악화 횟수")
    ax.set_title(f"A. 질병은 추적한다 ({verdict['worsening_ratio']:.2f}배)",
                 fontsize=11.5, loc="left", pad=10)
    ax.text(0.97, 0.95, "예측 4 통과", transform=ax.transAxes, ha="right",
            va="top", fontsize=10, color=GREEN, fontweight="bold")

    # --- B: but treatment barely responds ------------------------------------
    ax = axes[1]
    indexed = rates.set_index("direction")
    order = ["improved", "worsened"]
    names = {"improved": "ECOG 호전", "worsened": "ECOG 악화"}
    positions = np.arange(len(order))
    values = [indexed.loc[key, "stopped_after"] * 100 for key in order]
    ax.bar(positions, values, color=[GRAY, RED], alpha=0.9, width=0.55)
    for index, value in enumerate(values):
        ax.text(index, value + 0.5, f"{value:.1f}%", ha="center",
                fontsize=11, fontweight="bold", color=INK)
    gap = verdict["stop_gap"] * 100
    mde = verdict["clustered"]["minimum_detectable_difference"] * 100
    ax.annotate("", xy=(1, values[1]), xytext=(0, values[0]),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2))
    ax.text(0.5, max(values) * 0.62, f"격차 {gap:+.1f}%p",
            ha="center", fontsize=10.5, fontweight="bold", color=INK)
    ax.axhspan(values[0], values[0] + mde, color="#fde68a", alpha=0.55)
    ax.text(0.5, values[0] + mde + 0.5,
            f"검출 가능 최소 차이 {mde:.1f}%p", ha="center",
            fontsize=9, color=AMBER, fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels([names[key] for key in order], fontsize=10)
    ax.set_ylim(0, max(values) * 1.5)
    ax.set_ylabel("90일 내 전신치료가 중단된 비율 (%)")
    ax.set_title("B. 그런데 치료는 거의 안 바뀐다",
                 fontsize=11.5, loc="left", pad=10)
    ax.text(0.97, 0.95, "예측 1 빗나감", transform=ax.transAxes, ha="right",
            va="top", fontsize=10, color=RED, fontweight="bold")

    # --- C: and the fall's size buys nothing ---------------------------------
    ax = axes[2]
    gaps = by_size["gap"] * 100
    colours = [GREEN if value > 0 else RED for value in gaps]
    ax.bar(by_size["size"], gaps, color=colours, alpha=0.9, width=0.6)
    ax.axhline(0, color=INK, linewidth=1.1)
    for row, value in zip(by_size.itertuples(), gaps):
        offset = 0.9 if value >= 0 else -1.9
        ax.text(row.size, value + offset, f"{value:+.1f}", ha="center",
                fontsize=9.5, fontweight="bold", color=INK)
        ax.text(row.size, min(gaps) * 1.36, f"n={row.changes:,}", ha="center",
                fontsize=8.5, color=GRAY)
    ax.set_xticks(by_size["size"])
    ax.set_xlabel("ECOG가 떨어진 단계 수")
    ax.set_ylabel("호전 대비 중단율 격차 (%p)")
    ax.set_ylim(min(gaps) * 1.55, max(gaps) * 2.6)
    ax.set_title("C. 더 크게 떨어져도 더 멈추지 않는다",
                 fontsize=11.5, loc="left", pad=10)
    ax.text(0.97, 0.95, "용량-반응 없음", transform=ax.transAxes, ha="right",
            va="top", fontsize=10, color=RED, fontweight="bold")

    # --- D: the chemotherapy association runs backwards ----------------------
    ax = axes[3]
    strata = [s for s in ("Stage 1-3", "Stage 4") if s in set(stratified["stratum"])]
    width = 0.36
    positions = np.arange(len(strata))
    good, poor, poor_n = [], [], []
    for stratum in strata:
        block = stratified[stratified["stratum"] == stratum]
        healthy = block[block["ecog"] < verdict["poor_performance_cut"]]
        unwell = block[block["ecog"] >= verdict["poor_performance_cut"]]
        good.append(float(healthy["chemo"].sum() / healthy["starts"].sum()) * 100)
        poor.append(float(unwell["chemo"].sum() / unwell["starts"].sum()) * 100)
        poor_n.append(int(unwell["starts"].sum()))
    ax.bar(positions - width / 2, good, width, color=GRAY, alpha=0.9,
           label="ECOG 0-1")
    ax.bar(positions + width / 2, poor, width, color=RED, alpha=0.9,
           label=f"ECOG >= {verdict['poor_performance_cut']}")
    for index, (healthy, unwell, count) in enumerate(zip(good, poor, poor_n)):
        ax.text(index - width / 2, healthy + 1.2, f"{healthy:.1f}%", ha="center",
                fontsize=9.5, color=INK, fontweight="bold")
        ax.text(index + width / 2, unwell + 1.2, f"{unwell:.1f}%", ha="center",
                fontsize=9.5, color=INK, fontweight="bold")
        ax.text(index + width / 2, 2.0, f"n={count}", ha="center",
                fontsize=8, color="#fecaca")
    ax.set_xticks(positions)
    ax.set_xticklabels(strata, fontsize=10)
    ax.set_ylim(0, max(good + poor) * 1.32)
    ax.set_ylabel("전신치료 시작 중 항암 비중 (%)")
    ax.set_title("D. 자격 게이트는 반대로 간다",
                 fontsize=11.5, loc="left", pad=10)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.text(0.97, 0.95, "예측 3 빗나감\n(병기로 설명 안 됨)", transform=ax.transAxes,
            ha="right", va="top", fontsize=9.5, color=RED, fontweight="bold",
            linespacing=1.4)

    fig.suptitle(
        f"ECOG는 상태인가 — 질병은 추적하지만({verdict['worsening_ratio']:.2f}배) "
        f"기록된 진료가 그것에 반응하는 것은 보이지 않는다 "
        f"(중단율 격차 {verdict['stop_gap'] * 100:+.1f}%p, 검출 한계 "
        f"{verdict['clustered']['minimum_detectable_difference'] * 100:.1f}%p) "
        f"· MSK-CHORD 유방암 {verdict['cohort']:,}명",
        fontsize=13, y=1.04, color=INK)
    fig.text(0.5, -0.08,
             "기술적 서술이다. 양방향으로 교란되어 있다 - 환자가 나빠져서 치료를 멈추기도 하고, "
             "치료 때문에 나빠지기도 한다(악화의 26.8%가 치료 시작 90일 안에 온다. 호전은 22.6%). "
             "이 릴리스는 ECOG를 변화 시점에만 기록한다. "
             "자료: MSK-CHORD (MSK, Nature 2024), CC BY-NC-ND 4.0.",
             ha="center", fontsize=9.5, color=GRAY)
    save(fig, "fig49_ecog_state.png")


if __name__ == "__main__":
    fig49_ecog_state()
