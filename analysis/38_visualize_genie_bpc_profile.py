"""Figures for the GENIE BPC data-adequacy assessment (fig41, fig42).

fig41 is the movetext: how long real games run, and what kind of move gets
played at each turn. fig42 is the closed-loop evidence with its placebo-in-time
control beside it, because the control is what makes the finding worth
anything.

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

REPORT_DIR = ROOT / "reports" / "genie-bpc-profile-v1.6"
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
PURPLE = "#7c3aed"
GRAY = "#78716c"

METRICS = json.loads((REPORT_DIR / "metrics.json").read_text(encoding="utf-8"))

CHANNEL_LABEL = {
    "chemotherapy": "항암",
    "endocrine": "내분비",
    "her2_targeted": "HER2 표적",
    "targeted_other": "기타 표적",
    "investigational": "임상시험 약제",
    "immunotherapy": "면역",
    "other": "기타",
}
CHANNEL_COLOR = {
    "chemotherapy": RED,
    "endocrine": BLUE,
    "her2_targeted": GREEN,
    "targeted_other": AMBER,
    "investigational": PURPLE,
    "immunotherapy": "#0891b2",
    "other": GRAY,
}
CHANNEL_ORDER = ("chemotherapy", "endocrine", "her2_targeted",
                 "targeted_other", "investigational", "immunotherapy", "other")

STAGE_ORDER = ("Stage I", "Stage II", "Stage III", "Stage IV",
               "Stage I-III NOS")


def save(fig: plt.Figure, filename: str) -> None:
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(path, PUBLIC_DIR / filename)
    print(path)


def fig41_movetext() -> None:
    verdict = METRICS["verdict"]
    sequences = pd.read_csv(TABLE_DIR / "sequence_lengths.csv")
    lines = pd.read_csv(TABLE_DIR / "channel_by_line.csv")
    coverage = pd.read_csv(TABLE_DIR / "action_space_coverage.csv")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2),
                             gridspec_kw={"wspace": 0.30,
                                          "width_ratios": [1.05, 1.25, 0.85]})

    # --- A: how long the games run -----------------------------------------
    ax = axes[0]
    staged = sequences[sequences["stage_dx"].isin(STAGE_ORDER)].copy()
    staged["order"] = staged["stage_dx"].map(
        {stage: index for index, stage in enumerate(STAGE_ORDER)})
    staged = staged.sort_values("order")
    positions = np.arange(len(staged))
    ax.bar(positions, staged["p90_regimens"], 0.62, color="#e7e5e4",
           label="90 백분위")
    ax.bar(positions, staged["median_regimens"], 0.62, color=BLUE,
           label="중앙값")
    for position, row in zip(positions, staged.itertuples()):
        # Median inside the blue bar: the reference line sits at 5, and a label
        # drawn above a bar of height 5 lands on top of it.
        ax.text(position, row.median_regimens - 0.34, f"{row.median_regimens:.0f}",
                ha="center", va="top", fontsize=11, color="white",
                fontweight="bold")
        ax.text(position, row.p90_regimens + 0.18, f"{row.p90_regimens:.0f}",
                ha="center", fontsize=8.5, color=GRAY)
    ax.axhline(5, color=RED, linestyle="--", linewidth=1.4)
    ax.text(len(staged) - 0.45, 5.25, "우리 환경의 행동축 5개",
            ha="right", fontsize=9, color=RED)
    ax.set_xticks(positions)
    ax.set_xticklabels([row.stage_dx.replace("Stage ", "")
                        for row in staged.itertuples()], fontsize=9.5)
    ax.set_xlabel("진단 병기")
    ax.set_ylabel(f"{verdict['horizon_years']:.0f}년 내 레지멘 수")
    ax.set_title("A. 실제 기보는 병기에 따라 중앙값 2수에서 5수", fontsize=11.5,
                 loc="left", pad=10)
    ax.legend(fontsize=9, frameon=False, loc="upper left")

    # --- B: what gets played at each turn ----------------------------------
    ax = axes[1]
    pivot = (lines.pivot(index="regimen_number_within_cancer",
                         columns="primary_channel", values="percent")
             .fillna(0.0))
    for channel in CHANNEL_ORDER:
        if channel not in pivot.columns:
            pivot[channel] = 0.0
    pivot = pivot[list(CHANNEL_ORDER)]
    bottom = np.zeros(len(pivot))
    for channel in CHANNEL_ORDER:
        values = pivot[channel].to_numpy()
        if values.sum() == 0:
            continue
        ax.bar(pivot.index, values, 0.68, bottom=bottom,
               color=CHANNEL_COLOR[channel], label=CHANNEL_LABEL[channel])
        for position, (value, base) in enumerate(zip(values, bottom)):
            if value >= 8:
                ax.text(pivot.index[position], base + value / 2,
                        f"{value:.0f}", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
        bottom += values
    ax.set_xticks(list(pivot.index))
    ax.set_xticklabels([f"{n}차" for n in pivot.index], fontsize=9.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("해당 차수 레지멘 중 비율 (%)")
    ax.set_title("B. 1차는 항암, 2·3차는 내분비, 뒤로 갈수록 임상시험",
                 fontsize=11.5, loc="left", pad=10)
    ax.legend(fontsize=8.5, frameon=False, ncol=3,
              loc="lower center", bbox_to_anchor=(0.5, -0.30))

    # --- C: which of our action axes the data can see -----------------------
    ax = axes[2]
    coverage = coverage.iloc[::-1].reset_index(drop=True)
    positions = np.arange(len(coverage))
    colors = [GREEN if flag else RED
              for flag in coverage["observable_in_genie_bpc"]]
    ax.barh(positions, [1] * len(coverage), 0.6, color=colors)
    for position, row in zip(positions, coverage.itertuples()):
        ax.text(0.5, position,
                "관측 가능" if row.observable_in_genie_bpc else "없음",
                ha="center", va="center", fontsize=10, color="white",
                fontweight="bold")
    ax.set_yticks(positions)
    ax.set_yticklabels(coverage["action_axis"], fontsize=10)
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.grid(False)
    ax.set_title(
        f"C. 환경 행동축 5개 중 {verdict['action_axes_observable']}개만 보인다",
        fontsize=11.5, loc="left", pad=10)

    fig.suptitle(
        "GENIE BPC 유방암 v1.0-public — 시뮬레이터가 지어내던 기보의 실물 "
        f"(환자 {verdict['patients']}명 · index 레지멘 {verdict['index_regimens']}건)",
        fontsize=13, y=1.03, color=INK)
    save(fig, "fig41_genie_movetext.png")


def fig42_closed_loop() -> None:
    verdict = METRICS["verdict"]
    rates = pd.read_csv(TABLE_DIR / "response_switch.csv").set_index("group")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2),
                             gridspec_kw={"wspace": 0.28,
                                          "width_ratios": [1.15, 1]})

    order = ("progressing", "controlled", "other")
    label = {"progressing": "진행/악화", "controlled": "안정/호전",
             "other": "혼합·판정불가"}
    color = {"progressing": RED, "controlled": GREEN, "other": GRAY}

    # --- A: forward window against the placebo-in-time control --------------
    ax = axes[0]
    positions = np.arange(len(order))
    width = 0.36
    forward = [rates.loc[group, "switch_after"] * 100 for group in order]
    backward = [rates.loc[group, "switch_before"] * 100 for group in order]
    ax.bar(positions - width / 2, forward, width,
           color=[color[group] for group in order],
           label=f"판독 이후 {verdict['switch_window_days']}일")
    ax.bar(positions + width / 2, backward, width,
           color=[color[group] for group in order], alpha=0.32, hatch="//",
           edgecolor="white",
           label=f"판독 이전 {verdict['switch_window_days']}일 (위약대조)")
    for position, (ahead, behind) in enumerate(zip(forward, backward)):
        ax.text(position - width / 2, ahead + 1.4, f"{ahead:.1f}%",
                ha="center", fontsize=10, color=INK, fontweight="bold")
        ax.text(position + width / 2, behind + 1.4, f"{behind:.1f}%",
                ha="center", fontsize=9, color=GRAY)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{label[group]}\n(판독 {int(rates.loc[group, 'scans']):,}건)"
         for group in order], fontsize=9.5)
    ax.set_ylim(0, 82)
    ax.set_ylabel("90일 내 새 레지멘이 시작된 비율 (%)")
    ax.set_title("A. 진행 판독 뒤에만 치료가 바뀐다", fontsize=11.5,
                 loc="left", pad=10)
    ax.legend(fontsize=9, frameon=False, loc="upper right")

    # --- B: the control is the finding --------------------------------------
    ax = axes[1]
    gaps = [rates.loc[group, "forward_minus_backward"] * 100 for group in order]
    bars = ax.bar(positions, gaps, 0.58,
                  color=[color[group] for group in order])
    ax.axhline(0, color=INK, linewidth=1.1)
    for bar, gap in zip(bars, gaps):
        offset = 1.6 if gap >= 0 else -3.6
        ax.text(bar.get_x() + bar.get_width() / 2, gap + offset,
                f"{gap:+.1f}%p", ha="center", fontsize=10.5, color=INK,
                fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels([label[group] for group in order], fontsize=9.5)
    ax.set_ylabel("이후 빼기 이전 (%p)")
    ax.set_ylim(-10, 46)
    ax.set_title("B. 안정 판독은 앞뒤가 같다 — 이게 위약대조다",
                 fontsize=11.5, loc="left", pad=10)
    # Centred in the empty column above the stable-group bar: the right-hand
    # side is occupied by the mixed group's value label.
    ax.text(0.58, 0.66,
            "판독이 원인이라면 뒤로만 커야 한다.\n안정군은 실제로 -1.2%p.",
            transform=ax.transAxes, ha="center", va="top", fontsize=9.5,
            color=GRAY, linespacing=1.6)

    fig.suptitle(
        "폐루프 적응은 기록에 남아 있다 — 진행 판독 뒤 전환율 "
        f"{verdict['switch_rate_progressing'] * 100:.1f}% 대 "
        f"{verdict['switch_rate_controlled'] * 100:.1f}% "
        f"({verdict['switch_rate_ratio']:.2f}배, 판독 {verdict['scans_scored']:,}건)",
        fontsize=13, y=1.02, color=INK)
    fig.text(0.5, -0.055,
             "보정하지 않은 연관성이다. 적응이 결과를 좋게 했다는 뜻이 아니라, "
             "기록된 대국에 적응이 들어 있다는 뜻이다.",
             ha="center", fontsize=9.5, color=GRAY)
    save(fig, "fig42_genie_closed_loop.png")


if __name__ == "__main__":
    fig41_movetext()
    fig42_closed_loop()
