"""Figure for the adaptive-channel stakes map (fig51).

Four panels in the order the argument runs: turning the channel's size up does
nothing for either arm, because the declared ladder has a dominant rung; pricing
strength properly opens a narrow window where the lines actually trade off; in
that window the grades stop agreeing; and at the window's peak the information
is worth 25x what v2.4 measured, on half the stakes - but still 87% of the
noise floor rather than past it.

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

REPORT_DIR = ROOT / "reports" / "stakes-map-v2.5"
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
PALE = "#a8a29e"

METRICS = json.loads((REPORT_DIR / "metrics.json").read_text(encoding="utf-8"))

LINE_LABEL = {"none": "거절", "mild": "약한 단", "systemic": "표준 단",
              "intensive": "강한 단"}
LINE_COLOUR = {"none": PALE, "mild": GREEN, "systemic": BLUE, "intensive": RED}
ORDER = ("none", "mild", "systemic", "intensive")


def save(fig: plt.Figure, filename: str) -> None:
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(path, PUBLIC_DIR / filename)
    print(path)


def fig51_stakes_map() -> None:
    verdict = METRICS["verdict"]
    summary = pd.read_csv(TABLE_DIR / "stakes_summary.csv")
    costs = pd.read_csv(TABLE_DIR / "cost_sweep.csv")
    choices = pd.read_csv(TABLE_DIR / "line_choice_by_cost.csv")
    post = verdict["cost_sweep_posthoc"]
    noise = verdict["action_value_noise"]

    fig, axes = plt.subplots(1, 4, figsize=(20.5, 5.3),
                             gridspec_kw={"wspace": 0.33,
                                          "width_ratios": [1.0, 1.05, 1.1, 1.1]})

    # --- A: making the channel bigger does nothing ---------------------------
    ax = axes[0]
    for arm, colour, label in (("binary", GRAY, "이진 (오늘 구조)"),
                               ("ladder", BLUE, "사다리 (선택지 4개)")):
        block = summary[summary["arm"] == arm].sort_values("scale")
        ax.plot(block["scale"], block["channel_stakes"], marker="o", markersize=4.5,
                color=colour, linewidth=2, label=label)
    today = verdict["today_config_scale"]
    ax.axvline(today, color=INK, linestyle="--", linewidth=1.2)
    ax.text(today + 0.02, summary["channel_stakes"].max() * 0.90,
            f"오늘의 config\n({today:.2f})", fontsize=9, color=INK,
            fontweight="bold", linespacing=1.4)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("판돈 척도 (0 = 채널 없음)")
    ax.set_ylabel("채널 판돈 (최선 선택의 평균 이점)")
    ax.set_title("A. 채널을 키우면 판돈은 커진다", fontsize=11.5, loc="left", pad=10)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.text(0.97, 0.12, "그런데 정보가치는 둘 다 0", transform=ax.transAxes,
            ha="right", fontsize=10, color=RED, fontweight="bold")

    # --- B: pricing strength opens a window ----------------------------------
    ax = axes[1]
    ax.plot(costs["cost_multiple"], costs["grade_switch_share"] * 100,
            marker="o", markersize=4.5, color=AMBER, linewidth=2,
            label="등급에 따라 답이 갈림")
    ax.plot(costs["cost_multiple"], costs["treat_share"] * 100, marker="s",
            markersize=4, color=GRAY, linewidth=1.6, linestyle="--",
            label="치료를 고름")
    window = costs[costs["grade_switch_share"] > 0]
    if len(window):
        ax.axvspan(float(window["cost_multiple"].min()) - 0.1,
                   float(window["cost_multiple"].max()) + 0.1,
                   color="#fde68a", alpha=0.45)
    ax.axvline(1.0, color=INK, linestyle="--", linewidth=1.2)
    ax.text(1.12, 92, "오늘의 선언", fontsize=9, color=INK, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 4, 6, 8, 12])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(-4, 104)
    ax.set_xlabel("강한 단의 값을 몇 배로 (판돈 1.0 고정)")
    ax.set_ylabel("결정의 비율 (%)")
    ax.set_title("B. 값을 매겨야 창이 열린다", fontsize=11.5, loc="left", pad=10)
    ax.legend(fontsize=8.5, frameon=False, loc="center left")

    # --- C: what gets chosen, by grade, at the peak --------------------------
    ax = axes[2]
    peak_multiple = post["peak_cost_multiple"]
    block = choices[np.isclose(choices["cost_multiple"], peak_multiple)]
    grades = ["controlled", "indeterminate", "progressing"]
    grade_label = {"controlled": "안정", "indeterminate": "판정 보류",
                   "progressing": "진행"}
    bottom = np.zeros(len(grades))
    for line in ORDER:
        values = []
        for grade in grades:
            row = block[(block["grade"] == grade) & (block["best_line"] == line)]
            total = block[block["grade"] == grade]["decisions"].sum()
            values.append(float(row["decisions"].sum() / total * 100) if total else 0.0)
        values = np.array(values)
        if values.sum() == 0:
            continue
        ax.bar(range(len(grades)), values, bottom=bottom, width=0.58,
               color=LINE_COLOUR[line], alpha=0.9, label=LINE_LABEL[line])
        for index, value in enumerate(values):
            if value >= 7:
                ax.text(index, bottom[index] + value / 2, f"{value:.0f}%",
                        ha="center", va="center", fontsize=9.5, color="white",
                        fontweight="bold")
        bottom += values
    ax.set_xticks(range(len(grades)))
    ax.set_xticklabels([grade_label[grade] for grade in grades], fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_ylabel("그 단을 고른 결정 (%)")
    ax.set_title(f"C. 봉우리(값 {peak_multiple:.1f}배)에서는 등급마다 다르게 고른다",
                 fontsize=11.5, loc="left", pad=10)
    ax.legend(fontsize=8.5, frameon=False, loc="lower left", ncol=2)

    # --- D: and what is it worth? --------------------------------------------
    ax = axes[3]
    ax.plot(costs["cost_multiple"], costs["information_per_decision"],
            marker="o", markersize=4.5, color=BLUE, linewidth=2)
    ax.axhline(noise, color=RED, linewidth=2.2)
    ax.text(1.05, noise * 1.03, f"예산 1024의 행동값 잡음 {noise:.2f}",
            fontsize=9.5, color=RED, fontweight="bold", va="bottom")
    best = post["max_information_per_decision"]
    ax.annotate(
        f"봉우리 {best:.5f}\n잡음의 {post['share_of_noise_floor']:.0%}",
        xy=(peak_multiple, best), xytext=(peak_multiple * 1.35, best * 0.52),
        fontsize=10, fontweight="bold", color=INK, linespacing=1.4,
        arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.axhline(verdict["v24_information_value"], color=GRAY, linestyle=":",
               linewidth=1.6)
    ax.text(1.05, verdict["v24_information_value"] * 1.4,
            f"v2.4 (등급만, 이진) {verdict['v24_information_value']:.5f}",
            fontsize=9, color=GRAY, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 4, 6, 8, 12])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0, noise * 1.22)
    ax.set_xlabel("강한 단의 값을 몇 배로")
    ax.set_ylabel("등급을 아는 것의 값 (완전정보)")
    ax.set_title("D. 25배가 됐지만, 아직 잡음 아래다",
                 fontsize=11.5, loc="left", pad=10)

    fig.suptitle(
        f"적응 채널은 작은 게 아니라 모양이 문제였다 — 봉우리에서 판돈은 v2.4의 절반"
        f"({post['peak_channel_stakes']:.4f} 대 {verdict['v24_channel_stakes']})인데 "
        f"정보가치는 {post['information_multiple_over_v24']:.0f}배"
        f"({best:.5f} 대 {verdict['v24_information_value']}), "
        f"판돈 대비 {post['peak_grade_conditional_share_of_stakes']:.1%} 대 "
        f"{post['v24_grade_conditional_share_of_stakes']:.1%}. "
        f"같은 판돈의 이진 팔은 0 (환자 40명 · v0.7 · 탐색 없음)",
        fontsize=12.5, y=1.045, color=INK)
    fig.text(0.5, -0.09,
             "전부 우리가 선언한 파라미터의 성질이지 환자의 성질이 아니다. 사다리의 끝점과 "
             "값은 선언값이고 어떤 자료로도 추정할 수 없다. 사전 등록한 사다리는 강한 단이 "
             "모두를 지배해 선택지 넷이 하나처럼 굴었다 — 값 다이얼은 그 설계 오류를 재기 위한 "
             "사후 장치다. 질병 등급은 분리 1.0(v2.4).",
             ha="center", fontsize=9.5, color=GRAY)
    save(fig, "fig51_stakes_map.png")


if __name__ == "__main__":
    fig51_stakes_map()
