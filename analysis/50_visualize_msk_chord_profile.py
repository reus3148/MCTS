"""Figure for the MSK-CHORD eligibility assessment (fig48).

Five panels, in the order the report argues them: how many adaptation
opportunities real care contains against the one our environment offers; how
that count depends on when the patient entered the record, which is the
headline's largest caveat; how much of our action space the release can see;
whether treatment follows the progression signal, with the backward window as
the control; and the bimodal chemotherapy-to-surgery gap that explains why the
timing prediction missed.

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

REPORT_DIR = ROOT / "reports" / "msk-chord-profile-v2.2"
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


def save(fig: plt.Figure, filename: str) -> None:
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(path, PUBLIC_DIR / filename)
    print(path)


def fig48_msk_chord_profile() -> None:
    verdict = METRICS["verdict"]
    points = pd.read_csv(TABLE_DIR / "decision_point_distribution.csv")
    coverage = pd.read_csv(TABLE_DIR / "action_space_coverage.csv")
    rates = pd.read_csv(TABLE_DIR / "response_switch.csv")
    intervals = pd.read_csv(TABLE_DIR / "timing_intervals.csv")
    entry = pd.read_csv(TABLE_DIR / "opportunity_by_entry.csv")
    opportunities = verdict["opportunities"]
    observed = verdict["opportunity_entry_within_90d_posthoc"]

    fig, axes = plt.subplots(1, 5, figsize=(25.5, 5.4),
                             gridspec_kw={"wspace": 0.32,
                                          "width_ratios": [1.15, 1.0, 1.0, 1.1, 1.15]})

    # --- A: how many places there are to adapt -------------------------------
    ax = axes[0]
    ax.bar(points["opportunities"], points["patients"], color=BLUE,
           alpha=0.85, width=0.85)
    tallest = float(points["patients"].max())
    median = opportunities["median"]
    ax.axvline(median, color=INK, linestyle="--", linewidth=1.4)
    ax.annotate(f"중앙값 {median:.0f}", xy=(median, tallest * 0.92),
                xytext=(median + 2.2, tallest * 0.92),
                color=INK, fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    ax.axvline(verdict["environment_opportunities"], color=RED, linewidth=2.2)
    ax.annotate(
        f"우리 환경: 1회\n(에피소드의 "
        f"{verdict['environment_opportunity_share']:.1%})",
        xy=(1, tallest * 0.45), xytext=(11.5, tallest * 0.62),
        color=RED, fontsize=9.5, fontweight="bold", linespacing=1.4,
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.set_xlim(0, 33)
    ax.set_xlabel("5년 내 적응 기회 (30일 내 판독은 1회로 합산)")
    ax.set_ylabel("환자 수")
    ax.set_title("A. 실제 진료에는 자리가 많다", fontsize=11.5, loc="left", pad=10)

    # --- B: when the patient entered the record ------------------------------
    ax = axes[1]
    label = {"<= 90d": "90일 내", "91-365d": "91-365일",
             "1-3y": "1-3년", "> 3y": "3년 초과"}
    positions = np.arange(len(entry))
    ax.bar(positions, entry["median"], color=BLUE, alpha=0.85, width=0.62)
    for index, row in enumerate(entry.itertuples()):
        ax.text(index, row.median + 0.22, f"{row.median:.0f}회", ha="center",
                fontsize=10, color=INK, fontweight="bold")
        ax.text(index, -1.05, f"0회 {row.zero:.0%}", ha="center", fontsize=8.5,
                color=RED if row.zero > 0.3 else GRAY)
        ax.text(index, -1.75, f"n={row.patients:,}", ha="center", fontsize=8,
                color=GRAY)
    ax.axhline(verdict["environment_opportunities"], color=RED, linewidth=2.0,
               linestyle="-")
    ax.text(0.03, 0.93, "빨간 선 = 우리 환경 1회", transform=ax.transAxes,
            color=RED, fontsize=9.5, fontweight="bold", ha="left")
    ax.set_ylim(-2.2, max(entry["median"]) * 1.26)
    ax.set_xticks(positions)
    ax.set_xticklabels([label[bucket] for bucket in entry["entry_bucket"]],
                       fontsize=9.5)
    ax.set_xlabel("진단 -> 시퀀싱 (기록 진입 시점)")
    ax.set_ylabel("적응 기회 중앙값")
    ax.set_title("B. 늦게 들어온 환자는 창이 비어 있다",
                 fontsize=11.5, loc="left", pad=10)

    # --- C: what the release can see -----------------------------------------
    ax = axes[2]
    order = ["timing", "surgery", "chemo", "endocrine", "radiation"]
    axis_label = {"timing": "선행/보조 순서", "surgery": "수술", "chemo": "항암",
                  "endocrine": "내분비", "radiation": "방사선"}
    indexed = coverage.set_index("axis").loc[order]
    colour = {"True": GREEN, "partial": AMBER, "False": PALE}
    bars = ax.barh(range(len(order)),
                   [indexed.loc[axis, "share"] * 100 for axis in order],
                   color=[colour[str(indexed.loc[axis, "level_resolution"])]
                          for axis in order], alpha=0.9)
    for index, axis in enumerate(order):
        share = indexed.loc[axis, "share"] * 100
        ax.text(share + 1.5, index, f"{share:.0f}%", va="center",
                fontsize=9.5, color=INK, fontweight="bold")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([axis_label[axis] for axis in order], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xlabel("그 수가 기록된 환자 (%)")
    ax.set_title("C. 다섯 축 모두 '뒀는지'는 보인다", fontsize=11.5, loc="left", pad=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colour[key], alpha=0.9)
               for key in ("True", "partial", "False")]
    ax.legend(handles, ["단까지 판독", "부분 판독", "단은 판독 불가"],
              fontsize=8.5, frameon=False, loc="lower right")
    del bars

    # --- D: does treatment follow the signal? --------------------------------
    ax = axes[3]
    groups = ["controlled", "indeterminate", "progressing"]
    names = {"controlled": "안정", "indeterminate": "판정 보류",
             "progressing": "진행"}
    indexed = rates.set_index("group")
    group_positions = np.arange(len(groups))
    width = 0.38
    forward = [indexed.loc[group, "switch_after"] * 100 for group in groups]
    backward = [indexed.loc[group, "switch_before"] * 100 for group in groups]
    ax.bar(group_positions - width / 2, forward, width, color=RED, alpha=0.9,
           label="판독 이후 90일")
    ax.bar(group_positions + width / 2, backward, width, color=GRAY, alpha=0.75,
           label="판독 이전 90일 (위약대조)")
    for index, (ahead, behind) in enumerate(zip(forward, backward)):
        ax.text(index - width / 2, ahead + 1.6, f"{ahead:.0f}%", ha="center",
                fontsize=9.5, color=INK, fontweight="bold")
        ax.text(index + width / 2, behind + 1.6, f"{behind:.0f}%", ha="center",
                fontsize=9, color=GRAY)
    ax.set_xticks(group_positions)
    ax.set_xticklabels([names[group] for group in groups], fontsize=10)
    ax.set_ylim(0, 82)
    ax.set_ylabel("전신치료가 새로 시작된 비율 (%)")
    ax.set_title(f"D. 신호 뒤에 치료가 바뀐다 ({verdict['switch_rate_ratio']:.2f}배)",
                 fontsize=11.5, loc="left", pad=10)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")

    # --- E: the gap that explains the missed prediction ----------------------
    ax = axes[4]
    shares = intervals["share"] * 100
    colours = [GRAY if "adjuvant (surgery" in reading or "chemo" in reading
               else (BLUE if "neoadjuvant" in reading else PALE)
               for reading in intervals["reading"]]
    ax.bar(range(len(intervals)), shares, color=colours, alpha=0.9)
    ax.set_xticks(range(len(intervals)))
    ax.set_xticklabels(intervals["days_surgery_minus_chemo"], rotation=45,
                       ha="right", fontsize=8.5)
    for index, share in enumerate(shares):
        if share >= 3:
            ax.text(index, share + 0.9, f"{share:.0f}%", ha="center",
                    fontsize=9, color=INK, fontweight="bold")
    ax.set_ylim(0, max(shares) * 1.22)
    ax.set_xlabel("수술일 - 첫 항암일 (일)")
    ax.set_ylabel("환자 (%)")
    raw = verdict["neoadjuvant_share"] * 100
    restricted = verdict["neoadjuvant_share_interval_restricted_posthoc"] * 100
    ax.set_title(f"E. 날짜만 보면 {raw:.0f}%, 간격을 제한하면 {restricted:.0f}%",
                 fontsize=11.5, loc="left", pad=10)
    ax.text(0.97, 0.95,
            f"사전 구간 {verdict['neoadjuvant_band'][0]:.0%}-"
            f"{verdict['neoadjuvant_band'][1]:.0%}\n예측 6 빗나감",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            color=RED, fontweight="bold", linespacing=1.4)

    fig.suptitle(
        f"MSK-CHORD 유방암 {verdict['cohort']:,}명 — 진단 90일 내 진입군의 적응 기회 "
        f"중앙값 {observed['median']:.0f}회·2회 이상 {observed['at_least_two']:.0%} "
        f"(우리 환경 1회·{verdict['environment_opportunity_share']:.1%}), "
        f"진행 판독 뒤 치료 변경 {verdict['switch_rate_progressing']:.0%} 대 안정 "
        f"{verdict['switch_rate_controlled']:.0%} "
        f"(위약대조 {verdict['forward_minus_backward_progressing']:+.2f} 대 "
        f"{verdict['forward_minus_backward_controlled']:+.2f})",
        fontsize=13, y=1.04, color=INK)
    fig.text(0.5, -0.10,
             "기술적 서술이다. 전환율은 보정하지 않았고 적응증에 교란되어 있다. "
             "코호트는 진단에서 시퀀싱까지 생존한 환자로 조건화되어 있고"
             f"(중앙값 {verdict['days_dx_to_sequencing_median']:.0f}일, p75 1,905일) "
             "이 코호트의 생존을 좌측 절단 처리 없이 쓰지 않는다. "
             "자료: MSK-CHORD (MSK, Nature 2024), CC BY-NC-ND 4.0.",
             ha="center", fontsize=9.5, color=GRAY)
    save(fig, "fig48_msk_chord_profile.png")


if __name__ == "__main__":
    fig48_msk_chord_profile()
