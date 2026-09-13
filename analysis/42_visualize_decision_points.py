"""Figure for the second-decision-point experiment (fig44).

Three things side by side: whether the gap moved when a decision was added
(and whether the inert version of the same phase moved it), how much more of
the game the policy could now adapt in, and where in the game MCTS chose to
decline - the horizon-artefact diagnostic that decides how the first panel may
be read.

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

REPORT_DIR = ROOT / "reports" / "decision-points-v1.8"
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

ORDER = ("v05", "inert", "decide", "defer")
ARM_LABEL = {
    "v05": "A v0.5\n(결정 없음)",
    "inert": "B v0.6 영대조\n(효과 없음)",
    "decide": "C v0.6\nMCTS 결정",
    "defer": "D v0.6\n규칙에 위임",
}
ARM_COLOR = {"v05": GRAY, "inert": AMBER, "decide": BLUE, "defer": "#93c5fd"}


def save(fig: plt.Figure, filename: str) -> None:
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(path, PUBLIC_DIR / filename)
    print(path)


def fig44_decision_points() -> None:
    verdict = METRICS["verdict"]
    arms = pd.read_csv(TABLE_DIR / "arms.csv").set_index("arm")
    by_year = pd.read_csv(TABLE_DIR / "salvage_by_recurrence_year.csv")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4),
                             gridspec_kw={"wspace": 0.34,
                                          "width_ratios": [1.3, 0.9, 1.0]})

    # --- A: the gap under four arms ------------------------------------------
    ax = axes[0]
    positions = np.arange(len(ORDER))
    gaps = [arms.loc[arm, "utility_gap"] for arm in ORDER]
    errors = [arms.loc[arm, "standard_error"] for arm in ORDER]
    ax.bar(positions, gaps, 0.6, color=[ARM_COLOR[arm] for arm in ORDER],
           yerr=errors, capsize=5, ecolor=INK)
    top = max(g + e for g, e in zip(gaps, errors))
    for position, (gap, error) in enumerate(zip(gaps, errors)):
        ax.text(position, gap + error + top * 0.025, f"{gap:+.4f}", ha="center",
                fontsize=10.5, color=INK, fontweight="bold")
    # Bracket C-D: the value of the decision point.
    c_index, d_index = ORDER.index("decide"), ORDER.index("defer")
    bracket_y = top * 1.22
    ax.plot([c_index, c_index, d_index, d_index],
            [bracket_y - top * 0.03, bracket_y, bracket_y, bracket_y - top * 0.03],
            color=INK, linewidth=1.1)
    value = verdict["value_of_decision_point"]
    z = verdict["decide_vs_defer"]["z"]
    ax.text((c_index + d_index) / 2, bracket_y + top * 0.03,
            f"C - D = {value:+.4f}  (z = {z:.2f})", ha="center", fontsize=10,
            color=INK, fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels([ARM_LABEL[arm] for arm in ORDER], fontsize=9)
    ax.set_ylabel("MCTS 빼기 NCCN 효용 격차")
    ax.set_ylim(0, top * 1.42)
    ax.set_title("A. 결정 지점을 하나 더 주면 격차가 어떻게 되나",
                 fontsize=11.5, loc="left", pad=10)

    # --- B: how much of the game can now be adapted in -----------------------
    ax = axes[1]
    pair = ("v05", "decide")
    positions = np.arange(len(pair))
    shares = [arms.loc[arm, "mcts_episodes_with_adaptation_pct"] for arm in pair]
    bars = ax.bar(positions, shares, 0.55, color=[GRAY, BLUE])
    for bar, share in zip(bars, shares):
        ax.text(bar.get_x() + bar.get_width() / 2, share + 1.2, f"{share:.1f}%",
                ha="center", fontsize=11, color=INK, fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels(["v0.5\n(응답만)", "v0.6\n(응답 + 재발)"], fontsize=9.5)
    ax.set_ylabel("적응할 것이 하나라도 있는 에피소드 (%)")
    ax.set_ylim(0, max(max(shares) * 1.3, 10))
    recurrence = arms.loc["decide", "mcts_recurrence_pct"]
    ax.set_title(f"B. 적응 기회 — 재발 {recurrence:.1f}%가 더해진다",
                 fontsize=11.5, loc="left", pad=10)

    # --- C: where MCTS declines ----------------------------------------------
    ax = axes[2]
    if by_year.empty:
        ax.text(0.5, 0.5, "구제 결정 없음", ha="center", va="center",
                transform=ax.transAxes)
    else:
        positions = np.arange(len(by_year))
        colors = [RED if int(y) == 4 else BLUE for y in by_year["recurrence_year"]]
        bars = ax.bar(positions, by_year["declined_pct"], 0.6, color=colors)
        for bar, (pct, n) in zip(bars, zip(by_year["declined_pct"], by_year["decisions"])):
            ax.text(bar.get_x() + bar.get_width() / 2, pct + 1.5,
                    f"{pct:.0f}%\n(n={int(n):,})", ha="center", fontsize=9,
                    color=INK)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{int(y)}년차" for y in by_year["recurrence_year"]],
                           fontsize=9.5)
        # Percent axis: leave room for the value labels without running past
        # 100 by more than the labels need.
        ax.set_ylim(0, min(max(float(by_year["declined_pct"].max()) * 1.35, 10), 122))
    ax.set_ylabel("MCTS가 구제를 거절한 비율 (%)")
    ax.set_title("C. 언제 거절하나 — 지평 인공물 진단",
                 fontsize=11.5, loc="left", pad=10)
    ax.text(0.02, 0.97,
            "5년차 재발은 치료할 해가 남지 않아\n결정 자체가 열리지 않는다.",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
            color=GRAY, linespacing=1.5)

    fig.suptitle(
        "재발에 결정 지점을 하나 더 열었을 때 — v0.6 환경 "
        "(환자 40명 · 시드 12 · 예산 1024 · 치료 중립 보상모형)",
        fontsize=13, y=1.03, color=INK)
    fig.text(0.5, -0.05,
             "구제치료의 이득(사망 위험비 0.85)·독성(0.20)·부담(0.10)은 우리가 선언한 합성 가정이다. "
             "임상 검토 전까지 효과가 아니다.",
             ha="center", fontsize=9.5, color=GRAY)
    save(fig, "fig44_decision_points.png")


if __name__ == "__main__":
    fig44_decision_points()
