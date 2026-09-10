"""Figure for the closed-loop value experiment (fig43).

Three things have to be visible at once: how little the gap moves when the
planner is blinded, that NCCN does not move at all (the design's own control),
and what blinding does to the one action the option value lives in.

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

REPORT_DIR = ROOT / "reports" / "closed-loop-value-v1.7"
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

ARM_LABEL = {
    "none": "A 그대로",
    "label": "B 라벨만 가림",
    "full": "C 완전 블라인드",
}
ARM_COLOR = {"none": BLUE, "label": AMBER, "full": RED}
ORDER = ("none", "label", "full")


def save(fig: plt.Figure, filename: str) -> None:
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(path, PUBLIC_DIR / filename)
    print(path)


def fig43_closed_loop_value() -> None:
    verdict = METRICS["verdict"]
    arms = pd.read_csv(TABLE_DIR / "arms.csv").set_index("arm")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2),
                             gridspec_kw={"wspace": 0.32,
                                          "width_ratios": [1.15, 1.15, 1]})

    positions = np.arange(len(ORDER))

    # --- A: the gap barely moves -------------------------------------------
    ax = axes[0]
    gaps = [arms.loc[arm, "utility_gap"] for arm in ORDER]
    errors = [arms.loc[arm, "standard_error"] for arm in ORDER]
    ax.bar(positions, gaps, 0.58, color=[ARM_COLOR[arm] for arm in ORDER],
           yerr=errors, capsize=5, ecolor=INK)
    for position, (gap, error) in enumerate(zip(gaps, errors)):
        ax.text(position, gap + error + 0.0006, f"{gap:+.4f}", ha="center",
                fontsize=10.5, color=INK, fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels([ARM_LABEL[arm] for arm in ORDER], fontsize=9.5)
    ax.set_ylabel("MCTS 빼기 NCCN 효용 격차")
    ax.set_ylim(0, max(gap + err for gap, err in zip(gaps, errors)) * 1.30)
    ax.set_title(
        f"A. 응답을 완전히 가려도 {verdict['closed_loop_value']:+.4f}만 움직인다",
        fontsize=11.5, loc="left", pad=10)

    # --- B: which policy moved ---------------------------------------------
    ax = axes[1]
    width = 0.36
    mcts = [arms.loc[arm, "mcts_utility"] for arm in ORDER]
    nccn = [arms.loc[arm, "nccn_utility"] for arm in ORDER]
    ax.bar(positions - width / 2, mcts, width, color=BLUE, label="MCTS")
    ax.bar(positions + width / 2, nccn, width, color=GRAY, label="NCCN")
    for position, (m, n) in enumerate(zip(mcts, nccn)):
        ax.text(position - width / 2, m + 0.0012, f"{m:.4f}", ha="center",
                fontsize=9, color=INK)
        ax.text(position + width / 2, n + 0.0012, f"{n:.4f}", ha="center",
                fontsize=9, color=GRAY)
    low = min(min(mcts), min(nccn))
    high = max(max(mcts), max(nccn))
    span = max(high - low, 1e-4)
    ax.set_ylim(low - span * 1.2, high + span * 1.6)
    ax.set_xticks(positions)
    ax.set_xticklabels([ARM_LABEL[arm] for arm in ORDER], fontsize=9.5)
    ax.set_ylabel("평균 효용")
    ax.set_title("B. NCCN은 세 팔에서 완전히 같다 — 설계의 자체 점검",
                 fontsize=11.5, loc="left", pad=10)
    ax.legend(fontsize=9, frameon=False, loc="upper right")

    # --- C: the option value ------------------------------------------------
    ax = axes[2]
    shares = [arms.loc[arm, "mcts_neoadjuvant_pct"] for arm in ORDER]
    bars = ax.bar(positions, shares, 0.58,
                  color=[ARM_COLOR[arm] for arm in ORDER])
    for bar, share in zip(bars, shares):
        ax.text(bar.get_x() + bar.get_width() / 2, share + max(shares) * 0.02,
                f"{share:.1f}%", ha="center", fontsize=10.5, color=INK,
                fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels([ARM_LABEL[arm] for arm in ORDER], fontsize=9.5)
    ax.set_ylabel("MCTS가 선행치료를 고른 비율 (%)")
    ax.set_ylim(0, max(max(shares) * 1.85, 1.0))
    ax.set_title("C. 적응할 기회 자체가 5분의 1뿐이고, 가려도 안 변한다",
                 fontsize=11.5, loc="left", pad=10)
    ax.text(0.5, 0.98,
            "선행치료를 고른 에피소드만 응답을 뽑는다.\n"
            f"나머지 {100 - shares[0]:.1f}%는 애초에 적응할 것이 없다.\n"
            f"블라인드 계획이 불법이 된 경우 {verdict['fallbacks_full']}회.",
            transform=ax.transAxes, ha="center", va="top", fontsize=9,
            color=GRAY, linespacing=1.6)

    fig.suptitle(
        "폐루프 적응의 값 — 응답 채널을 탐색 정책에서 떼어냈을 때 "
        f"(환자 40명 · 시드 12 · 예산 1024 · 치료 중립 보상모형)",
        fontsize=13, y=1.03, color=INK)
    fig.text(0.5, -0.05,
             "환경은 세 팔 모두 동일하다. 바뀌는 것은 계획하는 쪽이 무엇을 보는가뿐이다.",
             ha="center", fontsize=9.5, color=GRAY)
    save(fig, "fig43_closed_loop_value.png")


if __name__ == "__main__":
    fig43_closed_loop_value()
