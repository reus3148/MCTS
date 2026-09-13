"""Figure for the horizon experiment (fig45).

Four things side by side: what the cliff was cutting off (GENIE BPC survival
after advanced disease), whether the salvage decision still collapses by
recurrence year once life past the horizon counts, whether the guideline
policy still loses on the salvage channel, and where the gap lands under the
four v0.7 arms next to the v0.5 baseline.

Every number is read from the committed ``metrics.json`` and ``tables/`` of
this report and of ``reports/decision-points-v1.8``.
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

REPORT_DIR = ROOT / "reports" / "horizon-v1.9"
TABLE_DIR = REPORT_DIR / "tables"
FIGURE_DIR = REPORT_DIR / "figures"
PUBLIC_DIR = ROOT / "public" / "figures"
PRIOR_DIR = ROOT / "reports" / "decision-points-v1.8"
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
PRIOR = json.loads((PRIOR_DIR / "metrics.json").read_text(encoding="utf-8"))


def save(fig: plt.Figure, filename: str) -> None:
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(path, PUBLIC_DIR / filename)
    print(path)


def fig45_horizon() -> None:
    verdict = METRICS["verdict"]
    arms = pd.read_csv(TABLE_DIR / "arms.csv").set_index("arm")
    by_year_new = pd.read_csv(TABLE_DIR / "salvage_by_recurrence_year.csv")
    by_year_old = pd.read_csv(PRIOR_DIR / "tables" / "salvage_by_recurrence_year.csv")
    genie = pd.read_csv(TABLE_DIR / "genie_post_advanced_survival.csv")

    fig, axes = plt.subplots(1, 4, figsize=(20.5, 5.3),
                             gridspec_kw={"wspace": 0.36,
                                          "width_ratios": [0.85, 1.05, 0.8, 1.3]})

    # --- A: what the cliff was cutting off -----------------------------------
    ax = axes[0]
    ax.plot(genie["years"], genie["survival"] * 100, marker="o", color=INK,
            linewidth=1.8)
    for year, value in zip(genie["years"], genie["survival"]):
        # The five-year point sits on the horizon line; label it to the left.
        ax.text(year - (0.45 if year == 5 else 0), value * 100 + 3.5,
                f"{value * 100:.0f}%", ha="center", fontsize=9, color=INK)
    ax.axvline(5, color=RED, linestyle="--", linewidth=1.3)
    ax.text(5.15, 92, "우리 지평\n(5년)", fontsize=9, color=RED, va="top")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 105)
    ax.set_xlabel("진행성 질환 진단 후 경과 (년)")
    ax.set_ylabel("생존 (%)")
    survival = verdict["genie_post_advanced_survival"]
    ax.set_title(
        f"A. GENIE BPC 진행 후 생존 — 중앙 {survival['median_years']:.1f}년",
        fontsize=11.5, loc="left", pad=10)

    # --- B: the artefact, before and after -----------------------------------
    ax = axes[1]
    years = sorted(set(by_year_old["recurrence_year"]) | set(by_year_new["recurrence_year"]))
    positions = np.arange(len(years))
    width = 0.38
    old = [float(by_year_old.set_index("recurrence_year")["declined_pct"].get(y, np.nan))
           for y in years]
    new = [float(by_year_new.set_index("recurrence_year")["declined_pct"].get(y, np.nan))
           for y in years]
    ax.bar(positions - width / 2, old, width, color="#fca5a5", label="v0.6 (5년 뒤 = 0)")
    ax.bar(positions + width / 2, new, width, color=BLUE, label="v0.7 (말기 가치 10년)")
    for position, (o, n) in enumerate(zip(old, new)):
        if not np.isnan(o):
            ax.text(position - width / 2, o + 2, f"{o:.0f}", ha="center", fontsize=9, color=RED)
        if not np.isnan(n):
            ax.text(position + width / 2, n + 2, f"{n:.0f}", ha="center", fontsize=9,
                    color=INK, fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{int(y)}년차" for y in years], fontsize=9.5)
    ax.set_ylim(0, 122)
    ax.set_ylabel("MCTS가 구제를 거절한 비율 (%)")
    ax.set_title("B. 재발 연도별 거절률 — 지평 인공물이 사라졌나",
                 fontsize=11.5, loc="left", pad=10)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")

    # --- C: does the guideline still lose on the channel ---------------------
    ax = axes[2]
    old_cost = verdict["nccn_forced_salvage_cost_v1_8"]
    new_cost = verdict["nccn_forced_salvage_cost"]
    values = [old_cost["difference"], new_cost["difference"]]
    errors = [old_cost["standard_error"], new_cost["standard_error"]]
    colors = [RED if v < 0 else GREEN for v in values]
    bars = ax.bar([0, 1], values, 0.55, color=colors, yerr=errors, capsize=5, ecolor=INK)
    ax.axhline(0, color=INK, linewidth=1.0)
    for bar, value, error, stat in zip(bars, values, errors, (old_cost, new_cost)):
        # Labels sit outside the bar on the side it points to, clear of the
        # error bar and the zero line.
        if value >= 0:
            y, va = value + error + 0.0002, "bottom"
        else:
            y, va = value - error - 0.0002, "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y,
                f"{value:+.4f}\n(z = {stat['z']:.1f})", ha="center", va=va,
                fontsize=9.5, color=INK, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["v0.6\n(v1.8)", "v0.7\n(v1.9)"], fontsize=9.5)
    span = max(abs(v) + e for v, e in zip(values, errors)) * 1.9
    ax.set_ylim(-span, span)
    ax.set_ylabel("NCCN 효용: 선언된 구제 - 효과 없는 구제")
    ax.set_title("C. 항상 치료하는 정책이 구제로 손해를 보나",
                 fontsize=11.5, loc="left", pad=10)

    # --- D: the gap under five conditions ------------------------------------
    ax = axes[3]
    labels = ["v0.5\n(v1.8 A)", "A v0.7\n말기만", "B v0.7\n영대조", "C v0.7\nMCTS 결정",
              "D v0.7\n규칙 위임"]
    v05 = PRIOR["arms"]["v05"]
    gaps = [v05["utility_gap"]] + [arms.loc[a, "utility_gap"] for a in ("tail", "inert", "decide", "defer")]
    errors = [v05["standard_error"]] + [arms.loc[a, "standard_error"] for a in ("tail", "inert", "decide", "defer")]
    colors = [GRAY, "#a8a29e", AMBER, BLUE, "#93c5fd"]
    positions = np.arange(len(labels))
    ax.bar(positions, gaps, 0.6, color=colors, yerr=errors, capsize=4, ecolor=INK)
    top = max(g + e for g, e in zip(gaps, errors))
    for position, (gap, error) in enumerate(zip(gaps, errors)):
        ax.text(position, gap + error + top * 0.02, f"{gap:+.4f}", ha="center",
                fontsize=9.5, color=INK, fontweight="bold")
    c_index, d_index = 3, 4
    bracket_y = top * 1.2
    ax.plot([c_index, c_index, d_index, d_index],
            [bracket_y - top * 0.03, bracket_y, bracket_y, bracket_y - top * 0.03],
            color=INK, linewidth=1.1)
    value = verdict["value_of_decision_point"]
    z = verdict["decide_vs_defer"]["z"]
    ax.text((c_index + d_index) / 2, bracket_y + top * 0.03,
            f"C - D = {value:+.4f} (z = {z:.2f})", ha="center", fontsize=9.5,
            color=INK, fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("MCTS 빼기 NCCN 효용 격차")
    ax.set_ylim(0, top * 1.4)
    tail_shift = verdict["tail_vs_v05"]
    ax.set_title(
        f"D. 격차 — 말기 가치만으로 {tail_shift['difference']:+.4f} (z = {tail_shift['z']:.2f})",
        fontsize=11.5, loc="left", pad=10)

    fig.suptitle(
        "5년 뒤를 0으로 치던 지평에 말기 가치 10년을 더했을 때 — v0.7 환경 "
        "(환자 40명 · 시드 12 · 예산 1024 · 치료 중립 보상모형)",
        fontsize=13, y=1.03, color=INK)
    fig.text(0.5, -0.05,
             "말기 가치는 환경 자체의 연간 사건 모형을 결정 없이 10년 더 이어 간 기댓값이다. "
             "구제 파라미터는 v0.6과 같다 — 움직인 것은 전부 지평의 몫이다.",
             ha="center", fontsize=9.5, color=GRAY)
    save(fig, "fig45_horizon.png")


if __name__ == "__main__":
    fig45_horizon()
