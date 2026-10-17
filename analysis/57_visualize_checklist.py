"""The reusable contribution, as one figure (fig52).

Eight versions of this project were spent discovering that a comparison between
a searched policy and a guideline can be wrong in ways that have nothing to do
with either. This figure is what we would hand someone starting the same work:
five diagnostics, what each one asks, and what it cost us to learn it.

Unlike every other figure here it is not a plot of a result - it is the
procedure. It belongs in the discussion section, where the claim is that the
procedure transfers to other diseases and other algorithms.

Numbers come from the committed ``metrics.json`` of the report that produced
each diagnostic, so the figure cannot drift from the results it summarises.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "reports" / "procedure-checklist"
FIGURE_DIR = REPORT_DIR / "figures"
PUBLIC_DIR = ROOT / "public" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams.update({
    "font.family": "Malgun Gothic",
    "axes.unicode_minus": False,
    "figure.facecolor": "#fafaf9",
    "axes.facecolor": "#fafaf9",
})

INK = "#1c1917"
BLUE = "#2563eb"
GREEN = "#15803d"
RED = "#dc2626"
AMBER = "#d97706"
GRAY = "#78716c"
PALE = "#e7e5e4"


def metric(label: str) -> dict:
    return json.loads(
        (ROOT / "reports" / label / "metrics.json").read_text(encoding="utf-8"))["verdict"]


def build_rows() -> list[dict]:
    """The five diagnostics, with their numbers read from committed metrics."""
    v14 = metric("reward-confounding-v1.4")
    v15 = metric("channel-decomposition-v1.5")
    v17 = metric("closed-loop-value-v1.7")
    v21 = metric("benefit-tipping-v2.1")
    v22 = metric("msk-chord-profile-v2.2")
    v25 = metric("stakes-map-v2.5")

    share_reward = v14["share_of_gap_from_reward_model"]
    share_cost = v15["share_of_gap_from_declared_cost"] * (1 - share_reward)
    observed = abs(v17["closed_loop_value"])
    mde = v17["minimum_detectable_difference"]
    opportunities = v22["opportunity_entry_within_90d_posthoc"]

    return [
        {
            "n": "1",
            "title": "채널 대칭성",
            "ask": "두 정책이 같은 채널에 접근하는가?",
            "how": "한쪽만 쓰는 수준에 붙은 이득·비용을 끄고\n격차를 다시 잰다",
            "found": f"아니오 — 격차의 {share_cost:.0%}가\n한쪽만 무는 비용이었다",
            "cite": "v1.5",
            "colour": RED,
        },
        {
            "n": "2",
            "title": "미선언 이득",
            "ask": "보상모형이 치료 효과까지 학습했는가?",
            "how": "치료 계수를 중립화하고 짝지은 차이를 본다",
            "found": f"예 — 격차의 {share_reward:.0%}\n(z = {v14['paired_z']:.2f})",
            "cite": "v1.4",
            "colour": RED,
        },
        {
            "n": "3",
            "title": "적응 자리",
            "ask": "순차적 방법이 쓸 분기가 있는가?",
            "how": "응답을 계획자에게서 가리고,\n실제 기보의 적응 기회를 센다",
            "found": (f"거의 없음 — 환경 1회·"
                      f"{v22['environment_opportunity_share']:.0%}\n실제 "
                      f"{opportunities['median']:.0f}회·"
                      f"{opportunities['at_least_two']:.0%}"),
            "cite": "v1.7 · v2.2",
            "colour": AMBER,
        },
        {
            "n": "4",
            "title": "결정 지점의 값",
            "ask": "판돈 × 상태 의존 × 해상도가 다 있는가?",
            "how": "탐색 전에 환경의 산술로 지도를 그린다 (5초)",
            "found": (f"다섯 번 중 네 번 하나씩 빠짐\n"
                      f"최선도 잡음의 "
                      f"{v25['cost_sweep_posthoc']['share_of_noise_floor']:.0%}"),
            "cite": "v1.8~v2.5",
            "colour": AMBER,
        },
        {
            "n": "5",
            "title": "선언 민감도",
            "ask": "결론이 어느 파라미터에 달렸는가?",
            "how": "값을 고르지 말고 0~1로 쓸어\n전환점을 보고한다",
            "found": (f"표준치료 이득 — 문헌값의\n"
                      f"{v21['tipping_strength']:.1%}에서 부호 반전"),
            "cite": "v2.1",
            "colour": GREEN,
        },
    ]


def save(fig: plt.Figure, filename: str) -> None:
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(path, PUBLIC_DIR / filename)
    print(path)


def fig52_checklist() -> None:
    rows = build_rows()
    fig, ax = plt.subplots(figsize=(15.5, 8.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 97.5,
            "비교가 성립하는지 먼저 묻는다 — 질환·알고리즘과 무관하게 쓰는 다섯 진단",
            ha="center", fontsize=15.5, fontweight="bold", color=INK)
    ax.text(50, 93.2,
            "탐색 정책은 시뮬레이터의 목적함수를 직접 최적화한다. "
            "가정이 틀리면 잘 탐색할수록 더 틀린다.",
            ha="center", fontsize=11, color=GRAY, style="italic")

    columns = (4, 20, 45, 74)
    ax.text(columns[1], 88.4, "무엇을 묻나", fontsize=10.5, fontweight="bold", color=GRAY)
    ax.text(columns[2], 88.4, "어떻게 재나", fontsize=10.5, fontweight="bold", color=GRAY)
    ax.text(columns[3], 88.4, "우리의 답", fontsize=10.5, fontweight="bold", color=GRAY)
    ax.plot([3, 97], [86.8, 86.8], color=INK, linewidth=1.2)

    top, height, gap = 83.0, 13.4, 1.6
    for index, row in enumerate(rows):
        y = top - index * (height + gap)
        ax.add_patch(FancyBboxPatch(
            (3, y - height), 94, height,
            boxstyle="round,pad=0.4,rounding_size=1.2",
            facecolor="#ffffff", edgecolor=PALE, linewidth=1.1))
        ax.add_patch(FancyBboxPatch(
            (3, y - height), 1.5, height,
            boxstyle="square,pad=0", facecolor=row["colour"], edgecolor="none"))

        centre = y - height / 2
        ax.text(columns[0] + 3.2, centre, row["n"], fontsize=20, fontweight="bold",
                color=row["colour"], ha="center", va="center")
        ax.text(columns[1] - 7.5, centre + 3.4, row["title"], fontsize=12,
                fontweight="bold", color=INK, va="center")
        ax.text(columns[1], centre - 1.6, row["ask"], fontsize=10.5, color=INK,
                va="center")
        ax.text(columns[2], centre, row["how"], fontsize=10, color=GRAY,
                va="center", linespacing=1.45)
        ax.text(columns[3], centre, row["found"], fontsize=10.5, color=row["colour"],
                fontweight="bold", va="center", linespacing=1.45)
        ax.text(96.2, y - height + 1.6, row["cite"], fontsize=8.5, color=GRAY,
                ha="right", va="bottom", style="italic")

    ax.text(50, 6.2,
            "세 가지 작업 습관이 이 진단들을 쓸 수 있게 만든다 — "
            "값을 못 정하면 다이얼로 · 열기 전에 탐색 없이 지도부터 · 예측은 실행 전 코드에",
            ha="center", fontsize=10.5, color=INK, fontweight="bold")
    ax.text(50, 2.6,
            "수치는 각 리포트의 committed metrics.json에서 읽는다. "
            "MCTS-ONC, 경북대학교 의과대학 의과학 연구동아리, 2026.",
            ha="center", fontsize=9, color=GRAY)
    save(fig, "fig52_procedure_checklist.png")


if __name__ == "__main__":
    fig52_checklist()
