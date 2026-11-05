"""IJCAI-27 roadmap (D130) as a Gantt chart: PNG and PDF (T215)."""
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
for name in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

EX, WR, SB = ("#E1F5EE", "#0F6E56", "#085041"), ("#EEEDFE", "#534AB7", "#3C3489"), ("#FAECE7", "#993C1D", "#712B13")
D = lambda m, d, y=2026: date(y, m, d)
TASKS = [  # label, start, end (inclusive), style, bar text
    ("하루짜리 run 분석", D(11, 5), D(11, 6), EX, "11/6 확인"),
    ("설계 확정·서버 준비", D(11, 5), D(11, 9), EX, "11/9까지"),
    ("서버 본 실험 5M", D(11, 12), D(11, 30), EX, "11/12 시작 · 4 mode × 다수 seed"),
    ("PC 강건성 run", D(11, 10), D(11, 23), EX, "인접 목적지·보상 잡음"),
    ("추가 seed(필요 시)", D(12, 15), D(12, 28), EX, "분산이 크면"),
    ("분석 도구·그림", D(11, 10), D(12, 7), WR, "학습 속도·정밀도/재현율·그림 3 교체"),
    ("방법·관련 연구 최종본", D(11, 17), D(11, 30), WR, "그림 1 원본 수정 포함"),
    ("결과 집필·초고 v1", D(12, 1), D(12, 14), WR, "12/14 초고 완성"),
    ("교수님 검토 반영", D(12, 15), D(12, 28), WR, "초고 v2"),
    ("제출 준비", D(12, 15), D(1, 4, 2027), SB, "체크리스트·익명화·Overleaf"),
    ("초록 제출", D(12, 29), D(1, 4, 2027), SB, "1/4 마감"),
    ("본문·보조 자료 제출", D(1, 5, 2027), D(1, 11, 2027), SB, "1/11 마감(AoE)"),
]
MILESTONES = [(D(11, 6), "11/6 설계 판단"), (D(11, 9), "11/9 규모 결정"), (D(11, 12), "11/12 본 실험 시작"),
              (D(12, 14), "12/14 초고 v1"), (D(1, 4, 2027), "1/4 초록"), (D(1, 11, 2027), "1/11 본문")]


def main():
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    n = len(TASKS)
    for k, (label, s, e, (fill, edge, text), note) in enumerate(TASKS):
        y = n - 1 - k
        width = (e - s).days + 1
        ax.barh(y, width, left=mdates.date2num(s), height=.62, color=fill, edgecolor=edge, linewidth=.8, zorder=3)
        x_text = mdates.date2num(s) + (width + .5 if width <= 3 else .4)  # short bars: label after the bar
        ax.text(x_text, y, note, va="center", ha="left", fontsize=8.5, color=text, zorder=4,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none") if width <= 3 else None)
    ax.set_yticks(range(n))
    ax.set_yticklabels([t[0] for t in TASKS][::-1], fontsize=9.5)
    start, end = D(11, 3), D(1, 12, 2027)
    ax.set_xlim(mdates.date2num(start), mdates.date2num(end))
    weeks = [D(11, 3) + timedelta(days=7 * w) for w in range(11)]
    ax.set_xticks([mdates.date2num(w) for w in weeks])
    ax.set_xticklabels([f"{w.month}/{w.day}" for w in weeks], fontsize=8.5)
    ax.grid(axis="x", color="#E1E0D9", lw=.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for k, (d, label) in enumerate(MILESTONES):
        x = mdates.date2num(d) + .5
        ax.axvline(x, color="#993C1D", lw=.9, ls="--", alpha=.6, zorder=1)
        ax.text(x, n - .3 + (k % 3) * .45, label, ha="right" if k == len(MILESTONES) - 1 else "center",
                va="bottom", fontsize=8, color="#712B13", zorder=4,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    ax.set_ylim(-.7, n + 1.2)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c[0], ec=c[1]) for c in (EX, WR, SB)]
    ax.legend(handles, ["실험", "분석·집필", "제출"], loc="lower left", ncol=3, frameon=False, fontsize=9,
              bbox_to_anchor=(0, -.16))
    ax.set_title("PGCT · IJCAI-27 제출 로드맵 (2026-11-05 기준, 초록 1/4 · 본문 1/11 AoE)", fontsize=11, loc="left",
                 pad=22)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"pgct_roadmap_ijcai27.{ext}", dpi=200)
    print("saved")


if __name__ == "__main__":
    main()
