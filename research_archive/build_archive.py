"""Merge the project's scattered Markdown records into one archive file.

Sources are read only; the archive is regenerated from scratch each run:
    python research_archive/build_archive.py
Output: research_archive/RESEARCH_ARCHIVE.md
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "RESEARCH_ARCHIVE.md"

# (part title, [source paths relative to ROOT]) in reading order.
PARTS = [
    ("인수인계 요약", ["HANDOFF.md"]),
    ("공용 지침", ["AGENTS.md", "CLAUDE.md"]),
    ("현재 연구 상태", ["RESEARCH_STATE.md"]),
    ("질문·결론·설계 판단", ["DECISIONS.md"]),
    ("실험·평가 기록", ["EXPERIMENTS.md"]),
    ("분석 보고서 (RWARE/output/, 작성일 순)", [
        "RWARE/output/gating_diagnosis/README.md",
        "RWARE/output/pgct_spatial_analysis/REPORT.md",
        "RWARE/output/learning_review_20261013/REPORT.md",
        "RWARE/output/latent_team/exp_a_20261013/REPORT.md",
        "RWARE/output/scratch_review_20261015/REPORT.md",
        "RWARE/output/ijcai_review_20261015/REPORT.md",
        "RWARE/output/env_view_20261015/probes/README.md",
    ]),
    ("실행기·구현 문서", [
        "RWARE/latent_team/README.md",
        "RWARE/latent_team/BASELINES.md",
        "RWARE/examples/README.md",
        "RWARE/examples/EX5_PGCT.md",
        "RWARE/examples/PGCT_PHYSICAL_MIXING_CURRICULUM.md",
        "RWARE/examples/PGCT_SPATIAL_EXPERIMENTS.md",
        "RWARE/examples/PGCT_TRANSFER_EXPERIMENTS.md",
        "RWARE/examples/HARD_RWARE_EXPERIMENTS.md",
    ]),
    ("대화 기록 원문 (Codex + Claude Code)", ["CONVERSATION_LOG.md"]),
]

FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^(#{1,6})(\s.*)$")
# Timeline entries: decision, experiment and conversation-turn headings.
ENTRY = re.compile(r"^#{2}\s+((?:D|E|L)\d{2}[^\n]*|원문 기록 T\d{3}[^\n]*)$")
DATE = re.compile(r"(20\d\d-\d\d-\d\d)")
TOOL = re.compile(r"(Claude Code|Codex)")


def demote(text: str, levels: int) -> str:
    """Push headings down so each source nests under its part; code fences untouched."""
    out, in_fence = [], False
    for line in text.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
        m = None if in_fence else HEADING.match(line)
        if m:
            depth = min(len(m.group(1)) + levels, 6)
            line = "#" * depth + m.group(2)
        out.append(line)
    return "\n".join(out)


def timeline(sources: dict[str, str]) -> list[str]:
    rows = []
    for rel in ("DECISIONS.md", "EXPERIMENTS.md", "CONVERSATION_LOG.md"):
        in_fence = False
        for line in sources.get(rel, "").splitlines():
            if FENCE.match(line):
                in_fence = not in_fence
            m = None if in_fence else ENTRY.match(line)
            if not m:
                continue
            title = m.group(1).replace("|", "/")
            date = DATE.search(title)
            tool = TOOL.search(title)
            rows.append((date.group(1) if date else "날짜 미상", tool.group(1) if tool else "-", rel, title))
    # Stable sort keeps each file's own order within a date; undated rows first.
    rows.sort(key=lambda r: (r[0] != "날짜 미상", r[0]))
    lines = ["| 날짜 | 작성 | 출처 | 항목 |", "|---|---|---|---|"]
    lines += [f"| {d} | {t} | {s} | {title[:110]} |" for d, t, s, title in rows]
    return lines


def main() -> None:
    sources: dict[str, str] = {}
    missing = []
    for _, paths in PARTS:
        for rel in paths:
            p = ROOT / rel
            if p.exists():
                sources[rel] = p.read_text(encoding="utf-8-sig")
            else:
                missing.append(rel)

    now = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    doc = [
        "# PGCT 연구 통합 기록",
        "",
        f"자동 생성: {now} — `python research_archive/build_archive.py`. "
        "원본 파일은 수정하지 않았고, 이 파일은 원본을 순서대로 합친 사본이다. "
        "원본이 갱신되면 이 파일은 낡으므로 다시 생성한다. 편집은 원본에 한다.",
        "",
        "새 세션은 **Part 1(인수인계 요약)만 먼저 읽고**, 필요할 때 아래 목차에서 해당 부분을 찾는다.",
        "",
        "## 목차",
        "",
    ]
    for i, (title, paths) in enumerate(PARTS, 1):
        doc.append(f"- Part {i}. {title}")
        for rel in paths:
            if rel in sources:
                kb = len(sources[rel].encode("utf-8")) / 1024
                doc.append(f"  - `{rel}` ({kb:.0f} KB)")
    doc.append("- 부록. 통합 타임라인 (D=결정, E=실험, L/T=대화)")
    if missing:
        doc += ["", "누락된 원본: " + ", ".join(f"`{m}`" for m in missing)]

    for i, (title, paths) in enumerate(PARTS, 1):
        doc += ["", "---", "", f"# Part {i}. {title}"]
        for rel in paths:
            if rel not in sources:
                continue
            p = ROOT / rel
            mtime = dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            doc += ["", f"## 출처: `{rel}` (수정 {mtime})", "", demote(sources[rel], 2)]

    doc += ["", "---", "", "# 부록. 통합 타임라인", "",
            "DECISIONS·EXPERIMENTS·CONVERSATION_LOG의 항목 제목을 날짜순으로 모은 색인이다. "
            "같은 날짜 안에서는 파일별 원래 순서를 따른다.", ""]
    doc += timeline(sources)
    OUT.write_text("\n".join(doc) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB, {len(sources)} sources, missing {missing})")


if __name__ == "__main__":
    main()
