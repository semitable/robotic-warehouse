"""Offline check (T210): a disagreement-count test that does not split probes by action.

Idea: if shared and team-specific reward events fall in the SAME public action, the per-action maximum mixes them
again. A count of probes on which the two predictions clearly differ (|p - q| > delta) is not diluted by shared
events, because agreeing probes add nothing to the count. Here all probes are pooled (no action split, no mirrored
copies), on the saved predictors and probes of every checkpoint of the v2 discovery run (seed 1, 5M), mature pairs.
Reported per checkpoint: same-team maximum and different-team minimum of the count and of the share of probes.
Read-only, no training.
"""
from pathlib import Path
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import stratified_gate_check as SG  # noqa: E402

RUN = ROOT / "runs/acv2_s1_20261029_084340/b6_corner_indiv_pen005_acv2_pgct_s1"
DELTAS = (0.3, 0.5)


def counts(cfg, preds, i, j):
    b, a, f = (np.concatenate([preds[k][x] for k in (i, j)]) for x in ("b", "a", "f"))
    p, q = preds[i]["model"].predict(b, a, f), preds[j]["model"].predict(b, a, f)
    diff = np.abs(p - q)
    return {d: (int((diff > d).sum()), float((diff > d).mean())) for d in DELTAS}, len(p)


lines, rows = [], []
for ck in sorted(RUN.glob("checkpoint_*.pt")):
    steps, cfg, preds = SG.load(ck)
    same, cross = {d: [] for d in DELTAS}, {d: [] for d in DELTAS}
    for i in range(6):
        for j in range(i + 1, 6):
            if not (preds[i]["mature"] and preds[j]["mature"]):
                continue
            c, n = counts(cfg, preds, i, j)
            for d in DELTAS:
                (same if SG.TEAM[i] == SG.TEAM[j] else cross)[d].append(c[d])
    row = dict(step=steps / 1e6)
    parts = [f"{steps / 1e6:.2f}M"]
    for d in DELTAS:
        sc, cc = max(x[0] for x in same[d]), min(x[0] for x in cross[d])
        ss, cs = max(x[1] for x in same[d]), min(x[1] for x in cross[d])
        row[f"d{d}"] = dict(same_max_count=sc, cross_min_count=cc, same_max_share=ss, cross_min_share=cs)
        parts.append(f"delta {d}: count same<= {sc} / cross>= {cc}, share same<= {ss:.2f} / cross>= {cs:.2f}")
    rows.append(row)
    lines.append(" | ".join(parts))
    print(lines[-1])
(HERE / "disagreement_count_check_stdout.txt").write_text("\n".join(lines), encoding="utf-8")
(HERE / "disagreement_count_check.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
