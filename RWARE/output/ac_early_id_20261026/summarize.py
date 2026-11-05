"""Tables from the *_curve.json files written by stream_eval.py."""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
NAMES = ["acs1", "corner1M_s1", "corner1M_s2", "adjacent1M_s1"]
MARKS = [102400, 204800, 307200, 409600, 512000, 614400, 716800, 819200, 921600, 1003520]


def fmt(v):
    return "—" if v is None else f"{v:.2f}"


def first(rows, cond):
    return next((r["env_steps"] for r in rows if cond(r)), None)


summary = {}
for name in NAMES:
    path = OUT / f"{name}_curve.json"
    if not path.exists():
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["curve"]
    evals = [r for r in rows if "v2_auroc" in r]
    print(f"== {name} (teams {data['teams']})")
    print("  steps    mature  gate s/c (cum)  v2 AUROC (pairs)  contrast-corr AUROC (pairs)  sign AUROC (pairs)")
    cum_s = cum_c = 0
    marks = list(MARKS)
    for r in rows:
        cum_s += r["gate_same_arcs"]; cum_c += r["gate_cross_arcs"]
        r["cum_same"], r["cum_cross"] = cum_s, cum_c
    for m in marks:
        e = min(evals, key=lambda r: abs(r["env_steps"] - m), default=None)
        if e is None or abs(e["env_steps"] - m) > 30000:
            continue
        print(f"  {e['env_steps']:>8} {sum(e['mature'])}/6    {e['cum_same']}/{e['cum_cross']:<10}"
              f"  {fmt(e['v2_auroc'])} ({e['v2_pairs']:>2})        {fmt(e['contrast_corr_auroc'])} ({e['contrast_pairs']:>2})"
              f"                 {fmt(e['contrast_sign_auroc'])} ({e['sign_pairs']:>2})")
    s = dict(
        first_mature_pair=first(rows, lambda r: sum(r["mature"]) >= 2),
        first_gate_same=first(rows, lambda r: r["gate_same_arcs"] > 0),
        first_gate_cross=first(rows, lambda r: r["gate_cross_arcs"] > 0),
        gate_same_arc_rounds=sum(r["gate_same_arcs"] > 0 for r in rows),
        gate_cross_arc_rounds=sum(r["gate_cross_arcs"] > 0 for r in rows),
        rounds=len(rows),
        first_contrast_auroc_ge_09=first(evals, lambda r: (r["contrast_corr_auroc"] or 0) >= .9),
        first_v2_auroc_ge_09=first(evals, lambda r: (r["v2_auroc"] or 0) >= .9),
        mean_contrast_auroc=float(np.mean([r["contrast_corr_auroc"] for r in evals if r["contrast_corr_auroc"] is not None])),
        mean_v2_auroc=float(np.mean([r["v2_auroc"] for r in evals if r["v2_auroc"] is not None]))
        if any(r["v2_auroc"] is not None for r in evals) else None,
        final_mature=evals[-1]["mature"] if evals else None,
        final_own_events=rows[-1]["own_events"])
    summary[name] = s
    print("  ", json.dumps(s))
(OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
