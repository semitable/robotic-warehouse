"""Offline check (T211): what if the reward rules of two teams differ only in rare situations?

Emulation on real data: at a late checkpoint of the v2 discovery run (seed 1, 4.35M), each agent's probe set is
rebuilt with exactly k team-specific events (its most recent own deliveries, FORWARD) and 32 - k shared events (its
most recent own shelf returns, TOGGLE), taken from the agent's saved 512-event buffer. Small k = the situation in
which the teams differ is rare in experience. For k = 0..32 we report, over pairs of mature agents that have enough
events of both kinds, the largest same-team and smallest different-team value of
  pooled distance, per-action maximum with n_min = 8 and n_min = 2, and the disagreement count (|p - q| > .5).
No mirrored probes. Read-only, no training.
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
from latent_team import reward_probe_ac_v2 as V2  # noqa: E402
import torch  # noqa: E402

CKPT = ROOT / "runs/acv2_s1_20261029_084340/b6_corner_indiv_pen005_acv2_pgct_s1/checkpoint_04250.pt"
FORWARD, TOGGLE = 1, 4
KS = (0, 1, 2, 4, 6, 8, 12, 16, 32)


def full_events(path):
    s = torch.load(path, map_location="cpu", weights_only=False)["reward_probe_ac_v2"]
    out = []
    for p in s["predictors"]:
        b, a, f = p["event_before"], p["event_action"], p["event_after"]
        carry_drop = (b[:, 0] > .5) & (f[:, 0] < .5)
        out.append(dict(deliv=np.flatnonzero(a == FORWARD), ret=np.flatnonzero((a == TOGGLE) & carry_drop), b=b, a=a, f=f))
    return out


def probes(ev, k):
    idx = np.r_[ev["deliv"][:k], ev["ret"][:32 - k]]  # saved buffers are newest first
    return ev["b"][idx], ev["a"][idx], ev["f"][idx]


def main():
    steps, cfg, preds = SG.load(CKPT)
    events = full_events(CKPT)
    usable = [k for k in range(6) if preds[k]["mature"] and len(events[k]["deliv"]) >= 32 and len(events[k]["ret"]) >= 32]
    lines = [f"{steps / 1e6:.2f}M, agents with >=32 deliveries and >=32 returns in their buffer: {usable}"]
    rows = []
    for k in KS:
        res = {"same": {}, "cross": {}}
        for x, i in enumerate(usable):
            for j in usable[x + 1:]:
                parts = [probes(events[i], k), probes(events[j], k)]
                b, a, f = (np.concatenate(z) for z in zip(*parts))
                p, q = preds[i]["model"].predict(b, a, f), preds[j]["model"].predict(b, a, f)
                vals = dict(pooled=V2.response_distance(p, q),
                            max8=V2.action_max_distance(p, q, a, parts[0][1], parts[1][1], 8)[0],
                            max2=V2.action_max_distance(p, q, a, parts[0][1], parts[1][1], 2)[0],
                            count=int((np.abs(p - q) > .5).sum()))
                rel = "same" if SG.TEAM[i] == SG.TEAM[j] else "cross"
                for key, v in vals.items():
                    res[rel].setdefault(key, []).append(v)
        row = dict(k=k)
        text = [f"k={k:2d}"]
        for key in ("pooled", "max8", "max2", "count"):
            s = [v for v in res["same"].get(key, []) if v is not None]
            c = [v for v in res["cross"].get(key, []) if v is not None]
            smax = max(s) if s else None
            cmin = min(c) if c else None
            row[key] = dict(same_max=smax, cross_min=cmin, n_same=len(s), n_cross=len(c))
            fmt = (lambda v: "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v)))
            text.append(f"{key}: same<= {fmt(smax)} / cross>= {fmt(cmin)}")
        rows.append(row)
        lines.append(" | ".join(text))
    print("\n".join(lines))
    (HERE / "rare_difference_check_stdout.txt").write_text("\n".join(lines), encoding="utf-8")
    (HERE / "rare_difference_check.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
