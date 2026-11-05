"""Read-only: Oracle vs IPPO/PGCT/All s1 curves, peer-loss scale, actual AC v1 allocation."""
import json
from pathlib import Path
import numpy as np

RUN = Path(r"C:\RWARE\MARL_RWARE\third_party\robotic-warehouse\runs\ac_20261023_185556")
MODES = ["ippo", "oracle", "pgct", "all"]


def load(mode):
    rows = []
    with open(RUN / f"b6_corner_indiv_pen005_ac_{mode}_s1" / "metrics.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append((d["env_steps"], d["metrics"]))
    return rows


def evals(rows):
    out = []
    for step, m in rows:
        if "eval/deliveries" in m and "eval/agent_1_delivery_count" in m:
            out.append((step, m["eval/deliveries"], [m[f"eval/agent_{k}_delivery_count"] for k in range(1, 7)]))
    return out


data = {m: load(m) for m in MODES}
ev = {m: evals(data[m]) for m in MODES}

print("== eval team deliveries/episode (5-eval moving mean) at checkpoints")
marks = [0.5e6, 1e6, 1.5e6, 2e6, 2.5e6, 3e6, 3.5e6, 4e6, 4.5e6, 5e6]
for m in MODES:
    s = np.array([e[0] for e in ev[m]]); v = np.array([e[1] for e in ev[m]])
    line = []
    for t in marks:
        idx = np.where(s <= t)[0]
        if len(idx) == 0 or s[-1] < t - 30000:
            line.append("  -  "); continue
        k = idx[-1]
        line.append(f"{v[max(0, k-4):k+1].mean():5.1f}")
    print(f"{m:7s}", " ".join(line), f"(last step {s[-1]/1e6:.2f}M, n_eval {len(s)})")

print("\n== per-agent eval deliveries (5-eval mean) at 1M/2M/3M/5M")
for m in MODES:
    s = np.array([e[0] for e in ev[m]]); a = np.array([e[2] for e in ev[m]])
    for t in [1e6, 2e6, 3e6, 5e6]:
        idx = np.where(s <= t)[0]
        if len(idx) == 0 or s[-1] < t - 30000:
            continue
        k = idx[-1]
        print(f"{m:7s} {t/1e6:.0f}M", np.round(a[max(0, k-4):k+1].mean(0), 1), "sum", round(float(a[max(0, k-4):k+1].mean(0).sum()), 1))

print("\n== slope after the team first reaches a level (deliveries per episode gained per 1M steps)")
for level in [5, 10, 15]:
    out = []
    for m in MODES:
        s = np.array([e[0] for e in ev[m]]); v = np.array([e[1] for e in ev[m]])
        sm = np.convolve(v, np.ones(5) / 5, mode="same")
        hit = np.where(sm >= level)[0]
        if not len(hit):
            out.append(f"{m}: not reached"); continue
        t0 = s[hit[0]]
        nxt = np.where(sm >= level + 5)[0]
        dt = (s[nxt[0]] - t0) / 1e6 if len(nxt) else float("nan")
        out.append(f"{m}: reach {level} at {t0/1e6:.2f}M, +5 more in {dt:.2f}M")
    print(f"level {level}:", " | ".join(out))

print("\n== Oracle peer loss vs value loss (update metrics), by phase")
for m in ["oracle", "all", "pgct"]:
    rows = data[m]
    for lo, hi in [(0.1e6, 0.5e6), (0.5e6, 1e6), (1e6, 2e6), (2e6, 3e6), (3e6, 5e6)]:
        pl, pw, vl, arcs = [], [], [], []
        for step, mm in rows:
            if lo <= step < hi and "update/pgct_peer_loss" in mm:
                pl.append(mm["update/pgct_peer_loss"]); pw.append(mm["update/pgct_peer_weight"])
                vl.append(mm["update/value_loss"]); arcs.append(mm.get("update/actual_transfer_arcs", np.nan))
        if pl:
            pl, vl = np.array(pl), np.array(vl)
            print(f"{m:7s} {lo/1e6:.1f}-{hi/1e6:.1f}M  peer_loss {pl.mean():.4f}  x0.01 = {0.01*pl.mean():.5f}  "
                  f"value_loss {vl.mean():.4f} x0.5 = {0.5*vl.mean():.4f}  ratio {0.01*pl.mean()/(0.5*vl.mean()):.3f}  "
                  f"arcs/update {np.nanmean(arcs):.2f}  frac updates with transfer {np.mean(np.array(pw) > 0):.2f}")

print("\n== actual allocation from transfer.jsonl")
for m in MODES:
    same = cross = upd = withx = 0
    cand_cross = cand_same = 0
    for line in open(RUN / f"b6_corner_indiv_pen005_ac_{m}_s1" / "transfer.jsonl", encoding="utf-8"):
        r = json.loads(line); upd += 1
        same += r["same_team_arcs"]; cross += r["cross_team_arcs"]
        withx += (r["same_team_arcs"] + r["cross_team_arcs"]) > 0
        c = np.array(r["physical_candidates"], dtype=bool)
        idx = np.arange(6) % 2
        st = idx[:, None] == idx[None, :]
        cand_same += int((c & st).sum()); cand_cross += int((c & ~st).sum())
    print(f"{m:7s} updates {upd}  same arcs {same}  cross arcs {cross}  updates with any transfer {withx}  "
          f"physical cand same {cand_same} cross {cand_cross}")

print("\n== probe/cross_team_gate vs physical cross candidates (PGCT)")
cg, sg = [], []
for step, mm in data["pgct"]:
    if "probe/cross_team_gate" in mm:
        cg.append(mm["probe/cross_team_gate"]); sg.append(mm["probe/same_team_gate"])
print("probe/cross_team_gate mean", np.mean(cg), "same_team_gate mean", np.mean(sg), "n", len(cg))
