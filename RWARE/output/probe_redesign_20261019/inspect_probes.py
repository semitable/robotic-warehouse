"""Which v2 E1 gate probes separate teams consistently, and why?  Read-only.

Joins E32's Delta_m (output/probe_weight_20261019/results.json) with each probe's scenario
(position, heading, team frame) and with the actors' action distributions on that probe,
for the raw and the team-relative-ablated bank.

    python output/probe_redesign_20261019/inspect_probes.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "output/probe_connect_20261019"))
sys.path.insert(0, str(ROOT / "output/probe_weight_20261019"))
from examples.diagnose_pgct_rollout_compatibility import load_model  # noqa: E402
from analyze import ablate  # noqa: E402
from delta_m import E1_MODELS, E1_BANK, per_probe  # noqa: E402

OUT = Path(__file__).resolve().parent
DIRS = ["UP", "DOWN", "LEFT", "RIGHT"]
ACT = ["NOOP", "FWD", "LEFT", "RIGHT", "TOGGLE"]
GOALS = {0: (4, 10), 1: (5, 10)}  # E1: red goal (4,10), blue goal (5,10)

res = json.loads((ROOT / "output/probe_weight_20261019/results.json").read_text(encoding="utf-8"))
bank = torch.load(E1_BANK, map_location="cpu", weights_only=False)
sc = bank["gate_scenarios"]
rows = []
for fam, seq in (("E1 v2", bank["sequences"].float()[:, 0]), ("E1 v2 ablated", ablate(bank["sequences"].float())[:, 0])):
    D = np.array(list(res[fam]["deltas"].values()))  # [model, probe]
    pos_all = (D > 0).all(0)
    team_P = {}
    for name, ck in E1_MODELS.items():
        agents, _ = load_model(ck)
        _, P = per_probe(agents, seq)  # [agent, probe, action]
        team_P[name] = np.stack([P[[0, 2]].mean(0), P[[1, 3]].mean(0)])  # [team, probe, action]
    TP = np.mean(list(team_P.values()), 0)
    for m, s in enumerate(sc):
        x, y, d = s["x"], s["y"], s["direction"]
        rows.append(dict(bank=fam, probe=m, consistent=bool(pos_all[m]), n_pos=int((D[:, m] > 0).sum()),
                         delta_mean=float(D[:, m].mean()), x=x, y=y, heading=DIRS[d], frame_team=s["team"],
                         dx_to_red_goal=GOALS[0][0] - x, dx_to_blue_goal=GOALS[1][0] - x, dy_to_goal=10 - y,
                         red_policy={a: round(float(v), 2) for a, v in zip(ACT, TP[0, m])},
                         blue_policy={a: round(float(v), 2) for a, v in zip(ACT, TP[1, m])},
                         team_gap_action=ACT[int(np.argmax(np.abs(TP[0, m] - TP[1, m])))]))

(OUT / "probe_table.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
for fam in ("E1 v2", "E1 v2 ablated"):
    R = [r for r in rows if r["bank"] == fam]
    print(f"== {fam}: consistent probes")
    for r in sorted(R, key=lambda r: -r["delta_mean"])[:10]:
        print(f"  #{r['probe']:2d} n_pos={r['n_pos']} Δ={r['delta_mean']:+.3f} ({r['x']},{r['y']}) {r['heading']:5s} frameT{r['frame_team']} "
              f"red={r['red_policy']} blue={r['blue_policy']} gap@{r['team_gap_action']}")
    for key in ("heading", "frame_team"):
        vals = sorted({r[key] for r in R})
        print("  by", key, {v: round(float(np.mean([r["delta_mean"] for r in R if r[key] == v])), 3) for v in vals})
    for lo, hi in ((0, 3), (3, 6), (6, 11)):
        sel = [r for r in R if lo <= r["dy_to_goal"] < hi]
        if sel:
            print(f"  dy_to_goal {lo}-{hi}: n={len(sel)} Δ mean={np.mean([r['delta_mean'] for r in sel]):+.3f} "
                  f"consistent={sum(r['consistent'] for r in sel)}")
    for side, f in (("left of goals x<4", lambda r: r["x"] < 4), ("right x>5", lambda r: r["x"] > 5),
                    ("between 4-5", lambda r: 4 <= r["x"] <= 5)):
        sel = [r for r in R if f(r)]
        if sel:
            print(f"  {side}: n={len(sel)} Δ mean={np.mean([r['delta_mean'] for r in sel]):+.3f} "
                  f"consistent={sum(r['consistent'] for r in sel)}")
