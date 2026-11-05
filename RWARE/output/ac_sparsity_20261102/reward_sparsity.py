"""Reward density and A+C v2 event/maturity timeline of the v2 PGCT s1 run (read-only)."""
import json
from pathlib import Path

import numpy as np

job = Path(r"C:\RWARE\MARL_RWARE\third_party\robotic-warehouse\runs\acv2_s1_20261029_084340"
           r"\b6_corner_indiv_pen005_acv2_pgct_s1")
rows = []
for line in (job / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        r = json.loads(line)
        rows.append((r["env_steps"], r.get("metrics", {})))

keys = ["update/reward_density", "update/positive_reward_density", "update/negative_reward_density",
        "update/ac_v2_own_events_mean", "update/ac_v2_mature_agents", "update/ac_v2_predictor_loss",
        "update/ac_v2_audit_distance_same", "update/ac_v2_audit_distance_cross",
        "update/ac_v2_pairs_compared", "update/ac_v2_audit_same_accepted", "update/ac_v2_audit_cross_accepted"]
edges = [0, .1e6, .25e6, .5e6, 1e6, 2e6, 3e6, 4e6, 5.01e6]
for lo, hi in zip(edges[:-1], edges[1:]):
    window = [m for s, m in rows if lo < s <= hi]
    out = {}
    for k in keys:
        vals = [m[k] for m in window if k in m]
        if vals:
            out[k.split("/")[-1]] = round(float(np.mean(vals)), 4)
    last = [m for m in window if "update/ac_v2_own_events_mean" in m]
    if last:
        out["own_events_mean_end"] = round(last[-1]["update/ac_v2_own_events_mean"], 1)
        out["mature_end"] = last[-1].get("update/ac_v2_mature_agents")
    deliveries = [[m.get(f"train/agent_{a}_delivery_count") for a in range(1, 7)] for m in window
                  if "train/agent_1_delivery_count" in m]
    if deliveries:
        out["train_deliveries_per_agent"] = np.round(np.nanmean(np.array(deliveries, dtype=float), 0), 2).tolist()
    print(f"({lo/1e6:.2f}, {hi/1e6:.2f}]M", json.dumps(out, ensure_ascii=False))

# Share of transitions that are v2 events (reward >= .5) per agent: approximate from positive density
# and delivery counts. Exact event share: deliveries+returns per agent / steps per agent.
