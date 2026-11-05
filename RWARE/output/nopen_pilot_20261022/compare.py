"""nopen pilot (E45) vs the earlier team-reward pilot (E43), same seed, first 1M steps.  Read-only; no training.

Pre-registered metrics (T089):
  1. TOGGLE per episode per 0.1M window (does it survive past 0.2M?)
  2. requested-shelf pickups per episode
  3. first evaluation with deliveries >= 1, and onset (3 consecutive evaluations >= 1)
  4. number of delivering agents
  5. side effects: non-requested pickups (all pickups - requested), wrong returns

    python output/nopen_pilot_20261022/compare.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OLD = "runs/pilot_20261019_180044_{c}/{c}_ippo_s{s}"
NEW = "runs/pilot_20261021_133652_{c}_nopen/{c}_nopen_ippo_s{s}"
RUNS = {}
for c, short, n in [("a_e1f4_team", "A", 4), ("b_e1f6_comm3", "B", 6)]:
    for s in (1, 2):
        RUNS[f"{short} s{s} old"] = (ROOT / OLD.format(c=c, s=s), n)
        RUNS[f"{short} s{s} nopen"] = (ROOT / NEW.format(c=c, s=s), n)
LIMIT = 1.0005e6
WINDOWS = np.arange(0, 1.0e6, 1e5)


def load(d):
    return [json.loads(l) for l in open(d / "metrics.jsonl", encoding="utf-8")]


def window_mean(rows, fn):
    out = []
    for lo in WINDOWS:
        v = [fn(r["metrics"]) for r in rows if lo <= r["env_steps"] < lo + 1e5]
        v = [x for x in v if x is not None and np.isfinite(x)]
        out.append(round(float(np.mean(v)), 3) if v else None)
    return out


def analyse(d, n):
    rows = [r for r in load(d) if r["env_steps"] <= LIMIT]
    train = [r for r in rows if "train/deliveries" in r["metrics"]]
    ev = [r for r in rows if "eval/deliveries" in r["metrics"]]
    pick = lambda m: sum(m.get(f"train/agent_{k}_pickup_count", 0.0) for k in range(1, n + 1))
    res = {
        "toggles/ep": window_mean(train, lambda m: m.get("train/toggle_count")),
        "P(toggle|pickup valid)": window_mean(train, lambda m: m.get("train/toggle_policy_probability_when_pickup_valid")),
        "req pickups/ep": window_mean(train, lambda m: m.get("train/requested_shelf_pickups")),
        "all pickups/ep": window_mean(train, pick),
        "non-req pickups/ep": window_mean(train, lambda m: pick(m) - m.get("train/requested_shelf_pickups", 0.0)),
        "wrong returns/ep": window_mean(train, lambda m: m.get("train/wrong_returns")),
        "train deliveries/ep": window_mean(train, lambda m: m.get("train/deliveries")),
        "shaping/ep": window_mean(train, lambda m: m.get("train/episode_shaping_return")),
    }
    steps = [r["env_steps"] for r in ev]
    dl = [r["metrics"]["eval/deliveries"] for r in ev]
    first = next((s for s, x in zip(steps, dl) if x >= 1), None)
    onset = next((steps[i] for i in range(len(dl) - 2) if min(dl[i:i + 3]) >= 1), None)
    # delivering agents: train deliveries per agent per episode over the last 0.1M, and eval over the last 5 evals
    last_train = [r["metrics"] for r in train if r["env_steps"] >= 0.9e6]
    per_agent_train = [round(float(np.mean([m.get(f"train/agent_{k}_delivery_count", 0.0) for m in last_train])), 3)
                       for k in range(1, n + 1)]
    # eval/agent_k_delivery_count is summed over the 20 evaluation episodes; divide to get per episode
    last_ev = [r["metrics"] for r in ev[-5:]]
    per_agent_eval = [round(float(np.mean([m.get(f"eval/agent_{k}_delivery_count", 0.0) for m in last_ev])) / 20, 3)
                      for k in range(1, n + 1)]
    total_train, total_eval = sum(per_agent_train), sum(per_agent_eval)
    res.update({
        "first eval >=1": first, "onset (3 evals >=1)": onset,
        "eval deliveries last5": round(float(np.mean(dl[-5:])), 2), "eval deliveries max": max(dl),
        "per-agent train deliveries/ep (0.9-1M)": per_agent_train,
        "agents >=0.1 train deliveries/ep (0.9-1M)": sum(x >= 0.1 for x in per_agent_train),
        "top-agent share of train deliveries (0.9-1M)": round(max(per_agent_train) / total_train, 2) if total_train else None,
        "per-agent eval deliveries/ep (last 5)": per_agent_eval,
        "top-agent share of eval deliveries (last 5)": round(max(per_agent_eval) / total_eval, 2) if total_eval else None,
        "last step": rows[-1]["env_steps"],
    })
    return res


def main():
    out = {name: analyse(d, n) for name, (d, n) in RUNS.items()}
    hashes = {name: (d / "initial_weights_sha256.txt").read_text().split()[0] for name, (d, _) in RUNS.items()}
    out["_init_hash_match"] = {k.replace(" old", ""): hashes[k] == hashes[k.replace("old", "nopen")]
                               for k in hashes if k.endswith("old")}
    (OUT / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["_init_hash_match"]))
    for name, r in out.items():
        if name.startswith("_"):
            continue
        print(f"\n== {name} (last step {r['last step']})")
        for k, v in r.items():
            if k != "last step":
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
