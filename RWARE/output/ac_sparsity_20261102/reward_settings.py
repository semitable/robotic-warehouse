"""Print reward-related env kwargs and A+C v2 event statistics of the v2 PGCT s1 run (read-only)."""
import json
from pathlib import Path

import numpy as np

root = Path(r"C:\RWARE\MARL_RWARE\third_party\robotic-warehouse\runs\acv2_s1_20261029_084340")
exp = json.loads((root / "experiment.json").read_text(encoding="utf-8"))


def find_env_kwargs(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "env_kwargs_json" and isinstance(value, str):
                return json.loads(value)
            if key == "env_kwargs" and isinstance(value, dict):
                return value
            found = find_env_kwargs(value)
            if found:
                return found
    if isinstance(obj, list):
        for value in obj:
            found = find_env_kwargs(value)
            if found:
                return found
    return None


kwargs = find_env_kwargs(exp)
if kwargs is None:
    print("no env kwargs; top keys:", list(exp)[:30])
else:
    for key in sorted(kwargs):
        if any(s in key for s in ("reward", "penalty", "progress", "max_steps", "n_agents", "team")):
            print(f"  {key} = {kwargs[key]}")

# Event statistics over training from metrics.jsonl
job = root / "b6_corner_indiv_pen005_acv2_pgct_s1"
rows = [json.loads(line) for line in (job / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
keys = ["update/ac_v2_own_events_mean", "update/ac_v2_mature_agents", "update/ac_v2_predictor_loss",
        "update/ac_v2_audit_distance_same", "update/ac_v2_audit_distance_cross",
        "update/ac_v2_audit_same_accepted", "update/ac_v2_audit_cross_accepted"]
step_key = next((k for k in ("global_step", "timesteps", "step", "env_steps") if k in rows[-1]), None)
print("step key:", step_key, "rows:", len(rows))
marks = [0.1e6, 0.25e6, 0.5e6, 1e6, 2e6, 3e6, 4e6, 5e6]
for m in marks:
    cand = [r for r in rows if step_key and r.get(step_key, 0) <= m and any(k in r for k in keys)]
    if not cand:
        continue
    r = cand[-1]
    print(f"{m/1e6:.2f}M", {k.split('/')[-1]: (round(r[k], 3) if isinstance(r.get(k), float) else r.get(k)) for k in keys})
