"""When do A+C PGCT s1 and IPPO s1 stop being identical? Compares logged metrics step by step
and lists the updates with a nonzero actual transfer allocation."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "runs/ac_20261023_185556"
runs = {m: ROOT / f"b6_corner_indiv_pen005_ac_{m}_s1" for m in ("ippo", "pgct")}


def load(run):
    out = {}
    for line in open(run / "metrics.jsonl", encoding="utf-8"):
        r = json.loads(line)
        out.setdefault(r["env_steps"], {}).update(r["metrics"])
    return out


ippo, pgct = load(runs["ippo"]), load(runs["pgct"])
transfers = []
for line in open(runs["pgct"] / "transfer.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if any(v > 0 for row in r["actual_allocation"] for v in row):
        pairs = [(i + 1, j + 1) for i, row in enumerate(r["actual_allocation"]) for j, v in enumerate(row) if v > 0]
        transfers.append((r["env_steps"], r["update"], pairs))
print("updates with actual transfer (env_steps, update, receiver->donor, 1-based):", transfers)

compare = ["train/episode_return", "update/value_loss", "update/policy_loss", "update/entropy", "eval/deliveries"]
first = {}
for step in sorted(set(ippo) & set(pgct)):
    for k in compare:
        a, b = ippo[step].get(k), pgct[step].get(k)
        if a is None or b is None or k in first:
            continue
        if a != b:
            first[k] = (step, a, b)
print("first differing logged value per metric (env_steps, ippo, pgct):")
for k in compare:
    print(" ", k, first.get(k))
common = sorted(set(ippo) & set(pgct))
same = sum(all(ippo[s].get(k) == pgct[s].get(k) for k in compare if k in ippo[s] and k in pgct[s]) for s in common
           if not transfers or s < transfers[0][0])
print(f"logged steps before first transfer: {sum(1 for s in common if not transfers or s < transfers[0][0])}, "
      f"of which all compared metrics bit-identical: {same}")
