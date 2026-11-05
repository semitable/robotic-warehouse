"""Offline gate v1 vs v2 evaluation on rollout streams (no policy training).

Each rollouts*.npz (collect.py) is streamed in collection order (checkpoint by checkpoint, episode
by episode) through the real gate classes: reward_probe_ac.ACController (v1, exact-match empirical
mean) and reward_probe_ac_v2.ACV2Controller (v2). v1's virtual (A) branches need the full
observation and are not replayed; they were never answered in the 2.87M A+C run either.
Every pair is in contact at every round, so this measures decisions given contact, not how often
neighbours meet. Team labels only score the decisions.
Usage: python output/ac_offline_20261026/gate_eval.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "latent_team"))
import reward_probe_ac as AC  # noqa: E402
import reward_probe_ac_v2 as V2  # noqa: E402

OUT = Path(__file__).resolve().parent
DATASETS = {
    "corner/indiv A+C IPPO s1, 0.26-5M": "rollouts.npz",
    "corner/indiv IPPO s1, 0.26-1M": "rollouts_corner_indiv_1M_s1.npz",
    "corner/indiv IPPO s2, 0.26-1M": "rollouts_corner_indiv_1M_s2.npz",
    "adjacent/indiv IPPO s1, 0.26-1M": "rollouts_adjacent_indiv_1M_s1.npz",
    "adjacent/indiv IPPO s2, 0.26-1M": "rollouts_adjacent_indiv_1M_s2.npz",
}
ROUND_STEPS = 1000  # environment steps between gate rounds in the replay
torch.set_num_threads(4)


def auroc(sim, same):
    a, b = sim[same], sim[~same]
    if not len(a) or not len(b):
        return None
    return float(np.mean([(x > y) + .5 * (x == y) for x in a for y in b]))


def status_v1(row):
    return {"accepted": "accept", "distance_signal_or_evidence": "reject"}.get(row["reason"], "abstain")


def status_v2(row, min_rounds):
    if row["accepted"]:
        return "accept"
    if row["reason"] == "distance_or_rounds" and row["rounds"] >= min_rounds:
        return "reject"
    return "abstain"


def summarise(rows, status, ema, rounds, teams, pairs):
    same = np.array([teams[i] == teams[j] for i, j in pairs])
    st = {(r["receiver"], r["donor"]): status(r) for r in rows}
    s = [st.get(p, "abstain") for p in pairs]
    out = dict(
        true_accept_same=sum(x == "accept" and y for x, y in zip(s, same)),
        false_accept_cross=sum(x == "accept" and not y for x, y in zip(s, same)),
        reject_same=sum(x == "reject" and y for x, y in zip(s, same)),
        reject_cross=sum(x == "reject" and not y for x, y in zip(s, same)),
        abstain_same=sum(x == "abstain" and y for x, y in zip(s, same)),
        abstain_cross=sum(x == "abstain" and not y for x, y in zip(s, same)))
    decided = len(pairs) - out["abstain_same"] - out["abstain_cross"]
    correct = out["true_accept_same"] + out["reject_cross"]
    out.update(decided=decided, decision_accuracy=round(correct / decided, 3) if decided else None)
    has = np.array([rounds[i, j] > 0 for i, j in pairs])
    sim = np.array([-ema[i, j] for i, j in pairs])
    out["auroc_ema_distance"] = auroc(sim[has], same[has]) if has.any() else None
    out["pairs_with_distance"] = int(has.sum())
    return out


def evaluate(path, v2config):
    D = np.load(OUT / path)
    B, A, ACT, R, AG, EP, TEAM = (D[k] for k in ("before", "after", "action", "reward", "agent", "episode", "team"))
    n = int(AG.max()) + 1
    teams = np.array([int(TEAM[AG == i][0]) for i in range(n)])
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    steps = len(R) // n
    v1 = AC.ACController(n, AC.ACConfig())
    v2 = V2.ACV2Controller(n, v2config)
    everyone = np.ones((n, n), bool)
    timeline, k = [], 0
    for t in range(steps):
        sl = slice(t * n, (t + 1) * n)
        assert (AG[sl] == np.arange(n)).all()
        for i in range(n):
            idx = t * n + i
            q = AC.Query(B[idx].astype(np.float32), int(ACT[idx]), A[idx].astype(np.float32), "experience")
            v1.predictors[i].learn(q, float(R[idx]), int(EP[idx]))
        v2.observe_views(B[sl], ACT[sl], A[sl], R[sl])
        if (t + 1) % ROUND_STEPS == 0 or t == steps - 1:
            k += 1
            a1, _ = v1(k * v1.config.interval, everyone, active=True)
            a2, _ = v2(k * v2.config.interval, everyone, active=True)
            same = teams[:, None] == teams[None, :]
            timeline.append(dict(round=k, env_steps_replayed=t + 1, ckpt=int(D["ckpt"][t * n]),
                                 v1_same_arcs=int(((a1 > 0) & same).sum()), v1_cross_arcs=int(((a1 > 0) & ~same).sum()),
                                 v2_same_arcs=int(((a2 > 0) & same).sum()), v2_cross_arcs=int(((a2 > 0) & ~same).sum()),
                                 v2_mature=int(sum(v2.last_log["ac_mature"]))))
    deliveries = [int(D["delivered"][AG == i].sum()) for i in range(n)]
    res = dict(file=path, agents=n, teams=teams.tolist(), deliveries_per_agent=deliveries, rounds=k,
               own_events_v2=[p.events_seen for p in v2.predictors],
               v1=summarise(v1.last_log.get("ac_pairs", []), status_v1, v1.ema, v1.rounds, teams, pairs),
               v2=summarise(v2.last_log.get("ac_pairs", []), lambda r: status_v2(r, v2.config.min_rounds),
                            v2.ema, v2.rounds, teams, pairs),
               v2_mature=v2.last_log["ac_mature"],
               v2_distance_matrix=np.where(v2.rounds + v2.rounds.T > 0, np.round(v2.ema + v2.ema.T, 3), np.nan).tolist(),
               first_round_v2_accept=next((r["round"] for r in timeline if r["v2_same_arcs"] + r["v2_cross_arcs"]), None),
               timeline=timeline)
    return res


def _np(o):
    return o.item() if hasattr(o, "item") else None


def main():
    results = {}
    for variant, cfg in (("v2", V2.ACV2Config()), ("v2_no_mirror", replace(V2.ACV2Config(), mirror=False))):
        for name, path in DATASETS.items():
            if not (OUT / path).exists():
                print("missing", path)
                continue
            t0 = time.time()
            r = evaluate(path, cfg)
            results.setdefault(name, {})[variant] = r
            print(f"[{variant}] {name}: rounds {r['rounds']} deliveries {r['deliveries_per_agent']} "
                  f"mature {r['v2_mature']}\n   v1 {json.dumps(r['v1'], default=_np)}\n   v2 {json.dumps(r['v2'], default=_np)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    (OUT / "gate_eval.json").write_text(json.dumps(results, indent=1, default=_np), encoding="utf-8")


if __name__ == "__main__":
    main()
