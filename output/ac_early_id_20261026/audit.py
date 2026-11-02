"""Reward-component identifiability audit (offline diagnostic, ladder step (i); not part of any gate).

For each reward component and each public feature set, the exact conditional mean
mu_i(g) = E[component_i | public feature group g] is estimated from agent i's own replayed experience.
This is the best any reward predictor restricted to those features could learn, so it separates
'not observable' from 'predictor too weak'. Team labels only score the result:
  team_gap   sum_g w_g |mu_team0(g) - mu_team1(g)| / sum_g w_g (|mu_team0| + |mu_team1|)
  split_rank rank of the true 3+3 team split among all 10 balanced agent splits (1 = most separated)
  agent_auroc  same- vs cross-team pairs ranked by sum_g w_g |mu_i(g) - mu_j(g)| (15 pairs)
Windows: experience of each checkpoint window and cumulative from the start.
Usage: python output/ac_early_id_20261026/audit.py acs1 corner1M_s1 ...
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
COMPONENTS = ["delivery", "return_", "pickup", "goal_progress", "rest", "reward"]
FEATURES = {"dir": ("carrying", "heading", "action"), "dir+landmark": ("carrying", "heading", "action", "landmark")}
MIN_TEAM, MIN_AGENT = 20, 10


def group_ids(D, mask, feats):
    keys = np.stack([D[f][mask].astype(np.int64) for f in feats], 1)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    return inv.ravel()


def means(values, groups, sel, n_groups):
    cnt = np.bincount(groups[sel], minlength=n_groups).astype(float)
    tot = np.bincount(groups[sel], weights=values[sel], minlength=n_groups)
    with np.errstate(invalid="ignore", divide="ignore"):
        return tot / cnt, cnt


def team_gap(values, groups, members_a, members_b, agent, n_groups):
    ma, ca = means(values, groups, np.isin(agent, members_a), n_groups)
    mb, cb = means(values, groups, np.isin(agent, members_b), n_groups)
    ok = (ca >= MIN_TEAM) & (cb >= MIN_TEAM)
    if not ok.any():
        return np.nan
    w = (ca + cb)[ok]
    mass = np.sum(w * (np.abs(ma[ok]) + np.abs(mb[ok])))
    return float(np.sum(w * np.abs(ma[ok] - mb[ok])) / mass) if mass > 0 else np.nan


def auroc(score, same):
    ok = np.isfinite(score)
    a, b = score[ok & same], score[ok & ~same]
    if not len(a) or not len(b):
        return None
    return float(np.mean([(x > y) + .5 * (x == y) for x in a for y in b]))


def audit(name):
    D = np.load(OUT / f"{name}_audit.npz")
    teams = D["teams"]
    n = len(teams)
    true_a = tuple(np.flatnonzero(teams == teams[0]))
    splits = [s for s in itertools.combinations(range(n), n // 2) if 0 in s]  # 10 distinct 3+3 splits
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    same = np.array([teams[i] == teams[j] for i, j in pairs])
    ends = D["ckpt_ends"].tolist()
    windows = [(f"ckpt{k}", D["ckpt"] == k) for k in range(len(ends))] + \
              [(f"<= {ends[k] / 1e6:.2f}M", D["ckpt"] <= k) for k in range(len(ends))]
    result = dict(teams=teams.tolist(), ckpt_ends=ends, windows={})
    for wname, wmask in windows:
        res = {"transitions_per_agent": int(wmask.sum() // n)}
        for fname, feats in FEATURES.items():
            groups = group_ids(D, wmask, feats)
            G = int(groups.max()) + 1
            agent = D["agent"][wmask]
            for comp in COMPONENTS:
                v = D[comp][wmask].astype(np.float64)
                gaps = {s: team_gap(v, groups, list(s), [a for a in range(n) if a not in s], agent, G) for s in splits}
                true = gaps[true_a]
                finite = [g for g in gaps.values() if np.isfinite(g)]
                rank = int(1 + sum(g > true for g in finite)) if np.isfinite(true) else None
                mu = []
                for i in range(n):
                    m, c = means(v, groups, agent == i, G)
                    mu.append((m, c))
                dist = []
                for i, j in pairs:
                    ok = (mu[i][1] >= MIN_AGENT) & (mu[j][1] >= MIN_AGENT)
                    w = (mu[i][1] + mu[j][1])[ok]
                    dist.append(np.sum(w * np.abs(mu[i][0][ok] - mu[j][0][ok])) / w.sum() if ok.any() else np.nan)
                dist = np.array(dist)
                events = int((v != 0).sum())
                res[f"{fname}|{comp}"] = dict(nonzero=events,
                                              team_gap=None if not np.isfinite(true) else round(true, 3),
                                              split_rank=rank, agent_auroc=auroc(-dist, same))
        result["windows"][wname] = res
    return result


def main():
    names = sys.argv[1:] or ["acs1", "corner1M_s1", "corner1M_s2", "adjacent1M_s1"]
    out = {}
    for name in names:
        if not (OUT / f"{name}_audit.npz").exists():
            print("missing", name)
            continue
        out[name] = audit(name)
        print(f"== {name}")
        for wname, res in out[name]["windows"].items():
            if not wname.startswith("<="):
                continue
            print(f"  {wname} ({res['transitions_per_agent']} steps/agent)")
            for key, r in res.items():
                if key == "transitions_per_agent":
                    continue
                print(f"    {key:28s} nonzero {r['nonzero']:>7} gap {r['team_gap']} rank {r['split_rank']} "
                      f"agentAUROC {r['agent_auroc']}")
    (OUT / "audit.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
