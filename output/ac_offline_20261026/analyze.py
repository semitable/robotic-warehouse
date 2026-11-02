"""A+C gate diagnosis on offline rollouts (collect.py). No training of the policy.

1. Observability (condition 1): are deliveries at the two goals distinguishable in the public view?
2. Experience coverage (condition 2): does each agent ever carry a shelf onto the OTHER team's goal?
3. Exact-match predictor (the running A+C rule) and coarser keys: support and identification.
4. Generalising predictor: a small MLP r_hat_i(before, action, after) fitted on agent i's own
   transitions, queried on shared probes (C: pooled experience, A: left-right mirrored events).
Team labels are used only to score pairs (AUROC: same-team pairs vs cross-team pairs).
"""
from __future__ import annotations

import json
from collections import defaultdict, OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

OUT = Path(__file__).resolve().parent
D = np.load(OUT / "rollouts.npz")
B, A2, ACT, R = D["before"], D["after"], D["action"], D["reward"].astype(np.float64)
AG, TEAM, EP, CK = D["agent"], D["team"], D["episode"], D["ckpt"]
DELIV, RET, GOAL_AFTER = D["delivered"], D["returned"], D["goal_after"]
N_AG = int(AG.max()) + 1
TEAMS = np.array([int(TEAM[AG == i][0]) for i in range(N_AG)])
PAIRS = [(i, j) for i in range(N_AG) for j in range(i + 1, N_AG)]
SAME = np.array([TEAMS[i] == TEAMS[j] for i, j in PAIRS])
CFG = dict(min_samples=8, min_episodes=3, max_std=.10, max_stderr=.04, min_queries=4, min_signal=.20, threshold=.12)
torch.manual_seed(0)
torch.set_num_threads(4)
report = {}

# public view layout: 6 ego (carrying, up, down, left, right, highway) + 11 channels x 5 x 5
# channels: 0 oob, 1 highway, 2 storage, 3 shelf, 4 agent, 5-8 agent dir (u, d, l, r), 9 loaded, 10 goal
EGO, S = 6, 5


def ch(c):
    return slice(EGO + c * S * S, EGO + (c + 1) * S * S)


def auroc(score_same_high, mask=None):
    s = np.asarray(score_same_high, dtype=float)
    same, cross = SAME, ~SAME
    if mask is not None:
        same, cross = same & mask, cross & mask
    a, b = s[same], s[cross]
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return None
    return float(np.mean([(x > y) + .5 * (x == y) for x in a for y in b]))


NO_AGENT0 = np.array([0 not in p for p in PAIRS])

# ---- 1. observability of the two goals in the public view --------------------------------------------
land = np.concatenate([B[:, :EGO], B[:, ch(0)], B[:, ch(10)]], 1)  # self state + walls + goal cells
dmask = DELIV == 1
keys_by_goal = defaultdict(set)
for row, g in zip(land[dmask], GOAL_AFTER[dmask]):
    keys_by_goal[int(g)].add(row.tobytes())
overlap = len(keys_by_goal[0] & keys_by_goal[1])
report["1_observability"] = dict(
    deliveries_by_goal_team={int(g): int(((GOAL_AFTER == g) & dmask).sum()) for g in (0, 1)},
    distinct_landmark_views={int(g): len(v) for g, v in keys_by_goal.items()},
    views_shared_by_both_goals=overlap)

# ---- 2. coverage: carrying onto own vs other team's goal ------------------------------------------------
carrying = B[:, 0] == 1
cov = {}
for i in range(N_AG):
    m = (AG == i) & carrying
    own = m & (GOAL_AFTER == TEAMS[i])
    other = m & (GOAL_AFTER == 1 - TEAMS[i])
    cov[i] = dict(team=int(TEAMS[i]), transitions=int((AG == i).sum()), deliveries=int(((AG == i) & dmask).sum()),
                  carry_onto_own_goal=int(own.sum()), carry_onto_other_goal=int(other.sum()),
                  mean_reward_onto_other_goal=float(R[other].mean()) if other.any() else None)
report["2_coverage"] = cov


# ---- 3. exact / coarse keys ----------------------------------------------------------------------------
def drop_agents(v):
    v = v.copy()
    for c in range(4, 10):
        sl = v[:, ch(c)].reshape(-1, S, S)
        centre = sl[:, 2, 2].copy()
        sl[:] = 0
        sl[:, 2, 2] = centre
        v[:, ch(c)] = sl.reshape(len(v), -1)
    return v


def crop(v, r):
    parts = [v[:, :EGO]]
    lo, hi = 2 - r, 2 + r + 1
    for c in range(11):
        parts.append(v[:, ch(c)].reshape(-1, S, S)[:, lo:hi, lo:hi].reshape(len(v), -1))
    return np.concatenate(parts, 1)


def landmark(v):
    return np.concatenate([v[:, :EGO], v[:, ch(0)], v[:, ch(3)], v[:, ch(10)]], 1)


VARIANTS = {
    "exact (running A+C)": (B, A2),
    "no other agents": (drop_agents(B), drop_agents(A2)),
    "no agents, 3x3": (crop(drop_agents(B), 1), crop(drop_agents(A2), 1)),
    "landmarks only (wall/shelf/goal)": (landmark(B), landmark(A2)),
}


def key_array(before, after):
    act = ACT.astype(np.uint8)[:, None]
    full = np.ascontiguousarray(np.concatenate([before.astype(np.uint8), act, after.astype(np.uint8)], 1))
    return full.view(np.dtype((np.void, full.shape[1]))).ravel()


def stats_per_agent(keys):
    out = []
    for i in range(N_AG):
        m = AG == i
        k, r, e = keys[m], R[m], EP[m]
        uniq, inv = np.unique(k, return_inverse=True)
        cnt = np.bincount(inv)
        mean = np.bincount(inv, weights=r) / cnt
        var = np.maximum(np.bincount(inv, weights=r * r) / cnt - mean ** 2, 0) * cnt / np.maximum(cnt - 1, 1)
        neps = np.array([len(s) for s in _episode_sets(inv, e, len(uniq))])
        std = np.sqrt(var)
        loose = (cnt >= CFG["min_samples"]) & (neps >= CFG["min_episodes"])
        strict = loose & (std <= CFG["max_std"]) & (std / np.sqrt(cnt) <= CFG["max_stderr"])
        out.append(dict(uniq=uniq, inv=inv, cnt=cnt, mean=mean, std=std, loose=loose, strict=strict))
    return out


def _episode_sets(inv, e, n):
    sets = [set() for _ in range(n)]
    for a, b in zip(inv, e):
        if len(sets[a]) < 3:
            sets[a].add(int(b))
    return sets


def pair_identification(st, rule):
    rows = []
    for i, j in PAIRS:
        a, b = st[i], st[j]
        ka = {k.tobytes(): x for k, x, s in zip(a["uniq"], a["mean"], a[rule]) if s}
        kb = {k.tobytes(): x for k, x, s in zip(b["uniq"], b["mean"], b[rule]) if s}
        common = sorted(set(ka) & set(kb))
        if len(common) < CFG["min_queries"]:
            rows.append(dict(n=len(common), decision=None, d_mean=np.nan, d_max=np.nan, signal=0., n_big=0))
            continue
        x = np.array([ka[k] for k in common]); y = np.array([kb[k] for k in common])
        d = np.abs(x - y)
        signal = min(np.ptp(x), np.ptp(y))
        big = (np.abs(x) >= .2) | (np.abs(y) >= .2)
        accept = signal >= CFG["min_signal"] and d.mean() < CFG["threshold"]
        rows.append(dict(n=len(common), decision=bool(accept), d_mean=float(d.mean()), d_max=float(d.max()),
                         signal=float(signal), n_big=int(big.sum()),
                         d_big=float(d[big].mean()) if big.any() else np.nan))
    return rows


def lru_supported(keys, cap=4096):
    """Strict-rule supported keys at the end when memory is an LRU of `cap` entries, like the running gate."""
    res = []
    for i in range(N_AG):
        m = np.flatnonzero(AG == i)
        mem = OrderedDict()
        for idx in m:
            k = keys[idx].tobytes()
            s = mem.get(k)
            if s is None:
                s = [0, 0., 0., set()]
                mem[k] = s
            s[0] += 1; s[1] += R[idx]; s[2] += R[idx] ** 2
            if len(s[3]) < 3:
                s[3].add(int(EP[idx]))
            mem.move_to_end(k)
            if len(mem) > cap:
                mem.popitem(last=False)
        n_ok = n_big = 0
        for c, sm, sq, eps in mem.values():
            if c >= 8 and len(eps) >= 3:
                mu = sm / c
                sd = np.sqrt(max(sq / c - mu * mu, 0) * c / max(c - 1, 1))
                if sd <= .10 and sd / np.sqrt(c) <= .04:
                    n_ok += 1
                    n_big += abs(mu) >= .2
        res.append(dict(supported=n_ok, supported_reward_bearing=int(n_big)))
    return res


rep3 = {}
for name, (bv, av) in VARIANTS.items():
    keys = key_array(bv, av)
    st = stats_per_agent(keys)
    per = []
    for i in range(N_AG):
        s = st[i]
        m = AG == i
        sup_t = s["strict"][s["inv"]]
        big = np.abs(s["mean"]) >= .2
        dl = DELIV[m] == 1
        per.append(dict(agent=i, transitions=int(m.sum()), unique_keys=int(len(s["uniq"])),
                        frac_transitions_supported_strict=round(float(sup_t.mean()), 3),
                        supported_keys_strict=int(s["strict"].sum()),
                        supported_reward_bearing_strict=int((s["strict"] & big).sum()),
                        supported_reward_bearing_loose=int((s["loose"] & big).sum()),
                        deliveries=int(dl.sum()),
                        deliveries_with_strict_support=int(sup_t[dl].sum()),
                        deliveries_with_loose_support=int(s["loose"][s["inv"]][dl].sum())))
    out = dict(per_agent=per)
    for rule in ("strict", "loose"):
        rows = pair_identification(st, rule)
        decided = [r for r in rows if r["decision"] is not None]
        out[f"pairs_{rule}"] = dict(
            mutual_queries=[r["n"] for r in rows],
            mutual_reward_bearing=[r.get("n_big", 0) for r in rows],
            decided=len(decided),
            accepted_same=sum(1 for r, s in zip(rows, SAME) if r["decision"] and s),
            accepted_cross=sum(1 for r, s in zip(rows, SAME) if r["decision"] and not s),
            auroc_mean_distance=auroc([-r["d_mean"] for r in rows]),
            auroc_reward_bearing_distance=auroc([-r.get("d_big", np.nan) for r in rows]),
            auroc_mean_distance_without_agent0=auroc([-r["d_mean"] for r in rows], NO_AGENT0))
    if name == "exact (running A+C)":
        out["lru4096_end_of_rollouts"] = lru_supported(keys)
    rep3[name] = out
report["3_keys"] = rep3


# ---- 4. generalising predictor ---------------------------------------------------------------------
def features(before, action, after):
    oh = np.eye(5, dtype=np.float32)[action]
    return np.concatenate([before.astype(np.float32), oh, after.astype(np.float32)], 1)


def mirror(v):
    v = v.copy()
    v[:, [3, 4]] = v[:, [4, 3]]  # ego dir left <-> right
    for c in range(11):
        v[:, ch(c)] = v[:, ch(c)].reshape(-1, S, S)[:, :, ::-1].reshape(len(v), -1)
    a, b = v[:, ch(7)].copy(), v[:, ch(8)].copy()  # other agents' dir left <-> right
    v[:, ch(7)], v[:, ch(8)] = b, a
    return v


def mirror_action(a):
    return np.where(a == 2, 3, np.where(a == 3, 2, a))


X = features(B, ACT, A2)
train_mask = (EP % 5) != 0


def fit(i, ckpt_max=None):
    m = (AG == i) & train_mask
    if ckpt_max is not None:
        m &= CK <= ckpt_max
    x = torch.as_tensor(X[m]); y = torch.as_tensor(R[m], dtype=torch.float32)
    net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    n = len(x)
    for epoch in range(20):
        perm = torch.randperm(n)
        for k in range(0, n, 512):
            idx = perm[k:k + 512]
            loss = ((net(x[idx]).squeeze(-1) - y[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


@torch.inference_mode()
def predict(net, x):
    return net(torch.as_tensor(x)).squeeze(-1).numpy().astype(np.float64)


def evaluate(nets, tag):
    held = ~train_mask
    acc = []
    for i, net in enumerate(nets):
        m = (AG == i) & held
        p = predict(net, X[m])
        dl = DELIV[m] == 1
        acc.append(dict(agent=i, heldout_mse=float(np.mean((p - R[m]) ** 2)),
                        heldout_own_delivery_pred=float(p[dl].mean()) if dl.any() else None,
                        heldout_nonevent_pred=float(p[~dl & (np.abs(R[m]) < .2)].mean())))
    rng = np.random.default_rng(0)
    ev = np.flatnonzero((DELIV == 1) | (RET == 1))
    rnd = rng.choice(len(R), size=3000, replace=False)
    probes = {
        "C random experience (3000)": X[rnd],
        "C delivery/return events": X[ev],
        "A mirrored events": features(mirror(B[ev]), mirror_action(ACT[ev]), mirror(A2[ev])),
    }
    P = {name: np.stack([predict(net, q) for net in nets]) for name, q in probes.items()}
    out = dict(predictor_check=acc, probes={})
    for name, p in P.items():
        dmean = np.array([np.mean(np.abs(p[i] - p[j])) for i, j in PAIRS])
        dq95 = np.array([np.quantile(np.abs(p[i] - p[j]), .95) for i, j in PAIRS])
        out["probes"][name] = dict(n=int(p.shape[1]),
                                   auroc_mean_abs=auroc(-dmean), auroc_q95_abs=auroc(-dq95),
                                   auroc_mean_abs_without_agent0=auroc(-dmean, NO_AGENT0),
                                   same_team_mean_distance=float(dmean[SAME].mean()),
                                   cross_team_mean_distance=float(dmean[~SAME].mean()))
    # 2x2: predictions of team-k agents on deliveries made at team-m goal (C) and the mirrored ones (A)
    dl = np.flatnonzero(DELIV == 1)
    q_c = X[dl]
    q_a = features(mirror(B[dl]), mirror_action(ACT[dl]), mirror(A2[dl]))
    pc = np.stack([predict(net, q_c) for net in nets]); pa = np.stack([predict(net, q_a) for net in nets])
    g = GOAL_AFTER[dl]
    table = {}
    for i in range(N_AG):
        table[f"agent{i} (team {TEAMS[i]})"] = {
            "C delivery at team0 goal": round(float(pc[i][g == 0].mean()), 3),
            "C delivery at team1 goal": round(float(pc[i][g == 1].mean()), 3),
            "A mirror of team0-goal delivery": round(float(pa[i][g == 0].mean()), 3),
            "A mirror of team1-goal delivery": round(float(pa[i][g == 1].mean()), 3)}
    out["delivery_table"] = table
    return out


report["4_mlp_all_experience"] = evaluate([fit(i) for i in range(N_AG)], "all")
report["4_mlp_experience_to_1.5M"] = evaluate([fit(i, ckpt_max=2) for i in range(N_AG)], "early")
report["meta"] = dict(transitions=int(len(R)), agents=N_AG, teams=TEAMS.tolist(),
                      ckpt_steps=D["ckpt_steps"].tolist(), pairs=len(PAIRS), same_pairs=int(SAME.sum()),
                      config=CFG)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


(OUT / "results.json").write_text(json.dumps(clean(report), indent=1), encoding="utf-8")
print(json.dumps(clean(report), indent=1))
