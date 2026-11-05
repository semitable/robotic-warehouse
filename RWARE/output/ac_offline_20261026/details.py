"""Two details behind results.json: (a) why delivery keys fail the strict support rule,
(b) the 6x6 event-probe distance matrix of the generalising predictor (all experience)."""
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

OUT = Path(__file__).resolve().parent
D = np.load(OUT / "rollouts.npz")
B, A2, ACT, R, AG, EP = D["before"], D["after"], D["action"], D["reward"], D["agent"], D["episode"]
DELIV, RET, TEAM = D["delivered"], D["returned"], D["team"]
torch.manual_seed(0)
torch.set_num_threads(4)
res = {}

# (a) exact A+C keys of each delivery: how often the same key paid ~+1
from collections import defaultdict
rows = []
for i in range(int(AG.max()) + 1):
    m = np.flatnonzero(AG == i)
    rewards, eps = defaultdict(list), defaultdict(set)
    keys = [B[t].tobytes() + bytes([int(ACT[t])]) + A2[t].tobytes() for t in m]
    for key, t in zip(keys, m):
        rewards[key].append(float(R[t])); eps[key].add(int(EP[t]))
    for key in {k for k, t in zip(keys, m) if DELIV[t] == 1}:
        rr = np.array(rewards[key])
        rows.append(dict(agent=i, n=len(rr), episodes=len(eps[key]), frac_paid=float((rr > .5).mean()),
                         std=float(rr.std(ddof=1)) if len(rr) > 1 else None))
big = [x for x in rows if x["n"] >= 8 and x["episodes"] >= 3]
res["a_delivery_keys"] = dict(distinct_delivery_keys=len(rows), keys_with_8plus_samples_3plus_episodes=len(big),
                              median_frac_paid=float(np.median([x["frac_paid"] for x in big])) if big else None,
                              median_std=float(np.median([x["std"] for x in big])) if big else None,
                              keys_passing_std_010=sum(1 for x in big if x["std"] <= .10))

# (b) event-probe distance matrix
X = np.concatenate([B.astype(np.float32), np.eye(5, dtype=np.float32)[ACT], A2.astype(np.float32)], 1)
train = (EP % 5) != 0
nets = []
for i in range(int(AG.max()) + 1):
    m = (AG == i) & train
    x = torch.as_tensor(X[m]); y = torch.as_tensor(R[m], dtype=torch.float32)
    net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(20):
        perm = torch.randperm(len(x))
        for s in range(0, len(x), 512):
            idx = perm[s:s + 512]
            loss = ((net(x[idx]).squeeze(-1) - y[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    nets.append(net.eval())
ev = np.flatnonzero((DELIV == 1) | (RET == 1))
with torch.inference_mode():
    P = np.stack([n(torch.as_tensor(X[ev])).squeeze(-1).numpy() for n in nets])
M = np.array([[np.mean(np.abs(P[i] - P[j])) for j in range(len(nets))] for i in range(len(nets))])
res["b_event_distance_matrix"] = M.round(3).tolist()
res["b_teams"] = [int(TEAM[AG == i][0]) for i in range(len(nets))]
(OUT / "details.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))
