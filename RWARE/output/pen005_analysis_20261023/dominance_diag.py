"""Why does one agent take over?  Three read-only diagnostics on the four 1M runs (D76).  No training.

1. Dominant-agent removal: evaluate the frozen 1M policies (20 episodes x 500 steps, deterministic as in the
   trainer's evaluation and also stochastic) under three conditions:
     normal    - as trained
     noop      - the dominant agent always takes NOOP (it still occupies its cell)
     parked    - at reset the dominant agent is moved to a far top-corner cell, then always NOOP
                 (an evaluation-only change to a copy of the state; the environment code is unchanged)
   Deliveries are counted per carrier and split into dominant / its teammates / the other team.
2. Learned-agent policy probe across runs: the dominant agent of each run (the only one that learned) is
   taken from all four runs and compared on shared probes (PGCT gate rule: zero hidden, unmasked softmax,
   mean JS).  Same-goal pairs vs different-goal pairs (goal = the dominant agent's team goal in its run).
3. Absolute per-agent training deliveries per 0.1M window (deliveries/episode x 200 episodes x agent share).

    python output/pen005_analysis_20261023/dominance_diag.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "latent_team"))
from examples import train_recurrent_ippo_consensus as T  # noqa: E402
from examples.diagnose_pgct_rollout_compatibility import model_hash  # noqa: E402
from reward_correlation_baseline import load_model  # noqa: E402

OUT = Path(__file__).resolve().parent
PEN = ROOT / "runs/pilot_20261022_112521_b_e1f6_comm3_pen005"
IND = ROOT / "runs/queue_20261022_140114_b6_corner_indiv_pen005"
RUNS = {  # name: (run dir, canonical bank, dominant agent index, 0-based)
    "corner_team s1": (PEN / "b_e1f6_comm3_pen005_ippo_s1", PEN / "canonical_bank_b_e1f6_comm3_pen005_s1.pt", 1),
    "corner_team s2": (PEN / "b_e1f6_comm3_pen005_ippo_s2", PEN / "canonical_bank_b_e1f6_comm3_pen005_s2.pt", 4),
    "corner_indiv s1": (IND / "b6_corner_indiv_pen005_ippo_s1", IND / "canonical_bank_b6_corner_indiv_pen005_s1.pt", 2),
    "corner_indiv s2": (IND / "b6_corner_indiv_pen005_ippo_s2", IND / "canonical_bank_b6_corner_indiv_pen005_s2.pt", 0),
}
V2_GATE = ROOT / "output/env_view_20261015/probes_v2/probe_bank_v2_e1f_s1.pt"
EPISODES, HORIZON, SEED = 20, 500, 9300001


@torch.inference_mode()
def evaluate(ck, dominant, condition, deterministic):
    agents, _ = load_model(ck)
    h0 = model_hash(agents)
    c = T._load_checkpoint(str(ck), torch.device("cpu"))
    cfg, kw = c["config"], dict(c["effective_env_kwargs"])
    n = len(agents)
    per_carrier = np.zeros(n)
    rng = np.random.default_rng(SEED)
    teams = None
    for e in range(EPISODES):
        with T.make_env(cfg["env_id"], kw, cfg["observation_format"]) as env:
            obs, _ = env.reset(seed=SEED + e)
            base = env.unwrapped
            teams = [int(t) for t in np.asarray(base.agent_team_ids)]
            if condition == "parked":
                height, width = base.grid_size
                occupied = {(a.x, a.y) for a in base.agents}
                spot = next(p for p in [(0, 0), (width - 1, 0), (1, 0), (width - 2, 0)] if p not in occupied)
                base.agents[dominant].x, base.agents[dominant].y = spot
                base._recalc_grid()  # the first step's observation is the pre-move one; one step only
            ah = torch.zeros((n, cfg["recurrent_hidden_dim"]))
            ch = torch.zeros_like(ah)
            for _ in range(HORIZON):
                x = torch.as_tensor(np.asarray(T.flatten_multi_agent_obs(env, obs), dtype=np.float32))
                masks = torch.as_tensor(np.asarray(T.local_rware_action_masks(env, n, 5, enabled=True), dtype=np.float32))
                acts = []
                for i, net in enumerate(agents):
                    a, _, _, nah, nch = net.act(x[i], ah[i], ch[i], deterministic=deterministic, action_mask=masks[i])
                    ah[i], ch[i] = nah.squeeze(0), nch.squeeze(0)
                    acts.append(0 if (condition != "normal" and i == dominant) else int(a.item()))
                obs, r, term, trunc, info = env.step(acts)
                for ev in info.get("delivery_events", []):
                    k = int(ev["carrier_id"]) - 1
                    if 0 <= k < n:
                        per_carrier[k] += 1
                if bool(np.any(term)) or bool(np.any(trunc)):
                    break
    assert h0 == model_hash(agents)
    per_ep = per_carrier / EPISODES
    mates = [k for k in range(n) if k != dominant and teams[k] == teams[dominant]]
    others = [k for k in range(n) if teams[k] != teams[dominant]]
    return dict(per_agent=np.round(per_ep, 3).tolist(), dominant=round(float(per_ep[dominant]), 3),
                teammates=round(float(per_ep[mates].sum()), 3), other_team=round(float(per_ep[others].sum()), 3),
                teams=teams)


def js_matrix(nets, x):
    with torch.inference_mode():
        P = []
        for net in nets:
            h, _ = net.initial_hidden(len(x), torch.device("cpu"))
            logits, _ = net._actor_step(x, h)
            P.append(logits.softmax(-1).double().numpy())
    k = len(nets)
    D = np.zeros((k, k))
    A = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            m = 0.5 * (P[i] + P[j])
            kl = lambda a, b: (a * (np.log(np.maximum(a, 1e-12)) - np.log(np.maximum(b, 1e-12)))).sum(-1)
            D[i, j] = float((0.5 * kl(P[i], m) + 0.5 * kl(P[j], m)).mean())
            A[i, j] = float((P[i].argmax(-1) != P[j].argmax(-1)).mean())
    return D, A


def learned_probe():
    nets, goals, names = [], [], []
    for name, (run, _, dom) in RUNS.items():
        ck = run / "final.pt"
        agents, _ = load_model(ck)
        c = T._load_checkpoint(str(ck), torch.device("cpu"))
        kw = c["effective_env_kwargs"]
        with T.make_env(c["config"]["env_id"], dict(kw), c["config"]["observation_format"]) as env:
            env.reset(seed=0)
            team = int(np.asarray(env.unwrapped.agent_team_ids)[dom])
        nets.append(agents[dom])
        goals.append(tuple(kw["team_goal_positions"][team][0]))
        names.append(name)
    banks = {"v2_gate": torch.load(V2_GATE, map_location="cpu", weights_only=False)["sequences"][:, 0].float()}
    for name, (_, bank, _) in RUNS.items():
        banks[f"canonical {name}"] = torch.load(bank, map_location="cpu", weights_only=False)["sequences"][:, 0].float()
    out = dict(agents=names, goals=[list(g) for g in goals], banks={})
    k = len(nets)
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    for bname, x in banks.items():
        D, A = js_matrix(nets, x)
        same = [D[i, j] for i, j in pairs if goals[i] == goals[j]]
        diff = [D[i, j] for i, j in pairs if goals[i] != goals[j]]
        auc = float(np.mean([1.0 if s < d else 0.5 if s == d else 0.0 for s in same for d in diff]))
        out["banks"][bname] = dict(js=np.round(D, 3).tolist(), argmax_disagree=np.round(A, 3).tolist(),
                                   same_goal_mean=round(float(np.mean(same)), 4), diff_goal_mean=round(float(np.mean(diff)), 4),
                                   auroc_same_closer=round(auc, 3))
    return out


def absolute_counts():
    out = {}
    for name, (run, _, dom) in RUNS.items():
        rows = [json.loads(l) for l in open(run / "metrics.jsonl", encoding="utf-8")]
        tr = [r for r in rows if "train/toggle_count" in r["metrics"]]
        table = []
        for lo in np.arange(0, 1e6, 1e5):
            w = [r["metrics"] for r in tr if lo <= r["env_steps"] < lo + 1e5]
            pa = np.array([np.mean([m.get(f"train/agent_{k}_delivery_count", 0.0) for m in w]) for k in range(1, 7)])
            total = float(np.mean([m["train/deliveries"] for m in w])) * 200
            counts = (pa / pa.sum() * total) if pa.sum() > 0 else np.zeros(6)
            table.append(dict(window=f"{lo / 1e6:.1f}M", total=round(total, 1), per_agent=np.round(counts, 1).tolist()))
        out[name] = dict(dominant=dom, windows=table)
    return out


def main():
    torch.set_num_threads(4)
    res = {"absolute": absolute_counts()}
    print("== absolute training deliveries per 0.1M window (count)", flush=True)
    for name, v in res["absolute"].items():
        print(f"  {name} (dominant agent {v['dominant'] + 1}):", flush=True)
        for w in v["windows"]:
            print(f"     {w['window']} total {w['total']:6.1f}  " + " ".join(f"{x:5.1f}" for x in w["per_agent"]), flush=True)
    res["learned_probe"] = learned_probe()
    print("== learned-agent policy probe across runs; goals:", res["learned_probe"]["goals"], flush=True)
    for b, v in res["learned_probe"]["banks"].items():
        print(f"  {b}: same-goal mean JS {v['same_goal_mean']}  diff-goal {v['diff_goal_mean']}  AUROC(same closer) {v['auroc_same_closer']}", flush=True)
    res["removal"] = {}
    for name, (run, _, dom) in RUNS.items():
        for det in (True, False):
            for cond in ("normal", "noop", "parked"):
                r = evaluate(run / "final.pt", dom, cond, det)
                key = f"{name} {'det' if det else 'stoch'} {cond}"
                res["removal"][key] = r
                print(f"  {key:32s} dominant {r['dominant']:5.2f}  teammates {r['teammates']:5.2f}  other team {r['other_team']:5.2f}  per agent {r['per_agent']}", flush=True)
    (OUT / "dominance_diag.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
