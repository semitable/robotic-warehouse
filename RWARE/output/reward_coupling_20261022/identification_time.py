"""How fast does the reward-coupling probe identify teams online?  Read-only; no training.

Same probe as reward_coupling.py (E49): agent i keeps, per shared probe "a delivery happens at goal cell g",
the mean of its own reward at steps where it saw another agent do that inside its view, minus its running
baseline reward.  Episodes are played in order and the tables are updated online.

A pair (i, j) can decide only when it meets: inside communication range (Chebyshev <= communication_range),
both have seen some common probe at least K times.  Decision: same team iff the mean |r_i - r_j| over those
probes is < 0.5.  The latest decision of each pair is kept until the next meeting.  Undecided pairs fall back
to no transfer, as an empty PGCT donor set does.

Policies: frozen B nopen checkpoints (0.26M, 0.51M, 0.77M, 1.00M; stochastic masked policy) and a uniform
random policy over valid actions (a stand-in for the untrained start).  Per run: correct / wrong / undecided
pairs (of 15) at fixed env-step marks, and the first step at which all 15 pairs are decided correctly.

    python output/reward_coupling_20261022/identification_time.py
"""
from __future__ import annotations

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
PLAN = ROOT / "runs/pilot_20261021_133652_b_e1f6_comm3_nopen"
CKPTS = ["checkpoint_00250.pt", "checkpoint_00500.pt", "checkpoint_00750.pt", "final.pt"]
EPISODES, HORIZON, SEED = 80, 500, 9200001
MARKS = [1000, 2500, 5000, 10000, 20000, 40000]
KS = (1, 3)


@torch.inference_mode()
def rollout(ck, use_policy):
    agents, _ = load_model(ck)
    h0 = model_hash(agents)
    c = T._load_checkpoint(str(ck), torch.device("cpu"))
    cfg, kw = c["config"], dict(c["effective_env_kwargs"])
    rng = np.random.default_rng(SEED)
    steps = []
    with T.make_env(cfg["env_id"], kw, cfg["observation_format"]) as env:
        base = env.unwrapped
        sensor, comm = int(base.sensor_range), int(base.communication_range)
        for e in range(EPISODES):
            obs, _ = env.reset(seed=SEED + e)
            n = len(base.agents)
            teams = [int(t) for t in np.asarray(base.agent_team_ids)]
            goals = {tuple(int(v) for v in g) for g in base.goals}
            hidden = [net.initial_hidden(1, torch.device("cpu"))[0] for net in agents]
            for _ in range(HORIZON):
                masks = np.asarray(T.local_rware_action_masks(env, n, 5, enabled=True), dtype=np.float64)
                if use_policy:
                    x = np.asarray(T.flatten_multi_agent_obs(env, obs), dtype=np.float32)
                    acts = []
                    for i, net in enumerate(agents):
                        logits, hidden[i] = net._actor_step(torch.as_tensor(x[i]).unsqueeze(0), hidden[i])
                        p = net._masked_logits(logits, torch.as_tensor(masks[i], dtype=torch.float32)).softmax(-1)
                        p = p.numpy()[0].astype(np.float64)
                        acts.append(int(rng.choice(5, p=p / p.sum())))
                else:
                    acts = [int(rng.choice(5, p=m / m.sum())) for m in masks]
                obs, r, term, trunc, info = env.step(acts)
                events = [(tuple(int(v) for v in ev["goal"]), int(ev["carrier_id"]) - 1)
                          for ev in info.get("delivery_events", [])]
                own = {k for _, k in events} | {int(ev["carrier_id"]) - 1 for ev in info.get("return_events", [])}
                steps.append(dict(r=np.asarray(r, dtype=np.float64), pos=[(int(a.x), int(a.y)) for a in base.agents],
                                  events=[ev for ev in events if ev[0] in goals], own=own))
                if bool(np.any(term)) or bool(np.any(trunc)):
                    break
    assert h0 == model_hash(agents)
    return steps, teams, sorted(goals), sensor, comm


def online(steps, teams, goals, sensor, comm, k):
    n, G = len(teams), len(goals)
    gi = {g: m for m, g in enumerate(goals)}
    S, C = np.zeros((n, G)), np.zeros((n, G))
    bs, bc = np.zeros(n), np.zeros(n)
    decision = np.full((n, n), -1)  # -1 undecided, 0 different, 1 same
    truth = np.array([[int(teams[i] == teams[j]) for j in range(n)] for i in range(n)])
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    marks, all_correct, first_event_seen, first_wrong = {}, None, None, None
    for t, st in enumerate(steps, start=1):
        seen = np.zeros(n, bool)
        for g, carrier in st["events"]:
            for i in range(n):
                if i == carrier or i in st["own"]:
                    continue
                if max(abs(st["pos"][i][0] - g[0]), abs(st["pos"][i][1] - g[1])) <= sensor:
                    S[i, gi[g]] += st["r"][i]; C[i, gi[g]] += 1; seen[i] = True
        for i in range(n):
            if not seen[i] and i not in st["own"]:
                bs[i] += st["r"][i]; bc[i] += 1
        if first_event_seen is None and seen.any():
            first_event_seen = t
        b = bs / np.maximum(bc, 1)
        R = S / np.maximum(C, 1) - b[:, None]
        for i, j in pairs:
            pi, pj = st["pos"][i], st["pos"][j]
            if max(abs(pi[0] - pj[0]), abs(pi[1] - pj[1])) > comm:
                continue
            ok = [m for m in range(G) if C[i, m] >= k and C[j, m] >= k]
            if ok:
                d = int(np.mean([abs(R[i, m] - R[j, m]) for m in ok]) < 0.5)
                decision[i, j] = decision[j, i] = d
                if d != truth[i, j] and first_wrong is None:
                    first_wrong = t
        state = [decision[i, j] for i, j in pairs]
        correct = sum(d == truth[i, j] for d, (i, j) in zip(state, pairs))
        wrong = sum(d != -1 and d != truth[i, j] for d, (i, j) in zip(state, pairs))
        if all_correct is None and correct == len(pairs):
            all_correct = t
        if t in MARKS:
            marks[t] = dict(correct=int(correct), wrong=int(wrong), undecided=int(len(pairs) - correct - wrong))
    return dict(first_event_seen=first_event_seen, all_15_correct_at=all_correct, first_wrong_decision_at=first_wrong,
                marks=marks, seen_per_agent=C.sum(1).astype(int).tolist())


def main():
    torch.set_num_threads(4)
    results = {}
    for s in (1, 2):
        run = PLAN / f"b_e1f6_comm3_nopen_ippo_s{s}"
        jobs = [("uniform random", run / "checkpoint_00250.pt", False)] + [(ck, run / ck, True) for ck in CKPTS]
        for label, ck, use_policy in jobs:
            steps, teams, goals, sensor, comm = rollout(ck, use_policy)
            deliveries = sum(len(st["events"]) for st in steps)
            for k in KS:
                res = online(steps, teams, goals, sensor, comm, k)
                res.update(deliveries=deliveries, total_steps=len(steps))
                results[f"s{s} {label} K={k}"] = res
                print(f"s{s} {label:20s} K={k} deliveries={deliveries:4d} first_seen={res['first_event_seen']} "
                      f"all15={res['all_15_correct_at']} first_wrong={res['first_wrong_decision_at']} "
                      f"marks={ {m: (v['correct'], v['wrong'], v['undecided']) for m, v in res['marks'].items()} }", flush=True)
    (OUT / "identification_time.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
