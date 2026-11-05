"""Why do agents stop picking shelves up between ~0.5M and 1.5M steps?  Read-only; no training.

Part 1 (logs)    per 0.1M window of train metrics: P(TOGGLE | pickup valid) under the policy, pickups,
                 pickup success, carrying at episode end, shaping return.
Part 2 (frozen)  for checkpoints along each run (and a uniform random policy), roll the stochastic masked
                 policy and attribute every agent's per-step reward to components:
                   pickup_own (+.15) / pickup_wrong_team (-.05) / pickup_unrequested (-.02)   exact (wrapped
                   _pickup_shaping_reward), failed_forward (event x coef, outside shaping), invalid_toggle,
                   blocking, deadlock (events x coef), premature_drop (detected drop x coef),
                   task (delivery/return), other_shaping (residual: progress terms, noop/turn/stationary
                   penalties, wrong-return penalty).
                 Carry segments: from a pickup until the shelf is no longer carried (or the episode ends);
                 the summed reward of the segment by component, its length, and how it ended.

    python output/pickup_stall_20261021/pickup_stall.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "latent_team"))
from examples import train_recurrent_ippo_consensus as T  # noqa: E402
from reward_correlation_baseline import load_model  # noqa: E402  (any agent count)
from examples.diagnose_pgct_rollout_compatibility import model_hash  # noqa: E402

OUT = Path(__file__).resolve().parent
RUNS = {
    "A s1": ROOT / "runs/pilot_20261019_180044_a_e1f4_team/a_e1f4_team_ippo_s1",
    "A s2": ROOT / "runs/pilot_20261019_180044_a_e1f4_team/a_e1f4_team_ippo_s2",
    "B s1": ROOT / "runs/pilot_20261019_180044_b_e1f6_comm3/b_e1f6_comm3_ippo_s1",
    "B s2": ROOT / "runs/pilot_20261019_180044_b_e1f6_comm3/b_e1f6_comm3_ippo_s2",
    "E1-F IPPO": ROOT / "runs/e1f_20261018_124343_main_canonical/e1f_ippo_s1",
    "E1 IPPO": ROOT / "runs/scratch_20261015_093349_stage2/e1_ippo_s1",
}
CKPTS = ["checkpoint_00250.pt", "checkpoint_00500.pt", "checkpoint_00750.pt", "checkpoint_01000.pt",
         "checkpoint_01250.pt", "checkpoint_01500.pt"]
ROLLOUT_RUNS = ["A s1", "A s2", "B s2", "E1-F IPPO"]
LOG_KEYS = {"P(toggle|pickup valid)": "train/toggle_policy_probability_when_pickup_valid",
            "pickup valid rate": "train/valid_pickup_opportunity_rate",
            "toggles/ep": "train/toggle_count", "req pickups/ep": "train/requested_shelf_pickups",
            "pickup success": "train/pickup_success_rate", "carry at end": "train/carrying_at_episode_end",
            "shaping/ep": "train/episode_shaping_return", "deliveries/ep": "train/deliveries"}
COMPONENTS = ["pickup_own", "pickup_wrong_team", "pickup_unrequested", "failed_forward", "invalid_toggle",
              "blocking", "deadlock", "premature_drop", "task", "other_shaping"]


def part1():
    out = {}
    for name, d in RUNS.items():
        rows = [json.loads(l) for l in open(d / "metrics.jsonl", encoding="utf-8")]
        rows = [r for r in rows if "train/deliveries" in r["metrics"] and r["env_steps"] <= 2.0e6]
        series = {}
        for label, key in LOG_KEYS.items():
            vals = []
            for lo in np.arange(0, 2.0e6, 1e5):
                v = [r["metrics"][key] for r in rows if lo <= r["env_steps"] < lo + 1e5 and key in r["metrics"]
                     and np.isfinite(r["metrics"][key])]
                vals.append(round(float(np.mean(v)), 4) if v else None)
            series[label] = vals
        out[name] = series
    return out


@torch.inference_mode()
def rollouts(policy_ck, env_ck, episodes, horizon, seed):
    agents = load_model(policy_ck)[0] if policy_ck else None
    h0 = model_hash(agents) if agents else None
    c = T._load_checkpoint(str(env_ck), torch.device("cpu"))
    cfg, env_kwargs = c["config"], dict(c["effective_env_kwargs"])
    rng = np.random.default_rng(seed)
    comp = defaultdict(float)
    segs = []
    steps = 0
    with T.make_env(cfg["env_id"], env_kwargs, cfg["observation_format"]) as env:
        u = env.unwrapped
        coef = dict(failed=u.failed_forward_penalty, invalid=u.invalid_toggle_load_penalty,
                    blocking=u.blocking_agent_penalty, deadlock=u.deadlock_penalty, drop=u.premature_drop_penalty)
        original = u._pickup_shaping_reward
        pick_log = []

        def wrapped(agent, shelf):
            v = original(agent, shelf)
            kind = "pickup_own" if v > 0 or (v == 0 and u._is_agent_requested_shelf(agent, shelf)) else \
                ("pickup_wrong_team" if u._is_requested_shelf(shelf) else "pickup_unrequested")
            pick_log.append((agent.id - 1, kind, v))
            return v
        u._pickup_shaping_reward = wrapped
        for e in range(episodes):
            obs, _ = env.reset(seed=seed + e)
            n = len(u.agents)
            hidden = [net.initial_hidden(1, torch.device("cpu"))[0] for net in agents] if agents else None
            open_seg = [None] * n
            for t in range(horizon):
                carrying_before = [a.carrying_shelf is not None for a in u.agents]
                masks = np.asarray(T.local_rware_action_masks(env, n, 5, enabled=True), dtype=np.float64)
                if agents is None:
                    actions = [int(rng.choice(5, p=m / m.sum())) for m in masks]
                else:
                    x = np.asarray(T.flatten_multi_agent_obs(env, obs), dtype=np.float32)
                    actions = []
                    for i, net in enumerate(agents):
                        logits, hidden[i] = net._actor_step(torch.as_tensor(x[i]).unsqueeze(0), hidden[i])
                        p = net._masked_logits(logits, torch.as_tensor(masks[i], dtype=torch.float32)).softmax(-1)
                        p = p.numpy()[0].astype(np.float64)
                        actions.append(int(rng.choice(5, p=p / p.sum())))
                pick_log.clear()
                obs, rewards, term, trunc, info = env.step(actions)
                steps += 1
                r = np.asarray(rewards, dtype=np.float64)
                shaping = np.asarray(info["reward_shaping"], dtype=np.float64)
                task = r - shaping
                per = [defaultdict(float) for _ in range(n)]
                for i, kind, v in pick_log:
                    per[i][kind] += v
                for ev in info.get("failed_forward_events", []):
                    per[int(ev["agent_id"]) - 1]["failed_forward"] -= coef["failed"]
                for ev in info.get("invalid_toggle_events", []):
                    per[int(ev["agent_id"]) - 1]["invalid_toggle"] -= coef["invalid"]
                for ev in info.get("blocking_agent_events", []):
                    per[int(ev.get("blocker_id", ev.get("agent_id", 1))) - 1]["blocking"] -= coef["blocking"]
                for ev in info.get("deadlock_events", []):
                    per[int(ev["agent_id"]) - 1]["deadlock"] -= coef["deadlock"]
                delivered = {int(ev.get("carrier_id", 0)) - 1 for ev in info.get("delivery_events", [])}
                returned = {int(ev.get("agent_id", ev.get("carrier_id", 0))) - 1 for ev in info.get("return_events", [])}
                for i, a in enumerate(u.agents):
                    if carrying_before[i] and a.carrying_shelf is None and i not in returned and i not in delivered:
                        per[i]["premature_drop"] -= coef["drop"]
                    # failed_forward is subtracted from rewards outside the shaping vector, so remove it from task
                    explained_shaping = sum(per[i][k] for k in ("pickup_own", "pickup_wrong_team", "pickup_unrequested",
                                                                "invalid_toggle", "blocking", "deadlock", "premature_drop"))
                    per[i]["task"] = task[i] - per[i]["failed_forward"]
                    per[i]["other_shaping"] = shaping[i] - explained_shaping
                    for k, v in per[i].items():
                        comp[k] += v
                    # carry segments
                    if not carrying_before[i] and a.carrying_shelf is not None:
                        open_seg[i] = dict(start=t, total=0.0, parts=defaultdict(float), end="episode_end")
                    if open_seg[i] is not None:
                        open_seg[i]["total"] += float(r[i])
                        for k, v in per[i].items():
                            open_seg[i]["parts"][k] += v
                        if a.carrying_shelf is None:
                            open_seg[i]["end"] = "delivered" if i in delivered else ("returned" if i in returned else "dropped")
                            open_seg[i]["len"] = t - open_seg[i]["start"] + 1
                            segs.append(open_seg[i]); open_seg[i] = None
                if bool(np.any(term)) or bool(np.any(trunc)):
                    break
            for i in range(n):
                if open_seg[i] is not None:
                    open_seg[i]["len"] = horizon - open_seg[i]["start"]
                    segs.append(open_seg[i])
        u._pickup_shaping_reward = original
    if agents:
        assert h0 == model_hash(agents)
    agent_steps = steps * n
    per_100 = {k: round(100 * comp[k] / agent_steps, 4) for k in COMPONENTS}
    seg_summary = dict(count=len(segs), per_agent_episode=round(len(segs) / (episodes * n), 3))
    if segs:
        seg_summary.update(mean_len=round(float(np.mean([s["len"] for s in segs])), 1),
                           mean_return=round(float(np.mean([s["total"] for s in segs])), 4),
                           ends={k: sum(s["end"] == k for s in segs) for k in ("delivered", "returned", "dropped", "episode_end")},
                           mean_parts={k: round(float(np.mean([s["parts"].get(k, 0.0) for s in segs])), 4) for k in COMPONENTS})
    return dict(reward_per_100_agent_steps=per_100, carry_segments=seg_summary,
                pickup_reward_totals={k: round(comp[k], 3) for k in ("pickup_own", "pickup_wrong_team", "pickup_unrequested")})


def main():
    torch.set_num_threads(2)
    results = dict(logs=part1(), rollouts={})
    for name in RUNS:
        print(f"[log] {name}: " + " | ".join(f"{k}: {v[2]}->{v[7]}->{v[14]}" for k, v in results["logs"][name].items()
                                             if k in ("P(toggle|pickup valid)", "pickup valid rate", "shaping/ep")), flush=True)
    for name in ROLLOUT_RUNS:
        d = RUNS[name]
        env_ck = d / CKPTS[0]
        stages = [("random", None)] + [(c[11:16], d / c) for c in CKPTS if (d / c).exists()]
        results["rollouts"][name] = {}
        for label, ck in stages:
            row = rollouts(ck, env_ck, episodes=6, horizon=500, seed=7900001)
            results["rollouts"][name][label] = row
            s = row["carry_segments"]
            print(f"[{name} {label}] per100 {row['reward_per_100_agent_steps']} | segs/agent-ep {s['per_agent_episode']} "
                  f"len {s.get('mean_len')} return {s.get('mean_return')} ends {s.get('ends')}", flush=True)
        (OUT / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
