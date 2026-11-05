"""Offline rollouts for the A+C gate diagnosis (no training, no gradient step).

Runs saved IPPO checkpoints of runs/ac_20261023_185556 (corner goals, individual reward,
self_x/self_y masked as in training) with the stochastic masked policy and stores, per agent
and step, the A+C public view before/after (latent_team/reward_probe_ac.public_view, radius 2),
the action and the agent's OWN reward. Hidden labels (team, position, carrying-requested,
delivery carrier) are stored for evaluation only and never enter a predictor.

Experience pooled over several checkpoints stands in for "what the agent saw during training".
Usage: python output/ac_offline_20261026/collect.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "latent_team"))
from examples import train_recurrent_ippo_consensus as T  # noqa: E402
from reward_correlation_baseline import load_model  # noqa: E402
import reward_probe_ac as AC  # noqa: E402

OUT = Path(__file__).resolve().parent
RUN = ROOT / "runs/ac_20261023_185556/b6_corner_indiv_pen005_ac_ippo_s1"
CKPTS = ["checkpoint_00250.pt", "checkpoint_00750.pt", "checkpoint_01500.pt", "checkpoint_02500.pt",
         "checkpoint_03500.pt", "checkpoint_04750.pt", "final.pt"]
EPISODES, HORIZON, SEED, RADIUS = 8, 500, 9300001, 2
MASK_POLICY_COORDS = True  # A+C runs were trained with self_x/self_y masked; older runs were not
torch.set_num_threads(4)


@torch.inference_mode()
def rollout(ck_index, ck_name, rng, episode_offset):
    path = RUN / ck_name
    agents, meta = load_model(path)
    c = T._load_checkpoint(str(path), torch.device("cpu"))
    cfg, kw = c["config"], dict(c["effective_env_kwargs"])
    assert kw["team_reward_mode"] == "individual"
    rows = {k: [] for k in ("before", "after", "action", "reward", "agent", "team", "episode", "ckpt",
                            "pos_before", "pos_after", "req_before", "delivered", "returned", "goal_after")}
    with T.make_env(cfg["env_id"], kw, cfg["observation_format"]) as env:
        base = env.unwrapped
        for e in range(EPISODES):
            ep = episode_offset + e
            obs, _ = env.reset(seed=SEED + ep)
            n = len(base.agents)  # agents exist only after reset
            goals = {tuple(int(v) for v in g): int(t) for g, t in base.goal_team_ids.items()}
            teams = np.asarray(base.agent_team_ids, dtype=np.int8)
            hidden = [net.initial_hidden(1, torch.device("cpu"))[0] for net in agents]
            raw = np.asarray(T.flatten_multi_agent_obs(env, obs), dtype=np.float32)
            x = AC.strip_coordinates(raw)
            for t in range(HORIZON):
                masks = np.asarray(T.local_rware_action_masks(env, n, 5, enabled=True), dtype=np.float32)
                acts = []
                policy_in = x if MASK_POLICY_COORDS else raw
                for i, net in enumerate(agents):
                    logits, hidden[i] = net._actor_step(torch.as_tensor(policy_in[i]).unsqueeze(0), hidden[i])
                    p = net._masked_logits(logits, torch.as_tensor(masks[i])).softmax(-1).numpy()[0].astype(np.float64)
                    acts.append(int(rng.choice(5, p=p / p.sum())))
                pos_b = [(int(a.x), int(a.y)) for a in base.agents]
                obs, r, term, trunc, info = env.step(acts)
                raw = np.asarray(T.flatten_multi_agent_obs(env, obs), dtype=np.float32)
                y = AC.strip_coordinates(raw)
                pos_a = [(int(a.x), int(a.y)) for a in base.agents]
                carriers = {int(ev["carrier_id"]) - 1 for ev in info.get("delivery_events", [])}
                returners = {int(ev["carrier_id"]) - 1 for ev in info.get("return_events", [])}
                for i in range(n):
                    rows["before"].append(AC.public_view(x[i], RADIUS).astype(np.uint8))
                    rows["after"].append(AC.public_view(y[i], RADIUS).astype(np.uint8))
                    rows["action"].append(acts[i])
                    rows["reward"].append(float(r[i]))
                    rows["agent"].append(i)
                    rows["team"].append(int(teams[i]))
                    rows["episode"].append(ep)
                    rows["ckpt"].append(ck_index)
                    rows["pos_before"].append(pos_b[i])
                    rows["pos_after"].append(pos_a[i])
                    rows["req_before"].append(int(x[i][15] > .5))
                    rows["delivered"].append(int(i in carriers))
                    rows["returned"].append(int(i in returners))
                    rows["goal_after"].append(goals.get(pos_a[i], -1))  # team owning the goal cell, -1 = none
                x = y
                if bool(np.any(term)) or bool(np.any(trunc)):
                    break
    return rows, int(meta["env_steps"])


def main():
    global RUN, CKPTS, MASK_POLICY_COORDS
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, help="Run folder (default: A+C IPPO s1)")
    parser.add_argument("--ckpts", nargs="+")
    parser.add_argument("--out", default="rollouts.npz")
    parser.add_argument("--unmasked-policy", action="store_true", help="Policy trained with coordinates")
    args = parser.parse_args()
    RUN = (ROOT / args.run) if args.run else RUN
    CKPTS = args.ckpts or CKPTS
    MASK_POLICY_COORDS = not args.unmasked_policy
    rng = np.random.default_rng(SEED)
    allrows, steps = None, []
    for k, name in enumerate(CKPTS):
        t0 = time.time()
        rows, env_steps = rollout(k, name, rng, k * EPISODES)
        steps.append(env_steps)
        if allrows is None:
            allrows = rows
        else:
            for key in allrows:
                allrows[key].extend(rows[key])
        print(f"{name}: env_steps {env_steps}, transitions {len(rows['reward'])}, "
              f"deliveries {sum(rows['delivered'])}, {time.time() - t0:.0f}s", flush=True)
    data = {k: np.asarray(v) for k, v in allrows.items()}
    np.savez_compressed(OUT / args.out, ckpt_steps=np.asarray(steps), ckpt_names=np.asarray(CKPTS),
                        run=str(RUN), masked_policy=MASK_POLICY_COORDS, **data)
    print("saved", OUT / args.out, {k: v.shape for k, v in data.items()})


if __name__ == "__main__":
    main()
