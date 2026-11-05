"""Reward-correlation baseline for team identification (D63).  No training; weights hashed.

Under team reward, teammates receive delivery/return rewards at the same step, so the reward streams alone
may reveal the teams.  Any probe-based team discovery has to beat this.  For a frozen checkpoint (in its
own training environment) this rolls out the policy, records every agent's reward per step and scores
pairs by
  corr_all     Pearson correlation of the full reward streams
  corr_events  correlation of large-reward events (|r| >= --event-threshold, i.e. deliveries / returns)
  cooccur      Jaccard overlap of the steps with large-reward events
Higher = more likely teammates.  Team labels only score (AUROC over same- vs cross-team pairs,
nearest-is-teammate).  Also reports how early the signal appears (first step with any event).

    python latent_team/reward_correlation_baseline.py --checkpoint runs/<plan>/<job>/checkpoint_01500.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples import train_recurrent_ippo_consensus as T  # noqa: E402
from examples.diagnose_pgct_rollout_compatibility import model_hash  # noqa: E402


def load_model(path):
    """Like diagnose_pgct_rollout_compatibility.load_model, but for any agent count (6/8-agent candidates)."""
    from types import SimpleNamespace
    checkpoint = T._load_checkpoint(str(path), torch.device("cpu"))
    cfg = checkpoint["config"]
    spec = T.get_semantic_observation_spec(SimpleNamespace(sensor_range=cfg["sensor_range"]))
    agents = []
    for state in checkpoint["agents"]:
        net = T.RecurrentActorCritic(spec.obs_dim, 5, cfg["mlp_hidden_dim"], cfg["recurrent_hidden_dim"],
                                     cfg["obs_encoder"], spec.spatial_channels, spec.spatial_size, spec.ego_dim)
        net.load_state_dict(state)
        net.eval()
        agents.append(net)
    return agents, dict(env_steps=checkpoint["env_steps"], config=cfg)


def pair_scores(R, threshold):
    """R [steps, n] rewards -> dict of n x n similarity matrices."""
    n = R.shape[1]
    ev = (np.abs(R) >= threshold).astype(np.float64)
    out = {k: np.full((n, n), np.nan) for k in ("corr_all", "corr_events", "cooccur")}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for key, a, b in (("corr_all", R[:, i], R[:, j]), ("corr_events", ev[:, i], ev[:, j])):
                if a.std() > 0 and b.std() > 0:
                    out[key][i, j] = float(np.corrcoef(a, b)[0, 1])
            union = ((ev[:, i] + ev[:, j]) > 0).sum()
            out["cooccur"][i, j] = float((ev[:, i] * ev[:, j]).sum() / union) if union else np.nan
    return out


def score(S, teams):
    n = len(teams)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    same = [S[i, j] for i, j in pairs if teams[i] == teams[j] and np.isfinite(S[i, j])]
    cross = [S[i, j] for i, j in pairs if teams[i] != teams[j] and np.isfinite(S[i, j])]
    if not same or not cross:
        return dict(auroc=None, nearest_is_mate=None, defined_pairs=len(same) + len(cross), pairs=len(pairs))
    auc = float(np.mean([1.0 if s > c else 0.5 if s == c else 0.0 for s in same for c in cross]))
    nearest = 0
    for i in range(n):
        row = [(S[i, j], j) for j in range(n) if j != i and np.isfinite(S[i, j])]
        if row:
            nearest += int(teams[max(row)[1]] == teams[i])
    return dict(auroc=auc, nearest_is_mate=nearest, defined_pairs=len(same) + len(cross), pairs=len(pairs))


@torch.inference_mode()
def rollouts(checkpoint, episodes, horizon, seed):
    agents, meta = load_model(checkpoint)
    h0 = model_hash(agents)
    cfg = meta["config"]
    env_kwargs = dict(T._load_checkpoint(str(checkpoint), torch.device("cpu"))["effective_env_kwargs"])
    rng = np.random.default_rng(seed)
    rewards = []
    with T.make_env(cfg["env_id"], env_kwargs, cfg["observation_format"]) as env:
        base = env.unwrapped
        for e in range(episodes):
            obs, _ = env.reset(seed=seed + e)
            n = len(base.agents)  # agents exist only after reset
            teams = [int(t) for t in np.asarray(base.agent_team_ids)]
            hidden = [net.initial_hidden(1, torch.device("cpu"))[0] for net in agents]
            for _ in range(horizon):
                x = np.asarray(T.flatten_multi_agent_obs(env, obs), dtype=np.float32)
                masks = np.asarray(T.local_rware_action_masks(env, n, 5, enabled=True), dtype=np.float32)
                actions = []
                for i, net in enumerate(agents):
                    logits, hidden[i] = net._actor_step(torch.as_tensor(x[i]).unsqueeze(0), hidden[i])
                    p = net._masked_logits(logits, torch.as_tensor(masks[i])).softmax(-1).numpy()[0].astype(np.float64)
                    actions.append(int(rng.choice(5, p=p / p.sum())))
                obs, r, term, trunc, _ = env.step(actions)
                rewards.append(np.asarray(r, dtype=np.float64))
                if bool(np.any(term)) or bool(np.any(trunc)):
                    break
    assert h0 == model_hash(agents)
    return np.stack(rewards), teams, env_kwargs.get("team_reward_mode"), int(meta["env_steps"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=500)
    ap.add_argument("--event-threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=7800001)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    torch.set_num_threads(2)
    results = {}
    for ck in args.checkpoint:
        R, teams, mode, steps = rollouts(ck, args.episodes, args.horizon, args.seed)
        S = pair_scores(R, args.event_threshold)
        ev = np.abs(R) >= args.event_threshold
        first = int(np.argmax(ev.any(1))) if ev.any() else None
        row = dict(team_reward_mode=mode, env_steps=steps, teams=teams, reward_steps=int(len(R)),
                   events_per_agent=ev.sum(0).tolist(), first_event_step=first,
                   scores={k: score(v, teams) for k, v in S.items()})
        results[str(ck)] = row
        line = " ".join(f"{k}={v['auroc'] if v['auroc'] is None else round(v['auroc'], 2)}" for k, v in row["scores"].items())
        print(f"{ck} [{mode}, {steps / 1e6:.2f}M] events/agent={row['events_per_agent']} {line}", flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(args={k: str(v) for k, v in vars(args).items()}, results=results),
                                          indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
