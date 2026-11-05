"""Communication-graph diagnostics for a candidate environment (D63).  No training.

Builds the environment exactly as train_from_scratch would for (variant, seed) -- including the trainer's
own sensor-range and agent-count overrides -- and rolls out a uniform random policy over valid actions
(or a frozen checkpoint).  Per step, agents i and j are neighbours when max(|dx|, |dy|) <= the env's
communication range (the trainer's peer graph rule).  Reports:

  mean_neighbors          neighbours per agent
  isolated_rate           agent-steps with no neighbour
  fully_connected_rate    steps where every pair is in range
  teammate_available      agent-steps with at least one teammate in range
  same_pair_in_range      fraction of same-team pairs in range; cross_pair_in_range likewise

    python latent_team/connectivity_check.py --variants b_e1f6_comm3 a_e1f4_team [--episodes 5] [--checkpoint ck.pt]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples import train_recurrent_ippo_consensus as T  # noqa: E402

try:
    from . import common as C
    from . import train_from_scratch as TS
except ImportError:
    import common as C
    import train_from_scratch as TS


def trainer_env(variant, seed):
    """(env_id, env_kwargs, observation_format) as T.train would build them for this plan entry."""
    reference = json.loads((C.REFERENCE_EXPERIMENT / "experiment.json").read_text(encoding="utf-8"))
    cfg = TS.base_config(reference, "e1", seed=seed, timesteps=1024, home_ratio=.5, learning_rate=2e-4,
                         entropy_coef=.05, save_interval=0, output=Path("unused"), track=False, save_video=False,
                         smoke=False, env_variant=variant)
    config = T.TrainConfig(**cfg)
    env_kwargs = T.parse_env_kwargs(config.env_kwargs_json)
    if config.sensor_range > 0:
        env_kwargs["sensor_range"] = int(config.sensor_range)
    env_kwargs = T.apply_agent_overrides(config.env_id, env_kwargs, config.agent_count, config.team_count,
                                         config.auto_scale_request_queue)
    return config.env_id, env_kwargs, config.observation_format


def comm_graph(base):
    pos = np.asarray([(a.x, a.y) for a in base.agents])
    adj = np.max(np.abs(pos[:, None] - pos[None, :]), axis=-1) <= int(base.communication_range)
    np.fill_diagonal(adj, False)
    return adj


@torch.inference_mode()
def measure(env_id, env_kwargs, obs_format, episodes, horizon, seed, agents=None):
    rng = np.random.default_rng(seed)
    with T.make_env(env_id, env_kwargs, obs_format) as env:
        base = env.unwrapped
        stats = dict(steps=0, neighbors=0.0, isolated=0, full=0, mate=0, same_in=0, same_all=0, cross_in=0, cross_all=0)
        for e in range(episodes):
            obs, _ = env.reset(seed=seed + e)
            n = len(base.agents)  # agents exist only after reset
            teams = np.asarray(base.agent_team_ids)
            same = (teams[:, None] == teams[None, :]) & ~np.eye(n, dtype=bool)
            cross = teams[:, None] != teams[None, :]
            hidden = [net.initial_hidden(1, torch.device("cpu"))[0] for net in agents] if agents else None
            for _ in range(horizon):
                adj = comm_graph(base)
                stats["steps"] += 1
                stats["neighbors"] += adj.sum() / n
                stats["isolated"] += int((adj.sum(1) == 0).sum())
                stats["full"] += int(adj[~np.eye(n, dtype=bool)].all())
                stats["mate"] += int((adj & same).any(1).sum())
                stats["same_in"] += int((adj & same).sum()); stats["same_all"] += int(same.sum())
                stats["cross_in"] += int((adj & cross).sum()); stats["cross_all"] += int(cross.sum())
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
                obs, _, term, trunc, _ = env.step(actions)
                if bool(np.any(term)) or bool(np.any(trunc)):
                    break
        s = stats
        return dict(n_agents=n, communication_range=int(base.communication_range), sensor_range=int(base.sensor_range),
                    grid_size=list(map(int, base.grid_size)), steps=s["steps"],
                    mean_neighbors=s["neighbors"] / s["steps"], isolated_rate=s["isolated"] / (s["steps"] * n),
                    fully_connected_rate=s["full"] / s["steps"], teammate_available=s["mate"] / (s["steps"] * n),
                    same_pair_in_range=s["same_in"] / max(s["same_all"], 1),
                    cross_pair_in_range=s["cross_in"] / max(s["cross_all"], 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=500)
    ap.add_argument("--checkpoint", type=Path, help="Frozen policy instead of the random policy (same env).")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    rows = {}
    from reward_correlation_baseline import load_model  # any agent count
    agents = load_model(args.checkpoint)[0] if args.checkpoint else None
    for v in args.variants:
        env_id, kwargs, fmt = trainer_env(v, args.seed)
        rows[v] = measure(env_id, kwargs, fmt, args.episodes, args.horizon, seed=7700001, agents=agents)
        r = rows[v]
        print(f"{v:20s} agents={r['n_agents']} comm={r['communication_range']} grid={r['grid_size']} "
              f"neighbors={r['mean_neighbors']:.2f} isolated={r['isolated_rate']:.2f} "
              f"fully_connected={r['fully_connected_rate']:.2f} teammate_available={r['teammate_available']:.2f} "
              f"same/cross pair in range={r['same_pair_in_range']:.2f}/{r['cross_pair_in_range']:.2f}", flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(policy=str(args.checkpoint) if args.checkpoint else "uniform random",
                                               seed=args.seed, episodes=args.episodes, results=rows), indent=2),
                               encoding="utf-8")


if __name__ == "__main__":
    main()
