"""Check that Experiment A's goal evidence is readable from the observer's own observation.

Experiment A classifies pickups/deliveries with environment events and request/goal
sets. This frozen replay checks, for every counted (observer, target, event), that the
observer's post-step semantic observation shows the same thing at the target's relative
cell: an agent, loaded, and the observer's requested_shelf (pickup) or relevant_goal
(delivery) channel equal to the evidence label. The target's team id is never read.

What the observation does NOT contain is the target's identity: agent cells are
anonymous. Attributing the event to a specific neighbour therefore assumes the same
neighbour identification that PGCT already assumes for communication.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from . import common as C
    from .exp_a_team_identifiability import MODELS, environment_for, visible
except ImportError:
    import common as C
    from exp_a_team_identifiability import MODELS, environment_for, visible

from examples import train_recurrent_ippo_consensus as T
from examples.diagnose_pgct_rollout_compatibility import load_model, model_hash
from rware.utils.semantic_observation import SEMANTIC_SPATIAL_CHANNELS

CH = {name: k for k, name in enumerate(SEMANTIC_SPATIAL_CHANNELS)}


@torch.inference_mode()
def verify(name, episodes, horizon, seed):
    checkpoint, env_source, _ = MODELS[name]
    agents, metadata = load_model(checkpoint)
    env_id, env_kwargs = environment_for(checkpoint, env_source)
    cfg = metadata["config"]
    before = model_hash(agents)
    rng = np.random.default_rng(seed)
    counts = {k: dict(checked=0, agent_and_loaded=0, label_matches_observation=0)
              for k in ("pickup", "delivery")}
    for e in range(episodes):
        with T.make_env(env_id, env_kwargs, cfg["observation_format"]) as env:
            obs, _ = env.reset(seed=seed + e)
            base = env.unwrapped
            spec = T.get_semantic_observation_spec(env)
            r, s = int(base.sensor_range), spec.spatial_size
            team_ids = np.asarray(base.agent_team_ids)
            goals_by_team = [{tuple(int(v) for v in g) for g in goals} for goals in base.goals_by_team]
            hidden = [net.initial_hidden(1, torch.device("cpu"))[0] for net in agents]
            for _ in range(horizon):
                x = np.asarray(T.flatten_multi_agent_obs(env, obs), dtype=np.float32)
                masks = np.asarray(T.local_rware_action_masks(env, 4, 5, enabled=cfg["action_mask"]), dtype=np.float32)
                requested_by_team = [{int(sh.id) for sh in q} for q in base.team_request_queues]
                actions = []
                for i, net in enumerate(agents):
                    logits, hidden[i] = net._actor_step(torch.as_tensor(x[i]).unsqueeze(0), hidden[i])
                    p = net._masked_logits(logits, torch.as_tensor(masks[i])).softmax(-1).numpy()[0].astype(np.float64)
                    actions.append(int(rng.choice(5, p=p / p.sum())))
                obs, _, terminated, truncated, info = env.step(actions)
                post = np.asarray(T.flatten_multi_agent_obs(env, obs), dtype=np.float32)
                grid = post[:, spec.ego_dim:].reshape(4, spec.spatial_channels, s, s)
                positions = [(int(a.x), int(a.y)) for a in base.agents]
                seen = visible(positions, r)
                events = [("pickup", int(ev["agent_id"]) - 1, int(ev["shelf_id"]), None)
                          for ev in info.get("pickup_events", []) if not ev.get("return_phase")]
                events += [("delivery", int(ev["carrier_id"]) - 1, None, tuple(int(v) for v in ev["goal"]))
                           for ev in info.get("delivery_events", []) if int(ev.get("carrier_id", 0)) > 0]
                for kind, j, shelf, goal in events:
                    for i in np.flatnonzero(seen[:, j]):
                        row, col = positions[j][1] - positions[i][1] + r, positions[j][0] - positions[i][0] + r
                        cell = grid[i, :, row, col]
                        if kind == "pickup":
                            label = shelf in requested_by_team[team_ids[i]]
                            shown = bool(cell[CH["requested_shelf"]])
                        else:
                            label = goal in goals_by_team[team_ids[i]]
                            shown = bool(cell[CH["relevant_goal"]])
                        c = counts[kind]
                        c["checked"] += 1
                        c["agent_and_loaded"] += int(cell[CH["agent"]] == 1 and cell[CH["agent_loaded"]] == 1)
                        c["label_matches_observation"] += int(label == shown)
                if bool(np.any(terminated)) or bool(np.any(truncated)):
                    break
    if before != model_hash(agents):
        raise RuntimeError("Model weights changed during frozen replay.")
    return dict(checkpoint=str(Path(checkpoint).resolve()), episodes=episodes, horizon=horizon, seed=seed,
                weights_unchanged=True, counts=counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=["e1_ippo_s1", "e2_ippo_s1"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--seed", type=int, default=5200001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    result = {name: verify(name, args.episodes, args.horizon, args.seed) for name in args.models}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({n: r["counts"] for n, r in result.items()}, indent=1))


if __name__ == "__main__":
    main()
