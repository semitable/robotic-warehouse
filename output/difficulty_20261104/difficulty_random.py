"""How much harder are the two candidate variants for early exploration? Read-only, no training (T180).

Uniform random policy over valid actions (as E44/E55), 20 episodes x 500 steps, seed 1 layout, on the planned
environment of each variant (individual reward, 6 agents, penalty .005, contact range 5):
  base  b6_corner_indiv_comm5_pen005
  wide  b6_wide_corner_indiv_comm5_pen005   (11 x 16 map)
  req2  b6_corner_indiv_comm5_req2_pen005   (2 requested shelves per team)
Counted per agent-episode: pickups of a shelf the agent's own team requested, deliveries, reward events
(r >= .5, the gate's event threshold). Also the mean grid distance from shelves to the own team goal.
A random policy ignores rewards, so this measures how often an untrained learner meets a reward, not learnability.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples import train_recurrent_ippo_consensus as T  # noqa: E402
from latent_team import common as C  # noqa: E402
from latent_team.train_from_scratch import plan  # noqa: E402

OUT = Path(__file__).resolve().parent
VARIANTS = dict(base="b6_corner_indiv_comm5_pen005", wide="b6_wide_corner_indiv_comm5_pen005",
                req2="b6_corner_indiv_comm5_req2_pen005")
EPISODES, HORIZON, SEED = 20, 500, 8200001


def job_config(variant, tmp):
    args = SimpleNamespace(reference_experiment=C.REFERENCE_EXPERIMENT, output_dir=tmp / variant, maps=["e1"],
                           seeds=[1], modes=["ippo"], timesteps=4096, home_ratio=.5, learning_rate=2e-4,
                           entropy_coef=.05, save_interval=250, track=False, save_video=False, smoke_test=False,
                           eval_seed=5000001, stage_label="difficulty", wandb_project="test", wandb_mode="offline",
                           probe_bank="canonical", team_reward="individual", env_variant=variant,
                           reward_probe_ac=True, ac_gate="v2")
    cfg = plan(args)[1]["jobs"][0]["config"]
    kw = json.loads(cfg["env_kwargs_json"])
    kw.update(n_agents=cfg["agent_count"], sensor_range=cfg["sensor_range"])
    return cfg, kw


def measure(cfg, kw):
    rng = np.random.default_rng(SEED)
    own = deliv = events = 0
    dist = []
    with T.make_env(cfg["env_id"], kw, cfg["observation_format"]) as env:
        u = env.unwrapped
        for e in range(EPISODES):
            env.reset(seed=SEED + e)
            n = len(u.agents)
            for a in u.agents:
                goal = u.goals_by_team[u._agent_team_id(a)][0]
                dist.append(np.mean([abs(s.x - goal[0]) + abs(s.y - goal[1]) for s in u.shelfs]))
            for _ in range(HORIZON):
                masks = np.asarray(T.local_rware_action_masks(env, n, 5, enabled=True), dtype=np.float64)
                acts = [int(rng.choice(5, p=m / m.sum())) for m in masks]
                _, r, term, trunc, info = env.step(acts)
                events += int(np.sum(np.asarray(r, dtype=np.float64) >= .5))
                deliv += len(info.get("delivery_events", []))
                for ev in info.get("pickup_events", []):
                    a = u.agents[int(ev["agent_id"]) - 1]
                    sh = next(s for s in u.shelfs if s.id == ev["shelf_id"])
                    own += int(u._is_agent_requested_shelf(a, sh))
        grid = tuple(int(v) for v in u.grid_size)
        requests = list(u.request_queue_size_per_team)
    per = EPISODES * n
    return dict(grid_hw=grid, requests_per_team=requests, own_requested_pickups=own / per, deliveries=deliv / per, reward_events=events / per,
                mean_shelf_goal_distance=float(np.mean(dist)))


def main():
    import tempfile
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, variant in VARIANTS.items():
            out[name] = dict(variant=variant, **measure(*job_config(variant, Path(tmp))))
            r = out[name]
            print(f"{name:5s} grid {r['grid_hw']} requests {r['requests_per_team']}  per agent-episode: own requested pickups {r['own_requested_pickups']:.3f}, "
                  f"deliveries {r['deliveries']:.4f}, reward events {r['reward_events']:.4f}; "
                  f"mean shelf-goal distance {r['mean_shelf_goal_distance']:.1f}", flush=True)
    (OUT / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
