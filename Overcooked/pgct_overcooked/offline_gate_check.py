"""Offline gate check: replay a trained job's policies and feed the SAME experience to fresh gates that differ
only in the public probe frame (absolute v1, facing v2, facing_cross v3). No training, nothing written.

  python -m pgct_overcooked.offline_gate_check --job Overcooked/runs/oc_pilot_500k/ippo_s1 \
         --schedule checkpoint_00250.pt:20 final.pt:20 --frames absolute facing facing_cross

--schedule replays several checkpoints in order (e.g. a mid-training policy that still explores, then the
final one), so the predictors see experience closer to what they see during training than with the final
policy alone. Policies act stochastically with a fixed torch seed, so every frame sees identical trajectories.
The gate uses the job's own configuration (contact_snapshot forced as in training). Labels are used only for
the printed identification summary.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from . import REPO_ROOT  # noqa: F401
from .experiment import build
from pgct_core.metrics import identification_summary
from pgct_core.reward_gate import RewardProbeGate
from pgct_core.trainer import load_agents


def parse_schedule(items):
    schedule = []
    for item in items:
        name, _, count = item.rpartition(":")
        if not name or not count.isdigit() or int(count) < 1:
            raise ValueError(f"schedule entries look like checkpoint.pt:UPDATES, got {item!r}")
        schedule.append((name, int(count)))
    return schedule


def owner_split(gate, i, j):
    """Analysis only (not a gate rule): distance on each agent's own probes separately, and their maximum."""
    from pgct_core.reward_gate import response_distance
    halves = []
    for k in (i, j):
        b, a, f = gate.predictors[k].event_probes()
        if not len(b):
            return None
        halves.append(response_distance(gate.predictors[i].predict(b, a, f), gate.predictors[j].predict(b, a, f)))
    return dict(on_i=halves[0], on_j=halves[1], owner_max=max(h for h in halves if h is not None) if all(
        h is not None for h in halves) else None)


def replay(job, frame, schedule, seed, gate_overrides=None):
    config = json.loads((job / "config.json").read_text(encoding="utf-8"))
    experiment = json.loads(json.dumps(config["experiment"]))
    experiment["adapter"]["probe_frame"] = frame
    make_adapter, trainer_cfg, ppo_cfg, gate_cfg, *_ = build(experiment, "pgct_d2", experiment["trainer"]["seed"])
    gate_cfg = replace(gate_cfg, contact_snapshot=True, **(gate_overrides or {}))
    adapter = make_adapter()
    gate = RewardProbeGate(adapter.n_agents, gate_cfg, adapter.projection)
    n, hidden = adapter.n_agents, ppo_cfg.recurrent_hidden_dim
    records, deliveries, update = [], {"own": np.zeros(n), "other": np.zeros(n)}, 0
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        obs = np.asarray(adapter.reset(seed=seed), dtype=np.float32)
        for checkpoint, updates in schedule:
            agents, _ = load_agents(job / checkpoint, adapter.obs_dim, adapter.n_actions, ppo_cfg)
            ah, ch = torch.zeros((n, hidden)), torch.zeros((n, hidden))
            for _ in range(updates):
                update += 1
                gate.snapshot()
                gate.active = True
                contact = np.zeros((n, n), dtype=bool)
                for _ in range(ppo_cfg.rollout_steps):
                    obs_t = torch.as_tensor(obs)
                    actions = []
                    with torch.no_grad():
                        for i, net in enumerate(agents):
                            a, _, _, ah_i, ch_i = net.act(obs_t[i], ah[i], ch[i])
                            actions.append(int(a.item()))
                            ah[i], ch[i] = ah_i.squeeze(0), ch_i.squeeze(0)
                    after, rewards, terminated, truncated, info = adapter.step(np.asarray(actions))
                    after = np.asarray(after, dtype=np.float32)
                    gate.observe(list(obs), np.asarray(actions), list(after), rewards)
                    contact |= adapter.contact_matrix()
                    deliveries["own"] += info["events"]["deliveries_own"]
                    deliveries["other"] += info["events"]["deliveries_other"]
                    if terminated or truncated:
                        after = np.asarray(adapter.reset(), dtype=np.float32)
                        ah.zero_(), ch.zero_()
                    obs = after
                gate.active = False
                gate(update, contact, active=True)
                records.append(dict(env_steps=update * ppo_cfg.rollout_steps, approved=gate.approved(),
                                    pairs=gate.last_log.get("ac_pairs", [])))
    return gate, adapter.oracle_team_ids(), records, deliveries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--schedule", nargs="+", default=["final.pt:40"], help="checkpoint:updates, in order")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--frames", nargs="+", default=["absolute", "facing", "facing_cross"])
    parser.add_argument("--gate", nargs="*", default=[], metavar="KEY=VALUE",
                        help="gate config overrides for this analysis, e.g. event_reward=0.1 min_event_response=0.06")
    parser.add_argument("--out", type=Path, help="optional JSON file for the results (never inside the job folder)")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    schedule = parse_schedule(args.schedule)
    overrides = {}
    for item in args.gate:
        key, _, value = item.partition("=")
        overrides[key] = json.loads(value)
    out = {}
    for frame in args.frames:
        gate, teams, records, deliveries = replay(args.job, frame, schedule, args.seed, overrides)
        ema = gate.ema + gate.ema.T
        iu = np.triu_indices(len(teams), 1)
        summary = identification_summary(records, teams)
        out[frame] = dict(schedule=args.schedule, teams=teams.tolist(), own_deliveries=deliveries["own"].tolist(),
                          other_deliveries=deliveries["other"].tolist(),
                          events_seen=[p.events_seen for p in gate.predictors], mature=gate.maturity().tolist(),
                          pairs=[f"{i}-{j}" for i, j in zip(*iu)], rounds=gate.rounds[iu].tolist(),
                          ema=np.round(ema[iu], 3).tolist(), approved=gate.approved().astype(int).tolist(),
                          final=summary["final"], distance_same=summary["distance_same_mean"],
                          distance_cross=summary["distance_cross_mean"],
                          first_correct=summary["first_correct_approval_step"],
                          first_wrong=summary["first_wrong_approval_step"], gate_overrides=overrides,
                          owner_split={f"{i}-{j}": owner_split(gate, i, j) for i, j in zip(*iu)
                                       if gate.predictors[i].events_seen and gate.predictors[j].events_seen})
        print(f"== {frame}: " + json.dumps(out[frame]), flush=True)
    if args.out is not None:
        if args.job.resolve() in args.out.resolve().parents:
            raise SystemExit("--out must not point inside the job folder")
        args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    main()
