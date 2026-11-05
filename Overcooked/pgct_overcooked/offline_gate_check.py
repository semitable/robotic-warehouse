"""Offline gate check: replay a trained job's policies and feed the SAME experience to fresh gates that differ
only in the public probe frame (absolute v1 vs facing v2). No training, nothing written to the job folder.

  python -m pgct_overcooked.offline_gate_check --job Overcooked/runs/oc_pilot_500k/ippo_s1 --updates 40

Policies act stochastically (as in training), with a fixed torch seed so both frames see identical trajectories.
The gate uses the job's own gate configuration (contact_snapshot forced as in training). Labels are used only
for the printed identification summary.
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
from pgct_core.reward_gate import RewardGateConfig, RewardProbeGate
from pgct_core.trainer import load_agents


def replay(job, frame, updates, seed, checkpoint):
    config = json.loads((job / "config.json").read_text(encoding="utf-8"))
    experiment = json.loads(json.dumps(config["experiment"]))
    experiment["adapter"]["probe_frame"] = frame
    make_adapter, trainer_cfg, ppo_cfg, gate_cfg, *_ = build(experiment, "pgct_d2", experiment["trainer"]["seed"])
    gate_cfg = replace(gate_cfg, contact_snapshot=True)
    adapter = make_adapter()
    agents, _ = load_agents(job / checkpoint, adapter.obs_dim, adapter.n_actions, ppo_cfg)
    gate = RewardProbeGate(adapter.n_agents, gate_cfg, adapter.projection)
    n, hidden = adapter.n_agents, ppo_cfg.recurrent_hidden_dim
    records, deliveries = [], np.zeros(n)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        obs = np.asarray(adapter.reset(seed=seed), dtype=np.float32)
        ah, ch = torch.zeros((n, hidden)), torch.zeros((n, hidden))
        for update in range(1, updates + 1):
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
                deliveries += info["events"]["deliveries_own"]
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
    parser.add_argument("--checkpoint", default="final.pt")
    parser.add_argument("--updates", type=int, default=40)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--frames", nargs="+", default=["absolute", "facing"])
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    out = {}
    for frame in args.frames:
        gate, teams, records, deliveries = replay(args.job, frame, args.updates, args.seed, args.checkpoint)
        ema = gate.ema + gate.ema.T
        summary = identification_summary(records, teams)
        out[frame] = dict(teams=teams.tolist(), own_deliveries=deliveries.tolist(),
                          events_seen=[p.events_seen for p in gate.predictors], mature=gate.maturity().tolist(),
                          rounds=gate.rounds[np.triu_indices(len(teams), 1)].tolist(),
                          ema_upper=np.round(ema[np.triu_indices(len(teams), 1)], 3).tolist(),
                          approved=gate.approved().astype(int).tolist(), final=summary["final"],
                          distance_same=summary["distance_same_mean"], distance_cross=summary["distance_cross_mean"],
                          first_correct=summary["first_correct_approval_step"],
                          first_wrong=summary["first_wrong_approval_step"])
        print(f"== {frame}: " + json.dumps(out[frame]), flush=True)
    return out


if __name__ == "__main__":
    main()
