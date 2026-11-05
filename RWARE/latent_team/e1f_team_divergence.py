"""Do teams behave differently in E1-F, and do the probes see it?  Frozen checkpoints only.

For every selected checkpoint of a from-scratch run:

  behaviour  stochastic masked rollouts; while an agent carries a requested shelf towards a goal
             (not the return phase), count its horizontal moves.  left_share = left / (left + right).
             E1-F goals are at the two bottom corners, so a team-aware policy has red ~1, blue ~0.
  probes     JS between the four actors on the v2 gate bank (carrying, no goal in view) and on the
             run's canonical bank; same-team pairs should be lower than cross-team pairs.

Team labels only score the results.  Writes only --output-dir.

    python latent_team/e1f_team_divergence.py --run runs/<plan folder> [--jobs e1f_ippo_s1] [--every 1000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples import train_recurrent_ippo_consensus as T  # noqa: E402
from examples.diagnose_pgct_rollout_compatibility import load_model, model_hash  # noqa: E402
from examples.pgct_probe_diagnostics import common_probe_response  # noqa: E402

try:
    from . import probes_v2 as P2
except ImportError:
    import probes_v2 as P2

PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]


def checkpoints(job_dir, every):
    cks = sorted(job_dir.glob("checkpoint_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    chosen = [p for p in cks if every <= 0 or int(p.stem.split("_")[1]) % every == 0]
    final = job_dir / "final.pt"
    return chosen + ([final] if final.exists() else [])


@torch.inference_mode()
def rollouts(agents, env_id, env_kwargs, cfg, *, episodes, horizon, seed):
    device = torch.device("cpu")
    rng = np.random.default_rng(seed)
    left, right, deliveries = np.zeros(4), np.zeros(4), np.zeros(4)
    team_ids = goals_x = None
    with T.make_env(env_id, env_kwargs, cfg["observation_format"]) as env:
        for e in range(episodes):
            obs, _ = env.reset(seed=seed + e)
            base = env.unwrapped
            team_ids = np.asarray(base.agent_team_ids).copy()
            goals_x = [float(np.mean([g[0] for g in goals])) for goals in base.goals_by_team]
            hidden = [net.initial_hidden(1, device)[0] for net in agents]
            for _ in range(horizon):
                x = np.asarray(T.flatten_multi_agent_obs(env, obs), dtype=np.float32)
                masks = np.asarray(T.local_rware_action_masks(env, 4, 5, enabled=cfg["action_mask"]), dtype=np.float32)
                delivering = [bool(x[i, P2.EGO["self_carrying_requested_shelf"]])
                              and not bool(x[i, P2.EGO["self_return_phase"]]) for i in range(4)]
                before = [int(a.x) for a in base.agents]
                actions = []
                for i, net in enumerate(agents):
                    logits, hidden[i] = net._actor_step(torch.as_tensor(x[i]).unsqueeze(0), hidden[i])
                    p = net._masked_logits(logits, torch.as_tensor(masks[i])).softmax(-1).numpy()[0].astype(np.float64)
                    actions.append(int(rng.choice(5, p=p / p.sum())))
                obs, _, terminated, truncated, info = env.step(actions)
                for i, a in enumerate(base.agents):
                    dx = int(a.x) - before[i]
                    if delivering[i] and dx:
                        (left if dx < 0 else right)[i] += 1
                for event in info.get("delivery_events", []):
                    if int(event.get("carrier_id", 0)) > 0:
                        deliveries[int(event["carrier_id"]) - 1] += 1
                if bool(np.any(terminated)) or bool(np.any(truncated)):
                    break
    moves = left + right
    share = np.where(moves > 0, left / np.maximum(moves, 1), np.nan)
    return dict(team_ids=team_ids.tolist(), team_goal_x=goals_x, left_moves=left.tolist(), right_moves=right.tolist(),
                left_share=share.tolist(), deliveries_per_episode=(deliveries / episodes).tolist())


def probe_scores(agents, bank, team_ids):
    m = np.asarray(common_probe_response(agents, bank, torch.device("cpu"))["mean_js"])
    same = [m[i, j] for i, j in PAIRS if team_ids[i] == team_ids[j]]
    cross = [m[i, j] for i, j in PAIRS if team_ids[i] != team_ids[j]]
    return dict(same=float(np.mean(same)), cross=float(np.mean(cross)), matrix=m.tolist())


def load_banks(run, cfg, out):
    v2_files = sorted(run.glob("probe_bank_v2_*.pt"))
    canonical_files = sorted(run.glob("canonical_bank_*.pt"))
    env_kwargs = json.loads(cfg["env_kwargs_json"])
    if v2_files:
        gate = torch.load(v2_files[0], map_location="cpu", weights_only=False)["sequences"].float()
    else:
        bank, _ = P2.build_bank(cfg["env_id"], env_kwargs, seed=int(cfg["seed"]))
        gate = bank["gate"]["sequences"].float()
    if canonical_files:
        canonical = torch.load(canonical_files[0], map_location="cpu", weights_only=False)["sequences"].float()
    else:  # same rule as the trainer's first canonical probe call (train_from_scratch.build_bank)
        from latent_team import train_from_scratch as TS
        path = out / "canonical_bank_rebuilt.pt"
        TS.build_bank(cfg, path)
        canonical = torch.load(path, map_location="cpu", weights_only=False)["sequences"].float()
    return dict(v2_gate=gate, canonical=canonical)


def analyse_job(run, job, args, out):
    rows, banks = [], None
    for ck in checkpoints(run / job, args.every):
        agents, meta = load_model(ck)
        cfg = meta["config"]
        env_kwargs = dict(T._load_checkpoint(str(ck), torch.device("cpu"))["effective_env_kwargs"])
        if banks is None:
            banks = load_banks(run, cfg, out)
        before = model_hash(agents)
        behaviour = rollouts(agents, cfg["env_id"], env_kwargs, cfg, episodes=args.episodes, horizon=args.horizon,
                             seed=args.seed)
        teams = behaviour["team_ids"]
        row = dict(checkpoint=ck.name, env_steps=int(meta["env_steps"]), behaviour=behaviour,
                   probes={name: probe_scores(agents, bank, teams) for name, bank in banks.items()})
        if before != model_hash(agents):
            raise RuntimeError("weights changed during frozen analysis")
        ls = np.asarray(behaviour["left_share"], float)
        tid = np.asarray(teams)
        row["team_left_share"] = [float(np.nanmean(ls[tid == t])) if np.isfinite(ls[tid == t]).any() else None
                                  for t in range(2)]
        rows.append(row)
        print(f"[{job}] {ck.name}: left_share={np.round(ls, 2).tolist()} "
              f"v2 same/cross={row['probes']['v2_gate']['same']:.3f}/{row['probes']['v2_gate']['cross']:.3f}", flush=True)
    return rows


def plot(results, path):
    jobs = list(results)
    fig, axes = plt.subplots(len(jobs), 3, figsize=(16, 3.6 * len(jobs)), squeeze=False, constrained_layout=True)
    colors = ["#d62728", "#1f77b4"]
    for r, job in enumerate(jobs):
        rows = results[job]
        x = [row["env_steps"] / 1e6 for row in rows]
        ax = axes[r, 0]
        for a in range(4):
            team = rows[0]["behaviour"]["team_ids"][a] if rows else a % 2
            ax.plot(x, [row["behaviour"]["left_share"][a] for row in rows], marker="o", color=colors[team],
                    ls=["-", "--"][a // 2], label=f"A{a + 1} (team {team})")
        ax.set_ylim(-.05, 1.05)
        ax.set_title(f"{job}: share of LEFT moves while delivering\n(red goal left -> ~1, blue goal right -> ~0)")
        ax.legend(fontsize=7)
        for c, name in enumerate(["v2_gate", "canonical"], start=1):
            ax = axes[r, c]
            ax.plot(x, [row["probes"][name]["same"] for row in rows], marker="o", color="#2ca02c", label="same team")
            ax.plot(x, [row["probes"][name]["cross"] for row in rows], marker="x", color="#7f7f7f", label="cross team")
            ax.axhline(.12, color="k", ls=":", lw=1)
            ax.set_title(f"{name} probe JS (teams separable if green < grey)")
            ax.legend(fontsize=7)
        for ax in axes[r]:
            ax.set_xlabel("env steps (M)")
    fig.savefig(path, dpi=120)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="Plan folder (contains experiment.json).")
    ap.add_argument("--jobs", nargs="+", help="Job folder names; default: every job in the plan.")
    ap.add_argument("--every", type=int, default=1000, help="Checkpoint update interval to analyse (0 = all).")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=500)
    ap.add_argument("--seed", type=int, default=5300001)
    ap.add_argument("--output-dir", type=Path)
    args = ap.parse_args()
    run = args.run.resolve()
    jobs = args.jobs or [j["name"] for j in json.loads((run / "experiment.json").read_text(encoding="utf-8"))["jobs"]]
    out = (args.output_dir or ROOT / "output" / f"e1f_divergence_{run.name}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = {job: analyse_job(run, job, args, out) for job in jobs}
    (out / "divergence.json").write_text(json.dumps(dict(run=str(run), args={k: str(v) for k, v in vars(args).items()},
                                                         results=results), indent=2), encoding="utf-8")
    plot(results, out / "divergence.png")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
