"""Source-swap check for E37 (D58).  Evaluated models fixed; only the policy that GENERATES the probe
windows changes.  No training; weights hashed (via trajectory_probe helpers).

  sources   Stage04 PGCT (curriculum lineage)  vs  scratch E1 All 6.4M (from-scratch lineage, 4/4 agents deliver)
            each in its own E1 environment (both E1 map, home .5)
  models    the 10 E1 models of E37; each source model is excluded from its own source's scores,
            and the headline comparison uses the 8 models that are neither source
  (a) policy  len1 (frame alone) | traj_all | traj_late (last half) | traj_last (last step) | traj_last2
  (b) reward  ONE GRU per agent (own rollouts) queried two ways:
                gru_len1   every frame alone (hidden reset), mean over steps
                gru_traj   the whole window, hidden carried, mean over steps
                gru_last   the whole window, last step only
              plus the E37 MLP length-1 baseline (mlp_len1)

    python output/trajectory_probe_20261019/source_swap.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trajectory_probe as TP  # noqa: E402

OUT = Path(__file__).resolve().parent
SOURCES = {"Stage04 (curriculum)": TP.CUR / "stage04_4ag_physical_home0.500/final.pt",
           "scratch All (from scratch)": TP.SCR / "e1_all_s1/checkpoint_06250.pt"}
SOURCE_MODEL = {"Stage04 (curriculum)": "Stage04 PGCT", "scratch All (from scratch)": "scratch E1 All 6.4M"}
E1_MODELS = {n: ck for n, (fam, ck) in TP.MODELS.items() if fam == "E1"}


def policy_scores(agents, obs, teams):
    seq, alone = TP.policy_probe(agents, obs)
    L = seq.shape[2]
    per = {k: {} for k in ("len1", "traj_all", "traj_late", "traj_last", "traj_last2")}
    for i, j in TP.PAIRS:
        per["len1"][(i, j)] = TP.js(alone[i], alone[j]).mean()
        d = TP.js(seq[i], seq[j])
        per["traj_all"][(i, j)] = d.mean()
        per["traj_late"][(i, j)] = d[:, L // 2:].mean()
        per["traj_last"][(i, j)] = d[:, -1].mean()
        per["traj_last2"][(i, j)] = d[:, -2:].mean()
    return {m: TP.score(TP.mat(v), teams) for m, v in per.items()}


@torch.no_grad()
def reward_scores(nets, scale, obs, act, teams):
    N, L, D = obs.shape
    mlp = np.stack([n[0](obs.reshape(N * L, D), act.reshape(-1)).numpy().reshape(N, L) for n in nets]) / scale
    g1 = np.stack([n[1](obs.reshape(N * L, 1, D), act.reshape(N * L, 1)).numpy().reshape(N, L) for n in nets]) / scale
    gt = np.stack([n[1](obs, act).numpy() for n in nets]) / scale
    per = {k: {} for k in ("mlp_len1", "gru_len1", "gru_traj", "gru_last")}
    for i, j in TP.PAIRS:
        per["mlp_len1"][(i, j)] = np.abs(mlp[i] - mlp[j]).mean()
        per["gru_len1"][(i, j)] = np.abs(g1[i] - g1[j]).mean()
        per["gru_traj"][(i, j)] = np.abs(gt[i] - gt[j]).mean()
        per["gru_last"][(i, j)] = np.abs(gt[i, :, -1] - gt[j, :, -1]).mean()
    return {m: TP.score(TP.mat(v), teams) for m, v in per.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--per-type", type=int, default=64)
    ap.add_argument("--source-episodes", type=int, default=60)
    ap.add_argument("--reward-episodes", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(2)
    if args.smoke:
        args.source_episodes, args.reward_episodes, args.epochs, args.per_type = 2, 2, 1, 8
    horizon = 120 if args.smoke else 500
    out = OUT / ("smoke_swap" if args.smoke else "results_swap")
    out.mkdir(parents=True, exist_ok=True)
    banks = {}
    for sname, ck in SOURCES.items():
        t0 = time.time()
        src = TP.rollouts(ck, ck, args.source_episodes, horizon, seed=8300001)
        w = TP.windows(src, args.L, args.per_type, seed=8300001)
        banks[sname] = {k: (TP.window_arrays(src, it, args.L) if it else None) for k, it in w.items()}
        print(f"[source {sname}] windows " + ", ".join(f"{k}={len(v)}" for k, v in w.items())
              + f" ({time.time() - t0:.0f}s)", flush=True)
    models = dict(list(E1_MODELS.items())[:2]) if args.smoke else E1_MODELS
    results = dict(args=vars(args), bank_sizes={s: {k: 0 if v is None else int(v[0].shape[0]) for k, v in b.items()}
                                                for s, b in banks.items()}, models={})
    for name, ck in models.items():
        t0 = time.time()
        cfg, env_kwargs = TP.env_of(ck)
        teams = TP.teams_in(cfg, env_kwargs)
        agents, _ = TP.load_model(ck)
        h0 = TP.model_hash(agents)
        row = dict(teams=teams, policy={}, reward={})
        own = TP.rollouts(ck, ck, args.reward_episodes, horizon, seed=8200001)
        nets, r2 = [], []
        for a in range(4):
            mlp, gru, rm, rg, scale = TP.train_reward(own, a, seed=300 + a, epochs=args.epochs)
            nets.append((mlp, gru)); r2.append((rm, rg))
        row["reward_r2"] = dict(mlp=float(np.mean([x[0] for x in r2])), gru=float(np.mean([x[1] for x in r2])))
        for sname, bank in banks.items():
            row["policy"][sname], row["reward"][sname] = {}, {}
            for kind, arr in bank.items():
                if arr is None:
                    continue
                row["policy"][sname][kind] = policy_scores(agents, arr[0], teams)
                row["reward"][sname][kind] = reward_scores(nets, scale, arr[0], arr[1], teams)
        assert h0 == TP.model_hash(agents)
        results["models"][name] = row
        print(f"[{name}] R2 mlp={row['reward_r2']['mlp']:.2f} gru={row['reward_r2']['gru']:.2f} ({time.time() - t0:.0f}s)",
              flush=True)
        (out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    summarise(results, out / "summary.md")
    print(f"Wrote {out / 'summary.md'}", flush=True)


def summarise(results, path):
    sources = list(SOURCES)
    neither = [n for n in results["models"] if n not in SOURCE_MODEL.values()]
    groups = {"8 models that are neither source": neither,
              "curriculum (Stage01-03, sp E1) excl. sources": [n for n in neither if not n.startswith("scratch")],
              "from scratch excl. sources": [n for n in neither if n.startswith("scratch")]}
    lines = ["# Source swap (E1): same evaluated models, probe windows from two source policies",
             "AUROC mean over models (same team closer = 1, chance .5; seed 1; 6 pairs per model).", ""]
    for part in ("policy", "reward"):
        for gname, members in groups.items():
            members = [n for n in members if results["models"][n][part]]
            if not members:
                continue
            lines += [f"## {part} — {gname} ({len(members)})", ""]
            metrics = list(next(iter(next(iter(results["models"][members[0]][part].values())).values())))
            lines.append("| source | window | " + " | ".join(metrics) + " |")
            lines.append("|---|---|" + "---|" * len(metrics))
            for s in sources:
                for kind in ("carry_memory", "carry", "random"):
                    vals = [results["models"][n][part][s][kind] for n in members if kind in results["models"][n][part][s]]
                    if vals:
                        n_win = results["bank_sizes"][s][kind]
                        lines.append(f"| {s} | {kind} (n={n_win}) | " + " | ".join(
                            f"{np.mean([v[m]['auroc'] for v in vals]):.2f}" for m in metrics) + " |")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
