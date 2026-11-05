"""Counterfactual split of the E38 reward probe (D60).  No training; weights hashed.

For every query of the real probe windows (state of the source agent at step t, its recorded action):
  * the source episode is replayed from its reset seed with the recorded joint actions (deterministic;
    checked against the recorded rewards),
  * at step t the base env is deep-copied twice, the source agent's team id is set to red / blue, the same
    joint action is stepped, and that agent's individual reward is read: r_red(q), r_blue(q).
  Q_diff = queries with r_red != r_blue,  Q_same = the rest.

Evaluated models (the 10 E1 models of E38) keep the E38 MLP length-1 reward predictor (own rollouts).
Per model x source x window type x subset {all, Q_diff, Q_same}:
  AUROC of D_ij = mean |r_hat_i - r_hat_j|  (same team closer = 1)
  corr(r_hat_i, r_own-team) vs corr(r_hat_i, r_other-team) over Q_diff       (hypothesis A)

    python output/trajectory_probe_20261019/counterfactual_split.py [--smoke]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trajectory_probe as TP  # noqa: E402
from source_swap import SOURCES, SOURCE_MODEL, E1_MODELS  # noqa: E402

OUT = Path(__file__).resolve().parent
SEED = 8300001  # same source rollouts / windows as E38


def counterfactual(src_ck, src, items, L, seed):
    """r[team, n, l] for each window step, and the replay check against recorded rewards."""
    cfg, env_kwargs = TP.env_of(src_ck)
    need = {}
    for n, (e, a, t) in enumerate(items):
        for l in range(L):
            need.setdefault(e, {}).setdefault(t + l, []).append((n, l, a))
    R = np.full((2, len(items), L), np.nan)
    mismatches = checks = 0
    with TP.T.make_env(cfg["env_id"], env_kwargs, cfg["observation_format"]) as env:
        for e in sorted(need):
            env.reset(seed=seed + e)
            last = max(need[e])
            for t in range(last + 1):
                joint = src["act"][e, t].tolist()
                for n, l, a in need[e].get(t, []):
                    for team in (0, 1):
                        base = copy.deepcopy(env.unwrapped)
                        ids = np.asarray(base.agent_team_ids).copy()
                        own = int(ids[a])
                        ids[a] = team
                        base.agent_team_ids = ids
                        _, rewards, *_ = base.step(joint)
                        R[team, n, l] = float(np.asarray(rewards)[a])
                        if team == own:
                            checks += 1
                            mismatches += int(abs(R[team, n, l] - float(src["rew"][e, t, a])) > 1e-6)
                env.step(joint)
    return R, dict(checks=checks, mismatches=mismatches)


def auroc_subset(pred, mask, teams):
    if not mask.any():
        return None
    per = {(i, j): np.abs(pred[i][mask] - pred[j][mask]).mean() for i, j in TP.PAIRS}
    return TP.score(TP.mat(per), teams)


def corr(x, y):
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


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
    out = OUT / ("smoke_cf" if args.smoke else "results_cf")
    out.mkdir(parents=True, exist_ok=True)
    banks, info = {}, {}
    for sname, ck in SOURCES.items():
        t0 = time.time()
        src = TP.rollouts(ck, ck, args.source_episodes, horizon, seed=SEED)
        w = TP.windows(src, args.L, args.per_type, seed=SEED)
        banks[sname] = {}
        for kind in ("carry", "random", "carry_memory"):
            if not w[kind]:
                continue
            obs, act = TP.window_arrays(src, w[kind], args.L)
            R, chk = counterfactual(ck, src, w[kind], args.L, seed=SEED)
            diff = np.abs(R[0] - R[1]) > 1e-9
            banks[sname][kind] = dict(obs=obs, act=act, R=R, diff=diff)
            info.setdefault(sname, {})[kind] = dict(windows=len(w[kind]), queries=int(diff.size),
                                                    q_diff=int(diff.sum()), replay=chk,
                                                    mean_abs_team_gap=float(np.abs(R[0] - R[1]).mean()))
            print(f"[source {sname}] {kind}: windows={len(w[kind])} Q_diff={int(diff.sum())}/{diff.size} "
                  f"replay mismatches={chk['mismatches']}/{chk['checks']}", flush=True)
        print(f"  ({time.time() - t0:.0f}s)", flush=True)
    models = dict(list(E1_MODELS.items())[:2]) if args.smoke else E1_MODELS
    results = dict(args=vars(args), sources=info, models={})
    for name, ck in models.items():
        t0 = time.time()
        cfg, env_kwargs = TP.env_of(ck)
        teams = TP.teams_in(cfg, env_kwargs)
        own = TP.rollouts(ck, ck, args.reward_episodes, horizon, seed=8200001)
        nets = []
        for a in range(4):
            mlp, _, _, _, scale = TP.train_reward(own, a, seed=300 + a, epochs=args.epochs)
            nets.append(mlp)
        row = dict(teams=teams, scores={})
        for sname, bank in banks.items():
            for kind, b in bank.items():
                N, L, D = b["obs"].shape
                with torch.no_grad():
                    pred = np.stack([m(b["obs"].reshape(N * L, D), b["act"].reshape(-1)).numpy().reshape(N, L) / scale
                                     for m in nets])
                cell = {}
                for sub, mask in (("all", np.ones_like(b["diff"])), ("Q_diff", b["diff"]), ("Q_same", ~b["diff"])):
                    cell[sub] = auroc_subset(pred, mask, teams)
                d = b["diff"]
                cell["corr_Q_diff"] = [dict(agent=i + 1, team=teams[i],
                                            own=corr(pred[i][d], b["R"][teams[i]][d]),
                                            other=corr(pred[i][d], b["R"][1 - teams[i]][d])) for i in range(4)]
                cell["corr_all"] = [dict(agent=i + 1, own=corr(pred[i].ravel(), b["R"][teams[i]].ravel()),
                                         other=corr(pred[i].ravel(), b["R"][1 - teams[i]].ravel())) for i in range(4)]
                row["scores"][f"{sname} | {kind}"] = cell
        results["models"][name] = row
        print(f"[{name}] ({time.time() - t0:.0f}s) " + " ; ".join(
            f"{k.split(' |')[0][:7]}/{k.split('| ')[1]}: " + " ".join(
                f"{s}={v[s]['auroc']:.2f}" if v[s] else f"{s}=-" for s in ("all", "Q_diff", "Q_same"))
            for k, v in row["scores"].items() if "carry_memory" not in k), flush=True)
        (out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    summarise(results, out / "summary.md")
    print(f"Wrote {out / 'summary.md'}", flush=True)


def summarise(results, path):
    neither = [n for n in results["models"] if n not in SOURCE_MODEL.values()]
    groups = {"8 models (neither source)": neither,
              "curriculum excl. sources": [n for n in neither if not n.startswith("scratch")],
              "from scratch excl. sources": [n for n in neither if n.startswith("scratch")]}
    lines = ["# Counterfactual split of the reward probe (E1, MLP length-1 predictors; seed 1)", "",
             "| source | window | windows | Q_diff / queries | mean |r_red - r_blue| |", "|---|---|---|---|---|"]
    for s, kinds in results["sources"].items():
        for k, v in kinds.items():
            lines.append(f"| {s} | {k} | {v['windows']} | {v['q_diff']} / {v['queries']} | {v['mean_abs_team_gap']:.4f} |")
    lines.append("")
    for gname, members in groups.items():
        members = [n for n in members if n in results["models"]]
        if not members:
            continue
        lines += [f"## {gname} ({len(members)})", "",
                  "| source / window | AUROC all | AUROC Q_diff | AUROC Q_same | corr own (Q_diff) | corr other (Q_diff) |",
                  "|---|---|---|---|---|---|"]
        for key in results["models"][members[0]]["scores"]:
            cells = [results["models"][n]["scores"][key] for n in members]
            def m(sub):
                v = [c[sub]["auroc"] for c in cells if c[sub]]
                return f"{np.mean(v):.2f}" if v else "-"
            def cc(which):
                v = [x[which] for c in cells for x in c["corr_Q_diff"] if x[which] is not None]
                return f"{np.mean(v):.2f}" if v else "-"
            lines.append(f"| {key} | {m('all')} | {m('Q_diff')} | {m('Q_same')} | {cc('own')} | {cc('other')} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
