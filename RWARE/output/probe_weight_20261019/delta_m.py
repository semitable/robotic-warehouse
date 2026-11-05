"""Per-probe team separation Delta_m and probe weighting for the policy probe.  Frozen models only.

Delta_m = mean_{cross-team pairs} JS_m - mean_{same-team pairs} JS_m  (PGCT rule: zero hidden, unmasked, ln JS).
Team labels are used only offline, to score probes and to build diagnostic (oracle) weights.

Weightings of the per-probe JS (applied to every pair alike, so D stays symmetric):
  uniform            current PGCT
  reward magnitude   max_a |r(s_m, a)| from the real env, averaged over teams      (user's idea, literal)
  team reward diff   max_a |r_red(s_m, a) - r_blue(s_m, a)| from the real env       (designer prior)
  agent spread       variance of the 4 actors' action distributions on probe m     (label-free)
  oracle held-out    max(Delta_m, 0) estimated on OTHER data (leave-one-model-out / previous checkpoint)
  oracle in-sample   max(Delta_m, 0) on the same data (optimistic, upper bound only)

Banks are shared across the models compared, so held-out weights are defined on the same probes:
  E1 family (Stage01-04, sp E1 IPPO/PGCT): v2 E1 gate bank, and the same bank with team-relative inputs zeroed
  E1-F (IPPO/PGCT/Oracle/All/PGCT-v2 checkpoints): v2 E1-F gate bank

    python output/probe_weight_20261019/delta_m.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples import train_recurrent_ippo_consensus as T  # noqa: E402
from examples.diagnose_pgct_rollout_compatibility import load_model, model_hash  # noqa: E402
from latent_team import probes_v2 as P2  # noqa: E402

sys.path.insert(0, str(ROOT / "output/probe_connect_20261019"))
from analyze import ablate  # noqa: E402  (same team-relative input removal as E30)

OUT = Path(__file__).resolve().parent
CUR = ROOT / "runs/pgct09700_physical_mixing_20260929_155338_079437"
SP = ROOT / "runs/sp_20261008_120638_983201"
E1F_C = ROOT / "runs/e1f_20261018_124343_main_canonical"
E1F_V = ROOT / "runs/e1f_20261018_124343_main_v2"
E1_BANK = ROOT / "output/env_view_20261015/probes_v2/probe_bank_v2_e1_s1.pt"
E1F_BANK = E1F_V / "probe_bank_v2_e1f_s1.pt"
E1_MODELS = {
    "Stage01": CUR / "stage01_4ag_physical_home1.000/final.pt",
    "Stage02": CUR / "stage02_4ag_physical_home0.875/final.pt",
    "Stage03": CUR / "stage03_4ag_physical_home0.750/final.pt",
    "Stage04 PGCT": CUR / "stage04_4ag_physical_home0.500/final.pt",
    "E1 IPPO": SP / "e1_ippo_s1/final.pt",
    "E1 PGCT": SP / "e1_pgct_s1/final.pt",
}
E1F_JOBS = {"IPPO": E1F_C / "e1f_ippo_s1", "PGCT": E1F_C / "e1f_pgct_s1", "Oracle": E1F_C / "e1f_oracle_s1",
            "All": E1F_C / "e1f_all_s1", "PGCT-v2": E1F_V / "e1f_pgct_s1"}
E1F_CKPTS = ["checkpoint_00500.pt", "checkpoint_01000.pt", "checkpoint_01500.pt"]
PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]
N_ACT = 5


@torch.inference_mode()
def per_probe(agents, seq):
    P = []
    for net in agents:
        h, _ = net.initial_hidden(len(seq), torch.device("cpu"))
        logits, _ = net._actor_step(seq, h)
        P.append(logits.softmax(-1).numpy().astype(np.float64))
    P = np.stack(P)  # [agent, probe, action]
    kl = lambda a, b: (a * (np.log(np.maximum(a, 1e-12)) - np.log(np.maximum(b, 1e-12)))).sum(-1)
    js = {}
    for i, j in PAIRS:
        m = (P[i] + P[j]) / 2
        js[(i, j)] = 0.5 * kl(P[i], m) + 0.5 * kl(P[j], m)
    return js, P


def delta(js, teams):
    same = np.mean([v for (i, j), v in js.items() if teams[i] == teams[j]], 0)
    cross = np.mean([v for (i, j), v in js.items() if teams[i] != teams[j]], 0)
    return cross - same


def score(js, teams, w):
    D = np.zeros((4, 4))
    for (i, j), v in js.items():
        D[i, j] = D[j, i] = float(np.sum(w * v) / max(np.sum(w), 1e-12))
    mate = [next(j for j in range(4) if j != i and teams[j] == teams[i]) for i in range(4)]
    same = [D[i, j] for i, j in PAIRS if teams[i] == teams[j]]
    cross = [D[i, j] for i, j in PAIRS if teams[i] != teams[j]]
    auc = float(np.mean([1.0 if s < c else 0.5 if s == c else 0.0 for s in same for c in cross]))
    nearest = int(sum(min((j for j in range(4) if j != i), key=lambda j: D[i, j]) == mate[i] for i in range(4)))
    return dict(auroc=auc, nearest_is_mate=nearest, gap=float(np.mean(cross) - np.mean(same)))


def oracle_rewards(ck, bank):
    """r[team, probe, action] from the real env with an agent of that team in the probe's physical state."""
    agents, meta = load_model(ck)
    cfg = meta["config"]
    env_kwargs = dict(T._load_checkpoint(str(ck), torch.device("cpu"))["effective_env_kwargs"])
    real = P2.ProbeRealiser(cfg["env_id"], env_kwargs, cfg["observation_format"], seed=1)
    names = {f.name for f in fields(P2.Scenario)}
    R = np.zeros((2, len(bank["gate_scenarios"]), N_ACT))
    try:
        for t in range(2):
            for m, sc in enumerate(bank["gate_scenarios"]):
                for a in range(N_ACT):
                    real.realise(P2.Scenario(**{**{n: sc[n] for n in names if n in sc}, "team": t}))
                    _, rewards, *_ = real.env.step([a, 0, 0, 0])
                    R[t, m, a] = float(np.asarray(rewards)[0])
    finally:
        real.close()
    return R


def teams_of(ck):
    agents, meta = load_model(ck)
    cfg = meta["config"]
    env_kwargs = dict(T._load_checkpoint(str(ck), torch.device("cpu"))["effective_env_kwargs"])
    with T.make_env(cfg["env_id"], env_kwargs, cfg["observation_format"]) as env:
        env.reset(seed=0)
        return [int(t) for t in np.asarray(env.unwrapped.agent_team_ids)]


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def analyse(family, items, bank_seq, R, holdout):
    """items: {name: (js, P, teams)}; holdout: 'loo' or 'previous' (ordered names)."""
    w_mag = np.abs(R).max(2).mean(0)
    w_diff = np.abs(R[0] - R[1]).max(1)
    deltas = {n: delta(js, teams) for n, (js, P, teams) in items.items()}
    names = list(items)
    rows = {}
    for k, n in enumerate(names):
        js, P, teams = items[n]
        if holdout == "loo":
            ref = [deltas[o] for o in names if o != n]
        else:  # previous checkpoint of the same job
            prev = names[k - 1] if k > 0 and names[k - 1].split("@")[0] == n.split("@")[0] else None
            ref = [deltas[prev]] if prev else []
        w_ho = np.maximum(np.mean(ref, 0), 0) if ref else None
        weights = {"uniform": np.ones(len(bank_seq)), "reward magnitude": w_mag, "team reward diff": w_diff,
                   "agent spread": P.var(0).sum(-1), "oracle in-sample": np.maximum(deltas[n], 0)}
        if w_ho is not None and w_ho.sum() > 0:
            weights["oracle held-out"] = w_ho
        d = deltas[n]
        rows[n] = dict(delta_mean=float(d.mean()), delta_pos_frac=float((d > 0).mean()),
                       delta_max=float(d.max()), delta_min=float(d.min()),
                       spearman_delta_vs_reward_magnitude=spearman(d, w_mag),
                       spearman_delta_vs_team_reward_diff=spearman(d, w_diff),
                       weighted={k2: score(js, teams, w) for k2, w in weights.items()})
        line = " ".join(f"{k2}={v['auroc']:.2f}" for k2, v in rows[n]["weighted"].items())
        print(f"[{family}] {n:22s} Δ mean={d.mean():+.4f} Δ>0 {100 * (d > 0).mean():3.0f}% | AUROC {line} | "
              f"ρ(Δ,|r|)={rows[n]['spearman_delta_vs_reward_magnitude']:+.2f} ρ(Δ,team diff)="
              f"{rows[n]['spearman_delta_vs_team_reward_diff']:+.2f}", flush=True)
    return rows, deltas, dict(reward_magnitude=w_mag.tolist(), team_reward_diff=w_diff.tolist(),
                              probes_with_team_reward_diff=int((w_diff > 1e-9).sum()))


def main():
    torch.set_num_threads(2)
    results = {}
    # E1 family, common v2 E1 bank (+ ablated)
    e1_bank = torch.load(E1_BANK, map_location="cpu", weights_only=False)
    R_e1 = oracle_rewards(SP / "e1_ippo_s1/final.pt", e1_bank)
    for label, seq in (("E1 v2", e1_bank["sequences"][:, 0].float()),
                       ("E1 v2 ablated", ablate(e1_bank["sequences"].float())[:, 0])):
        items = {}
        for n, ck in E1_MODELS.items():
            agents, _ = load_model(ck); h = model_hash(agents)
            js, P = per_probe(agents, seq)
            assert h == model_hash(agents)
            items[n] = (js, P, [0, 1, 0, 1])
        rows, deltas, rew = analyse(label, items, seq, R_e1, "loo")
        results[label] = dict(rows=rows, reward=rew, deltas={k: v.tolist() for k, v in deltas.items()})
    # E1-F checkpoints, common v2 E1-F bank
    f_bank = torch.load(E1F_BANK, map_location="cpu", weights_only=False)
    seq = f_bank["sequences"][:, 0].float()
    R_f = oracle_rewards(E1F_C / "e1f_ippo_s1" / E1F_CKPTS[-1], f_bank)
    teams = teams_of(E1F_C / "e1f_ippo_s1" / E1F_CKPTS[-1])
    items = {}
    for job, d in E1F_JOBS.items():
        for c in E1F_CKPTS:
            agents, meta = load_model(d / c); h = model_hash(agents)
            js, P = per_probe(agents, seq)
            assert h == model_hash(agents)
            items[f"{job}@{meta['env_steps'] / 1e6:.2f}M"] = (js, P, teams)
    rows, deltas, rew = analyse("E1-F v2", items, seq, R_f, "previous")
    results["E1-F v2"] = dict(rows=rows, reward=rew, teams=teams, deltas={k: v.tolist() for k, v in deltas.items()})
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot(results)
    print(f"probes with any team reward difference: E1 {results['E1 v2']['reward']['probes_with_team_reward_diff']}/48,"
          f" E1-F {results['E1-F v2']['reward']['probes_with_team_reward_diff']}/48", flush=True)


def plot(results):
    fams = list(results)
    fig, axes = plt.subplots(len(fams), 1, figsize=(13, 3.6 * len(fams)), constrained_layout=True)
    for ax, fam in zip(axes, fams):
        D = np.array(list(results[fam]["deltas"].values()))
        im = ax.imshow(D, aspect="auto", cmap="RdBu_r", vmin=-0.3, vmax=0.3)
        ax.set_yticks(range(len(D)), list(results[fam]["deltas"]), fontsize=7)
        ax.set_xlabel("probe (v2 gate state)")
        diff = np.array(results[fam]["reward"]["team_reward_diff"]) > 1e-9
        for m in np.flatnonzero(diff):
            ax.axvline(m, color="k", lw=0.4, alpha=0.4)
        ax.set_title(f"{fam}: Δ_m = cross-team JS − same-team JS (red = helps team discovery); "
                     f"grey lines = probes whose true reward differs by team", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(OUT / "delta_m.png", dpi=120)


if __name__ == "__main__":
    main()
