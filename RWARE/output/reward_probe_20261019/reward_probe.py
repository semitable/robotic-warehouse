"""Reward probe, offline: can agents' own learned reward predictors tell teammates apart?

No agent is trained.  Policies only generate data (frozen checkpoints, weights hashed, or a uniform random
policy over valid actions).  Per condition:

  1. rollouts   stochastic masked policy, record every agent's (flat obs, action, individual reward)
  2. predictor  per agent, a separate small MLP r_hat_i(obs, action) trained on that agent's own data only
                (several init seeds); it never touches the actor/critic
  3. probe      shared queries = v2 gate states (carrying own requested shelf, no goal in view) x 5 actions;
                D_ij = mean |r_hat_i - r_hat_j| over the queries (also restricted to FORWARD)
  4. oracle     counterfactual upper bound: the real env places an agent of team t in each probe's physical
                state (ProbeRealiser) and steps the action; D_oracle compares the teams' true rewards
  5. baseline   policy probe (actor JS on the same states) for the checkpoint policies

Team labels only score: AUROC (same team closer = 1), nearest-is-teammate, softmax teammate share.

    python output/reward_probe_20261019/reward_probe.py [--episodes 30] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples import train_recurrent_ippo_consensus as T  # noqa: E402
from examples.diagnose_pgct_rollout_compatibility import load_model, model_hash  # noqa: E402
from latent_team import probes_v2 as P2  # noqa: E402

OUT = Path(__file__).resolve().parent
E1F_V2 = ROOT / "runs/e1f_20261018_124343_main_v2/probe_bank_v2_e1f_s1.pt"
E1_V2 = ROOT / "output/env_view_20261015/probes_v2/probe_bank_v2_e1_s1.pt"
E1F_RUN = ROOT / "runs/e1f_20261018_124343_main_canonical/e1f_ippo_s1"
E1_RUN = ROOT / "runs/scratch_20261015_093349_stage2/e1_ippo_s1"
STAGE = ROOT / "runs/pgct09700_physical_mixing_20260929_155338_079437"
CONDITIONS = [  # (name, checkpoint giving env kwargs, policy checkpoint or None = random, bank)
    ("E1-F random", E1F_RUN / "checkpoint_00250.pt", None, E1F_V2),
    ("E1-F IPPO 0.26M", E1F_RUN / "checkpoint_00250.pt", E1F_RUN / "checkpoint_00250.pt", E1F_V2),
    ("E1-F IPPO 0.77M", E1F_RUN / "checkpoint_00750.pt", E1F_RUN / "checkpoint_00750.pt", E1F_V2),
    ("E1-F IPPO 1.54M", E1F_RUN / "checkpoint_01500.pt", E1F_RUN / "checkpoint_01500.pt", E1F_V2),
    ("E1 random", E1_RUN / "checkpoint_00250.pt", None, E1_V2),
    ("E1 IPPO 1.54M", E1_RUN / "checkpoint_01500.pt", E1_RUN / "checkpoint_01500.pt", E1_V2),
    ("E1 IPPO 6.40M", E1_RUN / "checkpoint_06250.pt", E1_RUN / "checkpoint_06250.pt", E1_V2),
    # added 2026-10-19 (T069): well-trained curriculum models, each in its own env (E1 map, home .75 / .5)
    ("Stage03 3.0M", STAGE / "stage03_4ag_physical_home0.750/final.pt",
     STAGE / "stage03_4ag_physical_home0.750/final.pt", E1_V2),
    ("Stage04 PGCT 4.0M", STAGE / "stage04_4ag_physical_home0.500/final.pt",
     STAGE / "stage04_4ag_physical_home0.500/final.pt", E1_V2),
]
PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]
N_ACT = 5
FORWARD = 1


def env_setup(ck):
    agents, meta = load_model(ck)
    cfg = meta["config"]
    env_kwargs = dict(T._load_checkpoint(str(ck), torch.device("cpu"))["effective_env_kwargs"])
    return agents, cfg, env_kwargs


@torch.inference_mode()
def rollouts(cfg, env_kwargs, agents, episodes, horizon, seed):
    rng = np.random.default_rng(seed)
    obs_l, act_l, rew_l = [], [], []
    device = torch.device("cpu")
    with T.make_env(cfg["env_id"], env_kwargs, cfg["observation_format"]) as env:
        teams = None
        for e in range(episodes):
            obs, _ = env.reset(seed=seed + e)
            teams = [int(t) for t in np.asarray(env.unwrapped.agent_team_ids)]
            hidden = [net.initial_hidden(1, device)[0] for net in agents] if agents else None
            for _ in range(horizon):
                x = np.asarray(T.flatten_multi_agent_obs(env, obs), dtype=np.float32)
                masks = np.asarray(T.local_rware_action_masks(env, 4, N_ACT, enabled=True), dtype=np.float32)
                actions = []
                for i in range(4):
                    if agents is None:
                        p = masks[i] / masks[i].sum()
                    else:
                        logits, hidden[i] = agents[i]._actor_step(torch.as_tensor(x[i]).unsqueeze(0), hidden[i])
                        p = agents[i]._masked_logits(logits, torch.as_tensor(masks[i])).softmax(-1).numpy()[0]
                        p = p.astype(np.float64)
                        p = p / p.sum()
                    actions.append(int(rng.choice(N_ACT, p=p)))
                obs, rewards, terminated, truncated, _ = env.step(actions)
                obs_l.append(x.astype(np.float16))
                act_l.append(actions)
                rew_l.append(np.asarray(rewards, dtype=np.float32))
                if bool(np.any(terminated)) or bool(np.any(truncated)):
                    break
    return np.stack(obs_l), np.asarray(act_l), np.stack(rew_l), teams


class RewardNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim + N_ACT, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, 1))

    def forward(self, obs, act):
        return self.net(torch.cat([obs, nn.functional.one_hot(act, N_ACT).float()], -1)).squeeze(-1)


def train_predictor(obs, act, rew, seed, epochs, scale=10.0):
    torch.manual_seed(seed)
    n = len(obs)
    idx = np.random.default_rng(seed).permutation(n)
    val, tr = idx[: n // 10], idx[n // 10:]
    X = torch.as_tensor(obs.astype(np.float32)); A = torch.as_tensor(act).long(); Y = torch.as_tensor(rew) * scale
    net = RewardNet(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(epochs):
        perm = torch.as_tensor(np.random.default_rng(seed + _).permutation(tr))
        for b in range(0, len(perm), 256):
            k = perm[b:b + 256]
            loss = nn.functional.mse_loss(net(X[k], A[k]), Y[k])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pv = net(X[val], A[val]); yv = Y[val]
        r2 = float(1 - ((pv - yv) ** 2).mean() / max(float(yv.var()), 1e-12))
        rare = yv.abs() >= 0.1 * scale  # pickups (+0.15), wrong pickups (-0.05 is below), deliveries/returns
        rare_mae = float((pv[rare] - yv[rare]).abs().mean() / scale) if bool(rare.any()) else float("nan")
    net.eval()
    return net, r2, scale, rare_mae, int(rare.sum())


def query_matrix(bank_obs):
    q_obs = bank_obs.repeat_interleave(N_ACT, 0)
    q_act = torch.arange(N_ACT).repeat(len(bank_obs))
    return q_obs, q_act


def scores(D, teams, distance=True):
    S = -np.asarray(D) if distance else np.asarray(D)
    mate = [next(j for j in range(4) if j != i and teams[j] == teams[i]) for i in range(4)]
    same = [S[i, j] for i, j in PAIRS if teams[i] == teams[j]]
    cross = [S[i, j] for i, j in PAIRS if teams[i] != teams[j]]
    auc = float(np.mean([1.0 if s > c else 0.5 if s == c else 0.0 for s in same for c in cross]))
    nearest = int(sum(max((j for j in range(4) if j != i), key=lambda j: S[i, j]) == mate[i] for i in range(4)))
    off = np.abs(S[~np.eye(4, dtype=bool)])
    temp = max(float(np.median(off)), 1e-9)
    W = np.exp(S / temp) * ~np.eye(4, dtype=bool)
    share = float(np.mean([W[i, mate[i]] / W[i].sum() for i in range(4)]))
    return dict(auroc=auc, nearest_is_mate=nearest, softmax_mate_share=share,
                same_mean=float(-np.mean(same)), cross_mean=float(-np.mean(cross)))


def pairwise(vectors):
    D = np.zeros((4, 4))
    for i, j in PAIRS:
        D[i, j] = D[j, i] = float(np.mean(np.abs(vectors[i] - vectors[j])))
    return D


def oracle(cfg, env_kwargs, scenarios, teams_agents):
    """True reward of an agent of team t taking each action in each probe's physical state."""
    real = P2.ProbeRealiser(cfg["env_id"], env_kwargs, cfg["observation_format"], seed=1)
    names = {f.name for f in fields(P2.Scenario)}
    try:
        R = np.zeros((2, len(scenarios) * N_ACT))
        for t in range(2):
            k = 0
            for sc in scenarios:
                for a in range(N_ACT):
                    s = P2.Scenario(**{**{n: sc[n] for n in names if n in sc}, "team": t})
                    real.realise(s)
                    _, rewards, *_ = real.env.step([a, 0, 0, 0])
                    R[t, k] = float(np.asarray(rewards)[0]); k += 1
    finally:
        real.close()
    vec = [R[teams_agents[i]] for i in range(4)]
    return R, vec


@torch.inference_mode()
def policy_js(agents, bank_obs):
    P = []
    for net in agents:
        h, _ = net.initial_hidden(len(bank_obs), torch.device("cpu"))
        logits, _ = net._actor_step(bank_obs, h)
        P.append(logits.softmax(-1).numpy().astype(np.float64))
    per = {}
    for i, j in PAIRS:
        p, q = P[i], P[j]; m = (p + q) / 2
        kl = lambda a, b: (a * (np.log(np.maximum(a, 1e-12)) - np.log(np.maximum(b, 1e-12)))).sum(-1)
        per[(i, j)] = 0.5 * kl(p, m) + 0.5 * kl(q, m)  # per probe state
    return per


def weighted_matrix(per_pair, w):
    D = np.zeros((4, 4))
    for (i, j), v in per_pair.items():
        D[i, j] = D[j, i] = float(np.sum(w * v) / max(np.sum(w), 1e-12))
    return D


def hist_js(rew, teams):
    """Reward-history baseline: distribution of each agent's received reward values (no predictor)."""
    vals = np.round(rew, 3)
    bins = np.unique(vals)
    H = np.stack([np.array([(vals[:, i] == b).sum() for b in bins], float) + 1e-3 for i in range(4)])
    H /= H.sum(1, keepdims=True)
    D = np.zeros((4, 4))
    for i, j in PAIRS:
        m = (H[i] + H[j]) / 2
        D[i, j] = D[j, i] = float(0.5 * (H[i] * np.log(H[i] / m)).sum() + 0.5 * (H[j] * np.log(H[j] / m)).sum())
    return D


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=3, help="Predictor init seeds per agent.")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", nargs="+", help="Condition names to run.")
    ap.add_argument("--results-dir", help="Subfolder for results (default: results).")
    args = ap.parse_args()
    torch.set_num_threads(2)  # leave CPU for the running E1-F jobs
    if args.smoke:
        args.episodes, args.horizon, args.seeds, args.epochs = 2, 60, 1, 2
    out = OUT / ("smoke" if args.smoke else (args.results_dir or "results"))
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, env_ck, pol_ck, bank_path in CONDITIONS:
        if args.only and name not in args.only:
            continue
        t0 = time.time()
        _, cfg, env_kwargs = env_setup(env_ck)
        agents, before = None, None
        if pol_ck is not None:
            agents, _ = load_model(pol_ck)
            before = model_hash(agents)
        bank = torch.load(bank_path, map_location="cpu", weights_only=False)
        bank_obs = bank["sequences"][:, 0].float()
        obs, act, rew, teams = rollouts(cfg, env_kwargs, agents, args.episodes, args.horizon, seed=7100001)
        if agents is not None and before != model_hash(agents):
            raise RuntimeError("policy weights changed")
        q_obs, q_act = query_matrix(bank_obs)
        fwd = (q_act == FORWARD).numpy()
        preds, r2s, rare_maes = [], [], []
        for i in range(4):
            per_seed = []
            for s in range(args.seeds):
                net, r2, scale, rare_mae, n_rare = train_predictor(obs[:, i], act[:, i], rew[:, i], seed=100 * i + s,
                                                                   epochs=args.epochs)
                with torch.no_grad():
                    per_seed.append(net(q_obs, q_act).numpy() / scale)
                r2s.append(r2); rare_maes.append(rare_mae)
            preds.append(np.stack(per_seed))  # [seed, query]
        mean_pred = [p.mean(0) for p in preds]
        # predictor noise floor: same agent, different init seeds
        noise = float(np.mean([np.mean(np.abs(p[a] - p[b])) for p in preds for a in range(len(p))
                               for b in range(a + 1, len(p))])) if args.seeds > 1 else float("nan")
        R_or, vec_or = oracle(cfg, env_kwargs, bank["gate_scenarios"], teams)
        # --- probe weighting (equal votes vs reward-informed votes) ---
        P4 = np.stack(mean_pred)                                        # [agent, query]
        within = np.stack([p.var(0) for p in preds]).mean(0) if args.seeds > 1 else np.zeros(P4.shape[1])
        weights = {"equal": np.ones(P4.shape[1]),
                   "reward_magnitude": np.abs(P4).mean(0),              # user's idea, taken literally
                   "action_sensitivity": np.repeat(np.ptp(P4.reshape(4, -1, N_ACT), axis=2).mean(0), N_ACT),
                   "between/within (SNR)": P4.var(0) / (within + 1e-8),  # unsupervised discriminability
                   "oracle |r_red - r_blue| (labels)": np.abs(R_or[0] - R_or[1])}
        per_pair = {(i, j): np.abs(P4[i] - P4[j]) for i, j in PAIRS}
        weighted = {k: scores(weighted_matrix(per_pair, w), teams) for k, w in weights.items()}
        # share of each weighting that falls on queries where the true team rewards differ
        disc = np.abs(R_or[0] - R_or[1]) > 1e-9
        weight_on_disc = {k: float(w[disc].sum() / max(w.sum(), 1e-12)) for k, w in weights.items()}
        # in-distribution check: samples where the agent carries its own requested shelf (not returning)
        carry = (obs[..., P2.EGO["self_carrying_requested_shelf"]] > 0) & ~(obs[..., P2.EGO["self_return_phase"]] > 0)
        coverage = carry.sum(0).tolist()
        row = dict(policy=str(pol_ck.relative_to(ROOT)) if pol_ck else "uniform random over valid actions",
                   samples_per_agent=int(len(obs)), teams=teams,
                   reward_stats=dict(mean=rew.mean(0).tolist(), nonzero=(rew != 0).mean(0).tolist(),
                                     pickups=(rew >= 0.1).sum(0).tolist()),
                   predictor_val_r2=float(np.mean(r2s)), predictor_seed_noise=noise,
                   reward_probe=dict(matrix=pairwise(mean_pred).round(5).tolist(), **scores(pairwise(mean_pred), teams)),
                   reward_probe_forward=scores(pairwise([p[fwd] for p in mean_pred]), teams),
                   oracle=dict(team_reward_diff=float(np.mean(np.abs(R_or[0] - R_or[1]))),
                               queries_differing=int(np.sum(np.abs(R_or[0] - R_or[1]) > 1e-9)),
                               n_queries=int(R_or.shape[1]), **scores(pairwise(vec_or), teams)),
                   seconds=round(time.time() - t0, 1),
                   predictor_rare_event_mae=float(np.nanmean(rare_maes)) if not np.all(np.isnan(rare_maes)) else None,
                   carrying_samples_per_agent=coverage,
                   weighted_reward_probe=weighted, weight_share_on_discriminative=weight_on_disc,
                   reward_history=scores(hist_js(rew, teams), teams))
        if agents is not None:
            per_state = policy_js(agents, bank_obs)
            sens = np.ptp(P4.reshape(4, -1, N_ACT), axis=2).mean(0)        # reward at stake per probe state
            row["policy_probe"] = scores(weighted_matrix(per_state, np.ones(len(bank_obs))), teams)
            row["policy_probe_reward_weighted"] = scores(weighted_matrix(per_state, sens), teams)
        np.savez_compressed(out / f"{name.replace(' ', '_')}.npz", preds=np.stack(preds), oracle=R_or,
                            q_act=q_act.numpy(), teams=np.asarray(teams), rew_hist=np.round(rew, 3))
        # does each predictor agree with its own team's true reward more than with the other team's?
        row["predictor_vs_oracle"] = [dict(agent=i + 1, team=teams[i],
                                           corr_own=float(np.corrcoef(mean_pred[i], R_or[teams[i]])[0, 1]),
                                           corr_other=float(np.corrcoef(mean_pred[i], R_or[1 - teams[i]])[0, 1]))
                                      for i in range(4)]
        results[name] = row
        rp, orc = row["reward_probe"], row["oracle"]
        pp = row.get("policy_probe", {})
        print(f"{name:18s} n={len(obs)} R2={row['predictor_val_r2']:.2f} noise={noise:.4f} | reward probe AUROC="
              f"{rp['auroc']:.2f} nearest={rp['nearest_is_mate']}/4 share={rp['softmax_mate_share']:.2f} "
              f"same/cross={rp['same_mean']:.4f}/{rp['cross_mean']:.4f} | fwd AUROC={row['reward_probe_forward']['auroc']:.2f} "
              f"| oracle AUROC={orc['auroc']:.2f} diff_q={orc['queries_differing']}/{orc['n_queries']} "
              f"| policy AUROC={pp.get('auroc', float('nan')):.2f} | corr own/other="
              f"{np.mean([r['corr_own'] for r in row['predictor_vs_oracle']]):.2f}/"
              f"{np.mean([r['corr_other'] for r in row['predictor_vs_oracle']]):.2f} ({row['seconds']}s)", flush=True)
        print("    weighted AUROC: " + ", ".join(f"{k}={v['auroc']:.2f}" for k, v in weighted.items())
              + " | weight on discriminative queries: " + ", ".join(f"{k}={v:.2f}" for k, v in weight_on_disc.items())
              + f" | reward history AUROC={row['reward_history']['auroc']:.2f}"
              + (f" | policy reward-weighted AUROC={row['policy_probe_reward_weighted']['auroc']:.2f}" if agents else "")
              + f" | rare MAE={row['predictor_rare_event_mae']} carrying samples={coverage}", flush=True)
        (out / "results.json").write_text(json.dumps(dict(args=vars(args), results=results), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
