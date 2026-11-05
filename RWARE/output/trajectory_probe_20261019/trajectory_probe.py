"""Trajectory (sequence) probes vs length-1 probes, offline.  No agent is trained; weights are hashed.

Shared probe trajectories (same input for every actor, label-free selection):
  source   rollouts of one fixed, well-trained policy (Stage04 PGCT) in the target environment
           (E1 or E1-F).  Its agents only generate realistic observation sequences.
  windows  length L (default 12) from one agent's own stream, balanced over the 4 source agents:
             carry_memory  carrying a requested shelf throughout; own goal in view during the first
                           3 steps and out of view during the last 4 (goal seen, then lost)
             carry         carrying a requested shelf throughout (any goal visibility)
             random        any window (control)

(a) policy probe   each actor reads the window with its hidden state carried (zero at window start);
                   D_ij = mean over windows and steps of JS(pi_i, pi_j)   [traj_all]
                   same, last half of the window only                     [traj_late]
                   length-1 baseline on the SAME frames: every frame alone, zero hidden [len1]
(b) reward probe   per agent, from that agent's own rollouts (the evaluated model's policy):
                   MLP  r(o_t, a_t)                       -> length-1 baseline [len1]
                   GRU  r(o_<=t, a_t), hidden carried     -> per-step [traj_step] and window return [traj_sum]
                   queried with the window's recorded actions.

Scores: AUROC (same team closer = 1), nearest-is-teammate.  Team labels only score.

    python output/trajectory_probe_20261019/trajectory_probe.py [--part policy|reward|both] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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
CUR = ROOT / "runs/pgct09700_physical_mixing_20260929_155338_079437"
SP = ROOT / "runs/sp_20261008_120638_983201"
SCR = ROOT / "runs/scratch_20261015_093349_stage2"
E1F = ROOT / "runs/e1f_20261018_124343_main_canonical"
E1F_V2 = ROOT / "runs/e1f_20261018_124343_main_v2"
SOURCE = CUR / "stage04_4ag_physical_home0.500/final.pt"
ENV_OF = {"E1": CUR / "stage04_4ag_physical_home0.500/final.pt", "E1-F": E1F / "e1f_ippo_s1/checkpoint_01500.pt"}
MODELS = {  # name: (env family, checkpoint)
    "Stage01": ("E1", CUR / "stage01_4ag_physical_home1.000/final.pt"),
    "Stage02": ("E1", CUR / "stage02_4ag_physical_home0.875/final.pt"),
    "Stage03": ("E1", CUR / "stage03_4ag_physical_home0.750/final.pt"),
    "Stage04 PGCT": ("E1", CUR / "stage04_4ag_physical_home0.500/final.pt"),
    "sp E1 IPPO": ("E1", SP / "e1_ippo_s1/final.pt"),
    "sp E1 PGCT": ("E1", SP / "e1_pgct_s1/final.pt"),
    "scratch E1 IPPO 6.4M": ("E1", SCR / "e1_ippo_s1/checkpoint_06250.pt"),
    "scratch E1 PGCT 6.4M": ("E1", SCR / "e1_pgct_s1/checkpoint_06250.pt"),
    "scratch E1 Oracle 6.4M": ("E1", SCR / "e1_oracle_s1/checkpoint_06250.pt"),
    "scratch E1 All 6.4M": ("E1", SCR / "e1_all_s1/checkpoint_06250.pt"),
    "E1-F IPPO 1.54M": ("E1-F", E1F / "e1f_ippo_s1/checkpoint_01500.pt"),
    "E1-F PGCT 1.54M": ("E1-F", E1F / "e1f_pgct_s1/checkpoint_01500.pt"),
    "E1-F Oracle 1.54M": ("E1-F", E1F / "e1f_oracle_s1/checkpoint_01500.pt"),
    "E1-F All 1.54M": ("E1-F", E1F / "e1f_all_s1/checkpoint_01500.pt"),
    "E1-F PGCT-v2 1.54M": ("E1-F", E1F_V2 / "e1f_pgct_s1/checkpoint_01500.pt"),
}
REWARD_MODELS = ["Stage03", "Stage04 PGCT", "scratch E1 IPPO 6.4M", "E1-F IPPO 1.54M", "E1-F PGCT 1.54M"]
PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]
N_ACT = 5
EGO_DIM = len(P2.SEMANTIC_EGO_FEATURES)
N_CH = len(P2.SEMANTIC_SPATIAL_CHANNELS)


def env_of(ck):
    agents, meta = load_model(ck)
    cfg = meta["config"]
    return cfg, dict(T._load_checkpoint(str(ck), torch.device("cpu"))["effective_env_kwargs"])


def teams_in(cfg, env_kwargs):
    with T.make_env(cfg["env_id"], env_kwargs, cfg["observation_format"]) as env:
        env.reset(seed=0)
        return [int(t) for t in np.asarray(env.unwrapped.agent_team_ids)]


@torch.inference_mode()
def rollouts(policy_ck, env_ck, episodes, horizon, seed):
    """Per agent: obs [E, H, D] float16, actions, rewards, carry flag, own-goal-visible flag."""
    agents, _ = load_model(policy_ck)
    h0 = model_hash(agents)
    cfg, env_kwargs = env_of(env_ck)
    rng = np.random.default_rng(seed)
    O, A, R = [], [], []
    with T.make_env(cfg["env_id"], env_kwargs, cfg["observation_format"]) as env:
        for e in range(episodes):
            obs, _ = env.reset(seed=seed + e)
            hidden = [net.initial_hidden(1, torch.device("cpu"))[0] for net in agents]
            eo, ea, er = [], [], []
            for _ in range(horizon):
                x = np.asarray(T.flatten_multi_agent_obs(env, obs), dtype=np.float32)
                masks = np.asarray(T.local_rware_action_masks(env, 4, N_ACT, enabled=True), dtype=np.float32)
                acts = []
                for i, net in enumerate(agents):
                    logits, hidden[i] = net._actor_step(torch.as_tensor(x[i]).unsqueeze(0), hidden[i])
                    p = net._masked_logits(logits, torch.as_tensor(masks[i])).softmax(-1).numpy()[0].astype(np.float64)
                    acts.append(int(rng.choice(N_ACT, p=p / p.sum())))
                obs, rewards, term, trunc, _ = env.step(acts)
                eo.append(x.astype(np.float16)); ea.append(acts); er.append(np.asarray(rewards, np.float32))
                if bool(np.any(term)) or bool(np.any(trunc)):
                    break
            O.append(np.stack(eo)); A.append(np.asarray(ea)); R.append(np.stack(er))
    assert h0 == model_hash(agents)
    O, A, R = np.stack(O), np.stack(A), np.stack(R)  # [E, H, 4, D] ...
    carry = (O[..., P2.EGO["self_carrying_requested_shelf"]] > 0) & ~(O[..., P2.EGO["self_return_phase"]] > 0)
    grid = O[..., EGO_DIM:].reshape(*O.shape[:-1], N_CH, -1)
    goal_vis = grid[..., P2.CH["relevant_goal"], :].astype(np.float32).max(-1) > 0
    return dict(obs=O, act=A, rew=R, carry=carry, goal_vis=goal_vis, teams=teams_in(cfg, env_kwargs))


def windows(src, L, per_type, seed):
    rng = np.random.default_rng(seed)
    E, H = src["act"].shape[:2]
    kinds = {"carry_memory": [], "carry": [], "random": []}
    for e in range(E):
        for a in range(4):
            c, g = src["carry"][e, :, a], src["goal_vis"][e, :, a]
            for t in range(0, H - L + 1):
                item = (e, a, t)
                kinds["random"].append(item)
                if c[t:t + L].all():
                    kinds["carry"].append(item)
                    if g[t:t + 3].any() and not g[t + L - 4:t + L].any():
                        kinds["carry_memory"].append(item)
    out = {}
    for k, items in kinds.items():
        by_agent = [[it for it in items if it[1] == a] for a in range(4)]
        pick = []
        for a in range(4):  # balanced over source agents (no team labels needed)
            if by_agent[a]:
                idx = rng.choice(len(by_agent[a]), size=min(per_type // 4, len(by_agent[a])), replace=False)
                pick += [by_agent[a][i] for i in idx]
        out[k] = pick
    return out


def window_arrays(src, items, L):
    obs = np.stack([src["obs"][e, t:t + L, a] for e, a, t in items]).astype(np.float32)  # [N, L, D]
    act = np.stack([src["act"][e, t:t + L, a] for e, a, t in items])
    return torch.as_tensor(obs), torch.as_tensor(act).long()


def js(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (np.log(np.maximum(a, 1e-12)) - np.log(np.maximum(b, 1e-12)))).sum(-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def score(D, teams):
    mate = [next(j for j in range(4) if j != i and teams[j] == teams[i]) for i in range(4)]
    same = [D[i, j] for i, j in PAIRS if teams[i] == teams[j]]
    cross = [D[i, j] for i, j in PAIRS if teams[i] != teams[j]]
    auc = float(np.mean([1.0 if s < c else 0.5 if s == c else 0.0 for s in same for c in cross]))
    nearest = int(sum(min((j for j in range(4) if j != i), key=lambda j: D[i, j]) == mate[i] for i in range(4)))
    return dict(auroc=auc, nearest=nearest, same=float(np.mean(same)), cross=float(np.mean(cross)))


def mat(per_pair):
    D = np.zeros((4, 4))
    for (i, j), v in per_pair.items():
        D[i, j] = D[j, i] = float(v)
    return D


@torch.inference_mode()
def policy_probe(agents, obs):
    """obs [N, L, D] -> probabilities with hidden carried [agent, N, L, A] and frame-alone [agent, N, L, A]."""
    N, L, D = obs.shape
    seq, alone = [], []
    for net in agents:
        h, _ = net.initial_hidden(N, torch.device("cpu"))
        steps = []
        for t in range(L):
            logits, h = net._actor_step(obs[:, t], h)
            steps.append(logits.softmax(-1).numpy())
        seq.append(np.stack(steps, 1))
        h1, _ = net.initial_hidden(N * L, torch.device("cpu"))
        logits, _ = net._actor_step(obs.reshape(N * L, D), h1)
        alone.append(logits.softmax(-1).numpy().reshape(N, L, -1))
    return np.stack(seq).astype(np.float64), np.stack(alone).astype(np.float64)


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d + N_ACT, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, o, a):
        return self.f(torch.cat([o, nn.functional.one_hot(a, N_ACT).float()], -1)).squeeze(-1)


class GRUR(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d + N_ACT, 256), nn.ReLU())
        self.gru = nn.GRU(256, 128, batch_first=True)
        self.out = nn.Linear(128, 1)

    def forward(self, o, a):  # o [B, L, D], a [B, L]
        z = self.inp(torch.cat([o, nn.functional.one_hot(a, N_ACT).float()], -1))
        y, _ = self.gru(z)
        return self.out(y).squeeze(-1)


def train_reward(src, agent, seed, epochs, chunk=16, scale=10.0):
    """MLP and GRU reward predictors from one agent's own stream; returns nets and validation R2."""
    torch.manual_seed(seed)
    O = torch.as_tensor(src["obs"][:, :, agent].astype(np.float32))  # [E, H, D]
    A = torch.as_tensor(src["act"][:, :, agent]).long()
    Y = torch.as_tensor(src["rew"][:, :, agent]) * scale
    E, H = A.shape
    n_val = max(1, E // 10)
    tr, va = slice(n_val, E), slice(0, n_val)
    mlp, gru = MLP(O.shape[-1]), GRUR(O.shape[-1])
    om, og = torch.optim.Adam(mlp.parameters(), 1e-3), torch.optim.Adam(gru.parameters(), 1e-3)
    Xo, Xa, Xy = O[tr].reshape(-1, O.shape[-1]), A[tr].reshape(-1), Y[tr].reshape(-1)
    Co = O[tr, :H // chunk * chunk].reshape(-1, chunk, O.shape[-1])
    Ca = A[tr, :H // chunk * chunk].reshape(-1, chunk)
    Cy = Y[tr, :H // chunk * chunk].reshape(-1, chunk)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(Xo), generator=g)
        for b in range(0, len(perm), 256):
            k = perm[b:b + 256]
            loss = nn.functional.mse_loss(mlp(Xo[k], Xa[k]), Xy[k]); om.zero_grad(); loss.backward(); om.step()
        perm = torch.randperm(len(Co), generator=g)
        for b in range(0, len(perm), 32):
            k = perm[b:b + 32]
            loss = nn.functional.mse_loss(gru(Co[k], Ca[k]), Cy[k]); og.zero_grad(); loss.backward(); og.step()
    with torch.no_grad():
        yv = Y[va].reshape(-1)
        pm = mlp(O[va].reshape(-1, O.shape[-1]), A[va].reshape(-1))
        vo = O[va, :H // chunk * chunk].reshape(-1, chunk, O.shape[-1])
        pg = gru(vo, A[va, :H // chunk * chunk].reshape(-1, chunk)).reshape(-1)
        yg = Y[va, :H // chunk * chunk].reshape(-1)
        r2 = lambda p, y: float(1 - ((p - y) ** 2).mean() / max(float(y.var()), 1e-12))
        return mlp.eval(), gru.eval(), r2(pm, yv), r2(pg, yg), scale


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["policy", "reward", "both"], default="both")
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--per-type", type=int, default=64)
    ap.add_argument("--source-episodes", type=int, default=40)
    ap.add_argument("--reward-episodes", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(2)
    if args.smoke:
        args.source_episodes, args.reward_episodes, args.epochs, args.per_type = 2, 2, 1, 8
    out = OUT / ("smoke" if args.smoke else "results")
    out.mkdir(parents=True, exist_ok=True)
    horizon = 120 if args.smoke else 500
    # shared probe windows per environment family (one fixed source policy)
    banks = {}
    for fam, env_ck in ENV_OF.items():
        t0 = time.time()
        src = rollouts(SOURCE, env_ck, args.source_episodes, horizon, seed=8100001)
        w = windows(src, args.L, args.per_type, seed=8100001)
        banks[fam] = {k: (window_arrays(src, items, args.L) if items else None) for k, items in w.items()}
        print(f"[bank {fam}] windows: " + ", ".join(f"{k}={len(v)}" for k, v in w.items())
              + f" ({time.time() - t0:.0f}s)", flush=True)
    results = {"args": vars(args), "bank_sizes": {f: {k: (0 if v is None else int(v[0].shape[0]))
                                                       for k, v in b.items()} for f, b in banks.items()},
               "models": {}}
    for name, (fam, ck) in MODELS.items():
        if args.smoke and name not in ("Stage04 PGCT", "E1-F IPPO 1.54M"):
            continue
        cfg, env_kwargs = env_of(ck)
        teams = teams_in(cfg, env_kwargs)
        row = dict(family=fam, checkpoint=str(ck.relative_to(ROOT)), teams=teams, policy={}, reward={})
        if args.part in ("policy", "both"):
            agents, _ = load_model(ck)
            h0 = model_hash(agents)
            for kind, arr in banks[fam].items():
                if arr is None:
                    continue
                seq, alone = policy_probe(agents, arr[0])
                L = seq.shape[2]
                per = {"len1": {}, "traj_all": {}, "traj_late": {}}
                for i, j in PAIRS:
                    per["len1"][(i, j)] = js(alone[i], alone[j]).mean()
                    d = js(seq[i], seq[j])  # [N, L]
                    per["traj_all"][(i, j)] = d.mean()
                    per["traj_late"][(i, j)] = d[:, L // 2:].mean()
                row["policy"][kind] = {m: score(mat(v), teams) for m, v in per.items()}
            assert h0 == model_hash(agents)
            line = " | ".join(f"{k}: " + " ".join(f"{m}={v['auroc']:.2f}" for m, v in r.items())
                              for k, r in row["policy"].items())
            print(f"[policy] {name:24s} {line}", flush=True)
        if args.part in ("reward", "both") and (name in REWARD_MODELS or args.smoke):
            t0 = time.time()
            own = rollouts(ck, ck, args.reward_episodes, horizon, seed=8200001)
            nets, r2m, r2g = [], [], []
            for a in range(4):
                mlp, gru, rm, rg, scale = train_reward(own, a, seed=300 + a, epochs=args.epochs)
                nets.append((mlp, gru)); r2m.append(rm); r2g.append(rg)
            row["reward_r2"] = dict(mlp=float(np.mean(r2m)), gru=float(np.mean(r2g)))
            for kind, arr in banks[fam].items():
                if arr is None:
                    continue
                obs, act = arr
                N, L, D = obs.shape
                with torch.no_grad():
                    m1 = np.stack([n[0](obs.reshape(N * L, D), act.reshape(-1)).numpy().reshape(N, L) / scale for n in nets])
                    g1 = np.stack([n[1](obs, act).numpy() / scale for n in nets])
                per = {"len1": {}, "traj_step": {}, "traj_sum": {}}
                for i, j in PAIRS:
                    per["len1"][(i, j)] = np.abs(m1[i] - m1[j]).mean()
                    per["traj_step"][(i, j)] = np.abs(g1[i] - g1[j]).mean()
                    per["traj_sum"][(i, j)] = np.abs(g1[i].sum(1) - g1[j].sum(1)).mean()
                row["reward"][kind] = {m: score(mat(v), teams) for m, v in per.items()}
            line = " | ".join(f"{k}: " + " ".join(f"{m}={v['auroc']:.2f}" for m, v in r.items())
                              for k, r in row["reward"].items())
            print(f"[reward] {name:24s} R2 mlp={row['reward_r2']['mlp']:.2f} gru={row['reward_r2']['gru']:.2f} | "
                  f"{line} ({time.time() - t0:.0f}s)", flush=True)
        results["models"][name] = row
        (out / f"results_{args.part}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    summarise(results, out / f"summary_{args.part}.md")


def summarise(results, path):
    lines = ["# Trajectory vs length-1 probes (AUROC; same team closer = 1, chance .5; seed 1)", ""]
    for part in ("policy", "reward"):
        for fam_group, pred in (("E1 curriculum (Stage01-04, sp E1)", lambda n: n.startswith(("Stage", "sp "))),
                                ("E1 from scratch (6.4M)", lambda n: n.startswith("scratch")),
                                ("E1-F checkpoints (1.54M)", lambda n: n.startswith("E1-F"))):
            rows = {n: r[part] for n, r in results["models"].items() if pred(n) and r.get(part)}
            if not rows:
                continue
            lines += [f"## {part} — {fam_group}", ""]
            kinds = sorted({k for r in rows.values() for k in r})
            metrics = sorted({m for r in rows.values() for k in r.values() for m in k})
            lines.append("| window type | " + " | ".join(metrics) + " |")
            lines.append("|---|" + "---|" * len(metrics))
            for k in kinds:
                vals = [np.mean([r[k][m]["auroc"] for r in rows.values() if k in r]) for m in metrics]
                lines.append(f"| {k} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
