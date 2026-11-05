"""Reward-coupling probe vs policy probe on the B nopen pilot (team reward, 6 agents 3+3).  Read-only.

No agent is trained (weights hashed before/after).  The reward model of each agent is a table of means of
its own reward, so nothing is fitted by gradient descent either.  Team labels are used only to score.

Rollouts: frozen checkpoint, stochastic masked policy (as in training), its own training environment.

Signals compared (each gives an n x n matrix; AUROC = P(same-team pair more similar than cross-team pair))

  coupling_probe  (probe form, D71 (나))  Shared probe events, built from the map only: "a delivery happens at
                  goal cell g" for every goal cell, and "a return happens in the left / right half" (control).
                  Agent i estimates r_i(m) = mean of its own reward at steps where it saw event m done by
                  another agent inside its view (Chebyshev distance <= sensor range), minus its own baseline
                  reward.  Steps where i itself delivered or returned are skipped.  Pairs compare responses:
                  D_ij = mean_m |r_i(m) - r_j(m)| over probes both have seen >= MIN_OBS times.
  coupling_probe_global  the same with every event visible to every agent: an observability ceiling, not a
                  deployable signal.
  direct_coupling (evidence form, reference only; D48 reservation)  c_ij = mean reward of i at steps where it
                  saw agent j deliver or return in view, minus baseline.  No comparison of responses.
  policy_canonical  PGCT gate rule on the run's canonical bank (48 probes): zero hidden, unmasked softmax,
                  mean JS.
  policy_v2_gate  same rule on the E1-F v2 gate bank (48 probes: carrying, goals out of view).
  policy_visited  same rule on observations sampled from these rollouts (real states, all agents' states pooled).

Episode bootstrap (200 resamples) gives an interval for the coupling AUROCs.

    python output/reward_coupling_20261022/reward_coupling.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "latent_team"))
from examples import train_recurrent_ippo_consensus as T  # noqa: E402
from examples.diagnose_pgct_rollout_compatibility import model_hash  # noqa: E402
from reward_correlation_baseline import load_model, score  # noqa: E402

OUT = Path(__file__).resolve().parent
PLAN = ROOT / "runs/pilot_20261021_133652_b_e1f6_comm3_nopen"
RUNS = {f"B nopen s{s}": (PLAN / f"b_e1f6_comm3_nopen_ippo_s{s}", PLAN / f"canonical_bank_b_e1f6_comm3_nopen_s{s}.pt")
        for s in (1, 2)}
CKPTS = ["checkpoint_00750.pt", "final.pt"]
V2_GATE = ROOT / "output/env_view_20261015/probes_v2/probe_bank_v2_e1f_s1.pt"
EPISODES, HORIZON, SEED = 80, 500, 9100001
MIN_OBS, BOOT, VISITED = 3, 200, 256


def js_matrix(agents, x):
    """x [M, obs] -> n x n mean JS (PGCT gate rule: zero hidden, unmasked softmax)."""
    with torch.inference_mode():
        P = []
        for net in agents:
            h, _ = net.initial_hidden(len(x), torch.device("cpu"))
            logits, _ = net._actor_step(x, h)
            P.append(logits.softmax(-1).double().numpy())
    n = len(agents)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            p, q = P[i], P[j]
            m = 0.5 * (p + q)
            kl = lambda a, b: (a * (np.log(np.maximum(a, 1e-12)) - np.log(np.maximum(b, 1e-12)))).sum(-1)
            D[i, j] = float((0.5 * kl(p, m) + 0.5 * kl(q, m)).mean())
    return D


@torch.inference_mode()
def rollout(ck):
    agents, meta = load_model(ck)
    h0 = model_hash(agents)
    c = T._load_checkpoint(str(ck), torch.device("cpu"))
    cfg, kw = c["config"], dict(c["effective_env_kwargs"])
    sensor = int(kw.get("sensor_range", cfg["sensor_range"]))
    rng = np.random.default_rng(SEED)
    episodes, visited = [], []
    with T.make_env(cfg["env_id"], kw, cfg["observation_format"]) as env:
        base = env.unwrapped
        for e in range(EPISODES):
            obs, _ = env.reset(seed=SEED + e)
            n = len(base.agents)
            teams = [int(t) for t in np.asarray(base.agent_team_ids)]
            goals = sorted({tuple(int(v) for v in g) for g in base.goals})
            width = base.grid_size[1]
            hidden = [net.initial_hidden(1, torch.device("cpu"))[0] for net in agents]
            steps = []
            for t in range(HORIZON):
                x = np.asarray(T.flatten_multi_agent_obs(env, obs), dtype=np.float32)
                if rng.random() < 0.02:
                    visited.append(x[rng.integers(n)])
                masks = np.asarray(T.local_rware_action_masks(env, n, 5, enabled=True), dtype=np.float32)
                acts = []
                for i, net in enumerate(agents):
                    logits, hidden[i] = net._actor_step(torch.as_tensor(x[i]).unsqueeze(0), hidden[i])
                    p = net._masked_logits(logits, torch.as_tensor(masks[i])).softmax(-1).numpy()[0].astype(np.float64)
                    acts.append(int(rng.choice(5, p=p / p.sum())))
                obs, r, term, trunc, info = env.step(acts)
                pos = [(int(a.x), int(a.y)) for a in base.agents]
                events = []  # (kind, location, carrier index); no team fields are read
                for ev in info.get("delivery_events", []):
                    events.append(("delivery", tuple(int(v) for v in ev["goal"]), int(ev["carrier_id"]) - 1))
                for ev in info.get("return_events", []):
                    if ev.get("home") is not None:
                        events.append(("return", tuple(int(v) for v in ev["home"]), int(ev["carrier_id"]) - 1))
                steps.append(dict(r=np.asarray(r, dtype=np.float64), pos=pos, events=events))
                if bool(np.any(term)) or bool(np.any(trunc)):
                    break
            episodes.append(dict(steps=steps, teams=teams, goals=goals, width=width))
    assert h0 == model_hash(agents)
    return agents, episodes, np.stack(visited), sensor, int(meta["env_steps"])


def probe_key(kind, loc, goals, width):
    if kind == "delivery":
        return f"delivery@{loc}" if loc in goals else None
    return "return@left" if loc[0] < width / 2 else "return@right"


def coupling(episodes, n, sensor, global_view=False):
    """Returns (probe responses n x M with counts, direct coupling n x n, probe names)."""
    names = sorted({f"delivery@{g}" for ep in episodes for g in ep["goals"]}) + ["return@left", "return@right"]
    idx = {k: m for m, k in enumerate(names)}
    S, C = np.zeros((n, len(names))), np.zeros((n, len(names)))
    DS, DC = np.zeros((n, n)), np.zeros((n, n))
    base_sum, base_cnt = np.zeros(n), np.zeros(n)
    for ep in episodes:
        for st in ep["steps"]:
            own = {c for _, _, c in st["events"]}
            seen_any = np.zeros(n, bool)
            for kind, loc, carrier in st["events"]:
                key = probe_key(kind, loc, ep["goals"], ep["width"])
                for i in range(n):
                    if i == carrier or i in own:
                        continue
                    px, py = st["pos"][i]
                    if not global_view and max(abs(px - loc[0]), abs(py - loc[1])) > sensor:
                        continue
                    seen_any[i] = True
                    if key is not None:
                        S[i, idx[key]] += st["r"][i]; C[i, idx[key]] += 1
                    if 0 <= carrier < n:
                        DS[i, carrier] += st["r"][i]; DC[i, carrier] += 1
            for i in range(n):
                if not seen_any[i] and i not in own:
                    base_sum[i] += st["r"][i]; base_cnt[i] += 1
    b = base_sum / np.maximum(base_cnt, 1)
    R = np.where(C > 0, S / np.maximum(C, 1) - b[:, None], np.nan)
    direct = np.where(DC > 0, DS / np.maximum(DC, 1) - b[:, None], np.nan)
    return R, C, direct, DC, names


def probe_distance(R, C, cols=None):
    n = len(R)
    cols = range(R.shape[1]) if cols is None else cols
    D = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            ok = [m for m in cols if C[i, m] >= MIN_OBS and C[j, m] >= MIN_OBS]
            if i != j and ok:
                D[i, j] = float(np.mean([abs(R[i, m] - R[j, m]) for m in ok]))
    return D


def sym(M):
    """direct coupling is directed; score pairs by the mean of the defined directions."""
    out = np.full_like(M, np.nan)
    for i in range(len(M)):
        for j in range(len(M)):
            v = [x for x in (M[i, j], M[j, i]) if np.isfinite(x)]
            out[i, j] = float(np.mean(v)) if v and i != j else np.nan
    return out


def coupling_scores(episodes, n, sensor, teams):
    out = {}
    for label, glob in (("coupling_probe", False), ("coupling_probe_global", True)):
        R, C, direct, DC, names = coupling(episodes, n, sensor, glob)
        delivery_cols = [m for m, k in enumerate(names) if k.startswith("delivery")]
        return_cols = [m for m, k in enumerate(names) if k.startswith("return")]
        out[label] = score(-probe_distance(R, C, delivery_cols), teams)
        out[label + "_returns_only"] = score(-probe_distance(R, C, return_cols), teams)
        out["direct_coupling" + ("_global" if glob else "")] = score(sym(np.where(DC >= MIN_OBS, direct, np.nan)), teams)
        if not glob:
            detail = dict(probes=names, responses=np.round(R, 3).tolist(), counts=C.astype(int).tolist(),
                          direct=np.round(direct, 3).tolist(), direct_counts=DC.astype(int).tolist())
    return out, detail


def bootstrap(episodes, n, sensor, teams, rng):
    keys = ["coupling_probe", "coupling_probe_global", "direct_coupling"]
    vals = {k: [] for k in keys}
    for _ in range(BOOT):
        sample = [episodes[k] for k in rng.integers(len(episodes), size=len(episodes))]
        s, _ = coupling_scores(sample, n, sensor, teams)
        for k in keys:
            if s[k]["auroc"] is not None:
                vals[k].append(s[k]["auroc"])
    return {k: dict(defined=len(v), lo=round(float(np.percentile(v, 2.5)), 3) if v else None,
                    hi=round(float(np.percentile(v, 97.5)), 3) if v else None) for k, v in vals.items()}


def main():
    torch.set_num_threads(4)
    v2 = torch.load(V2_GATE, map_location="cpu", weights_only=False)["sequences"][:, 0].float()
    results = {}
    for name, (run, bank_path) in RUNS.items():
        canon = torch.load(bank_path, map_location="cpu", weights_only=False)["sequences"][:, 0].float()
        for ck in CKPTS:
            agents, episodes, visited, sensor, steps = rollout(run / ck)
            n, teams = len(agents), episodes[0]["teams"]
            assert all(ep["teams"] == teams for ep in episodes)
            row = dict(env_steps=steps, teams=teams, episodes=len(episodes),
                       deliveries=sum(1 for ep in episodes for st in ep["steps"] for ev in st["events"] if ev[0] == "delivery"),
                       returns=sum(1 for ep in episodes for st in ep["steps"] for ev in st["events"] if ev[0] == "return"))
            sc, detail = coupling_scores(episodes, n, sensor, teams)
            row["scores"] = sc
            row["bootstrap"] = bootstrap(episodes, n, sensor, teams, np.random.default_rng(SEED))
            for label, x in (("policy_canonical", canon), ("policy_v2_gate", v2),
                             ("policy_visited", torch.as_tensor(visited[:VISITED]))):
                row["scores"][label] = score(-js_matrix(agents, x), teams)
            row["detail"] = detail
            results[f"{name} {ck}"] = row
            line = "  ".join(f"{k}={'-' if v['auroc'] is None else round(v['auroc'], 2)}({v['defined_pairs']})"
                             for k, v in row["scores"].items())
            print(f"== {name} {ck} ({steps / 1e6:.2f}M) deliveries={row['deliveries']} returns={row['returns']}", flush=True)
            print("  ", line, flush=True)
            print("   bootstrap 95%:", row["bootstrap"], flush=True)
            print("   seen counts (agent x probe):", detail["probes"], detail["counts"], flush=True)
            print("   responses:", detail["responses"], flush=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
