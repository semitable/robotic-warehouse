"""Do pre-E1 trained models connect teammates through the probes?  And would a soft gate help?

Frozen checkpoints only (weights hashed before/after).  PGCT gate rule: identical inputs, zero hidden,
unmasked softmax, natural-log JS averaged over probes.  Team labels only score the result.

Banks per model
  canonical   the model's own canonical_probes.pt (what its PGCT gate used)
  v2_gate     E1 v2 gate bank (carrying, own goal out of view); E1-map models only
  ablated     canonical with every team-relative input zeroed (requested_shelf, relevant_goal channels;
              ego carrying_requested / return_phase / shelf_home_*): the actor must fall back on its
              own learned prior, which is the only place a team difference could survive

Connection rules compared (receiver i chooses among the 3 others)
  hard        1[D_ij <= 0.12]                         (current PGCT)
  sigmoid     sigma((0.12 - D_ij) / 0.03), all connected
  softmax     exp(-D_ij / T) / sum_k, T = median off-diagonal D (scale-free)
Scored by teammate weight share (chance = 1/3) and AUROC (same-team closer = 1).
The E11 local-evidence score (output/latent_team/exp_a_20261013/*_scores.npz) is scored the same way.

    python output/probe_connect_20261019/analyze.py
"""
from __future__ import annotations

import json
import sys
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
from rware.utils.semantic_observation import SEMANTIC_EGO_FEATURES, SEMANTIC_SPATIAL_CHANNELS  # noqa: E402

OUT = Path(__file__).resolve().parent
CUR = ROOT / "runs/pgct09700_physical_mixing_20260929_155338_079437"
SP = ROOT / "runs/sp_20261008_120638_983201"
MODELS = {
    "Stage01 (home 1.0)": (CUR / "stage01_4ag_physical_home1.000", None),
    "Stage02 (home .875)": (CUR / "stage02_4ag_physical_home0.875", None),
    "Stage03 (home .75)": (CUR / "stage03_4ag_physical_home0.750", "stage03_init_e1"),
    "Stage04 PGCT (home .5)": (CUR / "stage04_4ag_physical_home0.500", "stage04_pgct"),
    "E1 IPPO": (SP / "e1_ippo_s1", "e1_ippo_s1"),
    "E1 PGCT": (SP / "e1_pgct_s1", "e1_pgct_s1"),
    "E2 IPPO": (SP / "e2_ippo_s1", "e2_ippo_s1"),
    "E2 PGCT": (SP / "e2_pgct_s1", "e2_pgct_s1"),
}
V2_E1 = ROOT / "output/env_view_20261015/probes_v2/probe_bank_v2_e1_s1.pt"
EVIDENCE = ROOT / "output/latent_team/exp_a_20261013"
PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]
E11_PAIRS = [(0, 2), (1, 3), (0, 1), (0, 3), (1, 2), (2, 3)]
TAU, SIG_T = 0.12, 0.03

EGO_DIM = len(SEMANTIC_EGO_FEATURES)
CH = {n: k for k, n in enumerate(SEMANTIC_SPATIAL_CHANNELS)}
EGO = {n: k for k, n in enumerate(SEMANTIC_EGO_FEATURES)}
TEAM_CHANNELS = ["requested_shelf", "relevant_goal"]
TEAM_EGO = ["self_carrying_requested_shelf", "self_return_phase", "self_shelf_home_known",
            "self_shelf_home_dx", "self_shelf_home_dy"]


def ablate(seq: torch.Tensor) -> torch.Tensor:
    x = seq.clone()
    n_ch = len(SEMANTIC_SPATIAL_CHANNELS)
    side = int(round(((x.shape[-1] - EGO_DIM) / n_ch) ** 0.5))
    grid = x[..., EGO_DIM:].reshape(*x.shape[:-1], n_ch, side, side)
    for c in TEAM_CHANNELS:
        grid[..., CH[c], :, :] = 0
    x[..., EGO_DIM:] = grid.reshape(*x.shape[:-1], -1)
    for e in TEAM_EGO:
        x[..., EGO[e]] = 0
    return x


@torch.inference_mode()
def responses(net, seq):
    h, _ = net.initial_hidden(len(seq), torch.device("cpu"))
    out = []
    for t in range(seq.shape[1]):
        logits, h = net._actor_step(seq[:, t], h)
        out.append(logits.softmax(-1).numpy())
    return np.stack(out, 1).astype(np.float64)


def js(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (np.log(np.maximum(a, 1e-12)) - np.log(np.maximum(b, 1e-12)))).sum(-1)
    return float((0.5 * kl(p, m) + 0.5 * kl(q, m)).mean())


def score_rules(D, teams, distance=True):
    """D: 4x4 symmetric (distance if distance else similarity score). Returns rule metrics."""
    D = np.asarray(D, float)
    S = -D if distance else D  # larger = more similar
    mate = [next(j for j in range(4) if j != i and teams[j] == teams[i]) for i in range(4)]
    same = [S[i, j] for i, j in PAIRS if teams[i] == teams[j]]
    cross = [S[i, j] for i, j in PAIRS if teams[i] != teams[j]]
    auc = float(np.mean([1.0 if s > c else 0.5 if s == c else 0.0 for s in same for c in cross]))
    nearest = int(sum(max((j for j in range(4) if j != i), key=lambda j: S[i, j]) == mate[i] for i in range(4)))
    res = dict(auroc=auc, nearest_is_mate=nearest)
    if distance:
        off = D[~np.eye(4, dtype=bool)]
        hard = (D <= TAU) & ~np.eye(4, dtype=bool)
        res["hard_arcs_same"] = int(sum(hard[i, mate[i]] for i in range(4)))
        res["hard_arcs_cross"] = int(hard.sum() - res["hard_arcs_same"])
        sig = 1 / (1 + np.exp(-(TAU - D) / SIG_T))
        temp = float(np.median(off))
        soft = np.exp(-D / max(temp, 1e-9))
    else:
        sig = 1 / (1 + np.exp(-(S - np.median(S[~np.eye(4, dtype=bool)])) / max(np.std(S), 1e-9)))
        soft = np.exp(S / max(np.std(S), 1e-9))
    for name, W in (("sigmoid", sig), ("softmax", soft)):
        W = W * ~np.eye(4, dtype=bool)
        res[f"{name}_mate_share"] = float(np.mean([W[i, mate[i]] / W[i].sum() for i in range(4)]))
        res[f"{name}_mean_weight"] = float(W.sum() / 12)
    return res


def team_ids(meta, ck):
    cfg = meta["config"]
    env_kwargs = dict(T._load_checkpoint(str(ck), torch.device("cpu")).get("effective_env_kwargs") or
                      json.loads(cfg["env_kwargs_json"]))
    with T.make_env(cfg["env_id"], env_kwargs, cfg["observation_format"]) as env:
        env.reset(seed=0)
        base = env.unwrapped
        return [int(t) for t in np.asarray(base.agent_team_ids)], (base.grid_size if hasattr(base, "grid_size") else None)


def main():
    v2 = torch.load(V2_E1, map_location="cpu", weights_only=False)["sequences"].float()
    rows = {}
    for name, (folder, ev_key) in MODELS.items():
        ck = folder / "final.pt"
        agents, meta = load_model(ck)
        before = model_hash(agents)
        teams, grid = team_ids(meta, ck)
        bank_file = folder / "canonical_probes.pt"
        if not bank_file.exists():  # IPPO jobs have no gate; use the same-map PGCT job's bank
            bank_file = folder.parent / folder.name.replace("ippo", "pgct") / "canonical_probes.pt"
        canonical = torch.load(bank_file, map_location="cpu", weights_only=False)["sequences"].float()
        banks = {"canonical": canonical, "ablated": ablate(canonical)}
        if name.startswith(("Stage", "E1")):
            banks["v2_gate"] = v2
            banks["v2_gate_ablated"] = ablate(v2)
        row = dict(checkpoint=str(ck.relative_to(ROOT)), env_steps=int(meta["env_steps"]), teams=teams, banks={})
        for bname, seq in banks.items():
            P = [responses(a, seq) for a in agents]
            D = np.zeros((4, 4))
            for i, j in PAIRS:
                D[i, j] = D[j, i] = js(P[i], P[j])
            row["banks"][bname] = dict(matrix=D.round(4).tolist(), **score_rules(D, teams))
        if ev_key and (EVIDENCE / f"{ev_key}_scores.npz").exists():
            z = np.load(EVIDENCE / f"{ev_key}_scores.npz")
            # E11 local evidence: Beta(1,1) posterior mean of "same team" over all 20 episodes
            sc = (z["evidence_same"].sum(0) + 1) / (z["evidence_same"].sum(0) + z["evidence_other"].sum(0) + 2)
            S = np.zeros((4, 4))
            for k, (i, j) in enumerate(E11_PAIRS):  # E11 stores pairs in latent_team/common.py order
                S[i, j] = S[j, i] = sc[k]
            row["banks"]["local_evidence(E11)"] = dict(matrix=S.round(4).tolist(),
                                                       **score_rules(S, [int(t) for t in z["team_ids"]], distance=False))
        if before != model_hash(agents):
            raise RuntimeError("weights changed")
        rows[name] = row
        for b, r in row["banks"].items():
            print(f"{name:24s} {b:20s} AUROC={r['auroc']:.2f} nearest_mate={r['nearest_is_mate']}/4 "
                  f"hard(same/cross)={r.get('hard_arcs_same', '-')}/{r.get('hard_arcs_cross', '-')} "
                  f"sigmoid_share={r['sigmoid_mate_share']:.2f} softmax_share={r['softmax_mate_share']:.2f}", flush=True)
    (OUT / "results.json").write_text(json.dumps(dict(rule=dict(tau=TAU, sigmoid_temperature=SIG_T,
                                                                softmax_temperature="median off-diagonal D"),
                                                      models=rows), indent=2), encoding="utf-8")
    plot(rows)


def plot(rows):
    names = list(rows)
    banks = ["canonical", "ablated", "v2_gate", "v2_gate_ablated", "local_evidence(E11)"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, key, title in ((axes[0], "auroc", "AUROC (same team closer = 1, chance .5)"),
                           (axes[1], "softmax_mate_share", "softmax weight on teammate (chance 1/3)")):
        width = 0.16
        for b_i, b in enumerate(banks):
            vals = [rows[n]["banks"].get(b, {}).get(key, np.nan) for n in names]
            ax.bar(np.arange(len(names)) + (b_i - 2) * width, vals, width, label=b)
        ax.axhline(0.5 if key == "auroc" else 1 / 3, color="k", ls=":")
        ax.set_xticks(range(len(names)), names, rotation=40, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
    axes[0].legend(fontsize=7)
    fig.savefig(OUT / "probe_connect.png", dpi=120)


if __name__ == "__main__":
    main()
