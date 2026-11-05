"""Do existing trained actors separate teams on the probe banks?  Read-only, frozen models.

Feeds each bank to all four actors (identical inputs, zero hidden, unmasked: the PGCT
convention), computes the 4x4 Jensen-Shannon matrix and compares same-team vs cross-team
pairs. Team labels only score the result.

    python output/probe_team_check_20261015/probe_team_check.py
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
from examples.diagnose_pgct_rollout_compatibility import load_model, model_hash  # noqa: E402
from examples.pgct_probe_diagnostics import common_probe_response  # noqa: E402

OUT = Path(__file__).resolve().parent
SP = ROOT / "runs/sp_20261008_120638_983201"
CUR = ROOT / "runs/pgct09700_physical_mixing_20260929_155338_079437"
MODELS = {  # all trained on the E1 11x10 map (team goals adjacent at (4,10)/(5,10))
    "Stage03 (curriculum, home .75)": CUR / "stage03_4ag_physical_home0.750/final.pt",
    "Stage04 PGCT (curriculum, home .5)": CUR / "stage04_4ag_physical_home0.500/final.pt",
    "E1 IPPO (Stage03 + 4M)": SP / "e1_ippo_s1/final.pt",
    "E1 PGCT (Stage03 + 4M)": SP / "e1_pgct_s1/final.pt",
}
CANONICAL = SP / "e1_pgct_s1/canonical_probes.pt"
V2 = ROOT / "output/env_view_20261015/probes_v2/probe_bank_v2_e1_s1.pt"
TEAM = np.array([0, 1, 0, 1])  # display agents 1,3 = team 0 (red); 2,4 = team 1 (blue)
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
SAME = [k for k, (i, j) in enumerate(PAIRS) if TEAM[i] == TEAM[j]]
CROSS = [k for k, (i, j) in enumerate(PAIRS) if TEAM[i] != TEAM[j]]


def banks():
    canonical = torch.load(CANONICAL, map_location="cpu", weights_only=False)["sequences"].float()
    v2 = torch.load(V2, map_location="cpu", weights_only=False)
    tr, tr_sc = v2["transfer_sequences"].float(), v2["transfer_scenarios"]
    pick = lambda phases: tr[[k for k, s in enumerate(tr_sc) if s["phase"] in phases]]
    return {
        "canonical 48\n(current PGCT)": canonical,
        "v2 gate 48\n(carrying, no goal in view)": v2["sequences"].float(),
        "v2: carrying,\nown goal VISIBLE": pick({"carry_goal_visible", "at_goal"}),
        "v2: carrying,\nown goal hidden": pick({"carry_goal_hidden"}),
        "v2: searching\n(empty-handed)": pick({"search"}),
    }


def summarise(js_pairs):
    m = js_pairs.mean(1)  # mean over probes, per pair
    mat = np.zeros((4, 4))
    for k, (i, j) in enumerate(PAIRS):
        mat[i, j] = mat[j, i] = m[k]
    hits = [int(TEAM[np.argmin(np.where(np.arange(4) == a, np.inf, mat[a]))] == TEAM[a]) for a in range(4)]
    # Per-probe AUROC-like score: fraction of (same, cross) pair comparisons where same < cross.
    s, c = m[SAME], m[CROSS]
    auroc = float(np.mean([(si < ci) + 0.5 * (si == ci) for si in s for ci in c]))
    return dict(matrix=mat.tolist(), same=float(s.mean()), cross=float(c.mean()),
                ratio=float(c.mean() / max(s.mean(), 1e-12)), auroc=auroc, teammate_hits=hits)


def main():
    bank_map = banks()
    results = {}
    for name, path in MODELS.items():
        agents, meta = load_model(path)
        before = model_hash(agents)
        results[name] = dict(checkpoint=str(path.relative_to(ROOT)), env_steps=int(meta["env_steps"]), banks={})
        for bname, seq in bank_map.items():
            r = common_probe_response(agents, seq, torch.device("cpu"))
            res = summarise(np.asarray(r["js_per_probe"]))
            res.update(n=int(len(seq)), unique=int(r["unique_probe_count"]))
            results[name]["banks"][bname] = res
        assert before == model_hash(agents), "weights changed"
        print(name, {b.split(chr(10))[0]: round(v["ratio"], 2) for b, v in results[name]["banks"].items()})
    (OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    plot(results, list(bank_map))


def plot(results, bank_names):
    models = list(results)
    vmax = max(np.max(b["matrix"]) for r in results.values() for b in r["banks"].values())
    fig, axes = plt.subplots(len(models), len(bank_names), figsize=(3.1 * len(bank_names), 3.0 * len(models) + 0.6))
    colors = ["#d62728", "#1f77b4"]
    labels = ["A1", "A2", "A3", "A4"]
    for r, m in enumerate(models):
        for c, b in enumerate(bank_names):
            ax, res = axes[r, c], results[m]["banks"][b]
            mat = np.array(res["matrix"])
            ax.imshow(mat, cmap="viridis", vmin=0, vmax=vmax)
            for i in range(4):
                for j in range(4):
                    if i != j:
                        same = TEAM[i] == TEAM[j]
                        ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=7,
                                color="white", fontweight="bold" if same else "normal")
                        if same:
                            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="white", lw=2))
            ax.set_xticks(range(4)); ax.set_yticks(range(4))
            ax.set_xticklabels(labels, fontsize=8); ax.set_yticklabels(labels, fontsize=8)
            for t, lab in zip(ax.get_xticklabels() + ax.get_yticklabels(), list(range(4)) * 2):
                t.set_color(colors[TEAM[lab]])
            verdict = "separates" if res["auroc"] == 1.0 and res["ratio"] > 1.5 else "does NOT separate"
            ax.set_title(f"cross/same = {res['ratio']:.2f}  ({verdict})\nteammate found {sum(res['teammate_hits'])}/4",
                         fontsize=8)
            if r == 0:
                ax.annotate(b + f"\n[{res['n']} probes]", (0.5, 1.42), xycoords="axes fraction", ha="center",
                            fontsize=9, fontweight="bold")
            if c == 0:
                ax.set_ylabel(m + f"\n{results[m]['env_steps'] / 1e6:.1f}M steps", fontsize=8)
    fig.suptitle("Do trained actors separate teams on the probes?  JS between agents (lower = more similar)\n"
                 "white box = same-team pair (red A1-A3, blue A2-A4). Teams separate only if boxed values are clearly"
                 " lower than the rest.  All models trained on E1 (goals adjacent).", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "js_matrices.png", dpi=130)

    # Compact bar view: same vs cross mean JS.
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.8), sharey=True)
    x = np.arange(len(bank_names))
    for ax, m in zip(axes, models):
        s = [results[m]["banks"][b]["same"] for b in bank_names]
        c = [results[m]["banks"][b]["cross"] for b in bank_names]
        ax.bar(x - .2, s, .4, label="same team", color="#2ca02c")
        ax.bar(x + .2, c, .4, label="cross team", color="#7f7f7f")
        ax.set_xticks(x); ax.set_xticklabels([b.replace("\n", " ").split(" (")[0].split(" [")[0] for b in bank_names],
                           rotation=35, ha="right", fontsize=7)
        ax.set_title(m, fontsize=9)
        ax.axhline(0.12, color="red", ls=":", lw=1)
    axes[0].set_ylabel("mean JS (nats)"); axes[0].legend(fontsize=8)
    fig.suptitle("Same-team vs cross-team policy distance (green should be clearly lower if teams were identifiable;"
                 " red dotted = PGCT gate 0.12)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "same_vs_cross.png", dpi=130)


if __name__ == "__main__":
    main()
