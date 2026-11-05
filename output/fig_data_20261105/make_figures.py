"""Data figures for the paper (T205). Read-only: training logs, saved checkpoints and saved check results.

1. discovery_log_v2.pdf: Figure 3 with a fourth panel. (a)-(c) are recomputed exactly as in
   output/pgct_docs_20261102/evidence_plots.py from runs/acv2_s1_20261029_084340 (logged EMA of the earlier pooled
   rule with mirrored probes, maturity, cumulative approved arcs). (d) is new: the raw distance recomputed offline from
   the saved predictors and probes of every checkpoint of the same run, without mirrored probes, for the pooled
   distance and the per-action maximum (same-team maximum and different-team minimum over mature pairs), using
   output/ac_drift_20261104/stratified_gate_check.py.
2. counterfactual.pdf: the counterfactual check of Section 5.3 (output/oracle_why_20261029/cf_*.json, four
   checkpoint datasets): Critic-predicted uplift dV against the measured return gain dMC (H = 100, 8 common-random-
   number repetitions), for lagging and leading receivers, by true relation of the peer.
"""
from pathlib import Path
import json
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "output/ac_drift_20261104"))
import stratified_gate_check as SG  # noqa: E402

GATE = ROOT / "runs/acv2_s1_20261029_084340/b6_corner_indiv_pen005_acv2_pgct_s1"
CF = ROOT / "output/oracle_why_20261029"
PAPER_FIG = Path(r"C:/Users/95101/OneDrive/Desktop/제출 논문/paper/figures")
BLUE, ORANGE, GREY = "#1F4E9A", "#C0504D", "#6B6B6B"
BIN = 100_000


def rows(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def logged_discovery():
    bins = {}
    for r in rows(GATE / "metrics.jsonl"):
        m, b = r["metrics"], r["env_steps"] // BIN
        for key, name in (("update/ac_v2_audit_distance_same", "same"), ("update/ac_v2_audit_distance_cross", "cross"),
                          ("update/ac_v2_mature_agents", "mature")):
            if key in m and m[key] == m[key]:
                bins.setdefault(b, {}).setdefault(name, []).append(m[key])
    disc = [dict(step=(b * BIN + BIN // 2) / 1e6, **{k: sum(v) / len(v) for k, v in d.items()})
            for b, d in sorted(bins.items())]
    same = cross = 0
    cumulative = []
    for r in rows(GATE / "transfer.jsonl"):
        same += r["same_team_arcs"]
        cross += r["cross_team_arcs"]
        cumulative.append((r["env_steps"] / 1e6, same, cross))
    return disc, cumulative


def offline_distances():
    out = []
    for ck in sorted(GATE.glob("checkpoint_*.pt")):
        steps, cfg, preds = SG.load(ck)
        vals = {"same": {"pooled": [], "max": []}, "cross": {"pooled": [], "max": []}}
        for i in range(6):
            for j in range(i + 1, 6):
                if not (preds[i]["mature"] and preds[j]["mature"]):
                    continue
                d = SG.pair(cfg, preds, i, j, mirror=False)
                rel = "same" if SG.TEAM[i] == SG.TEAM[j] else "cross"
                vals[rel]["pooled"].append(d["pooled"])
                if d["strat_m8"] is not None:
                    vals[rel]["max"].append(d["strat_m8"][0])
        out.append(dict(step=steps / 1e6,
                        same_pooled=max(vals["same"]["pooled"]), same_max=max(vals["same"]["max"]),
                        cross_pooled=min(vals["cross"]["pooled"]), cross_max=min(vals["cross"]["max"])))
    return out


def figure3(disc, cumulative, offline):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "axes.grid": True, "grid.alpha": .25})
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.25), constrained_layout=True)
    ax = axs[0]
    ax.plot([d["step"] for d in disc if "same" in d], [d["same"] for d in disc if "same" in d], color=BLUE, lw=2,
            label="Same team")
    ax.plot([d["step"] for d in disc if "cross" in d], [d["cross"] for d in disc if "cross" in d], color=ORANGE, lw=2,
            label="Different team")
    ax.axhline(.5, color=GREY, ls="--", lw=1.1, label="Threshold")
    ax.set(xlabel="Training steps (M)", ylabel="Distance EMA (earlier rule)", ylim=(0, 1.05), xlim=(0, 5.05))
    ax.legend(fontsize=6.5, loc="center right", frameon=False)
    ax = axs[1]
    ax.plot([d["step"] for d in disc if "mature" in d], [d["mature"] for d in disc if "mature" in d], color=BLUE, lw=2)
    ax.set(xlabel="Training steps (M)", ylabel="Mature reward predictors", ylim=(0, 6.3), yticks=[0, 2, 4, 6],
           xlim=(0, 5.05))
    ax = axs[2]
    ax.plot([c[0] for c in cumulative], [c[1] for c in cumulative], color=BLUE, lw=2, label="Same team")
    ax.plot([c[0] for c in cumulative], [c[2] for c in cumulative], color=ORANGE, lw=2, label="Different team")
    ax.set(xlabel="Training steps (M)", ylabel="Cumulative approved arcs", xlim=(0, 5.05))
    ax.legend(fontsize=6.5, loc="upper left", frameon=False)
    ax = axs[3]
    x = [o["step"] for o in offline]
    ax.plot(x, [o["cross_max"] for o in offline], color=ORANGE, lw=2, marker="o", ms=2.5, label="Diff. min, per-action")
    ax.plot(x, [o["cross_pooled"] for o in offline], color=ORANGE, lw=1.4, ls=":", marker="o", ms=2,
            label="Diff. min, pooled")
    ax.plot(x, [o["same_max"] for o in offline], color=BLUE, lw=2, marker="o", ms=2.5, label="Same max, per-action")
    ax.plot(x, [o["same_pooled"] for o in offline], color=BLUE, lw=1.4, ls=":", marker="o", ms=2,
            label="Same max, pooled")
    ax.axhline(.5, color=GREY, ls="--", lw=1.1)
    ax.set(xlabel="Training steps (M)", ylabel="Distance (offline, raw)", ylim=(0, 1.05), xlim=(0, 5.05))
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], color="#333333", lw=2), Line2D([], [], color="#333333", lw=1.4, ls=":")],
              ["per-action max", "pooled"], fontsize=6.5, loc="upper left", frameon=False,
              bbox_to_anchor=(0.0, 0.47), handlelength=1.6, labelspacing=.25)
    for a, tag in zip(axs, "abcd"):
        a.set_title(f"({tag})", fontsize=8, loc="left")
    fig.savefig(HERE / "discovery_log_v2.pdf")
    fig.savefig(HERE / "discovery_log_v2.png", dpi=200)
    plt.close(fig)


def figure_cf():
    data = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(CF.glob("cf_*.json")) if p.name != "cf_summary.json"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "axes.grid": True, "grid.alpha": .25})
    fig, axs = plt.subplots(1, 2, figsize=(3.35, 2.05), constrained_layout=True, sharey=True)
    summary = {}
    for ax, lag, title in zip(axs, (True, False), ("(a) Lagging receiver", "(b) Leading receiver")):
        for kind, color, label in (("same", BLUE, "Same-team peer"), ("cross", ORANGE, "Different-team peer")):
            pts = [(s["dV"], s["dMC"]) for d in data for s in d["state"] if s["lag"] == lag and s["kind"] == kind]
            xs, ys = np.array(pts).T
            ax.scatter(xs, ys, s=7, color=color, alpha=.65, lw=0, label=label)
            hi = ys[xs > .2]
            summary[f"{'lag' if lag else 'lead'}_{kind}"] = dict(n=int(len(xs)), n_dV_gt_02=int(len(hi)),
                                                                   mean_dMC_dV_gt_02=float(hi.mean()) if len(hi) else None,
                                                                   positive_share=float((hi > 0).mean()) if len(hi) else None)
        ax.axhline(0, color=GREY, lw=.8)
        ax.axvline(.2, color=GREY, lw=.8, ls=":")
        ax.set_title(title, fontsize=7, loc="left")
        ax.set_xlabel("Predicted uplift $V_j-V_i$")
    axs[0].set_ylabel("Measured return gain")
    axs[0].legend(fontsize=5.8, frameon=False, loc="lower right", handletextpad=.1)
    fig.savefig(HERE / "counterfactual.pdf")
    fig.savefig(HERE / "counterfactual.png", dpi=220)
    plt.close(fig)
    return summary


def main():
    disc, cumulative = logged_discovery()
    offline = offline_distances()
    figure3(disc, cumulative, offline)
    summary = figure_cf()
    report = dict(offline_distance=offline, counterfactual=summary)
    (HERE / "figure_data.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["counterfactual"], indent=1))
    print("offline min cross per-action", min(o["cross_max"] for o in offline),
          "min cross pooled", min(o["cross_pooled"] for o in offline),
          "max same per-action", max(o["same_max"] for o in offline))
    if PAPER_FIG.is_dir():
        for name in ("discovery_log_v2.pdf", "counterfactual.pdf"):
            shutil.copy(HERE / name, PAPER_FIG / name)


if __name__ == "__main__":
    main()
