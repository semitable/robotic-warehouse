"""Figure for the goal-axis statistic (E54).  Read-only; no training.

  (a) one discriminative gate probe, and the mean action probabilities of left-goal vs right-goal agents on it
      (all agents of the 6-agent corner-goal checkpoints)
  (b) s_i per agent, split by the agent's goal side, for each checkpoint group
  (c) pair-level team separation on the same 13 probes: AUROC of mean JS, AUROC of |s_i - s_j|, and the accuracy
      of the sign rule (same sign of s => teammate)

    python output/pgct_progress_paper_20261023/make_figures_axis.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "output/probe_discriminative_20261023"))
import discriminative_probes as DP  # noqa: E402
import make_figures as MF  # noqa: E402

OUT = HERE / "figures"
SHORT = {"corner/team .005 (B, 6 ag)": "corner/\nteam .005", "corner/indiv .005 (B, 6 ag)": "corner/\nindiv .005",
         "B team, penalty 0 (6 ag)": "B team\npen 0", "B team, penalty .02 (6 ag, 2M)": "B team\npen .02",
         "A team, penalty .02 (4 ag, 2M)": "A team\n4 ag", "E1-F indiv .02 (4 ag, 1.54M)": "E1-F\n4 ag"}
NAMES = ["NOOP", "FWD", "LEFT", "RIGHT", "TOGGLE"]


def main():
    torch.set_num_threads(4)
    gate, opt, disc = DP.discriminative_set()
    probe = 6
    res = json.loads((ROOT / "output/probe_discriminative_20261023/goal_axis_results.json").read_text(encoding="utf-8"))
    # (a) mean action probabilities on one probe, by goal side (6-agent corner groups)
    left_p, right_p = [], []
    for group, items in DP.GROUPS.items():
        if "6 ag" not in group:
            continue
        for run, ck in items:
            agents, _ = DP.load_model(run / ck)
            c = DP.T._load_checkpoint(str(run / ck), torch.device("cpu"))
            kw = c["effective_env_kwargs"]
            with DP.T.make_env(c["config"]["env_id"], dict(kw), c["config"]["observation_format"]) as env:
                env.reset(seed=0)
                teams = [int(t) for t in np.asarray(env.unwrapped.agent_team_ids)]
            _, P = DP.js_and_probs(agents, gate[[probe]])
            for i, t in enumerate(teams):
                (left_p if kw["team_goal_positions"][t][0][0] == 0 else right_p).append(P[i][0])
    left_p, right_p = np.array(left_p), np.array(right_p)

    fig = plt.figure(figsize=(10.5, 3.3))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.75, 1.0, 1.5, 1.4])
    ax0 = fig.add_subplot(gs[0])
    s = DP.discriminative_set  # noqa: F841
    MF.draw_probe(ax0, gate[probe], f"gate #{probe}: carrying,\ngoal hidden")
    lo, ro = sorted(opt[probe][0]), sorted(opt[probe][1])
    ax0.set_xlabel(f"optimal first action\nleft goal: {'/'.join(NAMES[a] for a in lo)}\nright goal: {'/'.join(NAMES[a] for a in ro)}", fontsize=7)
    ax1 = fig.add_subplot(gs[1])
    x = np.arange(5)
    ax1.bar(x - 0.2, left_p.mean(0), 0.4, color=MF.RED, label=f"left-goal agents (n={len(left_p)})")
    ax1.bar(x + 0.2, right_p.mean(0), 0.4, color=MF.BLUE, label=f"right-goal agents (n={len(right_p)})")
    ax1.set_xticks(x); ax1.set_xticklabels(NAMES, fontsize=6.5, rotation=40); ax1.set_ylabel("mean action prob.")
    ax1.legend(fontsize=6, frameon=False); ax1.set_title("(a) same probe, by goal side\n(6-agent corner checkpoints)", fontsize=8)

    ax2 = fig.add_subplot(gs[2])
    groups = list(SHORT)
    rng = np.random.default_rng(0)
    for g, name in enumerate(groups):
        L = [v for r in res.values() if r["group"] == name for v, gl in zip(r["s"], r["goal_left"]) if gl]
        Rr = [v for r in res.values() if r["group"] == name for v, gl in zip(r["s"], r["goal_left"]) if not gl]
        ax2.scatter(g - 0.15 + rng.uniform(-0.07, 0.07, len(L)), L, s=7, color=MF.RED, alpha=0.7, label="left-goal agent" if g == 0 else None)
        ax2.scatter(g + 0.15 + rng.uniform(-0.07, 0.07, len(Rr)), Rr, s=7, color=MF.BLUE, alpha=0.7, label="right-goal agent" if g == 0 else None)
    ax2.axhline(0, color="0.4", lw=0.8, ls=":")
    ax2.set_xticks(range(len(groups))); ax2.set_xticklabels([SHORT[g] for g in groups], fontsize=6.5)
    ax2.set_ylabel("s_i = p(left-optimal) - p(right-optimal)"); ax2.legend(fontsize=6, frameon=False, loc="lower left")
    ax2.set_title("(b) goal-axis value per agent (13 probes)", fontsize=8)

    ax3 = fig.add_subplot(gs[3])
    vals = {k: [] for k in ("js", "axis", "sign")}
    for name in groups:
        rs = [r for r in res.values() if r["group"] == name]
        vals["js"].append(np.mean([r["auroc_js13"] for r in rs]))
        vals["axis"].append(np.mean([r["auroc_axis"] for r in rs]))
        accs = []
        for r in rs:
            sv, gl = np.array(r["s"]), r["goal_left"]
            pairs = [(i, j) for i in range(len(sv)) for j in range(i + 1, len(sv))]
            accs.append(np.mean([(np.sign(sv[i]) == np.sign(sv[j])) == (gl[i] == gl[j]) for i, j in pairs]))
        vals["sign"].append(np.mean(accs))
    x = np.arange(len(groups))
    ax3.bar(x - 0.27, vals["js"], 0.27, color="0.7", label="AUROC, mean JS")
    ax3.bar(x, vals["axis"], 0.27, color="#2ca02c", label="AUROC, |s_i - s_j|")
    ax3.bar(x + 0.27, vals["sign"], 0.27, color="#9467bd", label="accuracy, sign rule")
    ax3.axhline(0.5, color="0.4", lw=0.8, ls=":")
    ax3.set_xticks(x); ax3.set_xticklabels([SHORT[g] for g in groups], fontsize=6.5); ax3.set_ylim(0, 1)
    ax3.legend(fontsize=6, frameon=False, loc="upper right"); ax3.set_title("(c) pair-level team separation", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_goal_axis.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_goal_axis.png", dpi=130, bbox_inches="tight")
    print("ok", len(left_p), len(right_p))


if __name__ == "__main__":
    main()
