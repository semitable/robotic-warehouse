"""Figure 2 from real data (T204): reward predictions of two agents on their common probes.

Source: the v2 discovery run (corner goals, individual rewards, seed 1), checkpoint_04250 (4.35M steps). The saved
reward predictors and event buffers are loaded exactly as in output/ac_drift_20261104/drift_analysis.py; probes are
the 32 most recent own events of both agents, without mirrored copies (current method), grouped by public action.
(a) teammates a3, a5; (b) different teams a2, a5. Read-only; writes the PDF next to this script and, if present,
into the paper figures folder.
"""
from pathlib import Path
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
import drift_analysis as DA  # noqa: E402
from latent_team import reward_probe_ac_v2 as V2  # noqa: E402

CKPT = 4250
PAIRS = [("a", 3, 5, "Same team"), ("b", 2, 5, "Different teams")]
ACTIONS = [(1, "Deliveries (FORWARD)"), (4, "Shelf returns (TOGGLE)")]
COL_I, COL_J, SHADE = "#1F1F1F", "#4F8FCF", "#EEF2F7"
PAPER_FIG = Path(r"C:/Users/95101/OneDrive/Desktop/제출 논문/paper/figures")


def pair_data(cfg, preds, i, j):
    parts = []
    for owner in (i, j):
        P = preds[owner]
        n = min(cfg.probes_per_agent, len(P["a"]))
        parts.append((P["b"][:n], P["a"][:n], P["f"][:n], np.full(n, owner)))
    b, a, f, own = (np.concatenate(x) for x in zip(*parts))
    p, q = preds[i]["model"].predict(b, a, f), preds[j]["model"].predict(b, a, f)
    pooled = V2.response_distance(p, q)
    per = {g: V2.response_distance(p[a == g], q[a == g]) for g, _ in ACTIONS}
    dmax, _ = V2.action_max_distance(p, q, a, parts[0][1], parts[1][1], cfg.min_probes)
    return a, own, p, q, pooled, per, dmax


def main():
    steps, cfg, preds, _ = DA.load(CKPT)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axs = plt.subplots(2, 1, figsize=(3.35, 3.0), constrained_layout=True)
    lines = []
    for ax, (tag, i, j, title) in zip(axs, PAIRS):
        a, own, p, q, pooled, per, dmax = pair_data(cfg, preds, i, j)
        x0 = 0
        for g, label in ACTIONS:
            idx = [k for o in (i, j) for k in np.flatnonzero((a == g) & (own == o))]
            xs = np.arange(x0, x0 + len(idx))
            ax.axvspan(x0 - .6, x0 + len(idx) - .4, color=SHADE, zorder=0)
            n_i = int(np.sum((a == g) & (own == i)))
            ax.axvline(x0 + n_i - .5, color="#9A9A9A", lw=.6, ls=":", zorder=1)
            ax.scatter(xs, p[idx], s=7, color=COL_I, zorder=3, label=f"$\\hat r$ of $a_{i}$")
            ax.scatter(xs, q[idx], s=9, facecolors="none", edgecolors=COL_J, lw=.8, marker="s", zorder=3,
                       label=f"$\\hat r$ of $a_{j}$")
            ax.text(x0 + len(idx) / 2, 1.22, f"{label}\n$D^{{a}}$ = {per[g]:.2f}", ha="center", va="bottom",
                    fontsize=6.5)
            ax.text(x0 + n_i / 2 - .5, -0.33, f"events of $a_{i}$", ha="center", fontsize=6, color="#555555")
            ax.text(x0 + n_i + (len(idx) - n_i) / 2 - .5, -0.33, f"of $a_{j}$", ha="center", fontsize=6,
                    color="#555555")
            x0 += len(idx) + 2
        ax.set_xlim(-1.5, x0 - 1.5)
        ax.set_ylim(-0.15, 1.2)
        ax.set_xticks([])
        ax.set_ylabel("Predicted reward")
        ax.set_title(f"({tag}) {title} ($a_{i}$, $a_{j}$): pooled {pooled:.2f}, max {dmax:.2f}",
                     fontsize=7, loc="left", pad=18)
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles[:2], [f"prediction of $a_{i}$", f"prediction of $a_{j}$"], fontsize=6, frameon=False,
                  loc="lower left" if tag == "a" else "lower right", ncol=1, handletextpad=.2,
                  bbox_to_anchor=(0.0, 0.08) if tag == "a" else (1.0, 0.08))
        lines.append(f"({tag}) a{i},a{j} {title}: pooled {pooled:.3f}, per-action "
                     f"{ {lbl: round(per[g], 3) for g, lbl in ACTIONS} }, max {dmax:.3f}, "
                     f"probes {len(a)} (FORWARD {int((a == 1).sum())}, TOGGLE {int((a == 4).sum())})")
    out = HERE / "fig_discovery_data.pdf"
    fig.savefig(out)
    fig.savefig(HERE / "fig_discovery_data.png", dpi=220)
    lines.insert(0, f"checkpoint {CKPT}: {steps} env steps")
    (HERE / "fig_discovery_data_stdout.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    if PAPER_FIG.is_dir():
        shutil.copy(out, PAPER_FIG / out.name)


if __name__ == "__main__":
    main()
