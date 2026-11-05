"""Actual A+C gate figure from saved logs (read-only): transfers, v2 distances, acceptances, maturity.

Usage: python plot_gate.py <job_dir> [out.png]
The W&B panels probe/same_team_gate and probe/cross_team_gate show the trainer's unrestricted gate
(physical-neighbour ratio), not the A+C decision, so this figure uses transfer.jsonl and update/ac_v2_*.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SAME, CROSS, NEUTRAL = "#2a78d6", "#eb6834", "#52514e"  # categorical slots 1-2, secondary ink
GRID, INK, SURFACE = "#e4e3df", "#0b0b0b", "#fcfcfb"
BIN = 50_000
THRESHOLD = .5


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def binned(steps, values, edges):
    steps, values = np.asarray(steps, float), np.asarray(values, float)
    out = np.full(len(edges) - 1, np.nan)
    for k in range(len(edges) - 1):
        mask = (steps > edges[k]) & (steps <= edges[k + 1]) & np.isfinite(values)
        if mask.sum() >= 10:  # skip the short tail bin after the last full 50k steps
            out[k] = values[mask].mean()
    return out


def series(rows, key):
    pairs = [(r["env_steps"], r["metrics"][key]) for r in rows if key in r.get("metrics", {})]
    return ([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else ([], [])


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    ax.set_ylabel(ylabel, fontsize=9, color=NEUTRAL)
    ax.grid(axis="y", color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=NEUTRAL, labelsize=8.5)


def end_label(ax, x, y, text, color):
    good = np.flatnonzero(np.isfinite(y))
    if len(good):
        ax.annotate(text, (x[good[-1]], y[good[-1]]), xytext=(4, 0), textcoords="offset points",
                    fontsize=8.5, color=INK, va="center")
        ax.plot(x[good[-1]], y[good[-1]], "o", ms=4, color=color, mec=SURFACE, mew=1.5)


def main():
    job = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).with_name(job.name + "_gate.png")
    transfer = read_jsonl(job / "transfer.jsonl")
    metrics = read_jsonl(job / "metrics.jsonl")
    last = max(r["env_steps"] for r in transfer)
    edges = np.arange(0, last + BIN, BIN)
    centers = (edges[:-1] + edges[1:]) / 2 / 1e6
    has_v2 = any("update/ac_v2_mature_agents" in r.get("metrics", {}) for r in metrics)

    if has_v2:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, facecolor=SURFACE)
    else:  # v1 logs only the actual allocation
        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
        axes = np.array([[ax]])
    fig.suptitle(f"A+C gate — {job.name}" + ("  " if has_v2 else "\n")
                 + "(actual allocation, not trainer probe/*_gate)",
                 x=.01, ha="left", fontsize=12, color=INK)

    # A. Cumulative actual transfer arcs.
    ax = axes[0, 0]
    x = np.array([r["env_steps"] for r in transfer]) / 1e6
    for key, color, name in (("same_team_arcs", SAME, "same team"), ("cross_team_arcs", CROSS, "cross team")):
        y = np.cumsum([r[key] for r in transfer])
        ax.plot(x, y, color=color, linewidth=2, label=name)
        ax.annotate(f"{name} {int(y[-1]):,}", (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=8.5, color=INK, va="center")
    held = [r["env_steps"] for r in transfer if r.get("warmup_hold")]
    if held:
        ax.axvspan(0, max(held) / 1e6, color=GRID, alpha=.6, linewidth=0)
        ax.text(max(held) / 1e6 + .05, .97, "warm-up", transform=ax.get_xaxis_transform(), ha="left",
                va="top", fontsize=8, color=NEUTRAL)
    style(ax, "A. Cumulative actual transfer arcs", "arc-updates")

    if has_v2:
        # B. v2 response distance (EMA), binned mean, with the acceptance threshold.
        ax = axes[0, 1]
        for key, color, name in (("update/ac_v2_audit_distance_same", SAME, "same"),
                                 ("update/ac_v2_audit_distance_cross", CROSS, "cross")):
            y = binned(*series(metrics, key), edges)
            ax.plot(centers, y, color=color, linewidth=2, label=name + " team")
            end_label(ax, centers, y, name, color)
        ax.axhline(THRESHOLD, color=NEUTRAL, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(.01, THRESHOLD + .02, f"accept below {THRESHOLD}", transform=ax.get_yaxis_transform(),
                fontsize=8, color=NEUTRAL)
        ax.set_ylim(0, 1.05)
        style(ax, "B. v2 reward-response distance (EMA)", "distance")

        # C. Accepted pairs per update, binned mean.
        ax = axes[1, 0]
        for key, color, name in (("update/ac_v2_audit_same_accepted", SAME, "same"),
                                 ("update/ac_v2_audit_cross_accepted", CROSS, "cross")):
            y = binned(*series(metrics, key), edges)
            ax.plot(centers, y, color=color, linewidth=2, label=name + " team")
            end_label(ax, centers, y, name, color)
        style(ax, "C. Accepted pairs per update (50k-step mean)", "pairs")

        # D. Mature agents.
        ax = axes[1, 1]
        y = binned(*series(metrics, "update/ac_v2_mature_agents"), edges)
        ax.plot(centers, y, color=NEUTRAL, linewidth=2)
        ax.set_ylim(0, 6.3)
        ax.set_yticks(range(7))
        style(ax, "D. Agents eligible to vote (mature r̂)", "agents")

    if has_v2:  # the single v1 panel is already direct-labelled
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("environment steps (M)", fontsize=9, color=NEUTRAL)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
