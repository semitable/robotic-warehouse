"""Evidence for the PGCT paper and slides, recomputed from raw training logs (read-only).

Discovery component: runs/acv2_s1_20261029_084340 (corner goals, individual rewards, coordinates masked,
seed 1, about 5M steps). Baseline: completed independent-PPO seeds 1 and 2 of runs/ac_20261023_185556.
Writes evidence.json, CSV files and figures for paper/figures and presentation/figures.
"""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
GATE = ROOT / "runs/acv2_s1_20261029_084340/b6_corner_indiv_pen005_acv2_pgct_s1"
IPPO = {s: ROOT / f"runs/ac_20261023_185556/b6_corner_indiv_pen005_ac_ippo_s{s}" for s in (1, 2)}
BIN = 100_000


def rows(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# Discovery: 100k-step bin means of the logged audit EMA distances and maturity; allocations from transfer.jsonl.
bins = {}
for r in rows(GATE / "metrics.jsonl"):
    m, b = r["metrics"], r["env_steps"] // BIN
    for key, name in (("update/ac_v2_audit_distance_same", "same"), ("update/ac_v2_audit_distance_cross", "cross"),
                      ("update/ac_v2_mature_agents", "mature")):
        if key in m and m[key] == m[key]:
            bins.setdefault(b, {}).setdefault(name, []).append(m[key])
discovery = [dict(step=(b * BIN + BIN // 2) / 1e6, **{k: sum(v) / len(v) for k, v in d.items()})
             for b, d in sorted(bins.items())]
same = cross = 0
cumulative, first_cross = [], None
for r in rows(GATE / "transfer.jsonl"):
    same += r["same_team_arcs"]
    cross += r["cross_team_arcs"]
    if r["cross_team_arcs"] and first_cross is None:
        first_cross = r["env_steps"]
    cumulative.append((r["env_steps"] / 1e6, same, cross))
last_step = cumulative[-1][0]

# Baseline: deterministic evaluation (20 fixed resets x 500 steps per point).
baseline = {}
for s, path in IPPO.items():
    baseline[s] = [(r["env_steps"] / 1e6, r["metrics"]["eval/deliveries"]) for r in rows(path / "metrics.jsonl")
                   if "eval/deliveries" in r["metrics"]]

evidence = dict(
    discovery=dict(source=str(GATE.relative_to(ROOT)), bin_steps=BIN, bins=discovery,
                   same_team_arc_updates=same, cross_team_arc_updates=cross,
                   same_share=same / (same + cross), first_cross_team_step=first_cross, last_step_million=last_step),
    baseline={str(s): dict(source=str(p.relative_to(ROOT)), points=len(baseline[s]), last=baseline[s][-1])
              for s, p in IPPO.items()})
(OUT / "evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
with open(OUT / "discovery_bins.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["step_million", "same_ema", "cross_ema", "mature_agents"])
    for d in discovery:
        w.writerow([d["step"], d.get("same", ""), d.get("cross", ""), d.get("mature", "")])
with open(OUT / "baseline_eval.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["seed", "step_million", "eval_deliveries"])
    for s, pts in baseline.items():
        w.writerows([s, x, y] for x, y in pts)
thin = cumulative[::25] + [cumulative[-1]]
with open(OUT / "allocations_cumulative.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["step_million", "same_team", "cross_team"])
    w.writerows(thin)

BLUE, ORANGE, GREY = "#1F4E9A", "#C0504D", "#6B6B6B"
x = [d["step"] for d in discovery]


def style(size):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": size, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "axes.grid": True, "grid.alpha": .25})


def ema_panel(ax, legend_size):
    ax.plot([d["step"] for d in discovery if "same" in d], [d["same"] for d in discovery if "same" in d],
            color=BLUE, lw=2.2, label="Same team")
    ax.plot([d["step"] for d in discovery if "cross" in d], [d["cross"] for d in discovery if "cross" in d],
            color=ORANGE, lw=2.2, label="Different team")
    ax.axhline(.5, color=GREY, ls="--", lw=1.3, label="Threshold 0.5")
    ax.set(xlabel="Training steps (million)", ylabel="Response distance (EMA)", ylim=(0, 1.05), xlim=(0, 5.05))
    ax.legend(fontsize=legend_size, loc="center right", frameon=False)


def alloc_panel(ax, legend_size):
    ax.plot([c[0] for c in cumulative], [c[1] for c in cumulative], color=BLUE, lw=2.2, label="Same team")
    ax.plot([c[0] for c in cumulative], [c[2] for c in cumulative], color=ORANGE, lw=2.2, label="Different team")
    ax.set(xlabel="Training steps (million)", ylabel="Cumulative approved arcs", xlim=(0, 5.05))
    ax.legend(fontsize=legend_size, loc="upper left", frameon=False)


def mature_panel(ax):
    ax.plot([d["step"] for d in discovery if "mature" in d], [d["mature"] for d in discovery if "mature" in d],
            color=BLUE, lw=2.2)
    ax.set(xlabel="Training steps (million)", ylabel="Mature reward predictors", ylim=(0, 6.3), yticks=[0, 2, 4, 6],
           xlim=(0, 5.05))


def baseline_panel(ax, legend_size):
    for s, c in ((1, BLUE), (2, "#5B9BD5")):
        ax.plot([p[0] for p in baseline[s]], [p[1] for p in baseline[s]], color=c, lw=2, label=f"Independent PPO, seed {s}")
    ax.set(xlabel="Training steps (million)", ylabel="Deliveries per episode", xlim=(0, 5.05), ylim=(0, None))
    ax.legend(fontsize=legend_size, loc="upper left", frameon=False)


paper, slides = ROOT / "paper/figures", ROOT / "presentation/figures"
paper.mkdir(parents=True, exist_ok=True)
slides.mkdir(parents=True, exist_ok=True)

style(9)
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.25), constrained_layout=True)
ema_panel(axs[0], 7)
mature_panel(axs[1])
alloc_panel(axs[2], 7)
for ax, tag in zip(axs, "abc"):
    ax.set_title(f"({tag})", fontsize=9, loc="left")
fig.savefig(paper / "discovery_log.pdf")
plt.close(fig)
fig, ax = plt.subplots(figsize=(3.35, 2.2), constrained_layout=True)
baseline_panel(ax, 7)
fig.savefig(paper / "baseline_ippo.pdf")
plt.close(fig)

style(15)
for name, draw in (("discovery_ema", lambda a: ema_panel(a, 13)), ("discovery_alloc", lambda a: alloc_panel(a, 13)),
                   ("baseline_ippo", lambda a: baseline_panel(a, 13))):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    draw(ax)
    fig.savefig(slides / f"{name}.pdf")
    fig.savefig(slides / f"{name}.png", dpi=200)
    plt.close(fig)

print(json.dumps(dict(same=same, cross=cross, share=round(same / (same + cross), 4), first_cross=first_cross,
                      last_step=last_step, baseline={s: v[-1] for s, v in baseline.items()},
                      bins=len(discovery)), indent=1))
