"""Figures for the PGCT progress paper (2026-10-23).  Read-only: loads logs, analysis json, checkpoints and probe
banks; never trains.  Writes PDFs into figures/ next to this script.

    python output/pgct_progress_paper_20261023/make_figures.py
"""
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "latent_team"))
from examples import train_recurrent_ippo_consensus as T  # noqa: E402
from rware.utils.semantic_observation import SEMANTIC_EGO_FEATURES, SEMANTIC_SPATIAL_CHANNELS  # noqa: E402
from reward_correlation_baseline import load_model  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
RED, BLUE = "#d62728", "#1f77b4"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight"})

R = ROOT / "runs"
RUNS = {
    "B, penalty .02": R / "pilot_20261019_180044_b_e1f6_comm3/b_e1f6_comm3_ippo_s{}",
    "B, penalty 0": R / "pilot_20261021_133652_b_e1f6_comm3_nopen/b_e1f6_comm3_nopen_ippo_s{}",
    "corner/team, .005": R / "pilot_20261022_112521_b_e1f6_comm3_pen005/b_e1f6_comm3_pen005_ippo_s{}",
    "corner/indiv, .005": R / "queue_20261022_140114_b6_corner_indiv_pen005/b6_corner_indiv_pen005_ippo_s{}",
    "adjacent/team, .005": R / "queue_20261022_140114_b6_adjacent_team_pen005/b6_adjacent_team_pen005_ippo_s{}",
    "adjacent/indiv, .005": R / "queue_20261022_140114_b6_adjacent_indiv_pen005/b6_adjacent_indiv_pen005_ippo_s{}",
}
STYLE = {"B, penalty .02": ("0.55", "--"), "B, penalty 0": ("#9467bd", ":"), "corner/team, .005": (BLUE, "-"),
         "corner/indiv, .005": (RED, "-"), "adjacent/team, .005": ("#2ca02c", "-."), "adjacent/indiv, .005": ("#ff7f0e", "-.")}


def windows(run, key, end=1.0e6, width=1e5):
    rows = [json.loads(l) for l in open(run / "metrics.jsonl", encoding="utf-8")]
    tr = [r for r in rows if key in r["metrics"]]
    xs, ys = [], []
    for lo in np.arange(0, end, width):
        v = [r["metrics"][key] for r in tr if lo <= r["env_steps"] < lo + width]
        if v:
            xs.append((lo + width / 2) / 1e6)
            ys.append(float(np.mean(v)))
    return xs, ys


# ---------------------------------------------------------------- 1. PGCT overview
def fig_overview():
    fig, ax = plt.subplots(figsize=(8.4, 2.1))
    ax.set_xlim(-0.2, 10.2); ax.set_ylim(0, 2.6); ax.axis("off")
    steps = [("1  Neighbours", "agents within\ncommunication range"),
             ("2  Same probes", "both actors answer\nthe shared probe bank"),
             ("3  Gate", "smoothed JS < tau\n=> treat as teammate"),
             ("4  Distil critic", "pull V_i toward V_j\non probe inputs")]
    for k, (title, body) in enumerate(steps):
        x = 0.0 + k * 2.6
        ax.add_patch(FancyBboxPatch((x, 0.45), 2.25, 1.75, boxstyle="round,pad=0.04", fc="#f3f3f3", ec="0.3"))
        ax.text(x + 1.125, 1.85, title, ha="center", va="center", weight="bold")
        ax.text(x + 1.125, 1.05, body, ha="center", va="center", fontsize=7.5)
        if k < 3:
            ax.add_patch(FancyArrowPatch((x + 2.3, 1.32), (x + 2.55, 1.32), arrowstyle="->", mutation_scale=12))
    ax.text(5, 0.12, "exchanged: network outputs on probes only (no observations, rewards, parameters or team labels); "
            "the actor learns from its own experience", ha="center", fontsize=7, color="0.3")
    fig.savefig(OUT / "fig_overview.pdf"); plt.close(fig)


# ---------------------------------------------------------------- 2. environment 2x2
def env_state(run_dir, seed=5000001):
    ck = sorted(run_dir.glob("checkpoint_*.pt"))[0]
    c = T._load_checkpoint(str(ck), torch.device("cpu"))
    kw = dict(c["effective_env_kwargs"])
    with T.make_env(c["config"]["env_id"], kw, c["config"]["observation_format"]) as env:
        env.reset(seed=seed)
        u = env.unwrapped
        H, W = map(int, u.grid_size)
        req = {int(s.id): t for t, q in enumerate(u.team_request_queues) for s in q}
        return dict(H=H, W=W, shelves=[(s.x, s.y, s.id) for s in u.shelfs], req=req,
                    agents=[(a.x, a.y) for a in u.agents], teams=[int(t) for t in np.asarray(u.agent_team_ids)],
                    goals=[tuple(g[0]) for g in kw["team_goal_positions"]], reward=kw["team_reward_mode"],
                    sensor=int(kw.get("sensor_range", 5)))


def draw_map(ax, st, title, event=None):
    H, W = st["H"], st["W"]
    ax.add_patch(Rectangle((-0.5, -0.5), W, H, fc="#f7f7f7", ec="0.6"))
    for x, y, sid in st["shelves"]:
        t = st["req"].get(sid)
        fc = "#c8c8c8" if t is None else (RED if t == 0 else BLUE)
        ax.add_patch(Rectangle((x - 0.42, y - 0.42), 0.84, 0.84, fc=fc, ec="0.4", lw=0.4, alpha=0.9 if t is not None else 1))
    for t, (gx, gy) in enumerate(st["goals"]):
        ax.add_patch(Rectangle((gx - 0.48, gy - 0.48), 0.96, 0.96, fc="none", ec=RED if t == 0 else BLUE, lw=2.2))
        ax.text(gx, gy, "G", ha="center", va="center", color=RED if t == 0 else BLUE, weight="bold", fontsize=8)
    for k, ((x, y), t) in enumerate(zip(st["agents"], st["teams"])):
        ax.add_patch(Circle((x, y), 0.36, fc="white", ec=RED if t == 0 else BLUE, lw=1.6))
        ax.text(x, y, str(k + 1), ha="center", va="center", fontsize=6.5)
    if event is not None:
        (ex, ey), (ox, oy) = event
        r = st["sensor"]
        ax.add_patch(Rectangle((ox - r - 0.5, oy - r - 0.5), 2 * r + 1, 2 * r + 1, fc="none", ec="0.2", ls="--", lw=0.9))
        ax.plot([ox], [oy], marker="x", color="0.2", ms=6)
        ax.text(ox + 0.3, oy - 0.3, "observer i\n(sensor window)", fontsize=6.5, color="0.2")
        ax.annotate("event $e_g$:\ndelivery at goal", (ex, ey), (ex + 2.6, ey - 3.2), fontsize=7,
                    arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlim(-0.6, W - 0.4); ax.set_ylim(H - 0.4, -0.6); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8.5)


def fig_env():
    cells = [("adjacent/indiv, .005", "adjacent goals, individual reward"), ("adjacent/team, .005", "adjacent goals, team reward"),
             ("corner/indiv, .005", "corner goals, individual reward"), ("corner/team, .005", "corner goals, team reward")]
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 7.2))
    for ax, (key, title) in zip(axes.ravel(), cells):
        st = env_state(Path(str(RUNS[key]).format(1)))
        event = None
        if key == "corner/team, .005":
            g = st["goals"][0]
            event = (g, (g[0] + 2, g[1] - 3))
        draw_map(ax, st, title, event)
    fig.text(0.5, 0.005, "Base: 11x10 map, 6 agents (red 1,3,5 / blue 2,4,6), sensor 5, communication 3, pickup penalties .005.\n"
             "Coloured shelves = currently requested; G = team goal (sides swap on even seeds).", ha="center", fontsize=7)
    fig.savefig(OUT / "fig_env_2x2.pdf"); plt.close(fig)


# ---------------------------------------------------------------- 3. probes
CH = {n: k for k, n in enumerate(SEMANTIC_SPATIAL_CHANNELS)}
EGO = len(SEMANTIC_EGO_FEATURES)


def decode(vec):
    v = np.asarray(vec, dtype=np.float32)
    n = len(SEMANTIC_SPATIAL_CHANNELS)
    side = int(round(((len(v) - EGO) / n) ** 0.5))
    g = v[EGO:].reshape(n, side, side)
    img = np.ones((side, side, 3)) * 0.97
    layers = [("storage_cell", (0.88, 0.88, 0.88)), ("shelf", (0.55, 0.55, 0.55)), ("requested_shelf", (1.0, 0.65, 0.0)),
              ("goal", (0.75, 0.75, 1.0)), ("relevant_goal", (0.1, 0.7, 0.2)), ("out_of_bounds", (0.15, 0.15, 0.15))]
    for name, col in layers:
        m = g[CH[name]] > 0.5
        img[m] = col
    agents = np.argwhere(g[CH["agent"]] > 0.5)
    return img, agents, side


def draw_probe(ax, vec, title):
    img, agents, side = decode(vec)
    ax.imshow(img, interpolation="nearest")
    c = side // 2
    ax.add_patch(Rectangle((c - 0.5, c - 0.5), 1, 1, fc="none", ec=RED, lw=1.6))
    for y, x in agents:
        if (y, x) != (c, c):
            ax.add_patch(Circle((x, y), 0.32, fc="white", ec="0.2", lw=0.8))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=7)


def fig_probes():
    bank = torch.load(ROOT / "output/env_view_20261015/probes_v2/probe_bank_v2_e1f_s1.pt", map_location="cpu", weights_only=False)
    gate, trans = bank["sequences"][:, 0], bank["transfer_sequences"][:, 0]
    gsc, tsc = bank["gate_scenarios"], bank["transfer_scenarios"]
    want_t = ["search", "carry_goal_visible", "return", "at_goal"]
    t_idx = [next(i for i, s in enumerate(tsc) if s["phase"] == p) for p in want_t]
    g_idx = [0, 5, 12, 30]
    # learned agents (the dominant agent of each 1M run) on the gate probes
    learned = [("corner/team s1", R / "pilot_20261022_112521_b_e1f6_comm3_pen005/b_e1f6_comm3_pen005_ippo_s1", 1),
               ("corner/team s2", R / "pilot_20261022_112521_b_e1f6_comm3_pen005/b_e1f6_comm3_pen005_ippo_s2", 4),
               ("corner/indiv s1", R / "queue_20261022_140114_b6_corner_indiv_pen005/b6_corner_indiv_pen005_ippo_s1", 2),
               ("corner/indiv s2", R / "queue_20261022_140114_b6_corner_indiv_pen005/b6_corner_indiv_pen005_ippo_s2", 0)]
    goals = ["(9,10)", "(9,10)", "(0,10)", "(9,10)"]
    probs = []
    with torch.inference_mode():
        for _, run, dom in learned:
            net = load_model(run / "final.pt")[0][dom]
            x = gate[g_idx].float()
            h, _ = net.initial_hidden(len(x), torch.device("cpu"))
            logits, _ = net._actor_step(x, h)
            probs.append(logits.softmax(-1).numpy())
    fig = plt.figure(figsize=(7.2, 5.4))
    for k, i in enumerate(g_idx):
        s = gsc[i]
        draw_probe(fig.add_subplot(3, 4, k + 1), gate[i], f"gate #{i}: carrying, goal hidden\nat ({s['x']},{s['y']}) dir {s['direction']}")
    for k, i in enumerate(t_idx):
        s = tsc[i]
        draw_probe(fig.add_subplot(3, 4, 5 + k), trans[i], f"distil #{i}: {s['phase'].replace('_', ' ')}\nat ({s['x']},{s['y']})")
    names = ["NOOP", "FWD", "LEFT", "RIGHT", "TOGGLE"]
    for k in range(4):
        ax = fig.add_subplot(3, 4, 9 + k)
        for a, (lab, _, _) in enumerate(learned):
            ax.bar(np.arange(5) + (a - 1.5) * 0.2, probs[a][k], 0.2, label=f"{lab}, goal {goals[a]}",
                   color=[BLUE, "#6baed6", RED, "#9ecae1"][a])
        ax.set_xticks(range(5)); ax.set_xticklabels(names, fontsize=6, rotation=45); ax.set_ylim(0, 1)
        ax.set_title(f"responses on gate #{g_idx[k]}", fontsize=7)
        if k == 0:
            ax.set_ylabel("action prob.")
    fig.legend(*fig.axes[-1].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=6.5, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.text(0.5, 1.0, "Self = red box (centre); orange = MY requested shelf; green = MY goal; light blue = other goal; "
             "dark = outside map", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(OUT / "fig_probes.pdf"); plt.close(fig)


# ---------------------------------------------------------------- 4. pickup stall and penalty
def fig_stall():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for key in ["B, penalty .02", "B, penalty 0", "corner/team, .005"]:
        col, ls = STYLE[key]
        for s in (1, 2):
            xs, ys = windows(Path(str(RUNS[key]).format(s)), "train/toggle_count")
            a.plot(xs, ys, color=col, ls=ls, marker="o", ms=2.5, label=key if s == 1 else None)
    a.set_yscale("log"); a.set_xlabel("env steps (M)"); a.set_ylabel("TOGGLE per episode"); a.legend(fontsize=7, frameon=False)
    a.set_title("(a) pickups collapse at .02 (team reward, seeds 1-2)", fontsize=8.5)
    sw = json.loads((ROOT / "output/penalty_sweep_20261022/results.json").read_text(encoding="utf-8"))
    ps = [0.0, 0.005, 0.01, 0.015, 0.02]
    for act, col in [("TOGGLE", RED), ("NOOP", "0.4"), ("LEFT", "0.65"), ("FORWARD", "0.85")]:
        b.plot(ps, [sw[f"A p={p}"]["immediate_by_action"][act]["mean"] for p in ps], marker="o", ms=3, color=col, label=act)
    b.axvline(0.0058, color="0.3", ls=":", lw=1); b.text(0.0061, -0.016, "crossing\n~.0056-.0059", fontsize=7)
    b.set_xlabel("pickup penalty p"); b.set_ylabel("mean immediate reward"); b.legend(fontsize=7, frameon=False)
    b.set_title("(b) uniform random policy, action value vs p", fontsize=8.5)
    fig.tight_layout(); fig.savefig(OUT / "fig_stall.pdf"); plt.close(fig)


# ---------------------------------------------------------------- 5. learning in the first 1M and dominance
def fig_learning():
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.4, 2.6), gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
    for key, (col, ls) in STYLE.items():
        for s in (1, 2):
            xs, ys = windows(Path(str(RUNS[key]).format(s)), "train/deliveries")
            a.plot(xs, ys, color=col, ls=ls, lw=1.2, label=key if s == 1 else None)
    a.set_xlabel("env steps (M)"); a.set_ylabel("training deliveries / episode"); a.legend(fontsize=6, frameon=False)
    a.set_title("(a) IPPO, first 1M (adjacent: in progress)", fontsize=8)
    dd = json.loads((ROOT / "output/pen005_analysis_20261023/dominance_diag.json").read_text(encoding="utf-8"))["absolute"]
    for ax, name, title in [(b, "corner_team s1", "(b) corner/team s1"), (c, "corner_indiv s1", "(c) corner/indiv s1")]:
        w = dd[name]["windows"]
        x = np.arange(len(w)) / 10 + 0.05
        bottom = np.zeros(len(w))
        cols = [RED, BLUE, "#ff9896", "#aec7e8", "#ffbb78", "#98df8a"]
        for k in range(6):
            vals = np.array([v["per_agent"][k] for v in w])
            ax.bar(x, vals, 0.09, bottom=bottom, color=cols[k], label=f"agent {k + 1}")
            bottom += vals
        ax.set_xlabel("env steps (M)"); ax.set_title(title + f" (dominant: agent {dd[name]['dominant'] + 1})", fontsize=8)
        if ax is b:
            ax.set_ylabel("deliveries per 0.1M window")
    c.legend(fontsize=6, frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "fig_learning_1m.pdf"); plt.close(fig)


# ---------------------------------------------------------------- 6. identification
def fig_identification():
    rc = json.loads((ROOT / "output/reward_coupling_20261022/results.json").read_text(encoding="utf-8"))
    pt = json.loads((ROOT / "output/pen005_analysis_20261023/results.json").read_text(encoding="utf-8"))
    pi = json.loads((ROOT / "output/pen005_analysis_20261023/results_corner_indiv.json").read_text(encoding="utf-8"))
    groups = {"B team, pen 0\n(0.77M,1M)": [rc[k]["scores"] for k in rc],
              "corner/team .005\n(0.77M,1M)": [v["scores"] for v in pt["auroc"].values()],
              "corner/indiv .005\n(0.77M,1M)": [v["scores"] for v in pi["auroc"].values()]}
    sigs = [("coupling_probe", "reward coupling", "#2ca02c"), ("policy_canonical", "policy, canonical", "0.35"),
            ("policy_v2_gate", "policy, v2 gate", "0.6"), ("policy_visited", "policy, visited states", "0.8")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 2.8), gridspec_kw=dict(width_ratios=[1.25, 1]))
    for gi, (gname, scores) in enumerate(groups.items()):
        for si, (key, lab, col) in enumerate(sigs):
            v = [s[key]["auroc"] for s in scores if s[key]["auroc"] is not None]
            m = float(np.mean(v))
            a.bar(gi + (si - 1.5) * 0.2, m, 0.2, color=col, label=lab if gi == 0 else None,
                  yerr=[[m - min(v)], [max(v) - m]], capsize=2, error_kw=dict(lw=0.7))
    a.axhline(0.5, color="0.4", ls=":", lw=0.8); a.set_xticks(range(3)); a.set_xticklabels(groups, fontsize=7)
    a.set_ylabel("team AUROC (same team closer)"); a.set_ylim(0, 1.05); a.legend(fontsize=6.5, frameon=False, ncol=2, loc="lower left")
    a.set_title("(a) mean over 2 seeds x 2 checkpoints (bars = min-max)", fontsize=8)
    for res, col, lab in [(pt, BLUE, "corner/team"), (pi, RED, "corner/indiv")]:
        for s in (1, 2):
            m = res["identification"][f"s{s} final.pt K=1"]["marks"]
            xs = sorted(int(k) for k in m)
            b.plot(np.array(xs) / 1e3, [m[str(x)]["correct"] for x in xs], color=col, marker="o", ms=3,
                   label=f"{lab}: correct" if s == 1 else None)
            b.plot(np.array(xs) / 1e3, [m[str(x)]["wrong"] for x in xs], color=col, ls="--", marker="x", ms=3,
                   label=f"{lab}: wrong" if s == 1 else None)
    b.set_xscale("log"); b.set_xlabel("env steps since start of online judging (k)"); b.set_ylabel("pairs (of 15)")
    b.set_ylim(-0.5, 15.5); b.legend(fontsize=6.5, frameon=False)
    b.set_title("(b) online reward-coupling judging, 1M policy, K=1", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_identification.pdf"); plt.close(fig)


# ---------------------------------------------------------------- 7. dominant-agent removal
def fig_removal():
    rem = json.loads((ROOT / "output/pen005_analysis_20261023/dominance_diag.json").read_text(encoding="utf-8"))["removal"]
    runs = ["corner_team s1", "corner_team s2", "corner_indiv s1", "corner_indiv s2"]
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    for k, cond in enumerate(["normal", "noop", "parked"]):
        others = [rem[f"{r} stoch {cond}"]["teammates"] + rem[f"{r} stoch {cond}"]["other_team"] for r in runs]
        ax.bar(np.arange(4) + (k - 1) * 0.25, others, 0.25, label=f"others, dominant {cond}", color=["0.3", "0.6", "0.85"][k])
    dom = [rem[f"{r} stoch normal"]["dominant"] for r in runs]
    ax.scatter(np.arange(4) - 0.25, dom, marker="*", s=60, color=RED, zorder=3, label="dominant agent itself (normal)")
    for k in (2, 3):
        ax.text(k, 0.04, "others = 0\nin all three", ha="center", fontsize=6.5, color="0.3")
    ax.set_xticks(range(4)); ax.set_xticklabels(runs, fontsize=7.5); ax.set_ylabel("deliveries / episode")
    ax.legend(fontsize=6.5, frameon=False, ncol=2); ax.set_title("1M policies, stochastic, 20 episodes", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_removal.pdf"); plt.close(fig)


# ---------------------------------------------------------------- 8. plan
def fig_plan():
    d = lambda s: dt.date.fromisoformat(s)
    rows = [("2x2 scan: adjacent cells (IPPO 1M)", "2026-10-22", "2026-10-24", "0.45", "running"),
            ("Main stage 1, wave 1 (4 jobs, 5M)", "2026-10-23", "2026-10-28", BLUE, "running/queued"),
            ("Main stage 1, wave 2 (4 jobs, 5M)", "2026-10-28", "2026-11-01", BLUE, "queued"),
            ("Main stage 1, wave 3 (1 job, 5M)", "2026-11-01", "2026-11-05", BLUE, "queued"),
            ("PGCT-reward + resume wrapper (no GPU)", "2026-10-24", "2026-10-31", "#2ca02c", "planned"),
            ("Stage 2: PGCT-reward / Random-gated / PGCT-policy", "2026-11-05", "2026-11-16", "#9467bd", "planned"),
            ("Corner/indiv cell 5M (E1-F role)", "2026-11-16", "2026-11-27", RED, "planned"),
            ("Second environment (LBF variant) or seeds 4-5", "2026-11-20", "2026-12-12", "#ff7f0e", "planned"),
            ("Writing", "2026-11-01", "2027-01-04", "0.7", "planned")]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    for k, (lab, s, e, col, st) in enumerate(rows):
        ax.barh(k, (d(e) - d(s)).days, left=mdates.date2num(d(s)), color=col, alpha=0.9 if st != "planned" else 0.55,
                hatch=None if st != "planned" else "//", edgecolor="0.3", lw=0.5)
        ax.text(mdates.date2num(d(e)) + 1, k, st, va="center", fontsize=6.5, color="0.3")
    for lab, s in [("today", "2026-10-23"), ("abstract", "2027-01-04"), ("paper", "2027-01-11")]:
        ax.axvline(mdates.date2num(d(s)), color=RED if lab != "today" else "0.2", ls=":", lw=1)
        ax.text(mdates.date2num(d(s)), -0.75, lab, fontsize=6.5, ha="center", va="bottom")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=7); ax.invert_yaxis()
    ax.xaxis_date(); ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d")); ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout(); fig.savefig(OUT / "fig_plan.pdf"); plt.close(fig)


def main():
    for name in ["fig1_learning_curves.pdf", "fig4_probe_js_gap.pdf", "fig3_critic_transfer.pdf"]:
        shutil.copy(ROOT / "output/ijcai_method_paper_20261018/figures" / name, OUT / ("e1_" + name))
    for f in (fig_overview, fig_env, fig_probes, fig_stall, fig_learning, fig_identification, fig_removal, fig_plan):
        f(); print("ok", f.__name__, flush=True)


if __name__ == "__main__":
    main()
