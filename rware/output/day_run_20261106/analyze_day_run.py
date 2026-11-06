"""Analysis of the one-day run (E100; recorded as T216/E86 on feature/PGCT_Change):
runs/day_base_1105_nv_b6_corner_indiv_comm5_pen005.

Base map 11x10, individual rewards, contact range 5, per-action distance gate (no mirrored probes), seed 1, 2M steps,
pgct_max vs ippo from the same random initial weights. Read-only.
Writes summary.json, stdout.txt and day_run.png/pdf (per-agent evaluation deliveries, gate statistics).
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # RWARE/
RUN_NAME = "runs/day_base_1105_nv_b6_corner_indiv_comm5_pen005"
# The run was made before the RWARE/ split (D131). Until scripts/migrate_local_layout.ps1 -Apply moves it,
# it stays in the repository-root runs/ folder.
RUN = next((p for p in (ROOT / RUN_NAME, ROOT.parent / RUN_NAME) if p.exists()), ROOT / RUN_NAME)
JOB = {m: RUN / f"b6_corner_indiv_comm5_pen005_acv2_amax_{m}_s1" for m in ("pgct_max", "ippo")}
TEAM = np.arange(6) % 2
EPISODES = 20
lines = []
say = lambda s="": (print(s), lines.append(s))


def jsonl(path):
    out = []
    for line in open(path, encoding="utf-8"):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def evals(job):
    rows = [r for r in jsonl(job / "metrics.jsonl") if "eval/deliveries" in r["metrics"]]
    steps = np.array([r["env_steps"] for r in rows]) / 1e6
    team = np.array([r["metrics"]["eval/deliveries"] for r in rows])
    agents = np.array([[r["metrics"].get(f"eval/agent_{k}_delivery_count", 0) / EPISODES for k in range(1, 7)]
                       for r in rows])
    return steps, team, agents


def first_reach(steps, series, level, window=3):
    """First step at which the rolling mean over `window` evaluations reaches `level`."""
    roll = np.convolve(series, np.ones(window) / window, mode="valid")
    idx = np.flatnonzero(roll >= level)
    return float(steps[idx[0] + window - 1]) if len(idx) else None


summary = {}
# ---------- learning ----------
ev = {m: evals(j) for m, j in JOB.items()}
say("== Learning (deterministic evaluation, 20 episodes x 500 steps)")
for m, (s, team, ag) in ev.items():
    last = slice(-3, None)
    say(f"{m}: evals {len(s)}, last eval {s[-1]:.2f}M, team deliveries/ep last-3 mean {team[last].mean():.2f}; "
        f"per agent last-3 {np.round(ag[last].mean(0), 2).tolist()}")
    reach = {lv: first_reach(s, team, lv) for lv in (0.5, 2, 5, 10)}
    say(f"   team first reach (3-eval mean): " + ", ".join(f">={lv}: {v if v is None else round(v, 2)}" for lv, v in reach.items()))
    per = [first_reach(s, ag[:, k], .5) for k in range(6)]
    say(f"   per agent first reach >=0.5/ep: {[None if v is None else round(v, 2) for v in per]}")
    summary[m] = dict(last3_team=float(team[last].mean()), last3_agents=ag[last].mean(0).tolist(), team_reach=reach,
                      agent_reach_05=per)

# training-time per-agent deliveries (smoother, stochastic policy)
say("\n== Training deliveries per agent and episode (stochastic policy), 0.25M bins")
for m, j in JOB.items():
    rows = [r for r in jsonl(j / "metrics.jsonl") if "train/agent_1_delivery_count" in r["metrics"]]
    out = []
    for lo in np.arange(0, 2.0e6, 2.5e5):
        seg = [r["metrics"] for r in rows if lo <= r["env_steps"] < lo + 2.5e5]
        out.append(np.round([np.mean([x[f"train/agent_{k}_delivery_count"] for x in seg]) for k in range(1, 7)], 2).tolist())
    say(f"{m}: " + " | ".join(f"{lo/1e6:.2f}M {v}" for lo, v in zip(np.arange(0, 2.0e6, 2.5e5), out)))
    summary[m]["train_agent_bins"] = out

# ---------- gate ----------
mg = jsonl(JOB["pgct_max"] / "max_gae.jsonl")
say("\n== Discovery (pgct_max)")
mature_first = {}
first_appr = {}
same_arcs = cross_arcs = 0
coverage, steps_c = [], []
dist = {}
strata = {}
for r in mg:
    st = r["env_steps"]
    for k, flag in enumerate(r.get("reward_mature", [])):
        if flag and k not in mature_first:
            mature_first[k] = st
    A = np.array(r["approved"], bool)
    for i in range(6):
        for j in range(i + 1, 6):
            if A[i, j] and (i, j) not in first_appr:
                first_appr[(i, j)] = st
    same_arcs += r.get("max_gae_approved_same_team_arcs", 0)
    cross_arcs += r.get("max_gae_approved_cross_team_arcs", 0)
    if "max_gae_same_team_approval_coverage" in r:
        coverage.append(r["max_gae_same_team_approval_coverage"]); steps_c.append(st)
    for p in r.get("ac_pairs", []):
        if p.get("distance") is None:
            continue
        key = (min(p["receiver"], p["donor"]), max(p["receiver"], p["donor"]))
        dist.setdefault(key, []).append((st, p["distance"], p.get("pooled_distance")))
        strata[p.get("stratum")] = strata.get(p.get("stratum"), 0) + 1
say(f"agents first mature: { {k: round(v / 1e6, 3) for k, v in sorted(mature_first.items())} }; "
    f"never mature: {[k for k in range(6) if k not in mature_first]}")
say(f"first approval per pair: { {f'{k}': round(v / 1e6, 3) for k, v in sorted(first_appr.items())} }")
say(f"approved arcs (directed, summed over updates): same {int(same_arcs)}, different {int(cross_arcs)}")
say(f"strata that gave the maximum (1=FORWARD, 4=TOGGLE): {strata}")
say("pair distances (first comparison / last comparison; per-action max vs pooled):")
late = {}
for key, v in sorted(dist.items()):
    rel = "same" if TEAM[key[0]] == TEAM[key[1]] else "DIFF"
    l = [x for x in v if x[0] >= 1.5e6]
    late[key] = (np.mean([x[1] for x in l]) if l else None, np.mean([x[2] for x in l]) if l else None)
    say(f"  {key} {rel}: n={len(v)} first {v[0][1]:.2f}/{v[0][2]:.2f} at {v[0][0]/1e6:.2f}M, last {v[-1][1]:.2f}/{v[-1][2]:.2f}"
        + (f", mean after 1.5M {late[key][0]:.2f}/{late[key][1]:.2f}" if late[key][0] is not None else ""))
summary["gate"] = dict(mature_first=mature_first, first_approval={f"{k}": v for k, v in first_appr.items()},
                       same_arcs=same_arcs, cross_arcs=cross_arcs, strata=strata,
                       late_mean={f"{k}": v for k, v in late.items()})

# ---------- Max-GAE ----------
say("\n== Max-GAE (pgct_max)")
bins = np.arange(0, 2.0e6, 2.5e5)
for key in ("max_gae_donor_arcs", "max_gae_receivers", "max_gae_select_frac", "max_gae_mean_b", "max_gae_eta_eff",
            "max_gae_adv_shift"):
    vals = []
    for lo in bins:
        seg = [r[key] for r in mg if lo <= r["env_steps"] < lo + 2.5e5 and key in r and r[key] == r[key]]
        vals.append(round(float(np.mean(seg)), 3) if seg else None)
    say(f"{key}: " + " ".join(str(v) for v in vals))
ev_k = [[r.get(f"max_gae_critic_ev_agent_{k}") for r in mg[-50:]] for k in range(6)]
say("critic EV per agent (mean of last 50 updates): " + str([round(float(np.nanmean([x for x in e if x is not None])), 2)
                                                             for e in ev_k]))

(HERE / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
(HERE / "stdout.txt").write_text("\n".join(lines), encoding="utf-8")

# ---------- figure ----------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "pdf.fonttype": 42})
fig, axs = plt.subplots(2, 4, figsize=(10, 4.6), constrained_layout=True)
colors = {"pgct_max": "#1F4E9A", "ippo": "#888780"}
for k in range(6):
    ax = axs.flat[k]
    for m, (s, team, ag) in ev.items():
        ax.plot(s[2:], np.convolve(ag[:, k], np.ones(3) / 3, mode="valid"), color=colors[m], lw=1.6,
                ls="-" if m == "pgct_max" else "--", label=m)
    ax.set_title(f"agent a{k} (team {'A' if TEAM[k] == 0 else 'B'})", fontsize=8, loc="left")
    ax.set_xlabel("steps (M)")
    if k % 4 == 0:
        ax.set_ylabel("eval deliveries / ep")
axs.flat[0].legend(fontsize=7, frameon=False)
ax = axs.flat[6]
for m, (s, team, ag) in ev.items():
    ax.plot(s[2:], np.convolve(team, np.ones(3) / 3, mode="valid"), color=colors[m], lw=1.6, ls="-" if m == "pgct_max" else "--")
ax.set_title("team total (3-eval mean)", fontsize=8, loc="left")
ax.set_xlabel("steps (M)")
ax = axs.flat[7]
cum_same, cum_cross, st = 0, 0, []
cs, cc = [], []
for r in mg:
    cum_same += r.get("max_gae_approved_same_team_arcs", 0); cum_cross += r.get("max_gae_approved_cross_team_arcs", 0)
    st.append(r["env_steps"] / 1e6); cs.append(cum_same); cc.append(cum_cross)
ax.plot(st, cs, color="#1F4E9A", lw=1.6, label="same team")
ax.plot(st, cc, color="#C0504D", lw=1.6, label="different team")
ax.set_title("approved arcs (pgct_max)", fontsize=8, loc="left")
ax.set_xlabel("steps (M)")
ax.legend(fontsize=7, frameon=False)
fig.savefig(HERE / "day_run.png", dpi=180)
fig.savefig(HERE / "day_run.pdf")
