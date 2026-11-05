"""Paper materials (figures, LaTeX tables, numbers) from the E1 from-scratch run stopped at ~6.5M steps.

Read-only: reads runs/scratch_20261015_093349_stage2 (and E17 for the reproducibility appendix) and writes
only this folder.  All statistics are from ONE training seed; captions and tables say so.

    python output/paper_e1_20261018/make_paper_materials.py
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
from latent_team import common as C  # noqa: E402

RUN = ROOT / "runs/scratch_20261015_093349_stage2"
E17 = ROOT / "runs/scratch_20261013_202352_stage2"
OUT = Path(__file__).resolve().parent
FIG, TAB = OUT / "figures", OUT / "tables"
MODES = ["ippo", "pgct", "oracle", "all"]
LABEL = dict(ippo="IPPO", pgct="PGCT", oracle="Oracle-team", all="All-peer")
# Categorical slots 1-4 of the reference palette; identity is also carried by line style and marker.
COLOR = dict(ippo="#2a78d6", pgct="#eb6834", oracle="#1baf7a", all="#eda100")
STYLE = dict(ippo="-", pgct="--", oracle="-.", all=":")
MARK = dict(ippo="o", pgct="s", oracle="^", all="D")
TEAM_COLOR = ["#e34948", "#2a78d6"]           # team 0 (agents 1, 3) red, team 1 (agents 2, 4) blue
TEAM = np.array([0, 1, 0, 1])
SAME_PAIRS = [(0, 2), (1, 3)]
ALL_PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]
COL_W, TEXT_W = 3.3, 6.9                     # IJCAI column / text width (inches)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
    "grid.linewidth": .5, "axes.edgecolor": "#888888", "lines.linewidth": 1.4, "pdf.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": .02})


def smooth(y, w=5):
    """Centered moving average; edges use the available points."""
    y = np.asarray(y, float)
    k = w // 2
    return np.asarray([np.nanmean(y[max(0, i - k):i + k + 1]) for i in range(len(y))])


def onset(steps, y, level, w=3):
    sm = np.convolve(y, np.ones(w) / w, mode="valid")
    hit = np.asarray(steps)[w - 1:][sm >= level]
    return int(hit[0]) if len(hit) else None


def load_eval(job_dir):
    steps, evals = C.evaluation_curve(job_dir)
    agent = lambda key: np.asarray([[e.get(f"agent_{i}_{key}", np.nan) / 20 for i in range(1, 5)] for e in evals])
    return dict(steps=np.asarray(steps, float), deliveries=np.asarray([e["deliveries"] for e in evals], float),
                returns=np.asarray([e["returns_home"] for e in evals], float),
                pickups=np.asarray([e["requested_shelf_pickups"] for e in evals], float),
                agent_deliveries=agent("delivery_count"), agent_pickups=agent("pickup_count"))


def load_transfer(job_dir):
    rows = C.read_jsonl(job_dir / "transfer.jsonl")
    steps = np.asarray([r["env_steps"] for r in rows], float)
    alloc = np.asarray([r["actual_allocation"] for r in rows]) > 0      # receiver x donor
    held = np.asarray([r["warmup_hold"] for r in rows])
    return steps, alloc, held


@torch.inference_mode()
def probe_js(job_dir, every=500):
    from examples.diagnose_pgct_rollout_compatibility import load_model
    from examples.pgct_probe_diagnostics import common_probe_response
    bank = torch.load(RUN / "canonical_bank_e1_s1.pt", map_location="cpu", weights_only=False)["sequences"].float()
    rows = []
    for ck in sorted(job_dir.glob("checkpoint_*.pt"), key=lambda p: int(p.stem.split("_")[1])):
        if int(ck.stem.split("_")[1]) % every:
            continue
        agents, meta = load_model(ck)
        m = np.asarray(common_probe_response(agents, bank, torch.device("cpu"))["mean_js"])
        same = float(np.mean([m[i, j] for i, j in SAME_PAIRS]))
        cross = float(np.mean([m[i, j] for i, j in ALL_PAIRS if (i, j) not in SAME_PAIRS]))
        rows.append(dict(env_steps=int(meta["env_steps"]), same=same, cross=cross, matrix=m.tolist()))
    return rows


def fmt_m(step):
    return "--" if step is None else f"{step / 1e6:.2f}"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    ev = {m: load_eval(RUN / f"e1_{m}_s1") for m in MODES}
    old = {m: load_eval(E17 / f"e1_{m}_s1") for m in MODES}
    tr = {m: load_transfer(RUN / f"e1_{m}_s1") for m in MODES if m != "ippo"}
    H = min(e["steps"][-1] for e in ev.values())               # common evaluated horizon
    numbers = dict(run=str(RUN.relative_to(ROOT)), horizon=H, seed=1, note=(
        "Single training seed; run stopped by the user at ~6.5M of a planned 15M. Evaluation every 51,200 steps: "
        "20 episodes x 500 steps, deterministic masked policy, reset seeds 5,000,001-5,000,020."), modes={})

    # ---------- Table 1: performance ----------
    for m in MODES:
        e = ev[m]
        k = e["steps"] <= H
        s, d = e["steps"][k], e["deliveries"][k]
        last = d[-5:]
        ad = e["agent_deliveries"][k][-5:].mean(0)
        learned_at = []
        for a in range(4):
            first = onset(s, e["agent_deliveries"][k][:, a], 1.0)
            learned_at.append(first)
        numbers["modes"][m] = dict(
            onset_ge1=onset(s, d, 1), onset_ge10=onset(s, d, 10), onset_ge20=onset(s, d, 20),
            mean_deliveries_over_training=float(np.trapezoid(d, s) / (s[-1] - s[0])),
            window_means={f"{lo/1e6:.0f}-{hi/1e6:.1f}M": float(d[(s >= lo) & (s <= hi)].mean())
                          for lo, hi in ((0, 2e6), (2e6, 4e6), (4e6, H))},
            final_mean=float(last.mean()), final_sd=float(last.std(ddof=1)),
            final_agent_deliveries=ad.round(2).tolist(), agents_learned=int((ad >= 1).sum()),
            agent_onset_ge1=learned_at,
            final_returns=float(e["returns"][k][-5:].mean()), final_pickups=float(e["pickups"][k][-5:].mean()))
    t1 = [r"\begin{tabular}{lcccccc}", r"\toprule",
          r"Method & \multicolumn{3}{c}{Onset (M steps)} & Mean deliv. & Final deliv. & Agents \\",
          r"\cmidrule(lr){2-4}", r" & $\geq$1 & $\geq$10 & $\geq$20 & 0--%.1fM & (last 5 evals) & learned \\" % (H / 1e6),
          r"\midrule"]
    for m in MODES:
        n = numbers["modes"][m]
        t1.append(f"{LABEL[m]} & {fmt_m(n['onset_ge1'])} & {fmt_m(n['onset_ge10'])} & {fmt_m(n['onset_ge20'])} & "
                  f"{n['mean_deliveries_over_training']:.1f} & {n['final_mean']:.1f} $\\pm$ {n['final_sd']:.1f} & "
                  f"{n['agents_learned']}/4 \\\\")
    t1 += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "table1_performance.tex").write_text("\n".join(t1) + "\n", encoding="utf-8")

    # ---------- Table 2: actual critic transfer ----------
    windows = [(0, 2e6), (2e6, 4e6), (4e6, H)]
    t2 = [r"\begin{tabular}{llccc}", r"\toprule",
          r"Method & Window & Updates w/ transfer & Same-team arcs/upd. & Cross-team arcs/upd. \\", r"\midrule"]
    for m in ("pgct", "oracle", "all"):
        steps, alloc, held = tr[m]
        numbers["modes"][m]["transfer"] = {}
        for lo, hi in windows:
            k = (steps >= lo) & (steps < hi) & ~held
            a = alloc[k]
            same = np.zeros(len(a))
            cross = np.zeros(len(a))
            for i in range(4):
                for j in range(4):
                    if i != j:
                        (same if TEAM[i] == TEAM[j] else cross)[:] += a[:, i, j]
            frac = float(a.reshape(len(a), -1).any(1).mean())
            pairs = {f"{i+1}-{j+1}": float((a[:, i, j] | a[:, j, i]).mean()) for i, j in ALL_PAIRS}
            label = f"{lo/1e6:.0f}--{hi/1e6:.1f}M"
            numbers["modes"][m]["transfer"][label] = dict(updates=int(k.sum()), fraction_with_transfer=frac,
                                                           same_arcs_per_update=float(same.mean()),
                                                           cross_arcs_per_update=float(cross.mean()), pair_share=pairs)
            t2.append(f"{LABEL[m] if lo == 0 else ''} & {label} & {100 * frac:.0f}\\% & {same.mean():.2f} & "
                      f"{cross.mean():.2f} \\\\")
        if m != "all":
            t2.append(r"\addlinespace")
    t2 += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "table2_transfer.tex").write_text("\n".join(t2) + "\n", encoding="utf-8")

    # ---------- Probe JS (policy similarity used by the PGCT gate) ----------
    js = {m: probe_js(RUN / f"e1_{m}_s1") for m in MODES}
    numbers["probe_js"] = js
    t3 = [r"\begin{tabular}{l" + "c" * 4 + "}", r"\toprule",
          r"Method & \multicolumn{4}{c}{JS same-team / cross-team at} \\", r"\cmidrule(lr){2-5}"]
    marks = [1e6, 2e6, 4e6, 6e6]
    pick = lambda rows, t: min(rows, key=lambda r: abs(r["env_steps"] - t))
    t3.append(" & " + " & ".join(f"{t/1e6:.0f}M" for t in marks) + r" \\")
    t3.append(r"\midrule")
    for m in MODES:
        t3.append(LABEL[m] + " & " + " & ".join(f"{pick(js[m], t)['same']:.2f} / {pick(js[m], t)['cross']:.2f}"
                                                for t in marks) + r" \\")
    t3 += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "table3_probe_js.tex").write_text("\n".join(t3) + "\n", encoding="utf-8")

    # ---------- Reproducibility vs E17 (same seed and initial weights) ----------
    numbers["reproducibility_vs_E17"] = {}
    for m in MODES:
        n = min(len(old[m]["steps"]), len(ev[m]["steps"]))
        diff = np.abs(old[m]["deliveries"][:n] - ev[m]["deliveries"][:n]) > 1e-9
        numbers["reproducibility_vs_E17"][m] = dict(
            compared_evals=int(n), identical=bool(not diff.any()),
            first_divergent_step=int(ev[m]["steps"][:n][diff][0]) if diff.any() else None,
            onset_ge1_E17=onset(old[m]["steps"], old[m]["deliveries"], 1), onset_ge1_this=numbers["modes"][m]["onset_ge1"])

    # ---------- Table A1: hyperparameters ----------
    cfg = json.loads((RUN / "experiment.json").read_text(encoding="utf-8"))["jobs"][1]["config"]
    env = json.loads(cfg["env_kwargs_json"])
    rows = [("Map / agents / teams", "11$\\times$10 E1 / 4 / 2"), ("Reward", env["team_reward_mode"]),
            ("Home-zone shelf ratio", env["shelf_soft_zone_ratio"]), ("Requests per team", env["request_queue_size_per_team"]),
            ("Sensor / communication range", f"{env['sensor_range']} / {env['communication_range']}"),
            ("Learning rate", cfg["learning_rate"]), ("Entropy coef.", cfg["entropy_coef"]),
            ("$\\gamma$ / GAE $\\lambda$", f"{cfg['gamma']} / {cfg['gae_lambda']}"),
            ("Rollout / BPTT length", f"{cfg['rollout_steps']} / {cfg['sequence_length']}"),
            ("PPO epochs / minibatches / clip", f"{cfg['ppo_epochs']} / {cfg['minibatch_chunks']} / {cfg['clip_coef']}"),
            ("Hidden (MLP / GRU)", f"{cfg['mlp_hidden_dim']} / {cfg['recurrent_hidden_dim']}"),
            ("Peer-loss coef. $\\lambda_{peer}$", cfg["pgct_peer_loss_coef"]),
            ("Gate: JS threshold / EMA $\\beta$ / warm-up", f"{cfg['pgct_distance_threshold']} / {cfg['pgct_distance_ema_beta']} / {cfg['pgct_warmup_updates']} upd."),
            ("Canonical probes", f"{cfg['policy_probe_batch_size']} (fixed, zero hidden, length {cfg['pgct_probe_sequence_length']})"),
            ("Evaluation", f"every {cfg['eval_interval']} updates, {cfg['eval_episodes']}$\\times${cfg['eval_horizon']} steps, deterministic")]
    ta = [r"\begin{tabular}{ll}", r"\toprule", r"Setting & Value \\", r"\midrule"] + \
         [f"{a} & {b} \\\\" for a, b in rows] + [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tableA1_hyperparameters.tex").write_text("\n".join(ta) + "\n", encoding="utf-8")

    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=2), encoding="utf-8")
    figures(ev, old, tr, js, H, numbers)
    print(json.dumps({m: dict(final=round(numbers["modes"][m]["final_mean"], 1),
                              learned=numbers["modes"][m]["agents_learned"],
                              mean=round(numbers["modes"][m]["mean_deliveries_over_training"], 2)) for m in MODES}))


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)


def figures(ev, old, tr, js, H, numbers):
    # Fig. 1: learning curves.
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    for m in MODES:
        e = ev[m]
        k = e["steps"] <= H
        x = e["steps"][k] / 1e6
        ax.plot(x, e["deliveries"][k], color=COLOR[m], lw=.6, alpha=.25)
        y = smooth(e["deliveries"][k])
        ax.plot(x, y, color=COLOR[m], ls=STYLE[m], label=LABEL[m], marker=MARK[m], markevery=12, ms=3.5)
        ax.annotate(LABEL[m], (x[-1], y[-1]), xytext=(3, 0), textcoords="offset points", fontsize=7,
                    va="center", color="#333333")
    ax.set_xlabel("Environment steps (M)")
    ax.set_ylabel("Deliveries per episode")
    ax.set_xlim(0, H / 1e6 * 1.22)
    ax.set_xticks(range(0, int(H / 1e6) + 1))
    ax.legend(loc="upper left", frameon=False)
    save(fig, "fig1_learning_curves")

    # Fig. 2: per-agent deliveries.
    fig, axes = plt.subplots(1, 4, figsize=(TEXT_W, 1.75), sharey=True)
    for ax, m in zip(axes, MODES):
        e = ev[m]
        k = e["steps"] <= H
        for a in range(4):
            ax.plot(e["steps"][k] / 1e6, smooth(e["agent_deliveries"][k][:, a]), color=TEAM_COLOR[TEAM[a]],
                    ls="-" if a < 2 else "--", label=f"Agent {a + 1} ({'red' if TEAM[a] == 0 else 'blue'})")
        ax.set_title(f"{LABEL[m]} ({numbers['modes'][m]['agents_learned']}/4 agents deliver)")
        ax.set_xlabel("Env. steps (M)")
    axes[0].set_ylabel("Deliveries per episode")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.08))
    fig.tight_layout(w_pad=.6)
    save(fig, "fig2_per_agent_deliveries")

    # Fig. 3: actual critic transfer.
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 1.9), gridspec_kw=dict(width_ratios=[1, 1.15]))
    bins = np.arange(0, H + 1, 5e5)
    for m in ("pgct", "oracle", "all"):
        steps, alloc, held = tr[m]
        same_t = np.zeros(len(alloc))
        cross_t = np.zeros(len(alloc))
        for i in range(4):
            for j in range(4):
                if i != j:
                    (same_t if TEAM[i] == TEAM[j] else cross_t)[:] += alloc[:, i, j]
        idx = np.digitize(steps, bins) - 1
        centres, s_m, c_m = [], [], []
        for b in range(len(bins) - 1):
            k = (idx == b) & ~held
            if k.any():
                centres.append((bins[b] + 2.5e5) / 1e6)
                s_m.append(same_t[k].mean())
                c_m.append(cross_t[k].mean())
        axes[0].plot(centres, s_m, color=COLOR[m], ls="-", marker=MARK[m], ms=3, label=f"{LABEL[m]} same-team")
        axes[0].plot(centres, c_m, color=COLOR[m], ls=":", marker=MARK[m], ms=3, mfc="white",
                     label=f"{LABEL[m]} cross-team")
    axes[0].set_xlabel("Environment steps (M)")
    axes[0].set_ylabel("Transfer arcs per update")
    axes[0].legend(frameon=False, ncol=3, fontsize=6, loc="lower left", bbox_to_anchor=(0, 1.0),
                   columnspacing=.8, handlelength=2.2)
    steps, alloc, held = tr["pgct"]
    idx = np.digitize(steps, bins) - 1
    order = [(0, 2), (1, 3), (0, 1), (0, 3), (1, 2), (2, 3)]
    grid = np.full((len(order), len(bins) - 1), np.nan)
    for b in range(len(bins) - 1):
        k = (idx == b) & ~held
        if k.any():
            for r, (i, j) in enumerate(order):
                grid[r, b] = (alloc[k, i, j] | alloc[k, j, i]).mean()
    im = axes[1].imshow(grid, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                        extent=(0, (len(bins) - 1) * .5, len(order) - .5, -.5))
    axes[1].set_yticks(range(len(order)))
    axes[1].set_yticklabels([f"{i+1}-{j+1} ({'same' if TEAM[i] == TEAM[j] else 'cross'})" for i, j in order])
    axes[1].axhline(1.5, color="white", lw=2)
    axes[1].set_xlabel("Environment steps (M)")
    axes[1].set_title("PGCT: share of updates with transfer, per pair")
    axes[1].grid(False)
    fig.colorbar(im, ax=axes[1], fraction=.04, pad=.02)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig3_critic_transfer")

    # Fig. 4: probe-JS gap between teammates and non-teammates.
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    for m in MODES:
        x = [r["env_steps"] / 1e6 for r in js[m]]
        ax.plot(x, [r["same"] - r["cross"] for r in js[m]], color=COLOR[m], ls=STYLE[m], marker=MARK[m], ms=3,
                label=LABEL[m])
    ax.axhline(0, color="#555555", lw=.8)
    ax.set_xlabel("Environment steps (M)")
    ax.set_ylabel("JS(same) $-$ JS(cross)")
    ax.text(.02, .04, "below 0: teammates more similar", transform=ax.transAxes, fontsize=6.5, color="#555555")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    save(fig, "fig4_probe_js_gap")

    # Fig. A1: reproducibility against E17 (same seed, same initial weights).
    fig, axes = plt.subplots(1, 4, figsize=(TEXT_W, 1.6), sharey=True)
    for ax, m in zip(axes, MODES):
        n = len(old[m]["steps"])
        ax.plot(ev[m]["steps"][:n] / 1e6, ev[m]["deliveries"][:n], color=COLOR[m], label="this run")
        ax.plot(old[m]["steps"] / 1e6, old[m]["deliveries"], color="#333333", ls=":", label="earlier run (E17)")
        r = numbers["reproducibility_vs_E17"][m]
        ax.set_title(f"{LABEL[m]}: " + ("identical" if r["identical"] else
                                       f"diverges at {r['first_divergent_step'] / 1e6:.2f}M"))
        ax.set_xlabel("Env. steps (M)")
    axes[0].set_ylabel("Deliveries per episode")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=.6)
    save(fig, "figA1_reproducibility")


if __name__ == "__main__":
    main()
