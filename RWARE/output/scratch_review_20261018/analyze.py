"""Read-only review of the running from-scratch E1 run (IPPO/PGCT/Oracle/All, seed 1); writes only this folder.

    python output/scratch_review_20261018/analyze.py [--no-probe]
"""
import argparse
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
OLD = ROOT / "runs/scratch_20261013_202352_stage2"  # E17, same seed/init, interrupted at ~2.5M
OUT = Path(__file__).resolve().parent
MODES = ["ippo", "pgct", "oracle", "all"]
COLORS = dict(ippo="#4c72b0", pgct="#dd8452", oracle="#55a868", all="#c44e52")
KEYS = ["deliveries", "requested_shelf_pickups", "returns_home", "task_return"]
WINDOWS = [(0, 1e6), (1e6, 2e6), (2e6, 3e6), (3e6, 4e6), (4e6, 5e6), (5e6, 6e6), (6e6, 7e6)]
SAME = [(0, 2), (1, 3)]


def smooth(y, w=5):
    y = np.asarray(y, float)
    return np.convolve(y, np.ones(w) / w, mode="valid") if len(y) >= w else y


def onset(steps, y, level, w=3):
    sm = smooth(y, w)
    st = np.asarray(steps)[w - 1:] if len(y) >= w else np.asarray(steps)
    hit = st[sm >= level]
    return int(hit[0]) if len(hit) else None


def load_eval(d):
    steps, evals = C.evaluation_curve(d)
    ev = {k: np.asarray([e.get(k, np.nan) for e in evals], float) for k in KEYS}
    ev["agents"] = np.asarray([[e.get(f"agent_{i + 1}_delivery_count", np.nan) / 20 for i in range(4)] for e in evals])
    return np.asarray(steps), ev


def transfer_windows(d):
    rows = C.read_jsonl(d / "transfer.jsonl")
    out = []
    for lo, hi in WINDOWS:
        rs = [r for r in rows if lo <= r["env_steps"] < hi and not r["warmup_hold"]]
        if not rs:
            out.append(None)
            continue
        a = np.asarray([r["actual_allocation"] for r in rs]) > 0
        pair = {f"{i + 1}-{j + 1}": int((a[:, i, j] | a[:, j, i]).sum()) for i in range(4) for j in range(i + 1, 4)}
        out.append(dict(updates=len(rs), with_transfer=int(a.reshape(len(a), -1).any(1).sum()),
                        same_arcs=int(sum(r["same_team_arcs"] for r in rs)),
                        cross_arcs=int(sum(r["cross_team_arcs"] for r in rs)), pairs=pair))
    return out


def train_windows(d):
    rows = [r for r in C.read_jsonl(d / "metrics.jsonl") if "train/env_steps" in r["metrics"]]
    keys = ["update/entropy", "update/value_loss", "update/approx_kl", "update/pgct_peer_loss", "train/sps"]
    res = []
    for lo, hi in WINDOWS:
        vals = {}
        for k in keys:
            v = [r["metrics"][k] for r in rows if lo <= r["env_steps"] < hi and k in r["metrics"]
                 and np.isfinite(r["metrics"][k])]
            vals[k.split("/")[-1]] = float(np.mean(v)) if v else None
        res.append(vals)
    return res


def probe_js(d, every=1000):
    """Canonical-bank JS between agents on saved checkpoints (frozen, CPU)."""
    from examples.diagnose_pgct_rollout_compatibility import load_model
    from examples.pgct_probe_diagnostics import common_probe_response
    bank = torch.load(RUN / "canonical_bank_e1_s1.pt", map_location="cpu", weights_only=False)["sequences"].float()
    res = []
    for ck in sorted(d.glob("checkpoint_*.pt")):
        if int(ck.stem.split("_")[1]) % every:
            continue
        agents, meta = load_model(ck)
        m = np.asarray(common_probe_response(agents, bank, torch.device("cpu"))["mean_js"])
        same = np.mean([m[i, j] for i, j in SAME])
        cross = np.mean([m[i, j] for i in range(4) for j in range(i + 1, 4) if (i, j) not in SAME])
        res.append(dict(checkpoint=ck.name, env_steps=int(meta["env_steps"]), same=float(same), cross=float(cross),
                        matrix=m.tolist()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-probe", action="store_true")
    args = ap.parse_args()
    rep = dict(run=str(RUN.relative_to(ROOT)), modes={})
    curves, old = {}, {}
    for mode in MODES:
        d = RUN / f"e1_{mode}_s1"
        steps, ev = load_eval(d)
        curves[mode] = (steps, ev)
        od = OLD / f"e1_{mode}_s1"
        if (od / "metrics.jsonl").exists():
            old[mode] = load_eval(od)
        status = json.loads((d / "job_status.json").read_text(encoding="utf-8"))["status"]
        last_train = [r for r in C.read_jsonl(d / "metrics.jsonl") if "train/env_steps" in r["metrics"]][-1]
        sps = last_train["metrics"].get("train/sps")
        rep["modes"][mode] = dict(
            status=status, last_env_steps=last_train["env_steps"], sps=sps,
            eta_days_to_15M=(15_000_576 - last_train["env_steps"]) / sps / 86400 if sps else None,
            last_eval_step=int(steps[-1]),
            last5_mean={k: float(np.nanmean(ev[k][-5:])) for k in KEYS},
            last5_agent_deliveries=np.nanmean(ev["agents"][-5:], 0).round(2).tolist(),
            peak_deliveries=float(np.nanmax(ev["deliveries"])),
            onset={f"deliveries_ge_{v}": onset(steps, ev["deliveries"], v) for v in (1, 5, 10, 20)},
            eval_noise_std=float(np.std(np.diff(ev["deliveries"][-10:]))),
            train=train_windows(d),
            transfer=transfer_windows(d) if mode != "ippo" else None)
    common = min(int(curves[m][0][-1]) for m in MODES)
    rep["common_eval_horizon"] = common
    for mode in MODES:
        steps, ev = curves[mode]
        k = steps <= common
        rep["modes"][mode]["delivery_auc_per_step"] = float(np.trapezoid(ev["deliveries"][k], steps[k]) / (steps[k][-1] - steps[k][0]))
        for lo, hi in ((0, 2e6), (2e6, 4e6), (4e6, common)):
            kk = (steps >= lo) & (steps <= hi)
            rep["modes"][mode][f"mean_deliveries_{lo/1e6:.0f}-{hi/1e6:.1f}M"] = float(np.nanmean(ev["deliveries"][kk]))
        if mode in old:
            os_, oe = old[mode]
            n = min(len(os_), np.searchsorted(steps, os_[-1], side="right"))
            same_steps = np.array_equal(steps[:n], os_[:n])
            rep["modes"][mode]["vs_E17"] = dict(
                common_evals=int(n), identical_eval_steps=bool(same_steps),
                first_divergent_eval_step=next((int(s) for s, a, b in zip(steps[:n], ev["deliveries"][:n], oe["deliveries"][:n])
                                                if abs(a - b) > 1e-9), None),
                E17_last_eval_step=int(os_[-1]))
    if not args.no_probe:
        rep["probe_js"] = {m: probe_js(RUN / f"e1_{m}_s1") for m in MODES}
    (OUT / "analysis.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    plot(rep, curves, old)
    for m in MODES:
        r = rep["modes"][m]
        print(m, r["last_env_steps"], r["last5_mean"]["deliveries"], r["last5_agent_deliveries"], r["delivery_auc_per_step"],
              r["onset"], r.get("vs_E17"))


def plot(rep, curves, old):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), dpi=120, constrained_layout=True)
    for mode in MODES:
        steps, ev = curves[mode]
        c = COLORS[mode]
        for ax, key in zip(axes[0], ("deliveries", "requested_shelf_pickups", "returns_home")):
            ax.plot(steps / 1e6, ev[key], color=c, alpha=.18)
            ax.plot(steps[4:] / 1e6, smooth(ev[key]), color=c, label=mode)
            if mode in old:
                ax.plot(old[mode][0] / 1e6, old[mode][1][key], color=c, ls=":", lw=1)
            ax.set_title(f"eval {key} per episode (20x500 det.; 5-pt mean)\ndotted = earlier interrupted run E17")
            ax.set_xlabel("env steps (M)")
    ax = axes[1, 0]
    for mode in MODES:
        steps, ev = curves[mode]
        for a, ls in zip(range(4), ["-", "--", "-.", ":"]):
            if mode in ("ippo", "oracle"):
                ax.plot(steps[4:] / 1e6, smooth(ev["agents"][:, a]), color=COLORS[mode], ls=ls, lw=1.2,
                        label=f"{mode} A{a + 1}")
    ax.set_title("per-agent eval deliveries: IPPO vs Oracle (A1,A3 red; A2,A4 blue)")
    ax.legend(fontsize=6, ncol=2)
    ax = axes[1, 1]
    for mode in ("pgct", "oracle", "all"):
        tw = rep["modes"][mode]["transfer"]
        x = [(lo + hi) / 2e6 for (lo, hi), t in zip(WINDOWS, tw) if t]
        ax.plot(x, [t["same_arcs"] / t["updates"] for t in tw if t], color=COLORS[mode], marker="o", label=f"{mode} same-team")
        ax.plot(x, [t["cross_arcs"] / t["updates"] for t in tw if t], color=COLORS[mode], marker="x", ls="--",
                label=f"{mode} cross-team")
    ax.set_title("transfer arcs per update (actual allocation, per 1M window)")
    ax.set_xlabel("env steps (M)")
    ax.legend(fontsize=7)
    ax = axes[1, 2]
    if "probe_js" in rep:
        for mode in MODES:
            p = rep["probe_js"][mode]
            x = [r["env_steps"] / 1e6 for r in p]
            ax.plot(x, [r["same"] for r in p], color=COLORS[mode], marker="o", label=f"{mode} same")
            ax.plot(x, [r["cross"] for r in p], color=COLORS[mode], marker="x", ls="--", label=f"{mode} cross")
        ax.axhline(0.12, color="k", ls=":", lw=1)
        ax.set_title("canonical-probe JS between agents (checkpoints)\nteams separable only if 'same' < 'cross'; dotted = gate 0.12")
        ax.set_xlabel("env steps (M)")
        ax.legend(fontsize=6, ncol=2)
    for ax in axes[0]:
        ax.legend(fontsize=8)
    fig.savefig(OUT / "overview.png")


if __name__ == "__main__":
    main()
