"""Read-only analysis of the interrupted from-scratch run (E17) up to ~2.5M steps; writes only this folder."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from latent_team import common as C  # noqa: E402

RUN = ROOT / "runs/scratch_20261013_202352_stage2"
OUT = Path(__file__).resolve().parent
MODES = ["ippo", "pgct", "oracle", "all"]
EVAL_KEYS = ["deliveries", "requested_shelf_pickups", "returns_home", "task_return", "full_cycle_success_rate",
             "requested_shelf_seen_rate", "pickup_conversion_rate", "invalid_toggle_rate", "failed_forward_rate"]
PAIR_SAME = {"1-3", "2-4"}


def smooth(y, w=3):
    y = np.asarray(y, dtype=float)
    return np.convolve(y, np.ones(w) / w, mode="valid") if len(y) >= w else y


def first_step(steps, values, level, w=3):
    sm = smooth(values, w)
    st = np.asarray(steps[w - 1:]) if len(values) >= w else np.asarray(steps)
    hit = st[sm >= level]
    return int(hit[0]) if len(hit) else None


def window_mean(rows, key, lo, hi):
    vals = [r["metrics"][key] for r in rows if lo <= r["env_steps"] < hi and key in r["metrics"]
            and np.isfinite(r["metrics"][key])]
    return float(np.mean(vals)) if vals else None


report = dict(run=str(RUN), note="Interrupted by a Windows Update reboot at 2026-10-15 01:01 KST; single training seed.",
              modes={})
curves = {}
for mode in MODES:
    d = RUN / f"e1_{mode}_s1"
    rows = C.read_jsonl(d / "metrics.jsonl")
    train = [r for r in rows if "train/env_steps" in r["metrics"]]
    steps, evals = C.evaluation_curve(d)
    curves[mode] = (np.asarray(steps), {k: np.asarray([e.get(k, np.nan) for e in evals]) for k in EVAL_KEYS})
    curves[mode][1]["agents"] = np.asarray([[e[f"agent_{i+1}_delivery_count"] / 20 for i in range(4)] for e in evals])
    transfer = C.read_jsonl(d / "transfer.jsonl")
    support = np.asarray([r["actual_allocation"] for r in transfer]) > 0
    cand = np.asarray([r["physical_candidates"] for r in transfer], dtype=bool)
    held = np.asarray([r["warmup_hold"] for r in transfer])
    pairs = {}
    for label, (i, j) in zip(C.PAIR_LABELS, C.PAIRS):
        pairs[label] = dict(transfer_updates=int((support[:, i, j] | support[:, j, i]).sum()),
                            candidate_updates=int(cand[:, i, j].sum()))
    after = ~held
    any_transfer = support.reshape(len(support), -1).any(1)
    gate_ratio = [r["metrics"]["update/gate_ratio"] for r in train if "update/gate_ratio" in r["metrics"]]
    report["modes"][mode] = dict(
        last_logged_step=train[-1]["env_steps"] if train else None, updates=len(transfer),
        last_eval_step=int(steps[-1]), evals=len(steps),
        last_eval={k: float(curves[mode][1][k][-1]) for k in EVAL_KEYS},
        last3_eval_mean={k: float(np.nanmean(curves[mode][1][k][-3:])) for k in EVAL_KEYS},
        peak_deliveries=float(np.nanmax(curves[mode][1]["deliveries"])),
        peak_step=int(steps[int(np.nanargmax(curves[mode][1]["deliveries"]))]),
        onset_smoothed3={f"pickups_ge_{v}": first_step(steps, curves[mode][1]["requested_shelf_pickups"], v) for v in (1, 5, 10)}
        | {f"deliveries_ge_{v}": first_step(steps, curves[mode][1]["deliveries"], v) for v in (1, 5, 10)},
        agent_deliveries_last_eval=curves[mode][1]["agents"][-1].tolist(),
        agent_deliveries_last3_mean=curves[mode][1]["agents"][-3:].mean(0).tolist(),
        eval_step_to_step_noise=float(np.std(np.diff(curves[mode][1]["deliveries"][-10:]))),
        train_windows={f"{lo/1e6:.1f}-{hi/1e6:.1f}M": {k.split('/')[-1]: window_mean(train, k, lo, hi) for k in
                                                     ("train/deliveries", "train/requested_shelf_pickups",
                                                      "train/episode_return", "update/value_loss", "update/entropy",
                                                      "update/approx_kl", "update/pgct_peer_loss", "train/sps")}
                       for lo, hi in ((0, 5e5), (5e5, 1.5e6), (1.5e6, 2.6e6))},
        transfer=dict(updates_after_warmup=int(after.sum()), updates_with_transfer=int(any_transfer.sum()),
                      fraction_after_warmup_with_transfer=float(any_transfer[after].mean()) if after.any() else 0.0,
                      same_team_arc_updates=int(sum(r["same_team_arcs"] for r in transfer)),
                      cross_team_arc_updates=int(sum(r["cross_team_arcs"] for r in transfer)),
                      mean_gate_ratio=float(np.mean(gate_ratio)) if gate_ratio else None, pairs=pairs))

# Common evaluated horizon for AUC (no extrapolation).
common = min(int(curves[m][0][-1]) for m in MODES)
report["common_eval_horizon"] = common
for mode in MODES:
    steps, ev = curves[mode]
    k = steps <= common
    for key in ("deliveries", "requested_shelf_pickups", "returns_home"):
        y = ev[key][k]
        report["modes"][mode][f"{key}_auc_per_step_to_common"] = float(np.trapezoid(y, steps[k]) / (steps[k][-1] - steps[k][0]))
# Reference: the earlier from-scratch PGCT (home 1.0, complete graph), eval deliveries near the same steps.
ref = OUT / "reference_home1_pgct_eval.json"
if ref.exists():
    report["reference_home1_pgct"] = json.loads(ref.read_text(encoding="utf-8"))
(OUT / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=130, constrained_layout=True)
for mode in MODES:
    steps, ev = curves[mode]
    for ax, key in zip(axes.flat[:3], ("deliveries", "requested_shelf_pickups", "returns_home")):
        ax.plot(steps / 1e6, ev[key], alpha=.25)
        ax.plot(steps[2:] / 1e6, smooth(ev[key]), label=mode, color=ax.lines[-1].get_color())
        ax.set_title(f"eval {key} (20x500, deterministic; 3-pt mean)")
        ax.set_xlabel("env steps (M)")
labels = C.PAIR_LABELS
x = np.arange(len(labels))
for n, mode in enumerate(["pgct", "oracle", "all"]):
    axes[1, 1].bar(x + n * .27, [report["modes"][mode]["transfer"]["pairs"][p]["transfer_updates"] for p in labels], .27,
                   label=mode)
axes[1, 1].set_xticks(x + .27, [p + (" same" if p in PAIR_SAME else " cross") for p in labels], rotation=30)
axes[1, 1].set_title("updates with critic transfer per pair")
for ax in axes.flat:
    ax.legend(fontsize=8)
fig.savefig(OUT / "curves.png")
print(json.dumps({m: dict(last3=report["modes"][m]["last3_eval_mean"]["deliveries"],
                          auc=report["modes"][m]["deliveries_auc_per_step_to_common"],
                          onset=report["modes"][m]["onset_smoothed3"]) for m in MODES}, indent=1))
