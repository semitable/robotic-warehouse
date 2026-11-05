"""Render probe response / pair-time heatmaps and matched learning curves."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from . import train_recurrent_ippo_consensus as T
    from .pgct_probe_diagnostics import PAIRS, PAIR_LABELS, ACTIONS, common_probe_response
except ImportError:
    import train_recurrent_ippo_consensus as T
    from pgct_probe_diagnostics import PAIRS, PAIR_LABELS, ACTIONS, common_probe_response


def read_jsonl(path):
    if not path.exists():
        return []
    # A live writer may leave one incomplete final line. Do not discard interior corruption.
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return rows


def probe_chart(row, path, title):
    p = np.asarray(row["probabilities"]).mean(2)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), dpi=140, constrained_layout=True)
    for i, ax in enumerate(axes.flat[:4]):
        m = ax.imshow(p[i].T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_yticks(range(5), ACTIONS)
        ax.set_xlabel("Common probe index (0-based)")
        ax.set_title(f"Agent {i+1} / team {i % 2}: action probabilities")
    fig.colorbar(m, ax=list(axes.flat[:4]), label="Probability", shrink=.7)
    ax = axes.flat[4]
    m = ax.imshow(row["js_per_probe"], aspect="auto", cmap="magma_r", vmin=0, vmax=np.log(2))
    ax.set_yticks(range(6), PAIR_LABELS)
    ax.set_xlabel("Common probe index (0-based)")
    ax.set_title("JS per probe (light = more similar)")
    fig.colorbar(m, ax=ax, label="JS (natural log)", shrink=.75)
    ax = axes.flat[5]
    mean = np.asarray(row["mean_js"])
    m = ax.imshow(mean, cmap="magma_r", vmin=0, vmax=np.log(2))
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{mean[i,j]:.3f}", ha="center", va="center", color="white" if mean[i,j] > .34 else "black")
    ax.set_xticks(range(4), range(1, 5)); ax.set_yticks(range(4), range(1, 5))
    ax.set_title("Mean JS / shared bank")
    fig.suptitle(title + "\nShared inputs, zero initial hidden state, unmasked; not transfer utility")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path); plt.close(fig)


def time_chart(steps, distances, path, title, *, gate=False):
    if not len(steps):
        return
    data = np.asarray(distances).T
    fig, ax = plt.subplots(figsize=(12, 4.1), dpi=150, constrained_layout=True)
    # Nonuniform checkpoint intervals keep their actual horizontal widths.
    x = np.asarray(steps, dtype=float) / 1e6
    edges = np.r_[x[0] - (x[1] - x[0]) / 2, (x[1:] + x[:-1]) / 2,
                  x[-1] + (x[-1] - x[-2]) / 2] if len(x) > 1 else np.array([x[0]-.0005, x[0]+.0005])
    m = ax.pcolormesh(edges, np.arange(7)-.5, data, cmap="Greens" if gate else "magma_r",
                      vmin=0, vmax=1 if gate else np.log(2), shading="flat")
    ax.invert_yaxis(); ax.set_yticks(range(6), PAIR_LABELS)
    ax.set_xlabel("Additional environment steps (millions)")
    ax.set_title(title)
    fig.colorbar(m, ax=ax, label="Transfer present (0/1)" if gate else "JS (natural log)")
    fig.savefig(path); plt.close(fig)


@torch.inference_mode()
def checkpoint_response(path, bank):
    cp = T._load_checkpoint(str(path), torch.device("cpu"))
    cfg = cp["config"]
    spec = T.get_semantic_observation_spec(type("SpecEnv", (), {"sensor_range": cfg["sensor_range"]})())
    nets = []
    for state in cp["agents"]:
        net = T.RecurrentActorCritic(bank.shape[-1], 5, cfg["mlp_hidden_dim"], cfg["recurrent_hidden_dim"],
                                    cfg["obs_encoder"], spec.spatial_channels, spec.spatial_size, spec.ego_dim)
        net.load_state_dict(state); net.eval(); nets.append(net)
    row = common_probe_response(nets, bank, torch.device("cpu"))
    row.update(env_steps=cp["env_steps"], checkpoint=str(path))
    return row


def plot_run(run_dir, output):
    output.mkdir(parents=True, exist_ok=True)
    probes = read_jsonl(run_dir / "probes.jsonl")
    # Prefer a final post-update snapshot over the pre-update row at the same step.
    probes = list({row["env_steps"]: row for row in probes}.values())
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    if probes:
        probe_chart(probes[-1], output / "response.png", run_dir.name + " / latest diagnostic")
        time_chart([p["env_steps"] for p in probes],
                   [[p["mean_js"][i][j] for i, j in PAIRS] for p in probes],
                   output / "js_time.png", run_dir.name + " / common probe JS over learning")
        time_chart([p["env_steps"] for p in probes],
                   [[p["unique_mean_js"][i][j] for i, j in PAIRS] for p in probes],
                   output / "unique_js_time.png", run_dir.name + " / deduplicated diagnostic JS")
    else:
        # Existing live runs already record raw pair JS at probe rounds.
        raw = [(r["env_steps"], [r["metrics"].get(f"probe/pair_{i}_{j}_distance", np.nan) for i, j in PAIRS])
               for r in metrics if "probe/pair_0_1_distance" in r["metrics"]]
        if raw:
            time_chart(*zip(*raw), output / "js_time.png", run_dir.name + " / raw training-probe JS")
        bank_path = run_dir / "canonical_probes.pt"
        if not bank_path.exists() and "_ippo_" in run_dir.name:
            # IPPO has no training probe. Forward-only reconstruction uses its
            # matched PGCT branch's bank, never probes from a different map.
            bank_path = run_dir.parent / run_dir.name.replace("_ippo_", "_pgct_") / "canonical_probes.pt"
        checkpoints = sorted(run_dir.glob("checkpoint_*.pt"))
        if (run_dir / "final.pt").exists():
            checkpoints.append(run_dir / "final.pt")
        if bank_path.exists() and checkpoints:
            bank = torch.load(bank_path, map_location="cpu", weights_only=False)["sequences"]
            snapshots = [checkpoint_response(cp, bank) for cp in checkpoints] if not raw else [checkpoint_response(checkpoints[-1], bank)]
            row = snapshots[-1]
            probe_chart(row, output / "response.png", f"{run_dir.name} / saved checkpoint {row['env_steps']} steps")
            (output / "checkpoint_response.json").write_text(json.dumps(row), encoding="utf-8")
            if not raw:
                snapshots = list({r["env_steps"]: r for r in snapshots}.values())
                time_chart([r["env_steps"] for r in snapshots],
                           [[r["mean_js"][i][j] for i, j in PAIRS] for r in snapshots],
                           output / "js_time.png", run_dir.name + " / checkpoint JS on matched PGCT bank")
    transfer = read_jsonl(run_dir / "transfer.jsonl")
    communication = read_jsonl(run_dir / "communication.jsonl")
    if communication:
        time_chart([r["env_steps"] for r in communication],
                   [[r["policy_distance_ema"][i][j] for i, j in PAIRS] for r in communication],
                   output / "ema_time.png", run_dir.name + " / neighbor-updated gate EMA (threshold 0.12)")
    edges = transfer or communication
    if edges:
        key = "actual_allocation" if transfer else "transfer_weights"
        time_chart([r["env_steps"] for r in edges],
                   [[float(r[key][i][j] > 0 or r[key][j][i] > 0) for i, j in PAIRS] for r in edges],
                   output / "transfer_time.png", run_dir.name + " / actual transfer pair-updates", gate=True)


def plot_experiment(experiment_dir, output):
    manifest = json.loads((experiment_dir / "experiment.json").read_text(encoding="utf-8"))
    curves = []
    for job in manifest["jobs"]:
        run_dir = Path(job.get("reused_run_dir", experiment_dir / job["name"]))
        if not run_dir.exists():
            continue
        plot_run(run_dir, output / job["name"])
        rows = read_jsonl(run_dir / "metrics.jsonl")
        evaluation = {}
        for row in rows:
            if "eval/deliveries" in row["metrics"]:
                evaluation[row["env_steps"]] = row["metrics"]
            elif "eval/final_deliveries" in row["metrics"]:
                evaluation[row["env_steps"]] = {"eval/" + k.removeprefix("eval/final_"): v
                                               for k, v in row["metrics"].items() if k.startswith("eval/final_")}
        if evaluation:
            curves.append((job, sorted(evaluation), evaluation))
    if not curves:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150, constrained_layout=True)
    for job, steps, values in curves:
        for ax, key, label in zip(axes, ["eval/deliveries", "eval/returns_home"], ["Delivery / episode", "Return / episode"]):
            ax.plot(np.asarray(steps)/1e6, [values[s][key] for s in steps], label=job["name"])
            ax.set_xlabel("Additional environment steps (millions)"); ax.set_ylabel(label)
    axes[0].legend(fontsize=8); fig.savefig(output / "learning.png"); plt.close(fig)
    # Per-agent curves expose cases where one successful agent dominates the total.
    fig, axes = plt.subplots(3, 4, figsize=(15, 9), dpi=140, constrained_layout=True)
    for job, steps, values in curves:
        episodes = job["config"]["eval_episodes"]
        for event_row, event in enumerate(("delivery", "return", "pickup")):
            for i in range(4):
                per_episode = f"eval/agent_{i+1}_{event}_per_episode"
                raw = f"eval/agent_{i+1}_{event}_count"
                y = [values[s].get(per_episode, values[s].get(raw, np.nan) / episodes) for s in steps]
                ax = axes[event_row, i]
                ax.plot(np.asarray(steps)/1e6, y, label=job["name"])
                ax.set_title(f"Agent {i+1} / team {i%2}: {event}"); ax.set_ylabel("Count / episode")
                ax.set_xlabel("Additional steps (millions)")
    axes[0, 0].legend(fontsize=7); fig.savefig(output / "agents.png"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=140, constrained_layout=True)
    available = False
    for job, steps, values in curves:
        for ax, key, label in zip(axes, ["eval/critic_mc_mse", "eval/critic_td_abs"],
                                  ["Finite-horizon stochastic MC MSE", "Stochastic one-step TD absolute error"]):
            xs = [s for s in steps if key in values[s]]
            if xs:
                ax.plot(np.asarray(xs)/1e6, [values[s][key] for s in xs], label=job["name"])
                available = True
            ax.set_title(label); ax.set_xlabel("Additional steps (millions)")
    if available:
        axes[0].legend(fontsize=8); fig.savefig(output / "critic.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.run_dir:
        plot_run(args.run_dir, args.output_dir)
    else:
        plot_experiment(args.experiment_dir, args.output_dir)
    print(f"Charts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
