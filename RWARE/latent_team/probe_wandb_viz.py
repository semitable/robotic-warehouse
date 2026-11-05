"""Probe-response colour maps for W&B.  Frozen checkpoints only; the training runs are not touched.

For every checkpoint of a from-scratch job this computes each actor's action distribution on the
probe bank(s) with the PGCT gate rule (zero hidden, unmasked softmax, natural-log JS) and logs:

  probe/<bank>/response    heatmap per agent: rows = probes, columns = NOOP/FWD/LEFT/RIGHT/TOGGLE
  probe/<bank>/js          4x4 mean JS matrix (tau marked) + per-probe JS for the 6 pairs
  probe/<bank>/drift       per-probe JS of each agent against its previous checkpoint
  probe/<bank>/js_same, js_cross, auroc, drift_mean   scalars (team labels only score, never gate)

Each job gets its own W&B run "<job>-probes" (group = plan folder, job_type = probe_viz), x axis = env
steps.  It never writes into the training run.  PNG copies go to --output-dir.  With --watch it keeps
polling for new checkpoints, so it can run next to a training job.

    python latent_team/probe_wandb_viz.py --run runs/<plan folder> [--jobs e1f_pgct_s1] \
        [--bank auto|canonical|v2] [--every 250] [--watch] [--wandb-mode online|offline|disabled]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples import train_recurrent_ippo_consensus as T  # noqa: E402
from examples.diagnose_pgct_rollout_compatibility import load_model, model_hash  # noqa: E402

ACTIONS = ["NOOP", "FWD", "LEFT", "RIGHT", "TOGGLE"]
PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]
TEAM_COLORS = ["#d62728", "#1f77b4"]
TAU = 0.12


def load_banks(run: Path, which: str) -> dict:
    banks = {}
    if which in ("auto", "canonical"):
        for f in sorted(run.glob("canonical_bank_*.pt")):
            b = torch.load(f, map_location="cpu", weights_only=False)
            banks["canonical"] = dict(seq=b["sequences"].float(), labels=None)
    if which in ("auto", "v2"):
        for f in sorted(run.glob("probe_bank_v2_*.pt")):
            b = torch.load(f, map_location="cpu", weights_only=False)
            labels = [f"{s['phase']}·T{s['team']}" for s in b.get("gate_scenarios", [])] or None
            banks["v2"] = dict(seq=b["sequences"].float(), labels=labels)
    if not banks:
        raise FileNotFoundError(f"no probe bank (canonical_bank_*.pt / probe_bank_v2_*.pt) in {run}")
    return banks


@torch.inference_mode()
def responses(net, seq) -> np.ndarray:
    h, _ = net.initial_hidden(len(seq), torch.device("cpu"))
    out = []
    for t in range(seq.shape[1]):
        logits, h = net._actor_step(seq[:, t], h)
        out.append(logits.softmax(-1).numpy())
    return np.stack(out, 1).astype(np.float64).mean(1)  # [probe, action], mean over probe time


def js(p, q) -> np.ndarray:
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (np.log(np.maximum(a, 1e-12)) - np.log(np.maximum(b, 1e-12)))).sum(-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)  # per probe


def team_ids(cfg, env_kwargs) -> list:
    with T.make_env(cfg["env_id"], env_kwargs, cfg["observation_format"]) as env:
        env.reset(seed=0)
        return [int(t) for t in np.asarray(env.unwrapped.agent_team_ids)]


def probe_order(labels, n):
    return list(range(n)) if labels is None else sorted(range(n), key=lambda k: (labels[k], k))


def fig_response(probs, labels, teams, title):
    n = probs.shape[1]
    order = probe_order(labels, n)
    fig, axes = plt.subplots(1, 4, figsize=(12, max(4, 0.16 * n + 1.5)), sharey=True, constrained_layout=True)
    for a, ax in enumerate(axes):
        im = ax.imshow(probs[a][order], aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(ACTIONS)), ACTIONS, rotation=60, fontsize=7)
        ax.set_title(f"A{a + 1} (team {teams[a]})", color=TEAM_COLORS[teams[a] % 2], fontsize=10)
    ytl = [labels[k] if labels else str(k) for k in order]
    axes[0].set_yticks(range(n), ytl, fontsize=5 if labels else 6)
    fig.colorbar(im, ax=axes, shrink=0.6, label="action probability")
    fig.suptitle(title, fontsize=10)
    return fig


def fig_js(per_probe, mean_js, labels, teams, title):
    n = per_probe.shape[1]
    order = probe_order(labels, n)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, max(4, 0.16 * n + 1.5)),
                                   gridspec_kw=dict(width_ratios=[1, 1.4]), constrained_layout=True)
    im0 = ax0.imshow(mean_js, cmap="magma_r", vmin=0, vmax=max(0.3, float(np.nanmax(mean_js))))
    for i in range(4):
        for j in range(4):
            if i != j:
                ax0.text(j, i, f"{mean_js[i, j]:.3f}{'*' if mean_js[i, j] <= TAU else ''}", ha="center",
                         va="center", fontsize=8, color="w" if mean_js[i, j] > 0.15 else "k")
    names = [f"A{a + 1}·T{teams[a]}" for a in range(4)]
    ax0.set_xticks(range(4), names, fontsize=8)
    ax0.set_yticks(range(4), names, fontsize=8)
    ax0.set_title(f"mean probe JS (* = below tau {TAU})", fontsize=9)
    fig.colorbar(im0, ax=ax0, shrink=0.6)
    im1 = ax1.imshow(per_probe[:, order].T, aspect="auto", cmap="magma_r", vmin=0, vmax=np.log(2))
    pair_names = [f"{i + 1}-{j + 1}{' (same)' if teams[i] == teams[j] else ''}" for i, j in PAIRS]
    ax1.set_xticks(range(len(PAIRS)), pair_names, rotation=60, fontsize=7)
    ax1.set_yticks(range(n), [labels[k] if labels else str(k) for k in order], fontsize=5 if labels else 6)
    ax1.set_title("per-probe JS by pair (max ln2)", fontsize=9)
    fig.colorbar(im1, ax=ax1, shrink=0.6)
    fig.suptitle(title, fontsize=10)
    return fig


def fig_drift(drift, labels, teams, title):
    n = drift.shape[1]
    order = probe_order(labels, n)
    fig, ax = plt.subplots(figsize=(5, max(4, 0.16 * n + 1.5)), constrained_layout=True)
    im = ax.imshow(drift[:, order].T, aspect="auto", cmap="magma_r", vmin=0, vmax=np.log(2))
    ax.set_xticks(range(4), [f"A{a + 1}·T{teams[a]}" for a in range(4)], fontsize=8)
    ax.set_yticks(range(n), [labels[k] if labels else str(k) for k in order], fontsize=5 if labels else 6)
    ax.set_title("JS vs own previous checkpoint", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.suptitle(title, fontsize=10)
    return fig


def auroc(mean_js, teams):
    same = [mean_js[i, j] for i, j in PAIRS if teams[i] == teams[j]]
    cross = [mean_js[i, j] for i, j in PAIRS if teams[i] != teams[j]]
    return float(np.mean([1.0 if s < c else 0.5 if s == c else 0.0 for s in same for c in cross]))


class JobViz:
    def __init__(self, run: Path, job: str, banks: dict, args, out: Path):
        self.run, self.job, self.banks, self.args = run, job, banks, args
        self.out = out / job
        self.out.mkdir(parents=True, exist_ok=True)
        self.state_file = self.out / "logged.json"
        self.done = set(json.loads(self.state_file.read_text())) if self.state_file.exists() else set()
        self.prev = {}
        self.teams = None
        self.wb = None

    def wandb_run(self, cfg):
        if self.wb is None and self.args.wandb_mode != "disabled":
            import wandb
            # Short id/dir: Windows MAX_PATH breaks wandb's offline-run-<time>-<id>/tmp/code folder otherwise.
            stamp = re.sub(r"\D", "", self.run.name)[-10:]
            variant = self.run.name.rsplit("_", 1)[-1][:10]
            rid = re.sub(r"[^A-Za-z0-9_-]", "-", f"pv-{stamp}-{variant}-{self.job}")[:64]
            self.wb = wandb.init(project=self.args.project, entity=self.args.entity or None, id=rid, resume="allow",
                                 name=f"{self.job}-probes", group=self.run.name, job_type="probe_viz",
                                 mode=self.args.wandb_mode, dir=str(self.out.parent),
                                 config=dict(source_run=str(self.run), job=self.job, banks=list(self.banks),
                                             rule="zero hidden, unmasked softmax, ln JS, frozen checkpoints",
                                             env_id=cfg["env_id"]))
            self.wb.define_metric("env_steps")
            self.wb.define_metric("probe/*", step_metric="env_steps")
        return self.wb

    def pending(self):
        cks = sorted((self.run / self.job).glob("checkpoint_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        fresh = time.time() - 60  # skip files that may still be being written
        return [c for c in cks if int(c.stem.split("_")[1]) % self.args.every == 0
                and c.name not in self.done and c.stat().st_mtime < fresh]

    def process(self, ck: Path):
        agents, meta = load_model(ck)
        cfg, steps = meta["config"], int(meta["env_steps"])
        before = model_hash(agents)
        if self.teams is None:
            env_kwargs = dict(T._load_checkpoint(str(ck), torch.device("cpu"))["effective_env_kwargs"])
            self.teams = team_ids(cfg, env_kwargs)
        payload = {"env_steps": steps}
        for name, bank in self.banks.items():
            probs = np.stack([responses(a, bank["seq"]) for a in agents])  # [agent, probe, action]
            per_probe = np.stack([js(probs[i], probs[j]) for i, j in PAIRS])  # [pair, probe]
            mean_js = np.zeros((4, 4))
            for k, (i, j) in enumerate(PAIRS):
                mean_js[i, j] = mean_js[j, i] = per_probe[k].mean()
            tag = f"{self.job} {name} @ {steps / 1e6:.2f}M"
            figs = {"response": fig_response(probs, bank["labels"], self.teams, tag),
                    "js": fig_js(per_probe, mean_js, bank["labels"], self.teams, tag)}
            same = [mean_js[i, j] for i, j in PAIRS if self.teams[i] == self.teams[j]]
            cross = [mean_js[i, j] for i, j in PAIRS if self.teams[i] != self.teams[j]]
            payload.update({f"probe/{name}/js_same": float(np.mean(same)),
                            f"probe/{name}/js_cross": float(np.mean(cross)),
                            f"probe/{name}/auroc": auroc(mean_js, self.teams),
                            f"probe/{name}/pairs_below_tau": int(sum(mean_js[i, j] <= TAU for i, j in PAIRS))})
            if name in self.prev:
                drift = np.stack([js(probs[a], self.prev[name][a]) for a in range(4)])
                figs["drift"] = fig_drift(drift, bank["labels"], self.teams, tag)
                payload[f"probe/{name}/drift_mean"] = float(drift.mean())
            self.prev[name] = probs
            wb = self.wandb_run(cfg)
            for kind, fig in figs.items():
                path = self.out / f"{name}_{kind}_{steps:09d}.png"
                fig.savefig(path, dpi=110)
                plt.close(fig)
                if wb is not None:
                    import wandb
                    payload[f"probe/{name}/{kind}"] = wandb.Image(str(path))
        if before != model_hash(agents):
            raise RuntimeError("weights changed during frozen probe visualisation")
        if self.wb is not None:
            self.wb.log(payload)
        self.done.add(ck.name)
        self.state_file.write_text(json.dumps(sorted(self.done)))
        scal = {k: round(v, 4) for k, v in payload.items() if isinstance(v, float)}
        print(f"[{self.job}] {ck.name} {steps / 1e6:.2f}M {scal}", flush=True)

    def finish_if_done(self):
        if (self.run / self.job / "final.pt").exists() and not self.pending():
            if self.wb is not None:
                self.wb.finish()
                self.wb = None
            return True
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="Plan folder (contains experiment.json).")
    ap.add_argument("--jobs", nargs="+", help="Job folder names; default: every job in the plan.")
    ap.add_argument("--bank", choices=["auto", "canonical", "v2"], default="auto")
    ap.add_argument("--every", type=int, default=250, help="Checkpoint update interval to log.")
    ap.add_argument("--watch", action="store_true", help="Keep polling for new checkpoints.")
    ap.add_argument("--poll-seconds", type=int, default=600)
    ap.add_argument("--project", default="rware-from-scratch")
    ap.add_argument("--entity", default="")
    ap.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    ap.add_argument("--output-dir", type=Path)
    args = ap.parse_args()
    torch.set_num_threads(1)
    run = args.run.resolve()
    jobs = args.jobs or [j["name"] for j in json.loads((run / "experiment.json").read_text(encoding="utf-8"))["jobs"]]
    out = (args.output_dir or ROOT / "output" / f"probe_viz_{run.name}").resolve()
    banks = load_banks(run, args.bank)
    vizzes = [JobViz(run, job, banks, args, out) for job in jobs]
    while True:
        for v in vizzes:
            for ck in v.pending():
                v.process(ck)
        if not args.watch or all(v.finish_if_done() for v in vizzes):
            break
        time.sleep(args.poll_seconds)
    for v in vizzes:
        if v.wb is not None:
            v.wb.finish()
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
