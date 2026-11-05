"""Aggregate a plan folder into identification / learning / cost tables (evaluation code: may use labels).

  python -m pgct_overcooked.aggregate --plan Overcooked/runs/<name> [--warmup-only-after]

Writes <plan>/summary.json, summary.md and jobs.csv. Unfinished jobs are reported with their status and the
rows logged so far. Nothing in a job folder is modified.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import REPO_ROOT  # noqa: F401  (pgct_core path)
from pgct_core.metrics import identification_summary, learning_auc, mean_std
from .env.kitchen import Kitchen, KitchenConfig

KEY_METRICS = ("final_eval_return", "eval_return_auc", "final_eval_deliveries_own", "final_eval_deliveries_other",
               "train_return_last10", "donor_arcs_per_update", "id_final_f1", "id_final_precision",
               "id_final_recall", "id_mean_false_approval_rate", "id_auroc", "id_first_correct_step",
               "id_gate_stability", "probe_bytes_total", "critic_bytes_total", "time_gate_s", "time_max_gae_s",
               "seconds", "rss_mb_max")


def read_jsonl(path):
    if not Path(path).exists():
        return []
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a partially written last line of a running job
    return rows


def team_ids(config):
    kitchen = dict(config["experiment"]["kitchen"])
    for key in ("team_recipes", "team_sizes"):
        kitchen[key] = tuple(kitchen[key])
    return Kitchen(KitchenConfig(**kitchen)).oracle_team_ids()


def mean_of(rows, key, start=0):
    vals = [r[key] for r in rows[start:] if r.get(key) is not None and np.isfinite(r[key])]
    return float(np.mean(vals)) if vals else None


def summarize_job(folder: Path):
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    status = json.loads((folder / "status.json").read_text(encoding="utf-8")) if (folder / "status.json").exists() \
        else dict(status="running_or_incomplete")
    metrics, gates, evals = (read_jsonl(folder / f) for f in ("metrics.jsonl", "gate.jsonl", "eval.jsonl"))
    teams = team_ids(config)
    warmup = config["trainer"]["warmup_updates"]
    out = dict(job=folder.name, method=config["trainer"]["method"], seed=config["trainer"]["seed"],
               status=status.get("status"), updates=len(metrics),
               env_steps=metrics[-1]["env_steps"] if metrics else 0, seconds=status.get("seconds"),
               team_ids=teams.tolist())
    if evals:
        steps = [e["env_steps"] for e in evals]
        out.update(final_eval_return=evals[-1]["eval_return_mean"],
                   final_eval_return_per_agent=evals[-1]["eval_return_per_agent"],
                   final_eval_deliveries_own=evals[-1].get("eval_deliveries_own_total", 0.0),
                   final_eval_deliveries_other=evals[-1].get("eval_deliveries_other_total", 0.0),
                   eval_return_auc=learning_auc(steps, [e["eval_return_mean"] for e in evals]),
                   **{k: v for k, v in evals[-1].items() if k.startswith("eval_team")})
    returns = [r["episode_return_mean"] for r in metrics if "episode_return_mean" in r]
    out["train_return_last10"] = float(np.mean(returns[-10:])) if returns else None
    out.update(donor_arcs_per_update=mean_of(metrics, "max_gae_donor_arcs", warmup),
               cross_team_donor_arcs_per_update=mean_of(metrics, "max_gae_cross_team_arcs", warmup),
               max_gae_mean_b=mean_of(metrics, "max_gae_mean_b"), max_gae_select_frac=mean_of(metrics, "max_gae_select_frac"),
               max_gae_adv_shift=mean_of(metrics, "max_gae_adv_shift"), max_gae_eta_eff=mean_of(metrics, "max_gae_eta_eff"),
               probe_exchanges_total=float(sum(r.get("cost_probe_exchanges", 0) for r in metrics)),
               probe_bytes_total=float(sum(r.get("cost_probe_bytes", 0) for r in metrics)),
               critic_bytes_total=float(sum(r.get("cost_critic_bytes", 0) for r in metrics)),
               time_gate_s=float(sum(r.get("time_gate_s", 0) for r in metrics)),
               time_max_gae_s=float(sum(r.get("time_max_gae_s", 0) for r in metrics)),
               time_rollout_s=float(sum(r.get("time_rollout_s", 0) for r in metrics)),
               time_ppo_s=float(sum(r.get("time_ppo_s", 0) for r in metrics)),
               rss_mb_max=max((r["rss_mb"] for r in metrics if r.get("rss_mb") is not None), default=None))
    if gates and "approved" in gates[0]:
        ident = identification_summary(gates, teams)
        final = ident.get("final", {})
        out.update(id_final_f1=final.get("f1"), id_final_precision=final.get("precision"),
                   id_final_recall=final.get("recall"), id_final_false_approval_rate=final.get("false_approval_rate"),
                   id_mean_false_approval_rate=ident.get("mean_false_approval_rate"), id_auroc=ident.get("auroc"),
                   id_first_correct_step=ident.get("first_correct_approval_step"),
                   id_first_wrong_step=ident.get("first_wrong_approval_step"),
                   id_gate_stability=ident.get("gate_stability"), id_approval_flips=ident.get("approval_flips"),
                   id_distance_same=ident.get("distance_same_mean"), id_distance_cross=ident.get("distance_cross_mean"),
                   id_mature_agents_final=float(np.sum(gates[-1]["reward_mature"])))
    return out


def aggregate(plan: Path):
    jobs = [summarize_job(p.parent) for p in sorted(plan.glob("*/config.json"))]
    by_method = {}
    for job in jobs:
        by_method.setdefault(job["method"], []).append(job)
    table = {m: {k: mean_std([j.get(k) for j in members]) for k in KEY_METRICS} | dict(seeds=sorted(j["seed"] for j in members))
             for m, members in sorted(by_method.items())}
    (plan / "summary.json").write_text(json.dumps(dict(jobs=jobs, by_method=table), indent=2), encoding="utf-8")
    keys = sorted({k for j in jobs for k in j if not isinstance(j[k], (list, dict))})
    with (plan / "jobs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(jobs)
    fmt = lambda s: "-" if s["mean"] is None else (f"{s['mean']:.3g}" + (f" +/- {s['std']:.2g}" if s["std"] is not None else ""))
    cols = ("final_eval_return", "eval_return_auc", "final_eval_deliveries_own", "donor_arcs_per_update",
            "id_final_f1", "id_mean_false_approval_rate", "id_auroc", "id_first_correct_step", "probe_bytes_total")
    lines = ["# Overcooked PGCT comparison", "",
             "Mean +/- std over training seeds (ddof=1). Identification uses ground-truth teams in evaluation only. "
             "Unfinished jobs are included with the rows logged so far; check the status column in jobs.csv.", "",
             "| Method | Seeds | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for method, row in table.items():
        lines.append(f"| {method} | {row['seeds']} | " + " | ".join(fmt(row[c]) for c in cols) + " |")
    (plan / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jobs, table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args(argv)
    jobs, _ = aggregate(args.plan)
    print((args.plan / "summary.md").read_text(encoding="utf-8"))
    print(f"{len(jobs)} jobs -> {args.plan / 'summary.json'}")


if __name__ == "__main__":
    main()
