"""Aggregate a plan folder into identification / learning / cost tables (evaluation code: may use labels).

Writes <plan>/summary.json, summary.md and jobs.csv. Unfinished jobs are reported with their status and the
rows logged so far. Nothing in a job folder is modified. Same metrics as Overcooked/pgct_overcooked/aggregate.py;
the headline event counters come from the ExperimentSpec, the true teams from the job's own adapter.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pgct_core.metrics import identification_summary, learning_auc, mean_std

BASE_METRICS = ("final_eval_return", "eval_return_auc", "train_return_last10", "donor_arcs_per_update",
                "cross_team_donor_arcs_per_update", "id_final_f1", "id_final_precision", "id_final_recall",
                "id_mean_false_approval_rate", "id_auroc", "id_first_correct_step", "id_gate_stability",
                "probe_bytes_total", "critic_bytes_total", "time_gate_s", "time_max_gae_s", "seconds", "rss_mb_max")
TABLE_COLUMNS = ("final_eval_return", "eval_return_auc", "donor_arcs_per_update", "cross_team_donor_arcs_per_update",
                 "id_final_f1", "id_mean_false_approval_rate", "id_auroc", "id_first_correct_step")


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


def mean_of(rows, key, start=0):
    vals = [r[key] for r in rows[start:] if r.get(key) is not None and np.isfinite(r[key])]
    return float(np.mean(vals)) if vals else None


def job_team_ids(spec, config):
    experiment = config["experiment"]
    make_adapter = spec.build(experiment, experiment["trainer"]["method"], experiment["trainer"]["seed"])[0]
    return np.asarray(make_adapter().oracle_team_ids())


def summarize_job(spec, folder: Path):
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    status = (json.loads((folder / "status.json").read_text(encoding="utf-8")) if (folder / "status.json").exists()
              else dict(status="running_or_incomplete"))
    metrics, gates, evals = (read_jsonl(folder / f) for f in ("metrics.jsonl", "gate.jsonl", "eval.jsonl"))
    teams = job_team_ids(spec, config)
    warmup = config["trainer"]["warmup_updates"]
    out = dict(job=folder.name, method=config["trainer"]["method"], seed=config["trainer"]["seed"],
               status=status.get("status"), updates=len(metrics),
               env_steps=metrics[-1]["env_steps"] if metrics else 0, seconds=status.get("seconds"),
               team_ids=teams.tolist())
    if evals:
        steps = [e["env_steps"] for e in evals]
        out.update(final_eval_return=evals[-1]["eval_return_mean"],
                   final_eval_return_per_agent=evals[-1]["eval_return_per_agent"],
                   eval_return_auc=learning_auc(steps, [e["eval_return_mean"] for e in evals]),
                   **{f"final_eval_{k}": evals[-1].get(f"eval_{k}_total", 0.0) for k in spec.headline_events},
                   **{k: v for k, v in evals[-1].items() if k.startswith("eval_team")})
    returns = [r["episode_return_mean"] for r in metrics if "episode_return_mean" in r]
    out["train_return_last10"] = float(np.mean(returns[-10:])) if returns else None
    out.update(donor_arcs_per_update=mean_of(metrics, "max_gae_donor_arcs", warmup),
               cross_team_donor_arcs_per_update=mean_of(metrics, "max_gae_cross_team_arcs", warmup),
               max_gae_mean_b=mean_of(metrics, "max_gae_mean_b"), max_gae_select_frac=mean_of(metrics, "max_gae_select_frac"),
               max_gae_adv_shift=mean_of(metrics, "max_gae_adv_shift"),
               max_gae_eta_eff=mean_of(metrics, "max_gae_eta_eff"),
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


def aggregate(spec, plan: Path):
    jobs = [summarize_job(spec, p.parent) for p in sorted(Path(plan).glob("*/config.json"))]
    keys = BASE_METRICS + tuple(f"final_eval_{k}" for k in spec.headline_events)
    by_method = {}
    for job in jobs:
        by_method.setdefault(job["method"], []).append(job)
    table = {m: {k: mean_std([j.get(k) for j in members]) for k in keys} | dict(seeds=sorted(j["seed"] for j in members))
             for m, members in sorted(by_method.items())}
    (plan / "summary.json").write_text(json.dumps(dict(jobs=jobs, by_method=table), indent=2), encoding="utf-8")
    fields = sorted({k for j in jobs for k in j if not isinstance(j[k], (list, dict))})
    with (plan / "jobs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(jobs)

    def fmt(s):
        if s["mean"] is None:
            return "-"
        return f"{s['mean']:.3g}" + (f" +/- {s['std']:.2g}" if s["std"] is not None else "")

    cols = TABLE_COLUMNS[:2] + tuple(f"final_eval_{k}" for k in spec.headline_events) + TABLE_COLUMNS[2:]
    lines = [f"# {spec.name} PGCT comparison", "",
             "Mean +/- std over training seeds (ddof=1). Identification uses ground-truth teams in evaluation only. "
             "Unfinished jobs are included with the rows logged so far; check the status column in jobs.csv.", "",
             "| Method | Seeds | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for method, row in table.items():
        lines.append(f"| {method} | {row['seeds']} | " + " | ".join(fmt(row[c]) for c in cols) + " |")
    (plan / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jobs, table


def aggregate_main(spec, argv=None):
    parser = argparse.ArgumentParser(description=f"{spec.name}: summarise a plan folder (summary.md/json, jobs.csv)")
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args(argv)
    jobs, _ = aggregate(spec, args.plan)
    print((args.plan / "summary.md").read_text(encoding="utf-8"))
    print(f"{len(jobs)} jobs -> {args.plan / 'summary.json'}")
    return 0
