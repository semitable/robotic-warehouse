"""Experiment B: does critic sharing with the *true* teammates help? Training runner.

Matched branches from the same Stage03 weights, environment, PPO settings,
canonical probe bank and evaluation seeds as the completed E1/E2 PGCT/IPPO runs.
Only the donor set of the existing PGCT critic peer loss changes:

  oracle  same-team physical neighbours (ground-truth labels = upper bound of team inference)
  all     every physical neighbour (no team selection)
  wrong   cross-team physical neighbours only

The original trainer is reused unchanged through its existing oracle/unrestricted/
wrong graph modes. Transfers start after the same warm-up as PGCT (100 updates by
default). The completed PGCT/IPPO runs are read, never written.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

try:
    from . import common as C
except ImportError:
    import common as C

from examples import train_recurrent_ippo_consensus as T
from examples.train_pgct_spatial_experiments import build_jobs, experiment_logging, geometric_adjacency
from examples.train_pgct_transfer_experiments import config_differences
from examples.diagnose_pgct_rollout_compatibility import model_hash

MODE_GRAPH = {"oracle": "oracle", "all": "unrestricted", "wrong": "wrong"}
UPPER = np.triu(np.ones((4, 4), dtype=bool), 1)


def team_matrix(team_ids):
    team_ids = np.asarray(team_ids)
    same = team_ids[:, None] == team_ids[None, :]
    np.fill_diagonal(same, False)
    return same


def communication_range(env, cfg):
    return int(getattr(env.unwrapped, "communication_range", cfg.sensor_range))


def allowed_donors(mode, team_ids, neighbors):
    """receiver x donor support permitted for a mode inside the current physical range."""
    same = team_matrix(team_ids)
    cross = ~same
    np.fill_diagonal(cross, False)
    allowed = {"oracle": same, "all": same | cross, "wrong": cross, "pgct": same | cross,
               "random": same | cross, "ippo": np.zeros_like(same)}[mode]
    return allowed & np.asarray(neighbors, dtype=bool)


def check_allocation(mode, allocation, team_ids, neighbors):
    support = np.asarray(allocation) > 0
    if np.any(support & ~allowed_donors(mode, team_ids, neighbors)):
        raise RuntimeError(f"{mode}: transfer outside the permitted team/physical donor set.")


def build_matched_jobs(args, output):
    checkpoint = T._load_checkpoint(str(args.init_checkpoint), torch.device("cpu"))
    templates = build_jobs(checkpoint, str(args.init_checkpoint), output, maps=args.maps, modes=["pgct"],
                           seeds=args.seeds, timesteps=args.timesteps, track=args.track,
                           save_video=args.save_video, smoke=args.smoke_test)
    del checkpoint
    reference = json.loads((args.reference_experiment / "experiment.json").read_text(encoding="utf-8"))
    jobs = []
    for template in templates:
        ref = next((j for j in reference["jobs"] if j["map"] == template["map"]
                    and j["seed"] == template["seed"] and j["mode"] == "pgct"), None)
        bank = args.reference_experiment / f"{template['map']}_pgct_s{template['seed']}" / "canonical_probes.pt"
        if not args.smoke_test:
            if ref is None:
                raise ValueError(f"No reference PGCT run for {template['map']} seed {template['seed']}.")
            differences = config_differences(ref["config"], T.TrainConfig(**template["config"]))
            if template["effective_timesteps"] < ref["effective_timesteps"]:
                differences = [d for d in differences if d != "total_timesteps"]
            if differences:
                raise ValueError(f"Template differs from the reference PGCT run: {differences}")
            if not bank.exists():
                raise ValueError(f"Missing reference canonical bank: {bank}")
        for mode in args.modes:
            job = copy.deepcopy(template)
            name = f"{template['map']}_{mode}_s{template['seed']}"
            job["config"].update(graph_mode=MODE_GRAPH[mode], exp_name=name, wandb_name=name)
            warmup = job["config"]["pgct_warmup_updates"] if args.transfer_warmup_updates is None \
                else args.transfer_warmup_updates
            job.update(name=name, mode=mode, output_checkpoint=str(output / name / "final.pt"),
                       transfer_warmup_updates=int(warmup),
                       canonical_bank=str(bank.resolve()) if bank.exists() else None,
                       reference_runs={m: str((args.reference_experiment / f"{template['map']}_{m}_s{template['seed']}").resolve())
                                       for m in ("pgct", "ippo")})
            if not args.smoke_test:
                changed = config_differences(ref["config"], T.TrainConfig(**job["config"]))
                # A shorter budget is a matched prefix: the trainer has no learning-rate
                # schedule, so total_timesteps only sets the number of updates.
                allowed = ["graph_mode"] + (["total_timesteps"] if template["effective_timesteps"]
                                            < ref["effective_timesteps"] else [])
                if sorted(changed) != sorted(allowed):
                    raise ValueError(f"{name} must differ from the reference only in {allowed}: {changed}")
            jobs.append(job)
    return jobs


@contextmanager
def matched_transfer(cfg, run_dir, mode, *, bank_path, warmup_updates, allocation_override=None):
    """Preload the reference bank, hold transfers during warm-up and log actual allocations.

    allocation_override(update, neighbors, active) -> (allocation, log_fields) replaces the trainer's
    donor set (random-gated control); None keeps every existing mode unchanged."""
    bank, distill = None, None
    if bank_path:
        saved = torch.load(bank_path, map_location="cpu", weights_only=False)
        bank = (saved["sequences"].clone(), dict(saved["metadata"]))
        # Probe bank v2 keeps a separate critic-distillation set; the gate (JS) set stays the cached bank.
        if saved.get("transfer_sequences") is not None:
            distill = saved["transfer_sequences"].clone()
    originals = {name: getattr(T, name) for name in ("make_env", "collect_rollout", "update_ippo")}
    state = dict(update=0, env=None)
    clock = time.perf_counter()
    with (run_dir / "transfer.jsonl").open("w", encoding="utf-8", buffering=1) as log:
        def make_env(*args, **kwargs):
            env = originals["make_env"](*args, **kwargs)
            if bank is not None:
                if int(env.observation_space[0].shape[-1]) != bank[0].shape[-1]:
                    raise ValueError("Reference canonical bank does not match the observation size.")
                env.unwrapped._pgct_canonical_bank = (bank[0].clone(), dict(bank[1]))
            return env

        def collect(env, *args, **kwargs):
            state["env"] = env
            return originals["collect_rollout"](env, *args, **kwargs)

        def update(agents, optimizers, rollout, config, graph_estimator, device, rng,
                   peer_allocation=None, peer_probe_sequences=None):
            state["update"] += 1
            if state["update"] == 1:
                # Collecting a rollout does not change weights: this is the initial model.
                (run_dir / "initial_weights_sha256.txt").write_text(model_hash(agents) + "\n", encoding="utf-8")
            n_agents = len(state["env"].unwrapped.agents)
            gate = np.zeros((n_agents, n_agents), dtype=np.float32) if peer_allocation is None                 else np.asarray(peer_allocation)
            held = state["update"] <= warmup_updates
            actual = np.zeros_like(gate) if held else gate.copy()
            env = state["env"]
            team_ids = env.unwrapped.get_oracle_team_assignments()
            # The trainer's peer graph uses the environment's communication range (equal to the sensor range
            # unless set separately); donors are audited, and random-gated drawn, within the same range.
            neighbors = geometric_adjacency(env, communication_range(env, cfg))
            extra = {}
            if allocation_override is not None:
                actual, extra = allocation_override(state["update"], neighbors,
                                                    active=not held and peer_allocation is not None)
                actual = np.asarray(actual, dtype=gate.dtype)
            check_allocation(mode, actual, team_ids, neighbors)
            same = team_matrix(team_ids)
            log.write(json.dumps(dict(env_steps=state["update"] * cfg.rollout_steps, update=state["update"],
                                      mode=mode, matrix_convention="receiver_donor", warmup_hold=held,
                                      trainer_allocation=gate.tolist(), actual_allocation=actual.tolist(),
                                      physical_candidates=neighbors.tolist(),
                                      same_team_arcs=int(np.sum((actual > 0) & same)),
                                      cross_team_arcs=int(np.sum((actual > 0) & ~same)),
                                      distill_probe_count=None if distill is None or peer_probe_sequences is None
                                      else int(len(distill)), **extra)) + "\n")
            metrics = originals["update_ippo"](agents, optimizers, rollout, config, graph_estimator, device, rng,
                                                peer_allocation=None if peer_allocation is None else actual,
                                                peer_probe_sequences=(distill.clone() if distill is not None
                                                                      and peer_probe_sequences is not None
                                                                      else peer_probe_sequences))
            candidate_arcs = int(neighbors.sum())
            metrics.update(same_team_arcs=float(np.sum((actual > 0) & same)),
                           cross_team_arcs=float(np.sum((actual > 0) & ~same)),
                           physical_candidate_arcs=float(candidate_arcs),
                           gate_ratio=float(np.count_nonzero(actual) / candidate_arcs) if candidate_arcs else 0.0,
                           actual_transfer_arcs=float(np.count_nonzero(actual)),
                           actual_allocation_mass=float(actual.sum()), transfer_warmup_hold=float(held),
                           elapsed_seconds=time.perf_counter() - clock)
            return metrics

        T.make_env, T.collect_rollout, T.update_ippo = make_env, collect, update
        try:
            yield
        finally:
            for name, func in originals.items():
                setattr(T, name, func)


def transfer_summary(run_dir):
    rows = C.read_jsonl(Path(run_dir) / "transfer.jsonl")
    summary = dict(updates=len(rows), warmup_held_updates=sum(r["warmup_hold"] for r in rows),
                   same_team_arc_updates=sum(r["same_team_arcs"] for r in rows),
                   cross_team_arc_updates=sum(r["cross_team_arcs"] for r in rows))
    if rows:
        support = np.asarray([r["actual_allocation"] for r in rows]) > 0
        candidates = np.asarray([r["physical_candidates"] for r in rows], dtype=bool)
        n = support.shape[-1]
        pairs = list(zip(C.PAIR_LABELS, C.PAIRS)) if n == 4 else             [(f"{i + 1}-{j + 1}", (i, j)) for i in range(n) for j in range(i + 1, n)]
        summary.update(directed_transfer_update_counts=support.sum(0).tolist(),
                       undirected_pair_updates={label: int((support[:, i, j] | support[:, j, i]).sum())
                                                for label, (i, j) in pairs},
                       physical_candidate_pair_updates={label: int(candidates[:, i, j].sum())
                                                        for label, (i, j) in pairs})
    return summary


def compare(manifest, output):
    """Read-only comparison with the completed PGCT/IPPO references."""
    table = {}
    for job in manifest["jobs"]:
        key = f"{job['map']}_s{job['seed']}"
        group = table.setdefault(key, {})
        for mode, run_dir in job["reference_runs"].items():
            if mode not in group and (Path(run_dir) / "metrics.jsonl").exists():
                # References are cut at this branch's budget so every row covers the same steps.
                group[mode] = dict(C.learning_summary(run_dir, max_step=job["effective_timesteps"]),
                                   source="reference (read-only)")
        if (output / job["name"] / "metrics.jsonl").exists():
            group[job["mode"]] = dict(C.learning_summary(output / job["name"]), status=job["status"],
                                      transfer=transfer_summary(output / job["name"]))
    (output / "comparison.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    lines = ["# Experiment B comparison", "",
             "Deterministic masked evaluation, 20 fixed reset seeds, 500 steps; AUC = trapezoid over logged "
             "evaluations / covered steps (no step-0 score). Reference runs are truncated at the branch budget. "
             "One training seed per row is not a replication.", "",
             "| Branch | Mode | Last eval step | Delivery AUC/step | Last 10 evals | Last delivery | Same-team arc updates | Cross-team arc updates |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for key, group in table.items():
        for mode in ("ippo", "pgct", "oracle", "all", "wrong"):
            if mode not in group:
                continue
            row = group[mode]
            transfer = row.get("transfer", {})
            fmt = lambda v: "—" if v is None else f"{v:.3f}"
            lines.append(f"| {key} | {mode} | {row.get('last_eval_step', '—')} | {fmt(row.get('delivery_auc_per_step'))} | "
                         f"{fmt(row.get('last10_deliveries'))} | {fmt(row.get('last_deliveries'))} | "
                         f"{transfer.get('same_team_arc_updates', '—')} | {transfer.get('cross_team_arc_updates', '—')} |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init-checkpoint", type=Path, default=C.STAGE03)
    parser.add_argument("--reference-experiment", type=Path, default=C.REFERENCE_EXPERIMENT)
    parser.add_argument("--maps", nargs="+", choices=["e1", "e2"], default=["e1"])
    parser.add_argument("--modes", nargs="+", choices=list(MODE_GRAPH), default=["oracle", "all"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--timesteps", type=int, default=4000000)
    parser.add_argument("--transfer-warmup-updates", type=int,
                        help="Default: the reference PGCT warm-up (100 updates; 0 in smoke tests).")
    parser.add_argument("--eval-seed", type=int, default=5000001)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--compare-only", type=Path, help="Rebuild comparison.md for an existing output folder.")
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.compare_only:
        manifest = json.loads((args.compare_only / "experiment.json").read_text(encoding="utf-8"))
        compare(manifest, args.compare_only.resolve())
        print(f"Comparison: {args.compare_only.resolve() / 'comparison.md'}", flush=True)
        return
    if len(set(args.modes)) != len(args.modes) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Modes and seeds must be unique.")
    if args.transfer_warmup_updates is not None and args.transfer_warmup_updates < 0:
        parser.error("Warm-up must be nonnegative.")
    if args.smoke_test and args.transfer_warmup_updates is None:
        args.transfer_warmup_updates = 0
    os.environ.setdefault("WANDB_DISABLE_GIT", "true")
    output = (args.output_dir or C.ROOT / "runs" / ("lt_b_" + datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    if output.exists():
        parser.error(f"Output already exists: {output}")
    try:
        jobs = build_matched_jobs(args, output)
    except ValueError as exc:
        parser.error(str(exc))
    output.mkdir(parents=True)
    manifest = dict(initial_checkpoint=str(args.init_checkpoint.resolve()),
                    reference_experiment=str(args.reference_experiment.resolve()), eval_seed=args.eval_seed,
                    smoke_test=args.smoke_test, jobs=jobs,
                    note="Conditional continuation from one Stage03 model; optimizer/graph restart. Only the donor "
                         "set of the existing critic peer loss differs from the reference PGCT run.")
    manifest_path = output / "experiment.json"
    save = lambda: manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    save()
    for job in jobs:
        print(f"{job['name']}: graph_mode={job['config']['graph_mode']} warmup={job['transfer_warmup_updates']} "
              f"{job['effective_timesteps']} steps", flush=True)
    print(f"Plan: {manifest_path}", flush=True)
    if args.dry_run:
        return
    if args.smoke_test:
        torch.set_num_threads(1)
    for job in jobs:
        cfg = T.TrainConfig(**job["config"])
        run_dir = output / job["name"]
        job["status"] = "running"
        save()
        try:
            with experiment_logging(cfg, run_dir, args.eval_seed):
                with matched_transfer(cfg, run_dir, job["mode"], bank_path=job["canonical_bank"],
                                      warmup_updates=job["transfer_warmup_updates"]):
                    T.train(cfg)
            if not Path(job["output_checkpoint"]).is_file():
                raise RuntimeError("Training did not produce final.pt")
            job["transfer_summary"] = transfer_summary(run_dir)
        except BaseException as exc:
            job.update(status="failed", error=str(exc))
            save()
            raise
        job["status"] = "complete"
        save()
        compare(manifest, output)
    print(f"Experiment B complete: {output}", flush=True)


if __name__ == "__main__":
    main()
