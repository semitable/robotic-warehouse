"""Stage03 -> E1 4M branches: PGCT, IPPO, dose-matched random and edge ablations.

The original trainer and spatial runner stay intact. Only this process's update
call is wrapped to substitute the documented ablation allocation. All diagnostic
forwards and local random draws are independent of the PPO/probe RNG streams.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

try:
    from . import train_recurrent_ippo_consensus as T
    from .train_pgct_spatial_experiments import ROOT, DEFAULT_CHECKPOINT, build_jobs, experiment_logging, geometric_adjacency
    from .pgct_probe_diagnostics import PAIRS, common_probe_response, critic_rollouts
    from .plot_pgct_probe_charts import read_jsonl, plot_run, plot_experiment, probe_chart
except ImportError:
    import train_recurrent_ippo_consensus as T
    from train_pgct_spatial_experiments import ROOT, DEFAULT_CHECKPOINT, build_jobs, experiment_logging, geometric_adjacency
    from pgct_probe_diagnostics import PAIRS, common_probe_response, critic_rollouts
    from plot_pgct_probe_charts import read_jsonl, plot_run, plot_experiment, probe_chart

MODES = ["pgct", "ippo", "random", "drop34", "drop34_refill"]
MATCH_KEYS = ["seed", "total_timesteps", "rollout_steps", "sequence_length", "minibatch_chunks",
              "learning_rate", "entropy_coef", "gamma", "gae_lambda", "ppo_epochs", "action_mask",
              "clip_coef", "value_clip_coef", "value_loss_coef", "normalize_advantages", "min_advantage_std",
              "skip_zero_reward_actor_update", "skip_no_positive_reward_actor_update", "reward_epsilon",
              "max_grad_norm", "mlp_hidden_dim", "recurrent_hidden_dim", "obs_encoder", "observation_format",
              "pgct_peer_loss_coef", "probe_interval", "policy_probe_batch_size", "probe_source",
              "fixed_canonical_probes", "policy_probe_team_conditioning", "pgct_probe_sequence_length",
              "pgct_distance_threshold", "pgct_distance_ema_beta", "pgct_warmup_updates", "min_probe_count",
              "pgct_min_policy_information", "pgct_alpha_epsilon", "pgct_gate_style",
              "communication_enabled", "graph_mode", "comm_graph_mode", "peer_transfer_mode",
              "critic_consensus_tau", "consensus_interval", "objective_probe_fraction",
              "pgct_distance_temperature", "pgct_gate_power", "policy_similarity_temperature",
              "eval_episodes", "eval_horizon", "eval_interval"]


def config_differences(reference, cfg):
    differences = [k for k in MATCH_KEYS if reference[k] != getattr(cfg, k)]
    if json.loads(reference["env_kwargs_json"]) != json.loads(cfg.env_kwargs_json):
        differences.append("env_kwargs_json")
    if Path(reference["init_checkpoint"]).resolve() != Path(cfg.init_checkpoint).resolve():
        differences.append("init_checkpoint")
    return differences


def ablation_allocation(mode, allocation, neighbors, rng, reference=None):
    """Receiver x donor. Match allocation mass, not unknown auxiliary gradient size."""
    allocation = np.asarray(allocation, dtype=np.float32).copy()
    neighbors = np.asarray(neighbors, dtype=bool).copy()
    np.fill_diagonal(neighbors, False)
    result = allocation.copy()
    requested = allocation if reference is None else np.asarray(reference, dtype=np.float32)
    if mode == "ippo":
        result.fill(0)
    elif mode == "random":
        if reference is None:
            raise ValueError("Random donor needs a completed matched PGCT reference schedule.")
        result.fill(0)
        for receiver in range(4):
            donors = np.flatnonzero(neighbors[receiver])
            count = min(len(donors), int(np.count_nonzero(requested[receiver])))
            if count:
                selected = rng.choice(donors, size=count, replace=False)
                # Preserve individual weights whenever enough donors exist;
                # otherwise preserve receiver mass, and record the count shortfall.
                weights = requested[receiver][requested[receiver] > 0].copy()
                if len(weights) != count:
                    weights = np.full(count, weights.sum() / count)
                result[receiver, selected] = rng.permutation(weights)
    elif mode in {"drop34", "drop34_refill"}:
        result[2, 3] = result[3, 2] = 0
        if mode == "drop34_refill":
            for receiver, removed_donor in ((2, 3), (3, 2)):
                candidates = np.flatnonzero(neighbors[receiver])
                candidates = candidates[candidates != removed_donor]
                if allocation[receiver, removed_donor] > 0 and len(candidates):
                    result[receiver, rng.choice(candidates)] += allocation[receiver, removed_donor]
    elif mode != "pgct":
        raise ValueError(f"Unknown mode: {mode}")
    np.fill_diagonal(result, 0)
    if np.any(result[~neighbors] > 0):
        raise ValueError("Ablation attempted transfer outside the current physical neighborhood.")
    return result, dict(requested_receiver_mass=requested.sum(1).tolist(),
                        actual_receiver_mass=result.sum(1).tolist(),
                        donor_count_shortfall=np.maximum((requested > 0).sum(1) - (result > 0).sum(1), 0).tolist(),
                        allocation_mass_shortfall=np.maximum(requested.sum(1) - result.sum(1), 0).tolist())


def matched_reference(experiment, cfg, *, allow_incomplete=False):
    manifest = json.loads((Path(experiment) / "experiment.json").read_text(encoding="utf-8"))
    matches = [job for job in manifest["jobs"] if job["mode"] == "pgct" and job["map"] == "e1" and job["seed"] == cfg.seed]
    if len(matches) != 1:
        raise ValueError(f"No unique E1 PGCT reference for training seed {cfg.seed} in {experiment}")
    job = matches[0]
    ref_cfg = job["config"]
    differences = config_differences(ref_cfg, cfg)
    if differences:
        raise ValueError(f"Reference conditions differ: {differences}")
    run_dir = Path(job.get("reused_run_dir", Path(experiment) / job["name"]))
    if not allow_incomplete and (job["status"] not in {"complete", "reused"} or not (run_dir / "final.pt").exists()):
        raise ValueError(f"Reference is still running/incomplete: {run_dir}. Wait for its completion.")
    return job, run_dir


def reference_schedule(run_dir, cfg):
    actual = read_jsonl(run_dir / "transfer.jsonl")
    if actual:
        schedule = {row["env_steps"]: np.asarray(row["actual_allocation"], dtype=np.float32) for row in actual}
    else:
        schedule = {}
        for row in read_jsonl(run_dir / "communication.jsonl"):
            gate = np.asarray(row["transfer_weights"], dtype=np.float32)
            schedule[row["env_steps"]] = gate / (gate.sum(1, keepdims=True) + max(cfg.pgct_alpha_epsilon, 1e-8))
    expected = set(range(cfg.rollout_steps, int(np.ceil(cfg.total_timesteps / cfg.rollout_steps)) * cfg.rollout_steps + 1,
                         cfg.rollout_steps))
    if set(schedule) != expected:
        raise ValueError("PGCT reference must contain the complete matched per-update transfer schedule.")
    return schedule


@contextmanager
def transfer_experiment(cfg, run_dir, mode, *, eval_seed, diagnostic_interval, plot_interval,
                        critic_episodes, schedule=None):
    raw_make_env = T.make_env
    clock_start = time.perf_counter()
    with experiment_logging(cfg, run_dir, eval_seed):
        originals = {name: getattr(T, name) for name in ("collect_rollout", "update_ippo", "evaluate", "log_to_wandb")}
        state = dict(update=0, env=None, latest=None, initial=False, agents=None, device=None)
        donor_rng = np.random.default_rng(cfg.seed + 781203)
        diagnostic_cfg = copy.deepcopy(cfg)
        diagnostic_cfg.seed = 1
        diagnostic_cfg.probe_source = "objective"
        diagnostic_cfg.policy_probe_team_conditioning = "shared"
        diagnostic_cfg.fixed_canonical_probes = True
        diagnostic_cfg.policy_probe_batch_size = 8 if cfg.total_timesteps == 64 else 48
        diagnostic_cfg.pgct_probe_sequence_length = 1
        # One bank is shared across methods and training seeds; never use their rollout states.
        reference_kwargs = json.loads(cfg.env_kwargs_json)
        reference_kwargs.update(communication_enabled=True, communication_topology="physical")
        with raw_make_env(cfg.env_id, reference_kwargs, cfg.observation_format) as reference_env:
            reference_env.reset(seed=1)
            bank, metadata = T.sample_policy_probe_sequences(diagnostic_cfg, reference_env, None,
                                                             np.random.default_rng(9174))
        torch.save(dict(sequences=bank, metadata=metadata, diagnostic_only=True), run_dir / "diagnostic_probes.pt")
        with (run_dir / "probes.jsonl").open("w", encoding="utf-8", buffering=1) as probes, \
                (run_dir / "transfer.jsonl").open("w", encoding="utf-8", buffering=1) as transfers, \
                (run_dir / "critic_eval.jsonl").open("w", encoding="utf-8", buffering=1) as critic_log:
            def diagnostic(agents, device, timing="before_current_update"):
                row = common_probe_response(agents, bank, device)
                row.update(env_steps=state["update"] * cfg.rollout_steps, update=state["update"],
                           model_timing=timing)
                probes.write(json.dumps(row) + "\n")
                state["latest"] = row
                return row

            def evaluate(*args, **kwargs):
                result, video = originals["evaluate"](*args, **kwargs)
                if critic_episodes:
                    # Existing evaluate signature: agents at 3, device at 9; fixed seeds for all branches.
                    record = critic_rollouts(cfg, args[3], args[1], args[9], episodes=critic_episodes,
                                             horizon=cfg.eval_horizon, seed=eval_seed + 100000, make_env=raw_make_env)
                    record.update(env_steps=state["update"] * cfg.rollout_steps)
                    critic_log.write(json.dumps(record) + "\n")
                    result.update({"eval_" + k: v for k, v in record["metrics"].items()})
                result["eval_elapsed_seconds"] = time.perf_counter() - clock_start
                for event in ("delivery", "return", "pickup"):
                    # Original per-agent evaluation counts sum over all evaluation episodes.
                    for i in range(4):
                        result[f"eval_agent_{i+1}_{event}_per_episode"] = result[f"eval_agent_{i+1}_{event}_count"] / args[4]
                    for team in range(2):
                        result[f"eval_team_{team}_{event}_per_episode"] = sum(
                            result[f"eval_agent_{i+1}_{event}_per_episode"] for i in range(4) if i % 2 == team)
                return result, video

            def collect(env, agents, runner_state, *args, **kwargs):
                state["env"] = env
                state["agents"] = agents
                state["device"] = next(agents[0].parameters()).device
                if not state["initial"]:
                    device = next(agents[0].parameters()).device
                    diagnostic(agents, device)
                    initial, _ = evaluate(cfg.env_id, json.loads(cfg.env_kwargs_json), cfg.observation_format,
                                          agents, cfg.eval_episodes, cfg.eval_horizon, cfg.gamma,
                                          env.observation_space[0].shape[0], cfg.recurrent_hidden_dim, device,
                                          seed=eval_seed, use_action_mask=cfg.action_mask)
                    originals["log_to_wandb"](None, T.eval_payload_from_metrics(initial), step=0)
                    state["initial"] = True
                result = originals["collect_rollout"](env, agents, runner_state, *args, **kwargs)
                state["update"] += 1
                return result

            def update(agents, optimizers, rollout, config, graph_estimator, device, rng,
                       peer_allocation=None, peer_probe_sequences=None):
                steps = state["update"] * cfg.rollout_steps
                allocation = np.zeros((4, 4), dtype=np.float32) if peer_allocation is None else peer_allocation.copy()
                neighbors = geometric_adjacency(state["env"], cfg.sensor_range)
                actual, exposure = ablation_allocation(mode, allocation, neighbors, donor_rng,
                                                      None if schedule is None else schedule[steps])
                if actual.any() and peer_probe_sequences is None:
                    peer_probe_sequences, _ = T.sample_policy_probe_sequences(cfg, state["env"], rollout,
                                                                               np.random.default_rng(cfg.seed + 9173))
                peer_mse = np.zeros((4, 4))
                if actual.any():
                    values = T.recurrent_critic_probe_targets(agents, peer_probe_sequences, device).cpu().numpy()
                    for i, j in zip(*np.nonzero(actual)):
                        peer_mse[i, j] = float(((values[i] - values[j]) ** 2).mean())
                transfers.write(json.dumps(dict(env_steps=steps, update=state["update"], mode=mode,
                    matrix_convention="receiver_donor", local_pgct_allocation=allocation.tolist(),
                    actual_allocation=actual.tolist(), physical_candidates=neighbors.tolist(),
                    pre_update_peer_mse=peer_mse.tolist(), peer_loss_coefficient=cfg.pgct_peer_loss_coef,
                    **exposure)) + "\n")
                if state["update"] % diagnostic_interval == 0:
                    diagnostic(agents, device)
                metrics = originals["update_ippo"](agents, optimizers, rollout, config, graph_estimator, device, rng,
                                                    peer_allocation=actual, peer_probe_sequences=peer_probe_sequences)
                metrics.update(actual_transfer_arcs=float(np.count_nonzero(actual)),
                               actual_allocation_mass=float(actual.sum()),
                               allocation_mass_shortfall=float(sum(exposure["allocation_mass_shortfall"])),
                               donor_count_shortfall=float(sum(exposure["donor_count_shortfall"])),
                               elapsed_seconds=time.perf_counter() - clock_start)
                if state["update"] % plot_interval == 0:
                    plot_run(run_dir, run_dir / "charts")
                return metrics

            def log(run, payload, step):
                payload = dict(payload)
                # Keep a normal eval endpoint too, so AUC covers the final budget.
                if "eval/final_deliveries" in payload:
                    payload.update({"eval/" + k.removeprefix("eval/final_"): v
                                    for k, v in list(payload.items()) if k.startswith("eval/final_")})
                if "train/env_steps" in payload and state["latest"] is not None:
                    row = state["latest"]
                    payload["diagnostic/probe_sample_env_steps"] = row["env_steps"]
                    for i, j in PAIRS:
                        payload[f"diagnostic/js_{i+1}_{j+1}"] = row["mean_js"][i][j]
                        payload[f"diagnostic/unique_js_{i+1}_{j+1}"] = row["unique_mean_js"][i][j]
                    image = run_dir / "charts" / "response.png"
                    if run is not None and state["update"] % plot_interval == 0 and image.exists():
                        import wandb
                        payload["diagnostic/probe_color_chart"] = wandb.Image(str(image))
                return originals["log_to_wandb"](run, payload, step=step)

            T.collect_rollout, T.update_ippo, T.evaluate, T.log_to_wandb = collect, update, evaluate, log
            try:
                yield
                if state["agents"] is not None:
                    diagnostic(state["agents"], state["device"], timing="after_final_update")
            finally:
                for name, func in originals.items():
                    setattr(T, name, func)
        plot_run(run_dir, run_dir / "charts")


def summarize(run_dir, performance_target):
    rows = read_jsonl(run_dir / "metrics.jsonl")
    evaluations = {}
    for row in rows:
        if "eval/deliveries" in row["metrics"]:
            evaluations[row["env_steps"]] = row["metrics"]
    steps = sorted(evaluations)
    result = dict(evaluated_steps=steps, performance_target=performance_target)
    if len(steps) >= 2:
        keys = ["deliveries", "returns_home", "task_return"] + [
            k.removeprefix("eval/") for k in evaluations[steps[-1]]
            if k.startswith(("eval/agent_", "eval/team_")) and k.endswith("_per_episode")]
        for key in keys:
            y = np.asarray([evaluations[s]["eval/" + key] for s in steps])
            # Point estimates of trapezoidal AUC; no extrapolation beyond evaluated range.
            auc = float(np.sum(np.diff(steps) * (y[1:] + y[:-1]) / 2))
            result[key + "_auc"] = auc
            result[key + "_auc_per_step"] = auc / (steps[-1] - steps[0])
    result["first_evaluated_step_reaching_target"] = next(
        (s for s in steps if evaluations[s]["eval/deliveries"] >= performance_target), None)
    transfer = read_jsonl(run_dir / "transfer.jsonl")
    if transfer:
        weights = np.asarray([r["actual_allocation"] for r in transfer])
        result.update(directed_transfer_update_counts=(weights > 0).sum(0).tolist(),
                      allocation_weight_sum=weights.sum(0).tolist(),
                      allocation_mass_shortfall=sum(sum(r["allocation_mass_shortfall"]) for r in transfer),
                      donor_count_shortfall=sum(sum(r["donor_count_shortfall"]) for r in transfer))
    result["final_eval"] = json.loads((run_dir / "final_eval_metrics.json").read_text(encoding="utf-8"))
    (run_dir / "transfer_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--modes", choices=MODES, nargs="+", default=["pgct", "ippo"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--timesteps", type=int, default=4000000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reference-experiment", type=Path)
    parser.add_argument("--reuse-experiment", type=Path)
    parser.add_argument("--eval-seed", type=int, default=5000001)
    parser.add_argument("--critic-episodes", type=int, default=5)
    parser.add_argument("--diagnostic-interval", type=int, default=5)
    parser.add_argument("--plot-interval", type=int, default=50)
    parser.add_argument("--performance-target", type=float, default=45)
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if len(set(args.modes)) != len(args.modes) or min(args.diagnostic_interval, args.plot_interval) <= 0 or args.critic_episodes < 0:
        parser.error("Modes must be unique; intervals positive; critic episodes nonnegative.")
    if "random" in args.modes and "pgct" not in args.modes and not args.reference_experiment:
        parser.error("Random requires --modes pgct random, or --reference-experiment from a completed matched PGCT run.")
    os.environ.setdefault("WANDB_DISABLE_GIT", "true")
    output = args.output_dir.resolve() if args.output_dir else ROOT / "runs" / ("tr_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    checkpoint = T._load_checkpoint(args.init_checkpoint, torch.device("cpu"))
    jobs = []
    try:
        base_jobs = build_jobs(checkpoint, args.init_checkpoint, output, maps=["e1"], modes=["pgct", "ippo"],
                               seeds=args.seeds, timesteps=args.timesteps, track=args.track,
                               save_video=args.save_video, smoke=args.smoke_test)
        for seed in args.seeds:
            for mode in [m for m in MODES if m in args.modes]:
                template = next(j for j in base_jobs if j["seed"] == seed and j["mode"] == ("ippo" if mode == "ippo" else "pgct"))
                job = copy.deepcopy(template)
                job.update(mode=mode, name=f"e1_{mode}_s{seed}", output_checkpoint=str(output / f"e1_{mode}_s{seed}" / "final.pt"))
                job["config"].update(exp_name=job["name"], wandb_name=job["name"])
                if args.reuse_experiment and mode in {"pgct", "ippo"}:
                    old_manifest = json.loads((args.reuse_experiment / "experiment.json").read_text(encoding="utf-8"))
                    if old_manifest["eval_seed"] != args.eval_seed:
                        raise ValueError("Reused experiment has a different evaluation seed list.")
                    old = next((j for j in old_manifest["jobs"] if j["map"] == "e1" and j["seed"] == seed and j["mode"] == mode), None)
                    if old is not None:
                        differences = config_differences(old["config"], T.TrainConfig(**job["config"]))
                        if differences:
                            raise ValueError(f"Reused {mode} conditions differ: {differences}")
                        # For IPPO validate via its paired PGCT config (communication intentionally differs).
                        paired_cfg = T.TrainConfig(**next(j for j in base_jobs if j["seed"] == seed and j["mode"] == "pgct")["config"])
                        matched_reference(args.reuse_experiment, paired_cfg, allow_incomplete=args.dry_run)
                        old_dir = args.reuse_experiment / old["name"]
                        if not args.dry_run and (old["status"] != "complete" or not (old_dir / "final.pt").exists()):
                            raise ValueError(f"Reuse requested for an incomplete run: {old_dir}. Wait for completion.")
                        job.update(status="reused", reused_run_dir=str(old_dir.resolve()),
                                   output_checkpoint=str(old_dir.resolve() / "final.pt"))
                jobs.append(job)
    except ValueError as exc:
        parser.error(str(exc))
    del checkpoint
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(initial_checkpoint=str(Path(args.init_checkpoint).resolve()), jobs=jobs,
                    note="Conditional continuation from one Stage03 model; optimizer and graph restart. E1 only.",
                    reference_experiment=str(args.reference_experiment) if args.reference_experiment else None,
                    diagnostics=dict(common_reference_bank_seed=1, interval=args.diagnostic_interval,
                                     critic_episodes=args.critic_episodes, stochastic_critic_seed=args.eval_seed + 100000),
                    eval_seed=args.eval_seed, smoke_test=args.smoke_test)
    manifest_path = output / "experiment.json"
    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    save()
    for job in jobs:
        print(f"{job['name']}: {job['effective_timesteps']} steps; {job['status']}", flush=True)
    print(f"Plan: {manifest_path}", flush=True)
    if args.dry_run:
        return
    if args.smoke_test:
        torch.set_num_threads(1)
    for job in jobs:
        if job["status"] == "reused":
            continue
        cfg = T.TrainConfig(**job["config"])
        schedule = None
        if job["mode"] == "random":
            _, ref_dir = matched_reference(args.reference_experiment or output, cfg)
            schedule = reference_schedule(ref_dir, cfg)
            job["random_reference_run"] = str(ref_dir)
        job["status"] = "running"; save()
        try:
            with transfer_experiment(cfg, output / job["name"], job["mode"], eval_seed=args.eval_seed,
                                     diagnostic_interval=1 if args.smoke_test else args.diagnostic_interval,
                                     plot_interval=2 if args.smoke_test else args.plot_interval,
                                     critic_episodes=min(args.critic_episodes, 1) if args.smoke_test else args.critic_episodes,
                                     schedule=schedule):
                T.train(cfg)
            job["summary"] = summarize(output / job["name"], args.performance_target)
        except BaseException as exc:
            job.update(status="failed", error=str(exc)); save(); raise
        job["status"] = "complete"; save()
    plot_experiment(output, output / "charts")
    print(f"Experiments complete: {output}", flush=True)


if __name__ == "__main__":
    main()
