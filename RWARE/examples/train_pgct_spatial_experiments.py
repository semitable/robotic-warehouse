"""Matched spatial-expansion training branches using the existing IPPO/PGCT trainer."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

try:
    from . import train_recurrent_ippo_consensus as T
    from .train_pgct_physical_mixing_curriculum import build_stages
except ImportError:
    import train_recurrent_ippo_consensus as T
    from train_pgct_physical_mixing_curriculum import build_stages

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / (
    "runs/pgct09700_physical_mixing_20260929_155338_079437/"
    "stage03_4ag_physical_home0.750/final.pt"
)
# Generator pilot: shelf density also changes. These are actual dimensions,
# not claims of exact 1.5x/2x scaling or density-controlled layouts.
MAPS = {
    "e1": dict(shelf_rows=1, shelf_columns=3, column_height=8),
    "e2": dict(shelf_rows=2, shelf_columns=5, column_height=6),
    "e3": dict(shelf_rows=2, shelf_columns=7, column_height=8),
}


def geometric_adjacency(env, radius):
    """Physical candidates even when IPPO disables the communication subsystem."""
    positions = np.asarray([(a.x, a.y) for a in env.unwrapped.agents])
    adjacency = np.max(np.abs(positions[:, None] - positions[None, :]), axis=-1) <= radius
    np.fill_diagonal(adjacency, False)
    return adjacency


def build_jobs(checkpoint, source, output, *, maps, modes, seeds, timesteps,
               home_ratio=.5, requests=4, learning_rate=1e-4, entropy_coef=.02,
               track=False, save_video=True, smoke=False):
    if len(checkpoint["agents"]) != 4:
        raise ValueError("This experiment requires a four-agent checkpoint (no actor cloning).")
    if checkpoint["effective_env_kwargs"].get("layout"):
        raise ValueError("Generator map overrides cannot be applied to a custom layout checkpoint.")
    if checkpoint["effective_env_kwargs"].get("team_goal_positions"):
        raise ValueError("Explicit goal coordinates must be reviewed before changing map geometry.")
    if not .5 <= home_ratio <= 1 or requests < 2 or requests % 2:
        raise ValueError("Home ratio must be in [.5,1]; request count must be positive and even.")
    for values in (maps, modes, seeds):
        if not values or len(set(values)) != len(values):
            raise ValueError("Map, mode and seed lists must be nonempty and contain no duplicates.")
    if any(m not in MAPS for m in maps) or any(m not in {"pgct", "ippo"} for m in modes):
        raise ValueError("Unknown map or learning mode.")
    smallest_capacity = min(2 * MAPS[m]["column_height"] *
                            (MAPS[m]["shelf_rows"] * MAPS[m]["shelf_columns"] - 1) for m in maps)
    if requests > smallest_capacity:
        raise ValueError("Request count exceeds the shelf capacity of a selected map.")
    if any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be nonnegative.")
    jobs = []
    for seed in seeds:
        template = build_stages(
            checkpoint, source, output, home_ratios=[1.0], timesteps=[timesteps],
            seed=seed, learning_rate=learning_rate, entropy_coef=entropy_coef,
            track=track, save_video=save_video, smoke=smoke,
        )[0]["config"]
        for map_name in maps:
            for mode in modes:
                cfg = copy.deepcopy(template)
                name = f"{map_name}_{mode}_s{seed}"
                env = json.loads(cfg["env_kwargs_json"])
                env.update(MAPS[map_name], shelf_soft_zone_ratio=home_ratio,
                           request_queue_size=requests, request_queue_size_per_team=requests // 2)
                cfg.update(exp_name=name, wandb_name=name, init_checkpoint=str(Path(source).resolve()),
                           env_kwargs_json=json.dumps(env), log_interval=10)
                config = T.TrainConfig(**cfg)
                if mode == "ippo":
                    config.communication_enabled = False
                    T.enforce_communication_config(config)
                    env.update(communication_enabled=False, communication_topology="none")
                    config.env_kwargs_json = json.dumps(env)
                shape = [(env["column_height"] + 1) * env["shelf_rows"] + 2,
                         env["shelf_columns"] * 3 + 1]
                jobs.append(dict(name=name, map=map_name, mode=mode, seed=seed,
                                 grid_shape=shape, config=asdict(config), status="planned",
                                 effective_timesteps=math.ceil(config.total_timesteps / config.rollout_steps)
                                 * config.rollout_steps,
                                 output_checkpoint=str(Path(output) / name / "final.pt")))
    return jobs


class SpatialMonitor:
    def __init__(self, radius):
        self.radius = radius
        self.active = False
        self.reset()

    def reset(self):
        self.steps = 0
        self.occupancy = None
        self.candidates = np.zeros((4, 4), dtype=np.int64)
        self.phase_counts = np.zeros((4, 3), dtype=np.int64)

    def record(self, env):
        base = env.unwrapped
        if self.occupancy is None:
            self.occupancy = np.zeros((4, *base.grid_size), dtype=np.int64)
        self.candidates += geometric_adjacency(env, self.radius)
        for i, agent in enumerate(base.agents):
            self.occupancy[i, agent.y, agent.x] += 1
            carrying = agent.carrying_shelf
            phase = (2 if carrying.id in base._shelves_awaiting_return else 1) if carrying else 0
            self.phase_counts[i, phase] += 1
        self.steps += 1


class MonitoredEnv(gym.Wrapper):
    def __init__(self, env, monitor):
        super().__init__(env)
        self.monitor = monitor

    def step(self, actions):
        if self.monitor.active:
            self.monitor.record(self.env)
        return self.env.step(actions)


@contextmanager
def experiment_logging(cfg, run_dir, eval_seed):
    """Observe training without changing PPO, probe, gate or environment dynamics."""
    run_dir.mkdir(parents=True, exist_ok=False)
    monitor = SpatialMonitor(cfg.sensor_range)
    originals = {name: getattr(T, name) for name in
                 ("make_env", "collect_rollout", "evaluate", "log_to_wandb")}
    envs = []
    update = 0
    evaluating = False
    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8", buffering=1) as metrics, \
            (run_dir / "spatial.jsonl").open("w", encoding="utf-8", buffering=1) as spatial:
        def make_env(*args, **kwargs):
            if evaluating:
                return originals["make_env"](*args, **kwargs)
            env = MonitoredEnv(originals["make_env"](*args, **kwargs), monitor)
            envs.append(env)
            return env

        def collect(env, *args, **kwargs):
            nonlocal update
            monitor.reset()
            monitor.active = True
            try:
                result = originals["collect_rollout"](env, *args, **kwargs)
            finally:
                monitor.active = False
            update += 1
            width = monitor.occupancy.shape[-1]
            row = dict(update=update, env_steps=update * cfg.rollout_steps,
                       sample_convention="pre_action_per_environment_step",
                       sampled_steps=monitor.steps,
                       geometric_candidate_step_counts=monitor.candidates.tolist(),
                       update_end_geometric_candidates=geometric_adjacency(env, cfg.sensor_range).tolist(),
                       mean_physical_neighbors=float(monitor.candidates.sum() / (4 * monitor.steps)),
                       left_occupancy_fraction=(monitor.occupancy[:, :, :width // 2].sum((1, 2))
                                                / monitor.steps).tolist(),
                       phase_counts=monitor.phase_counts.tolist(),
                       occupancy_counts=monitor.occupancy.tolist())
            spatial.write(json.dumps(row) + "\n")
            return result

        def evaluate(*args, **kwargs):
            nonlocal evaluating
            # A fixed list, shared across maps, methods and checkpoints.
            kwargs["seed"] = eval_seed
            evaluating = True
            try:
                return originals["evaluate"](*args, **kwargs)
            finally:
                evaluating = False

        def log(run, payload, step):
            numeric = {key: float(value) for key, value in payload.items()
                       if isinstance(value, (int, float, np.number)) and np.isfinite(value)}
            metrics.write(json.dumps(dict(env_steps=step, metrics=numeric)) + "\n")
            return originals["log_to_wandb"](run, payload, step=step)

        T.make_env, T.collect_rollout = make_env, collect
        T.evaluate, T.log_to_wandb = evaluate, log
        try:
            yield
        finally:
            for name, func in originals.items():
                setattr(T, name, func)
            for env in envs:
                env.close()


def summarize_run(run_dir):
    upper = np.triu(np.ones((4, 4), dtype=bool), 1)
    same = upper & (np.arange(4)[:, None] % 2 == np.arange(4)[None, :] % 2)
    cross = upper & ~same
    stats = dict(updates=0, per_step_candidate_pair_instances=0,
                 same_candidate_update_instances=0, cross_candidate_update_instances=0,
                 same_accepted_update_instances=0, cross_accepted_update_instances=0)
    with (run_dir / "spatial.jsonl").open(encoding="utf-8") as sf, \
            (run_dir / "communication.jsonl").open(encoding="utf-8") as cf:
        from itertools import zip_longest
        for sline, cline in zip_longest(sf, cf):
            if sline is None or cline is None:
                raise RuntimeError("Spatial and communication logs have different update counts.")
            s, c = json.loads(sline), json.loads(cline)
            if s["env_steps"] != c["env_steps"]:
                raise RuntimeError("Spatial and communication update steps disagree.")
            candidates = np.asarray(s["update_end_geometric_candidates"], dtype=bool)
            accepted = np.asarray(c["transfer_weights"]) > 0
            if np.any(accepted & ~candidates):
                raise RuntimeError("Transfer occurred outside the geometric communication range.")
            stats["updates"] += 1
            stats["per_step_candidate_pair_instances"] += int(
                np.asarray(s["geometric_candidate_step_counts"])[upper].sum())
            for label, mask in (("same", same), ("cross", cross)):
                stats[f"{label}_candidate_update_instances"] += int(candidates[mask].sum())
                stats[f"{label}_accepted_update_instances"] += int(accepted[mask].sum())
    for label in ("same", "cross"):
        candidates = stats[f"{label}_candidate_update_instances"]
        stats[f"{label}_acceptance_rate"] = (
            stats[f"{label}_accepted_update_instances"] / candidates if candidates else None)
    stats["final_eval"] = json.loads((run_dir / "final_eval_metrics.json").read_text(encoding="utf-8"))
    (run_dir / "summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--maps", choices=list(MAPS), nargs="+", default=list(MAPS))
    parser.add_argument("--modes", choices=["pgct", "ippo"], nargs="+", default=["pgct", "ippo"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--timesteps", type=int, default=4000000)
    parser.add_argument("--home-ratio", type=float, default=.5)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--entropy-coef", type=float, default=.02)
    parser.add_argument("--eval-seed", type=int, default=5000001)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if os.name == "nt":
        os.environ.setdefault("WANDB_DISABLE_GIT", "true")
    output = Path(args.output_dir).resolve() if args.output_dir else ROOT / "runs" / (
        "sp_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    checkpoint = T._load_checkpoint(args.init_checkpoint, torch.device("cpu"))
    try:
        jobs = build_jobs(checkpoint, args.init_checkpoint, output, maps=args.maps,
                          modes=args.modes, seeds=args.seeds, timesteps=args.timesteps,
                          home_ratio=args.home_ratio, requests=args.requests,
                          learning_rate=args.learning_rate, entropy_coef=args.entropy_coef,
                          track=args.track, save_video=not args.no_video, smoke=args.smoke_test)
    except ValueError as exc:
        parser.error(str(exc))
    del checkpoint
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(initial_checkpoint=str(Path(args.init_checkpoint).resolve()),
                    note="Independent branches from identical pretrained weights; optimizer/graph restart. "
                         "Generator pilot changes shelf density as well as map dimensions.",
                    eval_seed=args.eval_seed, smoke_test=args.smoke_test, jobs=jobs)
    manifest_path = output / "experiment.json"

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    save_manifest()
    for job in jobs:
        print(f"{job['name']}: {job['grid_shape']}; {job['effective_timesteps']} steps", flush=True)
    print(f"Plan: {manifest_path}", flush=True)
    if args.dry_run:
        return
    if args.smoke_test:
        torch.set_num_threads(1)
    for job in jobs:
        cfg = T.TrainConfig(**job["config"])
        run_dir = output / job["name"]
        job["status"] = "running"
        save_manifest()
        try:
            with experiment_logging(cfg, run_dir, args.eval_seed):
                T.train(cfg)
            if not Path(job["output_checkpoint"]).is_file():
                raise RuntimeError("Training did not produce final.pt")
            job["summary"] = summarize_run(run_dir)
        except BaseException as exc:
            job.update(status="failed", error=str(exc))
            save_manifest()
            raise
        job["status"] = "complete"
        save_manifest()
    print(f"Experiments complete: {output}", flush=True)


if __name__ == "__main__":
    main()
