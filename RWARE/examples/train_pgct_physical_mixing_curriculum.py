"""Fine-tune ex5 09700 with local PGCT links, then progressively mixed shelves."""
from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

try:
    from . import train_recurrent_ippo_consensus as T
except ImportError:
    import train_recurrent_ippo_consensus as T

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "runs/recurrent_ippo_consensus/ex5_semantic_cnn_policy_complete_from_scratch_4ag_sr5_square_home1.00_seed1/checkpoint_09700.pt"


def build_stages(checkpoint, source_path, output_dir, *, home_ratios, timesteps,
                 agent_count=None, seed=1, learning_rate=1e-4, entropy_coef=0.02,
                 track=False, save_video=True, smoke=False):
    if not home_ratios or home_ratios[0] != 1.0:
        raise ValueError("The first stage must use home ratio 1.0 (physical adaptation).")
    if len(home_ratios) != len(timesteps):
        raise ValueError("Provide one timestep budget for each home ratio.")
    if any(not 0.5 <= r <= 1 for r in home_ratios):
        raise ValueError("Home ratios must be in [0.5, 1.0].")
    if any(b > a for a, b in zip(home_ratios, home_ratios[1:])):
        raise ValueError("Home ratios must be non-increasing.")
    if any(t < 1 for t in timesteps) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Timestep budgets and learning rate must be positive.")
    if not math.isfinite(entropy_coef) or entropy_coef < 0:
        raise ValueError("Entropy coefficient must be finite and non-negative.")
    source_cfg = checkpoint["config"]
    if source_cfg["observation_format"] != "semantic" or source_cfg["obs_encoder"] != "cnn":
        raise ValueError("This curriculum requires a semantic CNN checkpoint.")
    source_n = len(checkpoint["agents"])
    n = source_n if agent_count is None else agent_count
    if source_n % 2 or n < 2 or n % 2:
        raise ValueError("Two-team policy transfer requires even source and target agent counts.")
    base_env = copy.deepcopy(checkpoint["effective_env_kwargs"])
    if base_env.get("n_teams", 2) != 2:
        raise ValueError("Balanced shelf mixing requires a two-team checkpoint.")
    sensor_range = int(base_env.get("sensor_range", source_cfg["sensor_range"]))
    # Start from today's defaults, then inherit the saved architecture and training
    # settings. Explicit overrides below define this curriculum's differences.
    defaults = vars(T.build_arg_parser().parse_args([]))
    defaults.update({k: v for k, v in source_cfg.items() if k in defaults})
    output_dir = Path(output_dir).resolve()
    previous = Path(source_path).resolve()
    stages = []
    for index, (ratio, budget) in enumerate(zip(home_ratios, timesteps), start=1):
        name = f"stage{index:02d}_{n}ag_physical_home{ratio:.3f}"
        env = copy.deepcopy(base_env)
        env.update(n_agents=n, n_teams=2, sensor_range=sensor_range,
                   communication_enabled=True, communication_topology="physical",
                   communication_range=sensor_range,
                   shelf_team_mode="balanced_soft_zones", shelf_soft_zone_ratio=ratio,
                   request_queue_size=n, request_queue_size_per_team=n // 2,
                   normalised_coordinates=True)
        env.pop("directional_observation", None)
        env.pop("communication_fov_degrees", None)
        cfg = dict(defaults)
        cfg.update(env_kwargs_json=json.dumps(env), agent_count=n, team_count=2,
                   sensor_range=sensor_range, curriculum_stages="", curriculum_stage=0,
                   init_checkpoint=str(previous), transfer_components="all", seed=seed,
                   communication_enabled=True, graph_mode="policy", comm_graph_mode="physical",
                   peer_transfer_mode="pgct", critic_consensus_tau=0, consensus_interval=0,
                   probe_source="objective", objective_probe_fraction=1.0,
                   policy_probe_team_conditioning="shared", fixed_canonical_probes=True,
                   policy_probe_batch_size=48, probe_interval=5, min_probe_count=8,
                   pgct_min_policy_information=0.0, pgct_gate_style="threshold",
                   pgct_probe_sequence_length=1, pgct_distance_threshold=0.12,
                   pgct_distance_temperature=0.25, pgct_distance_ema_beta=0.35,
                   pgct_peer_loss_coef=0.01, pgct_warmup_updates=100,
                   total_timesteps=budget, learning_rate=learning_rate, entropy_coef=entropy_coef,
                   rollout_steps=1024, sequence_length=64,
                   eval_interval=50, eval_episodes=20, eval_horizon=500, save_interval=100,
                   save_dir=str(output_dir), exp_name=name, render_eval=False,
                   save_eval_video=save_video, track=track, wandb_name=name,
                   wandb_group=output_dir.name, wandb_log_eval_video=track and save_video,
                   wandb_video_interval=5)
        if smoke:
            cfg.update(total_timesteps=64, rollout_steps=32, sequence_length=8,
                       ppo_epochs=1, minibatch_chunks=4, device="cpu",
                       probe_interval=1, min_probe_count=1, pgct_warmup_updates=0,
                       policy_probe_batch_size=8, eval_interval=0,
                       eval_episodes=1, eval_horizon=16, save_interval=0,
                       save_eval_video=False, track=False, wandb_log_eval_video=False)
        stage_cfg = T.TrainConfig(**cfg)
        previous = output_dir / name / "final.pt"
        stages.append(dict(config=asdict(stage_cfg), home_ratio=ratio,
                           mixed_fraction=1-ratio, output_checkpoint=str(previous),
                           effective_timesteps=math.ceil(stage_cfg.total_timesteps / stage_cfg.rollout_steps)
                           * stage_cfg.rollout_steps, status="planned"))
    return stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--agent-count", type=int)
    parser.add_argument("--home-ratios", type=float, nargs="+", default=[1.0, 0.875, 0.75, 0.5])
    parser.add_argument("--stage-timesteps", type=int, nargs="+", default=[1000000, 2000000, 3000000, 4000000])
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve() if args.output_dir else ROOT / "runs" / (
        "pgct09700_physical_mixing_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    checkpoint = T._load_checkpoint(args.init_checkpoint, torch.device("cpu"))
    try:
        stages = build_stages(checkpoint, args.init_checkpoint, output,
                              home_ratios=args.home_ratios, timesteps=args.stage_timesteps,
                              agent_count=args.agent_count, seed=args.seed,
                              learning_rate=args.learning_rate, entropy_coef=args.entropy_coef,
                              track=args.track, save_video=not args.no_video, smoke=args.smoke_test)
    except ValueError as exc:
        parser.error(str(exc))
    del checkpoint
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "curriculum.json"
    manifest = dict(initial_checkpoint=str(Path(args.init_checkpoint).resolve()),
                    note="Actor and critic are transferred; optimizer and graph restart at every stage.",
                    smoke_test=args.smoke_test, stages=stages)

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    save_manifest()
    print(f"Curriculum plan: {manifest_path}", flush=True)
    for stage in stages:
        print(f"{stage['config']['exp_name']}: {stage['effective_timesteps']} steps; "
              f"mixed={stage['mixed_fraction']:.1%}", flush=True)
    if args.dry_run:
        return
    if args.smoke_test:
        torch.set_num_threads(1)
    for stage in stages:
        cfg = T.TrainConfig(**stage["config"])
        if not Path(cfg.init_checkpoint).is_file():
            raise FileNotFoundError(cfg.init_checkpoint)
        stage["status"] = "running"
        save_manifest()
        try:
            T.train(cfg)
            if not Path(stage["output_checkpoint"]).is_file():
                raise RuntimeError("Training did not produce final.pt")
        except BaseException as exc:
            stage["status"] = "failed"
            stage["error"] = str(exc)
            save_manifest()
            raise
        stage["status"] = "complete"
        save_manifest()
    print(f"Curriculum complete: {stages[-1]['output_checkpoint']}", flush=True)


if __name__ == "__main__":
    main()
