"""Experiment configuration for multi-team LBF (one JSON file per comparison) and the pgct_runner spec.

Sections: foraging (environment), adapter (probe view / communication), ppo, gate (pgct_core reward gate),
max_gae, trainer. Only ``trainer.method`` and ``trainer.seed`` differ between the jobs of a plan.
``foraging.team_seed: null`` draws the hidden team assignment from the run seed.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from . import REPO_ROOT
from pgct_core.max_gae import MaxGAEConfig
from pgct_core.ppo import PPOConfig
from pgct_core.reward_gate import RewardGateConfig
from pgct_core.trainer import TrainerConfig
from pgct_runner.spec import ExperimentSpec

from .adapter import AdapterConfig, ForagingAdapter
from .env.foraging import ForagingConfig
from .render import render_adapter

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PACKAGE_ROOT / "configs"
SECTIONS = ("foraging", "adapter", "ppo", "gate", "max_gae", "trainer")

# Tiny budget and permissive, wiring-only gate values (any positive reward is an event, everything matures).
SMOKE = ("ppo.rollout_steps=64", "ppo.sequence_length=16", "ppo.minibatch_chunks=2", "ppo.ppo_epochs=1",
         "trainer.total_timesteps=256", "trainer.warmup_updates=0", "trainer.eval_interval=2",
         "trainer.eval_episodes=1", "trainer.save_interval=0", "trainer.log_interval=1", "foraging.horizon=25",
         "gate.interval=1", "gate.batch_size=16", "gate.train_steps=2", "gate.min_events=1",
         "gate.min_event_response_ratio=-1.0", "gate.min_event_response=-1.0", "gate.min_rounds=1",
         "gate.min_probes=1", "gate.distance_threshold=1.0")


def build(data, method, seed):
    """Typed configs for one job plus an adapter factory."""
    foraging = dict(data.get("foraging", {}))
    if foraging.get("team_seed") is None:
        foraging["team_seed"] = int(seed)
    if "team_sizes" in foraging:
        foraging["team_sizes"] = tuple(foraging["team_sizes"])
    foraging_cfg = ForagingConfig(**foraging)
    adapter_cfg = AdapterConfig(**data.get("adapter", {}))
    ppo_cfg = PPOConfig(**data.get("ppo", {}))
    gate_cfg = RewardGateConfig(**{"radius": max(1, adapter_cfg.probe_radius), **data.get("gate", {})})
    max_cfg = MaxGAEConfig(**{"donors": "gate", "scale": "shrink", **data.get("max_gae", {})})
    trainer = dict(data.get("trainer", {}))
    if trainer.get("init_checkpoint") and not Path(trainer["init_checkpoint"]).is_absolute():
        trainer["init_checkpoint"] = str(REPO_ROOT / trainer["init_checkpoint"])  # repo-relative in configs
    trainer_cfg = TrainerConfig(**{**trainer, "method": method, "seed": int(seed)})

    def make_adapter():
        return ForagingAdapter(foraging_cfg, adapter_cfg)

    resolved = dict(foraging=asdict(foraging_cfg), adapter=asdict(adapter_cfg), ppo=asdict(ppo_cfg),
                    gate=asdict(gate_cfg), max_gae=asdict(max_cfg), trainer=asdict(trainer_cfg))
    return make_adapter, trainer_cfg, ppo_cfg, gate_cfg, max_cfg, resolved


SPEC = ExperimentSpec(name="LBF", package="pgct_lbf", config_dir=CONFIG_DIR, output_root=REPO_ROOT / "LBF" / "runs",
                      sections=SECTIONS, build=build, default_config="main_4p2t.json", smoke=SMOKE,
                      wandb_project="pgct-lbf", plan_prefix="lbf", render=render_adapter,
                      headline_events=("collected_own", "collected_other"))


def load_config(path):
    return SPEC.load(path)


def apply_overrides(data, overrides):
    return SPEC.override(data, overrides)
