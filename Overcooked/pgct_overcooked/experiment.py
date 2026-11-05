"""Experiment configuration: one JSON file holds every setting shared by all methods of a comparison.

Sections: kitchen (environment), adapter (observation/probe/communication), ppo, gate (pgct_core reward
gate), max_gae, trainer. Only ``trainer.method`` and ``trainer.seed`` differ between the jobs of a plan.
``kitchen.team_seed: null`` draws the hidden team assignment from the run seed.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

from pgct_core.max_gae import MaxGAEConfig
from pgct_core.ppo import PPOConfig
from pgct_core.reward_gate import RewardGateConfig
from pgct_core.trainer import TrainerConfig

from .adapter import AdapterConfig, KitchenAdapter
from .env.kitchen import KitchenConfig

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PACKAGE_ROOT / "configs"
SECTIONS = ("kitchen", "adapter", "ppo", "gate", "max_gae", "trainer")


def load_config(path):
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / path
    data = json.loads(path.read_text(encoding="utf-8"))
    unknown = set(data) - set(SECTIONS) - {"description"}
    if unknown:
        raise ValueError(f"unknown config sections {sorted(unknown)}")
    return data


def apply_overrides(data, overrides):
    """'section.key=value' strings (value parsed as JSON when possible)."""
    data = copy.deepcopy(data)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        section, _, field = key.partition(".")
        if section not in SECTIONS or not field:
            raise ValueError(f"override must look like section.key=value: {item}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        data.setdefault(section, {})[field] = value
    return data


def build(data, method, seed):
    """Typed configs for one job plus an adapter factory."""
    kitchen = dict(data.get("kitchen", {}))
    if kitchen.get("team_seed") is None:
        kitchen["team_seed"] = int(seed)
    for key in ("team_recipes", "team_sizes"):
        if key in kitchen:
            kitchen[key] = tuple(kitchen[key])
    kitchen_cfg = KitchenConfig(**kitchen)
    adapter_cfg = AdapterConfig(**data.get("adapter", {}))
    ppo_cfg = PPOConfig(**data.get("ppo", {}))
    gate_cfg = RewardGateConfig(**{"radius": adapter_cfg.probe_radius or 1, **data.get("gate", {})})
    max_cfg = MaxGAEConfig(**{"donors": "gate", "scale": "shrink", **data.get("max_gae", {})})
    trainer_cfg = TrainerConfig(**{**data.get("trainer", {}), "method": method, "seed": int(seed)})

    def make_adapter():
        return KitchenAdapter(kitchen_cfg, adapter_cfg)

    resolved = dict(kitchen=asdict(kitchen_cfg), adapter=asdict(adapter_cfg), ppo=asdict(ppo_cfg),
                    gate=asdict(gate_cfg), max_gae=asdict(max_cfg), trainer=asdict(trainer_cfg))
    return make_adapter, trainer_cfg, ppo_cfg, gate_cfg, max_cfg, resolved
