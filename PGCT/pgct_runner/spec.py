"""ExperimentSpec: everything the shared command lines need to know about one experiment package."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple


def load_config(path, config_dir, sections):
    """A JSON experiment file; relative names are looked up in ``config_dir``. Unknown sections are refused."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = Path(config_dir) / path
    data = json.loads(path.read_text(encoding="utf-8"))
    unknown = set(data) - set(sections) - {"description"}
    if unknown:
        raise ValueError(f"unknown config sections {sorted(unknown)}")
    return data


def apply_overrides(data, overrides, sections):
    """'section.key=value' strings (value parsed as JSON when possible)."""
    data = copy.deepcopy(data)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        section, _, field = key.partition(".")
        if section not in sections or not field:
            raise ValueError(f"override must look like section.key=value with a section in {sections}: {item}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        data.setdefault(section, {})[field] = value
    return data


@dataclass(frozen=True)
class ExperimentSpec:
    """One experiment package (environment + adapter + configs).

    build(data, method, seed) -> (make_adapter, trainer_cfg, ppo_cfg, gate_cfg, max_gae_cfg, resolved) turns a
    config dictionary into typed configs for one job; ``resolved`` is the JSON-ready dictionary stored in
    plan.json and config.json. render(adapter, header) -> RGB uint8 frame draws the current state for videos
    (it may show hidden teams to the human viewer; it is never called by training).
    """
    name: str                              # display name ("LBF")
    package: str                           # python package ("pgct_lbf")
    config_dir: Path
    output_root: Path
    sections: Tuple[str, ...]
    build: Callable
    default_config: str
    smoke: Tuple[str, ...] = ()            # overrides for --smoke (tiny budget, permissive wiring-only gate)
    wandb_project: str = "pgct"
    plan_prefix: str = "run"
    render: Optional[Callable] = None
    headline_events: Tuple[str, ...] = ()  # per-agent event counters for video headers, W&B extras, summaries

    def load(self, path):
        return load_config(path, self.config_dir, self.sections)

    def override(self, data, overrides):
        return apply_overrides(data, overrides, self.sections)
