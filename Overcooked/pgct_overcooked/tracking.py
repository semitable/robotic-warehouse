"""Optional Weights & Biases tracking for Overcooked jobs.

The shared trainer stays independent of W&B.  It emits scalar metric/evaluation rows through a callback;
this adapter flattens per-agent vectors and sends them to one W&B run per (method, seed) job.
"""
from __future__ import annotations

import json
import numbers
from pathlib import Path

import numpy as np


def scalar_payload(row, prefix=""):
    """Keep scalars and flatten one-dimensional numeric vectors for dashboard logging."""
    payload = {}
    for key, value in row.items():
        target = f"{prefix}{key}" if key not in {"env_steps", "update"} else key
        if isinstance(value, (numbers.Number, np.number, bool)):
            payload[target] = value.item() if isinstance(value, np.generic) else value
        elif isinstance(value, (list, tuple, np.ndarray)):
            vector = np.asarray(value)
            if vector.ndim == 1 and all(isinstance(v, (numbers.Number, np.number, bool)) for v in vector):
                for index, item in enumerate(vector):
                    payload[f"{target}/agent_{index}"] = item.item() if isinstance(item, np.generic) else item
    return payload


class WandbTracker:
    """Own one W&B run and a small local descriptor that makes the remote run discoverable."""

    def __init__(self, out_dir, *, project, entity, group, run_name, method, seed, mode, config, tags=()):
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("W&B tracking requires `wandb` (install with `pip install -e .[overcooked]`)") from exc

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.failed = False
        self.run = wandb.init(
            project=project,
            entity=entity or None,
            group=group,
            name=run_name,
            job_type=method,
            tags=list(tags) + [method, f"seed{seed}"],
            config=config,
            mode=mode,
            dir=str(self.out_dir),
            reinit="finish_previous",
        )
        self.run.define_metric("env_steps")
        self.run.define_metric("*", step_metric="env_steps")
        descriptor = {
            "id": self.run.id,
            "url": self.run.url,
            "project": project,
            "entity": entity or None,
            "group": group,
            "name": run_name,
            "method": method,
            "seed": seed,
            "mode": mode,
        }
        (self.out_dir / "wandb.json").write_text(json.dumps(descriptor, indent=2), encoding="utf-8")

    def __call__(self, event, row):
        if self.failed:
            return
        prefix = "eval/" if event == "eval" else ""
        payload = scalar_payload(row, prefix)
        if payload:
            try:
                self.run.log(payload)
            except Exception as exc:  # telemetry must not destroy an expensive training run
                self._record_error(event, exc)

    def log_video(self, path, env_steps, extra=None):
        """Upload one evaluation video (MP4 file) as eval/video at this env step."""
        if self.failed:
            return
        try:
            import wandb
            self.run.log({"env_steps": env_steps, "eval/video": wandb.Video(str(path), format="mp4"),
                          **{f"eval/{k}": v for k, v in (extra or {}).items()}})
        except Exception as exc:  # a failed upload keeps the metric stream alive (no self.failed)
            (self.out_dir / "wandb_video_error.json").write_text(
                json.dumps({"env_steps": env_steps, "error": repr(exc)}, indent=2), encoding="utf-8")
            print(f"WARNING: W&B video upload failed; training continues: {exc!r}", flush=True)

    def finish(self, status, exit_code=0):
        try:
            self.run.summary["status"] = status
            self.run.finish(exit_code=exit_code)
        except Exception as exc:  # checkpoints and local JSONL remain authoritative
            self._record_error("finish", exc)

    def _record_error(self, event, exc):
        self.failed = True
        payload = {"event": event, "error": repr(exc)}
        (self.out_dir / "wandb_error.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"WARNING: W&B {event} failed; training continues with local logs: {exc!r}", flush=True)
