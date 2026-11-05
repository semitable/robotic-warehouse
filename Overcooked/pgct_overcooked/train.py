"""Train IPPO / Oracle / All / PGCT-D2 / PGCT-ID-only / random-gate on the multi-team kitchen.

Every method of a plan uses the same config file (environment, PPO, gate, Max-GAE, budget, evaluation) and
the same seeds; only the critic-access rule differs. Methods: ippo, oracle, all, pgct_d2, pgct_id_only,
random_gate (pgct_core.trainer.METHODS).

Examples (PowerShell, from the repository root):
  python -m pgct_overcooked.train --config main_4p2t.json --methods ippo pgct_d2 --seeds 1 --name smoke --smoke
  python -m pgct_overcooked.train --config main_4p2t.json --methods pgct_d2 --seeds 1 2 3 --name main_1105
Run with ``Overcooked`` on PYTHONPATH (scripts/*.ps1 do this) or from the Overcooked folder.
Outputs: <output-root>/<name>/plan.json and one folder per job (<method>_s<seed>) with config.json,
metrics.jsonl, gate.jsonl, eval.jsonl, checkpoints and status.json. Existing finished jobs are refused.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from . import REPO_ROOT
from .experiment import apply_overrides, build, load_config
from .tracking import WandbTracker
from .video import EvalVideoRecorder
from pgct_core.trainer import METHODS, run_training

SMOKE = ["ppo.rollout_steps=64", "ppo.sequence_length=16", "ppo.minibatch_chunks=2", "ppo.ppo_epochs=1",
         "trainer.total_timesteps=256", "trainer.warmup_updates=0", "trainer.eval_interval=2",
         "trainer.eval_episodes=1", "trainer.save_interval=0", "trainer.log_interval=1", "kitchen.horizon=50",
         "gate.interval=1", "gate.batch_size=16", "gate.train_steps=2", "gate.min_events=1",
         "gate.min_event_response=-1.0", "gate.event_reward=0.05", "gate.min_rounds=1", "gate.min_probes=1",
         "gate.distance_threshold=1.0"]


def read_plan(path, attempts=100):
    """Parallel job processes of one plan read it while another may be replacing it: retry briefly."""
    for attempt in range(attempts):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, PermissionError):
            if attempt == attempts - 1:
                raise
            time.sleep(.1)


def write_plan(path, payload, attempts=100):
    """Atomic replace (Windows refuses it while a reader holds the file open, so retry)."""
    temporary = Path(path).with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for attempt in range(attempts):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(.1)


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="main_4p2t.json", help="JSON in Overcooked/configs or a path")
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=["ippo", "pgct_d2"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--name", default=None, help="plan folder name (default: timestamp)")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "Overcooked" / "runs")
    parser.add_argument("--set", nargs="*", default=[], metavar="SECTION.KEY=VALUE", help="config overrides")
    parser.add_argument("--total-timesteps", type=int, help="shortcut for --set trainer.total_timesteps=N")
    parser.add_argument("--smoke", action="store_true", help="tiny budget and permissive wiring-only gate values")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--track", action="store_true", help="log each selected job to Weights & Biases")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "pgct-overcooked"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-mode", choices=("online", "offline"),
                        default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    parser.add_argument("--video-every", type=int, default=0,
                        help="record a greedy play video at every N-th evaluation and at the end (0: off); "
                             "uploaded as eval/video when --track is on")
    parser.add_argument("--video-keep", type=int, default=10, help="newest periodic videos kept per job (+ final.mp4)")
    args = parser.parse_args(argv)
    if len(set(args.methods)) != len(args.methods) or len(set(args.seeds)) != len(args.seeds):
        parser.error("methods and seeds must be unique")
    data = load_config(args.config)
    overrides = (SMOKE if args.smoke else []) + list(args.set)
    if args.total_timesteps:
        overrides.append(f"trainer.total_timesteps={args.total_timesteps}")
    data = apply_overrides(data, overrides)
    name = args.name or datetime.now().strftime("oc_%m%d_%H%M%S")
    root = args.output_root / name
    jobs = []
    for seed in args.seeds:
        for method in args.methods:
            *_, resolved = build(data, method, seed)
            jobs.append(dict(name=f"{method}_s{seed}", method=method, seed=seed, config=resolved))
    plan_path = root / "plan.json"
    planned, write = jobs, True
    if plan_path.exists():  # same plan launched in parts (scripts/run_overcooked.ps1): keep every job listed
        previous = read_plan(plan_path)
        if previous["config_source"] != data:
            parser.error(f"{root} holds a plan with a different configuration; choose a new --name")
        listed = {j["name"]: j for j in previous["jobs"]}
        write = any(listed.get(j["name"]) != j for j in jobs)  # queue workers find their job already planned
        names = {j["name"] for j in jobs}
        planned = [j for j in previous["jobs"] if j["name"] not in names] + jobs
    root.mkdir(parents=True, exist_ok=True)
    if write:
        write_plan(plan_path, dict(created=datetime.now().astimezone().isoformat(timespec="seconds"),
                                   git_commit=git_commit(), config_file=str(args.config), overrides=overrides,
                                   smoke=args.smoke, config_source=data,
                                   tracking=dict(enabled=args.track, project=args.wandb_project,
                                                 entity=args.wandb_entity, mode=args.wandb_mode,
                                                 tags=args.wandb_tags), jobs=planned))
    print(f"plan: {plan_path} ({len(planned)} jobs listed, {len(jobs)} selected)", flush=True)
    if args.plan_only:
        return 0
    failed = []
    for job in jobs:
        out = root / job["name"]
        if (out / "final.pt").exists():
            print(f"skip {job['name']}: already finished", flush=True)
            continue
        make_adapter, trainer_cfg, ppo_cfg, gate_cfg, max_cfg, resolved = build(data, job["method"], job["seed"])
        print(f"== {job['name']}", flush=True)
        tracker = None
        try:
            if args.track:
                tracker = WandbTracker(
                    out, project=args.wandb_project, entity=args.wandb_entity, group=name,
                    run_name=f"{name}_{job['name']}", method=job["method"], seed=job["seed"],
                    mode=args.wandb_mode,
                    config=dict(experiment=resolved, environment=make_adapter().describe(),
                                plan=name, job=job["name"], git_commit=git_commit()),
                    tags=args.wandb_tags,
                )
            recorder = (EvalVideoRecorder(make_adapter, out, seed=trainer_cfg.eval_seed, every=args.video_every,
                                          keep=args.video_keep, tracker=tracker) if args.video_every > 0 else None)
            result = run_training(make_adapter, out, trainer_cfg, ppo_cfg, gate_cfg, max_cfg,
                                  extra_config=dict(experiment=resolved, environment=make_adapter().describe()),
                                  event_callback=tracker, eval_hook=recorder)
            if tracker is not None:
                tracker.finish("complete")
            print(f"done {job['name']}: final eval return {result['final_eval']['eval_return_mean']:.3f}", flush=True)
        except Exception as exc:
            if tracker is not None:
                tracker.finish("failed", exit_code=1)
            out.mkdir(parents=True, exist_ok=True)
            (out / "status.json").write_text(json.dumps(dict(status="failed", error=repr(exc),
                                                             traceback=traceback.format_exc()), indent=2),
                                             encoding="utf-8")
            print(f"FAILED {job['name']}: {exc!r}", flush=True)
            failed.append(job["name"])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
