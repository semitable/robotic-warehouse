"""Play videos of trained Overcooked agents (MP4), optionally uploaded to Weights & Biases.

  python -m pgct_overcooked.video --job Overcooked/runs/<plan>/<method>_s<seed> [--checkpoint final.pt]
         [--seed 5000001] [--stochastic] [--fps 8] [--wandb]

Writes <job>/videos/<checkpoint>_seed<seed>_<det|stoch>.mp4 and never overwrites. With --wandb the video
goes to a separate W&B run "<training run name>_video" in the job's project and group (a running training
job is never resumed or touched). The same recorder is used by train.py --video-interval during training.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from . import REPO_ROOT  # noqa: F401  (pgct_core path)
from .experiment import build
from .render import render_rgb
from pgct_core.trainer import load_agents


def record_episode(adapter, agents, seed, deterministic=True, max_steps=None):
    """One episode with a fixed seed; returns (frames, per-agent returns, per-agent event totals)."""
    recipes = adapter.env.oracle_recipes()  # evaluation-only overlay
    frames = []
    device = next(agents[0].parameters()).device
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        obs = np.asarray(adapter.reset(seed=seed), dtype=np.float32)
        n = adapter.n_agents
        hidden = agents[0].recurrent_hidden_dim
        ah, ch = torch.zeros((n, hidden), device=device), torch.zeros((n, hidden), device=device)
        total, events, done, t = np.zeros(n), {}, False, 0
        while not done and (max_steps is None or t < max_steps):
            own = events.get("deliveries_own", np.zeros(n))
            frames.append(render_rgb(adapter.env, f"t={t}  return/agent={np.round(total, 1).tolist()}  "
                                                  f"own deliveries={own.astype(int).tolist()}", recipes))
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            actions = []
            with torch.no_grad():
                for i, net in enumerate(agents):
                    a, _, _, ah_i, ch_i = net.act(obs_t[i], ah[i], ch[i], deterministic=deterministic)
                    actions.append(int(a.item()))
                    ah[i], ch[i] = ah_i.squeeze(0), ch_i.squeeze(0)
            obs, rewards, terminated, truncated, info = adapter.step(np.asarray(actions))
            obs = np.asarray(obs, dtype=np.float32)
            total += np.asarray(rewards, dtype=np.float64)
            for key, value in (info.get("events") or {}).items():
                events[key] = events.get(key, 0) + np.asarray(value, dtype=np.float64)
            done, t = bool(terminated or truncated), t + 1
        frames.append(render_rgb(adapter.env, f"t={t} (end)  return/agent={np.round(total, 1).tolist()}", recipes))
    return frames, total, {k: v.tolist() for k, v in events.items()}


def write_mp4(frames, path, fps=8):
    import imageio.v2 as imageio
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=16) as writer:
        for frame in frames:
            writer.append_data(frame)
    return path


class EvalVideoRecorder:
    """pgct_core eval_hook: after every `every`-th evaluation (and the final one) record one greedy episode with
    the first evaluation seed, keep the newest `keep` MP4 files plus final.mp4, and log it to W&B as eval/video.

    Recording uses a separate environment and a forked torch RNG, so training is unaffected; a failure is
    written to video_error.json and never stops training."""

    def __init__(self, make_adapter, out_dir, *, seed, every=1, keep=10, fps=8, tracker=None):
        self.make_adapter, self.dir = make_adapter, Path(out_dir) / "videos"
        self.seed, self.every, self.keep, self.fps, self.tracker = seed, max(1, every), max(1, keep), fps, tracker
        self.count = 0

    def __call__(self, agents, row):
        final = bool(row.get("final"))
        self.count += 0 if final else 1
        if not final and self.count % self.every:
            return
        try:
            frames, returns, events = record_episode(self.make_adapter(), agents, self.seed, True)
            path = write_mp4(frames, self.dir / ("final.mp4" if final else f"eval_{row['env_steps']:09d}.mp4"),
                             self.fps)
            periodic = sorted(self.dir.glob("eval_*.mp4"))
            for old in periodic[:-self.keep]:
                old.unlink()
            if self.tracker is not None and hasattr(self.tracker, "log_video"):
                self.tracker.log_video(path, row["env_steps"], dict(
                    video_return_mean=float(np.mean(returns)),
                    video_deliveries_own=float(np.sum(events.get("deliveries_own", 0)))))
        except Exception as exc:  # a video must never cost a training run
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir.parent / "video_error.json").write_text(json.dumps(dict(env_steps=row.get("env_steps"),
                                                                              error=repr(exc)), indent=2),
                                                              encoding="utf-8")


def upload(path, job: Path, env_steps, summary, mode=None):
    """Log a finished video to a separate W&B run next to the job's training run."""
    import wandb
    descriptor = json.loads((job / "wandb.json").read_text(encoding="utf-8")) if (job / "wandb.json").exists() else {}
    plan = job.parent.name
    run = wandb.init(project=descriptor.get("project", "pgct-overcooked"), entity=descriptor.get("entity"),
                     group=descriptor.get("group", plan), name=f"{descriptor.get('name', plan + '_' + job.name)}_video",
                     job_type="video", mode=mode or descriptor.get("mode", "online"), dir=str(job),
                     config=dict(job=str(job), checkpoint_env_steps=env_steps), reinit="finish_previous")
    run.define_metric("env_steps")
    run.define_metric("*", step_metric="env_steps")
    run.log({"env_steps": env_steps, "eval/video": wandb.Video(str(path), format="mp4"),
             **{f"eval/video_{k}": v for k, v in summary.items()}})
    url = run.url
    run.finish()
    return url


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--checkpoint", default="final.pt")
    parser.add_argument("--seed", type=int, default=5_000_001, help="episode seed (default: first eval seed)")
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of the greedy policy")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--wandb", action="store_true", help="also upload to W&B (separate *_video run)")
    parser.add_argument("--wandb-mode", choices=("online", "offline"))
    args = parser.parse_args(argv)
    config = json.loads((args.job / "config.json").read_text(encoding="utf-8"))
    experiment = config["experiment"]
    make_adapter, _, ppo_cfg, *_ = build(experiment, experiment["trainer"]["method"], experiment["trainer"]["seed"])
    adapter = make_adapter()
    agents, payload = load_agents(args.job / args.checkpoint, adapter.obs_dim, adapter.n_actions, ppo_cfg)
    frames, returns, events = record_episode(adapter, agents, args.seed, not args.stochastic, args.max_steps)
    target = args.job / "videos" / f"{Path(args.checkpoint).stem}_seed{args.seed}_{'stoch' if args.stochastic else 'det'}.mp4"
    if target.exists():
        raise SystemExit(f"{target} exists; choose another --seed")
    write_mp4(frames, target, args.fps)
    summary = dict(return_mean=float(np.mean(returns)), deliveries_own=float(np.sum(events.get("deliveries_own", 0))),
                   deliveries_other=float(np.sum(events.get("deliveries_other", 0))))
    print(json.dumps(dict(video=str(target), env_steps=payload.get("env_steps"), frames=len(frames),
                          returns=returns.tolist(), **summary), indent=2))
    if args.wandb:
        print("W&B:", upload(target, args.job, payload.get("env_steps"), summary, args.wandb_mode))


if __name__ == "__main__":
    main()
