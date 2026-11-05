"""Evaluation play videos (MP4) for any ExperimentSpec with a ``render`` function.

record_episode runs one fixed-seed episode with a separate environment and a forked torch RNG; training is
never touched. EvalVideoRecorder is the pgct_core eval_hook used by ``train --video-every``; video_main is
the stand-alone command line for saved checkpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pgct_core.trainer import load_agents


def _header(t, total, events, keys, end=False):
    parts = [f"t={t}{' (end)' if end else ''}", f"return/agent={np.round(total, 2).tolist()}"]
    for key in keys:
        if key in events:
            parts.append(f"{key}={np.asarray(events[key]).astype(int).tolist()}")
    return "  ".join(parts)


def record_episode(spec, adapter, agents, seed, deterministic=True, max_steps=None):
    """One episode with a fixed seed; returns (frames, per-agent returns, per-agent event totals)."""
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
            frames.append(spec.render(adapter, _header(t, total, events, spec.headline_events)))
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
        frames.append(spec.render(adapter, _header(t, total, events, spec.headline_events, end=True)))
    return frames, total, {k: np.asarray(v).tolist() for k, v in events.items()}


def write_mp4(frames, path, fps=8):
    import imageio.v2 as imageio
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=16) as writer:
        for frame in frames:
            writer.append_data(frame)
    return path


def _summary(spec, returns, events):
    return dict(video_return_mean=float(np.mean(returns)),
                **{f"video_{k}": float(np.sum(events.get(k, 0))) for k in spec.headline_events})


class EvalVideoRecorder:
    """pgct_core eval_hook: after every `every`-th evaluation (and the final one) record one greedy episode with
    the first evaluation seed, keep the newest `keep` MP4 files plus final.mp4, and log it to W&B as eval/video.
    A failure is written to video_error.json and never stops training."""

    def __init__(self, spec, make_adapter, out_dir, *, seed, every=1, keep=10, fps=8, tracker=None):
        self.spec, self.make_adapter, self.dir = spec, make_adapter, Path(out_dir) / "videos"
        self.seed, self.every, self.keep, self.fps, self.tracker = seed, max(1, every), max(1, keep), fps, tracker
        self.count = 0

    def __call__(self, agents, row):
        final = bool(row.get("final"))
        self.count += 0 if final else 1
        if not final and self.count % self.every:
            return
        try:
            frames, returns, events = record_episode(self.spec, self.make_adapter(), agents, self.seed, True)
            path = write_mp4(frames, self.dir / ("final.mp4" if final else f"eval_{row['env_steps']:09d}.mp4"),
                             self.fps)
            for old in sorted(self.dir.glob("eval_*.mp4"))[:-self.keep]:
                old.unlink()
            if self.tracker is not None and hasattr(self.tracker, "log_video"):
                self.tracker.log_video(path, row["env_steps"], _summary(self.spec, returns, events))
        except Exception as exc:  # a video must never cost a training run
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir.parent / "video_error.json").write_text(
                json.dumps(dict(env_steps=row.get("env_steps"), error=repr(exc)), indent=2), encoding="utf-8")


def upload(spec, path, job: Path, env_steps, summary, mode=None):
    """Log a finished video to a separate W&B run "<training run>_video" (a training run is never resumed)."""
    import wandb
    descriptor = json.loads((job / "wandb.json").read_text(encoding="utf-8")) if (job / "wandb.json").exists() else {}
    plan = job.parent.name
    run = wandb.init(project=descriptor.get("project", spec.wandb_project), entity=descriptor.get("entity"),
                     group=descriptor.get("group", plan), name=f"{descriptor.get('name', plan + '_' + job.name)}_video",
                     job_type="video", mode=mode or descriptor.get("mode", "online"), dir=str(job),
                     config=dict(job=str(job), checkpoint_env_steps=env_steps), reinit="finish_previous")
    run.define_metric("env_steps")
    run.define_metric("*", step_metric="env_steps")
    run.log({"env_steps": env_steps, "eval/video": wandb.Video(str(path), format="mp4"),
             **{f"eval/{k}": v for k, v in summary.items()}})
    url = run.url
    run.finish()
    return url


def video_main(spec, argv=None):
    parser = argparse.ArgumentParser(description=f"{spec.name}: play video of a saved checkpoint "
                                                 "(writes <job>/videos/<checkpoint>_seed<seed>_<det|stoch>.mp4, "
                                                 "never overwrites)")
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--checkpoint", default="final.pt")
    parser.add_argument("--seed", type=int, default=5_000_001, help="episode seed (default: first eval seed)")
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of the greedy policy")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--wandb", action="store_true", help="also upload to W&B (separate *_video run)")
    parser.add_argument("--wandb-mode", choices=("online", "offline"))
    args = parser.parse_args(argv)
    experiment = json.loads((args.job / "config.json").read_text(encoding="utf-8"))["experiment"]
    make_adapter, _, ppo_cfg, *_ = spec.build(experiment, experiment["trainer"]["method"],
                                              experiment["trainer"]["seed"])
    adapter = make_adapter()
    agents, payload = load_agents(args.job / args.checkpoint, adapter.obs_dim, adapter.n_actions, ppo_cfg)
    target = (args.job / "videos" /
              f"{Path(args.checkpoint).stem}_seed{args.seed}_{'stoch' if args.stochastic else 'det'}.mp4")
    if target.exists():
        raise SystemExit(f"{target} exists; choose another --seed")
    frames, returns, events = record_episode(spec, adapter, agents, args.seed, not args.stochastic, args.max_steps)
    write_mp4(frames, target, args.fps)
    summary = _summary(spec, returns, events)
    print(json.dumps(dict(video=str(target), env_steps=payload.get("env_steps"), frames=len(frames),
                          returns=returns.tolist(), **summary), indent=2))
    if args.wandb:
        print("W&B:", upload(spec, target, args.job, payload.get("env_steps"), summary, args.wandb_mode))
    return 0
