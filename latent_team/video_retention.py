"""Opt-in rolling evaluation videos, with only the retained tail sent to W&B.

This wraps the shared trainer in-process; it does not edit the trainer or PPO.
Only videos successfully written by this context in its own run are pruned.
W&B metrics remain live. Media are uploaded once just before finishing (upload="finish"), or each video
right after it is saved (upload="live", T198), still through the short W&B directory.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import json
import math
from pathlib import Path
import re
import uuid
import warnings


class RetainedVideos:
    def __init__(self, cfg, run_dir, keep_last, wandb_root, upload="finish"):
        self.cfg = cfg
        self.run_dir = Path(run_dir).resolve()
        self.video_dir = self.run_dir / "videos"
        self.keep_last = keep_last
        self.wandb_root = Path(wandb_root).resolve()
        self.saved = deque()
        self.run = None
        self.last_step = 0
        self.finished = False
        self.upload_status = "pending"
        self.error = None
        self.active = False
        self.upload = upload
        self.live_logged = 0

    def record(self):
        data = dict(keep_last=self.keep_last,
                    upload_policy="live" if self.upload == "live" else "tail_at_finish",
                    live_logged=self.live_logged,
                    wandb_local_dir=str(self.wandb_root) if self.run is not None else None,
                    upload_status=self.upload_status, error=self.error,
                    videos=[dict(file=p.name, env_steps=step) for p, step in self.saved])
        (self.run_dir / "video_retention.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8")

    def save(self, writer, video, path, fps):
        path = Path(path).resolve()
        if path.parent != self.video_dir or not re.fullmatch(r"eval_\d+\.mp4|final\.mp4", path.name):
            raise ValueError(f"Video retention refuses a path outside this run's eval videos: {path}")
        if any(p == path for p, _ in self.saved) or path.exists():
            raise FileExistsError(f"Refusing to overwrite an existing evaluation video: {path}")
        writer(video, path, fps)
        if not path.is_file():
            raise RuntimeError(f"Evaluation video writer produced no file: {path}")
        step = (math.ceil(self.cfg.total_timesteps / self.cfg.rollout_steps) * self.cfg.rollout_steps
                if path.name == "final.mp4" else int(path.stem.split("_")[1]) * self.cfg.rollout_steps)
        self.saved.append((path, step))
        self.active = True
        while len(self.saved) > self.keep_last:
            old, _ = self.saved[0]
            # Recheck containment before deleting; never glob or prune pre-existing files.
            if old.resolve().parent != self.video_dir:
                raise ValueError(f"Refusing to prune a video outside this run: {old}")
            old.unlink()
            self.saved.popleft()
        if self.upload == "live" and self.run is not None:
            try:
                import wandb

                # Logged without an explicit step: W&B appends it after the last numeric row.
                self.run.log({"eval/video": wandb.Video(str(path), format="mp4",
                                                        caption=f"{path.name}; env_steps={step}")})
                self.live_logged += 1
            except Exception as exc:  # a failed upload must not stop training
                self.error = repr(exc)
                warnings.warn(f"Live W&B video logging failed; the local video is kept: {exc}", stacklevel=2)
        self.record()

    def finish(self, *args, **kwargs):
        if self.finished:
            return None
        self.finished = True
        if not self.active:
            return None
        try:
            if self.upload == "live":
                self.upload_status = "live" if self.live_logged else ("local_only" if self.saved else "no_videos")
            elif self.run is not None and self.saved:
                import wandb

                # File input reuses the encoded local mp4; no second encoding or frame buffer.
                payload = {f"eval/tail_{i:02d}": wandb.Video(str(path), format="mp4",
                           caption=f"{path.name}; env_steps={step}")
                           for i, (path, step) in enumerate(self.saved, 1)}
                payload["media/last_env_steps"] = self.last_step
                # Let W&B choose a fresh history row: the last numeric step is already committed.
                self.run.log(payload)
                self.upload_status = "logged"
            else:
                self.upload_status = "local_only" if self.saved else "no_videos"
        except Exception as exc:
            self.upload_status = "failed"
            self.error = repr(exc)
            warnings.warn(f"W&B tail video logging failed; local videos are preserved: {exc}", stacklevel=2)
        try:
            self.record()
        except Exception as exc:
            warnings.warn(f"Could not write video retention status: {exc}", stacklevel=2)
        if self.run is not None:
            return self.run.finish(*args, **kwargs)


class _RunProxy:
    def __init__(self, state):
        self._state = state

    def __getattr__(self, name):
        return getattr(self._state.run, name)

    def log(self, payload, *args, **kwargs):
        step = kwargs.get("step")
        if step is not None:
            self._state.last_step = max(self._state.last_step, step)
        return self._state.run.log(payload, *args, **kwargs)

    def finish(self, *args, **kwargs):
        return self._state.finish(*args, **kwargs)


@contextmanager
def install(trainer, cfg, run_dir, keep_last=0, *, wandb_root=None, upload="finish"):
    """keep_last=0 preserves legacy behavior; positive limits affect only new run files.

    A normal Python exception / Ctrl+C flushes the current tail. Force-killing the
    process cannot upload it, but already saved local videos remain bounded.
    """
    if keep_last < 0:
        raise ValueError("keep_last must be nonnegative")
    if upload not in ("finish", "live"):
        raise ValueError("upload must be 'finish' or 'live'")
    if keep_last == 0:
        yield None
        return
    if not cfg.save_eval_video:
        raise ValueError("Video retention needs --save-video")
    if wandb_root is None:
        wandb_root = Path(__file__).resolve().parents[1] / "wandb_data" / uuid.uuid4().hex[:12]
    state = RetainedVideos(cfg, run_dir, keep_last, wandb_root, upload)
    names = ("init_wandb", "save_local_eval_video", "make_wandb_video", "log_to_wandb")
    originals = {name: getattr(trainer, name) for name in names}

    def init(config, save_dir):
        state.active = True
        if config.track:
            state.wandb_root.mkdir(parents=True, exist_ok=False)
            state.run = originals["init_wandb"](config, state.wandb_root)
        if state.run is None:
            return None
        state.run.config.update(dict(video_keep_last=keep_last,
                                     video_upload_policy="live" if upload == "live" else "tail_at_finish",
                                     wandb_local_dir=str(state.wandb_root)), allow_val_change=True)
        state.record()
        return _RunProxy(state)

    def save(video, path, fps):
        return state.save(originals["save_local_eval_video"], video, path, fps)

    def log(run, payload, step):
        # The trainer attaches these when video capture is on. Defer media only;
        # every numeric metric still passes to experiment_logging and W&B live.
        payload = {key: value for key, value in payload.items() if key not in ("eval/video", "eval/final_video")}
        state.last_step = max(state.last_step, step)
        return originals["log_to_wandb"](run, payload, step=step)

    trainer.init_wandb = init
    trainer.save_local_eval_video = save
    trainer.make_wandb_video = lambda *args, **kwargs: None
    trainer.log_to_wandb = log
    try:
        try:
            yield state
        except BaseException:
            try:
                state.finish(exit_code=1)
            except Exception as exc:
                warnings.warn(f"W&B finish failed during error cleanup: {exc}", stacklevel=2)
            raise
        else:
            state.finish()
    finally:
        for name, original in originals.items():
            setattr(trainer, name, original)
