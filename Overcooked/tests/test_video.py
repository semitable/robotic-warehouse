"""Play-video rendering and the eval-time recorder (no W&B)."""
import json

import numpy as np
import torch

from pgct_core.trainer import run_training
from pgct_overcooked import train as TR
from pgct_overcooked.adapter import KitchenAdapter
from pgct_overcooked.experiment import apply_overrides, build, load_config
from pgct_overcooked.render import render_rgb
from pgct_overcooked.video import EvalVideoRecorder, record_episode, write_mp4


def test_frames_are_codec_friendly_rgb():
    adapter = KitchenAdapter()
    adapter.reset(seed=0)
    frame = render_rgb(adapter.env, "t=0", adapter.env.oracle_recipes())
    assert frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] == 3
    assert frame.shape[0] % 16 == 0 and frame.shape[1] % 16 == 0


def test_recorded_episode_reproduces_and_writes_mp4(tmp_path):
    data = apply_overrides(load_config("main_4p2t.json"), ["kitchen.horizon=20"])
    make_adapter, cfg, ppo, *_ = build(data, "ippo", 1)
    adapter = make_adapter()
    torch.manual_seed(0)
    from pgct_core.networks import RecurrentActorCritic
    agents = [RecurrentActorCritic(adapter.obs_dim, adapter.n_actions, 16, 8) for _ in range(4)]
    state = torch.get_rng_state()
    a = record_episode(make_adapter(), agents, 7, deterministic=False)
    b = record_episode(make_adapter(), agents, 7, deterministic=False)
    assert torch.equal(state, torch.get_rng_state())  # recording does not consume the training stream
    assert len(a[0]) == 21 and np.array_equal(a[1], b[1]) and all(np.array_equal(x, y) for x, y in zip(a[0], b[0]))
    path = write_mp4(a[0], tmp_path / "v.mp4")
    assert path.stat().st_size > 1000


def test_recorder_hook_keeps_training_identical(tmp_path):
    over = TR.SMOKE + ["trainer.total_timesteps=128", "trainer.eval_interval=1", "trainer.log_interval=0",
                       "kitchen.horizon=20"]
    data = apply_overrides(load_config("main_4p2t.json"), over)
    make_adapter, cfg, ppo, gate, mg, _ = build(data, "pgct_d2", 1)
    run_training(make_adapter, tmp_path / "plain", cfg, ppo, gate, mg)
    recorder = EvalVideoRecorder(make_adapter, tmp_path / "video", seed=cfg.eval_seed, every=1, keep=1)
    run_training(make_adapter, tmp_path / "video", cfg, ppo, gate, mg, eval_hook=recorder)
    a = torch.load(tmp_path / "plain" / "final.pt", weights_only=False)["agents"]
    b = torch.load(tmp_path / "video" / "final.pt", weights_only=False)["agents"]
    assert all(torch.equal(x[k], y[k]) for x, y in zip(a, b) for k in x)
    videos = sorted(p.name for p in (tmp_path / "video" / "videos").iterdir())
    assert videos == ["eval_000000128.mp4", "final.mp4"]  # keep=1 periodic + final
    assert not (tmp_path / "video" / "video_error.json").exists()
