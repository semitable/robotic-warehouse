"""Video lifecycle checks independent of PPO / environment randomness."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from latent_team.video_retention import install


class Config(dict):
    def update(self, values, **kwargs):
        super().update(values)


class Run:
    def __init__(self):
        self.config = Config()
        self.logs = []
        self.finished = []
        self.url = "offline-test"

    def log(self, payload, *args, **kwargs):
        self.logs.append((payload, kwargs))

    def finish(self, *args, **kwargs):
        self.finished.append(kwargs)


def setup(tmp_path, monkeypatch, track=True):
    run_dir = tmp_path / "new_run"
    run_dir.mkdir()
    cfg = SimpleNamespace(total_timesteps=130, rollout_steps=10, save_eval_video=True, track=track)
    run = Run()
    calls = []

    def init(cfg, path):
        calls.append(Path(path))
        return run if cfg.track else None

    def write(video, path, fps):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(video)

    def encode(*args, **kwargs):
        raise AssertionError("Per-evaluation W&B re-encoding must be suppressed")

    def log(run, payload, step):
        if run is not None:
            run.log(payload, step=step)

    videos = []

    def video(path, **kwargs):
        assert Path(path).is_file()
        videos.append(path)
        return {"path": path, **kwargs}

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Video=video))
    trainer = SimpleNamespace(init_wandb=init, save_local_eval_video=write,
                              make_wandb_video=encode, log_to_wandb=log)
    return trainer, cfg, run_dir, run, calls, videos


def test_ten_latest_including_final_uploaded_once_and_metrics_stay_live(tmp_path, monkeypatch):
    t, cfg, rd, run, calls, videos = setup(tmp_path, monkeypatch)
    originals = vars(t).copy()
    old = tmp_path / "old_run/videos/old.mp4"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"existing")
    unrelated = rd / "videos/manual.mp4"
    unrelated.parent.mkdir()
    unrelated.write_bytes(b"manual")
    short = tmp_path / "wb"
    with install(t, cfg, rd, 10, wandb_root=short):
        proxy = t.init_wandb(cfg, rd)
        assert proxy.url == run.url
        for i in range(1, 14):
            t.save_local_eval_video(bytes([i]), rd / f"videos/eval_{i:05d}.mp4", 10)
            t.log_to_wandb(proxy, {"eval/value": i, "eval/video": t.make_wandb_video(None)}, step=i * 10)
            assert not videos
            assert len(list((rd / "videos").glob("eval_*.mp4"))) <= 10
        t.save_local_eval_video(b"final", rd / "videos/final.mp4", 10)
        proxy.finish()
    assert vars(t) == originals
    assert calls == [short]
    assert len(run.logs) == 14  # 13 unmodified numeric rows + one tail upload
    assert [row[0] for row in run.logs[:-1]] == [{"eval/value": i} for i in range(1, 14)]
    tail, kwargs = run.logs[-1]
    assert kwargs == {}  # avoid logging videos at an already committed numeric step
    assert len(videos) == 10 and len(run.finished) == 1
    assert [Path(p).name for p in videos] == [f"eval_{i:05d}.mp4" for i in range(5, 14)] + ["final.mp4"]
    assert tail["media/last_env_steps"] == 130
    assert old.read_bytes() == b"existing" and unrelated.read_bytes() == b"manual"
    status = json.loads((rd / "video_retention.json").read_text())
    assert status["upload_status"] == "logged" and len(status["videos"]) == 10


def test_exception_flushes_tail_and_restores_hooks(tmp_path, monkeypatch):
    t, cfg, rd, run, _, videos = setup(tmp_path, monkeypatch)
    originals = vars(t).copy()
    with pytest.raises(KeyboardInterrupt):
        with install(t, cfg, rd, 10, wandb_root=tmp_path / "wb"):
            proxy = t.init_wandb(cfg, rd)
            t.save_local_eval_video(b"one", rd / "videos/eval_00001.mp4", 10)
            t.log_to_wandb(proxy, {"train/loss": 1.0}, step=10)
            raise KeyboardInterrupt
    assert vars(t) == originals and len(videos) == 1
    assert run.finished == [{"exit_code": 1}]


def test_upload_failure_preserves_local_videos_and_finishes(tmp_path, monkeypatch):
    t, cfg, rd, run, _, _ = setup(tmp_path, monkeypatch)

    def fail(*args, **kwargs):
        raise FileNotFoundError("media copy failed")

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Video=fail))
    with pytest.warns(UserWarning, match="local videos are preserved"):
        with install(t, cfg, rd, 10, wandb_root=tmp_path / "wb"):
            t.init_wandb(cfg, rd)
            t.save_local_eval_video(b"one", rd / "videos/eval_00001.mp4", 10)
    assert (rd / "videos/eval_00001.mp4").read_bytes() == b"one"
    assert len(run.finished) == 1
    assert json.loads((rd / "video_retention.json").read_text())["upload_status"] == "failed"


def test_disabled_is_legacy_and_local_only_also_bounded(tmp_path, monkeypatch):
    t, cfg, rd, run, _, videos = setup(tmp_path, monkeypatch, track=False)
    originals = vars(t).copy()
    with install(t, cfg, rd, 0):
        assert vars(t) == originals
    with install(t, cfg, rd, 2):
        assert t.init_wandb(cfg, rd) is None
        for i in range(1, 5):
            t.save_local_eval_video(b"one", rd / f"videos/eval_{i:05d}.mp4", 10)
    assert len(list((rd / "videos").glob("*.mp4"))) == 2
    assert not videos and not run.finished


def test_refuses_unowned_or_existing_video(tmp_path, monkeypatch):
    t, cfg, rd, _, _, _ = setup(tmp_path, monkeypatch, track=False)
    with install(t, cfg, rd, 1):
        t.init_wandb(cfg, rd)
        outside = tmp_path / "old.mp4"
        outside.write_bytes(b"old")
        with pytest.raises(ValueError):
            t.save_local_eval_video(b"new", outside, 10)
        own = rd / "videos/eval_00001.mp4"
        t.save_local_eval_video(b"first", own, 10)
        with pytest.raises(FileExistsError):
            t.save_local_eval_video(b"second", own, 10)
        assert outside.read_bytes() == b"old" and own.read_bytes() == b"first"


def test_context_never_writes_to_an_existing_run_on_entry_failure(tmp_path, monkeypatch):
    t, cfg, rd, _, _, _ = setup(tmp_path, monkeypatch)
    with pytest.raises(FileExistsError):
        with install(t, cfg, rd, 10):
            rd.mkdir(exist_ok=False)  # experiment_logging refuses an existing job folder
    assert list(rd.iterdir()) == []


def test_negative_or_missing_video_flag_rejected(tmp_path, monkeypatch):
    t, cfg, rd, _, _, _ = setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        with install(t, cfg, rd, -1):
            pass
    cfg.save_eval_video = False
    with pytest.raises(ValueError):
        with install(t, cfg, rd, 10):
            pass


def test_plan_records_opt_in_without_changing_training_config(tmp_path, monkeypatch):
    from latent_team import train_from_scratch as S
    from latent_team import common as C
    if not (C.REFERENCE_EXPERIMENT / "experiment.json").is_file():
        pytest.skip("Local reference missing")
    monkeypatch.setattr(S, "build_bank", lambda *args, **kwargs: None)
    args = dict(reference_experiment=C.REFERENCE_EXPERIMENT, maps=["e1"], seeds=[1], modes=["ippo"],
                timesteps=10000000, home_ratio=.5, learning_rate=2e-4, entropy_coef=.05,
                save_interval=250, track=True, save_video=True, smoke_test=False, stage_label="test",
                wandb_project="rware-main-amax-v2", wandb_mode="offline", env_variant="b6_wide_corner_indiv_comm5_pen005",
                team_reward="individual", probe_bank="canonical", eval_seed=5000001)
    _, legacy = S.plan(SimpleNamespace(**args, output_dir=tmp_path / "legacy"))
    _, retained = S.plan(SimpleNamespace(**args, output_dir=tmp_path / "retained", video_keep_last=10))
    assert "video_keep_last" not in legacy
    assert retained["video_keep_last"] == retained["jobs"][0]["video_keep_last"] == 10
    assert retained["jobs"][0]["effective_timesteps"] == 10000384
    a, b = legacy["jobs"][0]["config"], retained["jobs"][0]["config"]
    assert {k: v for k, v in a.items() if k not in ("save_dir", "wandb_group")} == \
        {k: v for k, v in b.items() if k not in ("save_dir", "wandb_group")}
    with pytest.raises(ValueError, match="needs --save-video"):
        S.plan(SimpleNamespace(**dict(args, save_video=False), output_dir=tmp_path / "invalid", video_keep_last=10))
