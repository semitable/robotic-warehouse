"""Tiny real-trainer / real-W&B-offline check. Not a training experiment."""
import json
import os
from datetime import datetime
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
tmp = OUT / "tmp"
tmp.mkdir(exist_ok=True)
os.environ.update(TEMP=str(tmp), TMP=str(tmp), WANDB_DISABLE_GIT="true", WANDB_SILENT="true",
                  OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

import torch
torch.set_num_threads(1)
from latent_team import common as C
from latent_team import train_from_scratch as S

args = SimpleNamespace(reference_experiment=C.REFERENCE_EXPERIMENT,
    output_dir=OUT / ("main_amax_" + datetime.now().strftime("%Y%m%d_%H%M%S") +
                     "_b6_wide_corner_indiv_comm5_pen005"),
    maps=["e1"], seeds=[1], modes=["ippo"], timesteps=64, home_ratio=.5,
    learning_rate=2e-4, entropy_coef=.05, save_interval=0, track=True, save_video=True,
    smoke_test=True, stage_label="video_retention_validation", wandb_project="rware-main-amax-v2",
    wandb_mode="offline", env_variant="b6_wide_corner_indiv_comm5_pen005", team_reward="individual",
    probe_bank="canonical", eval_seed=5000001, video_keep_last=10)
out, manifest = S.plan(args)
job = manifest["jobs"][0]
cfg = job["config"]
# Standard smoke disables video. Re-enable it here explicitly, retaining its tiny PPO settings.
cfg.update(total_timesteps=384, eval_interval=1, eval_episodes=1, eval_horizon=2,
           save_eval_video=True, wandb_log_eval_video=True, device="cpu")
job.update(video_keep_last=10, effective_timesteps=384)
(out / "experiment.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("running 12 tiny updates with real encoded video and W&B offline", flush=True)
S.run_job(out, manifest, job["name"], manifest["eval_seed"])
run_dir = out / job["name"]
status = json.loads((run_dir / "video_retention.json").read_text())
local = list((run_dir / "videos").glob("*.mp4"))
media_root = Path(status["wandb_local_dir"])
media = list(media_root.glob("wandb/offline-run-*/files/media/videos/**/*.mp4"))
result = dict(local_video_count=len(local), local_names=sorted(p.name for p in local),
              wandb_media_count=len(media), longest_wandb_path=max(map(lambda p: len(str(p)), media), default=0),
              local_run_path_chars=len(str(run_dir)), status=status,
              job_status=json.loads((run_dir / "job_status.json").read_text()))
assert len(local) == len(media) == 10, result
assert "final.mp4" in result["local_names"]
assert status["upload_status"] == "logged"
assert result["job_status"]["status"] == "complete"
assert (run_dir / "final.pt").is_file()
assert result["longest_wandb_path"] < 240
(OUT / "offline_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2), flush=True)
