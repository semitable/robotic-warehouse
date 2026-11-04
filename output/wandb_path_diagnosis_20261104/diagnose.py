"""Read-only run diagnosis and isolated filesystem reproduction results (no training)."""
import ast
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PLAN = ROOT / "runs/main_amax_20261104_115922_b6_wide_corner_indiv_comm5_pen005"
manifest = json.loads((PLAN / "experiment.json").read_text(encoding="utf-8"))
rows = []
for job in manifest["jobs"]:
    run_dir = PLAN / job["name"]
    status = json.loads((run_dir / "job_status.json").read_text(encoding="utf-8"))
    lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    metric = json.loads(lines[-1]) if lines else {}
    row = dict(mode=job["mode"], seed=job["seed"], status=status["status"],
               env_steps=metric.get("env_steps"), error=status.get("error"),
               checkpoints=[p.name for p in run_dir.glob("*.pt")],
               local_videos=[p.name for p in (run_dir / "videos").glob("*.mp4")])
    if status.get("traceback"):
        last = status["traceback"].splitlines()[-1]
        if "directory: " in last:
            target = ast.literal_eval(last.split("directory: ", 1)[1])
            row.update(target=target, target_chars=len(target), parent_exists=os.path.isdir(os.path.dirname(target)))
    rows.append(row)

repro = json.loads((OUT / "path_probe.json").read_text(encoding="utf-8"))

cfg = manifest["jobs"][0]["config"]
result = dict(checked_kst=datetime.now(ZoneInfo("Asia/Seoul")).isoformat(), python=sys.version,
              wandb_version="0.28.0 (training log)", plan=str(PLAN),
              config={key: cfg.get(key) for key in ["total_timesteps", "rollout_steps", "eval_interval",
                      "eval_episodes", "eval_horizon", "save_interval", "save_eval_video",
                      "wandb_log_eval_video", "wandb_video_interval"]},
              counts=dict(Counter(row["status"] for row in rows)), jobs=rows, reproduction=repro)
(OUT / "diagnosis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
