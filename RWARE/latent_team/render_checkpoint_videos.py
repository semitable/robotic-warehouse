"""Render one evaluation episode per frozen checkpoint to mp4.  No updates, no transfer.

Uses the trainer's own evaluate() (deterministic masked policy, rgb_array) and
save_local_eval_video(), so the video matches what the eval metrics measure.
Writes only --output-dir; the run folders are read-only.

    python latent_team/render_checkpoint_videos.py --run runs/<plan folder> [--jobs e1f_pgct_s1] \
        [--checkpoint latest|checkpoint_01250.pt] [--horizon 500] [--seed 5000001]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples import train_recurrent_ippo_consensus as T  # noqa: E402
from examples.diagnose_pgct_rollout_compatibility import load_model, model_hash  # noqa: E402


def pick_checkpoint(job_dir: Path, name: str) -> Path:
    if name != "latest":
        return job_dir / name
    final = job_dir / "final.pt"
    if final.exists():
        return final
    cks = sorted(job_dir.glob("checkpoint_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not cks:
        raise FileNotFoundError(f"no checkpoint in {job_dir}")
    return cks[-1]


def render_job(run: Path, job: str, args, out: Path) -> dict:
    ck = pick_checkpoint(run / job, args.checkpoint)
    agents, meta = load_model(ck)
    cfg = meta["config"]
    env_kwargs = dict(T._load_checkpoint(str(ck), torch.device("cpu"))["effective_env_kwargs"])
    env_kwargs["render_mode"] = "rgb_array"
    before = model_hash(agents)
    result, video = T.evaluate(cfg["env_id"], env_kwargs, cfg["observation_format"], agents, 1, args.horizon,
                               cfg["gamma"], 0, cfg["recurrent_hidden_dim"], torch.device("cpu"),
                               seed=args.seed, use_action_mask=cfg["action_mask"], collect_video=True)
    if before != model_hash(agents):
        raise RuntimeError("weights changed during frozen rendering")
    steps = int(meta["env_steps"])
    path = out / f"{job}_{steps / 1e6:.2f}M.mp4"
    T.save_local_eval_video(video, path, args.fps)
    keep = ["eval_deliveries", "eval_returns_home", "eval_return", "eval_deadlock_agent_rate"]
    row = dict(job=job, checkpoint=str(ck.relative_to(ROOT)), env_steps=steps, video=path.name,
               frames=int(video.shape[0]), metrics={k: float(result[k]) for k in keep if k in result},
               per_agent_deliveries=[float(result.get(f"eval_agent_{i + 1}_delivery_count", float("nan")))
                                     for i in range(len(agents))])
    print(json.dumps(row), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="Plan folder (contains experiment.json).")
    ap.add_argument("--jobs", nargs="+", help="Job folder names; default: every job in the plan.")
    ap.add_argument("--checkpoint", default="latest", help="'latest' or a checkpoint file name.")
    ap.add_argument("--horizon", type=int, default=500)
    ap.add_argument("--seed", type=int, default=5000001, help="Reset seed (eval seeds start at 5,000,001).")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    run = args.run.resolve()
    jobs = args.jobs or [j["name"] for j in json.loads((run / "experiment.json").read_text(encoding="utf-8"))["jobs"]]
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = [render_job(run, job, args, out) for job in jobs]
    summary = out / f"summary_{run.name}.json"
    summary.write_text(json.dumps(dict(run=str(run), policy="frozen, deterministic masked, 1 episode",
                                       args={k: str(v) for k, v in vars(args).items()}, rows=rows), indent=2),
                       encoding="utf-8")
    print(f"Wrote {summary}", flush=True)


if __name__ == "__main__":
    main()
