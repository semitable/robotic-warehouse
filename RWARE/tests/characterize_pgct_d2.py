"""Trainer-level characterization of the RWARE PGCT-D2 pipeline (64-step smoke jobs, CPU, no W&B).

Runs the real trainer through latent_team/train_from_scratch.py for every Max-GAE mode under three settings
and prints a JSON fingerprint per job: SHA-256 of the final actor-critic weights, of the saved reward-gate
state and of the max_gae.jsonl rows (timing-free fields). Used before and after the pgct_core extraction
(docs/MIGRATION.md); the fingerprints must match exactly on the same machine/torch build.

  closed   : v2 refits but never matures (min_events huge) -> every mode reduces to IPPO
  injected : approval/maturity replaced by the true teams (test-only bypass) -> Max-GAE donors active
  natural  : permissive wiring-only gate values (not research values) -> real compare/approve path

Usage: python tests/characterize_pgct_d2.py OUTPUT_DIR [--variant b6_corner_indiv_comm5_pen005]
Needs the local reference configuration (latent_team/common.py REFERENCE_EXPERIMENT).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latent_team import train_from_scratch as S  # noqa: E402
from latent_team import reward_probe_ac_v2 as V2  # noqa: E402
from examples import train_recurrent_ippo_consensus as T  # noqa: E402

MODES = ["ippo", "pgct_max", "oracle_mature_max", "oracle_max", "all_max", "allk_max"]
SETTINGS = {
    "closed": dict(interval=1, batch_size=16, train_steps=2, min_events=1000000),
    "injected": dict(interval=1, batch_size=16, train_steps=2, min_events=1000000),
    "natural": dict(interval=1, batch_size=16, train_steps=2, min_events=1, min_event_response=-1.0,
                    event_reward=.001, min_rounds=1, distance_threshold=1.0, min_probes=1, mirror=False),
}


def digest(obj):
    h = hashlib.sha256()

    def feed(x):
        if isinstance(x, torch.Tensor):
            h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                h.update(str(k).encode())
                feed(x[k])
        elif isinstance(x, (list, tuple)):
            for v in x:
                feed(v)
        else:
            h.update(repr(x).encode())
    feed(obj)
    return h.hexdigest()[:16]


def plan(output, variant, options_path, modes, distance):
    args = SimpleNamespace(
        reference_experiment=S.C.REFERENCE_EXPERIMENT, output_dir=output, maps=["e1"], seeds=[1], modes=modes,
        timesteps=64, home_ratio=.5, learning_rate=2e-4, entropy_coef=.05, save_interval=0, track=False,
        save_video=False, smoke_test=True, stage_label="characterize", wandb_project="test", wandb_mode="offline",
        env_variant=variant, team_reward="individual", probe_bank="canonical", eval_seed=5000001,
        reward_probe_ac=True, ac_gate="v2", ac_config=options_path, ac_distance=distance, init_checkpoint=None,
        transfer_warmup=0, max_gae_eta=.5, max_gae_scale="shrink", max_gae_lambda=None, max_gae_cap=None)
    return S.plan(args)


def fingerprint(folder):
    payload = torch.load(folder / "final.pt", map_location="cpu", weights_only=False)
    rows = []
    path = folder / "max_gae.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rows.append({k: v for k, v in row.items() if "time" not in k and "elapsed" not in k})
    gate = payload.get("reward_probe_ac_v2")
    if gate:  # config keys added after the split (D139) are not part of the pre-split fingerprint
        gate = dict(gate, config={k: v for k, v in gate["config"].items()
                                  if k not in ("event_threshold", "event_scale", "min_event_response_ratio")})
    return dict(weights=digest(payload["agents"]), gate=digest(gate) if gate else None,
                max_gae=digest(rows) if rows else None,
                donor_arcs=[r.get("max_gae_donor_arcs") for r in rows],
                compared=sum(p["distance"] is not None for r in rows for p in r.get("ac_pairs", [])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--variant", default="b6_corner_indiv_comm5_pen005")
    parser.add_argument("--distance", default="action_max")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.output.exists():
        raise SystemExit(f"{args.output} exists")
    args.output.mkdir(parents=True)
    result = {}
    for setting, options in SETTINGS.items():
        options_path = args.output / f"{setting}.json"
        options_path.write_text(json.dumps(options), encoding="utf-8")
        out, manifest = plan(args.output / setting, args.variant, options_path, MODES, args.distance)
        original = (V2.ACV2Controller.approved, V2.ACV2Controller.maturity)
        if setting == "injected":
            with S.EC.agent_count_support(manifest["jobs"][0]["config"]["agent_count"]):
                env = T.make_env(manifest["jobs"][0]["config"]["env_id"],
                                 json.loads(manifest["jobs"][0]["config"]["env_kwargs_json"]), "semantic")
                try:
                    env.reset(seed=1)
                    teams = np.asarray(env.unwrapped.get_oracle_team_assignments())
                finally:
                    env.close()
            oracle = (teams[:, None] == teams[None, :]) & ~np.eye(len(teams), dtype=bool)
            V2.ACV2Controller.approved = lambda self: oracle.copy()
            V2.ACV2Controller.maturity = lambda self: np.ones(len(teams), bool)
        try:
            for job in manifest["jobs"]:
                S.run_job(out, manifest, job["name"], manifest["eval_seed"])
                result[f"{setting}/{job['mode']}"] = fingerprint(out / job["name"])
        finally:
            V2.ACV2Controller.approved, V2.ACV2Controller.maturity = original
    text = json.dumps(dict(torch=torch.__version__, numpy=np.__version__, variant=args.variant, jobs=result),
                      indent=1)
    (args.output / "fingerprints.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
