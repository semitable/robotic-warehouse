"""Evaluate a saved Overcooked checkpoint with fixed seeds (writes a new JSON next to the job; never overwrites).

  python -m pgct_overcooked.evaluate --job Overcooked/runs/<name>/pgct_d2_s1 [--checkpoint final.pt]
         [--episodes 20] [--seed 7000001] [--stochastic]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from . import REPO_ROOT  # noqa: F401  (pgct_core path)
from pgct_core.trainer import evaluate, load_agents
from .experiment import build


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--checkpoint", default="final.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7000001)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads((args.job / "config.json").read_text(encoding="utf-8"))
    experiment = config["experiment"]
    method, seed = experiment["trainer"]["method"], experiment["trainer"]["seed"]
    make_adapter, trainer_cfg, ppo_cfg, *_ = build(experiment, method, seed)
    trainer_cfg = replace(trainer_cfg, eval_episodes=args.episodes, eval_seed=args.seed,
                          eval_deterministic=not args.stochastic)
    adapter = make_adapter()
    agents, payload = load_agents(args.job / args.checkpoint, adapter.obs_dim, adapter.n_actions, ppo_cfg)
    summary = evaluate(make_adapter, agents, ppo_cfg, trainer_cfg, "cpu", adapter.oracle_team_ids())
    target = args.job / f"eval_{Path(args.checkpoint).stem}_seed{args.seed}_{'stoch' if args.stochastic else 'det'}.json"
    if target.exists():
        raise SystemExit(f"{target} exists; choose another --seed")
    target.write_text(json.dumps(dict(checkpoint=args.checkpoint, env_steps=payload.get("env_steps"),
                                      episodes=args.episodes, **summary), indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"-> {target}")


if __name__ == "__main__":
    main()
