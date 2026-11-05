"""Penalty sweep for the pickup stall (D69/D70).  Read-only; no training.

Repeats the E44 calculation (output/pickup_stall_20261021/action_values.py): uniform random policy over valid
actions, 20 episodes x 500 steps, immediate reward of each action and what a TOGGLE leads to.  Only
unrequested_shelf_pickup_penalty and premature_drop_penalty change, both set to p.  Rewards do not affect a random
policy, so every p sees exactly the same trajectories; only the reward attached to them differs.

    python output/penalty_sweep_20261022/penalty_sweep.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "output/pickup_stall_20261021"))
import action_values as AV  # noqa: E402

OUT = Path(__file__).resolve().parent
ENVS = {"A": ROOT / "runs/pilot_20261019_180044_a_e1f4_team/a_e1f4_team_ippo_s1/checkpoint_00250.pt",
        "B": ROOT / "runs/pilot_20261019_180044_b_e1f6_comm3/b_e1f6_comm3_ippo_s1/checkpoint_00250.pt"}
PENALTIES = [0.0, 0.005, 0.01, 0.015, 0.02]
KEYS = ("unrequested_shelf_pickup_penalty", "premature_drop_penalty")

_load = AV.T._load_checkpoint


def patched(p):
    def load(path, device):
        c = _load(path, device)
        c["effective_env_kwargs"] = dict(c["effective_env_kwargs"], **{k: p for k in KEYS})
        return c
    return load


def main():
    out = {}
    for env, ck in ENVS.items():
        for p in PENALTIES:
            AV.T._load_checkpoint = patched(p)
            s = AV.run(ck)
            assert all(s["coefficients"][k] == p for k in KEYS)
            imm = {k: v["mean"] for k, v in s["immediate_by_action"].items()}
            rank = sorted(imm, key=imm.get)
            out[f"{env} p={p}"] = dict(s, toggle_rank_from_worst=rank.index("TOGGLE") + 1, worst=rank[0])
            own = {k: v for k, v in s["carry_segments"].items() if k.startswith("pickup_own")}
            print(f"== {env} p={p}: worst={rank[0]}  " + "  ".join(f"{k} {imm[k]:+.4f}" for k in AV.NAMES))
            print("   own-shelf segments:", {k: v["mean_return"] for k, v in own.items()})
            print("   unrequested segments:", {k: v["mean_return"] for k, v in s["carry_segments"].items()
                                               if k.startswith("pickup_unrequested")})
    AV.T._load_checkpoint = _load
    (OUT / "results.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
