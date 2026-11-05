"""How does the number of requested shelves per team change early exploration?  Read-only; no training.

Repeats the E44/E47 calculation (output/pickup_stall_20261021/action_values.py: uniform random policy over valid
actions, 20 episodes x 500 steps) in the corner / individual-reward cell (6 agents, penalty .005), changing only
request_queue_size_per_team (3 = current, 6, 9) and request_queue_size (= 2 x per team).  A random policy does not
react to rewards, and the shelf layout is the same, so only how many shelves count as "requested" changes.

    python output/request_sweep_20261023/request_sweep.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "output/pickup_stall_20261021"))
import action_values as AV  # noqa: E402

OUT = Path(__file__).resolve().parent
CK = ROOT / "runs/queue_20261022_140114_b6_corner_indiv_pen005/b6_corner_indiv_pen005_ippo_s1/checkpoint_00250.pt"
_load = AV.T._load_checkpoint


def patched(k):
    def load(path, device):
        c = _load(path, device)
        c["effective_env_kwargs"] = dict(c["effective_env_kwargs"], request_queue_size=2 * k, request_queue_size_per_team=[k, k])
        return c
    return load


def main():
    out = {}
    for k in (3, 6, 9):
        AV.T._load_checkpoint = patched(k)
        s = AV.run(CK)
        tog = s["toggle_outcomes"]
        n_toggle = sum(v["n"] for v in tog.values())
        own = tog.get("pickup_own", {"n": 0, "share": 0.0})
        seg = {kk: v for kk, v in s["carry_segments"].items() if kk.startswith("pickup_own")}
        out[k] = dict(immediate=s["immediate_by_action"], toggle_outcomes=tog, own_segments=seg,
                      own_pickups_per_agent_episode=round(own["n"] / 20 / 6, 3))
        imm = {a: v["mean"] for a, v in s["immediate_by_action"].items()}
        print(f"== {k} requests/team: TOGGLE immediate {imm['TOGGLE']:+.4f} (NOOP {imm['NOOP']:+.4f}, FWD {imm['FORWARD']:+.4f})  "
              f"own-requested share of toggles {own['share']:.3f}  own pickups/agent/episode {out[k]['own_pickups_per_agent_episode']}  "
              f"own carry segments {[(kk, v['n'], v['mean_return']) for kk, v in seg.items()]}", flush=True)
    AV.T._load_checkpoint = _load
    (OUT / "results.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
