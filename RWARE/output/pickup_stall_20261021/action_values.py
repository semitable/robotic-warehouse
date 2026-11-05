"""Immediate reward of each action under the initial (uniform random over valid actions) policy, and
what a TOGGLE leads to.  Read-only; no training.  Uses the environment of the A pilot (team reward) and
of E1-F (individual reward); shaping coefficients are the same.

For TOGGLE the outcome is split: pickup of own requested shelf / other team's requested shelf /
unrequested shelf, drop (before delivery), other.  For each pickup the reward the agent then collects
until the shelf leaves its hands (the "carry segment") is summed, by how the segment ends.

    python output/pickup_stall_20261021/action_values.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples import train_recurrent_ippo_consensus as T  # noqa: E402

OUT = Path(__file__).resolve().parent
ENVS = {"A (team reward)": ROOT / "runs/pilot_20261019_180044_a_e1f4_team/a_e1f4_team_ippo_s1/checkpoint_00250.pt",
        "E1-F (individual)": ROOT / "runs/e1f_20261018_124343_main_canonical/e1f_ippo_s1/checkpoint_00250.pt"}
NAMES = ["NOOP", "FORWARD", "LEFT", "RIGHT", "TOGGLE"]


def run(ck, episodes=20, horizon=500, seed=8100001):
    c = T._load_checkpoint(str(ck), torch.device("cpu"))
    cfg, kw = c["config"], dict(c["effective_env_kwargs"])
    coeffs = {k: kw.get(k) for k in kw if k.endswith(("_penalty", "_reward"))}
    rng = np.random.default_rng(seed)
    by_action = defaultdict(list)
    toggle = defaultdict(list)
    seg = defaultdict(list)
    with T.make_env(cfg["env_id"], kw, cfg["observation_format"]) as env:
        u = env.unwrapped
        for e in range(episodes):
            env.reset(seed=seed + e)
            n = len(u.agents)
            open_seg = [None] * n
            for _ in range(horizon):
                masks = np.asarray(T.local_rware_action_masks(env, n, 5, enabled=True), dtype=np.float64)
                acts = [int(rng.choice(5, p=m / m.sum())) for m in masks]
                before = [a.carrying_shelf for a in u.agents]
                _, r, term, trunc, info = env.step(acts)
                r = np.asarray(r, dtype=np.float64)
                picks = {int(ev["agent_id"]) - 1: ev for ev in info.get("pickup_events", [])}
                delivered = {int(ev.get("carrier_id", 0)) - 1 for ev in info.get("delivery_events", [])}
                for i, a in enumerate(u.agents):
                    by_action[NAMES[acts[i]]].append(r[i])
                    if acts[i] == 4:
                        if i in picks:
                            sh = next(s for s in u.shelfs if s.id == picks[i]["shelf_id"])
                            kind = ("pickup_own" if u._is_agent_requested_shelf(a, sh) else
                                    "pickup_other_team_request" if u._is_requested_shelf(sh) else "pickup_unrequested")
                        elif before[i] is not None and a.carrying_shelf is None:
                            kind = "drop"
                        else:
                            kind = "toggle_nothing"
                        toggle[kind].append(r[i])
                        if kind.startswith("pickup"):
                            open_seg[i] = dict(kind=kind, total=r[i], steps=1)
                            continue
                    if open_seg[i] is not None:
                        open_seg[i]["total"] += r[i]; open_seg[i]["steps"] += 1
                        if a.carrying_shelf is None:
                            end = "delivered" if i in delivered else "dropped/returned"
                            seg[(open_seg[i]["kind"], end)].append((open_seg[i]["total"], open_seg[i]["steps"]))
                            open_seg[i] = None
                if bool(np.any(term)) or bool(np.any(trunc)):
                    break
            for i in range(n):
                if open_seg[i] is not None:
                    seg[(open_seg[i]["kind"], "episode end")].append((open_seg[i]["total"], open_seg[i]["steps"]))
    summ = dict(coefficients=coeffs,
                immediate_by_action={k: dict(n=len(v), mean=round(float(np.mean(v)), 5)) for k, v in by_action.items()},
                toggle_outcomes={k: dict(n=len(v), share=round(len(v) / sum(map(len, toggle.values())), 4),
                                         mean=round(float(np.mean(v)), 5)) for k, v in toggle.items()},
                carry_segments={f"{k[0]} -> {k[1]}": dict(n=len(v), mean_return=round(float(np.mean([x[0] for x in v])), 4),
                                                          mean_steps=round(float(np.mean([x[1] for x in v])), 1))
                                for k, v in seg.items()})
    return summ


def main():
    out = {}
    for name, ck in ENVS.items():
        out[name] = s = run(ck)
        print(f"== {name}")
        print("  immediate reward by action:", s["immediate_by_action"])
        print("  TOGGLE outcomes:", s["toggle_outcomes"])
        for k, v in s["carry_segments"].items():
            print(f"  segment {k}: {v}")
    (OUT / "action_values.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
