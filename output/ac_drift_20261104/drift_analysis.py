"""Why does the different-team response distance fall late in the v2 discovery run? (read-only, no training)

Loads the saved reward predictors and event buffers of runs/acv2_s1_20261029_084340 (corner goals, individual
rewards, seed 1) at several checkpoints, rebuilds each pair's probes exactly as the run did (32 most recent own
events of both agents + left-right mirrors), and splits the response distance by probe type and owner.
Also counts different-team peer-access arcs per pair from transfer.jsonl.
"""
from pathlib import Path
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from latent_team import reward_probe_ac_v2 as V2  # noqa: E402

RUN = ROOT / "runs/acv2_s1_20261029_084340/b6_corner_indiv_pen005_acv2_pgct_s1"
TEAM = np.array([0, 1, 0, 1, 0, 1])          # from the saved EMA matrix: {0,2,4} vs {1,3,5}; checked below
CKPTS = [1000, 2000, 3000, 3500, 4000, 4250, 4500, 4750]
FORWARD, TOGGLE = 1, 4


def load(update):
    c = torch.load(RUN / f"checkpoint_{update:05d}.pt", map_location="cpu", weights_only=False)
    s = c["reward_probe_ac_v2"]
    cfg = V2.ACV2Config(**s["config"])
    preds = []
    for k, p in enumerate(s["predictors"]):
        rp = V2.RewardPredictor(cfg, 0)
        rp.net.load_state_dict(p["net"])
        preds.append(dict(model=rp, b=p["event_before"], a=p["event_action"], f=p["event_after"],
                          r=p["event_reward"], seen=p["events_seen"]))
    return c["env_steps"], cfg, preds, s


def kind(a, b, f):
    """delivery = FORWARD event; return = TOGGLE that puts a carried shelf down (ego[0] = carrying)."""
    if a == FORWARD:
        return "delivery"
    if a == TOGGLE and b[0] > 0.5 and f[0] < 0.5:
        return "return"
    return "other"


def pair_breakdown(cfg, preds, i, j):
    parts, tags = [], []
    for owner in (i, j):
        P = preds[owner]
        n = min(cfg.probes_per_agent, len(P["a"]))
        parts.append((P["b"][:n], P["a"][:n], P["f"][:n]))
        tags += [(owner, kind(P["a"][k], P["b"][k], P["f"][k])) for k in range(n)]
    b, a, f = (np.concatenate(x) for x in zip(*parts))
    tags = [(o, t, "orig") for o, t in tags] + [(o, t, "mirror") for o, t in tags]
    b, a, f = (np.concatenate([b, V2.mirror_view(b, cfg.radius)]), np.concatenate([a, V2.mirror_action(a)]),
               np.concatenate([f, V2.mirror_view(f, cfg.radius)]))
    p, q = preds[i]["model"].predict(b, a, f), preds[j]["model"].predict(b, a, f)
    num, den = np.abs(p - q), np.abs(p) + np.abs(q)
    out = {}
    for k, tg in enumerate(tags):
        key = ("own" if tg[0] == i else "peer") + ":" + tg[1] + ":" + tg[2]
        o = out.setdefault(key, [0.0, 0.0, 0, [], []])
        o[0] += num[k]; o[1] += den[k]; o[2] += 1; o[3].append(p[k]); o[4].append(q[k])
    return float(num.sum() / den.sum()), out


def main():
    lines = []
    say = lambda s="": (print(s), lines.append(s))
    for upd in CKPTS:
        steps, cfg, preds, s = load(upd)
        ema = s["ema"]
        say(f"=== checkpoint {upd} ({steps / 1e6:.2f}M steps)")
        say("events seen: " + ", ".join(f"a{k} {P['seen']}" for k, P in enumerate(preds)))
        comp = []
        for k, P in enumerate(preds):
            n = min(32, len(P["a"]))
            kinds = [kind(P["a"][m], P["b"][m], P["f"][m]) for m in range(n)]
            comp.append(f"a{k} D{kinds.count('delivery')}/R{kinds.count('return')}/O{kinds.count('other')}")
        say("recent 32 own events (Delivery/Return/Other): " + ", ".join(comp))
        same = [ema[i, j] for i in range(6) for j in range(i + 1, 6) if TEAM[i] == TEAM[j]]
        cross = [(i, j, ema[i, j]) for i in range(6) for j in range(i + 1, 6) if TEAM[i] != TEAM[j]]
        say(f"EMA same-team max {max(same):.2f}; cross-team " + ", ".join(f"({i},{j}) {e:.2f}" for i, j, e in cross))
        for i, j, e in cross:
            if len(preds[i]["a"]) == 0 or len(preds[j]["a"]) == 0:
                continue
            d, br = pair_breakdown(cfg, preds, i, j)
            share = {k: (v[0], v[1]) for k, v in br.items()}
            tot_den = sum(v[1] for v in share.values())
            agree = sum(v[1] - v[0] for v in share.values())
            top = sorted(br.items(), key=lambda kv: -(kv[1][1] - kv[1][0]))[:3]
            say(f"  ({i},{j}) distance {d:.2f}; shared mass {agree / tot_den:.2f} from: " +
                "; ".join(f"{k} n={v[2]} p={np.mean(v[3]):.2f} q={np.mean(v[4]):.2f}" for k, v in top))
        say()

    # Different-team peer access per pair over time (transfer.jsonl, update-end contacts)
    rows = [json.loads(l) for l in open(RUN / "transfer.jsonl", encoding="utf-8")]
    counts = {}
    for r in rows:
        A = np.asarray(r["actual_allocation"]) > 0
        for i in range(6):
            for j in range(6):
                if A[i, j] and TEAM[i] != TEAM[j]:
                    counts.setdefault((min(i, j), max(i, j)), []).append(r["env_steps"])
    say("=== different-team access arcs by pair (directed arcs counted)")
    for pair, steps in sorted(counts.items(), key=lambda kv: -len(kv[1])):
        say(f"  {pair}: {len(steps)} arcs, first {min(steps) / 1e6:.2f}M, last {max(steps) / 1e6:.2f}M")
    (Path(__file__).parent / "drift_analysis_stdout.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
