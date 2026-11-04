"""Offline check of a candidate fix for the late different-team approvals (D107). Read-only, no training.

Candidate: compute the response distance separately for each public action of the probe events (a stratum) and
take the maximum over strata on which BOTH agents have their own experience (at least M own events with that action
among their recent probes). Under equal reward functions every stratum agrees; a stratum both teams are rewarded on
(shelf returns) can no longer dilute the stratum on which they differ (deliveries). The experience rule keeps a
predictor from being judged on scenes it has never been rewarded in (e.g. agent 0 has almost no returns).

Compared with the gate's pooled distance, the earlier per-action maximum without the experience rule, and a lower
pooled threshold. Uses the same saved predictors / event buffers / probe construction as the gate (recent 32 own
events of both agents, with or without left-right mirrors). Gate maturity (>= min_events seen and mean own-event
prediction >= min_event_response) is applied first, as in the gate.

Runs: the v2 discovery run (seed 1, 5M, all checkpoints) and the PGCT-D2 branch runs (pgct_max seeds 1/3/5,
0.5M after the 1.02M IPPO checkpoint; predictors trained from scratch inside the branch).
"""
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from latent_team import reward_probe_ac_v2 as V2  # noqa: E402

TEAM = np.arange(6) % 2                     # rware _balanced_assignments: agent_id % n_teams
MS = (2, 4, 8)


def load(path):
    c = torch.load(path, map_location="cpu", weights_only=False)
    s = c["reward_probe_ac_v2"]
    cfg = V2.ACV2Config(**s["config"])
    preds = []
    for p in s["predictors"]:
        rp = V2.RewardPredictor(cfg, 0)
        rp.net.load_state_dict(p["net"])
        n = min(cfg.probes_per_agent, len(p["event_action"]))
        P = dict(model=rp, b=p["event_before"][:n], a=p["event_action"][:n], f=p["event_after"][:n],
                 seen=p["events_seen"])
        P["mature"] = bool(n and P["seen"] >= cfg.min_events and
                           float(np.mean(rp.predict(P["b"], P["a"], P["f"]))) >= cfg.min_event_response)
        preds.append(P)
    return c["env_steps"], cfg, preds


def pair(cfg, preds, i, j, mirror):
    b, a, f = (np.concatenate([preds[k][x] for k in (i, j)]) for x in ("b", "a", "f"))
    if mirror:
        b, a, f = (np.concatenate([b, V2.mirror_view(b, cfg.radius)]), np.concatenate([a, V2.mirror_action(a)]),
                   np.concatenate([f, V2.mirror_view(f, cfg.radius)]))
    p, q = preds[i]["model"].predict(b, a, f), preds[j]["model"].predict(b, a, f)
    out = {"pooled": V2.response_distance(p, q)}
    acts = np.unique(a)
    groups = {int(g): V2.response_distance(p[a == g], q[a == g]) for g in acts if (a == g).sum() >= 4}
    out["per_action"] = max(groups.values()) if groups else None
    for m in MS:
        ok = [g for g in groups if min((preds[k]["a"] == g).sum() for k in (i, j)) >= m]
        out[f"strat_m{m}"] = (max(groups[g] for g in ok), max(ok, key=groups.get)) if ok else None
    return out


def summarize(tag, path, mirror, lines):
    steps, cfg, preds = load(path)
    same, cross = {}, {}
    for i in range(6):
        for j in range(i + 1, 6):
            if preds[i]["mature"] and preds[j]["mature"]:
                (same if TEAM[i] == TEAM[j] else cross)[(i, j)] = pair(cfg, preds, i, j, mirror)
    if not same and not cross:
        return
    row = [f"{tag} {steps / 1e6:.2f}M mature {sum(P['mature'] for P in preds)}"]
    for key in ["pooled", "per_action"] + [f"strat_m{m}" for m in MS]:
        val = lambda d: None if d[key] is None else (d[key][0] if isinstance(d[key], tuple) else d[key])
        s = [(val(d), pr) for pr, d in same.items() if val(d) is not None]
        c = [(val(d), pr) for pr, d in cross.items() if val(d) is not None]
        smax = f"{max(s)[0]:.2f}{max(s)[1]}" if s else "-"
        cmin = f"{min(c)[0]:.2f}{min(c)[1]}" if c else "-"
        row.append(f"{key}: same max {smax} / cross min {cmin} (n {len(s)}/{len(c)})")
    line = " | ".join(row)
    print(line)
    lines.append(line)


def main():
    lines = []
    disc = ROOT / "runs/acv2_s1_20261029_084340/b6_corner_indiv_pen005_acv2_pgct_s1"
    branch = ROOT / "runs/pgctd2_br1M_20261102_180212"
    for mirror in (True, False):
        lines.append(f"===== mirror {mirror}")
        print(lines[-1])
        for ck in sorted(disc.glob("checkpoint_*.pt")):
            summarize("disc_s1", ck, mirror, lines)
        for seed in (1, 3, 5):
            run = branch / f"b6_corner_indiv_pen005_acv2_br01000_pgct_max_s{seed}"
            for ck in [run / "checkpoint_00250.pt", run / "final.pt"]:
                summarize(f"branch_s{seed}", ck, mirror, lines)
    (Path(__file__).parent / "stratified_gate_check_stdout.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
