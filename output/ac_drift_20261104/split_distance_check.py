"""Offline check of one candidate fix (not implemented in training): compute the response distance separately
for probes grouped by their public action and take the maximum over groups with enough probes. Same probes,
same saved predictors as drift_analysis.py. Rationale: equal reward rules imply agreement on every group, while
an event kind rewarded by every team (shelf returns) can no longer dilute a disagreement on another kind."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drift_analysis as DA  # noqa: E402
from latent_team import reward_probe_ac_v2 as V2  # noqa: E402

MIN_GROUP = 4


def distances(cfg, preds, i, j):
    parts = []
    for owner in (i, j):
        P = preds[owner]
        n = min(cfg.probes_per_agent, len(P["a"]))
        parts.append((P["b"][:n], P["a"][:n], P["f"][:n]))
    b, a, f = (np.concatenate(x) for x in zip(*parts))
    b, a, f = (np.concatenate([b, V2.mirror_view(b, cfg.radius)]), np.concatenate([a, V2.mirror_action(a)]),
               np.concatenate([f, V2.mirror_view(f, cfg.radius)]))
    p, q = preds[i]["model"].predict(b, a, f), preds[j]["model"].predict(b, a, f)
    d_all = V2.response_distance(p, q)
    groups = [V2.response_distance(p[a == act], q[a == act]) for act in np.unique(a) if (a == act).sum() >= MIN_GROUP]
    groups = [g for g in groups if g is not None]
    return d_all, max(groups) if groups else None


lines = []
for upd in DA.CKPTS:
    steps, cfg, preds, _ = DA.load(upd)
    same, cross = [], []
    for i in range(6):
        for j in range(i + 1, 6):
            if len(preds[i]["a"]) == 0 or len(preds[j]["a"]) == 0:
                continue
            d_all, d_max = distances(cfg, preds, i, j)
            (same if DA.TEAM[i] == DA.TEAM[j] else cross).append((d_all, d_max, (i, j)))
    row = (f"{steps / 1e6:.2f}M  pooled: same max {max(s[0] for s in same):.2f}, cross min {min(c[0] for c in cross):.2f} "
           f"{min(cross)[2]} | per-action max: same max {max(s[1] for s in same):.2f}, "
           f"cross min {min(c[1] for c in cross):.2f}")
    print(row)
    lines.append(row)
(Path(__file__).parent / "split_distance_check_stdout.txt").write_text("\n".join(lines), encoding="utf-8")
