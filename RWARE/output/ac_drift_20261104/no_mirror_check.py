"""Offline: the same saved predictors and recent-32 probes as drift_analysis.py, with and without the left-right
mirrors (the revised paper drops M). Predictors are not retrained, so this only shows whether probe composition
without mirrors still lets the shared return-event responses pull the different-team distance down."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drift_analysis as DA  # noqa: E402
from latent_team import reward_probe_ac_v2 as V2  # noqa: E402


def dist(cfg, preds, i, j, mirror):
    parts = []
    for owner in (i, j):
        P = preds[owner]
        n = min(cfg.probes_per_agent, len(P["a"]))
        parts.append((P["b"][:n], P["a"][:n], P["f"][:n]))
    b, a, f = (np.concatenate(x) for x in zip(*parts))
    if mirror:
        b, a, f = (np.concatenate([b, V2.mirror_view(b, cfg.radius)]), np.concatenate([a, V2.mirror_action(a)]),
                   np.concatenate([f, V2.mirror_view(f, cfg.radius)]))
    return V2.response_distance(preds[i]["model"].predict(b, a, f), preds[j]["model"].predict(b, a, f))


lines = []
for upd in DA.CKPTS:
    steps, cfg, preds, _ = DA.load(upd)
    out = {}
    for mirror in (True, False):
        same, cross = [], {}
        for i in range(6):
            for j in range(i + 1, 6):
                if len(preds[i]["a"]) == 0 or len(preds[j]["a"]) == 0:
                    continue
                d = dist(cfg, preds, i, j, mirror)
                if DA.TEAM[i] == DA.TEAM[j]:
                    same.append(d)
                else:
                    cross[(i, j)] = d
        out[mirror] = (max(same), cross)
    row = f"{steps / 1e6:.2f}M  same max: mirror {out[True][0]:.2f} / none {out[False][0]:.2f} | cross " + ", ".join(
        f"{p} {out[True][1][p]:.2f}/{out[False][1][p]:.2f}" for p in [(1, 2), (2, 3), (2, 5), (1, 4), (0, 1)]
        if p in out[True][1])
    print(row)
    lines.append(row)
(Path(__file__).parent / "no_mirror_check_stdout.txt").write_text("\n".join(lines), encoding="utf-8")
