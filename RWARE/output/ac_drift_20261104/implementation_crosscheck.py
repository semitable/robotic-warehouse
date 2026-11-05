"""Cross-check the implemented gate option (ACV2Config.distance_mode="action_max", D108) against the offline
analysis (stratified_gate_check.py, m = min_probes = 8) on the same saved checkpoints. Read-only, no training.

Each checkpoint's predictors (weights, version, events seen, event buffer in order) are restored into a fresh
ACV2Controller and every mature pair is compared through ACV2Controller.compare (the code path the trainer uses).
"""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stratified_gate_check as SG  # noqa: E402
from latent_team import reward_probe_ac_v2 as V2  # noqa: E402


def restore(path, mirror):
    c = torch.load(path, map_location="cpu", weights_only=False)
    s = c["reward_probe_ac_v2"]
    cfg = replace(V2.ACV2Config(**s["config"]), distance_mode="action_max", mirror=mirror, contact_snapshot=False)
    ctrl = V2.ACV2Controller(len(s["predictors"]), cfg)
    for pred, st in zip(ctrl.predictors, s["predictors"]):
        pred.net.load_state_dict(st["net"])
        pred.version, pred.events_seen = st["version"], st["events_seen"]
        n = len(st["event_action"])
        for k in range(n - 1, -1, -1):  # saved newest first; re-add oldest first so recent() matches
            pred.events.add(st["event_before"][k], st["event_action"][k], st["event_after"][k], st["event_reward"][k])
    return c["env_steps"], ctrl


def main():
    lines, worst = [], 0.0
    runs = [("disc_s1", sorted((SG.ROOT / "runs/acv2_s1_20261029_084340/b6_corner_indiv_pen005_acv2_pgct_s1")
                               .glob("checkpoint_*.pt")))]
    for seed in (1, 3, 5):
        run = SG.ROOT / f"runs/pgctd2_br1M_20261102_180212/b6_corner_indiv_pen005_acv2_br01000_pgct_max_s{seed}"
        runs.append((f"branch_s{seed}", [run / "checkpoint_00250.pt", run / "final.pt"]))
    for mirror in (False, True):
        same_max, cross_min, n_pairs = 0.0, 1.0, 0
        for tag, paths in runs:
            for path in paths:
                steps, ctrl = restore(path, mirror)
                _, cfg_off, preds_off = SG.load(path)
                mature = [p.mature() for p in ctrl.predictors]
                assert mature == [p["mature"] for p in preds_off], (tag, steps)
                for i in range(6):
                    for j in range(i + 1, 6):
                        if not (mature[i] and mature[j]):
                            continue
                        row = ctrl.compare(i, j, mature)
                        off = SG.pair(cfg_off, preds_off, i, j, mirror)
                        ref = off["strat_m8"]
                        assert (row["distance"] is None) == (ref is None), (tag, steps, i, j)
                        if ref is not None:
                            worst = max(worst, abs(row["distance"] - ref[0]))
                            assert row["stratum"] == ref[1], (tag, steps, i, j)
                        worst = max(worst, abs(row["pooled_distance"] - off["pooled"]))
                        n_pairs += 1
                        if row["distance"] is not None:
                            if SG.TEAM[i] == SG.TEAM[j]:
                                same_max = max(same_max, row["distance"])
                            else:
                                cross_min = min(cross_min, row["distance"])
        line = (f"mirror {mirror}: {n_pairs} mature pairs over 25 checkpoints; implemented action_max "
                f"same-team max {same_max:.3f}, different-team min {cross_min:.3f}")
        print(line)
        lines.append(line)
    lines.append(f"max |implemented - offline| over all distances: {worst:.2e}")
    print(lines[-1])
    (HERE / "implementation_crosscheck_stdout.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
