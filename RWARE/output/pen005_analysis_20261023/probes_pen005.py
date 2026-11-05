"""pen005 pilot: reward-coupling vs policy probe AUROC (as E49) and online identification time (as E50).  Read-only.

Reuses output/reward_coupling_20261022/{reward_coupling,identification_time}.py unchanged; only the runs differ.

    python output/pen005_analysis_20261023/probes_pen005.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "output/reward_coupling_20261022"))
import identification_time as IT  # noqa: E402
import reward_coupling as RC  # noqa: E402

OUT = Path(__file__).resolve().parent
# optional: python probes_pen005.py <plan dir> <candidate name> <output json name>
PLAN = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs/pilot_20261022_112521_b_e1f6_comm3_pen005"
CAND = sys.argv[2] if len(sys.argv) > 2 else "b_e1f6_comm3_pen005"
RESULT = sys.argv[3] if len(sys.argv) > 3 else "results.json"
RUN = CAND + "_ippo_s{}"


def main():
    torch.set_num_threads(4)
    v2 = torch.load(RC.V2_GATE, map_location="cpu", weights_only=False)["sequences"][:, 0].float()
    auroc, ident = {}, {}
    for s in (1, 2):
        run = PLAN / RUN.format(s)
        canon = torch.load(PLAN / f"canonical_bank_{CAND}_s{s}.pt", map_location="cpu",
                           weights_only=False)["sequences"][:, 0].float()
        for ck in ["checkpoint_00750.pt", "final.pt"]:
            agents, episodes, visited, sensor, steps = RC.rollout(run / ck)
            n, teams = len(agents), episodes[0]["teams"]
            sc, detail = RC.coupling_scores(episodes, n, sensor, teams)
            for label, x in (("policy_canonical", canon), ("policy_v2_gate", v2),
                             ("policy_visited", torch.as_tensor(visited[:RC.VISITED]))):
                sc[label] = RC.score(-RC.js_matrix(agents, x), teams)
            deliveries = sum(1 for ep in episodes for st in ep["steps"] for ev in st["events"] if ev[0] == "delivery")
            auroc[f"s{s} {ck}"] = dict(env_steps=steps, deliveries=deliveries, scores=sc, seen=detail["counts"],
                                       responses=detail["responses"])
            print(f"== AUROC s{s} {ck} ({steps / 1e6:.2f}M) deliveries={deliveries}  " +
                  "  ".join(f"{k}={'-' if v['auroc'] is None else round(v['auroc'], 2)}({v['defined_pairs']})"
                            for k, v in sc.items()), flush=True)
        jobs = [("uniform random", run / "checkpoint_00250.pt", False)] + \
               [(ck, run / ck, True) for ck in IT.CKPTS]
        for label, ck, use_policy in jobs:
            st, teams, goals, sensor, comm = IT.rollout(ck, use_policy)
            for k in IT.KS:
                res = IT.online(st, teams, goals, sensor, comm, k)
                res["deliveries"] = sum(len(x["events"]) for x in st)
                ident[f"s{s} {label} K={k}"] = res
                print(f"   ident s{s} {label:20s} K={k} deliveries={res['deliveries']:4d} all15={res['all_15_correct_at']} "
                      f"first_wrong={res['first_wrong_decision_at']} marks={ {m: (v['correct'], v['wrong'], v['undecided']) for m, v in res['marks'].items()} }",
                      flush=True)
    (OUT / RESULT).write_text(json.dumps(dict(auroc=auroc, identification=ident), indent=1, default=float),
                                      encoding="utf-8")


if __name__ == "__main__":
    main()
