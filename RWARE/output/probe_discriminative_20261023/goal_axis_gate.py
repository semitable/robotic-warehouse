"""Goal-axis statistic on the 13 discriminative gate probes vs mean JS.  Read-only; no training.

For each agent i and discriminative probe m (optimal first-action sets L_m for the left goal and R_m for the right
goal, both computed from the map alone):
    s_i = mean_m [ p_i(L_m) - p_i(R_m) ]        (probability mass; > 0 = behaves like a left-goal agent)
Pair distance |s_i - s_j| replaces the mean JS.  Every agent can compute s_i from the shared probes and the map; no
team label or goal assignment is needed.  Reported per checkpoint: team AUROC of |s_i - s_j| vs of mean JS on the
same 13 probes, and the AUROC of s_i itself against the agent's goal side (does s point to the right goal?).

    python output/probe_discriminative_20261023/goal_axis_gate.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import discriminative_probes as DP  # noqa: E402


def main():
    torch.set_num_threads(4)
    gate, opt, disc = DP.discriminative_set()
    out = {}
    rows = []
    for group, items in DP.GROUPS.items():
        for run, ck in items:
            agents, _ = DP.load_model(run / ck)
            c = DP.T._load_checkpoint(str(run / ck), torch.device("cpu"))
            kw = c["effective_env_kwargs"]
            with DP.T.make_env(c["config"]["env_id"], dict(kw), c["config"]["observation_format"]) as env:
                env.reset(seed=0)
                teams = [int(t) for t in np.asarray(env.unwrapped.agent_team_ids)]
            goal_left = [kw["team_goal_positions"][t][0][0] == 0 for t in teams]
            J, P = DP.js_and_probs(agents, gate)
            s = np.array([np.mean([P[i][k][sorted(opt[k][0])].sum() - P[i][k][sorted(opt[k][1])].sum() for k in disc])
                          for i in range(len(agents))])
            Dax = np.abs(s[:, None] - s[None, :])
            a_axis = DP.auroc(Dax, teams)
            a_js = DP.auroc(J[:, :, disc].mean(-1), teams)
            Ls = [s[i] for i in range(len(s)) if goal_left[i]]
            Rs = [s[i] for i in range(len(s)) if not goal_left[i]]
            a_dir = float(np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a in Ls for b in Rs]))
            key = f"{run.name} {ck}"
            out[key] = dict(group=group, steps=int(c["env_steps"]), s=np.round(s, 3).tolist(), goal_left=goal_left,
                            auroc_axis=a_axis, auroc_js13=a_js, auroc_direction=a_dir)
            rows.append((group, a_axis, a_js, a_dir))
            print(f"  {group:30s} {key:45s} {c['env_steps'] / 1e6:4.2f}M  pair AUROC: axis {a_axis:.2f} vs JS13 {a_js:.2f}   "
                  f"s points to own goal: {a_dir:.2f}   s = {np.round(s, 2).tolist()}", flush=True)
    arr = np.array([r[1:] for r in rows])
    print(f"== mean over {len(rows)} checkpoints: pair AUROC axis {arr[:, 0].mean():.3f}, JS13 {arr[:, 1].mean():.3f}; "
          f"direction {arr[:, 2].mean():.3f}; axis > JS13 in {(arr[:, 0] > arr[:, 1]).sum()}, < in {(arr[:, 0] < arr[:, 1]).sum()}", flush=True)
    for g in DP.GROUPS:
        sub = np.array([r[1:] for r in rows if r[0] == g])
        print(f"   {g:30s} axis {sub[:, 0].mean():.2f}  JS13 {sub[:, 1].mean():.2f}  direction {sub[:, 2].mean():.2f}  (n={len(sub)})", flush=True)
    (HERE / "goal_axis_results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
