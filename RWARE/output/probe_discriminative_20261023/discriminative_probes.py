"""Policy-probe team AUROC on the gate probes whose optimal action depends on the goal side.  Read-only; no training.

The E1-F v2 gate bank has 48 probes (carrying, goal out of view).  For each probe the shortest-path first action of a
loaded agent is computed for the left goal (0,10) and the right goal (9,10) (BFS on the map; loaded agents cannot
enter shelf cells).  13 probes have disjoint optimal-action sets ("discriminative"); the other 35 do not.

For every corner-goal checkpoint (frozen, weights hashed) this reports, with the PGCT gate rule (zero hidden,
unmasked softmax, mean JS):
  - team AUROC (same-team pair closer = 1) on all 48, on the 13 discriminative and on the 35 others
  - per agent, on the 13 probes: share of argmax actions in its OWN goal side's optimal set vs the OTHER side's
    ("goal consistency"; an agent that encodes its goal side should score own > other)
  - whether the agent learned (last 5 evaluations: deliveries >= 0.25 per episode)

    python output/probe_discriminative_20261023/discriminative_probes.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "latent_team"))
sys.path.insert(0, str(ROOT / "output/pgct_progress_paper_20261023"))
sys.path.insert(0, str(ROOT / "output/pen005_analysis_20261023"))
from examples import train_recurrent_ippo_consensus as T  # noqa: E402
from examples.diagnose_pgct_rollout_compatibility import model_hash  # noqa: E402
from reward_correlation_baseline import load_model  # noqa: E402
import make_figures as MF  # noqa: E402
import make_figures_plans as MP  # noqa: E402

R = ROOT / "runs"
ACT = {"NOOP": 0, "FWD": 1, "LEFT": 2, "RIGHT": 3}
GROUPS = {
    "corner/team .005 (B, 6 ag)": [(R / f"pilot_20261022_112521_b_e1f6_comm3_pen005/b_e1f6_comm3_pen005_ippo_s{s}", c)
                                   for s in (1, 2) for c in ("checkpoint_00500.pt", "checkpoint_00750.pt", "final.pt")],
    "corner/indiv .005 (B, 6 ag)": [(R / f"queue_20261022_140114_b6_corner_indiv_pen005/b6_corner_indiv_pen005_ippo_s{s}", c)
                                    for s in (1, 2) for c in ("checkpoint_00500.pt", "checkpoint_00750.pt", "final.pt")],
    "B team, penalty 0 (6 ag)": [(R / f"pilot_20261021_133652_b_e1f6_comm3_nopen/b_e1f6_comm3_nopen_ippo_s{s}", c)
                                 for s in (1, 2) for c in ("checkpoint_00750.pt", "final.pt")],
    "B team, penalty .02 (6 ag, 2M)": [(R / f"pilot_20261019_180044_b_e1f6_comm3/b_e1f6_comm3_ippo_s{s}", c)
                                       for s in (1, 2) for c in ("checkpoint_01000.pt", "final.pt")],
    "A team, penalty .02 (4 ag, 2M)": [(R / f"pilot_20261019_180044_a_e1f4_team/a_e1f4_team_ippo_s{s}", c)
                                       for s in (1, 2) for c in ("checkpoint_01000.pt", "final.pt")],
    "E1-F indiv .02 (4 ag, 1.54M)": [(R / f"e1f_20261018_124343_main_canonical/e1f_{m}_s1", "checkpoint_01500.pt")
                                     for m in ("ippo", "pgct", "oracle", "all")],
}


def discriminative_set():
    bank = torch.load(ROOT / "output/env_view_20261015/probes_v2/probe_bank_v2_e1f_s1.pt", map_location="cpu", weights_only=False)
    gate, gsc = bank["sequences"][:, 0].float(), bank["gate_scenarios"]
    st = MF.env_state(Path(str(MF.RUNS["corner/indiv, .005"]).format(1)))
    B = MP.blocked_map(st)
    H, W = B.shape
    opt = []
    for s in gsc:
        B2 = B.copy(); B2[s["y"], s["x"]] = False
        left = MP.first_actions(B2, (s["x"], s["y"]), s["direction"], (0, H - 1))
        right = MP.first_actions(B2, (s["x"], s["y"]), s["direction"], (W - 1, H - 1))
        opt.append(({ACT[a] for a in left}, {ACT[a] for a in right}))
    disc = [k for k, (l, r) in enumerate(opt) if l.isdisjoint(r)]
    return gate, opt, disc


def js_and_probs(agents, x):
    with torch.inference_mode():
        P = []
        for net in agents:
            h, _ = net.initial_hidden(len(x), torch.device("cpu"))
            logits, _ = net._actor_step(x, h)
            P.append(logits.softmax(-1).double().numpy())
    n = len(agents)
    J = np.zeros((n, n, len(x)))
    kl = lambda a, b: (a * (np.log(np.maximum(a, 1e-12)) - np.log(np.maximum(b, 1e-12)))).sum(-1)
    for i in range(n):
        for j in range(n):
            m = 0.5 * (P[i] + P[j])
            J[i, j] = 0.5 * kl(P[i], m) + 0.5 * kl(P[j], m)
    return J, P


def auroc(D, teams):
    n = len(teams)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    same = [D[i, j] for i, j in pairs if teams[i] == teams[j]]
    cross = [D[i, j] for i, j in pairs if teams[i] != teams[j]]
    return float(np.mean([1.0 if s < c else 0.5 if s == c else 0.0 for s in same for c in cross]))


def learned_agents(run, n):
    rows = [json.loads(l) for l in open(run / "metrics.jsonl", encoding="utf-8")]
    ev = [r["metrics"] for r in rows if "eval/deliveries" in r["metrics"]][-5:]
    return [float(np.mean([m.get(f"eval/agent_{k}_delivery_count", 0.0) for m in ev])) / 20 >= 0.25 for k in range(1, n + 1)]


def main():
    torch.set_num_threads(4)
    gate, opt, disc = discriminative_set()
    rest = [k for k in range(len(opt)) if k not in disc]
    print(f"discriminative gate probes: {len(disc)} / {len(opt)}  {disc}", flush=True)
    out = dict(discriminative=disc, optimal=[[sorted(l), sorted(r)] for l, r in opt], runs={})
    for group, items in GROUPS.items():
        print(f"== {group}", flush=True)
        for run, ck in items:
            agents, _ = load_model(run / ck)
            h0 = model_hash(agents)
            c = T._load_checkpoint(str(run / ck), torch.device("cpu"))
            kw = c["effective_env_kwargs"]
            with T.make_env(c["config"]["env_id"], dict(kw), c["config"]["observation_format"]) as env:
                env.reset(seed=0)
                teams = [int(t) for t in np.asarray(env.unwrapped.agent_team_ids)]
            goal_left = [kw["team_goal_positions"][t][0][0] == 0 for t in teams]
            J, P = js_and_probs(agents, gate)
            assert h0 == model_hash(agents)
            a_all, a_disc, a_rest = (auroc(J[:, :, idx].mean(-1), teams) for idx in (list(range(48)), disc, rest))
            cons = []
            for i in range(len(agents)):
                am = P[i][disc].argmax(-1)
                own = np.mean([am[q] in (opt[k][0] if goal_left[i] else opt[k][1]) for q, k in enumerate(disc)])
                oth = np.mean([am[q] in (opt[k][1] if goal_left[i] else opt[k][0]) for q, k in enumerate(disc)])
                cons.append((round(float(own), 2), round(float(oth), 2)))
            learned = learned_agents(run, len(agents))
            key = f"{run.name} {ck}"
            out["runs"][key] = dict(group=group, steps=int(c["env_steps"]), teams=teams, goal_left=goal_left,
                                    auroc_all48=a_all, auroc_disc13=a_disc, auroc_rest35=a_rest,
                                    consistency_own_other=cons, learned=learned)
            lc = [f"{'*' if learned[i] else ''}{i + 1}{'L' if goal_left[i] else 'R'}:{o:.2f}/{t:.2f}" for i, (o, t) in enumerate(cons)]
            print(f"  {key:45s} {c['env_steps'] / 1e6:4.2f}M  AUROC all48 {a_all:.2f} | disc13 {a_disc:.2f} | rest35 {a_rest:.2f}   "
                  f"own/other: {' '.join(lc)}", flush=True)
    (HERE / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
