"""Experience-volume rollouts for the early-identification check (no training of the policy).

Each checkpoint stands in for its training window: ckpt k is rolled out for the env steps between the
previous checkpoint and k (e.g. checkpoint_00250 for 0-0.256M), so the replayed experience volume
matches training. Per step and agent it stores
  (1) the A+C public features + own reward split into components (delivery, return, pickup,
      goal-progress shaping, rest) for the component audit (audit.py), and
  (2) streams the public transitions into the real gate v2 controller (reward_probe_ac_v2) with the
      run's rhythm: a round every 5 updates (5,120 env steps), warm-up 100 updates, physical
      neighbours within the communication range. Every 4 rounds all 15 pairs are scored with the same
      predictors by (a) v2's normalised absolute distance on reward-event probes and (b) the
      contrast measure c_i(q) = r_i(q) - r_i(mirror q) on carrying-move probes + events.
Team labels and reward components come from the environment and are used only for scoring/audit.
Usage: python output/ac_early_id_20261026/stream_eval.py --run <job dir> --ckpts checkpoint_00250.pt:256000 ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "latent_team"))
from examples import train_recurrent_ippo_consensus as T  # noqa: E402
from reward_correlation_baseline import load_model  # noqa: E402
import reward_probe_ac as AC  # noqa: E402
import reward_probe_ac_v2 as V2  # noqa: E402

OUT = Path(__file__).resolve().parent
HORIZON, SEED, R = 500, 9500001, 2
ROUND_STEPS, WARMUP_STEPS, EVAL_EVERY = 5 * 1024, 100 * 1024, 4
FORWARD = 1
GOAL_PROGRESS = .005
torch.set_num_threads(2)


def auroc(score, same):
    score, same = np.asarray(score, float), np.asarray(same, bool)
    ok = np.isfinite(score)
    a, b = score[ok & same], score[ok & ~same]
    if not len(a) or not len(b):
        return None
    return float(np.mean([(x > y) + .5 * (x == y) for x in a for y in b]))


class Probes:
    """Recent own carrying-FORWARD public transitions (selected by public features only)."""
    def __init__(self, n, size=128):
        self.q = [deque(maxlen=size) for _ in range(n)]

    def add(self, i, b, a, f):
        if b[0] == 1 and a == FORWARD:
            self.q[i].append((b, a, f))

    def get(self, i, k):
        items = list(self.q[i])[-k:]
        if not items:
            return None
        b, a, f = zip(*items)
        return np.stack(b), np.array(a), np.stack(f)


def contrast_scores(ctrl, probes, i, j, k=64):
    parts = [x for x in (probes.get(i, k), probes.get(j, k), ctrl.predictors[i].event_probes(),
                         ctrl.predictors[j].event_probes()) if x is not None and len(x[0])]
    if not parts:
        return np.nan, np.nan, 0
    b, a, f = (np.concatenate(x) for x in zip(*parts))
    mb, ma, mf = V2.mirror_view(b, R), V2.mirror_action(a), V2.mirror_view(f, R)
    ci = ctrl.predictors[i].predict(b, a, f) - ctrl.predictors[i].predict(mb, ma, mf)
    cj = ctrl.predictors[j].predict(b, a, f) - ctrl.predictors[j].predict(mb, ma, mf)
    corr = float(np.corrcoef(ci, cj)[0, 1]) if ci.std() > 1e-9 and cj.std() > 1e-9 else np.nan
    eps = .002
    conf = (np.abs(ci) > eps) & (np.abs(cj) > eps)
    sign = float(np.mean(np.sign(ci[conf]) == np.sign(cj[conf]))) if conf.sum() >= 8 else np.nan
    return corr, sign, int(len(b))


def v2_distance(ctrl, i, j):
    b, a, f = ctrl.probes(i, j)
    if len(b) < ctrl.config.min_probes:
        return np.nan
    d = V2.response_distance(ctrl.predictors[i].predict(b, a, f), ctrl.predictors[j].predict(b, a, f))
    return np.nan if d is None else d


def landmark_key(view, table):
    key = view[V2.EGO:V2.EGO + 25].tobytes() + view[V2.EGO + 10 * 25:V2.EGO + 11 * 25].tobytes()  # oob + goal
    return table.setdefault(key, len(table))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--ckpts", nargs="+", required=True, help="name:end_env_step, in training order")
    p.add_argument("--unmasked-policy", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--max-steps", type=int, default=0, help="debug: stop after this many env steps")
    args = p.parse_args()
    run = ROOT / args.run
    ckpts = [(c.split(":")[0], int(c.split(":")[1])) for c in args.ckpts]
    rng = np.random.default_rng(SEED)
    ctrl, probes, lm_table = None, None, {}
    FLOATS = ("reward", "delivery", "return_", "pickup", "goal_progress", "rest")
    INTS = ("agent", "team", "ckpt", "carrying", "heading", "action", "highway", "landmark")
    cap = ckpts[-1][1] * 8  # rows = env steps x agents (<= 8 agents); trimmed when saving
    cols = {k: np.zeros(cap, np.float32) for k in FLOATS}
    cols.update({k: np.zeros(cap, np.int32) for k in INTS})
    nrow = 0
    curve, total, start, episode = [], 0, 0, 0
    t0 = time.time()
    for ci_, (name, end) in enumerate(ckpts):
        agents, _ = load_model(run / name)
        c = T._load_checkpoint(str(run / name), torch.device("cpu"))
        cfg, kw = c["config"], dict(c["effective_env_kwargs"])
        assert kw["team_reward_mode"] == "individual"
        with T.make_env(cfg["env_id"], kw, cfg["observation_format"]) as env:
            base = env.unwrapped
            while total < end and not (args.max_steps and total >= args.max_steps):
                obs, _ = env.reset(seed=SEED + episode)
                episode += 1
                n = len(base.agents)
                if ctrl is None:
                    ctrl, probes = V2.ACV2Controller(n, V2.ACV2Config()), Probes(n)
                    teams = np.asarray(base.agent_team_ids).copy()
                    same = np.array([[teams[i] == teams[j] for j in range(n)] for i in range(n)])
                    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
                comm = int(getattr(base, "communication_range", cfg.get("sensor_range", 3)))
                hidden = [net.initial_hidden(1, torch.device("cpu"))[0] for net in agents]
                raw = np.asarray(T.flatten_multi_agent_obs(env, obs), dtype=np.float32)
                for t in range(HORIZON):
                    x = AC.strip_coordinates(raw)
                    policy_in = raw if args.unmasked_policy else x
                    masks = np.asarray(T.local_rware_action_masks(env, n, 5, enabled=True), dtype=np.float32)
                    acts = []
                    with torch.inference_mode():
                        for i, net in enumerate(agents):
                            logits, hidden[i] = net._actor_step(torch.as_tensor(policy_in[i]).unsqueeze(0), hidden[i])
                            pr = net._masked_logits(logits, torch.as_tensor(masks[i])).softmax(-1).numpy()[0].astype(np.float64)
                            acts.append(int(rng.choice(5, p=pr / pr.sum())))
                    # state needed for the reward components (audit only)
                    carried = [a.carrying_shelf for a in base.agents]
                    goal_d = [base._carried_requested_shelf_delivery_goal_distance(a) if a.carrying_shelf is not None
                              else None for a in base.agents]
                    picked_before = [set(s) for s in base._picked_request_shelf_ids_by_agent]
                    obs, r, term, trunc, info = env.step(acts)
                    raw = np.asarray(T.flatten_multi_agent_obs(env, obs), dtype=np.float32)
                    y = AC.strip_coordinates(raw)
                    comp = np.zeros((n, 4))  # delivery, return, pickup, goal progress
                    for ev in info.get("delivery_events", []):
                        for k in ev.get("rewarded_agents", []):
                            comp[k, 0] += base.delivery_reward
                    for ev in info.get("return_events", []):
                        for k in ev.get("rewarded_agents", []):
                            comp[k, 1] += base.shelf_return_reward
                    for ev in info.get("pickup_events", []):
                        k = int(ev["agent_id"]) - 1
                        shelf = base.shelfs[int(ev["shelf_id"]) - 1]
                        if ev["requested"]:
                            comp[k, 2] += 0. if shelf.id in picked_before[k] else base.requested_shelf_pickup_reward
                        elif base._is_requested_shelf(shelf):
                            comp[k, 2] -= base.wrong_shelf_pickup_penalty
                        else:
                            comp[k, 2] -= base.unrequested_shelf_pickup_penalty
                    for k, a in enumerate(base.agents):
                        if carried[k] is not None and a.carrying_shelf is carried[k] and goal_d[k] is not None:
                            after = base._carried_requested_shelf_delivery_goal_distance(a)
                            if after is not None:
                                comp[k, 3] += base.goal_progress_reward * (goal_d[k] - after)
                    bv = np.stack([AC.public_view(x[i], R) for i in range(n)]).astype(np.uint8)
                    av = np.stack([AC.public_view(y[i], R) for i in range(n)]).astype(np.uint8)
                    rr = np.asarray(r, dtype=np.float64)
                    ctrl.observe_views(bv, acts, av, rr)
                    sl = slice(nrow, nrow + n)
                    for i in range(n):
                        probes.add(i, bv[i], acts[i], av[i])
                    cols["agent"][sl] = np.arange(n); cols["team"][sl] = teams; cols["ckpt"][sl] = ci_
                    cols["carrying"][sl] = bv[:, 0]; cols["heading"][sl] = np.argmax(bv[:, 1:5], 1)
                    cols["action"][sl] = acts; cols["highway"][sl] = bv[:, 5]
                    cols["landmark"][sl] = [landmark_key(bv[i], lm_table) for i in range(n)]
                    cols["reward"][sl] = rr; cols["delivery"][sl] = comp[:, 0]; cols["return_"][sl] = comp[:, 1]
                    cols["pickup"][sl] = comp[:, 2]; cols["goal_progress"][sl] = comp[:, 3]
                    cols["rest"][sl] = rr - comp.sum(1)
                    nrow += n
                    total += 1
                    if total % ROUND_STEPS == 0:
                        rnd = total // ROUND_STEPS
                        pos = np.array([(a.x, a.y) for a in base.agents])
                        cheb = np.abs(pos[:, None, :] - pos[None, :, :]).max(-1)
                        alloc, log = ctrl(rnd * ctrl.config.interval, cheb <= comm, active=total > WARMUP_STEPS)
                        row = dict(env_steps=total, ckpt=name, mature=[bool(m) for m in log["ac_mature"]],
                                   own_events=[p_.events_seen for p_ in ctrl.predictors],
                                   gate_same_arcs=int(((alloc > 0) & same).sum()),
                                   gate_cross_arcs=int(((alloc > 0) & ~same).sum()),
                                   contacts=int(np.triu(cheb <= comm, 1).sum()))
                        if rnd % EVAL_EVERY == 0:
                            sm = np.array([same[i, j] for i, j in pairs])
                            d2 = np.array([v2_distance(ctrl, i, j) for i, j in pairs])
                            cs = [contrast_scores(ctrl, probes, i, j) for i, j in pairs]
                            corr = np.array([x[0] for x in cs]); sign = np.array([x[1] for x in cs])
                            both = np.array([row["mature"][i] and row["mature"][j] for i, j in pairs])
                            row.update(v2_auroc=auroc(-d2, sm), v2_pairs=int(np.isfinite(d2).sum()),
                                       contrast_corr_auroc=auroc(corr, sm), contrast_pairs=int(np.isfinite(corr).sum()),
                                       contrast_sign_auroc=auroc(sign, sm), sign_pairs=int(np.isfinite(sign).sum()),
                                       v2_auroc_mature_pairs=auroc(np.where(both, -d2, np.nan), sm),
                                       v2_distance=[None if not np.isfinite(v) else round(float(v), 4) for v in d2],
                                       contrast_corr=[None if not np.isfinite(v) else round(float(v), 4) for v in corr],
                                       contrast_sign=[None if not np.isfinite(v) else round(float(v), 4) for v in sign])
                            print(f"{total:>8} {name} mature {sum(row['mature'])} gate s/c "
                                  f"{row['gate_same_arcs']}/{row['gate_cross_arcs']} v2 {row['v2_auroc']} "
                                  f"({row['v2_pairs']}) corr {row['contrast_corr_auroc']} ({row['contrast_pairs']}) "
                                  f"sign {row['contrast_sign_auroc']} ({row['sign_pairs']}) {time.time() - t0:.0f}s",
                                  flush=True)
                        curve.append(row)
                    if bool(np.any(term)) or bool(np.any(trunc)) or total >= end:
                        break
        print(f"done {name} at {total} steps, {episode} episodes, {time.time() - t0:.0f}s", flush=True)
    arr = {k: v[:nrow] for k, v in cols.items()}
    np.savez_compressed(OUT / f"{args.out}_audit.npz", ckpt_names=np.asarray([c[0] for c in ckpts]),
                        ckpt_ends=np.asarray([c[1] for c in ckpts]), teams=teams, **arr)
    (OUT / f"{args.out}_curve.json").write_text(json.dumps(dict(
        run=str(run), ckpts=ckpts, teams=teams.tolist(), pairs=pairs, unmasked_policy=args.unmasked_policy,
        landmarks=len(lm_table), curve=curve), indent=1), encoding="utf-8")
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
