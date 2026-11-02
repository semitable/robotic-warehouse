"""Read-only counterfactual checks for D91 (T149). No training.

(1) State level: at a saved env state s on receiver i's own trajectory, keep i's seat and reward, and run
    i with its own policy or with donor j's policy (j's actor warmed up on i's observation history); every
    other agent keeps its own policy. The Monte Carlo difference dMC = G(pi_j) - G(pi_i) is the real gain.
    Compare it with dV = V_j(h_i) - V_i(h_i), the critic signal max-aggregation would use.
(2) Action level: for each valid action a at s, one-step A_self = r + g V_i(s') - V_i(s) and
    A_f = r + g f(s') - f(s), f = max(V_i, max_j V_j), against MC Q(s, a) with i continuing by its own policy
    (and by the best teammate policy). Common random numbers across arms.
(3) Gate: share of a mature laggard's steps with a mature teammate inside communication range (Chebyshev 3),
    and the uplift available from those donors only.
(4) h=0 queries: how often the donor-vs-self selection flips against the recurrent value.
Maturity per agent comes from the E60 V2 stream replay (output/ac_early_id_20261026/*_curve.json)."""
import copy
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(r"C:\RWARE\MARL_RWARE\third_party\robotic-warehouse")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "latent_team"))
from examples import train_recurrent_ippo_consensus as T  # noqa: E402
from reward_correlation_baseline import load_model  # noqa: E402
import reward_probe_ac as AC  # noqa: E402

torch.set_num_threads(2)
GAMMA, EPISODES, TAUS, H_STATE, H_ACTION, N_MC, COMM, MARGIN = 0.99, 3, (80, 180, 280), 100, 60, 8, 3, 0.05
OLD = ROOT / "runs/queue_20261022_140114_b6_corner_indiv_pen005"
DATASETS = {
    "acs1_1M": (ROOT / "runs/ac_20261023_185556/b6_corner_indiv_pen005_ac_ippo_s1/checkpoint_01000.pt", True, "acs1", 1024000),
    "acs1_2M": (ROOT / "runs/ac_20261023_185556/b6_corner_indiv_pen005_ac_ippo_s1/checkpoint_02000.pt", True, "acs1", 1024000),
    "corner1M_s1": (OLD / "b6_corner_indiv_pen005_ippo_s1/final.pt", False, "corner1M_s1", 998400),
    "corner1M_s2": (OLD / "b6_corner_indiv_pen005_ippo_s2/final.pt", False, "corner1M_s2", 998400),
}


def maturity(curve_name, step):
    d = json.load(open(ROOT / f"output/ac_early_id_20261026/{curve_name}_curve.json", encoding="utf-8"))
    row = min(d["curve"], key=lambda r: abs(r["env_steps"] - step))
    return np.asarray(row["mature"], dtype=bool)


class Runner:
    def __init__(self, path, masked):
        self.agents, _ = load_model(path)
        c = T._load_checkpoint(str(path), torch.device("cpu"))
        self.cfg, self.kw = c["config"], dict(c["effective_env_kwargs"])
        assert self.kw["team_reward_mode"] == "individual"
        self.masked = masked
        self.n = len(self.agents)

    def view(self, env, obs):
        raw = np.asarray(T.flatten_multi_agent_obs(env, obs), dtype=np.float32)
        return AC.strip_coordinates(raw) if self.masked else raw

    def masks(self, env):
        return np.asarray(T.local_rware_action_masks(env, self.n, 5, enabled=True), dtype=np.float32)

    def probs(self, k, x, h, m):
        logits, h2 = self.agents[k]._actor_step(torch.as_tensor(x), h)
        return self.agents[k]._masked_logits(logits, torch.as_tensor(m)).softmax(-1).numpy().astype(np.float64), h2

    @torch.inference_mode()
    def continue_arm(self, snap, i, policy, horizon, seeds, first_action=None):
        """Discounted return of agent i over `horizon` steps from snapshot, i acting with `policy`."""
        N = len(seeds)
        envs = []
        for s in seeds:
            e = copy.deepcopy(snap["env"]); e.unwrapped._np_random = np.random.default_rng(int(s)); envs.append(e)
        rngs = [np.random.default_rng(int(s) + 7) for s in seeds]
        x = np.repeat(snap["x"][None], N, 0)  # (N, n, D)
        hid = [snap["actor_h"][k][k].repeat(N, 1) for k in range(self.n)]
        hid[i] = snap["actor_h"][policy][i].repeat(N, 1)
        actors = list(range(self.n)); actors[i] = policy
        G = np.zeros(N); alive = np.ones(N, bool); disc = 1.0
        first_obs, first_r = None, None
        for step in range(horizon):
            m = np.stack([self.masks(e) for e in envs])  # (N, n, 5)
            acts = np.zeros((N, self.n), dtype=int)
            for k in range(self.n):
                p, hid[k] = self.probs(actors[k], x[:, k], hid[k], m[:, k])
                u = np.array([r.random() for r in rngs])
                acts[:, k] = np.minimum((np.cumsum(p / p.sum(1, keepdims=True), 1) < u[:, None]).sum(1), 4)
            if step == 0 and first_action is not None:
                acts[:, i] = first_action
            for b, e in enumerate(envs):
                if not alive[b]:
                    continue
                obs, r, term, trunc, _ = e.step([int(a) for a in acts[b]])
                G[b] += disc * float(r[i]); x[b] = self.view(e, obs)
                if step == 0 and b == 0:
                    first_r = float(r[i])
                if bool(np.any(term)) or bool(np.any(trunc)):
                    alive[b] = False
            if step == 0:
                first_obs = x[:, i].copy()
            disc *= GAMMA
            if not alive.any():
                break
        return G, (first_obs, first_r)

    @torch.inference_mode()
    def critic_values_next(self, snap, i, k, obs_next):
        """V_k at i's next observation, advancing k's critic hidden over i's stream by obs_t then obs_next."""
        h = snap["critic_h"][k][i:i + 1]
        _, h = self.agents[k]._critic_step(torch.as_tensor(snap["x"][i:i + 1]), h)
        v, _ = self.agents[k]._critic_step(torch.as_tensor(obs_next[None]), h)
        return float(v.reshape(-1)[0])

    @torch.inference_mode()
    def episode(self, seed, rng):
        n = self.n
        with T.make_env(self.cfg["env_id"], self.kw, self.cfg["observation_format"]) as env:
            obs, _ = env.reset(seed=seed)
            teams = np.asarray(env.unwrapped.agent_team_ids)
            x = self.view(env, obs)
            zero = self.agents[0].initial_hidden(n, torch.device("cpu"))
            actor_h = [zero[0].clone() for _ in range(n)]   # actor_h[j][i]: actor j over receiver i's stream
            critic_h = [zero[1].clone() for _ in range(n)]  # critic_h[k][i]
            V, V0, pos, deliv, snaps = [], [], [], np.zeros(n), {}
            for t in range(500):
                if t in TAUS:
                    snaps[t] = dict(env=copy.deepcopy(env), x=x.copy(),
                                    actor_h=[h.clone() for h in actor_h], critic_h=[h.clone() for h in critic_h])
                xt = torch.as_tensor(x)
                vt, v0t = np.zeros((n, n)), np.zeros((n, n))
                for k in range(n):
                    v, critic_h[k] = self.agents[k]._critic_step(xt, critic_h[k]); vt[k] = v.reshape(-1).numpy()
                    v0, _ = self.agents[k]._critic_step(xt, zero[1]); v0t[k] = v0.reshape(-1).numpy()
                V.append(vt); V0.append(v0t)
                pos.append([(a.x, a.y) for a in env.unwrapped.agents])
                m = self.masks(env)
                acts = []
                for j in range(n):
                    logits, actor_h[j] = self.agents[j]._actor_step(xt, actor_h[j])
                    p = self.agents[j]._masked_logits(logits[j:j + 1], torch.as_tensor(m[j])).softmax(-1).numpy()[0]
                    p = p.astype(np.float64)
                    acts.append(int(rng.choice(5, p=p / p.sum())))
                obs, r, term, trunc, info = env.step(acts)
                for ev in info.get("delivery_events", []):
                    deliv[int(ev["carrier_id"]) - 1] += 1
                x = self.view(env, obs)
                if bool(np.any(term)) or bool(np.any(trunc)):
                    break
        return dict(teams=teams, V=np.array(V), V0=np.array(V0), pos=np.array(pos), deliv=deliv, snaps=snaps)


def main(name):
    path, masked, curve, mstep = DATASETS[name]
    mature = maturity(curve, mstep)
    R = Runner(path, masked)
    rng = np.random.default_rng(11)
    state_rows, action_rows, gate_rows = [], [], []
    for ep in range(EPISODES):
        E = R.episode(9700001 + ep, rng)
        teams, V, V0, pos, deliv = E["teams"], E["V"], E["V0"], E["pos"], E["deliv"]
        for i in range(R.n):
            if not mature[i]:
                continue
            lag = deliv[i] == 0
            mates = [j for j in range(R.n) if j != i and teams[j] == teams[i] and mature[j]]
            others = [j for j in range(R.n) if teams[j] != teams[i] and mature[j]]
            if not mates:
                continue
            # (3) gate availability and (4) h=0 selection flips over every step of the episode
            dist = np.abs(pos[:, mates] - pos[:, i:i + 1]).max(-1)  # (T, mates)
            in_range = dist <= COMM
            vi = V[:, i, i]; vm = V[:, mates, i]; v0m = V0[:, mates, i]
            up_team = np.maximum(vm.max(1) - vi, 0)
            up_gate = np.where(in_range.any(1), np.maximum(np.where(in_range, vm, -np.inf).max(1) - vi, 0), 0.0)
            flip = ((vm.max(1) > vi + MARGIN) != (v0m.max(1) > vi + MARGIN))
            gate_rows.append(dict(lag=bool(lag), avail=float(in_range.any(1).mean()),
                                  higher_gate=float((up_gate > MARGIN).mean()), higher_team=float((up_team > MARGIN).mean()),
                                  up_gate=float(up_gate.mean()), up_team=float(up_team.mean()), flip=float(flip.mean())))
            for tau, snap in E["snaps"].items():
                if tau >= len(V):
                    continue
                seeds = [int(1e6 * (ep + 1) + 1000 * tau + 10 * i + s) for s in range(N_MC)]
                g_self, _ = R.continue_arm(snap, i, i, H_STATE, seeds)
                arms = [(j, "same") for j in mates] + ([(max(others, key=lambda o: deliv[o]), "cross")] if others else [])
                for j, kind in arms:
                    g_j, _ = R.continue_arm(snap, i, j, H_STATE, seeds)
                    d = g_j - g_self
                    state_rows.append(dict(lag=bool(lag), kind=kind, tau=tau, dV=float(V[tau, j, i] - V[tau, i, i]),
                                           dMC=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))),
                                           in_range=bool(np.abs(pos[tau, j] - pos[tau, i]).max() <= COMM),
                                           g_self=float(g_self.mean())))
                # (2) action level, laggards only
                if not lag:
                    continue
                valid = np.flatnonzero(R.masks(snap["env"])[i] > 0)
                if len(valid) < 2:
                    continue
                lead = max(mates, key=lambda j: deliv[j])
                f_s = max(V[tau, i, i], max(V[tau, j, i] for j in mates))
                rec = dict(tau=tau, a_self=[], a_f=[], q_self=[], q_lead=[])
                for a in valid:
                    q_self, (nxt, r0) = R.continue_arm(snap, i, i, H_ACTION, seeds, first_action=int(a))
                    q_lead, _ = R.continue_arm(snap, i, lead, H_ACTION, seeds, first_action=int(a))
                    v_next = {k: R.critic_values_next(snap, i, k, nxt[0]) for k in [i] + mates}
                    f_next = max(v_next[i], max(v_next[j] for j in mates))
                    # one-step advantages on the same sampled transition (clone 0, common random numbers)
                    rec["a_self"].append(r0 + GAMMA * v_next[i] - V[tau, i, i])
                    rec["a_f"].append(r0 + GAMMA * f_next - f_s)
                    rec["q_self"].append(float(q_self.mean())); rec["q_lead"].append(float(q_lead.mean()))
                action_rows.append({k: (np.asarray(v).tolist() if isinstance(v, list) else v) for k, v in rec.items()})
        print(f"[{name}] episode {ep} done: deliveries {deliv.tolist()} mature {mature.tolist()}", flush=True)
    out = ROOT / f"output/oracle_why_20261029/cf_{name}.json"
    out.write_text(json.dumps(dict(name=name, mature=mature.tolist(), state=state_rows, action=action_rows,
                                   gate=gate_rows), indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
