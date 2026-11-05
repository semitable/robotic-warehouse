"""Read-only premise checks for the transfer redesign (T148). No training.

On each receiver's real states (IPPO s1 policies, coords masked) every critic is run recurrently.
(1) Team specificity: corr(V_k, G_i) for own / same-team / cross-team critics, split by whether the
    receiver's own goal is in view. Out of view, the goal direction is team-specific knowledge.
(2) Max aggregation: uplift max_j V_j - V_i from same-team vs cross-team critics, by receiver competence,
    and the optimism max(V_i, V_j) - G_i compared with the own critic's V_i - G_i."""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(r"C:\RWARE\MARL_RWARE\third_party\robotic-warehouse")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "latent_team"))
from examples import train_recurrent_ippo_consensus as T  # noqa: E402
from reward_correlation_baseline import load_model  # noqa: E402
import reward_probe_ac as AC  # noqa: E402

torch.set_num_threads(4)
RUN = ROOT / "runs/ac_20261023_185556"
EPISODES, HORIZON, SEED, GAMMA = 3, 500, 9500001, 0.99


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 1e-8 and b.std() > 1e-8 else float("nan")


@torch.inference_mode()
def run(mode, ck):
    path = RUN / f"b6_corner_indiv_pen005_ac_{mode}_s1" / ck
    agents, _ = load_model(path)
    c = T._load_checkpoint(str(path), torch.device("cpu"))
    cfg, kw = c["config"], dict(c["effective_env_kwargs"])
    rng = np.random.default_rng(0)
    recs = []  # per receiver-episode
    with T.make_env(cfg["env_id"], kw, cfg["observation_format"]) as env:
        base = env.unwrapped
        for ep in range(EPISODES):
            obs, _ = env.reset(seed=SEED + ep)
            n = len(base.agents)
            teams = np.asarray(base.agent_team_ids)
            hidden = [net.initial_hidden(1, torch.device("cpu"))[0] for net in agents]
            x = AC.strip_coordinates(np.asarray(T.flatten_multi_agent_obs(env, obs), dtype=np.float32))
            seq = [[] for _ in range(n)]; rew = [[] for _ in range(n)]; deliv = np.zeros(n)
            for t in range(HORIZON):
                masks = np.asarray(T.local_rware_action_masks(env, n, 5, enabled=True), dtype=np.float32)
                acts = []
                for i, net in enumerate(agents):
                    seq[i].append(x[i].copy())
                    logits, hidden[i] = net._actor_step(torch.as_tensor(x[i]).unsqueeze(0), hidden[i])
                    p = net._masked_logits(logits, torch.as_tensor(masks[i])).softmax(-1).numpy()[0].astype(np.float64)
                    acts.append(int(rng.choice(5, p=p / p.sum())))
                obs, r, term, trunc, info = env.step(acts)
                for ev in info.get("delivery_events", []):
                    deliv[int(ev["carrier_id"]) - 1] += 1
                for i in range(n):
                    rew[i].append(float(r[i]))
                x = AC.strip_coordinates(np.asarray(T.flatten_multi_agent_obs(env, obs), dtype=np.float32))
                if bool(np.any(term)) or bool(np.any(trunc)):
                    break
            for i in range(n):
                s = np.stack(seq[i]); st = torch.as_tensor(s)
                vals = []
                for net in agents:
                    _, ch = net.initial_hidden(1, torch.device("cpu")); out = []
                    for k in range(len(st)):
                        v, ch = net._critic_step(st[k:k + 1], ch); out.append(float(v.reshape(-1)[0]))
                    vals.append(np.array(out))
                g = np.zeros(len(s)); acc = 0.0
                for k in range(len(s) - 1, -1, -1):
                    acc = rew[i][k] + GAMMA * acc; g[k] = acc
                grid = s[:, 20:].reshape(len(s), 13, -1)
                visible = grid[:, 12].max(1) > 0.5
                keep = np.arange(len(s)) < len(s) - 100  # drop the truncated tail of the return
                recs.append(dict(i=i, team=teams, vals=np.stack(vals), g=g, visible=visible, carry=s[:, 2] > .5,
                                 keep=keep, deliv=deliv[i]))
    return recs


def summarize(tag, recs):
    rows = {k: [] for k in ("own", "same", "cross")}
    split = {(k, vis): [] for k in ("own", "same", "cross") for vis in (True, False)}
    upl = {"lag": {"same": [], "cross": []}, "lead": {"same": [], "cross": []}}
    opt = {"own": [], "max_same": [], "max_cross": []}
    for r in recs:
        i, teams, V, g, keep = r["i"], r["team"], r["vals"], r["g"], r["keep"]
        mates = [j for j in range(6) if j != i and teams[j] == teams[i]]
        others = [j for j in range(6) if teams[j] != teams[i]]
        groups = {"own": [i], "same": mates, "cross": others}
        for k, js in groups.items():
            for j in js:
                rows[k].append(corr(V[j][keep], g[keep]))
                for vis in (True, False):
                    m = keep & (r["visible"] == vis) & r["carry"]
                    split[(k, vis)].append(corr(V[j][m], g[m]))
        group = "lag" if r["deliv"] == 0 else "lead"
        for k, js in (("same", mates), ("cross", others)):
            u = V[js].max(0) - V[i]
            upl[group][k].append(np.mean(np.maximum(u, 0)[keep]))
        opt["own"].append(np.mean((V[i] - g)[keep]))
        opt["max_same"].append(np.mean((np.maximum(V[i], V[mates].max(0)) - g)[keep]))
        opt["max_cross"].append(np.mean((np.maximum(V[i], V[others].max(0)) - g)[keep]))
    f = lambda xs: f"{np.nanmean(xs):.2f}"
    print(f"\n[{tag}] corr(V_k, G_i) all states: own {f(rows['own'])} same {f(rows['same'])} cross {f(rows['cross'])}")
    print(f"[{tag}] carrying, own goal IN view : own {f(split[('own', True)])} same {f(split[('same', True)])} "
          f"cross {f(split[('cross', True)])}")
    print(f"[{tag}] carrying, own goal OUT of view: own {f(split[('own', False)])} same {f(split[('same', False)])} "
          f"cross {f(split[('cross', False)])}")
    for grp in ("lag", "lead"):
        n = len(upl[grp]["same"])
        if n:
            print(f"[{tag}] mean relu uplift for {grp} receivers (n={n}): same-team {np.mean(upl[grp]['same']):.3f}  "
                  f"cross-team {np.mean(upl[grp]['cross']):.3f}")
    print(f"[{tag}] mean bias vs own return: own V {np.mean(opt['own']):+.3f}  max(own,same) {np.mean(opt['max_same']):+.3f}  "
          f"max(own,cross) {np.mean(opt['max_cross']):+.3f}")


for ck in ["checkpoint_01000.pt", "checkpoint_02000.pt", "checkpoint_03000.pt"]:
    summarize(f"ippo {ck}", run("ippo", ck))
