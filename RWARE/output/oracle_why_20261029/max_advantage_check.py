"""Read-only follow-up checks for D91 (T149). No training.

(1) Query accuracy: a donor critic evaluated on the receiver's states with h=0 (one step) or with a short
    window (L=8) versus the full recurrent pass over the receiver's own stream.
(2) Advantage direction: on a laggard's rollout (0 deliveries in the episode, with a delivering teammate),
    does the GAE advantage computed with f = max(V_i, max_j V_j) favour the actions the leading teammate's
    policy would pick, more than the advantage from the laggard's own critic? The other team's leader is the
    control. Leaders' advantages should barely change.

Datasets: A+C IPPO s1 (coords masked) and the older corner 1M IPPO s1/s2 (coords in the input)."""
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
EPISODES, HORIZON, SEED, GAMMA, LAM, WINDOW, CAP = 4, 500, 9600001, 0.99, 0.95, 8, 0.5
OLD = ROOT / "runs/queue_20261022_140114_b6_corner_indiv_pen005"
DATASETS = [
    ("acs1", ROOT / "runs/ac_20261023_185556/b6_corner_indiv_pen005_ac_ippo_s1", True,
     ["checkpoint_01000.pt", "checkpoint_02000.pt"]),
    ("corner1M_s1", OLD / "b6_corner_indiv_pen005_ippo_s1", False, ["checkpoint_00750.pt", "final.pt"]),
    ("corner1M_s2", OLD / "b6_corner_indiv_pen005_ippo_s2", False, ["checkpoint_00750.pt", "final.pt"]),
]


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 1e-9 and b.std() > 1e-9 else float("nan")


def critic_recurrent(net, s):
    _, h = net.initial_hidden(1, torch.device("cpu")); out = []
    for k in range(len(s)):
        v, h = net._critic_step(s[k:k + 1], h); out.append(float(v.reshape(-1)[0]))
    return np.array(out)


def critic_window(net, s, length, full):
    """Value at t from a fresh hidden state run over s[t-length+1..t] (h=0 when length == 1).
    For t < length-1 that window starts at the episode start, so it equals the full recurrent value."""
    n = len(s)
    out = np.array(full, copy=True)
    ts = np.arange(length - 1, n)
    if not len(ts):
        return out
    _, h = net.initial_hidden(len(ts), torch.device("cpu"))
    for step in range(length):
        v, h = net._critic_step(s[ts - length + 1 + step], h)
    out[ts] = v.reshape(-1).numpy()
    return out


def actor_probs(net, s, masks):
    h, _ = net.initial_hidden(1, torch.device("cpu")); out = []
    for k in range(len(s)):
        logits, h = net._actor_step(s[k:k + 1], h)
        out.append(net._masked_logits(logits, torch.as_tensor(masks[k])).softmax(-1).numpy()[0])
    return np.array(out)


def gae(r, f):
    """GAE with a value-like function f over one episode; bootstraps f at the truncation step."""
    n = len(r); adv = np.zeros(n); last = 0.0
    for t in range(n - 1, -1, -1):
        nxt = f[t + 1] if t + 1 < n else f[t]
        delta = r[t] + GAMMA * nxt - f[t]
        last = delta + GAMMA * LAM * last; adv[t] = last
    return adv


def preference_gap(adv, acts, probs):
    """Normalised advantage of actions the policy `probs` ranks first minus the others."""
    z = (adv - adv.mean()) / (adv.std() + 1e-8)
    top = probs.argmax(1) == acts
    return float(z[top].mean() - z[~top].mean()) if top.any() and (~top).any() else float("nan")


@torch.inference_mode()
def run(path, masked):
    agents, _ = load_model(path)
    c = T._load_checkpoint(str(path), torch.device("cpu"))
    cfg, kw = c["config"], dict(c["effective_env_kwargs"])
    assert kw["team_reward_mode"] == "individual"
    rng = np.random.default_rng(0)
    out = dict(acc={k: [] for k in ("h0", f"L{WINDOW}", "h0_uplift", f"L{WINDOW}_uplift", "h0_adv")},
               lag={k: [] for k in ("own", "same", "same_cap", "same_h0", "all")},
               lag_ctrl={k: [] for k in ("own", "same", "all")}, lead_keep=[], n_lag=0, n_lead=0)
    with T.make_env(cfg["env_id"], kw, cfg["observation_format"]) as env:
        base = env.unwrapped
        for ep in range(EPISODES):
            obs, _ = env.reset(seed=SEED + ep)
            n = len(base.agents)
            teams = np.asarray(base.agent_team_ids)
            hidden = [net.initial_hidden(1, torch.device("cpu"))[0] for net in agents]
            raw = np.asarray(T.flatten_multi_agent_obs(env, obs), dtype=np.float32)
            x = AC.strip_coordinates(raw) if masked else raw
            seq = [[] for _ in range(n)]; rew = [[] for _ in range(n)]; act = [[] for _ in range(n)]
            msk = [[] for _ in range(n)]; deliv = np.zeros(n)
            for t in range(HORIZON):
                masks = np.asarray(T.local_rware_action_masks(env, n, 5, enabled=True), dtype=np.float32)
                acts = []
                for i, net in enumerate(agents):
                    seq[i].append(x[i].copy()); msk[i].append(masks[i])
                    logits, hidden[i] = net._actor_step(torch.as_tensor(x[i]).unsqueeze(0), hidden[i])
                    p = net._masked_logits(logits, torch.as_tensor(masks[i])).softmax(-1).numpy()[0].astype(np.float64)
                    acts.append(int(rng.choice(5, p=p / p.sum())))
                obs, r, term, trunc, info = env.step(acts)
                for ev in info.get("delivery_events", []):
                    deliv[int(ev["carrier_id"]) - 1] += 1
                for i in range(n):
                    rew[i].append(float(r[i])); act[i].append(acts[i])
                raw = np.asarray(T.flatten_multi_agent_obs(env, obs), dtype=np.float32)
                x = AC.strip_coordinates(raw) if masked else raw
                if bool(np.any(term)) or bool(np.any(trunc)):
                    break
            for i in range(n):
                s = torch.as_tensor(np.stack(seq[i])); r_i = np.array(rew[i]); a_i = np.array(act[i])
                mates = [j for j in range(n) if j != i and teams[j] == teams[i]]
                others = [j for j in range(n) if teams[j] != teams[i]]
                rec = {k: critic_recurrent(agents[k], s) for k in range(n)}
                h0 = {k: critic_window(agents[k], s, 1, rec[k]) for k in mates + others}
                lw = {k: critic_window(agents[k], s, WINDOW, rec[k]) for k in mates + others}
                v_i = rec[i]
                for k in mates + others:
                    out["acc"]["h0"].append(corr(h0[k], rec[k]))
                    out["acc"][f"L{WINDOW}"].append(corr(lw[k], rec[k]))
                up_rec = np.maximum(np.max([rec[k] for k in mates], 0) - v_i, 0)
                up_h0 = np.maximum(np.max([h0[k] for k in mates], 0) - v_i, 0)
                up_lw = np.maximum(np.max([lw[k] for k in mates], 0) - v_i, 0)
                out["acc"]["h0_uplift"].append(corr(up_h0, up_rec))
                out["acc"][f"L{WINDOW}_uplift"].append(corr(up_lw, up_rec))
                f_same = v_i + up_rec
                f_cap = v_i + np.minimum(up_rec, CAP)
                f_h0 = v_i + up_h0
                f_all = np.maximum(v_i, np.max([rec[k] for k in mates + others], 0))
                adv = dict(own=gae(r_i, v_i), same=gae(r_i, f_same), same_cap=gae(r_i, f_cap),
                           same_h0=gae(r_i, f_h0), all=gae(r_i, f_all))
                out["acc"]["h0_adv"].append(corr(adv["same_h0"], adv["same"]))
                lead_mates = [j for j in mates if deliv[j] > 0]
                lead_others = [j for j in others if deliv[j] > 0]
                masks_i = np.stack(msk[i])
                if deliv[i] == 0 and lead_mates:
                    out["n_lag"] += 1
                    leader = max(lead_mates, key=lambda j: deliv[j])
                    p_lead = actor_probs(agents[leader], s, masks_i)
                    for k in out["lag"]:
                        out["lag"][k].append(preference_gap(adv[k], a_i, p_lead))
                    if lead_others:
                        ctrl = max(lead_others, key=lambda j: deliv[j])
                        p_ctrl = actor_probs(agents[ctrl], s, masks_i)
                        for k in out["lag_ctrl"]:
                            out["lag_ctrl"][k].append(preference_gap(adv[k], a_i, p_ctrl))
                elif deliv[i] > 0:
                    out["n_lead"] += 1
                    out["lead_keep"].append(corr(adv["same"], adv["own"]))
    return out


def report(tag, o):
    m = lambda xs: f"{np.nanmean(xs):.2f}" if len(xs) else "  - "
    a = o["acc"]
    print(f"\n[{tag}] laggard receivers {o['n_lag']}, leader receivers {o['n_lead']}")
    print(f"[{tag}] donor value vs full recurrent: corr h0 {m(a['h0'])}  L{WINDOW} {m(a[f'L{WINDOW}'])} | "
          f"same-team uplift corr h0 {m(a['h0_uplift'])}  L{WINDOW} {m(a[f'L{WINDOW}_uplift'])} | "
          f"advantage corr h0 vs recurrent {m(a['h0_adv'])}")
    g = o["lag"]
    print(f"[{tag}] laggard: z(A) gap for teammate-leader's top action: own {m(g['own'])}  max-same {m(g['same'])}  "
          f"max-same-cap {m(g['same_cap'])}  max-same-h0 {m(g['same_h0'])}  max-all {m(g['all'])}")
    cg = o["lag_ctrl"]
    print(f"[{tag}] control (other-team leader's top action): own {m(cg['own'])}  max-same {m(cg['same'])}  "
          f"max-all {m(cg['all'])}")
    print(f"[{tag}] leaders: corr(A_max-same, A_own) {m(o['lead_keep'])}")


for name, run_dir, masked, cks in DATASETS:
    for ck in cks:
        report(f"{name} {ck}", run(run_dir / ck, masked))
