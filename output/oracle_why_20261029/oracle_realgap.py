"""Read-only: does probe alignment carry over to the states agents actually visit?

Roll out each mode's own policies (coords masked, stochastic), then for every receiver i run all six
critics recurrently over i's real observation sequence and compare teammates' values there."""
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
EPISODES, HORIZON, SEED = 3, 500, 9400001


@torch.inference_mode()
def run(mode, ck):
    path = RUN / f"b6_corner_indiv_pen005_ac_{mode}_s1" / ck
    agents, _ = load_model(path)
    c = T._load_checkpoint(str(path), torch.device("cpu"))
    cfg, kw = c["config"], dict(c["effective_env_kwargs"])
    rng = np.random.default_rng(0)
    same_gap, cross_gap, own_err, mate_err = [], [], [], []
    with T.make_env(cfg["env_id"], kw, cfg["observation_format"]) as env:
        base = env.unwrapped
        for ep in range(EPISODES):
            obs, _ = env.reset(seed=SEED + ep)
            n = len(base.agents)
            teams = np.asarray(base.agent_team_ids)
            hidden = [net.initial_hidden(1, torch.device("cpu"))[0] for net in agents]
            x = AC.strip_coordinates(np.asarray(T.flatten_multi_agent_obs(env, obs), dtype=np.float32))
            seq = [[] for _ in range(n)]; rew = [[] for _ in range(n)]
            for t in range(HORIZON):
                masks = np.asarray(T.local_rware_action_masks(env, n, 5, enabled=True), dtype=np.float32)
                acts = []
                for i, net in enumerate(agents):
                    seq[i].append(x[i].copy())
                    logits, hidden[i] = net._actor_step(torch.as_tensor(x[i]).unsqueeze(0), hidden[i])
                    p = net._masked_logits(logits, torch.as_tensor(masks[i])).softmax(-1).numpy()[0].astype(np.float64)
                    acts.append(int(rng.choice(5, p=p / p.sum())))
                obs, r, term, trunc, _ = env.step(acts)
                for i in range(n):
                    rew[i].append(float(r[i]))
                x = AC.strip_coordinates(np.asarray(T.flatten_multi_agent_obs(env, obs), dtype=np.float32))
                if bool(np.any(term)) or bool(np.any(trunc)):
                    break
            for i in range(n):
                s = torch.as_tensor(np.stack(seq[i]))
                vals = []
                for net in agents:  # every critic reads receiver i's real stream with its own recurrent state
                    _, ch = net.initial_hidden(1, torch.device("cpu"))
                    out = []
                    for k in range(len(s)):
                        v, ch = net._critic_step(s[k:k + 1], ch)
                        out.append(float(v.reshape(-1)[0]))
                    vals.append(np.array(out))
                g = np.zeros(len(s)); acc = 0.0  # discounted return-to-go of agent i (gamma .99)
                for k in range(len(s) - 1, -1, -1):
                    acc = rew[i][k] + 0.99 * acc; g[k] = acc
                own_err.append(np.mean((vals[i] - g) ** 2))
                mates = [j for j in range(n) if j != i and teams[j] == teams[i]]
                others = [j for j in range(n) if teams[j] != teams[i]]
                same_gap += [np.mean((vals[i] - vals[j]) ** 2) for j in mates]
                cross_gap += [np.mean((vals[i] - vals[j]) ** 2) for j in others]
                mate_err += [np.mean((vals[j] - g) ** 2) for j in mates]
    print(f"{mode:7s} {ck}  real-state same-team gap {np.mean(same_gap):.4f}  cross {np.mean(cross_gap):.4f}  "
          f"MSE to own return: own critic {np.mean(own_err):.4f}  teammate critic {np.mean(mate_err):.4f}")


for ck in ["checkpoint_01000.pt", "checkpoint_02000.pt", "checkpoint_04750.pt"]:
    for mode in ["ippo", "oracle"]:
        run(mode, ck)
