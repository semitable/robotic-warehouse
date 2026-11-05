"""Read-only: would a JS-similarity weight track maturity? Policy JS on canonical probes over training."""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(r"C:\RWARE\MARL_RWARE\third_party\robotic-warehouse")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "latent_team"))
from reward_correlation_baseline import load_model  # noqa: E402

torch.set_num_threads(4)
RUN = ROOT / "runs/ac_20261023_185556"
bank = torch.load(RUN / "b6_corner_indiv_pen005_ac_oracle_s1/canonical_probes.pt", map_location="cpu",
                  weights_only=False)["sequences"].clone()
bank[..., :2] = 0
probes = bank.reshape(-1, bank.shape[-1])
team = np.arange(6) % 2


def js(p, q):
    m = (p + q) / 2
    kl = lambda a, b: np.sum(a * (np.log(a + 1e-12) - np.log(b + 1e-12)), -1)
    return float(np.mean(0.5 * kl(p, m) + 0.5 * kl(q, m)))


with torch.inference_mode():
    for mode in ["ippo", "oracle"]:
        for ck in ["checkpoint_00250.pt", "checkpoint_00500.pt", "checkpoint_01000.pt", "checkpoint_02000.pt",
                   "checkpoint_04750.pt"]:
            agents, _ = load_model(RUN / f"b6_corner_indiv_pen005_ac_{mode}_s1" / ck)
            P = []
            for net in agents:
                h, _ = net.initial_hidden(len(probes), torch.device("cpu"))
                logits, _ = net._actor_step(probes, h)
                P.append(logits.softmax(-1).numpy())
            same = [js(P[i], P[j]) for i in range(6) for j in range(i + 1, 6) if team[i] == team[j]]
            cross = [js(P[i], P[j]) for i in range(6) for j in range(i + 1, 6) if team[i] != team[j]]
            info = [float(np.mean(np.sum(p * np.log(p * 5 + 1e-12), -1))) for p in P]  # KL(pi || uniform)
            print(f"{mode:7s} {ck}  JS same {np.mean(same):.3f}  cross {np.mean(cross):.3f}  "
                  f"KL(pi||U) per agent {np.round(info, 2)}")
