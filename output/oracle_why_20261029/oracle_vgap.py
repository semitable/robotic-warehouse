"""Read-only: did Oracle distillation actually pull teammates' critics together on the probes?

For IPPO s1 and Oracle s1 checkpoints, evaluate every critic on the canonical probes (coords masked as in
training) and report mean squared value gap for same-team and cross-team pairs, plus value scale."""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(r"C:\RWARE\MARL_RWARE\third_party\robotic-warehouse")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "latent_team"))
from examples import train_recurrent_ippo_consensus as T  # noqa: E402
from reward_correlation_baseline import load_model  # noqa: E402

torch.set_num_threads(4)
RUN = ROOT / "runs/ac_20261023_185556"
bank = torch.load(RUN / "b6_corner_indiv_pen005_ac_oracle_s1/canonical_probes.pt", map_location="cpu",
                  weights_only=False)["sequences"].clone()
bank[..., :2] = 0  # A+C masks self_x/self_y in probes too
team = np.arange(6) % 2
CK = ["checkpoint_00250.pt", "checkpoint_00500.pt", "checkpoint_01000.pt", "checkpoint_02000.pt",
      "checkpoint_03000.pt", "checkpoint_04750.pt"]

with torch.inference_mode():
    for mode in ["ippo", "oracle", "all"]:
        for ck in CK:
            path = RUN / f"b6_corner_indiv_pen005_ac_{mode}_s1" / ck
            if not path.exists():
                continue
            agents, _ = load_model(path)
            v = T.recurrent_critic_probe_targets(agents, bank, torch.device("cpu")).squeeze(-1).numpy()
            v = v.reshape(6, -1)
            same, cross = [], []
            for i in range(6):
                for j in range(i + 1, 6):
                    gap = float(np.mean((v[i] - v[j]) ** 2))
                    (same if team[i] == team[j] else cross).append(gap)
            print(f"{mode:7s} {ck}  same-team gap {np.mean(same):.4f}  cross-team gap {np.mean(cross):.4f}  "
                  f"|V| mean {np.abs(v).mean():.3f}  V spread across probes {v.std(1).mean():.3f}")
