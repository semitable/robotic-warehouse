"""Visualise the canonical probe bank of the running from-scratch run (read-only).

1. Rebuild the bank exactly as latent_team/train_from_scratch.build_bank did (env reset(seed), rng seed+9173)
   and check it equals the saved canonical_bank_e1_s1.pt.
2. Save the repository's own probe-map renderings (rware.utils.probe_maps) into official/.
3. Decode every saved probe vector (ego features + 13 semantic channels) into decoded_bank.png and probes.csv.
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from latent_team import common as C  # noqa: E402,F401
from examples import train_recurrent_ippo_consensus as T  # noqa: E402
from rware.utils.probe_maps import RwareObjectiveProbeMapBuilder  # noqa: E402
from rware.utils.semantic_observation import SEMANTIC_EGO_FEATURES, SEMANTIC_SPATIAL_CHANNELS  # noqa: E402

RUN = ROOT / "runs/scratch_20261015_093349_stage2"
OUT = Path(__file__).resolve().parent
job = next(j for j in json.loads((RUN / "experiment.json").read_text(encoding="utf-8"))["jobs"] if j["mode"] == "pgct")
cfg = T.TrainConfig(**job["config"])
saved = torch.load(RUN / "canonical_bank_e1_s1.pt", weights_only=False)["sequences"][:, 0].numpy()

with T.make_env(cfg.env_id, json.loads(cfg.env_kwargs_json), cfg.observation_format) as env:
    env.reset(seed=cfg.seed)
    obs_dim = int(env.observation_space[0].shape[-1])
    rebuilt = RwareObjectiveProbeMapBuilder(env, obs_dim, np.random.default_rng(cfg.seed + 9173)).build(48)
    same = bool(np.array_equal(np.asarray(rebuilt).reshape(48, -1), saved))
    print("rebuilt bank identical to saved bank:", same)
    official = RwareObjectiveProbeMapBuilder(env, obs_dim, np.random.default_rng(cfg.seed + 9173)).save_probe_map_images(
        OUT / "official", batch_size=48, save_individual=True, save_contact_sheet=True)
    spec = T.get_semantic_observation_spec(env)
(OUT / "official_summary.json").write_text(json.dumps(official, indent=2, default=str), encoding="utf-8")

ego, grids = saved[:, :spec.ego_dim], saved[:, spec.ego_dim:].reshape(48, spec.spatial_channels, spec.spatial_size, spec.spatial_size)
_, inverse, counts = np.unique(saved, axis=0, return_inverse=True, return_counts=True)
ch = {n: k for k, n in enumerate(SEMANTIC_SPATIAL_CHANNELS)}
e = {n: k for k, n in enumerate(SEMANTIC_EGO_FEATURES)}
dirs = ["up", "down", "left", "right"]
rows = []
for p in range(48):
    g, v = grids[p], ego[p]
    rows.append(dict(probe=p, unique_id=int(inverse[p]), copies_in_bank=int(counts[inverse[p]]),
                     direction=dirs[int(np.argmax(v[e["self_dir_up"]:e["self_dir_up"] + 4]))],
                     carrying=int(v[e["self_carrying_shelf"]]), carrying_requested=int(v[e["self_carrying_requested_shelf"]]),
                     return_phase=int(v[e["self_return_phase"]]), on_highway=int(v[e["self_on_highway"]]),
                     x_norm=round(float(v[e["self_x"]]), 3), y_norm=round(float(v[e["self_y"]]), 3),
                     requested_cells=int(g[ch["requested_shelf"]].sum()), shelf_cells=int(g[ch["shelf"]].sum()),
                     relevant_goal_cells=int(g[ch["relevant_goal"]].sum()), other_agents=int(g[ch["agent"]].sum()) - 1))
with (OUT / "probes.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

fig, axes = plt.subplots(6, 8, figsize=(20, 16), dpi=110)
r = spec.spatial_size // 2
for p, ax in enumerate(axes.flat):
    g = grids[p]
    img = np.ones((spec.spatial_size, spec.spatial_size, 3)) * .95
    img[g[ch["highway"]] > 0] = (.98, .98, .98)
    img[g[ch["storage_cell"]] > 0] = (.86, .86, .86)
    img[g[ch["shelf"]] > 0] = (.55, .55, .55)
    img[g[ch["requested_shelf"]] > 0] = (1.0, .65, 0.0)
    img[g[ch["goal"]] > 0] = (.75, .75, 1.0)
    img[g[ch["relevant_goal"]] > 0] = (.1, .7, .2)
    img[g[ch["out_of_bounds"]] > 0] = (.15, .15, .15)
    ax.imshow(img, interpolation="nearest")
    for yy, xx in zip(*np.nonzero(g[ch["agent"]])):
        loaded = g[ch["agent_loaded"], yy, xx] > 0
        ax.add_patch(plt.Circle((xx, yy), .38, fc="black" if loaded else "white", ec="black", lw=1))
    row = rows[p]
    ax.add_patch(Rectangle((r - .5, r - .5), 1, 1, fill=False, ec="red", lw=2))
    state = "carry REQ" if row["carrying_requested"] else ("carry" if row["carrying"] else "empty")
    if row["return_phase"]:
        state += "+return"
    ax.set_title(f"#{p} u{row['unique_id']} x{row['copies_in_bank']} | {row['direction']}, {state}", fontsize=7.5,
                 color="crimson" if row["copies_in_bank"] > 1 else "black")
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"Canonical probe bank of {RUN.name} (48 probes, {len(counts)} unique; red title = duplicated input)\n"
             "orange = requested for MY team, green = MY goal, light blue = other goal, gray = shelf, "
             "white/black circle = agent unloaded/loaded, red square = self (center)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, .95))
fig.savefig(OUT / "decoded_bank.png")
print("unique inputs:", len(counts), "duplicate counts:", sorted(counts.tolist(), reverse=True)[:6])
print("states:", {k: sum(1 for r_ in rows if (("carry REQ" if r_["carrying_requested"] else ("carry" if r_["carrying"] else "empty")) == k)) for k in ("empty", "carry", "carry REQ")})
