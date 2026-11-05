"""Draw the environment used by the running from-scratch run (read-only; no training)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from latent_team import common as C  # noqa: E402,F401
from examples import train_recurrent_ippo_consensus as T  # noqa: E402
from rware.utils.semantic_observation import SEMANTIC_SPATIAL_CHANNELS  # noqa: E402

RUN = ROOT / "runs/scratch_20261015_093349_stage2"
OUT = Path(__file__).resolve().parent
TEAM = {0: ("#d62728", "Red team"), 1: ("#1f77b4", "Blue team")}
DIRS = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}  # up, down, left, right

job = next(j for j in json.loads((RUN / "experiment.json").read_text(encoding="utf-8"))["jobs"] if j["mode"] == "oracle")
cfg = job["config"]
kwargs = json.loads(cfg["env_kwargs_json"])
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 5000001
with T.make_env(cfg["env_id"], kwargs, cfg["observation_format"]) as env:
    obs, _ = env.reset(seed=seed)
    u = env.unwrapped
    H, W = map(int, u.grid_size)
    teams = np.asarray(u.agent_team_ids)
    requested = {int(s.id): t for t, q in enumerate(u.team_request_queues) for s in q}
    flat = np.asarray(T.flatten_multi_agent_obs(env, obs))
    spec = T.get_semantic_observation_spec(env)
    grids = flat[:, spec.ego_dim:].reshape(4, spec.spatial_channels, spec.spatial_size, spec.spatial_size)

    fig = plt.figure(figsize=(16, 7.5), dpi=130)
    ax = fig.add_axes([0.02, 0.06, 0.40, 0.86])
    for y in range(H):
        for x in range(W):
            ax.add_patch(Rectangle((x, y), 1, 1, fc="#f2f2f2" if u._is_highway(x, y) else "#dcdcdc", ec="white", lw=1))
    for s in u.shelfs:
        t = u.shelf_team_ids.get(int(s.id))
        color = TEAM[t][0] if t is not None else "gray"
        req = int(s.id) in requested
        ax.add_patch(Rectangle((s.x + .12, s.y + .12), .76, .76, fc=color, alpha=.85 if req else .22,
                               ec="black" if req else color, lw=2.2 if req else .6))
        if req:
            ax.text(s.x + .5, s.y + .5, "REQ", ha="center", va="center", fontsize=6.5, color="white", weight="bold")
    for t, goals in enumerate(u.goals_by_team):
        for gx, gy in goals:
            ax.add_patch(Rectangle((gx, gy), 1, 1, fc=TEAM[t][0], alpha=.35, ec=TEAM[t][0], lw=2))
            ax.text(gx + .5, gy + .5, "GOAL", ha="center", va="center", fontsize=7, color=TEAM[t][0], weight="bold")
    r = int(u.sensor_range)
    for i, a in enumerate(u.agents):
        color = TEAM[int(teams[i])][0]
        ax.add_patch(plt.Circle((a.x + .5, a.y + .5), .36, fc="white", ec=color, lw=3, zorder=5))
        ax.text(a.x + .5, a.y + .5, str(i + 1), ha="center", va="center", fontsize=11, weight="bold", color=color, zorder=6)
        dx, dy = DIRS[int(a.dir.value)]
        ax.add_patch(FancyArrow(a.x + .5 + .3 * dx, a.y + .5 + .3 * dy, .25 * dx, .25 * dy, width=.08, color=color, zorder=6))
    a0 = u.agents[0]
    ax.add_patch(Rectangle((a0.x - r, a0.y - r), 2 * r + 1, 2 * r + 1, fill=False, ls="--", ec=TEAM[int(teams[0])][0], lw=1.5))
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal"); ax.set_xticks(range(W)); ax.set_yticks(range(H))
    ax.set_title(f"E1 map {H}x{W} (reset seed {seed}): agents 1,3 = Red, 2,4 = Blue\n"
                 "faint = shelves owned by each team (home .5 mixes zones)\n"
                 "bold REQ = requested now, dashed = agent 1 sensor window (r=5)", fontsize=8.5)

    # Same moment through the agents' own eyes: team-relative channels.
    ch = {n: k for k, n in enumerate(SEMANTIC_SPATIAL_CHANNELS)}
    for col, i in enumerate((0, 1)):
        g = grids[i]
        img = np.ones((spec.spatial_size, spec.spatial_size, 3)) * .95
        img[g[ch["highway"]] > 0] = (.98, .98, .98)
        img[g[ch["storage_cell"]] > 0] = (.86, .86, .86)
        img[g[ch["shelf"]] > 0] = (.55, .55, .55)
        img[g[ch["requested_shelf"]] > 0] = (1.0, .65, 0.0)
        img[g[ch["goal"]] > 0] = (.75, .75, 1.0)
        img[g[ch["relevant_goal"]] > 0] = (0.1, .7, 0.2)
        img[g[ch["out_of_bounds"]] > 0] = (0.15, 0.15, 0.15)
        axo = fig.add_axes([0.46 + col * 0.27, 0.16, 0.24, 0.66])
        axo.imshow(img, interpolation="nearest")
        for (yy, xx) in zip(*np.nonzero(g[ch["agent"]])):
            axo.add_patch(plt.Circle((xx, yy), .35, fc="white", ec="black", lw=1.5))
        axo.add_patch(Rectangle((r - .5, r - .5), 1, 1, fill=False, ec=TEAM[int(teams[i])][0], lw=3))
        axo.set_xticks([]); axo.set_yticks([])
        axo.set_title(f"What agent {i+1} ({TEAM[int(teams[i])][1]}) observes (11x11)\n"
                      "orange = requested for MY team, green = MY goal\nlight blue = other goal, gray = shelf\n"
                      "white circle = some agent (anonymous)", fontsize=8)
    fig.text(0.46, 0.04, "Observation channels are team-relative: each agent sees only its own team's requests and goal\n"
             "as 'requested'/'relevant'; other agents' teams are not visible.", fontsize=9)
    fig.savefig(OUT / f"environment_seed{seed}.png")
    print(OUT / f"environment_seed{seed}.png")
    print("teams", teams.tolist(), "goals", u.goals_by_team, "requests", requested, "agents", [(a.x, a.y) for a in u.agents])
