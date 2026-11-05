"""Figure: what the gate 'sees' when an agent serves at the west vs the east window, for probe v1/v2/v3.

Cells are decoded from the actual projection outputs (pgct_overcooked.adapter), not drawn by hand.
  python Overcooked/docs/make_probe_view_figure.py   ->   Overcooked/docs/img/probe_views_v1_v2_v3.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parents[1] / "PGCT")]
from pgct_overcooked.adapter import CH, EGO_SIZE, N_CHANNELS, AdapterConfig, CROSS_CELLS, KitchenAdapter  # noqa: E402
from pgct_overcooked.env.kitchen import KitchenConfig  # noqa: E402

LABELS = [("serve", "S", "#5ca860"), ("dish_dispenser", "D", "#c8c8d0"), ("counter", "X", "#96806a"),
          ("onion_dispenser", "O", "#d6b548"), ("tomato_dispenser", "T", "#c4483c"), ("pot", "P", "#46464c")]
FRAMES = [("absolute", "v1: north-up 3x3"), ("facing", "v2: rotated to heading, 3x3"),
          ("facing_cross", "v3: rotated, cross (5 cells)")]
CASES = [("west window", (3, 1), 3), ("east window", (3, 9), 1)]


def cells(frame):
    adapter = KitchenAdapter(KitchenConfig(team_assignment="fixed"), AdapterConfig(probe_frame=frame))
    out = []
    for _, pos, facing in CASES:
        adapter.reset()
        for i, (p, f) in enumerate([(pos, facing), ((1, 5), 2), ((5, 4), 2), ((5, 6), 2)]):
            a = adapter.env.agents[i]
            a.pos, a.facing, a.held = p, f, ("soup_tomato" if i == 0 else None)
        view = adapter.projection.public_view(adapter.observations()[0])
        if frame == "facing_cross":
            grid = np.full((N_CHANNELS, 3, 3), np.nan)
            vals = view[EGO_SIZE:].reshape(N_CHANNELS, 5)
            for k, (r, c) in enumerate(CROSS_CELLS):
                grid[:, r, c] = vals[:, k]
        else:
            grid = view[EGO_SIZE:].reshape(N_CHANNELS, 3, 3).astype(float)
        heading = int(np.argmax(view[:4]))
        out.append((grid, heading))
    return out


fig, axes = plt.subplots(len(FRAMES), 2, figsize=(5.2, 7.2))
arrows = {0: (0, .32), 1: (.32, 0), 2: (0, -.32), 3: (-.32, 0)}
for row, (frame, title) in enumerate(FRAMES):
    views = cells(frame)
    same = np.array_equal(np.nan_to_num(views[0][0], nan=-1), np.nan_to_num(views[1][0], nan=-1))
    for col, ((case, _, _), (grid, heading)) in enumerate(zip(CASES, views)):
        ax = axes[row, col]
        for r in range(3):
            for c in range(3):
                y = 2 - r
                if np.isnan(grid[0, r, c]):
                    ax.add_patch(plt.Rectangle((c, y), 1, 1, color="#f2f2f2", hatch="//", ec="#cccccc"))
                    continue
                label, color = "", "#efe7d6"
                for name, short, col_ in LABELS:
                    if grid[CH[name], r, c]:
                        label, color = short, col_
                ax.add_patch(plt.Rectangle((c, y), 1, 1, color=color, ec="#888888"))
                if (r, c) == (1, 1):
                    ax.add_patch(plt.Circle((1.5, 1.5), .3, color="#3465cc"))
                    dx, dy = arrows[heading]
                    ax.arrow(1.5, 1.5, dx, dy, width=.05, color="white", length_includes_head=True)
                elif label:
                    ax.text(c + .5, y + .5, label, ha="center", va="center", fontsize=14, weight="bold")
        ax.set_xlim(0, 3), ax.set_ylim(0, 3), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(case, fontsize=9)
    axes[row, 0].set_ylabel(title + ("\n(identical)" if same else "\n(different)"), fontsize=9)
fig.suptitle("Probe view of 'serve a soup' at the two windows\nS window, D dish dispenser, X counter, blue = agent + heading",
             fontsize=9)
fig.tight_layout()
out = HERE / "img" / "probe_views_v1_v2_v3.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)
print(out)
