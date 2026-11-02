"""Render the slide equations in deck_spec.EQUATIONS to PNG for the PowerPoint deck (matplotlib mathtext)."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from deck_spec import EQUATIONS  # noqa: E402

OUT = HERE.parents[1] / "presentation/figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"mathtext.fontset": "cm"})
for key, (_, mathtext) in EQUATIONS.items():
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, mathtext, fontsize=22, color="#1A1A1A")
    fig.savefig(OUT / f"eq_{key}.png", dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("eq", key)
