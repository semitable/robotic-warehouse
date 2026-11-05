"""Variant B of the design rule: R1-R4 plus 'far band' (cell at least 9 rows from the goal row), headings all.
Chosen AFTER looking at E1-family probes, so E1 numbers are in-sample; E1-F is the out-of-sample test."""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from junction_bank import junction_bank, evaluate, ablate, E1_MODELS, E1F_JOBS, teams_of  # noqa: E402
torch.set_num_threads(2)
out = {}
for fam, env_ck, models, tf in (
        ("E1", E1_MODELS["E1 IPPO"], E1_MODELS, lambda ck: [0, 1, 0, 1]),
        ("E1-F", E1F_JOBS["IPPO"] / "checkpoint_01500.pt",
         {f"{j}@{c[11:16]}": d / c for j, d in E1F_JOBS.items() for c in ("checkpoint_00500.pt", "checkpoint_01000.pt", "checkpoint_01500.pt")},
         None)):
    seq, meta = junction_bank(env_ck)
    keep = [k for k, m in enumerate(meta) if 10 - m["y"] >= 9]
    band = ablate(seq[keep])
    teams = [0, 1, 0, 1] if tf else teams_of(env_ck)
    print(f"{fam}: far-band neutral probes {len(keep)}")
    S = evaluate(models, {"far_band_neutral": band}, lambda ck: teams)
    a = [S[m]["far_band_neutral"]["auroc"] for m in S]
    out[fam] = dict(n=len(keep), scores=S, auroc_mean=float(np.mean(a)), ge075=int(sum(x >= .75 for x in a)))
    print(f"== {fam}: AUROC mean {np.mean(a):.2f}, >=.75 in {sum(x >= .75 for x in a)}/{len(a)}")
(Path(__file__).resolve().parent / "far_band_results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
