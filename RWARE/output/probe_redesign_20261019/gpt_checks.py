"""Checks for GPT's review (read-only): duplicate probes, Delta_m(t) consistency in E1-F, and an
independent-lineage E1 test (from-scratch E24 models, never seen when the rule was derived)."""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from junction_bank import junction_bank, evaluate, ablate, E1_MODELS, E1_BANK, E1F_BANK  # noqa: E402
torch.set_num_threads(2)
uniq = lambda s: len(np.unique(s.reshape(len(s), -1).numpy(), axis=0))
v2 = torch.load(E1_BANK, map_location="cpu", weights_only=False)["sequences"].float()
fv2 = torch.load(E1F_BANK, map_location="cpu", weights_only=False)["sequences"].float()
jseq, jmeta = junction_bank(E1_MODELS["E1 IPPO"])
band = jseq[[k for k, m in enumerate(jmeta) if m["y"] <= 1]]
out = {"unique": {"E1 v2": [len(v2), uniq(v2)], "E1 v2 ablated": [len(v2), uniq(ablate(v2))],
                  "E1-F v2": [len(fv2), uniq(fv2)], "E1-F v2 ablated": [len(fv2), uniq(ablate(fv2))],
                  "E1 rule B neutral": [len(band), uniq(ablate(band))]}}
print("unique inputs (total, unique):", out["unique"])
# Delta_m(t) consistency, E1-F (from E32)
r = json.loads((ROOT / "output/probe_weight_20261019/results.json").read_text(encoding="utf-8"))["E1-F v2"]["deltas"]
def rho(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])
jobs = sorted({k.split("@")[0] for k in r})
within, across = [], []
for j in jobs:
    ks = sorted(k for k in r if k.startswith(j + "@"))
    within += [rho(r[ks[a]], r[ks[a + 1]]) for a in range(len(ks) - 1)]
last = [k for k in r if k.endswith("1.54M")]
across = [rho(r[a], r[b]) for i, a in enumerate(last) for b in last[i + 1:]]
D = np.array([r[k] for k in last])
out["delta_t"] = dict(within_job_consecutive_spearman=within, across_jobs_at_1_54M_spearman=across,
                      probes_positive_in_all_jobs_at_1_54M=np.flatnonzero((D > 0).all(0)).tolist())
print(f"E1-F Δ_m rank corr: consecutive checkpoints mean {np.mean(within):+.2f} (range {min(within):+.2f}..{max(within):+.2f}); "
      f"across jobs @1.54M mean {np.mean(across):+.2f}; probes Δ>0 in all 5 jobs @1.54M: {out['delta_t']['probes_positive_in_all_jobs_at_1_54M']}")
# independent lineage: from-scratch E1 (E24), last checkpoint 06250
S = ROOT / "runs/scratch_20261015_093349_stage2"
models = {f"scratch {m}": S / f"e1_{m}_s1/checkpoint_06250.pt" for m in ("ippo", "pgct", "oracle", "all")}
res = evaluate(models, {"v2": v2, "v2_ablated": ablate(v2), "ruleB_neutral": ablate(band)}, lambda ck: [0, 1, 0, 1])
out["scratch_e1"] = res
(Path(__file__).resolve().parent / "gpt_checks.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
