"""Summarise cf_*.json from counterfactual_check.py (T149)."""
import json
from pathlib import Path
import numpy as np


def rankdata(x):
    """Average ranks with ties (scipy is not installed in this env)."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):
        tie = x == v
        ranks[tie] = ranks[tie].mean()
    return ranks

HERE = Path(__file__).resolve().parent
NAMES = ["acs1_1M", "acs1_2M", "corner1M_s1", "corner1M_s2"]


def spear(a, b):
    a, b = rankdata(a), rankdata(b)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def mse(xs):
    xs = np.asarray([x for x in xs if np.isfinite(x)])
    return (f"{xs.mean():+.2f}±{xs.std(ddof=1) / np.sqrt(len(xs)):.2f} (n={len(xs)})" if len(xs) > 1 else "  -  ")


summary = {}
for name in NAMES:
    p = HERE / f"cf_{name}.json"
    if not p.exists():
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    print(f"\n===== {name}  mature {d['mature']}")
    S = d["state"]
    res = {}
    for lag in (True, False):
        for kind in ("same", "cross"):
            rows = [r for r in S if r["lag"] == lag and r["kind"] == kind]
            if len(rows) < 3:
                continue
            dV = np.array([r["dV"] for r in rows]); dMC = np.array([r["dMC"] for r in rows])
            se = np.array([r["se"] for r in rows])
            hi = dV > 0.2
            tag = f"{'laggard' if lag else 'leader '} {kind:5s}"
            line = (f"{tag} n={len(rows):3d} corr(dV,dMC)={np.corrcoef(dV, dMC)[0, 1]:+.2f}  "
                    f"dMC>0 overall {np.mean(dMC > 0):.2f}  mean dMC {dMC.mean():+.3f}  mean dV {dV.mean():+.2f}")
            print(line)
            for lab, m in (("dV>0.2", hi), ("dV<=0.2", ~hi)):
                if m.sum():
                    print(f"      {lab:8s} n={m.sum():3d} mean dMC {dMC[m].mean():+.3f}  dMC>0 {np.mean(dMC[m] > 0):.2f}  "
                          f"sig gain {np.mean(dMC[m] > 2 * se[m]):.2f}  sig loss {np.mean(dMC[m] < -2 * se[m]):.2f}")
            res[tag] = dict(n=len(rows), corr=float(np.corrcoef(dV, dMC)[0, 1]), pos=float(np.mean(dMC > 0)),
                            mean_dMC=float(dMC.mean()), hi_n=int(hi.sum()),
                            hi_pos=float(np.mean(dMC[hi] > 0)) if hi.any() else None,
                            hi_sig_gain=float(np.mean(dMC[hi] > 2 * se[hi])) if hi.any() else None,
                            hi_sig_loss=float(np.mean(dMC[hi] < -2 * se[hi])) if hi.any() else None)
    A = d["action"]
    if A:
        s_self = [spear(r["a_self"], r["q_self"]) for r in A]
        s_f = [spear(r["a_f"], r["q_self"]) for r in A]
        s_self_l = [spear(r["a_self"], r["q_lead"]) for r in A]
        s_f_l = [spear(r["a_f"], r["q_lead"]) for r in A]
        s_qq = [spear(r["q_self"], r["q_lead"]) for r in A]
        top_self = [float(np.argmax(r["a_self"]) == np.argmax(r["q_self"])) for r in A]
        top_f = [float(np.argmax(r["a_f"]) == np.argmax(r["q_self"])) for r in A]
        chance = np.mean([1 / len(r["q_self"]) for r in A])
        print(f"action level (laggard states n={len(A)}): spearman vs MC Q(own continuation): "
              f"A_self {mse(s_self)}  A_max {mse(s_f)}")
        print(f"      vs MC Q(leader continuation): A_self {mse(s_self_l)}  A_max {mse(s_f_l)}  | "
              f"Q_own vs Q_leader agreement {mse(s_qq)}")
        print(f"      top-1 match with MC best: A_self {np.mean(top_self):.2f}  A_max {np.mean(top_f):.2f}  chance {chance:.2f}")
        res["action"] = dict(n=len(A), self=float(np.nanmean(s_self)), max=float(np.nanmean(s_f)),
                             self_lead=float(np.nanmean(s_self_l)), max_lead=float(np.nanmean(s_f_l)),
                             qq=float(np.nanmean(s_qq)), top_self=float(np.mean(top_self)), top_max=float(np.mean(top_f)),
                             chance=float(chance))
    G = d["gate"]
    for lag in (True, False):
        rows = [r for r in G if r["lag"] == lag]
        if rows:
            m = lambda k: float(np.mean([r[k] for r in rows]))
            print(f"gate {'laggard' if lag else 'leader '} receivers n={len(rows)}: mature teammate in range {m('avail'):.2f} of steps; "
                  f"uplift>.05 from gated donors {m('higher_gate'):.2f} vs whole team {m('higher_team'):.2f}; "
                  f"mean uplift gated {m('up_gate'):.3f} vs team {m('up_team'):.3f}; h=0 selection flips {m('flip'):.2f}")
            res[f"gate_{'lag' if lag else 'lead'}"] = {k: m(k) for k in ("avail", "higher_gate", "higher_team",
                                                                         "up_gate", "up_team", "flip")}
    summary[name] = res
(HERE / "cf_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
