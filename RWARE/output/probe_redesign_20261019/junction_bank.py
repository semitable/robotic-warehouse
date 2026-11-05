"""Prototype bank built from a label-free design rule, evaluated offline.  No training, frozen models only.

Design rule (derived from inspect_probes.py; nothing below uses agent team labels):
  R1 decision points  highway cells where a carrying agent can go >= 3 ways (route branching)
  R2 goal hidden      no goal of any team inside the sensor window (only the learned prior can steer)
  R3 cue-neutral      team-relative inputs neutralised (requested_shelf / relevant_goal channels, ego
                      carrying_requested / return_phase / shelf_home_*); ego then reads "carrying an
                      unrequested shelf, home unknown", a state agents do visit
  R4 all headings     every heading at the cell, one probe per (cell, heading); duplicates dropped

Compared on the same models with uniform weighting (the current PGCT rule):
  v2 gate (48)             current E1-F bank design
  v2 gate ablated (48)     R3 only
  junction (R1+R2)         R1, R2, R4 with cues intact
  junction neutral         R1-R4

    python output/probe_redesign_20261019/junction_bank.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "output/probe_connect_20261019"))
sys.path.insert(0, str(ROOT / "output/probe_weight_20261019"))
from examples import train_recurrent_ippo_consensus as T  # noqa: E402
from examples.diagnose_pgct_rollout_compatibility import load_model, model_hash  # noqa: E402
from latent_team import probes_v2 as P2  # noqa: E402
from analyze import ablate  # noqa: E402
from delta_m import E1_MODELS, E1_BANK, E1F_BANK, E1F_JOBS, per_probe, score, teams_of  # noqa: E402

OUT = Path(__file__).resolve().parent


def junction_bank(env_ck):
    agents, meta = load_model(env_ck)
    cfg = meta["config"]
    env_kwargs = dict(T._load_checkpoint(str(env_ck), torch.device("cpu"))["effective_env_kwargs"])
    real = P2.ProbeRealiser(cfg["env_id"], env_kwargs, cfg["observation_format"], seed=1)
    try:
        cells = set(real.highway_cells())
        n_teams = len(real.u.goals_by_team)
        obs, meta_rows, seen = [], [], set()
        for (x, y) in sorted(cells):
            branches = sum((x + dx, y + dy) in cells for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            if branches < 3 or any(real.goal_visible(x, y, t) for t in range(n_teams)):
                continue
            for d in range(4):
                # team frame 0 only: after R3 the frame no longer matters, and with cues intact this keeps
                # one input per state (the same input is fed to every actor either way)
                o = real.realise(P2.Scenario("gate", "carry_goal_hidden", 0, x, y, d, carrying=True)).astype(np.float32)
                key = o.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                obs.append(o)
                meta_rows.append(dict(x=x, y=y, direction=d, branches=branches))
    finally:
        real.close()
    seq = torch.as_tensor(np.stack(obs))[:, None, :]
    return seq, meta_rows


def evaluate(models, banks, teams_fn):
    out = {}
    for name, ck in models.items():
        agents, meta = load_model(ck)
        h = model_hash(agents)
        teams = teams_fn(ck)
        out[name] = {}
        for bname, seq in banks.items():
            js, _ = per_probe(agents, seq[:, 0])
            out[name][bname] = score(js, teams, np.ones(len(seq)))
        assert h == model_hash(agents)
        print(f"{name:20s} " + " ".join(f"{b}={r['auroc']:.2f}/{r['nearest_is_mate']}" for b, r in out[name].items()),
              flush=True)
    return out


def main():
    torch.set_num_threads(2)
    results = {}
    # E1 family
    v2 = torch.load(E1_BANK, map_location="cpu", weights_only=False)["sequences"].float()
    jseq, jmeta = junction_bank(E1_MODELS["E1 IPPO"])
    print(f"E1 junction probes: {len(jseq)} (cells {sorted({(m['x'], m['y']) for m in jmeta})})")
    banks = {"v2": v2, "v2_ablated": ablate(v2), "junction": jseq, "junction_neutral": ablate(jseq)}
    results["E1"] = dict(n_junction=len(jseq), junction_cells=jmeta,
                         scores=evaluate(E1_MODELS, banks, lambda ck: [0, 1, 0, 1]))
    # E1-F: latest available common checkpoints
    fv2 = torch.load(E1F_BANK, map_location="cpu", weights_only=False)["sequences"].float()
    ckpts = sorted((E1F_JOBS["IPPO"]).glob("checkpoint_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    common = [c.name for c in ckpts if all((d / c.name).exists() for d in E1F_JOBS.values())]
    pick = [c for c in common if int(c.split("_")[1].split(".")[0]) % 500 == 0][-3:]
    fseq, fmeta = junction_bank(E1F_JOBS["IPPO"] / pick[-1])
    print(f"E1-F junction probes: {len(fseq)}; checkpoints {pick}")
    fbanks = {"v2": fv2, "v2_ablated": ablate(fv2), "junction": fseq, "junction_neutral": ablate(fseq)}
    models = {f"{job}@{c.split('_')[1][:5]}": d / c for job, d in E1F_JOBS.items() for c in pick}
    teams = teams_of(E1F_JOBS["IPPO"] / pick[-1])
    results["E1-F"] = dict(n_junction=len(fseq), junction_cells=fmeta, checkpoints=pick,
                           scores=evaluate(models, fbanks, lambda ck: teams))
    for fam in results:
        S = results[fam]["scores"]
        summ = {b: dict(auroc_mean=float(np.mean([S[m][b]["auroc"] for m in S])),
                        auroc_ge_075=int(sum(S[m][b]["auroc"] >= 0.75 for m in S)),
                        nearest_mean=float(np.mean([S[m][b]["nearest_is_mate"] for m in S])))
                for b in next(iter(S.values()))}
        results[fam]["summary"] = summ
        print(f"== {fam} summary ({len(S)} models): " + " | ".join(
            f"{b}: AUROC mean {v['auroc_mean']:.2f}, >=.75 in {v['auroc_ge_075']}, nearest {v['nearest_mean']:.1f}/4"
            for b, v in summ.items()), flush=True)
    (OUT / "junction_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
