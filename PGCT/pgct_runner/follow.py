"""Follower report (D145): how fast do novices learn next to a frozen, pre-trained expert?

Groups per job (evaluation code; uses the true teams): expert = frozen agents, followers = their hidden teammates,
controls = every other non-frozen agent (no expert in their team). For each method, seed and window of env steps
the report gives the mean per-agent own-reward-event count and return per training episode, and the first step at
which a group reaches a fraction of the expert's level. Writes <plan>/follow_report.json and .md; changes nothing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .aggregate import read_jsonl


def groups(spec, config):
    experiment = config["experiment"]
    teams = np.asarray(spec.build(experiment, experiment["trainer"]["method"],
                                  experiment["trainer"]["seed"])[0]().oracle_team_ids())
    frozen = list(config["trainer"].get("frozen_agents") or [])
    expert_teams = {int(teams[i]) for i in frozen}
    followers = [i for i in range(len(teams)) if i not in frozen and int(teams[i]) in expert_teams]
    controls = [i for i in range(len(teams)) if i not in frozen and int(teams[i]) not in expert_teams]
    return frozen, followers, controls


def job_curves(spec, folder: Path, window: int):
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    frozen, followers, controls = groups(spec, config)
    key = f"episode_{spec.headline_events[0]}_per_agent"
    rows = [r for r in read_jsonl(folder / "metrics.jsonl") if key in r]
    out = dict(method=config["trainer"]["method"], seed=config["trainer"]["seed"], frozen=frozen,
               followers=followers, controls=controls, windows=[])
    if not rows:
        return out
    last = rows[-1]["env_steps"]
    for start in range(0, last, window):
        part = [r for r in rows if start <= r["env_steps"] < start + window]
        if not part:
            continue
        events = np.mean([r[key] for r in part], axis=0)
        returns = np.mean([r["episode_return_per_agent"] for r in part], axis=0)
        out["windows"].append(dict(start=start, **{f"{g}_events": float(np.mean(events[idx])) if idx else None
                                                  for g, idx in (("expert", frozen), ("followers", followers),
                                                                 ("controls", controls))},
                                   **{f"{g}_return": float(np.mean(returns[idx])) if idx else None
                                      for g, idx in (("expert", frozen), ("followers", followers),
                                                     ("controls", controls))}))
    expert_level = np.mean([w["expert_events"] for w in out["windows"] if w["expert_events"] is not None] or [np.nan])
    for g in ("followers", "controls"):
        for frac in (.5, .8):
            hit = next((w["start"] for w in out["windows"] if w[f"{g}_events"] is not None
                        and w[f"{g}_events"] >= frac * expert_level), None)
            out[f"{g}_reach_{int(frac * 100)}pct_expert"] = hit
    out["expert_events_mean"] = float(expert_level)
    return out


def follow_report(spec, plan: Path, window: int = 250_000):
    jobs = [job_curves(spec, p.parent, window) for p in sorted(Path(plan).glob("*/config.json"))]
    (plan / "follow_report.json").write_text(json.dumps(dict(window=window, jobs=jobs), indent=2), encoding="utf-8")
    lines = [f"# {spec.name} follower report", "",
             f"Per-agent own-reward events per training episode, mean over {window:,}-step windows. Expert = frozen "
             "agents; followers = its hidden teammates; controls = agents of other teams. Ground-truth teams are used "
             "for this evaluation only.", "",
             "| job | first window F / C | last window F / C | expert | F reach 50% / 80% | C reach 50% / 80% |",
             "|---|---|---|---|---|---|"]
    for j in jobs:
        if not j["windows"]:
            continue
        a, b = j["windows"][0], j["windows"][-1]
        lines.append(f"| {j['method']}_s{j['seed']} | {a['followers_events']:.3f} / {a['controls_events']:.3f} | "
                     f"{b['followers_events']:.3f} / {b['controls_events']:.3f} | {j['expert_events_mean']:.3f} | "
                     f"{j['followers_reach_50pct_expert']} / {j['followers_reach_80pct_expert']} | "
                     f"{j['controls_reach_50pct_expert']} / {j['controls_reach_80pct_expert']} |")
    (plan / "follow_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jobs


def follow_main(spec, argv=None):
    parser = argparse.ArgumentParser(description=f"{spec.name}: follower report of a plan with a frozen expert")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--window", type=int, default=250_000)
    args = parser.parse_args(argv)
    follow_report(spec, args.plan, args.window)
    print((args.plan / "follow_report.md").read_text(encoding="utf-8"))
    return 0
