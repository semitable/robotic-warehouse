"""Read-only progress check of a running (or finished) main-experiment plan folder (T178, D110 stop check).

Usage:  python latent_team/main_check.py runs/<plan folder> [--min-sps 14]

Per job: status, environment steps, recent training speed. For gate jobs (pgct_max, oracle_mature_max) also
the reward-mature agents, same-/different-team approvals (from the audit fields of max_gae.jsonl; team labels
are used only here, never by the gate), the first same-team approval, and the share of mature pair checks that
found no common public action (action_max). Never writes into the run folders and never stops a job.

Convergence (T185): at 5/6/7/8/9M each job's evaluations are checked with convergence() below; a job that has
converged and is still running is listed with its PID so the user can stop it. Nothing is stopped here.

Stop criteria (D110), judged around 1.3M steps for each pgct_max job:
  no error, speed >= --min-sps, >= 4 mature agents, >= 1 same-team approval, no different-team approval,
  and no_common_action not the majority of mature pair checks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GATE_MODES = ("pgct_max", "oracle_mature_max")

# T185 convergence rule (user: 10M budget, stop at 5/6/7/8/9M once learning has converged). Judged on the
# deterministic evaluations: mean of window B = (M - .5M, M] against window A = (M - 1M, M - .5M]. Converged when
# the team delivery rate moved by at most max(5% of B, .5 per episode) AND every agent moved by at most .3 per
# episode (so an agent that only starts learning late keeps the job running). Needs >= 5 evaluations per window.
MILESTONES = tuple(int(m * 1e6) for m in (5, 6, 7, 8, 9))
WINDOW = 500_000
TEAM_REL, TEAM_ABS, AGENT_ABS, MIN_EVALS = .05, .5, .3, 5


def eval_rows(job_dir, episodes):
    """(env_steps, team deliveries per episode, [per-agent deliveries per episode]) of every evaluation."""
    rows = []
    for r in jsonl(job_dir / "metrics.jsonl"):
        m = r.get("metrics", {})
        if "eval/deliveries" in m:
            agents = sorted(k for k in m if k.startswith("eval/agent_") and k.endswith("_delivery_count"))
            agents.sort(key=lambda k: int(k.split("_")[1]))
            rows.append((r["env_steps"], float(m["eval/deliveries"]), [float(m[k]) / episodes for k in agents]))
    return rows


def convergence(rows, milestone):
    """Convergence verdict at one milestone, or None if the job has not reached it / too few evaluations."""
    a = [r for r in rows if milestone - 2 * WINDOW < r[0] <= milestone - WINDOW]
    b = [r for r in rows if milestone - WINDOW < r[0] <= milestone]
    if not rows or rows[-1][0] < milestone or len(a) < MIN_EVALS or len(b) < MIN_EVALS:
        return None
    mean = lambda xs: sum(xs) / len(xs)
    team_a, team_b = mean([r[1] for r in a]), mean([r[1] for r in b])
    n = min(len(r[2]) for r in a + b)
    agent_change = [abs(mean([r[2][k] for r in b]) - mean([r[2][k] for r in a])) for k in range(n)]
    team_ok = abs(team_b - team_a) <= max(TEAM_REL * team_b, TEAM_ABS)
    agents_ok = all(x <= AGENT_ABS for x in agent_change)
    return dict(milestone=milestone, converged=bool(team_ok and agents_ok), team_a=team_a, team_b=team_b,
                worst_agent=int(max(range(n), key=agent_change.__getitem__)) + 1 if n else None,
                worst_agent_change=max(agent_change) if n else 0.0)


def jsonl(path, tail=None):
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for line in lines[-tail:] if tail else lines:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:  # a line being written right now
            pass
    return rows


def check(job_dir, mode, min_sps, episodes=20):
    status = json.loads((job_dir / "job_status.json").read_text(encoding="utf-8")).get("status", "?") \
        if (job_dir / "job_status.json").is_file() else "not started"
    metrics = jsonl(job_dir / "metrics.jsonl", tail=60)
    steps = metrics[-1]["env_steps"] if metrics else 0
    sps = [m["metrics"]["train/sps"] for m in metrics if "train/sps" in m.get("metrics", {})][-20:]
    out = dict(job=job_dir.name, status=status, steps=steps, sps=sum(sps) / len(sps) if sps else None)
    if steps >= MILESTONES[0] - 2 * WINDOW:
        rows = eval_rows(job_dir, episodes)
        out["milestones"] = [v for v in (convergence(rows, m) for m in MILESTONES) if v is not None]
        out["converged_at"] = next((v["milestone"] for v in out["milestones"] if v["converged"]), None)
    if mode not in GATE_MODES:
        return out
    rows = jsonl(job_dir / "max_gae.jsonl")
    if not rows:
        return out
    last = rows[-1]
    same = int(sum(r.get("max_gae_approved_same_team_arcs", 0) for r in rows))
    cross = int(sum(r.get("max_gae_approved_cross_team_arcs", 0) for r in rows))
    first_same = next((r["env_steps"] for r in rows if r.get("max_gae_approved_same_team_arcs", 0) > 0), None)
    recent = [p for r in rows[-50:] for p in r.get("ac_pairs", []) if p.get("reason") != "immature"]
    no_common = sum(p.get("reason") == "no_common_action" for p in recent)
    out.update(mature=int(sum(bool(x) for x in last.get("reward_mature", []))), same_arcs=same, cross_arcs=cross,
               first_same=first_same, no_common_share=no_common / len(recent) if recent else None)
    if mode == "pgct_max":
        problems = []
        if status == "failed":
            problems.append("failed")
        if out["sps"] is not None and out["sps"] < min_sps:
            problems.append(f"speed {out['sps']:.1f} < {min_sps}")
        if out["mature"] < 4:
            problems.append(f"{out['mature']} mature agents")
        if same == 0:
            problems.append("no same-team approval")
        if cross > 0:
            problems.append(f"{cross} different-team approval arcs")
        if out["no_common_share"] is not None and out["no_common_share"] > .5:
            problems.append(f"no_common_action {out['no_common_share']:.0%} of mature checks")
        out["verdict"] = "OK" if not problems else "CHECK: " + "; ".join(problems)
        if steps < 1_300_000:
            out["verdict"] = f"(before 1.3M) {out['verdict']}"
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("plan", type=Path)
    parser.add_argument("--min-sps", type=float, default=14.0,
                        help="Per-job speed floor; ~16 steps/s was measured with 4 jobs at once, ~25 with 2.")
    args = parser.parse_args()
    manifest = json.loads((args.plan / "experiment.json").read_text(encoding="utf-8"))
    # run_queue.ps1 writes runs/<Name>_pids.json next to the plan folder runs/<Name>_<candidate>
    pid_file = args.plan.parent / (args.plan.name[:-len(manifest.get("env_variant", "")) - 1] + "_pids.json")
    pids = {}
    if pid_file.is_file():
        entries = json.loads(pid_file.read_text(encoding="utf-8-sig"))
        pids = {e["Job"]: e["PID"] for e in (entries if isinstance(entries, list) else [entries])}
    stop = []
    for job in manifest["jobs"]:
        r = check(args.plan / job["name"], job["mode"], args.min_sps, job["config"].get("eval_episodes", 20))
        line = f"{r['job']}: {r['status']}, {r['steps'] / 1e6:.2f}M"
        if r["sps"] is not None:
            line += f", {r['sps']:.1f} steps/s"
        if "mature" in r:
            share = "-" if r["no_common_share"] is None else f"{r['no_common_share']:.0%}"
            first = "-" if r["first_same"] is None else f"{r['first_same'] / 1e6:.2f}M"
            line += (f" | mature {r['mature']}/6, approved arcs same {r['same_arcs']} / different {r['cross_arcs']}, "
                     f"first same {first}, no_common_action {share}")
        if "verdict" in r:
            line += f" | {r['verdict']}"
        print(line)
        for v in r.get("milestones", []):
            print(f"    {v['milestone'] / 1e6:.0f}M: team {v['team_a']:.2f} -> {v['team_b']:.2f}/ep, largest agent change "
                  f"{v['worst_agent_change']:.2f} (agent_{v['worst_agent']}) -> "
                  f"{'converged' if v['converged'] else 'still changing'}")
        if r.get("converged_at") and r["status"] == "running":
            stop.append(f"{job['name']}: converged at {r['converged_at'] / 1e6:.0f}M, PID {pids.get(job['name'], '?')}")
    if stop:
        print("\nConverged and still running (the user may stop them; this script never stops anything):")
        print("\n".join("  " + s for s in stop))


if __name__ == "__main__":
    main()
