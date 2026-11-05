"""Watch an IPPO pilot plan, judge it on the pre-registered criteria, write a report.  Never starts training.

Waits (polling) until every job of the plan has job_status "complete" (or one has "failed"), then computes
the five pilot criteria (D74; thresholds fixed in T104/T105) per seed and writes
output/pilot_verdicts/<plan name>/verdict.{json,md}.  Existing run folders are only read.

Criteria (training logs = stochastic policy; windows of 0.1M env steps; one episode = 500 steps)
  1 TOGGLE kept      every window in [0.2M, end) has toggles/episode >= 5                 (all seeds)
  2 own pickups      mean requested-shelf pickups/episode over [0.2M, end) >= 1            (all seeds)
  3 deliveries       some seed reaches 3 consecutive evaluations with deliveries >= 1,
                     or every seed has training deliveries/episode >= 1.0 in the last window
  4 spread           share of the top agent's training deliveries in the last window <= 0.7 (all seeds);
                     a seed with fewer than 20 deliveries in that window (deliveries/episode x 200) is
                     "pending", not judged
  5 no toggle spam   mean non-requested pickups/episode over [0.2M, end) <= 50% of the reference run
                     of the same seed (--spam-reference, e.g. the nopen pilot); skipped without it
Overall: "fail" if any criterion fails, "pending" if none fails and some is pending or skipped,
"pass" otherwise.  The report also lists the same numbers for --compare plans (same seeds) for reading.

    python latent_team/pilot_verdict.py --plan runs/pilot_20261022_112521_b_e1f6_comm3_pen005 \
        --spam-reference runs/pilot_20261021_133652_b_e1f6_comm3_nopen \
        --compare runs/pilot_20261019_180044_b_e1f6_comm3 runs/pilot_20261021_133652_b_e1f6_comm3_nopen
    (add --once to judge now without waiting; --poll seconds between checks, default 300)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pilot_verdicts"
WINDOW, START = 1e5, 2e5
EPISODES_PER_WINDOW = WINDOW / 500


def jobs(plan):
    manifest = json.loads((plan / "experiment.json").read_text(encoding="utf-8"))
    return [(j["name"], int(j["seed"]), int(j["config"]["total_timesteps"]), plan / j["name"]) for j in manifest["jobs"]]


def status(run):
    try:
        return json.loads((run / "job_status.json").read_text(encoding="utf-8"))["status"]
    except (OSError, ValueError, KeyError):
        return "not started"


def metrics(run, end):
    rows = [json.loads(l) for l in open(run / "metrics.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["env_steps"] <= end + 1000]
    train = [r["metrics"] | {"_step": r["env_steps"]} for r in rows if "train/toggle_count" in r["metrics"]]
    ev = [(r["env_steps"], r["metrics"]["eval/deliveries"]) for r in rows if "eval/deliveries" in r["metrics"]]
    n = max(int(k.split("_")[1]) for m in train for k in m if k.startswith("train/agent_") and k.endswith("_delivery_count"))
    last_lo = end - WINDOW

    def window(lo, hi, fn):
        v = [fn(m) for m in train if lo <= m["_step"] < hi]
        v = [x for x in v if x is not None and np.isfinite(x)]
        return float(np.mean(v)) if v else None

    pick = lambda m: sum(m.get(f"train/agent_{k}_pickup_count", 0.0) for k in range(1, n + 1))
    toggles = [window(lo, lo + WINDOW, lambda m: m.get("train/toggle_count")) for lo in np.arange(START, end, WINDOW)]
    toggles = [t for t in toggles if t is not None]
    per_agent = [window(last_lo, end + 1, lambda m, k=k: m.get(f"train/agent_{k}_delivery_count", 0.0)) or 0.0
                 for k in range(1, n + 1)]
    last_deliv = window(last_lo, end + 1, lambda m: m.get("train/deliveries")) or 0.0
    dl = [d for _, d in ev]
    onset = next((ev[i][0] for i in range(len(dl) - 2) if min(dl[i:i + 3]) >= 1), None)
    return dict(
        last_step=rows[-1]["env_steps"] if rows else 0,
        min_window_toggles=round(min(toggles), 2) if toggles else None,
        toggles_by_window=[round(t, 2) for t in toggles],
        own_pickups=round(window(START, end + 1, lambda m: m.get("train/requested_shelf_pickups")) or 0.0, 3),
        non_requested_pickups=round(window(START, end + 1, lambda m: pick(m) - m.get("train/requested_shelf_pickups", 0.0)) or 0.0, 3),
        last_window_train_deliveries=round(last_deliv, 3),
        last_window_delivery_count=round(last_deliv * EPISODES_PER_WINDOW, 1),
        top_agent_share=round(max(per_agent) / sum(per_agent), 3) if sum(per_agent) > 0 else None,
        eval_onset=onset, eval_last5=round(float(np.mean(dl[-5:])), 3) if dl else None,
    )


def judge(m_by_seed, spam_ref):
    seeds = sorted(m_by_seed)
    res = {}
    res["1_toggle_kept"] = "pass" if all((m["min_window_toggles"] or 0) >= 5 for m in m_by_seed.values()) else "fail"
    res["2_own_pickups"] = "pass" if all(m["own_pickups"] >= 1 for m in m_by_seed.values()) else "fail"
    res["3_deliveries"] = ("pass" if any(m["eval_onset"] is not None for m in m_by_seed.values())
                           or all(m["last_window_train_deliveries"] >= 1.0 for m in m_by_seed.values()) else "fail")
    four = []
    for s in seeds:
        m = m_by_seed[s]
        if m["last_window_delivery_count"] < 20 or m["top_agent_share"] is None:
            four.append("pending")
        else:
            four.append("pass" if m["top_agent_share"] <= 0.7 else "fail")
    res["4_spread"] = "fail" if "fail" in four else "pending" if "pending" in four else "pass"
    if spam_ref is None:
        res["5_no_toggle_spam"] = "skipped"
    else:
        ok = [m_by_seed[s]["non_requested_pickups"] <= 0.5 * spam_ref[s]["non_requested_pickups"]
              for s in seeds if s in spam_ref]
        res["5_no_toggle_spam"] = "skipped" if not ok else "pass" if all(ok) else "fail"
    vals = list(res.values())
    overall = "fail" if "fail" in vals else "pending" if ("pending" in vals or "skipped" in vals) else "pass"
    return res, overall


def plan_metrics(plan, end=None):
    out = {}
    for name, seed, total, run in jobs(plan):
        if (run / "metrics.jsonl").exists():
            out[seed] = metrics(run, end or total)
    return out


def report(plan, spam_reference, compare):
    js = jobs(plan)
    end = js[0][2]
    m = plan_metrics(plan)
    spam = plan_metrics(spam_reference, end) if spam_reference else None
    crit, overall = judge(m, spam)
    others = {str(p): plan_metrics(p, end) for p in compare}
    dest = OUT / plan.name
    dest.mkdir(parents=True, exist_ok=True)
    data = dict(plan=str(plan), judged_at=time.strftime("%Y-%m-%d %H:%M:%S"), end_step=end, overall=overall,
                criteria=crit, per_seed=m, spam_reference=str(spam_reference) if spam_reference else None,
                compare=others)
    (dest / "verdict.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    lines = [f"# Pilot verdict: {plan.name}", "", f"Judged {data['judged_at']} at {end:,} steps. **Overall: {overall}**", "",
             "| Criterion | Result |", "|---|---|"] + [f"| {k} | {v} |" for k, v in crit.items()] + ["",
             "| Seed | min toggles/ep | own pickups/ep | non-req pickups/ep | last-window deliveries/ep (count) | top share | eval onset | eval last 5 |",
             "|---|---|---|---|---|---|---|---|"]
    for label, mm in [(plan.name, m)] + list(others.items()):
        for s, v in sorted(mm.items()):
            lines.append(f"| {Path(label).name} s{s} | {v['min_window_toggles']} | {v['own_pickups']} | {v['non_requested_pickups']} | "
                         f"{v['last_window_train_deliveries']} ({v['last_window_delivery_count']}) | {v['top_agent_share']} | "
                         f"{v['eval_onset']} | {v['eval_last5']} |")
    lines += ["", "This report never starts training. Look at the curves before starting the next queue."]
    (dest / "verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest, overall


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--spam-reference", type=Path)
    ap.add_argument("--compare", type=Path, nargs="*", default=[])
    ap.add_argument("--poll", type=int, default=300)
    ap.add_argument("--once", action="store_true", help="judge now, even if jobs are still running")
    args = ap.parse_args()
    while True:
        states = {name: status(run) for name, _, _, run in jobs(args.plan)}
        done = all(s == "complete" for s in states.values())
        failed = [n for n, s in states.items() if s == "failed"]
        if done or failed or args.once:
            break
        print(time.strftime("[%H:%M:%S]"), states, flush=True)
        time.sleep(args.poll)
    if failed:
        print("failed jobs:", failed, "- judging what exists", flush=True)
    dest, overall = report(args.plan, args.spam_reference, args.compare)
    print(f"overall={overall}; report: {dest / 'verdict.md'}", flush=True)


if __name__ == "__main__":
    main()
