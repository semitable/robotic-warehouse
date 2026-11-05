"""When do agents learn on the base (11 x 10) map, and how far is the wide-map main run? Read-only (T184).

Base runs: from-scratch 5M IPPO on b6_corner_indiv_pen005 (A+C v1 masking; contact range does not change IPPO,
E76), seeds 1 and 2, and the v2 PGCT discovery run (seed 1). Deterministic evaluation, 20 episodes x 500 steps
every 50 updates. Milestones: first evaluation reaching a total delivery rate, and for each agent the first
evaluation where it delivers >= .5 per episode (eval/agent_k_delivery_count is a 20-episode total).
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "base ippo s1": "runs/ac_20261023_185556/b6_corner_indiv_pen005_ac_ippo_s1",
    "base ippo s2": "runs/ac_20261023_185556/b6_corner_indiv_pen005_ac_ippo_s2",
    "base pgct(v2) s1": "runs/acv2_s1_20261029_084340/b6_corner_indiv_pen005_acv2_pgct_s1",
}
MAIN = "runs/main_amax_20261104_115922_b6_wide_corner_indiv_comm5_pen005"
LEVELS = (.5, 2, 5, 10, 20)


def evals(folder):
    rows = []
    for line in (ROOT / folder / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "eval/deliveries" in r["metrics"]:
            rows.append((r["env_steps"], r["metrics"]))
    return rows


def first(rows, test):
    return next((s for s, m in rows if test(m)), None)


def fmt(s):
    return "-" if s is None else f"{s / 1e6:.2f}M"


def main():
    lines = []
    say = lambda s: (print(s), lines.append(s))
    for name, folder in RUNS.items():
        rows = evals(folder)
        total = " ".join(f">={lv:g}:{fmt(first(rows, lambda m, lv=lv: m['eval/deliveries'] >= lv))}" for lv in LEVELS)
        agents = [fmt(first(rows, lambda m, k=k: m.get(f"eval/agent_{k}_delivery_count", 0) / 20 >= .5))
                  for k in range(1, 7)]
        at5 = [s for s, _ in rows if s <= 5.0e6][-1]
        last = [m for s, m in rows if s <= 5.0e6][-1]
        say(f"{name}: delivery/episode {total} | per agent (>=.5/ep) {agents} | at {at5 / 1e6:.2f}M "
            f"total {last['eval/deliveries']:.1f}, per agent "
            f"{[round(last.get(f'eval/agent_{k}_delivery_count', 0) / 20, 2) for k in range(1, 7)]}")
    main = ROOT / MAIN
    if main.is_dir():
        for job in sorted(p for p in main.iterdir() if p.is_dir()):
            if not (job / "metrics.jsonl").is_file():
                continue
            text = (job / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            rows = [json.loads(l) for l in text[-30:] if l.strip().endswith("}")]
            train = [r["metrics"].get("train/deliveries") for r in rows if "train/deliveries" in r["metrics"]]
            ev = [(r["env_steps"], r["metrics"]["eval/deliveries"]) for r in rows if "eval/deliveries" in r["metrics"]]
            say(f"wide {job.name.split('_amax_')[-1]}: {rows[-1]['env_steps'] / 1e6:.3f}M, recent train deliveries/ep "
                f"{[round(x, 2) for x in train[-5:]]}, evals {[(round(s / 1e6, 3), round(d, 2)) for s, d in ev[-3:]]}")
    (Path(__file__).parent / "learning_milestones_stdout.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
