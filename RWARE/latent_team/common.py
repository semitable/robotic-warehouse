"""Shared paths and read-only statistics for the latent-team experiments.

Everything original is imported through the ``examples`` package so that
monkey-patching the trainer module affects the single instance used by the
existing spatial/transfer runners.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REFERENCE_EXPERIMENT = ROOT / "runs/sp_20261008_120638_983201"
CURRICULUM = ROOT / "runs/pgct09700_physical_mixing_20260929_155338_079437"
STAGE03 = CURRICULUM / "stage03_4ag_physical_home0.750/final.pt"
STAGE04 = CURRICULUM / "stage04_4ag_physical_home0.500/final.pt"

PAIRS = [(0, 2), (1, 3), (0, 1), (0, 3), (1, 2), (2, 3)]
PAIR_LABELS = ["1-3", "2-4", "1-2", "1-4", "2-3", "3-4"]


def same_team_mask(team_ids):
    """Pair order PAIRS; labels are used for scoring only, never to compute a score."""
    team_ids = np.asarray(team_ids)
    return np.asarray([team_ids[i] == team_ids[j] for i, j in PAIRS], dtype=bool)


def auroc(scores, positive):
    """Mann-Whitney AUROC with average ranks for ties; None if a class is empty."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    positive = np.asarray(positive, dtype=bool).ravel()
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop + 1) / 2
        start = stop
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def teammate_recovery(pair_scores, team_ids):
    """Fraction of agents whose highest-scoring partner is a teammate (ties share credit).

    With 4 agents/2 teams the chance level is 1/3.
    """
    team_ids = np.asarray(team_ids)
    matrix = np.full((4, 4), -np.inf)
    for k, (i, j) in enumerate(PAIRS):
        matrix[i, j] = matrix[j, i] = pair_scores[k]
    hits = []
    for i in range(4):
        others = [j for j in range(4) if j != i]
        best = max(matrix[i, j] for j in others)
        tied = [j for j in others if matrix[i, j] == best]
        hits.append(sum(team_ids[j] == team_ids[i] for j in tied) / len(tied))
    return float(np.mean(hits))


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:  # only a partially written final line is tolerated
                raise
    return rows


def evaluation_curve(run_dir):
    """Evaluation deliveries by env step, as in output/learning_review_20261013/analyze.py."""
    evals = {}
    for row in read_jsonl(Path(run_dir) / "metrics.jsonl"):
        metrics, step = row["metrics"], row["env_steps"]
        if "eval/deliveries" in metrics:
            evals[step] = {k.removeprefix("eval/"): v for k, v in metrics.items() if k.startswith("eval/")}
        if "eval/final_deliveries" in metrics:
            evals[step] = {k.removeprefix("eval/final_"): v for k, v in metrics.items()
                           if k.startswith("eval/final_")}
    steps = sorted(evals)
    return steps, [evals[s] for s in steps]


def learning_summary(run_dir, max_step=None):
    """Observed-time-average AUC: trapezoid over logged evaluations only, no step-0 score.

    ``max_step`` truncates a longer reference run to a shorter matched budget.
    """
    steps, metrics = evaluation_curve(run_dir)
    if max_step is not None:
        keep = [k for k, s in enumerate(steps) if s <= max_step]
        steps, metrics = [steps[k] for k in keep], [metrics[k] for k in keep]
    result = dict(run_dir=str(run_dir), eval_count=len(steps), truncated_at=max_step)
    if not steps:
        return result
    delivery = np.asarray([m["deliveries"] for m in metrics], dtype=np.float64)
    result.update(first_eval_step=int(steps[0]), last_eval_step=int(steps[-1]),
                  last_deliveries=float(delivery[-1]), last10_deliveries=float(delivery[-10:].mean()),
                  last_returns_home=float(metrics[-1].get("returns_home", np.nan)))
    if len(steps) >= 2:
        result["delivery_auc_per_step"] = float(np.trapezoid(delivery, steps) / (steps[-1] - steps[0]))
    return result
