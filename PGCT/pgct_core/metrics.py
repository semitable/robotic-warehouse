"""Evaluation-side metrics. Team labels enter here (and in the Oracle baseline) only, never the gate.

Identification is measured from the gate's own records (approval matrices and pair distances per update)
against ground-truth teams; it is independent of learning performance.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


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


def same_team_matrix(team_ids):
    t = np.asarray(team_ids)
    return (t[:, None] == t[None, :]) & ~np.eye(len(t), dtype=bool)


def pair_confusion(approved, team_ids) -> Dict[str, float]:
    """Unordered-pair confusion of one symmetric approval matrix against the true teams."""
    approved = np.asarray(approved, dtype=bool)
    same = same_team_matrix(team_ids)
    iu = np.triu_indices(len(same), k=1)
    a, s = approved[iu], same[iu]
    tp, fp, fn, tn = int(np.sum(a & s)), int(np.sum(a & ~s)), int(np.sum(~a & s)), int(np.sum(~a & ~s))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else (0.0 if tp + fp + fn else None)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall, f1=f1,
                false_approval_rate=fp / (fp + tn) if fp + tn else None)


def identification_summary(updates: Sequence[dict], team_ids, steps_key="env_steps") -> Dict[str, object]:
    """Summary over a run's gate records (one dict per update with 'approved' and optional 'pairs').

    - final precision/recall/F1/false-approval rate (last update);
    - time-averaged false approval rate and recall;
    - first env step with a correct (same-team) approval, and with any wrong approval;
    - gate stability: fraction of consecutive updates with an unchanged approval matrix, and flip count;
    - AUROC of -EMA distance for same-team vs different-team pairs (last EMA of each compared pair);
    - mean last EMA distance for same-team and different-team pairs.
    """
    if not updates:
        return dict(updates=0)
    same = same_team_matrix(team_ids)
    first_correct = first_wrong = None
    fars, recalls, unchanged, flips = [], [], [], 0
    previous = None
    last_ema = {}
    for row in updates:
        approved = np.asarray(row["approved"], dtype=bool)
        conf = pair_confusion(approved, team_ids)
        if conf["false_approval_rate"] is not None:
            fars.append(conf["false_approval_rate"])
        if conf["recall"] is not None:
            recalls.append(conf["recall"])
        if first_correct is None and conf["tp"] > 0:
            first_correct = row.get(steps_key)
        if first_wrong is None and conf["fp"] > 0:
            first_wrong = row.get(steps_key)
        if previous is not None:
            changed = int(np.sum(np.triu(previous ^ approved, k=1)))
            flips += changed
            unchanged.append(changed == 0)
        previous = approved
        for p in row.get("pairs", []) or []:
            if p.get("distance_ema") is not None:
                last_ema[(p["receiver"], p["donor"])] = p["distance_ema"]
    final = pair_confusion(previous, team_ids)
    keys = sorted(last_ema)
    labels = [bool(same[i, j]) for i, j in keys]
    dists = [last_ema[k] for k in keys]
    out = dict(updates=len(updates), final=final,
               mean_false_approval_rate=float(np.mean(fars)) if fars else None,
               mean_recall=float(np.mean(recalls)) if recalls else None,
               first_correct_approval_step=first_correct, first_wrong_approval_step=first_wrong,
               gate_stability=float(np.mean(unchanged)) if unchanged else None, approval_flips=flips,
               compared_pairs=len(keys), auroc=auroc([-d for d in dists], labels))
    for name, flag in (("same", True), ("cross", False)):
        vals = [d for d, s in zip(dists, labels) if s == flag]
        out[f"distance_{name}_mean"] = float(np.mean(vals)) if vals else None
    return out


def learning_auc(steps: Iterable[float], values: Iterable[float]) -> Optional[float]:
    """Observed-time-average area under an evaluation curve (trapezoid; RWARE's delivery_auc_per_step)."""
    steps, values = np.asarray(list(steps), float), np.asarray(list(values), float)
    if len(steps) < 2:
        return None
    return float(np.trapezoid(values, steps) / (steps[-1] - steps[0]))


def mean_std(values: List[float]) -> Dict[str, Optional[float]]:
    values = [v for v in values if v is not None and np.isfinite(v)]
    return dict(n=len(values), mean=float(np.mean(values)) if values else None,
                std=float(np.std(values, ddof=1)) if len(values) > 1 else None)
