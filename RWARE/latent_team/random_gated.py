"""Random-gated critic sharing: PGCT's measured transfer budget, donors chosen at random.

Control for "does *choosing* the donor matter, or only how much is shared?"  At every update the
receiver x donor allocation keeps the number of directed transfer arcs that a reference PGCT run
actually used at the same update (its transfer.jsonl), but the arcs are drawn uniformly at random
among the physical neighbours available now (the same physical constraint as every other mode).
Team labels are never read.

  * The allocation is row-normalised like the trainer's own (each receiver's donors share weight 1).
  * If fewer physical arcs exist than the reference used, the shortfall is carried to later updates
    (debt, capped at one full graph) so the cumulative budget stays matched; what cannot be repaid
    is logged as `unmet_arcs`.
  * The draw uses its own generator (seeded from the run seed), so the trainer's RNG stream is untouched.

Used through train_from_scratch.py `--modes random --random-reference <PGCT transfer.jsonl>`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RNG_OFFSET = 7_340_033  # keeps the donor draw independent of every other seed derived from the run seed


def reference_targets(reference):
    """Per-update directed arc counts actually used by a PGCT run (index = update - 1)."""
    rows = [json.loads(line) for line in Path(reference).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty reference transfer log: {reference}")
    modes = {r.get("mode") for r in rows}
    if modes != {"pgct"}:
        raise ValueError(f"Reference must be a PGCT transfer log, found modes {sorted(map(str, modes))}.")
    rows.sort(key=lambda r: r["update"])
    if [r["update"] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("Reference transfer log has missing or duplicated updates.")
    steps = {r["env_steps"] / r["update"] for r in rows}
    return [int(np.count_nonzero(np.asarray(r["actual_allocation"]) > 0)) for r in rows], steps.pop() if len(steps) == 1 else None


class RandomGatedAllocator:
    def __init__(self, reference, seed):
        self.targets, self.steps_per_update = reference_targets(reference)
        self.reference = str(reference)
        self.rng = np.random.default_rng(int(seed) + RNG_OFFSET)
        self.debt = 0
        self.unmet = 0

    def __call__(self, update, neighbors, active):
        """Allocation for this update.  `active` is False when the trainer shares nothing this update
        (transfer warm-up, or no peer loss prepared); the reference target then becomes debt."""
        if update > len(self.targets):
            raise RuntimeError(f"Random-gated reference {self.reference} covers {len(self.targets)} updates; "
                               f"update {update} has no matched budget.")
        neighbors = np.asarray(neighbors, dtype=bool).copy()
        np.fill_diagonal(neighbors, False)
        max_arcs = neighbors.shape[0] * (neighbors.shape[0] - 1)  # debt cap = one full graph
        target = self.targets[update - 1]
        want = target + self.debt
        candidates = np.argwhere(neighbors)
        k = min(want, len(candidates)) if active else 0
        allocation = np.zeros(neighbors.shape, dtype=np.float32)
        if k:
            chosen = candidates[self.rng.choice(len(candidates), size=k, replace=False)]
            allocation[chosen[:, 0], chosen[:, 1]] = 1.0
            rows = allocation.sum(1, keepdims=True)
            allocation = np.divide(allocation, rows, out=np.zeros_like(allocation), where=rows > 0)
        owed = want - k
        self.debt = min(owed, max_arcs)
        self.unmet += owed - self.debt
        return allocation, dict(random_target_arcs=int(target), random_used_arcs=int(k),
                                random_candidate_arcs=int(len(candidates)), random_debt=int(self.debt),
                                random_unmet_total=int(self.unmet))
