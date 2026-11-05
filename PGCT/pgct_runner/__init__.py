"""pgct_runner: shared experiment plumbing for adapter-based PGCT experiments (LBF, VMAS).

Config files with sections, per-job plans, the train/evaluate/video/aggregate command lines, optional
Weights & Biases tracking and evaluation videos. The algorithm and the training loop live in pgct_core;
nothing here changes what a training run computes. An experiment package describes itself with one
:class:`ExperimentSpec` and wraps these entry points (see LBF/pgct_lbf/experiment.py).

Overcooked predates this package and keeps its own copies of the same plumbing (Overcooked/pgct_overcooked).
"""
from .spec import ExperimentSpec

__all__ = ["ExperimentSpec"]
