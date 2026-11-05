"""Train IPPO / Oracle / All / PGCT-D2 / PGCT-ID-only / random-gate on VMAS Team Discovery (pgct_runner.cli).

  python -m pgct_vmas.train --config main_4p2t.json --methods ippo pgct_d2 --seeds 1 --name smoke --smoke
Run with VMAS and PGCT on PYTHONPATH (scripts/run_experiment.ps1 -Experiment VMAS does this). Outputs go to
VMAS/runs/<name>/ (plan.json + one folder per <method>_s<seed>).
"""
import sys

from .experiment import SPEC
from pgct_runner.cli import train_main


def main(argv=None):
    return train_main(SPEC, argv)


if __name__ == "__main__":
    sys.exit(main())
