"""Summarise an LBF plan folder: summary.md/json and jobs.csv (evaluation code; may use labels).

  python -m pgct_lbf.aggregate --plan LBF/runs/<name>
"""
import sys

from .experiment import SPEC
from pgct_runner.aggregate import aggregate_main


def main(argv=None):
    return aggregate_main(SPEC, argv)


if __name__ == "__main__":
    sys.exit(main())
