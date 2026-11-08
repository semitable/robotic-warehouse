"""Follower report of a VMAS plan with a frozen expert (pgct_runner.follow).

  python -m pgct_vmas.follow --plan VMAS/runs/<name> [--window 250000]
"""
import sys

from .experiment import SPEC
from pgct_runner.follow import follow_main


def main(argv=None):
    return follow_main(SPEC, argv)


if __name__ == "__main__":
    sys.exit(main())
