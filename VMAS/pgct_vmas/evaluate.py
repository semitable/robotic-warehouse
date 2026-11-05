"""Evaluate a saved VMAS checkpoint with fixed seeds (new JSON next to the job; never overwrites).

  python -m pgct_vmas.evaluate --job VMAS/runs/<name>/pgct_d2_s1 [--checkpoint final.pt] [--episodes 20]
"""
import sys

from .experiment import SPEC
from pgct_runner.cli import evaluate_main


def main(argv=None):
    return evaluate_main(SPEC, argv)


if __name__ == "__main__":
    sys.exit(main())
