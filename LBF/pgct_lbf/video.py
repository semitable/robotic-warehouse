"""Play video of a saved LBF checkpoint (MP4 under <job>/videos; never overwrites).

  python -m pgct_lbf.video --job LBF/runs/<name>/pgct_d2_s1 [--checkpoint final.pt] [--stochastic] [--wandb]
"""
import sys

from .experiment import SPEC
from pgct_runner.video import video_main


def main(argv=None):
    return video_main(SPEC, argv)


if __name__ == "__main__":
    sys.exit(main())
