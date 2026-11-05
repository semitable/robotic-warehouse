import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))           # VMAS/ (pgct_vmas)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "PGCT"))  # pgct_core, pgct_runner
