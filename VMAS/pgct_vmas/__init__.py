"""VMAS Team Discovery experiment for PGCT (needs the optional `vmas` package; uses pgct_core and pgct_runner)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if importlib.util.find_spec("pgct_core") is None or importlib.util.find_spec("pgct_runner") is None:
    sys.path.insert(0, str(REPO_ROOT / "PGCT"))  # not pip-installed: use <repo>/PGCT
