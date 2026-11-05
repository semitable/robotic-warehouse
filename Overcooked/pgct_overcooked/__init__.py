"""Overcooked-style multi-team kitchen experiment for PGCT (uses the shared pgct_core package)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if importlib.util.find_spec("pgct_core") is None:  # not pip-installed: use <repo>/PGCT
    sys.path.insert(0, str(REPO_ROOT / "PGCT"))
