"""Make the shared ``pgct_core`` package importable without installation.

Repository layout: <repo>/PGCT/pgct_core (shared algorithm) and <repo>/RWARE/latent_team (this package).
If ``pip install -e <repo>`` was run, the installed package is used; otherwise <repo>/PGCT is put on sys.path.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

CORE_ROOT = Path(__file__).resolve().parents[2] / "PGCT"

if importlib.util.find_spec("pgct_core") is None:
    if not (CORE_ROOT / "pgct_core" / "__init__.py").is_file():
        raise ImportError(f"pgct_core not found at {CORE_ROOT}; run from the PGCT repository or pip install -e it")
    sys.path.insert(0, str(CORE_ROOT))
