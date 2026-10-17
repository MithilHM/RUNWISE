"""Pytest bootstrap: make ``app`` and ``tests`` importable.

The integration owner owns ``backend/pyproject.toml`` and the pytest configuration, so this
file keeps the suite runnable with a bare ``pytest`` from ``backend/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))