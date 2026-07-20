"""Shared pytest fixtures for the Phase-1 regression harness.

Kept intentionally small: this is the seed Phase 2 will grow. It only provides
a headless matplotlib backend and a pointer to the bundled ``examples/`` data.
"""

from pathlib import Path

import matplotlib

# All workflow modules import matplotlib; force a non-interactive backend so
# tests never try to open a GUI window (mirrors the headless smoke method).
matplotlib.use("Agg")

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"


@pytest.fixture
def examples_dir():
    """Absolute path to the repo's bundled example data (``examples/``)."""
    return EXAMPLES_DIR
