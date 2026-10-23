from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from judges_helpers import SUITE_DIR


@pytest.fixture
def tmp_suite(tmp_path: Path) -> Path:
    """A copy of the spec-writing judge files, safe to edit."""
    d = tmp_path / "suite"
    shutil.copytree(SUITE_DIR / "judges", d / "judges")
    shutil.copytree(SUITE_DIR / "rubrics", d / "rubrics")
    return d
