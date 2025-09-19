# conftest.py
from pathlib import Path
import pytest

ROOT_SENTINELS = ("pyproject.toml", "setup.cfg", "setup.py", ".git")

def _find_project_root(start: Path) -> Path:
    # Walk up until we find a typical project root sentinel; fallback to start
    for p in [start, *start.parents]:
        if any((p / s).exists() for s in ROOT_SENTINELS):
            return p
    return start

def _big_tests_exists() -> bool:
    here = Path(__file__).resolve()
    root = _find_project_root(here)
    return (root / "big_tests").is_dir()

def pytest_collection_modifyitems(config, items):
    # If no big_tests/ directory, skip anything marked big_test
    if _big_tests_exists():
        return
    skip_marker = pytest.mark.skip(reason="Skipped because project has no 'big_tests/' directory.")
    for item in items:
        if "big_test" in item.keywords:
            item.add_marker(skip_marker)
