"""Lint checks for the Search_and_Research package marker and docs."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _repo_root() -> Path:
    """Return the repository root for path-based lint checks."""
    return Path(__file__).resolve().parents[3]


def test_search_and_research_package_marker_uses_canonical_casing() -> None:
    """Require canonical package marker casing in Search_and_Research."""
    package_dir = _repo_root() / "tldw_Server_API" / "app" / "core" / "Search_and_Research"
    filenames = {path.name for path in package_dir.iterdir() if path.is_file()}

    assert "__init__.py" in filenames, "Search_and_Research must use canonical __init__.py casing"  # nosec B101
    assert "__Init__.py" not in filenames, "Search_and_Research must not keep non-canonical __Init__.py"  # nosec B101


def test_search_and_research_readme_uses_canonical_package_marker_name() -> None:
    """Require the README to mention the canonical marker without stale casing."""
    readme_path = (
        _repo_root()
        / "tldw_Server_API"
        / "app"
        / "core"
        / "Search_and_Research"
        / "README.md"
    )

    readme = readme_path.read_text(encoding="utf-8")

    assert "__Init__.py" not in readme, "Search_and_Research README must not mention __Init__.py"  # nosec B101
    assert "__init__.py" in readme, "Search_and_Research README must mention the canonical marker"  # nosec B101
