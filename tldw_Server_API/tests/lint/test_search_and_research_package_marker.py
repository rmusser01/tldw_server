from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_search_and_research_package_marker_uses_canonical_casing():
    package_dir = _repo_root() / "tldw_Server_API" / "app" / "core" / "Search_and_Research"
    filenames = {path.name for path in package_dir.iterdir() if path.is_file()}

    if "__init__.py" not in filenames:
        raise AssertionError("Search_and_Research must use canonical __init__.py casing")
    if "__Init__.py" in filenames:
        raise AssertionError("Search_and_Research must not keep the non-canonical __Init__.py")


def test_search_and_research_readme_uses_canonical_package_marker_name():
    readme_path = (
        _repo_root()
        / "tldw_Server_API"
        / "app"
        / "core"
        / "Search_and_Research"
        / "README.md"
    )

    readme = readme_path.read_text(encoding="utf-8")

    if "__Init__.py" in readme:
        raise AssertionError("Search_and_Research README must not mention __Init__.py")
    if "`__init__.py`" not in readme:
        raise AssertionError("Search_and_Research README must mention the canonical marker")
