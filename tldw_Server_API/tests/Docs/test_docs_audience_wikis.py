"""Contract tests for audience-focused documentation wiki entry points."""

from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _require(condition: bool, message: str) -> None:
    """Fail with a descriptive assertion message when a docs contract is broken."""
    if not condition:
        pytest.fail(message)


def test_docs_audience_wiki_source_and_published_pages_exist() -> None:
    """The docs site should expose source and generated audience wiki pages."""
    for relative_path in (
        "Docs/Wiki/index.md",
        "Docs/Wiki/User_Wiki.md",
        "Docs/Wiki/Developer_Wiki.md",
        "Docs/Published/Wiki/index.md",
        "Docs/Published/Wiki/User_Wiki.md",
        "Docs/Published/Wiki/Developer_Wiki.md",
    ):
        _require((REPO_ROOT / relative_path).is_file(), f"Missing {relative_path}")


def test_mkdocs_nav_exposes_audience_wikis() -> None:
    """MkDocs navigation should make the audience split visible at top level."""
    mkdocs_text = _read("Docs/mkdocs.yml")

    _require("Home: Wiki/index.md" in mkdocs_text, "MkDocs nav should use Wiki home")
    _require("User Wiki:" in mkdocs_text, "MkDocs nav should expose User Wiki")
    _require("Developer Wiki:" in mkdocs_text, "MkDocs nav should expose Developer Wiki")
    _require(
        "Start Here: Wiki/User_Wiki.md" in mkdocs_text,
        "User Wiki nav should start at the user wiki page",
    )
    _require(
        "Start Here: Wiki/Developer_Wiki.md" in mkdocs_text,
        "Developer Wiki nav should start at the developer wiki page",
    )


def test_readme_points_users_and_contributors_to_wikis() -> None:
    """README should route readers to the right documentation audience entry."""
    readme_text = _read("README.md")

    _require("Docs/Wiki/User_Wiki.md" in readme_text, "README should link User Wiki")
    _require(
        "Docs/Wiki/Developer_Wiki.md" in readme_text,
        "README should link Developer Wiki",
    )
    _require("User Wiki" in readme_text, "README should label the User Wiki")
    _require("Developer Wiki" in readme_text, "README should label the Developer Wiki")
