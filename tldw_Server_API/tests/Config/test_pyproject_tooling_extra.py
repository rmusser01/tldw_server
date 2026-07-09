"""Pyproject contract tests for packaging and tooling metadata."""

from pathlib import Path

import pytest

try:
    import tomllib  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


pytestmark = pytest.mark.unit


def test_pyproject_has_tooling_optional_dependency_group() -> None:
    """The tooling extra must remain available for helper scripts."""
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    optional = data["project"]["optional-dependencies"]
    assert "tooling" in optional  # nosec B101
    assert any(dep.startswith("requests") for dep in optional["tooling"])  # nosec B101


def test_pyproject_packages_db_migration_sql() -> None:
    """Installed packages must include DB migration SQL files."""
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    package_data = data["tool"]["setuptools"]["package-data"]["tldw_Server_API"]
    assert "app/core/DB_Management/migrations/*.sql" in package_data  # nosec B101
