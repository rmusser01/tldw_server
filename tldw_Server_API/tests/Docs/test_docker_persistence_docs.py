"""Regression tests for Docker persistence documentation."""

from pathlib import Path

import pytest


def _require(condition: bool, message: str) -> None:
    """Fail with a descriptive assertion message."""
    if not condition:
        pytest.fail(message)


def test_dockerfiles_readme_documents_persistence_contract() -> None:
    """Dockerfiles README should explain where quickstart data persists."""
    text = Path("Dockerfiles/README.md").read_text(encoding="utf-8")

    for snippet in (
        "app-data",
        "docker-compose.host-storage.yml",
        "docker compose down -v",
        "tldw_Server_API/Config_Files/.env",
        "No nested named volume is mounted under /app/Databases/user_databases",
    ):
        _require(
            snippet in text,
            f"Dockerfiles README should mention {snippet}",
        )


def test_docker_single_user_profile_documents_named_volumes_and_overlay() -> None:
    """Docker single-user setup should document default persistence and the optional overlay."""
    text = Path("Docs/Getting_Started/Profile_Docker_Single_User.md").read_text(
        encoding="utf-8"
    )

    for snippet in (
        "Docker named volumes",
        "docker-compose.host-storage.yml",
        "docker compose down -v",
        "app-data",
        "No nested named volume is mounted under /app/Databases/user_databases",
    ):
        _require(
            snippet in text,
            f"Docker single-user profile should mention {snippet}",
        )


def test_dockerfiles_readme_documents_public_multi_user_admin_bootstrap() -> None:
    """Docker docs should not tell public multi-user users to create admins manually."""
    text = Path("Dockerfiles/README.md").read_text(encoding="utf-8")

    _require(
        "ADMIN_USERNAME" in text and "ADMIN_PASSWORD" in text,
        "Dockerfiles README should document env-driven first admin bootstrap",
    )
    _require(
        "For `multi_user`, run AuthNZ initialization manually to create your admin account"
        not in text,
        "Dockerfiles README should not use stale manual AuthNZ init wording",
    )
