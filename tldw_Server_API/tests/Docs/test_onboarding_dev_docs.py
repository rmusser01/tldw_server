"""Regression tests for the development-focused onboarding docs."""

from pathlib import Path

import pytest


def _require(condition: bool, message: str) -> None:
    """Fail with a descriptive assertion message when a contract is broken."""
    if not condition:
        pytest.fail(message)


def test_local_single_user_profile_is_marked_for_development() -> None:
    """The local single-user profile should be clearly marked as a dev path."""
    text = Path("Docs/Getting_Started/Profile_Local_Single_User.md").read_text(
        encoding="utf-8"
    )
    _require(
        "development" in text.lower(),
        "Local single-user profile should be marked for development",
    )
    for command in (
        "make install-local",
        "make setup-local-single",
        "make start-local-single",
        "make verify-local-single",
    ):
        _require(command in text, f"Local single-user profile should include {command}")
    _require(
        "make quickstart-local" not in text,
        "Local single-user profile should not use make quickstart-local as the run command",
    )


def test_development_guide_links_local_dev_to_prod_paths() -> None:
    """Development guide should point local contributors to the right paths."""
    text = Path("apps/DEVELOPMENT.md").read_text(encoding="utf-8")
    _require("quickstart-install" in text, "Development guide should mention quickstart-install")
    _require("dev:webpack" in text, "Development guide should mention the webpack fallback")
    _require(
        "bun run --cwd apps/tldw-frontend dev" in text,
        "Development guide should use the correct frontend workspace path",
    )
    _require(
        "Docs/Getting_Started/Profile_Docker_Single_User.md" in text,
        "Development guide should link to the Docker single-user profile",
    )
    _require(
        "Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md" in text,
        "Development guide should link to the multi-user + Postgres profile",
    )


def test_website_quick_start_defaults_to_make_quickstart() -> None:
    """Website quick start should lead with make quickstart and keep the dev path secondary."""
    text = Path("Docs/Website/index.html").read_text(encoding="utf-8")
    _require(
        "Shortest alias: Docker single-user + WebUI" in text,
        "Website quick start should label the shortest Docker single-user + WebUI alias",
    )
    _require("make quickstart" in text, "Website quick start should mention make quickstart")
    _require(
        "make install-local" in text,
        "Website quick start should mention the local single-user install path",
    )
    _require(
        "Profile_Docker_Multi_User_Postgres.md" in text,
        "Website quick start should link to the multi-user + Postgres guide",
    )
    _require(
        text.index("make quickstart") < text.index("make install-local"),
        "Website quick start should present make quickstart before local development commands",
    )
