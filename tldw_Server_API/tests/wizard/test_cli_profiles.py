from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.cli.wizard import profiles


def test_normalize_profile_accepts_public_names() -> None:
    assert profiles.normalize_profile("docker-single-webui").name == "docker-single-webui"
    assert profiles.normalize_profile("docker-multi-postgres").auth_mode == "multi_user"
    assert profiles.normalize_profile("local-single").auth_mode == "single_user"


def test_normalize_profile_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unsupported setup profile"):
        profiles.normalize_profile("docker-team")


def test_repo_checkout_env_defaults_to_config_files(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "tldw_Server_API" / "Config_Files").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='tldw-server'\n", encoding="utf-8")

    env_path = profiles.resolve_env_path(
        profile=profiles.normalize_profile("docker-single-webui"),
        start_dir=root / "Docs",
        explicit_env_file=None,
    )

    assert env_path == root / "tldw_Server_API" / "Config_Files" / ".env"


def test_explicit_env_file_overrides_repo_default(tmp_path: Path) -> None:
    explicit = tmp_path / "custom.env"

    env_path = profiles.resolve_env_path(
        profile=profiles.normalize_profile("local-single"),
        start_dir=tmp_path,
        explicit_env_file=explicit,
    )

    assert env_path == explicit


def test_single_user_defaults_generate_maskable_api_key() -> None:
    defaults = profiles.build_profile_env(
        profile=profiles.normalize_profile("local-single"),
        existing_env={},
    )

    assert defaults["AUTH_MODE"] == "single_user"
    assert defaults["SINGLE_USER_API_KEY"].startswith("tldw_")
    assert "DATABASE_URL" in defaults


def test_multi_user_defaults_include_required_secrets() -> None:
    defaults = profiles.build_profile_env(
        profile=profiles.normalize_profile("docker-multi-postgres"),
        existing_env={},
        admin_username="admin",
        admin_password="CorrectHorseBatteryStaple1!",
        admin_email="admin@example.com",
    )

    for key in (
        "AUTH_MODE",
        "DATABASE_URL",
        "JWT_SECRET_KEY",
        "SESSION_ENCRYPTION_KEY",
        "MCP_JWT_SECRET",
        "MCP_API_KEY_SALT",
        "BYOK_ENCRYPTION_KEY",
        "ADMIN_USERNAME",
        "ADMIN_PASSWORD",
        "ADMIN_EMAIL",
    ):
        assert defaults[key]
    assert defaults["AUTH_MODE"] == "multi_user"
    assert defaults["DATABASE_URL"].startswith("postgresql://")
