from __future__ import annotations

from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from typer.testing import CliRunner

from tldw_Server_API.cli.wizard.cli import app
from tldw_Server_API.cli.wizard import profiles
from tldw_Server_API.tests.wizard.helpers import assert_action_field, assert_wizard_json


runner = CliRunner()


def _literal(*parts: str) -> str:
    return "".join(parts)


def test_normalize_profile_accepts_public_names() -> None:
    assert profiles.normalize_profile("docker-single-webui").name == "docker-single-webui"
    assert profiles.normalize_profile("docker-multi-postgres").auth_mode == "multi_user"
    assert profiles.normalize_profile("local-single").auth_mode == "single_user"


def test_normalize_profile_accepts_aliases() -> None:
    assert profiles.normalize_profile("docker-single").name == "docker-single-webui"
    assert profiles.normalize_profile("docker-webui").name == "docker-single-webui"
    assert profiles.normalize_profile("docker-multi").name == "docker-multi-postgres"
    assert profiles.normalize_profile("local").name == "local-single"


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
        admin_password=_literal("CorrectHorse", "BatteryStaple", "1!"),
        admin_email="admin@example.com",
    )

    for key in (
        "AUTH_MODE",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
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
    assert defaults["DATABASE_URL"] == ""
    assert defaults["JOBS_DB_URL"] == ""
    assert defaults["POSTGRES_USER"] == "tldw_user"
    assert defaults["POSTGRES_DB"] == "tldw_users"
    assert "TestPassword123!" not in "\n".join(defaults.values())


def test_multi_user_defaults_preserve_existing_postgres_credentials() -> None:
    pg_value = _literal("ExistingPostgres", "Password", "1!")
    defaults = profiles.build_profile_env(
        profile=profiles.normalize_profile("docker-multi-postgres"),
        existing_env={
            "POSTGRES_USER": "custom_user",
            "POSTGRES_PASSWORD": pg_value,
            "POSTGRES_DB": "custom_db",
        },
    )

    assert defaults["POSTGRES_USER"] == "custom_user"
    assert defaults["POSTGRES_PASSWORD"] == pg_value
    assert defaults["POSTGRES_DB"] == "custom_db"
    assert defaults["DATABASE_URL"] == ""
    assert defaults["JOBS_DB_URL"] == ""


def test_multi_user_defaults_url_quote_reserved_postgres_credentials() -> None:
    pg_value = _literal("abc", "@def", ":ghi", "/with", "#chars", "%")
    expected_url = "postgresql://custom%3Auser:abc%40def%3Aghi%2Fwith%23chars%25@postgres:5432/custom%2Fdb%20%231"
    assert (
        profiles.build_postgres_database_url(
            user="custom:user",
            password=pg_value,
            db="custom/db #1",
        )
        == expected_url
    )


def test_multi_user_session_key_is_fernet_compatible() -> None:
    defaults = profiles.build_profile_env(
        profile=profiles.normalize_profile("docker-multi-postgres"),
        existing_env={},
    )

    Fernet(defaults["SESSION_ENCRYPTION_KEY"])


def test_invalid_existing_session_key_is_regenerated() -> None:
    defaults = profiles.build_profile_env(
        profile=profiles.normalize_profile("docker-multi-postgres"),
        existing_env={"SESSION_ENCRYPTION_KEY": "abc"},
    )

    assert defaults["SESSION_ENCRYPTION_KEY"] != "abc"
    Fernet(defaults["SESSION_ENCRYPTION_KEY"])


@pytest.mark.parametrize(
    "key",
    (
        "SINGLE_USER_API_KEY",
        "JWT_SECRET_KEY",
        "SESSION_ENCRYPTION_KEY",
        "MCP_JWT_SECRET",
        "MCP_API_KEY_SALT",
        "BYOK_ENCRYPTION_KEY",
    ),
)
@pytest.mark.parametrize(
    "placeholder",
    (
        "",
        "change-me",
        "changeme",
        "default",
        "test-key",
        "CHANGE_ME_SECRET",
        "replace-with-real-secret",
    ),
)
def test_placeholder_secret_values_are_regenerated(key: str, placeholder: str) -> None:
    profile = profiles.normalize_profile("local-single" if key == "SINGLE_USER_API_KEY" else "docker-multi-postgres")

    defaults = profiles.build_profile_env(
        profile=profile,
        existing_env={key: placeholder},
    )

    assert defaults[key]
    assert defaults[key] != placeholder


def test_init_profile_writes_repo_env_path_in_dry_run(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    (repo / "tldw_Server_API" / "Config_Files").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='tldw-server'\n", encoding="utf-8")
    monkeypatch.chdir(repo)

    result = runner.invoke(
        app,
        ["init", "--profile", "docker-single-webui", "--dry-run", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = assert_wizard_json(result.output, command="init", status="ok")
    assert payload["paths"]["env"].endswith("tldw_Server_API/Config_Files/.env")
    actions = payload.get("actions") or []
    set_env = next(action["set_env"] for action in actions if "set_env" in action)
    assert_action_field(actions, "set_env", "AUTH_MODE", "single_user")
    assert str(set_env["SINGLE_USER_API_KEY"]).startswith("*")


def test_init_multi_user_profile_masks_admin_password_in_dry_run(
    tmp_path: Path, monkeypatch
) -> None:
    admin_pw = _literal("CorrectHorse", "BatteryStaple", "1!")
    repo = tmp_path / "repo"
    (repo / "tldw_Server_API" / "Config_Files").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='tldw-server'\n", encoding="utf-8")
    monkeypatch.chdir(repo)

    result = runner.invoke(
        app,
        [
            "init",
            "--profile",
            "docker-multi-postgres",
            "--admin-username",
            "admin",
            "--admin-password",
            admin_pw,
            "--admin-email",
            "admin@example.com",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "TestPassword123!" not in result.output
    assert admin_pw not in result.output
    payload = assert_wizard_json(result.output, command="init", status="ok")
    actions = payload.get("actions") or []
    set_env = next(action["set_env"] for action in actions if "set_env" in action)
    assert_action_field(actions, "set_env", "AUTH_MODE", "multi_user")
    assert str(set_env["SESSION_ENCRYPTION_KEY"]).startswith("*")
    assert_action_field(actions, "set_env", "ADMIN_USERNAME", "admin")
    assert str(set_env["ADMIN_PASSWORD"]).startswith("*")
    assert str(set_env["POSTGRES_PASSWORD"]).startswith("*")
    assert set_env["DATABASE_URL"] == ""
    assert set_env["JOBS_DB_URL"] == ""


def test_init_multi_user_profile_reads_admin_env_and_masks_password(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    (repo / "tldw_Server_API" / "Config_Files").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='tldw-server'\n", encoding="utf-8")
    monkeypatch.chdir(repo)
    monkeypatch.setenv("ADMIN_USERNAME", "env-admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "EnvPasswordWith$Dollar1!")
    monkeypatch.setenv("ADMIN_EMAIL", "env-admin@example.com")

    result = runner.invoke(
        app,
        [
            "init",
            "--profile",
            "docker-multi-postgres",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "EnvPasswordWith$Dollar1!" not in result.output
    payload = assert_wizard_json(result.output, command="init", status="ok")
    actions = payload.get("actions") or []
    set_env = next(action["set_env"] for action in actions if "set_env" in action)
    assert_action_field(actions, "set_env", "ADMIN_USERNAME", "env-admin")
    assert_action_field(actions, "set_env", "ADMIN_EMAIL", "env-admin@example.com")
    assert str(set_env["ADMIN_PASSWORD"]).startswith("*")


def test_init_invalid_profile_returns_json_error() -> None:
    result = runner.invoke(
        app,
        ["init", "--profile", "does-not-exist", "--dry-run", "--json"],
    )

    assert result.exit_code == 2, result.output
    payload = assert_wizard_json(result.output, command="init", status="error")
    actions = payload.get("actions") or []
    assert_action_field(actions, "profile", "valid", False)
    assert "Unsupported setup profile" in actions[0]["profile"]["reason"]
    assert "Unsupported setup profile" in payload["notes"][0]


def test_init_docker_multi_profile_defers_initializer_for_yes(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    env_path = tmp_path / "custom.env"
    (repo / "tldw_Server_API" / "Config_Files").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='tldw-server'\n", encoding="utf-8")

    def fail_run(*_args, **_kwargs):
        raise AssertionError("docker profile must not run local AuthNZ initializer")

    monkeypatch.setattr("tldw_Server_API.cli.wizard.cli.subprocess.run", fail_run)

    result = runner.invoke(
        app,
        [
            "init",
            "--install-dir",
            str(repo),
            "--profile",
            "docker-multi-postgres",
            "--env-file",
            str(env_path),
            "--yes",
            "--no-format",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = assert_wizard_json(result.output, command="init", status="ok")
    actions = payload.get("actions") or []
    initializer = next(action["authnz_initializer"] for action in actions if "authnz_initializer" in action)
    assert initializer["status"] == "deferred_to_docker"
    assert "inside the container" in initializer["reason"]
    written_env = profiles.env_utils.load_env(env_path)
    assert written_env["POSTGRES_PASSWORD"]
    assert written_env["DATABASE_URL"] == ""
    assert written_env["JOBS_DB_URL"] == ""


def test_init_docker_multi_profile_clears_stale_database_urls(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    env_path = tmp_path / "custom.env"
    (repo / "tldw_Server_API" / "Config_Files").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='tldw-server'\n", encoding="utf-8")
    env_path.write_text(
        "DATABASE_URL=postgresql://old:stale@postgres:5432/old\n"
        "JOBS_DB_URL=postgresql://old:stale@postgres:5432/old\n",
        encoding="utf-8",
    )

    def fail_run(*_args, **_kwargs):
        raise AssertionError("docker profile must not run local AuthNZ initializer")

    monkeypatch.setattr("tldw_Server_API.cli.wizard.cli.subprocess.run", fail_run)

    result = runner.invoke(
        app,
        [
            "init",
            "--install-dir",
            str(repo),
            "--profile",
            "docker-multi-postgres",
            "--env-file",
            str(env_path),
            "--admin-username",
            "admin",
            "--admin-password",
            _literal("CorrectHorse", "BatteryStaple", "1!"),
            "--yes",
            "--no-format",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    written_env = profiles.env_utils.load_env(env_path)
    assert written_env["POSTGRES_PASSWORD"]
    assert written_env["DATABASE_URL"] == ""
    assert written_env["JOBS_DB_URL"] == ""


def test_init_docker_multi_profile_writes_raw_env_values_for_compose(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    env_path = tmp_path / "custom.env"
    admin_secret = _literal("Admin ", "#", "$Dollar", "1!")
    postgres_secret = _literal("abc", "#def", "$ghi")
    (repo / "tldw_Server_API" / "Config_Files").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='tldw-server'\n", encoding="utf-8")
    env_path.write_text(f"POSTGRES_PASSWORD={postgres_secret}\n", encoding="utf-8")

    def fail_run(*_args, **_kwargs):
        raise AssertionError("docker profile must not run local AuthNZ initializer")

    monkeypatch.setattr("tldw_Server_API.cli.wizard.cli.subprocess.run", fail_run)

    result = runner.invoke(
        app,
        [
            "init",
            "--install-dir",
            str(repo),
            "--profile",
            "docker-multi-postgres",
            "--env-file",
            str(env_path),
            "--admin-username",
            "admin",
            "--admin-password",
            admin_secret,
            "--yes",
            "--no-format",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    content = env_path.read_text(encoding="utf-8")
    assert f"POSTGRES_PASSWORD={postgres_secret}\n" in content
    assert f"ADMIN_PASSWORD={admin_secret}\n" in content
    assert f'POSTGRES_PASSWORD="{postgres_secret}"' not in content
    assert f'ADMIN_PASSWORD="{admin_secret}"' not in content
