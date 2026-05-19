from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from tldw_Server_API.cli.wizard import cli as wizard_cli
from tldw_Server_API.cli.wizard.utils import env as env_utils
from tldw_Server_API.tests.wizard.helpers import assert_action_field, assert_action_fields, assert_wizard_json


runner = CliRunner()


def test_doctor_dry_run_recommends_env_gitignore(monkeypatch):
    monkeypatch.delenv("AUTH_MODE", raising=False)
    monkeypatch.delenv("SINGLE_USER_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setattr(env_utils, "generate_single_user_api_key", lambda: "sk-test-0000")
    monkeypatch.setattr(wizard_cli.detect_utils, "has_ffmpeg", lambda: False)
    monkeypatch.setattr(wizard_cli, "_port_available", lambda _port: True)

    with runner.isolated_filesystem():
        result = runner.invoke(wizard_cli.app, ["doctor", "--json", "--dry-run"])  # type: ignore[arg-type]
        assert result.exit_code == 0, result.output
        payload = assert_wizard_json(result.output, command="doctor", status="ok")
        actions = payload.get("actions") or []
        assert_action_field(actions, "env", "status", "would_update")
        assert_action_fields(
            actions,
            "set_env",
            {
                "AUTH_MODE": "single_user",
                "SINGLE_USER_API_KEY": env_utils.mask_value("sk-test-0000"),
            },
        )
        assert_action_field(actions, "gitignore", "status", "would_update")
        assert_action_field(actions, "ffmpeg", "status", "missing")
        assert not Path(".env").exists()
        assert not Path(".gitignore").exists()


def test_doctor_yes_applies_env_gitignore(monkeypatch):
    monkeypatch.delenv("AUTH_MODE", raising=False)
    monkeypatch.delenv("SINGLE_USER_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setattr(env_utils, "generate_single_user_api_key", lambda: "sk-test-1111")
    monkeypatch.setattr(wizard_cli.detect_utils, "has_ffmpeg", lambda: True)
    monkeypatch.setattr(wizard_cli, "_port_available", lambda _port: True)

    with runner.isolated_filesystem():
        result = runner.invoke(wizard_cli.app, ["doctor", "--json", "--yes"])  # type: ignore[arg-type]
        assert result.exit_code == 0, result.output
        env_path = Path(".env")
        assert env_path.exists()
        content = env_path.read_text(encoding="utf-8")
        assert "AUTH_MODE=single_user" in content
        assert "SINGLE_USER_API_KEY=sk-test-1111" in content
        gitignore = Path(".gitignore")
        assert gitignore.exists()
        gitignore_content = gitignore.read_text(encoding="utf-8")
        assert ".env" in gitignore_content
        assert ".env.*.bak" in gitignore_content
        assert ".env.local" in gitignore_content
        assert "wizard.log" in gitignore_content


def test_doctor_flags_invalid_database_url(monkeypatch):
    monkeypatch.setattr(wizard_cli.detect_utils, "has_ffmpeg", lambda: True)
    monkeypatch.setattr(wizard_cli, "_port_available", lambda _port: True)

    result = runner.invoke(
        wizard_cli.app,
        ["doctor", "--json"],
        env={"AUTH_MODE": "multi_user", "DATABASE_URL": "sqlite:///./Databases/users.db"},
    )  # type: ignore[arg-type]
    assert result.exit_code == 0, result.output
    payload = assert_wizard_json(result.output, command="doctor", status="ok")
    actions = payload.get("actions") or []
    assert_action_fields(actions, "validate_database_url", {"present": True, "valid": False})


def test_doctor_recommends_port_change(monkeypatch):
    monkeypatch.setattr(env_utils, "generate_single_user_api_key", lambda: "sk-test-2222")
    monkeypatch.setattr(wizard_cli.detect_utils, "has_ffmpeg", lambda: True)
    monkeypatch.setattr(wizard_cli, "_port_available", lambda _port: False)
    monkeypatch.setattr(wizard_cli, "_pick_free_port", lambda: 8123)

    result = runner.invoke(wizard_cli.app, ["doctor", "--json", "--dry-run"])  # type: ignore[arg-type]
    assert result.exit_code == 0, result.output
    payload = assert_wizard_json(result.output, command="doctor", status="ok")
    actions = payload.get("actions") or []
    assert_action_field(actions, "port", "suggested_port", 8123)
    assert_action_field(actions, "set_env", "TLDW_SERVER_PORT", "8123")
