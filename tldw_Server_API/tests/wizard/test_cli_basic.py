from __future__ import annotations

from types import SimpleNamespace

from typer.testing import CliRunner

from tldw_Server_API.cli.wizard import cli as wizard_cli
from tldw_Server_API.cli.wizard.cli import app
from tldw_Server_API.tests.wizard.helpers import (
    assert_action_field,
    assert_action_fields,
    assert_wizard_error,
    assert_wizard_json,
)


runner = CliRunner()


def test_init_dry_run_json():
    result = runner.invoke(app, ["init", "--dry-run", "--json"])  # type: ignore[arg-type]
    assert result.exit_code == 0, result.output
    data = assert_wizard_json(result.output, command="init", status="ok")
    assert "facts" in data
    assert "actions" in data


def test_init_dry_run_honors_env_file_without_profile(tmp_path):
    env_path = tmp_path / "custom.env"

    result = runner.invoke(app, ["init", "--env-file", str(env_path), "--dry-run", "--json"])  # type: ignore[arg-type]

    assert result.exit_code == 0, result.output
    data = assert_wizard_json(result.output, command="init", status="ok")
    assert data["paths"]["env"] == str(env_path)


def test_init_dry_run_env_file_preserves_single_user_key_over_shell(tmp_path):
    env_path = tmp_path / "custom.env"
    env_path.write_text("AUTH_MODE=single_user\nSINGLE_USER_API_KEY=file-key-1234567890\n", encoding="utf-8")

    result = runner.invoke(
        app,
        ["init", "--env-file", str(env_path), "--dry-run", "--json"],
        env={"SINGLE_USER_API_KEY": "shell-key-should-not-write"},
    )

    assert result.exit_code == 0, result.output
    assert "shell-key-should-not-write" not in result.output
    payload = assert_wizard_json(result.output, command="init", status="ok")
    actions = payload.get("actions") or []
    set_env = next(action["set_env"] for action in actions if "set_env" in action)
    assert set_env["SINGLE_USER_API_KEY"] == "***************7890"


def test_auth_single_user_json():
    result = runner.invoke(app, ["auth", "--mode", "single_user", "--json", "--dry-run"])  # type: ignore[arg-type]
    assert result.exit_code == 0, result.output
    data = assert_wizard_json(result.output, command="auth", status="ok")
    assert data.get("mode") == "single_user"


def test_verify_json(monkeypatch):
    def probe(url: str, path: str, *, timeout: float = 2.0):
        return {"url": f"{url}{path}", "status_code": 200, "ok": True}

    monkeypatch.setattr(wizard_cli, "_probe_endpoint", probe)

    result = runner.invoke(app, ["verify", "--json"])  # type: ignore[arg-type]
    assert result.exit_code == 0, result.output
    data = assert_wizard_json(result.output, command="verify", status="ok")
    assert "facts" in data


def test_verify_dry_run_json(monkeypatch):
    def probe(*_args, **_kwargs):
        raise AssertionError("unexpected probe in dry-run")

    def start_ephemeral(*_args, **_kwargs):
        raise AssertionError("unexpected server spawn in dry-run")

    monkeypatch.setattr(wizard_cli, "_probe_endpoint", probe)
    monkeypatch.setattr(wizard_cli, "_start_ephemeral_server", start_ephemeral)

    result = runner.invoke(app, ["verify", "--json", "--dry-run"])  # type: ignore[arg-type]
    assert result.exit_code == 0, result.output
    payload = assert_wizard_json(result.output, command="verify", status="ok")
    assert payload.get("dry_run") is True
    actions = payload.get("actions") or []
    assert_action_field(actions, "server", "mode", "dry_run")


def test_init_multi_user_missing_database_url_errors():
    with runner.isolated_filesystem():
        result = runner.invoke(
            app,
            ["init", "--dry-run", "--json"],
            env={"AUTH_MODE": "multi_user", "DATABASE_URL": ""},
        )
        assert result.exit_code == 2, result.output
        payload = assert_wizard_json(result.output, command="init", status="error")
        assert_wizard_error(payload, action_key="validate_database_url")
        actions = payload.get("actions") or []
        assert_action_fields(actions, "validate_database_url", {"present": False, "valid": False})


def test_init_local_initializer_uses_resolved_env_file_values(tmp_path, monkeypatch):
    env_path = tmp_path / "custom.env"
    env_path.write_text(
        "AUTH_MODE=multi_user\nDATABASE_URL=postgresql://custom_user:secret@localhost:5432/custom_db\n",
        encoding="utf-8",
    )
    seen: dict[str, str] = {}

    def fake_run(_cmd, *, check, env=None):
        assert check is False
        assert env is not None
        seen.update(env)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(wizard_cli.subprocess, "run", fake_run)

    result = runner.invoke(
        app,
        ["init", "--env-file", str(env_path), "--yes", "--no-format", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert seen["AUTH_MODE"] == "multi_user"
    assert seen["DATABASE_URL"] == "postgresql://custom_user:secret@localhost:5432/custom_db"
