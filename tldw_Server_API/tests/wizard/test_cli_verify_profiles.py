from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from tldw_Server_API.cli.wizard import cli as wizard_cli
from tldw_Server_API.cli.wizard import profile_verify
from tldw_Server_API.tests.wizard.helpers import assert_action_field, assert_wizard_json

runner = CliRunner()


def test_docker_profile_verify_does_not_spawn_ephemeral_server(monkeypatch) -> None:
    def fake_run_checks(*, profile, base_url, webui_url, env_path, first_value, timeout):
        return {
            "status": "ok",
            "actions": [
                {"server": {"mode": "existing", "profile": profile.name}},
                {"endpoints": {"health": {"ok": True}, "ready": {"ok": True}, "docs": {"ok": True}}},
            ],
            "notes": [],
        }

    def start_ephemeral(*_args, **_kwargs):
        raise AssertionError("docker profile verify must not spawn a local server")

    monkeypatch.setattr(profile_verify, "run_profile_checks", fake_run_checks)
    monkeypatch.setattr(wizard_cli, "_start_ephemeral_server", start_ephemeral)

    result = runner.invoke(wizard_cli.app, ["verify", "--profile", "docker-single-webui", "--json"])

    assert result.exit_code == 0, result.output
    payload = assert_wizard_json(result.output, command="verify", status="ok")
    actions = payload.get("actions") or []
    assert_action_field(actions, "server", "profile", "docker-single-webui")


def test_verify_first_value_reports_provider_missing(monkeypatch, tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("AUTH_MODE=single_user\nSINGLE_USER_API_KEY=tldw_test.key\n", encoding="utf-8")

    def fake_run_checks(*, profile, base_url, webui_url, env_path, first_value, timeout):
        return {
            "status": "ok",
            "actions": [
                {"chat": {"status": "provider_missing", "env_examples": ["OPENAI_API_KEY=sk-..."]}},
                {"first_value": {"ingest": "ok", "search": "ok"}},
            ],
            "notes": ["No provider key configured; chat verification skipped."],
        }

    monkeypatch.setattr(profile_verify, "run_profile_checks", fake_run_checks)

    result = runner.invoke(
        wizard_cli.app,
        [
            "verify",
            "--profile",
            "local-single",
            "--env-file",
            str(env_path),
            "--first-value",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = assert_wizard_json(result.output, command="verify", status="ok")
    actions = payload.get("actions") or []
    assert_action_field(actions, "chat", "status", "provider_missing")
    assert_action_field(actions, "first_value", "search", "ok")
