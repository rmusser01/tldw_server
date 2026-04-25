from __future__ import annotations

import json
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
    assert payload.get("check_provider") is False
    actions = payload.get("actions") or []
    assert_action_field(actions, "server", "profile", "docker-single-webui")


def test_verify_invalid_profile_json_exits_2() -> None:
    result = runner.invoke(
        wizard_cli.app,
        ["verify", "--profile", "does-not-exist", "--dry-run", "--json"],
    )

    assert result.exit_code == 2, result.output
    payload = assert_wizard_json(result.output, command="verify", status="error")
    actions = payload.get("actions") or []
    assert_action_field(actions, "profile", "valid", False)


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


def test_provider_check_reports_missing_when_provider_entries_are_unconfigured(monkeypatch) -> None:
    def fake_request(*_args, **_kwargs):
        return {
            "url": "http://127.0.0.1:8000/api/v1/llm/providers",
            "status_code": 200,
            "ok": True,
            "body": {
                "total_configured": 2,
                "providers": [
                    {"name": "openai", "is_configured": False},
                    {"name": "anthropic", "is_configured": False},
                ],
            },
        }

    monkeypatch.setattr(profile_verify, "_request", fake_request)

    result = profile_verify._provider_check("http://127.0.0.1:8000", {}, 5.0)

    assert result["status"] == "provider_missing"
    assert result["configured"] == 0


def test_provider_check_reports_configured_when_any_provider_entry_is_configured(monkeypatch) -> None:
    def fake_request(*_args, **_kwargs):
        return {
            "url": "http://127.0.0.1:8000/api/v1/llm/providers",
            "status_code": 200,
            "ok": True,
            "body": {
                "total_configured": 2,
                "providers": {
                    "openai": {"is_configured": False},
                    "anthropic": {"is_configured": True},
                },
            },
        }

    monkeypatch.setattr(profile_verify, "_request", fake_request)

    result = profile_verify._provider_check("http://127.0.0.1:8000", {}, 5.0)

    assert result["status"] == "provider_configured"
    assert result["configured"] == 1


def test_first_value_empty_search_results_fail_without_raw_body(monkeypatch) -> None:
    def fake_request(method, _base_url, path, **_kwargs):
        if path == "/api/v1/media/add":
            return {
                "url": "http://127.0.0.1:8000/api/v1/media/add",
                "status_code": 200,
                "ok": True,
                "body": {"id": 1, "secret": "SECRET_SHOULD_NOT_LEAK"},
            }
        if path == "/api/v1/media/search":
            return {
                "url": "http://127.0.0.1:8000/api/v1/media/search",
                "status_code": 200,
                "ok": True,
                "body": {"results": []},
            }
        raise AssertionError(f"unexpected request {method} {path}")

    monkeypatch.setattr(profile_verify, "_request", fake_request)

    result = profile_verify._first_value_check("http://127.0.0.1:8000", {}, 5.0)

    assert result["ok"] is False
    assert result["search"] == "error"
    assert result["details"]["search"]["matched"] is False
    assert "body" not in result["details"]["ingest"]
    assert "body" not in result["details"]["search"]


def test_first_value_matching_search_results_pass(monkeypatch) -> None:
    def fake_request(method, _base_url, path, **_kwargs):
        if path == "/api/v1/media/add":
            return {
                "url": "http://127.0.0.1:8000/api/v1/media/add",
                "status_code": 200,
                "ok": True,
                "body": {"id": 1},
            }
        if path == "/api/v1/media/search":
            return {
                "url": "http://127.0.0.1:8000/api/v1/media/search",
                "status_code": 200,
                "ok": True,
                "body": {
                    "results": [
                        {
                            "title": "tldw onboarding verification",
                            "content": "contains tldw-onboarding-verification-unique",
                        }
                    ]
                },
            }
        raise AssertionError(f"unexpected request {method} {path}")

    monkeypatch.setattr(profile_verify, "_request", fake_request)

    result = profile_verify._first_value_check("http://127.0.0.1:8000", {}, 5.0)

    assert result["ok"] is True
    assert result["search"] == "ok"
    assert result["details"]["search"]["matched"] is True


def test_profile_checks_do_not_emit_raw_response_bodies(monkeypatch, tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("AUTH_MODE=single_user\nSINGLE_USER_API_KEY=tldw_test.key\n", encoding="utf-8")

    def fake_request(method, _base_url, path, **_kwargs):
        response = {
            "url": f"http://127.0.0.1:8000{path}",
            "status_code": 200,
            "ok": True,
            "body": {"debug": "SECRET_SHOULD_NOT_LEAK"},
        }
        if path == "/api/v1/llm/providers":
            response["body"] = {
                "total_configured": 0,
                "providers": [
                    {
                        "name": "openai",
                        "is_configured": False,
                        "debug": "SECRET_SHOULD_NOT_LEAK",
                    }
                ],
            }
        if path == "/api/v1/media/search":
            response["body"] = {
                "results": [
                    {
                        "title": "tldw onboarding verification",
                        "content": "tldw-onboarding-verification-unique SECRET_SHOULD_NOT_LEAK",
                    }
                ]
            }
        return response

    monkeypatch.setattr(profile_verify, "_request", fake_request)

    result = profile_verify.run_profile_checks(
        profile=profile_verify.SetupProfile(
            name="local-single",
            auth_mode="single_user",
            docker=False,
            includes_webui=False,
            includes_postgres=False,
        ),
        base_url="http://127.0.0.1:8000",
        webui_url=None,
        env_path=env_path,
        first_value=True,
        timeout=5.0,
    )

    assert "SECRET_SHOULD_NOT_LEAK" not in json.dumps(result["actions"])
