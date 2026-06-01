from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from tldw_Server_API.cli.wizard import cli as wizard_cli
from tldw_Server_API.cli.wizard import profile_verify
from tldw_Server_API.app.api.v1.endpoints import auth as auth_endpoint
from tldw_Server_API.tests.wizard.helpers import assert_action_field, assert_wizard_json

runner = CliRunner()


def test_docker_profile_verify_does_not_spawn_ephemeral_server(monkeypatch) -> None:
    def fake_run_checks(*, profile, base_url, webui_url, env_path, first_value, check_provider, timeout):
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


def test_verify_profile_dry_run_sanitizes_url_userinfo() -> None:
    result = runner.invoke(
        wizard_cli.app,
        [
            "verify",
            "--profile",
            "docker-single-webui",
            "--base-url",
            "http://user:pass@127.0.0.1:8000",
            "--webui-url",
            "http://web:secret@127.0.0.1:8080",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = assert_wizard_json(result.output, command="verify", status="ok")
    encoded = json.dumps(payload)
    assert "user:pass" not in encoded
    assert "web:secret" not in encoded
    assert payload.get("facts", {}).get("base_url") == "http://127.0.0.1:8000"
    assert payload.get("facts", {}).get("webui_url") == "http://127.0.0.1:8080"


def test_verify_first_value_reports_provider_missing(monkeypatch, tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("AUTH_MODE=single_user\nSINGLE_USER_API_KEY=tldw_test.key\n", encoding="utf-8")

    def fake_run_checks(*, profile, base_url, webui_url, env_path, first_value, check_provider, timeout):
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


def test_first_value_metadata_search_result_fetches_detail_for_content(monkeypatch) -> None:
    seen_paths: list[str] = []

    def fake_request(method, _base_url, path, **_kwargs):
        seen_paths.append(path)
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
                            "id": 1,
                            "title": "tldw onboarding verification",
                            "url": "/api/v1/media/1",
                            "type": "document",
                        }
                    ]
                },
            }
        if path == "/api/v1/media/1":
            return {
                "url": "http://127.0.0.1:8000/api/v1/media/1",
                "status_code": 200,
                "ok": True,
                "body": {
                    "content": {
                        "text": "contains tldw-onboarding-verification-unique",
                    }
                },
            }
        raise AssertionError(f"unexpected request {method} {path}")

    monkeypatch.setattr(profile_verify, "_request", fake_request)

    result = profile_verify._first_value_check("http://127.0.0.1:8000", {}, 5.0)

    assert result["ok"] is True
    assert result["search"] == "ok"
    assert result["details"]["search"]["matched"] is True
    assert result["details"]["detail"]["ok"] is True
    assert "/api/v1/media/1" in seen_paths


def test_first_value_title_only_search_results_fail(monkeypatch) -> None:
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
                "body": {"results": [{"title": "tldw onboarding verification"}]},
            }
        raise AssertionError(f"unexpected request {method} {path}")

    monkeypatch.setattr(profile_verify, "_request", fake_request)

    result = profile_verify._first_value_check("http://127.0.0.1:8000", {}, 5.0)

    assert result["ok"] is False
    assert result["search"] == "error"
    assert result["details"]["search"]["matched"] is False


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
        check_provider=True,
        timeout=5.0,
    )

    assert "SECRET_SHOULD_NOT_LEAK" not in json.dumps(result["actions"])


def test_profile_checks_uses_api_v1_login_and_me_routes_with_process_env(monkeypatch, tmp_path: Path) -> None:
    assert auth_endpoint.router.prefix == "/auth"
    env_path = tmp_path / ".env"
    env_path.write_text(
        "AUTH_MODE=multi_user\nADMIN_USERNAME=file_admin\nADMIN_PASSWORD=file_pass\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ADMIN_USERNAME", "process_admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "process_pass")
    seen: dict[str, object] = {"paths": [], "login_data": None}

    def fake_request(method, _base_url, path, **kwargs):
        seen["paths"].append(path)
        if path == "/api/v1/auth/login":
            seen["login_data"] = kwargs.get("data")
            return {
                "url": "http://127.0.0.1:8000/api/v1/auth/login",
                "status_code": 200,
                "ok": True,
                "body": {"access_token": "jwt.token"},
            }
        return {
            "url": f"http://127.0.0.1:8000{path}",
            "status_code": 200,
            "ok": True,
            "body": {},
        }

    monkeypatch.setattr(profile_verify, "_request", fake_request)

    result = profile_verify.run_profile_checks(
        profile=profile_verify.SetupProfile(
            name="docker-multi-postgres",
            auth_mode="multi_user",
            docker=True,
            includes_webui=False,
            includes_postgres=True,
        ),
        base_url="http://127.0.0.1:8000",
        webui_url=None,
        env_path=env_path,
        first_value=False,
        check_provider=False,
        timeout=5.0,
    )

    assert result["status"] == "ok"
    assert "/api/v1/auth/login" in seen["paths"]
    assert "/api/v1/auth/me" in seen["paths"]
    assert "/api/v1/login" not in seen["paths"]
    assert "/api/v1/me" not in seen["paths"]
    assert seen["login_data"] == {"username": "process_admin", "password": "process_pass"}


def test_profile_checks_check_provider_false_skips_provider_probe(monkeypatch, tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("AUTH_MODE=single_user\nSINGLE_USER_API_KEY=file.key\n", encoding="utf-8")
    seen_paths: list[str] = []

    def fake_request(method, _base_url, path, **_kwargs):
        seen_paths.append(path)
        if path == "/api/v1/llm/providers":
            raise AssertionError("provider endpoint should not be probed")
        return {
            "url": f"http://127.0.0.1:8000{path}",
            "status_code": 200,
            "ok": True,
            "body": {},
        }

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
        first_value=False,
        check_provider=False,
        timeout=5.0,
    )

    assert result["status"] == "ok"
    assert "/api/v1/llm/providers" not in seen_paths
    assert not any("chat" in action for action in result["actions"])
    assert (
        "Provider verification skipped; pass --check-provider to check chat provider configuration." in result["notes"]
    )


def test_profile_checks_report_first_chat_state_without_completing_setup(
    monkeypatch, tmp_path: Path
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("AUTH_MODE=single_user\nSINGLE_USER_API_KEY=file.key\n", encoding="utf-8")

    def fake_request(method, _base_url, path, **_kwargs):
        if path == "/api/v1/setup/first-run/state":
            return {
                "url": "http://127.0.0.1:8000/api/v1/setup/first-run/state",
                "status_code": 200,
                "ok": True,
                "body": {
                    "status": "in_progress",
                    "first_chat": {"completed": False},
                },
            }
        return {
            "url": f"http://127.0.0.1:8000{path}",
            "status_code": 200,
            "ok": True,
            "body": {},
        }

    monkeypatch.setattr(profile_verify, "_request", fake_request)

    result = profile_verify.run_profile_checks(
        profile=profile_verify.SetupProfile(
            name="docker-single-webui",
            auth_mode="single_user",
            docker=True,
            includes_webui=True,
            includes_postgres=False,
            default_webui_url="http://127.0.0.1:8080",
        ),
        base_url="http://127.0.0.1:8000",
        webui_url="http://127.0.0.1:8080",
        env_path=env_path,
        first_value=False,
        check_provider=False,
        timeout=5.0,
    )

    assert result["status"] == "ok"
    assert_action_field(result["actions"], "first_run", "status", "first_chat_not_complete")
    assert_action_field(result["actions"], "first_run", "setup_status", "in_progress")
    assert (
        "First chat is not complete yet; open the WebUI to complete first-time setup."
        in result["notes"]
    )


def test_provider_endpoint_failure_is_fatal_and_sanitized_when_checked(monkeypatch, tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("AUTH_MODE=single_user\nSINGLE_USER_API_KEY=file.key\n", encoding="utf-8")

    def fake_request(method, _base_url, path, **_kwargs):
        if path == "/api/v1/llm/providers":
            return {
                "url": "http://127.0.0.1:8000/api/v1/llm/providers",
                "ok": False,
                "error": "SECRET_SHOULD_NOT_LEAK connection exploded",
            }
        return {
            "url": f"http://127.0.0.1:8000{path}",
            "status_code": 200,
            "ok": True,
            "body": {},
        }

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
        first_value=False,
        check_provider=True,
        timeout=5.0,
    )

    assert result["status"] == "error"
    actions_json = json.dumps(result["actions"])
    assert "SECRET_SHOULD_NOT_LEAK" not in actions_json
    assert_action_field(result["actions"], "chat", "status", "endpoint_failed")
    assert_action_field(result["actions"], "chat", "error", "request_failed")
    assert "Provider endpoint failed during verification." in result["notes"]


def test_single_user_header_uses_process_env_over_env_file(monkeypatch, tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("AUTH_MODE=single_user\nSINGLE_USER_API_KEY=file.key\n", encoding="utf-8")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "process.key")
    seen_headers: dict[str, str] = {}

    def fake_request(method, _base_url, path, **kwargs):
        if path == "/api/v1/auth/me":
            seen_headers.update(kwargs.get("headers") or {})
        return {
            "url": f"http://127.0.0.1:8000{path}",
            "status_code": 200,
            "ok": True,
            "body": {},
        }

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
        first_value=False,
        check_provider=False,
        timeout=5.0,
    )

    assert result["status"] == "ok"
    assert seen_headers == {"X-API-KEY": "process.key"}


def test_response_summary_sanitizes_url_userinfo() -> None:
    result = profile_verify._response_summary(
        {
            "url": "http://user:pass@127.0.0.1:8000/health",
            "status_code": 200,
            "ok": True,
        }
    )

    assert result["url"] == "http://127.0.0.1:8000/health"


def test_provider_check_sanitizes_url_userinfo(monkeypatch) -> None:
    def fake_request(_method, _base_url, path, **_kwargs):
        assert path == "/api/v1/llm/providers"
        return {
            "url": "http://user:pass@127.0.0.1:8000/api/v1/llm/providers",
            "status_code": 200,
            "ok": True,
            "body": {"providers": [{"name": "openai", "is_configured": True}]},
        }

    monkeypatch.setattr(profile_verify, "_request", fake_request)

    result = profile_verify._provider_check("http://user:pass@127.0.0.1:8000", {}, 5.0)
    encoded = json.dumps(result)

    assert "user:pass" not in encoded
    assert result["url"] == "http://127.0.0.1:8000/api/v1/llm/providers"
