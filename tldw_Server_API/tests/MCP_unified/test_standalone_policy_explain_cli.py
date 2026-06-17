"""Tests for the standalone MCP gateway policy explain CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mcp_unified.gateway import cli as gateway_cli
from mcp_unified.gateway.remote_admin import (
    RemoteGatewayAdminClient,
    RemoteGatewayAdminConfig,
    RemoteGatewayAdminError,
)

pytestmark = pytest.mark.unit


class _Response:
    """Small stdlib urlopen-compatible response double."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_remote_gateway_admin_client_explain_policy_posts_json_body() -> None:
    """Policy explain remote calls POST the supplied JSON body."""

    seen: list[tuple[str, str, dict[str, str], dict[str, Any]]] = []

    def opener(request: Any, *, timeout: float) -> _Response:
        del timeout
        seen.append(
            (
                request.full_url,
                request.get_method(),
                {key.lower(): value for key, value in request.header_items()},
                json.loads(request.data.decode("utf-8")),
            )
        )
        return _Response(b'{"ok": true, "final_outcome": "allow"}')

    client = RemoteGatewayAdminClient(
        RemoteGatewayAdminConfig(gateway_url="http://example.test/mcp"),
        opener=opener,
    )

    assert client.explain_policy(
        {
            "profile_id": "backend-engineer",
            "tool_name": "fs.patch",
            "arguments": {"path": "workspace/app.py"},
        }
    ) == {"ok": True, "final_outcome": "allow"}
    assert seen == [
        (
            "http://example.test/mcp/policy/explain",
            "POST",
            {
                "accept": "application/json",
                "content-type": "application/json",
            },
            {
                "profile_id": "backend-engineer",
                "tool_name": "fs.patch",
                "arguments": {"path": "workspace/app.py"},
            },
        )
    ]


def test_remote_gateway_admin_client_preview_profile_tools_posts_json_body() -> None:
    """Profile tool preview remote calls POST the supplied JSON body."""

    seen: list[tuple[str, str, dict[str, Any]]] = []

    def opener(request: Any, *, timeout: float) -> _Response:
        del timeout
        seen.append(
            (
                request.full_url,
                request.get_method(),
                json.loads(request.data.decode("utf-8")),
            )
        )
        return _Response(b'{"ok": true, "tools": []}')

    client = RemoteGatewayAdminClient(
        RemoteGatewayAdminConfig(gateway_url="http://example.test/mcp"),
        opener=opener,
    )

    assert client.preview_profile_tools(
        "backend engineer",
        {
            "category": "filesystem",
            "include_denied": False,
            "include_recommendations": False,
            "limit": 25,
        },
    ) == {"ok": True, "tools": []}
    assert seen == [
        (
            "http://example.test/mcp/profiles/backend%20engineer/tool-preview",
            "POST",
            {
                "category": "filesystem",
                "include_denied": False,
                "include_recommendations": False,
                "limit": 25,
            },
        )
    ]


def test_remote_gateway_admin_error_preserves_message_payload() -> None:
    """Remote admin errors preserve gateway message fields for CLI stderr."""

    error = RemoteGatewayAdminError.from_payload(
        {"message": "Policy explanation failed", "reason_code": "policy_failed"},
        status_code=403,
    )

    assert error.to_payload() == {
        "error": "Policy explanation failed",
        "message": "Policy explanation failed",
        "ok": False,
        "reason_code": "policy_failed",
        "status_code": 403,
    }


def test_gateway_cli_explain_policy_reads_args_json_file_for_remote_payload(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """--args-json-file supplies tool arguments without command-line JSON."""

    args_file = tmp_path / "tool-args.json"
    args_file.write_text('{"path": "workspace/app.py"}', encoding="utf-8")
    configs: list[RemoteGatewayAdminConfig] = []
    calls: list[dict[str, Any]] = []

    class _Client:
        def __init__(self, config: RemoteGatewayAdminConfig) -> None:
            configs.append(config)

        def explain_policy(self, payload: dict[str, Any]) -> dict[str, Any]:
            calls.append(payload)
            return {"ok": True, "payload": payload}

    monkeypatch.setattr(gateway_cli, "RemoteGatewayAdminClient", _Client)
    monkeypatch.setenv("MCP_UNIFIED_GATEWAY_ADMIN_KEY", "admin-secret")

    exit_code = gateway_cli.main(
        [
            "explain-policy",
            "--gateway-url",
            "http://127.0.0.1:8000/mcp",
            "--admin-header-name",
            "X-Admin",
            "--profile",
            "backend-engineer",
            "--tool",
            "fs.patch",
            "--args-json-file",
            str(args_file),
            "--capability",
            "filesystem",
            "--session-id",
            "session-1",
            "--static-policy-only",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "ok": True,
        "payload": {
            "arguments": {"path": "workspace/app.py"},
            "capability": "filesystem",
            "mode": "static_policy_only",
            "profile_id": "backend-engineer",
            "session_id": "session-1",
            "tool_name": "fs.patch",
        },
    }
    assert calls == [
        {
            "arguments": {"path": "workspace/app.py"},
            "capability": "filesystem",
            "mode": "static_policy_only",
            "profile_id": "backend-engineer",
            "session_id": "session-1",
            "tool_name": "fs.patch",
        }
    ]
    assert configs[0].gateway_url == "http://127.0.0.1:8000/mcp"
    assert configs[0].admin_header_name == "X-Admin"
    assert configs[0].admin_key == "admin-secret"


def test_gateway_cli_explain_policy_defaults_to_local_when_gateway_env_is_set(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ambient gateway env does not silently switch policy explain to remote mode."""

    calls: list[tuple[str, str]] = []

    def local_handler(args: Any, operation: Any) -> int:
        del operation
        calls.append((args.profile, args.tool))
        print(json.dumps({"ok": True, "mode": "local"}))
        return 0

    monkeypatch.setenv("MCP_UNIFIED_GATEWAY_URL", "http://127.0.0.1:8000/mcp")
    monkeypatch.setattr(
        gateway_cli,
        "_handle_local_policy_explain_command",
        local_handler,
    )

    exit_code = gateway_cli.main(
        [
            "explain-policy",
            "--profile",
            "backend-engineer",
            "--tool",
            "fs.patch",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert json.loads(captured.out) == {"ok": True, "mode": "local"}
    assert calls == [("backend-engineer", "fs.patch")]


def test_gateway_cli_preview_profile_tools_remote_flag_uses_gateway_env(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--remote explicitly selects env-backed remote gateway mode."""

    configs: list[RemoteGatewayAdminConfig] = []

    class _Client:
        def __init__(self, config: RemoteGatewayAdminConfig) -> None:
            configs.append(config)

        def preview_profile_tools(
            self,
            profile_id: str,
            payload: dict[str, Any],
        ) -> dict[str, Any]:
            return {"ok": True, "profile_id": profile_id, "payload": payload}

    monkeypatch.setenv("MCP_UNIFIED_GATEWAY_URL", "http://127.0.0.1:8000/mcp")
    monkeypatch.setenv("MCP_UNIFIED_GATEWAY_ADMIN_KEY", "admin-secret")
    monkeypatch.setattr(gateway_cli, "RemoteGatewayAdminClient", _Client)

    exit_code = gateway_cli.main(
        [
            "preview-profile-tools",
            "--remote",
            "--profile",
            "backend-engineer",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "ok": True,
        "payload": {
            "include_denied": True,
            "include_recommendations": True,
            "mode": "runtime_effective",
            "profile_id": "backend-engineer",
        },
        "profile_id": "backend-engineer",
    }
    assert configs == [
        RemoteGatewayAdminConfig(
            gateway_url="http://127.0.0.1:8000/mcp",
            admin_key="admin-secret",
        )
    ]


def test_gateway_cli_explain_policy_rejects_empty_args_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Explicit empty --args-json is invalid instead of being ignored."""

    exit_code = gateway_cli.main(
        [
            "explain-policy",
            "--profile",
            "backend-engineer",
            "--tool",
            "fs.patch",
            "--args-json",
            "",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "error": "Invalid args JSON: Expecting value",
        "ok": False,
    }


def test_gateway_cli_explain_policy_rejects_local_with_gateway_url(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Local mode cannot be combined with an explicit remote URL."""

    exit_code = gateway_cli.main(
        [
            "explain-policy",
            "--local",
            "--gateway-url",
            "http://127.0.0.1:8000/mcp",
            "--profile",
            "backend-engineer",
            "--tool",
            "fs.patch",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "error": "--local cannot be combined with --gateway-url",
        "ok": False,
    }


def test_gateway_cli_explain_policy_rejects_admin_key_argument(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Policy explain commands do not accept command-line admin secrets."""

    exit_code = gateway_cli.main(
        [
            "explain-policy",
            "--gateway-url",
            "http://127.0.0.1:8000/mcp",
            "--admin-key",
            "admin-secret",
            "--profile",
            "backend-engineer",
            "--tool",
            "fs.patch",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["ok"] is False
    assert "--admin-key" in payload["error"]


def test_gateway_cli_preview_profile_tools_maps_remote_payload_flags(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preview command registration maps payload flags to the remote client."""

    calls: list[tuple[str, dict[str, Any]]] = []

    class _Client:
        def __init__(self, config: RemoteGatewayAdminConfig) -> None:
            del config

        def preview_profile_tools(
            self,
            profile_id: str,
            payload: dict[str, Any],
        ) -> dict[str, Any]:
            calls.append((profile_id, payload))
            return {"ok": True, "payload": payload, "profile_id": profile_id}

    monkeypatch.setattr(gateway_cli, "RemoteGatewayAdminClient", _Client)

    exit_code = gateway_cli.main(
        [
            "preview-profile-tools",
            "--gateway-url",
            "http://127.0.0.1:8000/mcp",
            "--profile",
            "backend-engineer",
            "--category",
            "filesystem",
            "--exclude-recommendations",
            "--exclude-denied",
            "--limit",
            "25",
            "--session-id",
            "session-1",
            "--static-policy-only",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "ok": True,
        "payload": {
            "category": "filesystem",
            "include_denied": False,
            "include_recommendations": False,
            "limit": 25,
            "mode": "static_policy_only",
            "profile_id": "backend-engineer",
            "session_id": "session-1",
        },
        "profile_id": "backend-engineer",
    }
    assert calls == [
        (
            "backend-engineer",
            {
                "category": "filesystem",
                "include_denied": False,
                "include_recommendations": False,
                "limit": 25,
                "mode": "static_policy_only",
                "profile_id": "backend-engineer",
                "session_id": "session-1",
            },
        )
    ]
