"""Tests for remote runtime CLI commands against a running gateway."""

from __future__ import annotations

import io
import json
import urllib.error
from collections.abc import Mapping
from typing import Any

import pytest

from mcp_unified.gateway import cli as gateway_cli
from mcp_unified.gateway.remote_admin import (
    RemoteGatewayAdminClient,
    RemoteGatewayAdminConfig,
    RemoteGatewayAdminError,
)


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


def test_remote_gateway_admin_config_preserves_mounted_base_path() -> None:
    """The base URL is the mounted gateway prefix, not just the server origin."""

    prefixed = RemoteGatewayAdminClient(
        RemoteGatewayAdminConfig(gateway_url="http://example.test/mcp/")
    )
    origin_only = RemoteGatewayAdminClient(
        RemoteGatewayAdminConfig(gateway_url="http://example.test")
    )

    assert (
        prefixed.endpoint_url("/external-servers/runtime")
        == "http://example.test/mcp/external-servers/runtime"
    )
    assert (
        origin_only.endpoint_url("/external-servers/runtime")
        == "http://example.test/external-servers/runtime"
    )


def test_remote_gateway_admin_client_sends_env_style_admin_header() -> None:
    """Admin auth is passed as a header value when configured."""

    seen_headers: list[dict[str, str]] = []

    def opener(request: Any, *, timeout: float) -> _Response:
        del timeout
        seen_headers.append(
            {key.lower(): value for key, value in request.header_items()}
        )
        return _Response(b'{"ok": true, "servers": []}')

    client = RemoteGatewayAdminClient(
        RemoteGatewayAdminConfig(
            gateway_url="http://example.test/mcp",
            admin_header_name="X-Admin-Key",
            admin_key="secret-value",
        ),
        opener=opener,
    )

    assert client.list_runtime_servers() == {"ok": True, "servers": []}
    assert seen_headers[0]["x-admin-key"] == "secret-value"


@pytest.mark.parametrize("admin_key", ["secret\nvalue", "secret\rvalue"])
def test_remote_gateway_admin_config_rejects_admin_key_line_breaks(
    admin_key: str,
) -> None:
    """Admin header values cannot contain line breaks."""

    with pytest.raises(ValueError, match="admin_key cannot contain line breaks"):
        RemoteGatewayAdminConfig(
            gateway_url="http://example.test/mcp",
            admin_key=admin_key,
        )


def test_remote_gateway_admin_client_omits_admin_header_when_absent() -> None:
    """The admin header is not sent when no admin key is configured."""

    seen_headers: list[dict[str, str]] = []

    def opener(request: Any, *, timeout: float) -> _Response:
        del timeout
        seen_headers.append(
            {key.lower(): value for key, value in request.header_items()}
        )
        return _Response(b'{"ok": true, "servers": []}')

    client = RemoteGatewayAdminClient(
        RemoteGatewayAdminConfig(gateway_url="http://example.test/mcp"),
        opener=opener,
    )

    assert client.list_runtime_servers()["ok"] is True
    assert "x-mcp-gateway-admin-key" not in seen_headers[0]


def test_remote_gateway_admin_client_passes_through_json_payloads() -> None:
    """Successful JSON object responses are returned unchanged."""

    def opener(request: Any, *, timeout: float) -> _Response:
        assert request.full_url == "http://example.test/mcp/external-servers/search/start"
        assert request.get_method() == "POST"
        assert timeout == 3.5
        return _Response(
            b'{"ok": true, "reason_code": "external_server_started", '
            b'"server_id": "search"}'
        )

    client = RemoteGatewayAdminClient(
        RemoteGatewayAdminConfig(
            gateway_url="http://example.test/mcp",
            timeout_seconds=3.5,
        ),
        opener=opener,
    )

    assert client.start_server("search") == {
        "ok": True,
        "reason_code": "external_server_started",
        "server_id": "search",
    }


@pytest.mark.parametrize(
    "response_body",
    [b"not json", b'["not", "an", "object"]'],
)
def test_remote_gateway_admin_client_sanitizes_malformed_responses(
    response_body: bytes,
) -> None:
    """Malformed gateway responses do not echo raw response bytes."""

    def opener(request: Any, *, timeout: float) -> _Response:
        del request, timeout
        return _Response(response_body)

    client = RemoteGatewayAdminClient(
        RemoteGatewayAdminConfig(gateway_url="http://example.test/mcp"),
        opener=opener,
    )

    with pytest.raises(RemoteGatewayAdminError) as exc_info:
        client.list_runtime_servers()

    assert exc_info.value.to_payload() == {
        "error": "Gateway returned an invalid JSON object",
        "ok": False,
        "reason_code": "gateway_invalid_response",
    }


def test_remote_gateway_admin_client_sanitizes_connection_failures() -> None:
    """Connection failures return a generic envelope without raw secret values."""

    def opener(request: Any, *, timeout: float) -> _Response:
        del request, timeout
        raise urllib.error.URLError("offline secret-value")

    client = RemoteGatewayAdminClient(
        RemoteGatewayAdminConfig(gateway_url="http://example.test/mcp"),
        opener=opener,
    )

    with pytest.raises(RemoteGatewayAdminError) as exc_info:
        client.list_runtime_servers()

    payload = exc_info.value.to_payload()
    assert payload == {
        "error": "Unable to reach gateway",
        "error_type": "URLError",
        "ok": False,
        "reason_code": "gateway_connection_failed",
    }
    assert "secret-value" not in json.dumps(payload)


@pytest.mark.parametrize(
    ("status_code", "reason_code"),
    [
        (401, "admin_auth_missing"),
        (404, "external_server_not_found"),
        (503, "external_runtime_unavailable"),
    ],
)
def test_remote_gateway_admin_client_preserves_http_error_json_payloads(
    status_code: int,
    reason_code: str,
) -> None:
    """HTTP error bodies preserve public gateway reason fields."""

    def opener(request: Any, *, timeout: float) -> _Response:
        del timeout
        body = json.dumps(
            {
                "error": "Gateway rejected request",
                "ok": False,
                "reason_code": reason_code,
                "server_id": "search",
            }
        ).encode("utf-8")
        raise urllib.error.HTTPError(
            request.full_url,
            status_code,
            "error",
            hdrs={},
            fp=io.BytesIO(body),
        )

    client = RemoteGatewayAdminClient(
        RemoteGatewayAdminConfig(gateway_url="http://example.test/mcp"),
        opener=opener,
    )

    with pytest.raises(RemoteGatewayAdminError) as exc_info:
        client.stop_server("search")

    assert exc_info.value.to_payload() == {
        "error": "Gateway rejected request",
        "ok": False,
        "reason_code": reason_code,
        "server_id": "search",
        "status_code": status_code,
    }


@pytest.mark.parametrize(
    ("argv", "method_name", "method_args"),
    [
        (["runtime-list"], "list_runtime_servers", ()),
        (["runtime-start", "search"], "start_server", ("search",)),
        (["runtime-stop", "search"], "stop_server", ("search",)),
        (["runtime-restart", "search"], "restart_server", ("search",)),
        (["runtime-refresh"], "refresh_server", (None,)),
        (["runtime-refresh", "search"], "refresh_server", ("search",)),
        (["runtime-reconcile"], "reconcile", (None,)),
        (["runtime-reconcile", "search"], "reconcile", ("search",)),
        (["runtime-install", "search"], "install_server", ("search",)),
        (["runtime-update", "search"], "update_server", ("search",)),
    ],
)
def test_gateway_cli_runtime_commands_call_remote_gateway(
    argv: list[str],
    method_name: str,
    method_args: tuple[object, ...],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime CLI commands call the remote admin client, not local processes."""

    calls: list[tuple[str, tuple[object, ...]]] = []
    configs: list[RemoteGatewayAdminConfig] = []

    class _Client:
        def __init__(self, config: RemoteGatewayAdminConfig) -> None:
            configs.append(config)

        def __getattr__(self, name: str) -> Any:
            def _method(*args: object) -> dict[str, object]:
                calls.append((name, args))
                return {"ok": True, "method": name, "args": list(args)}

            return _method

    monkeypatch.setattr(gateway_cli, "RemoteGatewayAdminClient", _Client)

    exit_code = gateway_cli.main(
        [*argv, "--gateway-url", "http://127.0.0.1:8000/mcp"]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert payload == {
        "args": list(method_args),
        "method": method_name,
        "ok": True,
    }
    assert calls == [(method_name, method_args)]
    assert configs[0].gateway_url == "http://127.0.0.1:8000/mcp"


def test_gateway_cli_runtime_commands_use_gateway_url_and_admin_key_env(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime commands accept URL/admin config from environment variables."""

    configs: list[RemoteGatewayAdminConfig] = []

    class _Client:
        def __init__(self, config: RemoteGatewayAdminConfig) -> None:
            configs.append(config)

        def list_runtime_servers(self) -> dict[str, object]:
            return {"ok": True, "servers": []}

    monkeypatch.setattr(gateway_cli, "RemoteGatewayAdminClient", _Client)
    monkeypatch.setenv("MCP_UNIFIED_GATEWAY_URL", "http://127.0.0.1:8000/mcp")
    monkeypatch.setenv("MCP_UNIFIED_GATEWAY_ADMIN_KEY", "admin-secret")

    exit_code = gateway_cli.main(["runtime-list", "--admin-header-name", "X-Admin"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out) == {"ok": True, "servers": []}
    assert captured.err == ""
    assert configs[0].gateway_url == "http://127.0.0.1:8000/mcp"
    assert configs[0].admin_header_name == "X-Admin"
    assert configs[0].admin_key == "admin-secret"


def test_gateway_cli_runtime_commands_require_gateway_url(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime commands require --gateway-url or MCP_UNIFIED_GATEWAY_URL."""

    monkeypatch.delenv("MCP_UNIFIED_GATEWAY_URL", raising=False)

    exit_code = gateway_cli.main(["runtime-list"])

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 2
    assert captured.out == ""
    assert payload == {
        "error": (
            "--gateway-url is required unless MCP_UNIFIED_GATEWAY_URL is set"
        ),
        "ok": False,
    }


def test_gateway_cli_runtime_commands_do_not_accept_admin_key_argument(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Runtime CLI avoids command-line admin-key secrets."""

    exit_code = gateway_cli.main(
        [
            "runtime-list",
            "--gateway-url",
            "http://127.0.0.1:8000/mcp",
            "--admin-key",
            "secret",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 2
    assert captured.out == ""
    assert payload["ok"] is False
    assert "admin-key" in payload["error"]


def test_gateway_cli_runtime_commands_preserve_remote_error_payload(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI stderr preserves remote gateway reason_code payloads."""

    class _Client:
        def __init__(self, config: RemoteGatewayAdminConfig) -> None:
            del config

        def stop_server(self, server_id: str) -> dict[str, object]:
            raise RemoteGatewayAdminError.from_payload(
                {
                    "error": "Gateway rejected request",
                    "ok": False,
                    "reason_code": "external_server_not_found",
                    "server_id": server_id,
                    "status_code": 404,
                }
            )

    monkeypatch.setattr(gateway_cli, "RemoteGatewayAdminClient", _Client)

    exit_code = gateway_cli.main(
        [
            "runtime-stop",
            "search",
            "--gateway-url",
            "http://127.0.0.1:8000/mcp",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "error": "Gateway rejected request",
        "ok": False,
        "reason_code": "external_server_not_found",
        "server_id": "search",
        "status_code": 404,
    }
