"""Managed ingress preserves production MCP IP and authentication policy."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi import FastAPI, HTTPException
from starlette.requests import Request
from starlette.websockets import WebSocket

from tldw_Server_API.app.core.MCP_unified.config import get_config
from tldw_Server_API.app.core.MCP_unified.security import ip_filter
from tldw_Server_API.app.core.MCP_unified.server import MCPServer, _websocket_transport_metadata
from tldw_Server_API.app.core.MCP_unified.tests.support import clear_mcp_singleton_state

SECRET = "managed-gateway-test-secret-at-least-32-characters"
ORIGIN = "http://127.0.0.1:18090"


@pytest.fixture(autouse=True)
def production_policy(monkeypatch):
    for name, value in {
        "AUTH_MODE": "single_user",
        "TLDW_MANAGED_GATEWAY": "1",
        "TLDW_GATEWAY_HOP_SECRET": SECRET,
        "TLDW_MANAGED_PUBLIC_ORIGIN": ORIGIN,
        "MCP_ALLOWED_IPS": '["127.0.0.1", "::1"]',
        "MCP_BLOCKED_IPS": "[]",
        "MCP_TRUST_X_FORWARDED": "false",
        "MCP_WS_ALLOWED_ORIGINS": json.dumps([ORIGIN]),
        "MCP_WS_AUTH_REQUIRED": "true",
        "TEST_MODE": "false",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(ip_filter, "is_explicit_pytest_runtime", lambda: False)
    monkeypatch.setattr(ip_filter, "is_test_mode", lambda: False)
    monkeypatch.setattr(MCPServer, "_is_explicit_pytest_runtime", lambda self: False)
    monkeypatch.setattr(MCPServer, "_is_test_mode", lambda self: False)
    clear_mcp_singleton_state()
    yield
    clear_mcp_singleton_state()


def scope(*, peer="172.18.0.3", changes=None, extra=(), path="/api/v1/mcp/ws"):
    headers = {
        "host": "127.0.0.1:18090",
        "origin": ORIGIN,
        "x-tldw-gateway-hop": SECRET,
        "x-forwarded-for": "172.18.0.1",
        "x-forwarded-host": "127.0.0.1:18090",
        "x-forwarded-port": "18090",
        "x-forwarded-proto": "http",
    }
    for name, value in (changes or {}).items():
        if value is None:
            headers.pop(name, None)
        else:
            headers[name] = value
    return {
        "type": "http",
        "scheme": "http",
        "method": "GET",
        "path": path,
        "server": ("app", 8000),
        "client": (peer, 40000),
        "query_string": b"",
        "headers": [(k.encode(), v.encode()) for k, v in headers.items()] + list(extra),
        "app": FastAPI(),
    }


async def run_websocket(request_scope, *, authenticated=False):
    """Run the production server against an ASGI socket that disconnects after accept."""
    request_scope.update(type="websocket", scheme="ws")
    if authenticated:
        request_scope["state"] = {
            "single_user_session_id": "validated-cookie-session",
            "user_id": "1",
            "auth_principal": SimpleNamespace(roles=["user"], permissions=[]),
        }
    messages = []
    incoming = iter([{"type": "websocket.connect"}, {"type": "websocket.disconnect", "code": 1000}])
    server = MCPServer()

    async def receive():
        return next(incoming)

    async def send(message):
        messages.append(message)
        # The connection still exists at accept time; inspect the real metadata.
        if message["type"] == "websocket.accept":
            messages.append({"metadata": next(iter(server.connections.values())).metadata.copy()})

    await server.handle_websocket(WebSocket(request_scope, receive, send))
    return messages


@pytest.mark.asyncio
async def test_managed_http_bridge_reaches_existing_local_policy():
    await ip_filter.enforce_ip_allowlist(Request(scope(path="/api/v1/mcp/status")))


@pytest.mark.asyncio
async def test_managed_websocket_bridge_accepts_validated_cookie_without_storing_hop():
    messages = await run_websocket(scope(), authenticated=True)
    assert any(m.get("type") == "websocket.accept" for m in messages)
    metadata = next(m["metadata"] for m in messages if "metadata" in m)
    assert metadata["client_ip"] == "127.0.0.1"
    assert metadata["permissions"] == []
    assert SECRET not in repr(messages)
    assert "x-tldw-gateway-hop" not in repr(messages)


@pytest.mark.asyncio
async def test_managed_websocket_hop_does_not_authenticate_user():
    messages = await run_websocket(scope())
    assert messages == [{"type": "websocket.close", "code": 1008, "reason": "Authentication required"}]


@pytest.mark.asyncio
async def test_explicit_loopback_block_still_rejects_managed_http_and_websocket():
    get_config().blocked_client_ips = ["127.0.0.1"]
    with pytest.raises(HTTPException, match="Client IP not allowed"):
        await ip_filter.enforce_ip_allowlist(Request(scope()))
    assert await run_websocket(scope(), authenticated=True) == [
        {"type": "websocket.close", "code": 1008, "reason": "IP not allowed"}
    ]


INVALID_ENVELOPES = [
    {"changes": {"x-tldw-gateway-hop": None}},
    {"changes": {"x-tldw-gateway-hop": "forged"}},
    {"changes": {"host": "localhost:18090"}},
    {"changes": {"origin": "http://attacker.test"}},
    {"changes": {"origin": "http://127.0.0.1:18091"}},
    {"changes": {"x-forwarded-for": "8.8.8.8"}},
    {"changes": {"x-forwarded-for": "172.18.0.1, 127.0.0.1"}},
    {"changes": {"x-forwarded-host": "attacker.test"}},
    {"changes": {"x-forwarded-port": "018090"}},
    {"changes": {"x-forwarded-proto": "https"}},
    {"changes": {"x-forwarded-prefix": "/evil"}},
    {"changes": {"forwarded": "for=127.0.0.1"}},
    {"changes": {"x-real-ip": "127.0.0.1"}},
    {"peer": "8.8.8.8"},
    {"peer": "192.0.2.3"},
    {"path": "/api/v1/media"},
    {"path": "/api/v1/mcp-other"},
    {"path": "/api/v1/mcp//ws"},
    {"path": "/api/v1/mcp/%77s"},
] + [
    {"extra": [(name.encode(), b"duplicate")]}
    for name in (
        "host",
        "origin",
        "x-tldw-gateway-hop",
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-port",
        "x-forwarded-proto",
    )
]


@pytest.mark.asyncio
@pytest.mark.parametrize("options", INVALID_ENVELOPES)
async def test_invalid_hops_cannot_bypass_http_or_websocket_ip_policy(options):
    with pytest.raises(HTTPException, match="Client IP not allowed"):
        await ip_filter.enforce_ip_allowlist(Request(scope(**options)))
    assert await run_websocket(scope(**options), authenticated=True) == [
        {"type": "websocket.close", "code": 1008, "reason": "IP not allowed"}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,value",
    [
        ("TLDW_MANAGED_GATEWAY", "0"),
        ("AUTH_MODE", "multi_user"),
        ("TLDW_GATEWAY_HOP_SECRET", ""),
        ("TLDW_MANAGED_PUBLIC_ORIGIN", ""),
    ],
)
async def test_unconfigured_mode_does_not_trust_bridge(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(HTTPException, match="Client IP not allowed"):
        await ip_filter.enforce_ip_allowlist(Request(scope()))
    assert await run_websocket(scope(), authenticated=True) == [
        {"type": "websocket.close", "code": 1008, "reason": "IP not allowed"}
    ]


def test_transport_metadata_never_copies_private_hop():
    assert _websocket_transport_metadata(scope()) == {"mcp_transport": "websocket"}


def test_managed_compose_mcp_origin_configuration(monkeypatch):
    root = Path(__file__).resolve().parents[5]
    environment = yaml.safe_load((root / "Dockerfiles/app-bundle/compose.yaml").read_text())["services"]["app"][
        "environment"
    ]
    for name in ("MCP_WS_ALLOWED_ORIGINS", "MCP_CORS_ORIGINS"):
        monkeypatch.setenv(name, environment[name].replace("${TLDW_PUBLIC_PORT:?Public port is required}", "18090"))
    get_config.cache_clear()
    assert get_config().ws_allowed_origins == [ORIGIN]
    assert get_config().cors_origins == [ORIGIN]


@pytest.mark.asyncio
async def test_managed_hop_does_not_make_bridge_a_trusted_certificate_proxy():
    config = get_config()
    config.client_cert_required = True
    config.client_cert_header = "x-ssl-client-verify"
    config.client_cert_header_value = "SUCCESS"
    assert await run_websocket(scope(changes={"x-ssl-client-verify": "SUCCESS"}), authenticated=True) == [
        {"type": "websocket.close", "code": 1008, "reason": "Client certificate required"}
    ]


@pytest.mark.asyncio
async def test_ordinary_local_and_explicitly_trusted_hosted_proxy_policy_is_preserved(monkeypatch):
    monkeypatch.setenv("TLDW_MANAGED_GATEWAY", "0")
    await ip_filter.enforce_ip_allowlist(Request(scope(peer="127.0.0.1", changes={"x-tldw-gateway-hop": None})))
    config = get_config()
    config.trust_x_forwarded_for = True
    config.trusted_proxy_ips = ["172.18.0.3"]
    config.allowed_client_ips = ["198.51.100.5"]
    ip_filter.get_ip_access_controller.cache_clear()
    await ip_filter.enforce_ip_allowlist(
        Request(
            scope(
                changes={
                    "x-tldw-gateway-hop": None,
                    "x-forwarded-for": "198.51.100.5",
                }
            )
        )
    )


@pytest.mark.asyncio
async def test_direct_local_websocket_still_rejects_hostile_origin():
    messages = await run_websocket(
        scope(
            peer="127.0.0.1",
            changes={
                "x-tldw-gateway-hop": None,
                "origin": "http://attacker.test",
            },
        ),
        authenticated=True,
    )
    assert messages == [{"type": "websocket.close", "code": 1008, "reason": "Origin not allowed"}]
