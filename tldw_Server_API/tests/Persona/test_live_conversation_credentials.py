"""Chat transport keeps the socket's chosen identity and proxy trust inputs."""

from types import SimpleNamespace

import pytest
from starlette.requests import Request
from starlette.websockets import WebSocket

from tldw_Server_API.app.api.v1.endpoints import persona
from tldw_Server_API.app.core.AuthNZ.ip_allowlist import resolve_client_ip

pytestmark = pytest.mark.unit


async def _unused_io(*args):
    raise AssertionError("Credential projection must not perform socket I/O")


def _socket(method, headers, *, peer="192.0.2.10"):
    ws = WebSocket(
        {
            "type": "websocket",
            "path": "/api/v1/persona/stream",
            "headers": [(name.encode(), value.encode()) for name, value in headers],
            "client": (peer, 4321),
            "query_string": b"",
        },
        receive=_unused_io,
        send=_unused_io,
    )
    ws.state.persona_auth_method = method
    return ws


@pytest.mark.parametrize(
    "headers,token,key,expected",
    [
        ([("authorization", "Bearer ignored-key"), ("x-api-key", "chosen-key")], None, None, "chosen-key"),
        ([("authorization", "Bearer chosen-key")], None, None, "chosen-key"),
        ([], "chosen-key", None, "chosen-key"),
        ([("authorization", "Bearer expired-jwt")], None, "chosen-key", "chosen-key"),
    ],
)
def test_api_key_auth_forwards_only_the_selected_key(headers, token, key, expected):
    ws = _socket("api_key", headers + [("cookie", "session=ambient"), ("x-csrf-token", "unrelated")])
    result = persona._persona_conversation_headers(ws, token, key)
    assert result == {"x-api-key": expected}


@pytest.mark.parametrize("method", ["jwt_authnz", "jwt_mcp"])
@pytest.mark.parametrize("source", ["header", "subprotocol", "query"])
def test_jwt_auth_cannot_fall_back_to_a_different_key_or_cookie(method, source):
    headers = [("x-api-key", "other-owner"), ("cookie", "session=other-owner")]
    token = None
    if source == "header":
        headers.append(("authorization", "Bearer chosen-token"))
    elif source == "subprotocol":
        headers.append(("sec-websocket-protocol", "bearer, chosen-token"))
    else:
        token = "chosen-token"
    ws = _socket(method, headers)
    assert persona._persona_conversation_headers(ws, token, None) == {
        "authorization": "Bearer chosen-token",
    }


def test_cookie_credentials_require_successful_cookie_auth_state():
    headers = [
        ("cookie", "session=owned; csrf_token=owned-csrf"),
        ("origin", "https://trusted.example"),
        ("authorization", "Bearer other-owner"),
        ("x-api-key", "other-owner"),
        ("x-csrf-token", "untrusted-header"),
    ]
    ws = _socket("single_user_session", headers)
    assert persona._persona_conversation_headers(ws, None, None) == {
        "origin": "https://trusted.example",
        "cookie": "session=owned; csrf_token=owned-csrf",
        "x-csrf-token": "owned-csrf",
    }
    ws.state.persona_auth_method = ""
    assert persona._persona_conversation_headers(ws, None, None) == {
        "origin": "https://trusted.example",
    }


@pytest.mark.parametrize(
    "peer,forwarding,expected",
    [
        ("192.0.2.10", [("x-forwarded-for", "198.51.100.7")], "198.51.100.7"),
        ("203.0.113.2", [("x-forwarded-for", "198.51.100.7")], "203.0.113.2"),
        ("192.0.2.10", [("x-forwarded-for", "198.51.100.7"), ("x-forwarded-for", "203.0.113.8")], "203.0.113.8"),
        ("192.0.2.10", [("x-real-ip", "198.51.100.7"), ("x-real-ip", "203.0.113.8")], "192.0.2.10"),
    ],
)
def test_forwarding_preserves_original_proxy_trust_resolution(peer, forwarding, expected):
    ws = _socket("api_key", [("x-api-key", "chosen-key"), *forwarding], peer=peer)
    headers = persona._persona_conversation_headers(ws, None, None)
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/chat/completions",
            "headers": [(name.encode(), value.encode()) for name, value in headers.items()],
            "client": ws.client,
        }
    )
    settings = SimpleNamespace(AUTH_TRUSTED_PROXY_IPS=["192.0.2.10"], AUTH_TRUST_X_FORWARDED_FOR=True)
    assert resolve_client_ip(ws, settings) == expected
    assert resolve_client_ip(request, settings) == expected
