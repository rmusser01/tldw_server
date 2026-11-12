"""Chat transport forwards exactly the credential the socket was verified with, and proxy trust inputs."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from starlette.websockets import WebSocket

from tldw_Server_API.app.api.v1.endpoints import persona
from tldw_Server_API.app.core.AuthNZ import websocket_principal
from tldw_Server_API.app.core.AuthNZ.ip_allowlist import resolve_client_ip
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal

pytestmark = pytest.mark.unit


async def _unused_io(*args):
    raise AssertionError("Credential projection must not perform socket I/O")


def _socket(method, headers, *, peer="192.0.2.10", query=b""):
    ws = WebSocket(
        {
            "type": "websocket",
            "path": "/api/v1/persona/stream",
            "headers": [(name.encode(), value.encode()) for name, value in headers],
            "client": (peer, 4321),
            "query_string": query,
        },
        receive=_unused_io,
        send=_unused_io,
    )
    ws.state.persona_auth_method = method
    return ws


@pytest.fixture
def http_verifier(monkeypatch):
    """HTTP's choice: the JWT ``chosen-token`` first, else the API key (``X-API-KEY``, or a non-JWT bearer)."""

    async def verify(request):
        if request.headers.get("authorization") == "Bearer chosen-token":
            return AuthPrincipal(kind="user", user_id=1, token_type="access")
        if request.headers.get("x-api-key") or request.headers.get("authorization"):
            return AuthPrincipal(kind="user", user_id=1, token_type="api_key")
        raise HTTPException(status_code=401, detail="Not authenticated")

    monkeypatch.setattr(websocket_principal, "get_auth_principal", verify)


@pytest.mark.parametrize(
    "headers,query",
    [
        ([("authorization", "Bearer stale-jwt"), ("x-api-key", "chosen-key")], b""),
        ([("authorization", "Bearer chosen-key")], b""),
        ([], b"api_key=chosen-key"),
    ],
)
@pytest.mark.asyncio
async def test_api_key_auth_forwards_only_the_selected_key(http_verifier, headers, query):
    ws = _socket("", headers + [("cookie", "session=ambient"), ("x-csrf-token", "unrelated")], query=query)
    assert await persona._resolve_authenticated_user_id(ws) == ("1", True, True)
    assert persona._persona_conversation_headers(ws) == {"x-api-key": "chosen-key"}


@pytest.mark.parametrize("source", ["header", "subprotocol", "query"])
@pytest.mark.asyncio
async def test_jwt_auth_cannot_fall_back_to_a_different_key_or_cookie(http_verifier, source):
    headers = [("cookie", "session=other-owner")]
    query = b""
    if source == "header":
        headers += [("authorization", "Bearer chosen-token"), ("x-api-key", "other-owner")]
    elif source == "subprotocol":
        headers += [("sec-websocket-protocol", "bearer, chosen-token"), ("x-api-key", "other-owner")]
    else:  # a header credential would win over the query, so the query stands alone
        query = b"token=chosen-token"
    ws = _socket("", headers, query=query)
    assert await persona._resolve_authenticated_user_id(ws) == ("1", True, True)
    assert persona._persona_conversation_headers(ws) == {"authorization": "Bearer chosen-token"}


def test_cookie_credentials_require_successful_cookie_auth_state():
    headers = [
        ("cookie", "session=owned; csrf_token=owned-csrf"),
        ("origin", "https://trusted.example"),
        ("authorization", "Bearer other-owner"),
        ("x-api-key", "other-owner"),
        ("x-csrf-token", "untrusted-header"),
    ]
    ws = _socket("single_user_session", headers)
    assert persona._persona_conversation_headers(ws) == {
        "origin": "https://trusted.example",
        "cookie": "session=owned; csrf_token=owned-csrf",
        "x-csrf-token": "owned-csrf",
    }
    ws.state.persona_auth_method = ""
    assert persona._persona_conversation_headers(ws) == {
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
    headers = persona._persona_conversation_headers(ws)
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
