from types import SimpleNamespace

import pytest
from fastapi import WebSocketDisconnect

from tldw_Server_API.app.core.AuthNZ.exceptions import InvalidTokenError
from tldw_Server_API.app.core.MCP_unified.auth.authnz_rbac import Action, AuthNZRBAC, Resource
from tldw_Server_API.app.core.MCP_unified.auth.jwt_manager import get_jwt_manager
from tldw_Server_API.app.core.MCP_unified.config import get_config
from tldw_Server_API.app.core.MCP_unified.server import MCPServer


class FakeWebSocket:
    def __init__(self, headers: dict, client_host: str = "127.0.0.1"):
        self.headers = headers
        self.client = SimpleNamespace(host=client_host)
        self.accepted = False
        self.closed = False
        self.close_args = None
        self.sent = []

    async def accept(self):
        self.accepted = True

    async def close(self, code: int = 1000, reason: str = ""):
        self.closed = True
        self.close_args = (code, reason)

    async def send_json(self, data):
        self.sent.append(data)

    async def receive_json(self):
        # Simulate client disconnect immediately after accepting
        raise WebSocketDisconnect()


@pytest.mark.asyncio
async def test_ws_origin_denied(monkeypatch):
    # Configure allowed origins to a different host than provided header
    monkeypatch.setenv("MCP_WS_ALLOWED_ORIGINS", "https://allowed.com")
    # Reset config cache
    try:
        get_config.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        _ = None

    server = MCPServer()
    # Ensure config picked up in case env cache didn't propagate
    server.config.ws_allowed_origins = ["https://allowed.com"]
    ws = FakeWebSocket(headers={"origin": "https://other.com"})

    await server.handle_websocket(ws, client_id="c1")
    assert ws.closed is True
    assert ws.accepted is False
    assert ws.close_args[0] == 1008
    assert "Origin not allowed" in (ws.close_args[1] or "")


@pytest.mark.asyncio
async def test_ws_query_auth_disabled_requires_auth(monkeypatch):
    # Allow any origin for this test and require auth
    monkeypatch.setenv("MCP_WS_ALLOWED_ORIGINS", "*")
    monkeypatch.setenv("MCP_WS_AUTH_REQUIRED", "1")
    monkeypatch.setenv("MCP_WS_ALLOW_QUERY_AUTH", "0")  # default, explicit here
    try:
        get_config.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        _ = None

    server = MCPServer()
    server.config.ws_allowed_origins = ["*"]
    server.config.ws_auth_required = True
    ws = FakeWebSocket(headers={"origin": "https://any.com"})

    # Provide tokens only via query args (which should be ignored)
    await server.handle_websocket(ws, client_id="c2", auth_token="ignored", api_key="ignored")
    assert ws.closed is True
    assert ws.accepted is False
    # Expect authentication required close when query tokens ignored
    assert ws.close_args[0] == 1008
    assert "Authentication required" in (ws.close_args[1] or "")


@pytest.mark.asyncio
async def test_ws_header_bearer_auth_accepts(monkeypatch):
    # Allow any origin and allow auth via header
    monkeypatch.setenv("MCP_WS_ALLOWED_ORIGINS", "*")
    monkeypatch.setenv("MCP_WS_AUTH_REQUIRED", "1")
    monkeypatch.setenv("MCP_WS_ALLOW_QUERY_AUTH", "0")
    monkeypatch.setenv("MCP_JWT_SECRET", "x" * 64)  # strong secret
    try:
        get_config.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        _ = None

    # Create a token using the same JWT manager used by server
    _ = get_config()  # ensure config instantiated with env
    jwtm = get_jwt_manager()
    token = jwtm.create_access_token(subject="user1")

    server = MCPServer()
    server.config.ws_allowed_origins = ["*"]
    server.config.ws_auth_required = True
    ws = FakeWebSocket(headers={
        "origin": "https://any.com",
        "Authorization": f"Bearer {token}",
    })

    await server.handle_websocket(ws, client_id="c3")
    # Should have accepted, then disconnected immediately via FakeWebSocket.receive_json
    assert ws.accepted is True
    # It may or may not close explicitly in this path; but must not be closed with auth errors
    if ws.closed:
        # If closed, it should not be due to auth failures
        assert ws.close_args[0] != 1008


@pytest.mark.asyncio
async def test_ws_invalid_authnz_token_allows_api_key(monkeypatch):
    monkeypatch.setenv("MCP_WS_ALLOWED_ORIGINS", "*")
    monkeypatch.setenv("MCP_WS_AUTH_REQUIRED", "1")
    monkeypatch.setenv("MCP_WS_ALLOW_QUERY_AUTH", "0")
    try:
        get_config.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        _ = None

    class _ApiKeyMgr:
        async def validate_api_key(self, api_key: str, ip_address: str | None = None):
            return {"user_id": "1", "scopes": ["read"]}

    async def _get_api_key_manager_stub():
        return _ApiKeyMgr()

    async def _verify_jwt_and_fetch_user_stub(*_args, **_kwargs):
        raise InvalidTokenError("invalid")

    import tldw_Server_API.app.core.AuthNZ.api_key_manager as api_key_mod
    import tldw_Server_API.app.core.AuthNZ.User_DB_Handling as user_db_mod
    import tldw_Server_API.app.core.MCP_unified.server as server_mod

    monkeypatch.setattr(api_key_mod, "get_api_key_manager", _get_api_key_manager_stub)
    monkeypatch.setattr(user_db_mod, "verify_jwt_and_fetch_user", _verify_jwt_and_fetch_user_stub)
    monkeypatch.setattr(server_mod, "_is_authnz_access_token", lambda _token: True)

    server = MCPServer()
    server.config.ws_allowed_origins = ["*"]
    server.config.ws_auth_required = True
    ws = FakeWebSocket(
        headers={
            "origin": "https://any.com",
            "Authorization": "Bearer invalid",
            "X-API-KEY": "valid-key",
        }
    )

    await server.handle_websocket(ws, client_id="c4")
    assert ws.accepted is True
    if ws.closed:
        assert "Authentication failed" not in (ws.close_args[1] or "")


@pytest.mark.asyncio
async def test_ws_failed_authnz_token_does_not_fall_back_to_mcp_jwt():
    server = MCPServer()
    server.config.ws_allowed_origins = ["*"]
    server.config.ws_auth_required = True

    class _FailingAuthNZProvider:
        def get_mcp_jwt_manager(self):
            return object()

        def is_authnz_access_token(self, _token: str) -> bool:
            return True

        async def authenticate_authnz_websocket_token(self, _token: str, *, websocket):
            del websocket
            return None

        async def validate_api_key(self, _api_key: str, *, ip_address: str | None = None):
            del ip_address
            return None

        def normalize_api_key_permissions(self, info: dict | None) -> list[str]:
            return []

    class _JwtVerifier:
        def __init__(self) -> None:
            self.calls = 0

        def verify_token(self, _token: str):
            self.calls += 1
            raise AssertionError("MCP JWT verifier should not receive AuthNZ access tokens")

    jwt_verifier = _JwtVerifier()
    server.auth_provider = _FailingAuthNZProvider()
    server.jwt_manager = jwt_verifier  # type: ignore[assignment]
    ws = FakeWebSocket(
        headers={
            "origin": "https://any.com",
            "Authorization": "Bearer expired-authnz-token",
        }
    )

    await server.handle_websocket(ws, client_id="c-authnz-failed")

    assert jwt_verifier.calls == 0
    assert ws.closed is True
    assert ws.close_args == (1008, "Authentication failed")


@pytest.mark.asyncio
async def test_ws_unexpected_mcp_jwt_verifier_error_is_not_masked():
    server = MCPServer()
    server.config.ws_allowed_origins = ["*"]
    server.config.ws_auth_required = True

    def _boom(_token: str):
        raise RuntimeError("verifier exploded")

    server.jwt_manager.verify_token = _boom  # type: ignore[assignment]
    ws = FakeWebSocket(
        headers={
            "origin": "https://any.com",
            "Authorization": "Bearer opaque-token",
        }
    )

    with pytest.raises(RuntimeError, match="verifier exploded"):
        await server.handle_websocket(ws, client_id="c5")



@pytest.mark.asyncio
async def test_rbac_denies_unmapped_permissions(monkeypatch):
    policy = AuthNZRBAC(db_pool=None)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.MCP_unified.auth.authnz_rbac._map_to_permission",
        lambda *args, **kwargs: None,
    )
    allowed = await policy.check_permission("1", Resource.MEDIA, Action.CREATE)
    assert allowed is False


@pytest.mark.asyncio
async def test_rbac_does_not_seed_permission_for_unknown_tool(monkeypatch):
    class _Pool:
        async def fetchone(self, *_args, **_kwargs):
            return None

    policy = AuthNZRBAC(db_pool=_Pool())
    seeded = False

    async def _ensure_permission_exists_stub(*_args, **_kwargs):
        nonlocal seeded
        seeded = True

    monkeypatch.setattr(policy, "_ensure_permission_exists", _ensure_permission_exists_stub)

    allowed = await policy.check_permission("1", Resource.TOOL, Action.EXECUTE, "definitely_missing_tool")

    assert allowed is False
    assert seeded is False
