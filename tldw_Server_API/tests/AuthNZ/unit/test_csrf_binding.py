import types
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.core.AuthNZ.csrf_protection import CSRFTokenManager
from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService, reset_jwt_service
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.AuthNZ.single_user_session import SingleUserSessionIdentity


def _dummy_request(user_id: int | None):
    o = types.SimpleNamespace()
    o.state = types.SimpleNamespace()
    if user_id is not None:
        o.state.user_id = user_id
    return o


def test_csrf_binding_user_suffix_ok(monkeypatch):


    reset_settings()
    monkeypatch.setenv("CSRF_BIND_TO_USER", "true")
    mgr = CSRFTokenManager()
    req = _dummy_request(42)
    token = mgr.generate_token(req)
    # Correct user binding validates
    assert mgr.validate_token(token, token, user_id=42) is True
    # Wrong user id fails
    assert mgr.validate_token(token, token, user_id=43) is False


def test_csrf_binding_unbound_token_rejected(monkeypatch):


    reset_settings()
    monkeypatch.setenv("CSRF_BIND_TO_USER", "true")
    mgr = CSRFTokenManager()
    req = _dummy_request(None)
    token = mgr.generate_token(req)
    assert token.endswith(".unbound")
    assert mgr.validate_token(token, token, user_id=None) is True
    assert mgr.validate_token(token, token, user_id=1) is False


def test_csrf_no_binding(monkeypatch):


    reset_settings()
    monkeypatch.setenv("CSRF_BIND_TO_USER", "false")
    mgr = CSRFTokenManager()
    req = _dummy_request(None)
    token = mgr.generate_token(req)
    # No binding required, plain compare works
    assert mgr.validate_token(token, token, user_id=None) is True


def test_csrf_single_user_bearer_skips_protection(monkeypatch):


    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "bearer-test-key-1234567890")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    reset_settings()

    mgr = CSRFTokenManager()
    request = types.SimpleNamespace(
        method="POST",
        headers={
            "Authorization": "Bearer bearer-test-key-1234567890",
            "content-type": "application/json",
        },
        url=types.SimpleNamespace(path="/api/v1/chat/completions"),
        state=types.SimpleNamespace(),
    )
    assert mgr.should_protect(request) is False

    # Cleanup env to avoid leakage
    for key in ("AUTH_MODE", "SINGLE_USER_API_KEY", "DATABASE_URL"):
        monkeypatch.delenv(key, raising=False)
    reset_settings()


def test_single_user_csrf_protects_only_cookie_authenticated_requests(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_SESSION_COOKIE_NAME", "tldw_single_user_session")
    reset_settings()

    mgr = CSRFTokenManager()
    base = {
        "method": "DELETE",
        "headers": {},
        "url": types.SimpleNamespace(path="/api/v1/auth/single-user/session"),
        "state": types.SimpleNamespace(),
    }
    without_cookie = types.SimpleNamespace(**base, cookies={})
    with_cookie = types.SimpleNamespace(
        **base,
        cookies={"tldw_single_user_session": "opaque"},
    )

    assert mgr.should_protect(without_cookie) is False
    assert mgr.should_protect(with_cookie) is True
    reset_settings()


def test_effective_csrf_environment_override_drives_resolution(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import csrf_protection

    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("CSRF_ENABLED", "0")
    reset_settings()

    assert csrf_protection.resolve_effective_csrf_enabled() is False

    monkeypatch.setenv("CSRF_ENABLED", "1")
    assert csrf_protection.resolve_effective_csrf_enabled() is True
    reset_settings()


@pytest.mark.asyncio
async def test_csrf_binding_resolves_single_user_cookie_before_dependencies(monkeypatch):
    from fastapi import FastAPI
    from starlette.requests import Request

    from tldw_Server_API.app.core.AuthNZ import csrf_protection
    from tldw_Server_API.app.core.AuthNZ.csrf_protection import CSRFProtectionMiddleware

    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("CSRF_BIND_TO_USER", "true")
    reset_settings()
    identity = SingleUserSessionIdentity(
        session_id=9,
        user_id=1,
        expires_at=datetime.now(timezone.utc) + timedelta(days=1),
    )
    validate = AsyncMock(return_value=identity)
    monkeypatch.setattr(
        csrf_protection,
        "validate_single_user_session",
        validate,
        raising=False,
    )
    request = Request(
        {
            "type": "http",
            "method": "DELETE",
            "path": "/api/v1/auth/single-user/session",
            "headers": [(b"cookie", b"tldw_single_user_session=opaque")],
            "client": ("127.0.0.1", 0),
        }
    )

    middleware = CSRFProtectionMiddleware(FastAPI(), enabled=True)
    user_id = await middleware._resolve_user_id(request)

    assert user_id == 1
    assert request.state.user_id == 1
    assert request.state.single_user_session_id == 9
    validate.assert_awaited_once_with(request)
    reset_settings()


def test_csrf_middleware_rotates_unbound_cookie(monkeypatch):
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    from tldw_Server_API.app.core.AuthNZ.csrf_protection import CSRFProtectionMiddleware
    from tldw_Server_API.app.core.config import settings as global_settings

    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("CSRF_BIND_TO_USER", "true")
    reset_settings()

    original_csrf_setting = global_settings.get("CSRF_ENABLED")
    global_settings["CSRF_ENABLED"] = True

    app = FastAPI()
    app.add_middleware(CSRFProtectionMiddleware, enabled=True)

    @app.get("/public")
    async def public():
        return {"ok": True}

    @app.get("/login")
    async def login(request: Request):
        request.state.user_id = 42
        return {"ok": True}

    try:
        with TestClient(app) as client:
            resp1 = client.get("/public")
            token1 = resp1.cookies.get("csrf_token")
            assert token1 and token1.endswith(".unbound")

            resp2 = client.get("/login")
            token2 = resp2.cookies.get("csrf_token")
            assert token2 and token2 != token1
            assert not token2.endswith(".unbound")
    finally:
        if original_csrf_setting is None:
            global_settings.pop("CSRF_ENABLED", None)
        else:
            global_settings["CSRF_ENABLED"] = original_csrf_setting
        for key in ("AUTH_MODE", "CSRF_BIND_TO_USER"):
            monkeypatch.delenv(key, raising=False)
        reset_settings()


@pytest.mark.asyncio
async def test_csrf_binding_ignores_revoked_jwt(monkeypatch):
    from fastapi import FastAPI
    from starlette.requests import Request
    from tldw_Server_API.app.core.AuthNZ.csrf_protection import CSRFProtectionMiddleware

    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-csrf-secret-1234567890-abcdef")
    reset_settings()
    reset_jwt_service()

    jwt_service = JWTService()
    token = jwt_service.create_access_token(user_id=1, username="user", role="user")

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/test",
        "headers": [(b"authorization", f"Bearer {token}".encode("ascii"))],
        "client": ("127.0.0.1", 0),
    }
    request = Request(scope)

    class _StubSessionManager:
        async def is_token_blacklisted(self, _token: str, _jti=None) -> bool:
            return True

    async def _stub_get_session_manager():
        return _StubSessionManager()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.session_manager.get_session_manager",
        _stub_get_session_manager,
    )

    middleware = CSRFProtectionMiddleware(FastAPI(), enabled=True)
    user_id = await middleware._resolve_user_id(request)

    assert user_id is None
    reset_settings()
    reset_jwt_service()
