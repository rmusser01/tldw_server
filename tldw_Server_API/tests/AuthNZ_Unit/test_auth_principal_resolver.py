from types import SimpleNamespace
from typing import Any, Dict

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from starlette.types import Scope

from tldw_Server_API.app.core.AuthNZ import auth_principal_resolver as resolver
from tldw_Server_API.app.core.AuthNZ.auth_principal_resolver import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings


def _make_request(headers: Dict[str, str] | None = None) -> Request:
    scope: Scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in (headers or {}).items()],
        "client": ("127.0.0.1", 12345),
    }
    return Request(scope)


def test_build_principal_ignores_legacy_user_role_without_canonical_claims() -> None:
    user = User(
        id=44,
        username="legacy-admin",
        role="admin",
        roles=[],
        permissions=[],
        is_admin=False,
    )

    principal = resolver._build_principal_from_user(
        user,
        kind="user",
        request=_make_request(),
    )

    assert principal.roles == []
    assert principal.is_admin is False


@pytest.mark.asyncio
async def test_get_auth_principal_reuses_existing_context():
    req = _make_request()
    principal = AuthPrincipal(kind="anonymous")
    ctx = AuthContext(principal=principal, ip="127.0.0.1", user_agent=None, request_id=None)
    req.state.auth = ctx

    resolved = await get_auth_principal(req)
    assert resolved is principal


@pytest.mark.asyncio
async def test_get_auth_principal_single_user_mode(monkeypatch):
    async def _fake_authenticate_api_key_user(request, _api_key: str) -> User:
        # Simulate a bootstrapped single-user admin principal.
        user = User(id=1, username="single_user", is_active=True, roles=["admin"], permissions=["*"], is_admin=True)
        request.state.user_id = 1
        request.state.org_ids = []
        request.state.team_ids = []
        principal = AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=123,
            subject="single_user",
            token_type="api_key",
            jti=None,
            roles=list(user.roles or []),
            permissions=list(user.permissions or []),
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )
        ctx = AuthContext(principal=principal, ip="127.0.0.1", user_agent=None, request_id=None)
        request.state.auth = ctx
        request.state._auth_user = user
        return user

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.User_DB_Handling.authenticate_api_key_user",
        _fake_authenticate_api_key_user,
    )

    req = _make_request(headers={"X-API-KEY": "fixed-key"})

    principal = await get_auth_principal(req)
    assert principal.kind == "user"
    assert principal.subject == "single_user"
    assert principal.user_id == 1
    assert principal.is_admin is True
    assert "admin" in principal.roles
    assert isinstance(getattr(req.state, "auth", None), AuthContext)
    assert getattr(req.state, "user_id", None) == 1


@pytest.mark.asyncio
async def test_get_auth_principal_jwt_path(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    reset_settings()

    async def _fake_verify_jwt_and_fetch_user(request, _token: str = "") -> User:
        # simulate User with claims and membership already attached to request.state
        request.state.user_id = 42
        request.state.org_ids = [10]
        request.state.team_ids = [20]
        return User(
            id=42,
            username="jwt-user",
            is_active=True,
            roles=["user"],
            permissions=["media.read"],
            is_admin=False,
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.User_DB_Handling.verify_jwt_and_fetch_user",
        _fake_verify_jwt_and_fetch_user,
    )

    req = _make_request(headers={"Authorization": "Bearer token-abc"})

    try:
        principal = await get_auth_principal(req)
        assert principal.kind == "user"
        assert principal.user_id == 42
        assert principal.api_key_id is None
        assert principal.org_ids == [10]
        assert principal.team_ids == [20]
    finally:
        for env_key in ("AUTH_MODE", "DATABASE_URL"):
            monkeypatch.delenv(env_key, raising=False)
        reset_settings()


@pytest.mark.asyncio
async def test_get_auth_principal_jwt_path_ignores_boolean_is_admin_without_claims(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    reset_settings()

    async def _fake_verify_jwt_and_fetch_user(request, _token: str = "") -> User:
        request.state.user_id = 43
        request.state.org_ids = [11]
        request.state.team_ids = [21]
        return User(
            id=43,
            username="jwt-user",
            is_active=True,
            roles=["user"],
            permissions=[],
            is_admin=True,
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.User_DB_Handling.verify_jwt_and_fetch_user",
        _fake_verify_jwt_and_fetch_user,
    )

    req = _make_request(headers={"Authorization": "Bearer token-abc"})

    try:
        principal = await get_auth_principal(req)
        assert principal.kind == "user"
        assert principal.user_id == 43
        assert principal.is_admin is False
    finally:
        for env_key in ("AUTH_MODE", "DATABASE_URL"):
            monkeypatch.delenv(env_key, raising=False)
        reset_settings()


@pytest.mark.asyncio
async def test_get_auth_principal_api_key_path(monkeypatch):
    async def _fake_authenticate_api_key_user(request, _api_key: str) -> User:
        request.state.user_id = 7
        request.state.api_key_id = 100
        request.state.org_ids = [1, 2]
        request.state.team_ids = [3]
        return User(
            id=7,
            username="api-user",
            is_active=True,
            roles=["user"],
            permissions=["media.read"],
            is_admin=False,
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.User_DB_Handling.authenticate_api_key_user",
        _fake_authenticate_api_key_user,
    )

    req = _make_request(headers={"X-API-KEY": "vk-key"})

    principal = await get_auth_principal(req)
    assert principal.kind == "api_key"
    assert principal.user_id == 7
    assert principal.api_key_id == 100
    assert principal.org_ids == [1, 2]
    assert principal.team_ids == [3]


@pytest.mark.asyncio
async def test_get_auth_principal_falls_back_to_api_key_when_jwt_fails(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    reset_settings()

    async def _fake_verify_jwt_and_fetch_user(request, _token: str = "") -> User:
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    async def _fake_authenticate_api_key_user(request, _api_key: str) -> User:
        request.state.user_id = 7
        request.state.api_key_id = 100
        request.state.org_ids = [1]
        request.state.team_ids = [2]
        return User(
            id=7,
            username="api-user",
            is_active=True,
            roles=["user"],
            permissions=["media.read"],
            is_admin=False,
        )

    monkeypatch.setattr(
        resolver.User_DB_Handling,
        "verify_jwt_and_fetch_user",
        _fake_verify_jwt_and_fetch_user,
    )
    monkeypatch.setattr(
        resolver.User_DB_Handling,
        "authenticate_api_key_user",
        _fake_authenticate_api_key_user,
    )

    req = _make_request(
        headers={
            "Authorization": "Bearer aaa.bbb.ccc",
            "X-API-KEY": "vk-key",
        }
    )

    try:
        principal = await get_auth_principal(req)
        assert principal.kind == "api_key"
        assert principal.user_id == 7
        assert principal.api_key_id == 100
        assert principal.org_ids == [1]
        assert principal.team_ids == [2]
    finally:
        monkeypatch.delenv("AUTH_MODE", raising=False)
        reset_settings()


@pytest.mark.asyncio
async def test_get_auth_principal_missing_credentials_raises_401(monkeypatch):
    req = _make_request()

    with pytest.raises(HTTPException) as exc_info:
        await get_auth_principal(req)

    assert exc_info.value.status_code == 401
