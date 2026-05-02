from typing import Optional

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import authnz_debug as debug_mod
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal


def _build_app_with_overrides(
    principal: Optional[AuthPrincipal],
    *,
    fail_with_401: bool = False,
) -> FastAPI:
    app = FastAPI()
    app.include_router(debug_mod.router, prefix="/api/v1")

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
        if fail_with_401:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        assert principal is not None
        return principal

    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal
    return app


def _make_principal(
    *,
    is_admin: bool,
    roles: Optional[list[str]] = None,
    subject: Optional[str] = None,
) -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        subject=subject,
        token_type="access",
        jti=None,
        roles=roles or [],
        permissions=[],
        is_admin=is_admin,
        org_ids=[],
        team_ids=[],
    )


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []

    def debug(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.debugs.append(message)


class _BrokenAuthState:
    @property
    def auth(self):
        raise TypeError("auth context exploded at /private/authnz-debug.db")


@pytest.mark.asyncio
async def test_resolve_api_key_id_principal_fallback_log_is_sanitized(monkeypatch):
    request = type(
        "_Request",
        (),
        {
            "state": _BrokenAuthState(),
            "headers": {},
        },
    )()
    logger_stub = _LoggerStub()
    monkeypatch.setattr(debug_mod, "logger", logger_stub)

    resolved = await debug_mod._resolve_api_key_id(request, None)

    assert resolved == {"api_key_id": None, "user_id": None}
    assert logger_stub.debugs == [
        "_resolve_api_key_id: principal-first resolution failed, falling back",
    ]
    assert "auth context exploded" not in str(logger_stub.debugs)
    assert "/private/authnz-debug.db" not in str(logger_stub.debugs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    ["/api/v1/authnz/debug/api-key-id", "/api/v1/authnz/debug/budget-summary"],
)
async def test_authnz_debug_401_when_principal_missing(path: str):
    app = _build_app_with_overrides(principal=None, fail_with_401=True)

    with TestClient(app) as client:
        resp = client.get(path)

    assert resp.status_code == 401


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    ["/api/v1/authnz/debug/api-key-id", "/api/v1/authnz/debug/budget-summary"],
)
async def test_authnz_debug_403_when_plain_admin(path: str):
    principal = _make_principal(is_admin=True, roles=["admin"])
    app = _build_app_with_overrides(principal=principal)

    with TestClient(app) as client:
        resp = client.get(path)

    assert resp.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    ["/api/v1/authnz/debug/api-key-id", "/api/v1/authnz/debug/budget-summary"],
)
async def test_authnz_debug_403_when_not_admin(path: str):
    principal = _make_principal(is_admin=False, roles=["user"])
    app = _build_app_with_overrides(principal=principal)

    with TestClient(app) as client:
        resp = client.get(path)

    assert resp.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    ["/api/v1/authnz/debug/api-key-id", "/api/v1/authnz/debug/budget-summary"],
)
async def test_authnz_debug_200_for_single_user_admin(path: str):
    principal = _make_principal(
        is_admin=True,
        roles=["admin"],
        subject="single_user",
    )
    app = _build_app_with_overrides(principal=principal)

    with TestClient(app) as client:
        resp = client.get(path)

    assert resp.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    ["/api/v1/authnz/debug/api-key-id", "/api/v1/authnz/debug/budget-summary"],
)
@pytest.mark.parametrize("roles", [["super_admin"], ["owner"]])
async def test_authnz_debug_200_for_super_admin_or_owner(path: str, roles: list[str]):
    principal = _make_principal(is_admin=True, roles=roles)
    app = _build_app_with_overrides(principal=principal)

    with TestClient(app) as client:
        resp = client.get(path)

    assert resp.status_code == 200
