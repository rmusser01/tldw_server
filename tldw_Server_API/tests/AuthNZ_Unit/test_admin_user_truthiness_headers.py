from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException, Response
from starlette.requests import Request

from tldw_Server_API.app.api.v1.endpoints.admin import admin_user as admin_user_mod
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal


pytestmark = pytest.mark.unit


def _make_request(*, auth_header: str = "Bearer fake") -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/v1/admin/users",
        "headers": [(b"authorization", auth_header.encode("utf-8"))],
        "client": ("127.0.0.1", 12345),
    }
    return Request(scope)


def _make_principal() -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        subject="admin",
        token_type="access",
        jti=None,
        roles=["admin"],
        permissions=[],
        is_admin=True,
        org_ids=[],
        team_ids=[],
    )


@pytest.mark.asyncio
async def test_admin_list_users_test_mode_headers_accept_tldw_single_letter_y(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "y")

    class _FakePool:
        pool = None

    async def _fake_get_db_pool() -> _FakePool:
        return _FakePool()

    async def _fake_list_users(*args: Any, **kwargs: Any) -> tuple[list[Any], int]:
        _ = (args, kwargs)
        return [], 0

    monkeypatch.setattr("tldw_Server_API.app.core.AuthNZ.database.get_db_pool", _fake_get_db_pool)
    monkeypatch.setattr(admin_user_mod.admin_users_service, "list_users", _fake_list_users)

    response = Response()
    payload = await admin_user_mod.list_users(
        request=_make_request(),
        response=response,
        principal=_make_principal(),
        page=1,
        limit=20,
    )
    assert payload.total == 0
    assert response.headers.get("X-TLDW-Admin-DB") == "sqlite"
    assert response.headers.get("X-TLDW-Admin-Req") == "ok"


@pytest.mark.asyncio
async def test_admin_list_users_error_header_set_in_test_mode_y(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_MODE", "y")
    monkeypatch.delenv("TLDW_TEST_MODE", raising=False)

    class _FakePool:
        pool = None

    async def _fake_get_db_pool() -> _FakePool:
        return _FakePool()

    async def _failing_list_users(*args: Any, **kwargs: Any) -> tuple[list[Any], int]:
        _ = (args, kwargs)
        raise HTTPException(status_code=403, detail="forbidden")

    monkeypatch.setattr("tldw_Server_API.app.core.AuthNZ.database.get_db_pool", _fake_get_db_pool)
    monkeypatch.setattr(admin_user_mod.admin_users_service, "list_users", _failing_list_users)

    response = Response()
    with pytest.raises(HTTPException):
        await admin_user_mod.list_users(
            request=_make_request(),
            response=response,
            principal=_make_principal(),
            page=1,
            limit=20,
        )
    assert "forbidden" in response.headers.get("X-TLDW-Admin-Error", "")
