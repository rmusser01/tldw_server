from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_db_transaction
from tldw_Server_API.app.main import app

_MISSING_OVERRIDE = object()


@contextmanager
def _dependency_override_scope(
    overrides: dict[Any, Any],
) -> Iterator[None]:
    previous = {
        dependency: app.dependency_overrides.get(
            dependency,
            _MISSING_OVERRIDE,
        )
        for dependency in overrides
    }
    app.dependency_overrides.update(overrides)
    try:
        yield
    finally:
        for dependency, prior_override in previous.items():
            if prior_override is _MISSING_OVERRIDE:
                app.dependency_overrides.pop(dependency, None)
            else:
                app.dependency_overrides[dependency] = prior_override


def _active_user_context() -> dict[str, object]:
    return {
        "id": 1,
        "username": "legacy-user",
        "email": "legacy@example.com",
        "role": "user",
        "is_active": True,
        "is_verified": True,
        "storage_quota_mb": 5120,
        "storage_used_mb": 0.0,
        "created_at": datetime.utcnow(),
        "last_login": None,
    }


def test_users_me_update_returns_404_when_update_affects_no_rows(auth_headers, monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import users as users_endpoints

    class _FakeCursor:
        rowcount = 0

    class _FakeDB:
        async def execute(self, *_args, **_kwargs):
            return _FakeCursor()

    async def _fake_get_db_transaction():
        yield _FakeDB()

    async def _fake_resolve_user_context(_principal, *, allow_missing: bool = False):
        del allow_missing
        return _active_user_context()

    monkeypatch.setattr(users_endpoints, "_resolve_user_context", _fake_resolve_user_context)
    with _dependency_override_scope(
        {get_db_transaction: _fake_get_db_transaction}
    ):
        with TestClient(app) as client:
            resp = client.put(
                "/api/v1/users/me",
                headers=auth_headers,
                json={"email": "updated@example.com"},
            )

    assert resp.status_code == 404
    assert resp.json().get("detail") == "User not found"


def test_users_me_update_succeeds_when_row_is_updated(auth_headers, monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import users as users_endpoints

    class _FakeCursor:
        rowcount = 1

    class _FakeDB:
        calls: list[tuple[object, ...]] = []

        async def execute(self, *args, **_kwargs):
            self.calls.append(args)
            return _FakeCursor()

    fake_db = _FakeDB()

    async def _fake_get_db_transaction():
        yield fake_db

    async def _fake_resolve_user_context(_principal, *, allow_missing: bool = False):
        del allow_missing
        return _active_user_context()

    monkeypatch.setattr(users_endpoints, "_resolve_user_context", _fake_resolve_user_context)
    with _dependency_override_scope(
        {get_db_transaction: _fake_get_db_transaction}
    ):
        with TestClient(app) as client:
            resp = client.put(
                "/api/v1/users/me",
                headers=auth_headers,
                json={"email": "UPDATED@EXAMPLE.COM"},
            )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload.get("warning") == "deprecated_endpoint"
    assert payload.get("successor") == "/api/v1/users/me/profile"
    assert payload.get("email") == "updated@example.com"
    assert fake_db.calls == [
        (
            "UPDATE users SET email = $1, updated_at = CURRENT_TIMESTAMP WHERE id = $2",
            "updated@example.com",
            1,
        )
    ]


@pytest.mark.parametrize(
    "request_json",
    [
        {},
        {"email": None},
        {"email": "legacy@example.com"},
    ],
    ids=["omitted", "null", "unchanged"],
)
def test_users_me_update_no_email_change_is_400_without_sql(
    auth_headers,
    monkeypatch,
    request_json,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import users as users_endpoints

    class _FakeDB:
        async def execute(self, *_args, **_kwargs):
            raise AssertionError("no-op deprecated email request must not write")

    async def _fake_get_db_transaction():
        yield _FakeDB()

    async def _fake_resolve_user_context(_principal, *, allow_missing: bool = False):
        del allow_missing
        return _active_user_context()

    monkeypatch.setattr(users_endpoints, "_resolve_user_context", _fake_resolve_user_context)
    with _dependency_override_scope(
        {get_db_transaction: _fake_get_db_transaction}
    ):
        with TestClient(app) as client:
            resp = client.put(
                "/api/v1/users/me",
                headers=auth_headers,
                json=request_json,
            )

    assert resp.status_code == 400
    assert resp.json() == {"detail": "No updates provided"}


def test_dependency_override_scope_restores_preexisting_entry() -> None:
    async def _prior_override():
        yield object()

    async def _replacement_override():
        yield object()

    original_overrides = dict(app.dependency_overrides)
    app.dependency_overrides[get_db_transaction] = _prior_override
    try:
        with _dependency_override_scope(
            {get_db_transaction: _replacement_override}
        ):
            assert (
                app.dependency_overrides[get_db_transaction]
                is _replacement_override
            )
        assert app.dependency_overrides[get_db_transaction] is _prior_override
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original_overrides)
