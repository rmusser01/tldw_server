from types import SimpleNamespace

import pytest
from fastapi import HTTPException, status
from starlette.requests import Request

from tldw_Server_API.app.core.AuthNZ import User_DB_Handling as user_handling
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings


def _build_request(client_ip: str = "127.0.0.1") -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/test",
        "headers": [],
        "client": (client_ip, 0),
    }
    request = Request(scope)
    return request


@pytest.mark.asyncio
async def test_verify_single_user_api_key_accepts_bearer(monkeypatch):
    # Configure a synthetic single-user id for this test.
    fake_settings = SimpleNamespace(SINGLE_USER_FIXED_ID=99, PII_REDACT_LOGS=False)

    monkeypatch.setattr(user_handling, "get_settings", lambda: fake_settings)

    async def _fake_authenticate_api_key_user(request, api_key: str) -> user_handling.User:
        # The verification helper should pass the raw API key through.
        assert api_key == "test-api-key-abcdefghijklmnopqrstuvwxyz"
        return user_handling.User(id=fake_settings.SINGLE_USER_FIXED_ID, username="single_user", is_active=True)

    monkeypatch.setattr(user_handling, "authenticate_api_key_user", _fake_authenticate_api_key_user)

    request = _build_request()

    # Happy path: Authorization bearer with the configured key resolves to the single-user id.
    assert await user_handling.verify_single_user_api_key(
        request,
        api_key=None,
        authorization="Bearer test-api-key-abcdefghijklmnopqrstuvwxyz",
    ) is True

    # Missing credentials should raise 401.
    with pytest.raises(HTTPException):
        await user_handling.verify_single_user_api_key(request, api_key=None, authorization=None)

@pytest.mark.asyncio
async def test_authenticate_api_key_user_rejects_disallowed_single_user_ip(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("SINGLE_USER_ALLOWED_IPS", "[\"203.0.113.10\"]")
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "99")
    reset_settings()

    request = _build_request(client_ip="198.51.100.5")

    with pytest.raises(HTTPException) as exc_info:
        await user_handling.authenticate_api_key_user(request, "test-api-key-1234567890")

    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert exc_info.value.detail == "Invalid or missing API Key"

    for env_key in ("AUTH_MODE", "SINGLE_USER_API_KEY", "SINGLE_USER_ALLOWED_IPS", "SINGLE_USER_FIXED_ID"):
        monkeypatch.delenv(env_key, raising=False)
    reset_settings()


@pytest.mark.asyncio
async def test_authenticate_api_key_user_allows_allowed_single_user_ip(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("SINGLE_USER_ALLOWED_IPS", "[\"203.0.113.10\"]")
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "99")
    reset_settings()

    request = _build_request(client_ip="203.0.113.10")
    user = await user_handling.authenticate_api_key_user(request, "test-api-key-1234567890")

    assert user.id == 99
    assert user.is_admin is True

    for env_key in ("AUTH_MODE", "SINGLE_USER_API_KEY", "SINGLE_USER_ALLOWED_IPS", "SINGLE_USER_FIXED_ID"):
        monkeypatch.delenv(env_key, raising=False)
    reset_settings()


@pytest.mark.asyncio
async def test_authenticate_api_key_user_rejects_nonmatching_single_user_key_without_db_fallback(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "99")
    reset_settings()

    class _StubAPIKeyManager:
        async def validate_api_key(self, **_kwargs):
            raise AssertionError("single-user mismatch must not fall back to DB-backed API keys")

    async def _fake_get_api_key_manager():
        return _StubAPIKeyManager()

    monkeypatch.setattr(user_handling, "get_api_key_manager", _fake_get_api_key_manager)

    request = _build_request(client_ip="127.0.0.1")

    with pytest.raises(HTTPException) as exc_info:
        await user_handling.authenticate_api_key_user(request, "db-key-0987654321")

    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert exc_info.value.detail == "Invalid or missing API Key"

    for env_key in ("AUTH_MODE", "SINGLE_USER_API_KEY", "SINGLE_USER_FIXED_ID"):
        monkeypatch.delenv(env_key, raising=False)
    reset_settings()


@pytest.mark.asyncio
async def test_get_request_user_rejects_inactive_api_key_user(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    reset_settings()

    class StubAPIKeyManager:
        async def validate_api_key(self, api_key: str, ip_address=None):
            return {"id": 1, "user_id": 42}

    async def fake_get_api_key_manager():
        return StubAPIKeyManager()

    async def fake_get_user_by_id(user_id: int):
        return {
            "id": user_id,
            "username": "inactive-user",
            "email": "inactive@example.com",
            "is_active": 0,
            "roles": [],
            "permissions": [],
        }

    async def fake_list_memberships(user_id: int):
        return []

    # Patch dependencies
    monkeypatch.setattr(user_handling, "get_api_key_manager", fake_get_api_key_manager)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.DB_Management.Users_DB.get_user_by_id",
        fake_get_user_by_id,
    )
    monkeypatch.setattr(user_handling, "list_memberships_for_user", fake_list_memberships)
    monkeypatch.setattr(user_handling, "set_scope", lambda *_, **__: None)

    request = _build_request()

    with pytest.raises(HTTPException) as exc:
        await user_handling.get_request_user(request, api_key="test-api-key", token=None)

    assert exc.value.status_code == status.HTTP_400_BAD_REQUEST
    assert exc.value.detail == "Inactive user"

    for env_key in ("AUTH_MODE", "DATABASE_URL"):
        monkeypatch.delenv(env_key, raising=False)
    reset_settings()


@pytest.mark.asyncio
async def test_get_request_user_allows_active_api_key_user(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    reset_settings()

    class StubAPIKeyManager:
        async def validate_api_key(self, api_key: str, ip_address=None):
            return {"id": 5, "user_id": 77}

    async def fake_get_api_key_manager():
        return StubAPIKeyManager()

    async def fake_get_user_by_id(user_id: int):
        return {
            "id": user_id,
            "username": "active-user",
            "email": "active@example.com",
            "is_active": 1,
            "is_superuser": False,
            "roles": [],
            "permissions": [],
        }

    async def fake_list_memberships(user_id: int):
        return []

    monkeypatch.setattr(user_handling, "get_api_key_manager", fake_get_api_key_manager)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.DB_Management.Users_DB.get_user_by_id",
        fake_get_user_by_id,
    )
    monkeypatch.setattr(user_handling, "list_memberships_for_user", fake_list_memberships)
    monkeypatch.setattr(user_handling, "set_scope", lambda *_, **__: None)

    request = _build_request()
    user = await user_handling.get_request_user(request, api_key="valid-key", token=None)
    assert user.id == 77
    assert user.is_active is True

    for env_key in ("AUTH_MODE", "DATABASE_URL"):
        monkeypatch.delenv(env_key, raising=False)
    reset_settings()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "auth_mode,expected_subject",
    [
        ("multi_user", None),
        ("single_user", "single_user"),
    ],
)
async def test_api_key_principal_subject_single_user_only_in_single_user_mode(
    monkeypatch,
    auth_mode: str,
    expected_subject: str | None,
):
    monkeypatch.setenv("AUTH_MODE", auth_mode)
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "77")
    reset_settings()

    class StubAPIKeyManager:
        async def validate_api_key(self, api_key: str, ip_address=None):
            return {"id": 5, "user_id": 77}

    async def fake_get_api_key_manager():
        return StubAPIKeyManager()

    async def fake_get_user_by_id(user_id: int):
        return {
            "id": user_id,
            "username": "active-user",
            "email": "active@example.com",
            "is_active": 1,
            "is_superuser": True,
            "roles": [],
            "permissions": [],
        }

    async def fake_list_memberships(user_id: int):
        return []

    monkeypatch.setattr(user_handling, "get_api_key_manager", fake_get_api_key_manager)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.DB_Management.Users_DB.get_user_by_id",
        fake_get_user_by_id,
    )
    monkeypatch.setattr(user_handling, "list_memberships_for_user", fake_list_memberships)
    monkeypatch.setattr(user_handling, "set_scope", lambda *_, **__: None)

    request = _build_request()
    await user_handling.get_request_user(request, api_key="valid-key", token=None)
    ctx = getattr(request.state, "auth", None)
    assert ctx is not None
    assert getattr(ctx.principal, "subject", None) == expected_subject

    for env_key in ("AUTH_MODE", "DATABASE_URL", "SINGLE_USER_FIXED_ID"):
        monkeypatch.delenv(env_key, raising=False)
    reset_settings()
