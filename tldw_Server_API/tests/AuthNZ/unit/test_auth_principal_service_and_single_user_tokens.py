from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException, Response
from fastapi.security import HTTPAuthorizationCredentials
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.core.AuthNZ import User_DB_Handling, auth_principal_resolver
from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService, reset_jwt_service
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.AuthNZ.single_user_session import SingleUserSessionIdentity
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user


def _build_request(headers: dict[str, str], client_host: str = "127.0.0.1") -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/test",
        "headers": [
            (k.lower().encode("ascii"), v.encode("ascii")) for k, v in headers.items()
        ],
        "client": (client_host, 0),
    }
    return Request(scope)


@pytest.mark.asyncio
async def test_get_auth_principal_accepts_single_user_cookie_session(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "tldw_test_key_123456")
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")
    reset_settings()

    identity = SingleUserSessionIdentity(
        session_id=41,
        user_id=1,
        expires_at=datetime.now(timezone.utc) + timedelta(days=1),
    )
    validate = AsyncMock(return_value=identity)
    monkeypatch.setattr(
        auth_principal_resolver,
        "validate_single_user_session",
        validate,
        raising=False,
    )
    request = _build_request({"Cookie": "tldw_single_user_session=opaque"})

    principal = await auth_principal_resolver.get_auth_principal(request)

    assert principal.kind == "user"
    assert principal.subject == "single_user"
    assert principal.token_type == "single_user_session"
    assert request.state.single_user_session_id == 41
    assert request.state.user_id == 1
    validate.assert_awaited_once_with(request)
    reset_settings()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "credential_header",
    [
        {"X-API-KEY": "tldw_test_key_123456"},
        {"Authorization": "Bearer tldw_test_key_123456"},
    ],
)
async def test_explicit_credentials_take_precedence_over_cookie_session(
    monkeypatch,
    credential_header,
):
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "tldw_test_key_123456")
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")
    reset_settings()

    validate = AsyncMock()
    monkeypatch.setattr(
        auth_principal_resolver,
        "validate_single_user_session",
        validate,
        raising=False,
    )
    request = _build_request(
        {**credential_header, "Cookie": "tldw_single_user_session=opaque"}
    )

    principal = await auth_principal_resolver.get_auth_principal(request)

    assert principal.token_type == "api_key"
    validate.assert_not_awaited()
    reset_settings()


@pytest.mark.asyncio
async def test_get_auth_principal_accepts_service_token(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-service-secret-1234567890-abcdef")
    reset_settings()
    reset_jwt_service()

    jwt_service = JWTService()
    token = jwt_service.create_service_account_token(
        service_name="worker",
        permissions=["tools.execute:foo", "admin"],
    )
    request = _build_request({"Authorization": f"Bearer {token}"})

    principal = await auth_principal_resolver.get_auth_principal(request)

    assert principal.kind == "service"
    assert principal.subject == "service:worker"
    assert "tools.execute:foo" in principal.permissions
    assert principal.is_admin is True
    reset_settings()
    reset_jwt_service()


@pytest.mark.asyncio
async def test_get_auth_principal_rejects_service_token_from_nonlocal(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-service-secret-1234567890-abcdef")
    reset_settings()
    reset_jwt_service()

    jwt_service = JWTService()
    token = jwt_service.create_service_account_token(
        service_name="worker",
        permissions=["tools.execute:foo"],
    )
    request = _build_request(
        {"Authorization": f"Bearer {token}"},
        client_host="203.0.113.10",
    )

    with pytest.raises(HTTPException) as exc:
        await auth_principal_resolver.get_auth_principal(request)

    assert exc.value.status_code == 403
    reset_settings()
    reset_jwt_service()


@pytest.mark.asyncio
async def test_get_current_user_accepts_bearer_api_key_single_user(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "tldw_test_key_123456")
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")
    reset_settings()

    token = "tldw_test_key_123456"
    request = _build_request({"Authorization": f"Bearer {token}"})
    response = Response()
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    user = await auth_deps.get_current_user(
        request=request,
        response=response,
        credentials=creds,
        x_api_key=None,
    )

    assert user["id"] == 1
    assert "admin" in (user.get("roles") or [])
    reset_settings()


@pytest.mark.asyncio
async def test_get_auth_principal_accepts_bearer_api_key_single_user(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "tldw_test_key_123456")
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")
    reset_settings()

    token = "tldw_test_key_123456"
    request = _build_request({"Authorization": f"Bearer {token}"})

    principal = await auth_principal_resolver.get_auth_principal(request)

    assert principal.kind == "user"
    assert principal.subject == "single_user"
    assert principal.user_id == 1
    reset_settings()


@pytest.mark.asyncio
async def test_get_request_user_accepts_bearer_api_key_single_user(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "tldw_test_key_123456")
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")
    reset_settings()

    token = "tldw_test_key_123456"
    request = _build_request({"Authorization": f"Bearer {token}"})

    user = await get_request_user(request, token=token)

    assert getattr(user, "id_int", None) == 1
    reset_settings()


@pytest.mark.asyncio
async def test_single_user_legacy_bearer_token_rejected(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "tldw_test_key_123456")
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")
    reset_settings()

    token = "single-user-token-1"
    request = _build_request({"Authorization": f"Bearer {token}"})
    response = Response()
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    with pytest.raises(HTTPException) as exc:
        await auth_deps.get_current_user(
            request=request,
            response=response,
            credentials=creds,
            x_api_key=None,
        )

    assert exc.value.status_code == 401

    with pytest.raises(HTTPException) as exc2:
        await auth_principal_resolver.get_auth_principal(request)
    assert exc2.value.status_code == 401

    with pytest.raises(HTTPException) as exc3:
        await get_request_user(request, token=token)
    assert exc3.value.status_code == 401
    reset_settings()


@pytest.mark.asyncio
async def test_get_auth_principal_rejects_legacy_token_header(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    reset_settings()

    request = _build_request({"Token": "Bearer legacy-key"})

    with pytest.raises(HTTPException) as exc:
        await auth_principal_resolver.get_auth_principal(request)

    assert exc.value.status_code == 401
    reset_settings()


@pytest.mark.asyncio
async def test_single_user_test_key_rejected_outside_test_context_for_auth_principal(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "primary-key-123456")
    monkeypatch.setenv("SINGLE_USER_TEST_API_KEY", "test-key-abcdef")
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")
    reset_settings()

    monkeypatch.setattr(auth_principal_resolver, "_is_test_context", lambda: False)
    monkeypatch.setattr(User_DB_Handling, "_is_test_context", lambda: False)

    class _StubApiKeyManager:
        async def validate_api_key(self, **_kwargs):
            return None

    async def _stub_get_api_key_manager():
        return _StubApiKeyManager()

    monkeypatch.setattr(User_DB_Handling, "get_api_key_manager", _stub_get_api_key_manager)

    request = _build_request({"Authorization": "Bearer test-key-abcdef"})
    with pytest.raises(HTTPException) as exc:
        await auth_principal_resolver.get_auth_principal(request)

    assert exc.value.status_code == 401
    reset_settings()


@pytest.mark.asyncio
async def test_single_user_test_key_rejected_outside_test_context_for_get_request_user(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "primary-key-123456")
    monkeypatch.setenv("SINGLE_USER_TEST_API_KEY", "test-key-abcdef")
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")
    reset_settings()

    monkeypatch.setattr(User_DB_Handling, "_is_test_context", lambda: False)

    class _StubApiKeyManager:
        async def validate_api_key(self, **_kwargs):
            return None

    async def _stub_get_api_key_manager():
        return _StubApiKeyManager()

    monkeypatch.setattr(User_DB_Handling, "get_api_key_manager", _stub_get_api_key_manager)

    request = _build_request({})
    with pytest.raises(HTTPException) as exc:
        await get_request_user(request, api_key="test-key-abcdef", token=None)

    assert exc.value.status_code == 401
    reset_settings()


@pytest.mark.asyncio
async def test_get_request_user_bypass_requires_strict_test_context(monkeypatch):
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("EVALS_HEAVY_ADMIN_ONLY", "false")
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("TLDW_TEST_MODE", raising=False)
    reset_settings()

    monkeypatch.setattr(User_DB_Handling, "_is_strict_test_bypass_context", lambda: False)

    request = _build_request({})
    with pytest.raises(HTTPException) as exc:
        await get_request_user(request, api_key=None, token=None)

    assert exc.value.status_code == 401
    reset_settings()


@pytest.mark.asyncio
async def test_get_request_user_bypass_allows_strict_test_context(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("EVALS_HEAVY_ADMIN_ONLY", "false")
    monkeypatch.setenv("SINGLE_USER_TEST_API_KEY", "test-key-abcdef")
    reset_settings()

    request = _build_request({})
    user = await get_request_user(request, api_key=None, token=None)

    assert getattr(user, "username", None) == "single_user"
    reset_settings()


@pytest.mark.asyncio
async def test_get_request_user_bypass_blocked_in_environment_production(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("EVALS_HEAVY_ADMIN_ONLY", "false")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("tldw_production", raising=False)
    reset_settings()

    request = _build_request({})
    with pytest.raises(HTTPException) as exc:
        await get_request_user(request, api_key=None, token=None)

    assert exc.value.status_code == 401
    reset_settings()


def test_user_db_is_test_context_accepts_single_letter_y(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_MODE", "y")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    assert User_DB_Handling._is_test_context() is True


def test_user_db_strict_test_bypass_context_accepts_single_letter_y(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "y")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    assert User_DB_Handling._is_strict_test_bypass_context() is True


def test_auth_principal_resolver_is_test_context_accepts_single_letter_y(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.setenv("TESTING", "y")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    assert auth_principal_resolver._is_test_context() is True
