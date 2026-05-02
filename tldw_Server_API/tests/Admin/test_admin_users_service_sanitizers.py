from types import SimpleNamespace

import pytest
from fastapi import HTTPException, status

from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    AdminMfaRequirementRequest,
    AdminPasswordResetRequest,
    AdminUserCreateRequest,
    UserUpdateRequest,
)
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    DuplicateUserError,
    RegistrationDisabledError,
    RegistrationError,
    WeakPasswordError,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.services import admin_users_service as service


pytestmark = pytest.mark.unit


class _RegistrationService:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def register_user(self, **_kwargs):
        raise self.error


class _LeakyRegistrationDisabledError(RegistrationDisabledError):
    def __init__(self) -> None:
        Exception.__init__(self, "registration disabled token at /private/authnz-config.txt")


def _principal() -> AuthPrincipal:
    return AuthPrincipal(kind="user", user_id=7, roles=["admin"], permissions=["*"], is_admin=True)


def _payload() -> AdminUserCreateRequest:
    temporary_secret = "Temporary" + "Pass123!"
    return AdminUserCreateRequest(
        username="newuser",
        email="newuser@example.com",
        password=temporary_secret,
        role="user",
    )


async def _assert_admin_user_500_log_sanitized(call, expected_detail: str, expected_log: str, raw_marker: str) -> None:
    messages: list[str] = []
    sink_id = service.logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        with pytest.raises(HTTPException) as exc_info:
            await call()
    finally:
        service.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == expected_detail
    assert expected_log in joined
    assert raw_marker not in joined
    assert "/private/" not in joined


@pytest.mark.asyncio
async def test_create_user_sanitizes_generic_failure_log(monkeypatch) -> None:
    monkeypatch.setattr(service, "get_profile", lambda: "multi_user")

    await _assert_admin_user_500_log_sanitized(
        lambda: service.create_user(
            _payload(),
            _principal(),
            _RegistrationService(RuntimeError("create user failed at /private/users.db")),
        ),
        "Failed to create user",
        "Failed to create user",
        "create user failed",
    )


@pytest.mark.asyncio
async def test_list_users_sanitizes_generic_failure_log(monkeypatch) -> None:
    async def _allow_all_orgs(_principal):
        return None

    async def _raise_from_pool():
        raise RuntimeError("list users failed at /private/users.db")

    monkeypatch.setattr(service.admin_scope_service, "get_admin_org_ids", _allow_all_orgs)
    monkeypatch.setattr(service, "AuthnzUsersRepo", SimpleNamespace(from_pool=_raise_from_pool))

    await _assert_admin_user_500_log_sanitized(
        lambda: service.list_users(
            _principal(),
            page=1,
            limit=10,
            role=None,
            admin_capable=False,
            is_active=None,
            search=None,
            org_id=None,
        ),
        "Failed to retrieve users",
        "Failed to list users",
        "list users failed",
    )


@pytest.mark.asyncio
async def test_export_users_sanitizes_generic_failure_log(monkeypatch) -> None:
    async def _allow_all_orgs(_principal):
        return None

    async def _raise_from_pool():
        raise RuntimeError("export users failed at /private/users.db")

    monkeypatch.setattr(service.admin_scope_service, "get_admin_org_ids", _allow_all_orgs)
    monkeypatch.setattr(service, "AuthnzUsersRepo", SimpleNamespace(from_pool=_raise_from_pool))

    await _assert_admin_user_500_log_sanitized(
        lambda: service.export_users(
            _principal(),
            role=None,
            is_active=None,
            search=None,
            org_id=None,
            limit=10,
            offset=0,
            format="json",
        ),
        "Failed to export users",
        "Failed to export users",
        "export users failed",
    )


@pytest.mark.asyncio
async def test_get_user_details_sanitizes_generic_failure_log(monkeypatch) -> None:
    async def _allow_scope(*_args, **_kwargs) -> None:
        return None

    async def _raise_from_pool():
        raise RuntimeError("get user failed at /private/users.db")

    monkeypatch.setattr(service.admin_scope_service, "enforce_admin_user_scope", _allow_scope)
    monkeypatch.setattr(service, "AuthnzUsersRepo", SimpleNamespace(from_pool=_raise_from_pool))

    await _assert_admin_user_500_log_sanitized(
        lambda: service.get_user_details(_principal(), 42),
        "Failed to retrieve user details",
        "Failed to get user",
        "get user failed",
    )


@pytest.mark.asyncio
async def test_update_user_sanitizes_generic_failure_log(monkeypatch) -> None:
    async def _allow_scope(*_args, **_kwargs) -> None:
        return None

    async def _raise_is_pg() -> bool:
        raise RuntimeError("update user failed at /private/users.db")

    monkeypatch.setattr(service.admin_scope_service, "enforce_admin_user_scope", _allow_scope)

    await _assert_admin_user_500_log_sanitized(
        lambda: service.update_user(
            _principal(),
            42,
            UserUpdateRequest(email="newuser@example.com"),
            db=object(),
            password_service=object(),
            is_pg_fn=_raise_is_pg,
        ),
        "Failed to update user",
        "Failed to update user",
        "update user failed",
    )


@pytest.mark.asyncio
async def test_reset_user_password_sanitizes_generic_failure_log(monkeypatch) -> None:
    async def _allow_scope(*_args, **_kwargs) -> None:
        return None

    async def _allow_reauth(*_args, **_kwargs) -> str:
        return "Support case 123"

    async def _raise_is_pg() -> bool:
        raise RuntimeError("reset password failed at /private/users.db")

    temporary_secret = "Temp" + "Pass123!"
    monkeypatch.setattr(service.admin_scope_service, "enforce_admin_user_scope", _allow_scope)
    monkeypatch.setattr(service, "verify_privileged_action", _allow_reauth)
    monkeypatch.setattr(service, "hash_password", lambda _password: "hashed-password")

    await _assert_admin_user_500_log_sanitized(
        lambda: service.reset_user_password(
            _principal(),
            42,
            AdminPasswordResetRequest(
                reason="Support case 123",
                temporary_password=temporary_secret,
                force_password_change=True,
            ),
            db=object(),
            password_service=object(),
            is_pg_fn=_raise_is_pg,
        ),
        "Failed to reset password",
        "Failed to reset password",
        "reset password failed",
    )


@pytest.mark.asyncio
async def test_set_user_mfa_requirement_sanitizes_generic_failure_log(monkeypatch) -> None:
    async def _allow_scope(*_args, **_kwargs) -> None:
        return None

    async def _allow_reauth(*_args, **_kwargs) -> str:
        return "Support case 123"

    async def _raise_is_pg() -> bool:
        raise RuntimeError("mfa requirement failed at /private/users.db")

    monkeypatch.setattr(service.admin_scope_service, "enforce_admin_user_scope", _allow_scope)
    monkeypatch.setattr(service, "verify_privileged_action", _allow_reauth)

    await _assert_admin_user_500_log_sanitized(
        lambda: service.set_user_mfa_requirement(
            _principal(),
            42,
            AdminMfaRequirementRequest(require_mfa=True, reason="Support case 123"),
            db=object(),
            password_service=object(),
            is_pg_fn=_raise_is_pg,
        ),
        "Failed to update MFA requirement",
        "Failed to update MFA requirement",
        "mfa requirement failed",
    )


@pytest.mark.asyncio
async def test_delete_user_sanitizes_generic_failure_log(monkeypatch) -> None:
    async def _allow_scope(*_args, **_kwargs) -> None:
        return None

    async def _allow_reauth(*_args, **_kwargs) -> str:
        return "Support case 123"

    async def _raise_is_pg() -> bool:
        raise RuntimeError("delete user failed at /private/users.db")

    monkeypatch.setattr(service.admin_scope_service, "enforce_admin_user_scope", _allow_scope)
    monkeypatch.setattr(service, "verify_privileged_action", _allow_reauth)

    await _assert_admin_user_500_log_sanitized(
        lambda: service.delete_user(
            _principal(),
            42,
            SimpleNamespace(reason="Support case 123"),
            db=object(),
            password_service=object(),
            is_pg_fn=_raise_is_pg,
        ),
        "Failed to delete user",
        "Failed to delete user",
        "delete user failed",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "status_code", "detail", "raw_token"),
    [
        (
            DuplicateUserError("registration database path /private/users.db"),
            status.HTTP_409_CONFLICT,
            "User already exists.",
            "/private/users.db",
        ),
        (
            WeakPasswordError("password config token at /private/password-policy.txt"),
            status.HTTP_400_BAD_REQUEST,
            "Password does not meet requirements.",
            "/private/password-policy.txt",
        ),
        (
            _LeakyRegistrationDisabledError(),
            status.HTTP_403_FORBIDDEN,
            "Registration is currently disabled.",
            "/private/authnz-config.txt",
        ),
        (
            RegistrationError("registration backend token at /private/registration.db"),
            status.HTTP_400_BAD_REQUEST,
            "Registration failed.",
            "/private/registration.db",
        ),
    ],
)
async def test_create_user_sanitizes_registration_errors(
    monkeypatch,
    error: Exception,
    status_code: int,
    detail: str,
    raw_token: str,
) -> None:
    monkeypatch.setattr(service, "get_profile", lambda: "multi_user")

    with pytest.raises(HTTPException) as exc_info:
        await service.create_user(_payload(), _principal(), _RegistrationService(error))

    assert exc_info.value.status_code == status_code
    assert exc_info.value.detail == detail
    assert raw_token not in exc_info.value.detail
