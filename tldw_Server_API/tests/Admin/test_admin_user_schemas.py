import pytest
from pydantic import SecretStr, ValidationError

from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    AdminUserCreateRequest,
    AdminPasswordResetRequest,
    AdminPrivilegedActionRequest,
    AdminPasswordResetResponse,
    RegistrationCodeRequest,
    UserUpdateRequest,
)


pytestmark = pytest.mark.unit


def test_user_update_request_accepts_backend_role_vocabulary() -> None:
    payload = UserUpdateRequest(role="user")

    assert payload.role == "user"


def test_user_update_request_rejects_legacy_member_role() -> None:
    with pytest.raises(ValidationError):
        UserUpdateRequest(role="member")


@pytest.mark.parametrize(
    ("model", "kwargs"),
    [
        (UserUpdateRequest, {"role": "service"}),
        (
            AdminUserCreateRequest,
            {
                "username": "service-user",
                "email": "service@example.com",
                "password": "StrongPass123!",
                "role": "service",
            },
        ),
        (RegistrationCodeRequest, {"role_to_grant": "service"}),
    ],
)
def test_admin_user_contract_rejects_service_principal_as_human_role(model, kwargs) -> None:
    with pytest.raises(ValidationError):
        model(**kwargs)


@pytest.mark.parametrize("role", ["user", "admin"])
def test_admin_user_contract_accepts_seeded_system_roles(role: str) -> None:
    assert UserUpdateRequest(role=role).role == role
    assert AdminUserCreateRequest(
        username=f"{role}-account",
        email=f"{role}@example.com",
        password="StrongPass123!",
        role=role,
    ).role == role
    assert RegistrationCodeRequest(role_to_grant=role).role_to_grant == role


def test_admin_password_reset_response_omits_plaintext_password() -> None:
    payload = AdminPasswordResetResponse(
        user_id=42,
        force_password_change=True,
        message="Password reset successfully",
    )

    assert payload.model_dump() == {
        "user_id": 42,
        "force_password_change": True,
        "message": "Password reset successfully",
    }


def test_admin_password_reset_request_requires_temporary_password() -> None:
    with pytest.raises(ValidationError):
        AdminPasswordResetRequest(
            reason="Support case 123",
            admin_password="AdminPass123!",
        )


def test_admin_privileged_action_request_allows_blank_admin_password_for_single_user_mode() -> None:
    payload = AdminPrivilegedActionRequest(
        reason="Support case 123",
        admin_password="",
        admin_reauth_token="",
    )

    assert payload.admin_password is None
    assert payload.admin_reauth_token is None


def test_user_update_request_allows_blank_admin_password_for_single_user_mode() -> None:
    payload = UserUpdateRequest(
        is_active=False,
        reason="Support case 123",
        admin_password="",
    )

    assert payload.admin_password is None


def test_admin_privileged_action_request_repr_redacts_sensitive_fields() -> None:
    payload = AdminPrivilegedActionRequest(
        reason="Support case 123",
        admin_password="AdminPass123!",
        admin_reauth_token="reauth-token-123",
    )

    rendered = repr(payload)

    assert "AdminPass123!" not in rendered
    assert "reauth-token-123" not in rendered


def test_admin_privileged_action_request_stores_sensitive_fields_as_secretstr() -> None:
    payload = AdminPrivilegedActionRequest(
        reason="Support case 123",
        admin_password="AdminPass123!",
        admin_reauth_token="reauth-token-123",
    )

    assert isinstance(payload.admin_password, SecretStr)
    assert payload.admin_password.get_secret_value() == "AdminPass123!"
    assert isinstance(payload.admin_reauth_token, SecretStr)
    assert payload.admin_reauth_token.get_secret_value() == "reauth-token-123"
