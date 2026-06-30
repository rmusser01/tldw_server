"""Tests for admin impersonation endpoint."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.admin.admin_impersonation import (
    ImpersonationTokenResponse,
    create_impersonation_token,
)
from tldw_Server_API.app.core.Audit.unified_audit_service import MandatoryAuditWriteError
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal


class _LoggerStub:
    def __init__(self) -> None:
        self.error_records: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.error_records.append((message, args, kwargs))


def _admin_principal() -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        username="admin",
        roles=["admin"],
        is_admin=True,
    )


class TestImpersonationTokenResponse:
    def test_defaults(self):
        resp = ImpersonationTokenResponse(
            token="jwt.token.here",  # nosec B106
            impersonated_user_id=42,
            impersonated_by=1,
        )
        assert resp.token_type == "bearer"  # nosec
        assert resp.expires_in_minutes == 15  # nosec B101


class TestCreateImpersonationToken:
    @pytest.mark.asyncio
    async def test_success_uses_repositories_short_ttl_and_mandatory_audit(self):
        principal = _admin_principal()

        class UsersRepoStub:
            @classmethod
            async def from_pool(cls):
                return cls()

            async def get_user_by_id(self, user_id: int):
                assert user_id == 42  # nosec B101
                return {"id": 42, "username": "targetuser", "is_active": True, "role": "legacy"}

        class RbacRepoStub:
            def get_user_roles(self, user_id: int):
                assert user_id == 42  # nosec B101
                return [{"name": "user"}]

        mock_jwt_svc = MagicMock()
        mock_jwt_svc.create_impersonation_access_token = MagicMock(return_value="mock.jwt.token")
        audit = AsyncMock()

        with (
            patch(
                "tldw_Server_API.app.api.v1.endpoints.admin.admin_impersonation.AuthnzUsersRepo",
                UsersRepoStub,
            ),
            patch(
                "tldw_Server_API.app.api.v1.endpoints.admin.admin_impersonation.AuthnzRbacRepo",
                return_value=RbacRepoStub(),
            ),
            patch(
                "tldw_Server_API.app.api.v1.endpoints.admin.admin_impersonation.get_jwt_service",
                return_value=mock_jwt_svc,
            ),
            patch(
                "tldw_Server_API.app.api.v1.endpoints.admin.admin_impersonation.emit_impersonation_issuance_audit_event",
                audit,
            ),
        ):
            result = await create_impersonation_token(42, principal)

        assert result.token == "mock.jwt.token"  # nosec
        assert result.impersonated_user_id == 42  # nosec B101
        assert result.impersonated_by == 1  # nosec B101
        mock_jwt_svc.create_impersonation_access_token.assert_called_once()
        token_kwargs = mock_jwt_svc.create_impersonation_access_token.call_args.kwargs
        assert token_kwargs["user_id"] == 42  # nosec B101
        assert token_kwargs["username"] == "targetuser"  # nosec B101
        assert token_kwargs["role"] == "user"  # nosec B101
        assert token_kwargs["impersonated_by"] == 1  # nosec B101
        assert token_kwargs["expires_delta"].total_seconds() == 15 * 60  # nosec B101
        audit.assert_awaited_once_with(
            actor_id=1,
            target_user_id=42,
            expires_in_minutes=15,
        )

    @pytest.mark.asyncio
    async def test_mandatory_audit_failure_returns_503(self):
        principal = _admin_principal()

        class UsersRepoStub:
            @classmethod
            async def from_pool(cls):
                return cls()

            async def get_user_by_id(self, user_id: int):
                return {"id": 42, "username": "targetuser", "is_active": True, "role": "user"}

        class RbacRepoStub:
            def get_user_roles(self, user_id: int):
                return []

        mock_jwt_svc = MagicMock()
        mock_jwt_svc.create_impersonation_access_token = MagicMock(return_value="mock.jwt.token")

        with (
            patch(
                "tldw_Server_API.app.api.v1.endpoints.admin.admin_impersonation.AuthnzUsersRepo",
                UsersRepoStub,
            ),
            patch(
                "tldw_Server_API.app.api.v1.endpoints.admin.admin_impersonation.AuthnzRbacRepo",
                return_value=RbacRepoStub(),
            ),
            patch(
                "tldw_Server_API.app.api.v1.endpoints.admin.admin_impersonation.get_jwt_service",
                return_value=mock_jwt_svc,
            ),
            patch(
                "tldw_Server_API.app.api.v1.endpoints.admin.admin_impersonation.emit_impersonation_issuance_audit_event",
                AsyncMock(side_effect=MandatoryAuditWriteError("Mandatory audit persistence unavailable")),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await create_impersonation_token(42, principal)

        assert exc_info.value.status_code == 503  # nosec B101
        assert exc_info.value.detail == "Mandatory audit persistence unavailable"  # nosec B101

    @pytest.mark.asyncio
    async def test_user_not_found(self):
        principal = _admin_principal()

        class UsersRepoStub:
            @classmethod
            async def from_pool(cls):
                return cls()

            async def get_user_by_id(self, user_id: int):
                assert user_id == 999  # nosec B101
                return None

        with patch(
            "tldw_Server_API.app.api.v1.endpoints.admin.admin_impersonation.AuthnzUsersRepo",
            UsersRepoStub,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await create_impersonation_token(999, principal)
            assert exc_info.value.status_code == 404  # nosec B101

    @pytest.mark.asyncio
    async def test_inactive_user_rejected(self):
        principal = _admin_principal()

        class UsersRepoStub:
            @classmethod
            async def from_pool(cls):
                return cls()

            async def get_user_by_id(self, user_id: int):
                assert user_id == 42  # nosec B101
                return {"id": 42, "username": "inactive", "is_active": False, "role": "user"}

        with patch(
            "tldw_Server_API.app.api.v1.endpoints.admin.admin_impersonation.AuthnzUsersRepo",
            UsersRepoStub,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await create_impersonation_token(42, principal)
            assert exc_info.value.status_code == 400  # nosec B101

    @pytest.mark.asyncio
    async def test_sanitizes_generic_failure(self, monkeypatch: pytest.MonkeyPatch):
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_impersonation

        principal = _admin_principal()
        logger_stub = _LoggerStub()
        monkeypatch.setattr(admin_impersonation, "logger", logger_stub)

        class UsersRepoStub:
            @classmethod
            async def from_pool(cls):
                raise RuntimeError("impersonation backend exploded at /private/impersonation.db")

        with patch(
            "tldw_Server_API.app.api.v1.endpoints.admin.admin_impersonation.AuthnzUsersRepo",
            UsersRepoStub,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await create_impersonation_token(42, principal)

        assert exc_info.value.status_code == 500  # nosec B101
        assert exc_info.value.detail == "Impersonation token creation failed"  # nosec B101
        assert logger_stub.error_records == [("Impersonation token creation failed", (), {})]  # nosec B101
