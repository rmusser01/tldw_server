"""Tests for admin impersonation endpoint."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.core.Audit.unified_audit_service import AuditEventCategory, AuditEventType
from tldw_Server_API.app.api.v1.endpoints.admin.admin_impersonation import (
    ImpersonationTokenResponse,
    create_impersonation_token,
)
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
            token="jwt.token.here",
            impersonated_user_id=42,
            impersonated_by=1,
        )
        assert resp.token_type == "bearer"
        assert resp.expires_in_minutes == 15


class TestCreateImpersonationToken:
    @pytest.mark.asyncio
    async def test_success(self):
        principal = _admin_principal()

        mock_pool = MagicMock()
        mock_pool.fetchone = AsyncMock(
            side_effect=[
                (42, "targetuser", 1),
                ("user",),
            ],
        )

        mock_jwt_svc = MagicMock()
        mock_jwt_svc.create_access_token = MagicMock(return_value="mock.jwt.token")
        mock_audit = AsyncMock()

        with (
            patch(
                "tldw_Server_API.app.core.AuthNZ.database.get_db_pool",
                new_callable=AsyncMock,
                return_value=mock_pool,
            ),
            patch(
                "tldw_Server_API.app.core.AuthNZ.jwt_service.get_jwt_service",
                return_value=mock_jwt_svc,
            ),
            patch(
                "tldw_Server_API.app.api.v1.endpoints.admin.admin_impersonation._emit_admin_account_audit_event",
                new=mock_audit,
                create=True,
            ),
        ):
            result = await create_impersonation_token(42, principal)

        assert result.token == "mock.jwt.token"
        assert result.impersonated_user_id == 42
        assert result.impersonated_by == 1

        # Verify JWT was created with impersonation claims
        mock_jwt_svc.create_access_token.assert_called_once()
        call_kwargs = mock_jwt_svc.create_access_token.call_args
        assert call_kwargs.kwargs["expires_in_minutes"] == 15
        additional = call_kwargs.kwargs.get("additional_claims") or call_kwargs[1].get("additional_claims")
        assert additional["impersonated_by"] == 1
        assert additional["impersonation"] is True
        mock_audit.assert_awaited_once()
        audit_kwargs = mock_audit.await_args.kwargs
        assert audit_kwargs["actor_id"] == 1
        assert audit_kwargs["target_user_id"] == 42
        assert audit_kwargs["event_type"] is AuditEventType.AUTH_TOKEN_CREATED
        assert audit_kwargs["category"] is AuditEventCategory.AUTHENTICATION
        assert audit_kwargs["action"] == "admin.impersonation.token.create"
        assert audit_kwargs["metadata"]["impersonation"] is True
        assert audit_kwargs["metadata"]["impersonation_ttl_minutes"] == 15

    @pytest.mark.asyncio
    async def test_uses_backend_neutral_fetchone_for_user_and_role_lookups(self):
        principal = _admin_principal()

        mock_pool = MagicMock()
        mock_pool.fetchone = AsyncMock(
            side_effect=[
                (42, "targetuser", 1),
                ("user",),
            ],
        )
        mock_pool.acquire = MagicMock(side_effect=AssertionError("raw connection lookup should not run"))

        mock_jwt_svc = MagicMock()
        mock_jwt_svc.create_access_token = MagicMock(return_value="mock.jwt.token")

        with (
            patch(
                "tldw_Server_API.app.core.AuthNZ.database.get_db_pool",
                new_callable=AsyncMock,
                return_value=mock_pool,
            ),
            patch(
                "tldw_Server_API.app.core.AuthNZ.jwt_service.get_jwt_service",
                return_value=mock_jwt_svc,
            ),
            patch(
                "tldw_Server_API.app.api.v1.endpoints.admin.admin_impersonation._emit_admin_account_audit_event",
                new_callable=AsyncMock,
                create=True,
            ),
        ):
            result = await create_impersonation_token(42, principal)

        assert result.impersonated_user_id == 42
        assert mock_pool.fetchone.await_count == 2
        mock_pool.acquire.assert_not_called()

    @pytest.mark.asyncio
    async def test_user_not_found(self):
        principal = _admin_principal()

        mock_pool = MagicMock()
        mock_pool.fetchone = AsyncMock(return_value=None)

        from fastapi import HTTPException

        with patch(
            "tldw_Server_API.app.core.AuthNZ.database.get_db_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await create_impersonation_token(999, principal)
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_inactive_user_rejected(self):
        principal = _admin_principal()

        mock_pool = MagicMock()
        mock_pool.fetchone = AsyncMock(return_value=(42, "inactive", 0))  # is_active=0

        from fastapi import HTTPException

        with patch(
            "tldw_Server_API.app.core.AuthNZ.database.get_db_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await create_impersonation_token(42, principal)
            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_sanitizes_generic_failure(self, monkeypatch: pytest.MonkeyPatch):
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_impersonation

        principal = _admin_principal()
        logger_stub = _LoggerStub()
        monkeypatch.setattr(admin_impersonation, "logger", logger_stub)

        with patch(
            "tldw_Server_API.app.core.AuthNZ.database.get_db_pool",
            new_callable=AsyncMock,
            side_effect=RuntimeError("impersonation backend exploded at /private/impersonation.db"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await create_impersonation_token(42, principal)

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Impersonation token creation failed"
        assert logger_stub.error_records == [("Impersonation token creation failed", (), {})]
