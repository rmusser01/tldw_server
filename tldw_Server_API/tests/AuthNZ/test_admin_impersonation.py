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

        # Mock cursor for user lookup
        mock_cursor_user = AsyncMock()
        mock_cursor_user.fetchone = AsyncMock(return_value=(42, "targetuser", 1))

        # Mock cursor for role lookup
        mock_cursor_role = AsyncMock()
        mock_cursor_role.fetchone = AsyncMock(return_value=("user",))

        mock_conn = AsyncMock()
        # First call returns user info, second returns role
        mock_conn.execute = AsyncMock(side_effect=[mock_cursor_user, mock_cursor_role])
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_conn)

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
        ):
            result = await create_impersonation_token(42, principal)

        assert result.token == "mock.jwt.token"
        assert result.impersonated_user_id == 42
        assert result.impersonated_by == 1

        # Verify JWT was created with impersonation claims
        mock_jwt_svc.create_access_token.assert_called_once()
        call_kwargs = mock_jwt_svc.create_access_token.call_args
        additional = call_kwargs.kwargs.get("additional_claims") or call_kwargs[1].get("additional_claims")
        assert additional["impersonated_by"] == 1
        assert additional["impersonation"] is True

    @pytest.mark.asyncio
    async def test_user_not_found(self):
        principal = _admin_principal()

        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=None)

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value=mock_cursor)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_conn)

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

        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=(42, "inactive", 0))  # is_active=0

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value=mock_cursor)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_conn)

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
