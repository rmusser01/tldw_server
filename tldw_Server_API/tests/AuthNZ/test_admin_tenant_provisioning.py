"""Tests for admin tenant provisioning endpoint."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.endpoints.admin.admin_tenant_provisioning import (
    TenantProvisionRequest,
    TenantProvisionResponse,
    router,
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


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/admin")
    return app


class TestTenantProvisionRequest:
    def test_valid_request(self):
        req = TenantProvisionRequest(
            username="newuser",
            email="new@example.com",
            password="securepass123",
            org_name="NewOrg",
        )
        assert req.role == "owner"

    def test_custom_role(self):
        req = TenantProvisionRequest(
            username="newuser",
            email="new@example.com",
            password="securepass123",
            org_name="NewOrg",
            role="member",
        )
        assert req.role == "member"

    def test_password_too_short(self):
        with pytest.raises(ValidationError):
            TenantProvisionRequest(
                username="u",
                email="e@x.com",
                password="short",
                org_name="Org",
            )


class TestTenantProvisionResponse:
    def test_default_message(self):
        resp = TenantProvisionResponse(
            user_id=1,
            username="user",
            org_id=10,
            org_name="Org",
            role="owner",
        )
        assert resp.message == "Tenant provisioned successfully"


class TestProvisionEndpointUnit:
    """Unit tests for the provision_tenant endpoint logic using mocks."""

    @pytest.mark.asyncio
    async def test_provision_calls_steps(self):
        """Verify the endpoint function orchestrates user+org+member creation."""
        from tldw_Server_API.app.api.v1.endpoints.admin.admin_tenant_provisioning import (
            provision_tenant,
        )

        payload = TenantProvisionRequest(
            username="tenant_user",
            email="tenant@example.com",
            password="securepass123",
            org_name="TenantOrg",
        )
        principal = _admin_principal()

        # Build mock connection & pool
        lookup_cursor = AsyncMock()
        lookup_cursor.fetchone = AsyncMock(return_value=None)
        user_cursor = AsyncMock(lastrowid=42, rowcount=1)
        org_cursor = AsyncMock(lastrowid=10, rowcount=1)
        membership_cursor = AsyncMock(rowcount=1)

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(
            side_effect=[
                lookup_cursor,
                user_cursor,
                org_cursor,
                membership_cursor,
            ]
        )
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        mock_pool = MagicMock(pool=None)
        mock_pool.transaction = MagicMock(return_value=mock_conn)

        mock_pw_svc = MagicMock()
        mock_pw_svc.hash_password = MagicMock(return_value="hashed_pw")

        with (
            patch(
                "tldw_Server_API.app.core.AuthNZ.database.get_db_pool",
                new_callable=AsyncMock,
                return_value=mock_pool,
            ),
            patch(
                "tldw_Server_API.app.core.AuthNZ.password_service.get_password_service",
                return_value=mock_pw_svc,
            ),
        ):
            result = await provision_tenant(payload, principal)

        assert result.user_id == 42
        assert result.org_id == 10
        assert result.username == "tenant_user"
        assert result.org_name == "TenantOrg"
        assert result.role == "owner"
        mock_pool.transaction.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_provision_uses_backend_aware_postgres_transaction(self):
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_tenant_provisioning

        payload = TenantProvisionRequest(
            username="tenant_user",
            email="tenant@example.com",
            password="securepass123",
            org_name="TenantOrg",
        )
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(side_effect=[None, {"id": 10}])
        conn.__aenter__ = AsyncMock(return_value=conn)
        conn.__aexit__ = AsyncMock(return_value=False)
        pool = MagicMock(pool=object())
        pool.transaction = MagicMock(return_value=conn)
        gateway = AsyncMock()
        gateway.insert_user = AsyncMock(
            return_value=SimpleNamespace(affected_user_ids=(42,))
        )

        with (
            patch(
                "tldw_Server_API.app.core.AuthNZ.database.get_db_pool",
                new_callable=AsyncMock,
                return_value=pool,
            ),
            patch(
                "tldw_Server_API.app.core.AuthNZ.password_service.get_password_service"
            ) as password_service,
            patch.object(
                admin_tenant_provisioning,
                "VersionedUserWriteGateway",
                return_value=gateway,
            ) as gateway_type,
        ):
            password_service.return_value.hash_password.return_value = "hashed"
            result = await admin_tenant_provisioning.provision_tenant(
                payload,
                _admin_principal(),
            )

        assert result.user_id == 42
        assert result.org_id == 10
        gateway_type.assert_called_once_with("postgres")
        assert gateway.insert_user.await_args.kwargs["values"]["is_active"] is True
        pool.transaction.assert_called_once_with()
        assert conn.fetchrow.await_args_list[0].args[0].count("$1") == 1
        assert "RETURNING id" in conn.fetchrow.await_args_list[1].args[0]
        assert "INSERT INTO public.org_members" in conn.execute.await_args.args[0]


def test_tenant_role_rejects_unknown_values() -> None:
    with pytest.raises(ValidationError):
        TenantProvisionRequest(
            username="tenant_user",
            email="tenant@example.com",
            password="securepass123",
            org_name="TenantOrg",
            role="root",
        )

    @pytest.mark.asyncio
    async def test_provisioning_failure_exits_the_single_transaction_with_error(self):
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_tenant_provisioning

        payload = TenantProvisionRequest(
            username="tenant_user",
            email="tenant@example.com",
            password="securepass123",
            org_name="TenantOrg",
        )
        lookup_cursor = AsyncMock()
        lookup_cursor.fetchone = AsyncMock(return_value=None)
        conn = AsyncMock()

        async def _execute(statement: str, *_args: object) -> object:
            if "INSERT INTO main.org_members" in statement:
                raise RuntimeError("membership write failed")
            if "INSERT INTO main.organizations" in statement:
                return SimpleNamespace(lastrowid=10)
            return lookup_cursor

        conn.execute = AsyncMock(side_effect=_execute)
        conn.__aenter__ = AsyncMock(return_value=conn)
        conn.__aexit__ = AsyncMock(return_value=False)
        pool = MagicMock(pool=None)
        pool.transaction = MagicMock(return_value=conn)
        gateway = AsyncMock()
        gateway.insert_user = AsyncMock(
            return_value=SimpleNamespace(affected_user_ids=(42,))
        )

        with (
            patch(
                "tldw_Server_API.app.core.AuthNZ.database.get_db_pool",
                new_callable=AsyncMock,
                return_value=pool,
            ),
            patch(
                "tldw_Server_API.app.core.AuthNZ.password_service.get_password_service"
            ) as password_service,
            patch.object(
                admin_tenant_provisioning,
                "VersionedUserWriteGateway",
                return_value=gateway,
            ),
        ):
            password_service.return_value.hash_password.return_value = "hashed"
            with pytest.raises(Exception) as exc_info:
                await admin_tenant_provisioning.provision_tenant(
                    payload,
                    _admin_principal(),
                )

        assert exc_info.value.status_code == 500
        pool.transaction.assert_called_once_with()
        assert conn.__aexit__.await_args.args[0] is RuntimeError

    @pytest.mark.asyncio
    async def test_sanitizes_generic_failure(self, monkeypatch: pytest.MonkeyPatch):
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_tenant_provisioning

        payload = TenantProvisionRequest(
            username="tenant_user",
            email="tenant@example.com",
            password="securepass123",
            org_name="TenantOrg",
        )
        principal = _admin_principal()
        logger_stub = _LoggerStub()
        monkeypatch.setattr(admin_tenant_provisioning, "logger", logger_stub)

        with patch(
            "tldw_Server_API.app.core.AuthNZ.database.get_db_pool",
            new_callable=AsyncMock,
            side_effect=RuntimeError("tenant provisioning backend exploded at /private/tenant.db"),
        ):
            with pytest.raises(Exception) as exc_info:
                await admin_tenant_provisioning.provision_tenant(payload, principal)

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Tenant provisioning failed"
        assert logger_stub.error_records == [("Tenant provisioning failed", (), {})]
