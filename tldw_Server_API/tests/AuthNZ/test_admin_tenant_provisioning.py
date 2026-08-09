"""Tests for admin tenant provisioning endpoint."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.endpoints.admin.admin_tenant_provisioning import (
    TenantProvisionRequest,
    TenantProvisionResponse,
    router,
)
from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    ActorMembershipWriteContext,
    AnchorOwnership,
    MembershipAuthority,
    MembershipAuthorizationError,
    MembershipTargetNotFound,
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

    @pytest.mark.parametrize("role", ("admin", "lead", "member"))
    def test_non_owner_initial_role_is_rejected(self, role: str):
        with pytest.raises(ValidationError):
            TenantProvisionRequest(
                username="newuser",
                email="new@example.com",
                password="securepass123",
                org_name="NewOrg",
                role=role,
            )

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
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_tenant_provisioning

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
        repo = AsyncMock()
        creation_writer = AsyncMock()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(
            side_effect=[
                lookup_cursor,
                user_cursor,
                org_cursor,
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
            patch.object(
                admin_tenant_provisioning,
                "AuthnzOrgsTeamsRepo",
                return_value=repo,
            ),
            patch.object(
                admin_tenant_provisioning,
                "MembershipWriter",
                return_value=creation_writer,
            ),
        ):
            result = await admin_tenant_provisioning.provision_tenant(
                payload,
                principal,
            )

        assert result.user_id == 42
        assert result.org_id == 10
        assert result.username == "tenant_user"
        assert result.org_name == "TenantOrg"
        assert result.role == "owner"
        mock_pool.transaction.assert_called_once_with()
        creation_writer.authorize_organization_creation.assert_awaited_once()
        repo.provision_org_membership_on_connection.assert_awaited_once()

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
        repo = AsyncMock()
        creation_writer = AsyncMock()

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
            patch.object(
                admin_tenant_provisioning,
                "AuthnzOrgsTeamsRepo",
                return_value=repo,
            ),
            patch.object(
                admin_tenant_provisioning,
                "MembershipWriter",
                return_value=creation_writer,
            ),
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
        creation_writer.authorize_organization_creation.assert_awaited_once()
        repo.provision_org_membership_on_connection.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_provision_revalidates_admin_actor_on_caller_connection(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
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
        repo = AsyncMock()
        creation_writer = AsyncMock()
        monkeypatch.setattr(
            admin_tenant_provisioning,
            "AuthnzOrgsTeamsRepo",
            MagicMock(return_value=repo),
            raising=False,
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
            patch.object(
                admin_tenant_provisioning,
                "MembershipWriter",
                return_value=creation_writer,
                create=True,
            ),
        ):
            password_service.return_value.hash_password.return_value = "hashed"
            result = await admin_tenant_provisioning.provision_tenant(
                payload,
                _admin_principal(),
            )

        assert result.user_id == 42
        creation_writer.authorize_organization_creation.assert_awaited_once_with(
            conn=conn,
            context=ActorMembershipWriteContext(
                actor_user_id=1,
                required_authority=MembershipAuthority.PLATFORM_ADMIN,
            ),
            owner_user_id=42,
        )
        kwargs = repo.provision_org_membership_on_connection.await_args.kwargs
        assert kwargs["conn"] is conn
        assert kwargs["context"] == ActorMembershipWriteContext(
            actor_user_id=1,
            required_authority=MembershipAuthority.PLATFORM_ADMIN,
        )
        assert kwargs["anchor_ownership"] is AnchorOwnership.WRITER_OWNS_ANCHOR
        assert kwargs["org_id"] == 10
        assert kwargs["user_id"] == 42
        assert kwargs["org_role"] == "owner"
        assert kwargs["team_id"] is None
        assert kwargs["team_role"] is None
        assert kwargs["team_failure_is_best_effort"] is False
        assert isinstance(kwargs["operation_time"], datetime)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "writer_error",
        (MembershipAuthorizationError(), MembershipTargetNotFound()),
    )
    async def test_provision_maps_persisted_admin_rejection_to_forbidden(
        self,
        writer_error: Exception,
    ) -> None:
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
        repo = AsyncMock()
        creation_writer = AsyncMock()
        creation_writer.authorize_organization_creation.side_effect = writer_error

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
            patch.object(
                admin_tenant_provisioning,
                "AuthnzOrgsTeamsRepo",
                return_value=repo,
            ),
            patch.object(
                admin_tenant_provisioning,
                "MembershipWriter",
                return_value=creation_writer,
                create=True,
            ),
        ):
            password_service.return_value.hash_password.return_value = "hashed"
            with pytest.raises(HTTPException) as exc_info:
                await admin_tenant_provisioning.provision_tenant(
                    payload,
                    _admin_principal(),
                )

        assert exc_info.value.status_code == 403
        assert exc_info.value.detail == "Not authorized to provision tenants"
        assert conn.__aexit__.await_args.args[0] is type(writer_error)


    def test_tenant_role_rejects_unknown_values(self) -> None:
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
