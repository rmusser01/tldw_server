"""Tests for admin impersonation endpoint."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from jose import jwt
from tldw_Server_API.app.api.v1.endpoints.admin import admin_impersonation
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.admin.admin_impersonation import (
    ImpersonationTokenResponse,
    create_impersonation_token,
)
from tldw_Server_API.app.core.Audit.unified_audit_service import (
    AuditEventCategory,
    AuditEventType,
    MandatoryAuditWriteError,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.services.admin_audit_service import (
    emit_impersonation_issuance_audit_event,
)


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
        assert resp.token_type == "bearer"  # nosec B101, B105
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

        assert result.token == "mock.jwt.token"  # nosec B101, B105
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
    async def test_success_accepts_legacy_string_role_rows(self):
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
                return ["user"]

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

        assert result.token == "mock.jwt.token"  # nosec B101, B105
        token_kwargs = mock_jwt_svc.create_impersonation_access_token.call_args.kwargs
        assert token_kwargs["role"] == "user"  # nosec B101
        audit.assert_awaited_once()

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
        assert exc_info.value.detail["error"]["code"] == "audit_persistence_failure"  # nosec B101

    @pytest.mark.asyncio
    async def test_rbac_failure_does_not_issue_token_or_audit(self):
        principal = _admin_principal()

        class UsersRepoStub:
            @classmethod
            async def from_pool(cls):
                return cls()

            async def get_user_by_id(self, user_id: int):
                assert user_id == 42  # nosec B101
                return {"id": 42, "username": "targetuser", "is_active": True, "role": "user"}

        class RbacRepoStub:
            def get_user_roles(self, user_id: int):
                assert user_id == 42  # nosec B101
                raise RuntimeError("rbac unavailable")

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
            with pytest.raises(HTTPException) as exc_info:
                await create_impersonation_token(42, principal)

        assert exc_info.value.status_code == 500  # nosec B101
        assert exc_info.value.detail == "Impersonation token creation failed"  # nosec B101
        mock_jwt_svc.create_impersonation_access_token.assert_not_called()
        audit.assert_not_awaited()

    @pytest.mark.parametrize(
        "role_rows",
        [
            pytest.param([{}], id="empty-dict"),
            pytest.param([object()], id="plain-object"),
            pytest.param([""], id="empty-string"),
            pytest.param([123], id="integer"),
        ],
    )
    @pytest.mark.asyncio
    async def test_malformed_rbac_role_rows_do_not_issue_token_or_audit(self, role_rows: list[Any]):
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
                return role_rows

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
            with pytest.raises(HTTPException) as exc_info:
                await create_impersonation_token(42, principal)

        assert exc_info.value.status_code == 500  # nosec B101
        assert exc_info.value.detail == "Impersonation token creation failed"  # nosec B101
        mock_jwt_svc.create_impersonation_access_token.assert_not_called()
        audit.assert_not_awaited()

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


def _install_endpoint_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    target_user: Any,
    audit_error: Exception | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jwt_calls: list[dict[str, Any]] = []
    audit_calls: list[dict[str, Any]] = []

    class _StubRepo:
        async def get_user_by_id(self, _user_id: int) -> Any:
            return target_user

    async def _from_pool() -> _StubRepo:
        return _StubRepo()

    class _StubJWTService:
        def create_impersonation_access_token(self, **kwargs: Any) -> str:
            jwt_calls.append(kwargs)
            issued_at = datetime.now(timezone.utc)
            payload = {
                "sub": str(kwargs["user_id"]),
                "username": kwargs["username"],
                "role": kwargs["role"],
                "type": "access",
                "iat": int(issued_at.timestamp()),
                "exp": int((issued_at + kwargs["expires_delta"]).timestamp()),
                "impersonated_by": kwargs["impersonated_by"],
                "impersonation": True,
            }
            return jwt.encode(payload, "test-secret", algorithm="HS256")

    async def _emit(**kwargs: Any) -> None:
        audit_calls.append(kwargs)
        if audit_error is not None:
            raise audit_error

    monkeypatch.setattr(
        admin_impersonation,
        "AuthnzUsersRepo",
        SimpleNamespace(from_pool=_from_pool),
    )
    monkeypatch.setattr(admin_impersonation, "get_jwt_service", lambda: _StubJWTService(), raising=False)
    monkeypatch.setattr(admin_impersonation, "emit_impersonation_issuance_audit_event", _emit, raising=False)
    monkeypatch.setattr(admin_impersonation, "AuthnzRbacRepo", lambda: SimpleNamespace(get_user_roles=lambda _user_id: []))
    return jwt_calls, audit_calls


class TestCurrentDevImpersonationCompatibility:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "principal",
        [
            pytest.param(
                AuthPrincipal(
                    kind="user",
                    user_id=42,
                    username="nested-admin",
                    roles=["admin"],
                    is_admin=True,
                    impersonation=True,
                    impersonated_by=1,
                ),
                id="nested-impersonation",
            ),
            pytest.param(
                AuthPrincipal(
                    kind="service",
                    subject="service:local-admin",
                    permissions=["*"],
                    is_admin=True,
                ),
                id="service-without-user-actor",
            ),
        ],
    )
    async def test_rejects_issuer_without_unambiguous_user_actor(self, monkeypatch, principal):
        jwt_calls, audit_calls = _install_endpoint_stubs(
            monkeypatch,
            target_user={
                "id": 99,
                "username": "targetuser",
                "is_active": True,
                "role": "user",
            },
        )

        with pytest.raises(HTTPException) as exc_info:
            await create_impersonation_token(99, principal)

        assert exc_info.value.status_code == 403
        assert exc_info.value.detail == "Impersonation requires a non-impersonated user principal"
        assert jwt_calls == []
        assert audit_calls == [{"actor_id": 1, "target_user_id": 42, "expires_in_minutes": 15}]

    @pytest.mark.asyncio
    async def test_success_uses_backend_agnostic_user_repository(self, monkeypatch):
        principal = _admin_principal()
        jwt_calls, audit_calls = _install_endpoint_stubs(
            monkeypatch,
            target_user={
                "id": 42,
                "username": "targetuser",
                "is_active": True,
                "role": "user",
            },
        )

        result = await create_impersonation_token(42, principal)

        assert result.impersonated_user_id == 42
        claims = jwt.get_unverified_claims(result.token)
        assert int(claims["exp"]) - int(claims["iat"]) == 15 * 60
        assert jwt_calls == [
            {
                "user_id": 42,
                "username": "targetuser",
                "role": "user",
                "expires_delta": timedelta(minutes=15),
                "impersonated_by": 1,
            }
        ]
        assert audit_calls == [
            {
                "actor_id": 1,
                "target_user_id": 42,
                "event_type": admin_impersonation.AuditEventType.AUTH_TOKEN_CREATED,
                "category": admin_impersonation.AuditEventCategory.AUTHORIZATION,
                "resource_type": "user_impersonation",
                "resource_id": "42",
                "action": "admin.impersonation.token.create",
                "metadata": {
                    "impersonated_by": 1,
                    "impersonated_user_id": 42,
                    "expires_in_minutes": 15,
                    "impersonation": True,
                },
                "raise_on_failure": True,
            }
        ]

    @pytest.mark.asyncio
    async def test_success_accepts_user_model_object(self, monkeypatch):
        principal = _admin_principal()
        jwt_calls, _audit_calls = _install_endpoint_stubs(
            monkeypatch,
            target_user=SimpleNamespace(
                id=42,
                username="targetuser",
                is_active=True,
                role="researcher",
            ),
        )

        result = await create_impersonation_token(42, principal)

        assert result.impersonated_user_id == 42
        assert result.impersonated_by == 1
        assert jwt_calls[0]["role"] == "researcher"

    @pytest.mark.asyncio
    async def test_user_not_found(self, monkeypatch):
        principal = _admin_principal()
        _install_endpoint_stubs(monkeypatch, target_user=None)

        with pytest.raises(HTTPException) as exc_info:
            await create_impersonation_token(999, principal)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_inactive_user_rejected(self, monkeypatch):
        principal = _admin_principal()
        _install_endpoint_stubs(
            monkeypatch,
            target_user={
                "id": 42,
                "username": "inactive",
                "is_active": False,
                "role": "user",
            },
        )

        with pytest.raises(HTTPException) as exc_info:
            await create_impersonation_token(42, principal)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_mandatory_audit_failure_returns_sanitized_503(self, monkeypatch):
        principal = _admin_principal()
        _install_endpoint_stubs(
            monkeypatch,
            target_user={
                "id": 42,
                "username": "targetuser",
                "is_active": True,
                "role": "user",
            },
            audit_error=MandatoryAuditWriteError("private audit database path"),
        )

        with pytest.raises(HTTPException) as exc_info:
            await create_impersonation_token(42, principal)

        assert exc_info.value.status_code == 503
        assert exc_info.value.detail == {
            "error": {
                "message": "Mandatory audit persistence unavailable",
                "type": "audit_persistence_failure",
                "code": "audit_persistence_failure",
            }
        }

    @pytest.mark.asyncio
    async def test_sanitizes_generic_failure(self, monkeypatch: pytest.MonkeyPatch):
        principal = _admin_principal()
        logger_stub = _LoggerStub()
        monkeypatch.setattr(admin_impersonation, "logger", logger_stub)

        async def _failing_from_pool() -> None:
            raise RuntimeError("impersonation backend exploded at /private/impersonation.db")

        monkeypatch.setattr(
            admin_impersonation,
            "AuthnzUsersRepo",
            SimpleNamespace(from_pool=_failing_from_pool),
        )

        with pytest.raises(HTTPException) as exc_info:
            await create_impersonation_token(42, principal)

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Impersonation token creation failed"
        assert logger_stub.error_records == [("Impersonation token creation failed", (), {})]


@pytest.mark.asyncio
async def test_impersonation_audit_helper_records_token_created_event():
    service = MagicMock()
    service.log_event = AsyncMock()
    service.flush = AsyncMock()
    lookup = AsyncMock(return_value=service)

    with patch(
        "tldw_Server_API.app.services.admin_audit_service.get_or_create_audit_service_for_user_id_optional",
        lookup,
    ):
        await emit_impersonation_issuance_audit_event(
            actor_id=1,
            target_user_id=42,
            expires_in_minutes=15,
        )

    lookup.assert_awaited_once_with(1)
    service.log_event.assert_awaited_once()
    kwargs = service.log_event.await_args.kwargs
    ctx = kwargs["context"]
    assert ctx.user_id == "1"  # nosec B101
    assert ctx.endpoint == "/api/v1/admin/impersonate/{user_id}/token"  # nosec B101
    assert ctx.method == "POST"  # nosec B101
    assert kwargs["event_type"] is AuditEventType.AUTH_TOKEN_CREATED  # nosec B101
    assert kwargs["category"] is AuditEventCategory.AUTHENTICATION  # nosec B101
    assert kwargs["resource_type"] == "user_impersonation"  # nosec B101
    assert kwargs["resource_id"] == "42"  # nosec B101
    assert kwargs["action"] == "admin.impersonation.token_issued"  # nosec B101
    assert kwargs["metadata"]["actor_id"] == 1  # nosec B101
    assert kwargs["metadata"]["target_user_id"] == 42  # nosec B101
    assert kwargs["metadata"]["expires_in_minutes"] == 15  # nosec B101
    assert kwargs["metadata"]["impersonation"] is True  # nosec B101
    service.flush.assert_awaited_once_with(raise_on_failure=True)
