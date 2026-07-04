"""Tests for admin impersonation endpoint."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from jose import jwt

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
            token="jwt.token.here",
            impersonated_user_id=42,
            impersonated_by=1,
        )
        assert resp.token_type == "bearer"
        assert resp.expires_in_minutes == 15


class TestCreateImpersonationToken:
    @pytest.mark.asyncio
    async def test_success_uses_short_ttl_repo_lookup_and_mandatory_audit(self, monkeypatch):
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_impersonation

        principal = _admin_principal()
        audit_calls: list[dict[str, Any]] = []

        class _StubRepo:
            async def get_user_by_id(self, user_id: int) -> dict[str, Any] | None:
                assert user_id == 42
                return {
                    "id": 42,
                    "username": "targetuser",
                    "role": "user",
                    "is_active": True,
                }

        async def _fake_from_pool() -> _StubRepo:
            return _StubRepo()

        class _StubJWTService:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            def create_access_token(self, **kwargs: Any) -> str:
                self.calls.append(kwargs)
                issued_at = datetime.now(timezone.utc)
                expires_delta = kwargs["expires_delta"]
                payload = {
                    "sub": str(kwargs["user_id"]),
                    "username": kwargs["username"],
                    "role": kwargs["role"],
                    "type": "access",
                    "iat": int(issued_at.timestamp()),
                    "exp": int((issued_at + expires_delta).timestamp()),
                    **(kwargs.get("additional_claims") or {}),
                }
                return jwt.encode(payload, "test-secret", algorithm="HS256")

        jwt_svc = _StubJWTService()

        async def _fake_emit_admin_account_audit_event(**kwargs: Any) -> None:
            audit_calls.append(kwargs)

        async def _fail_raw_pool() -> None:
            raise AssertionError("raw AuthNZ database pool should not be used")

        monkeypatch.setattr(
            admin_impersonation,
            "AuthnzUsersRepo",
            SimpleNamespace(from_pool=_fake_from_pool),
            raising=False,
        )
        monkeypatch.setattr(admin_impersonation, "get_jwt_service", lambda: jwt_svc, raising=False)
        monkeypatch.setattr(
            admin_impersonation,
            "_emit_admin_account_audit_event",
            _fake_emit_admin_account_audit_event,
            raising=False,
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.AuthNZ.database.get_db_pool",
            _fail_raw_pool,
        )

        result = await create_impersonation_token(42, principal)

        claims = jwt.get_unverified_claims(result.token)
        assert int(claims["exp"]) - int(claims["iat"]) == 15 * 60
        assert result.impersonated_user_id == 42
        assert result.impersonated_by == 1

        assert jwt_svc.calls[0]["expires_delta"] == timedelta(minutes=15)
        additional = jwt_svc.calls[0]["additional_claims"]
        assert additional["impersonated_by"] == 1
        assert additional["impersonation"] is True
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
    async def test_user_not_found(self, monkeypatch):
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_impersonation

        principal = _admin_principal()

        class _StubRepo:
            async def get_user_by_id(self, user_id: int) -> dict[str, Any] | None:
                assert user_id == 999
                return None

        async def _fake_from_pool() -> _StubRepo:
            return _StubRepo()

        monkeypatch.setattr(
            admin_impersonation,
            "AuthnzUsersRepo",
            SimpleNamespace(from_pool=_fake_from_pool),
            raising=False,
        )

        with pytest.raises(HTTPException) as exc_info:
            await create_impersonation_token(999, principal)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_inactive_user_rejected(self, monkeypatch):
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_impersonation

        principal = _admin_principal()

        class _StubRepo:
            async def get_user_by_id(self, user_id: int) -> dict[str, Any] | None:
                assert user_id == 42
                return {
                    "id": 42,
                    "username": "inactive",
                    "role": "user",
                    "is_active": False,
                }

        async def _fake_from_pool() -> _StubRepo:
            return _StubRepo()

        monkeypatch.setattr(
            admin_impersonation,
            "AuthnzUsersRepo",
            SimpleNamespace(from_pool=_fake_from_pool),
            raising=False,
        )

        with pytest.raises(HTTPException) as exc_info:
            await create_impersonation_token(42, principal)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_mandatory_audit_failure_returns_503(self, monkeypatch):
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_impersonation

        principal = _admin_principal()

        class _StubRepo:
            async def get_user_by_id(self, user_id: int) -> dict[str, Any] | None:
                return {
                    "id": user_id,
                    "username": "targetuser",
                    "role": "user",
                    "is_active": True,
                }

        async def _fake_from_pool() -> _StubRepo:
            return _StubRepo()

        class _StubJWTService:
            def create_access_token(self, **_kwargs: Any) -> str:
                return "mock.jwt.token"

        async def _fail_emit_admin_account_audit_event(**_kwargs: Any) -> None:
            raise MandatoryAuditWriteError("Mandatory audit persistence unavailable")

        monkeypatch.setattr(
            admin_impersonation,
            "AuthnzUsersRepo",
            SimpleNamespace(from_pool=_fake_from_pool),
            raising=False,
        )
        monkeypatch.setattr(
            admin_impersonation,
            "get_jwt_service",
            lambda: _StubJWTService(),
            raising=False,
        )
        monkeypatch.setattr(
            admin_impersonation,
            "_emit_admin_account_audit_event",
            _fail_emit_admin_account_audit_event,
            raising=False,
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
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_impersonation

        principal = _admin_principal()
        logger_stub = _LoggerStub()
        monkeypatch.setattr(admin_impersonation, "logger", logger_stub)

        async def _failing_from_pool() -> None:
            raise RuntimeError("impersonation backend exploded at /private/impersonation.db")

        monkeypatch.setattr(
            admin_impersonation,
            "AuthnzUsersRepo",
            SimpleNamespace(from_pool=_failing_from_pool),
            raising=False,
        )

        with pytest.raises(HTTPException) as exc_info:
            await create_impersonation_token(42, principal)

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Impersonation token creation failed"
        assert logger_stub.error_records == [("Impersonation token creation failed", (), {})]
