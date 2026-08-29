"""Tests for admin impersonation endpoint."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from jose import jwt

from tldw_Server_API.app.api.v1.endpoints.admin import admin_impersonation
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
        def create_access_token(self, **kwargs: Any) -> str:
            jwt_calls.append(kwargs)
            issued_at = datetime.now(timezone.utc)
            payload = {
                "sub": str(kwargs["user_id"]),
                "username": kwargs["username"],
                "role": kwargs["role"],
                "type": "access",
                "iat": int(issued_at.timestamp()),
                "exp": int((issued_at + kwargs["expires_delta"]).timestamp()),
                **(kwargs.get("additional_claims") or {}),
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
    monkeypatch.setattr(admin_impersonation, "_emit_admin_account_audit_event", _emit, raising=False)
    return jwt_calls, audit_calls


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
                "additional_claims": {"impersonated_by": 1, "impersonation": True},
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
