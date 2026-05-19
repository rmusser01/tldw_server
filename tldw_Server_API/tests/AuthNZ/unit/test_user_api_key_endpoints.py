from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import users as users_endpoints
from tldw_Server_API.app.api.v1.schemas.api_key_schemas import (
    APIKeyCreateRequest,
    APIKeyRotateRequest,
)
from tldw_Server_API.app.core.Audit.unified_audit_service import MandatoryAuditWriteError
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal


class _FailingAPIKeyManager:
    async def create_api_key(self, **_kwargs: Any) -> dict[str, Any]:
        raise MandatoryAuditWriteError("Mandatory audit persistence unavailable")

    async def create_virtual_key(self, **_kwargs: Any) -> dict[str, Any]:
        raise MandatoryAuditWriteError("Mandatory audit persistence unavailable")

    async def rotate_api_key(self, **_kwargs: Any) -> dict[str, Any]:
        raise MandatoryAuditWriteError("Mandatory audit persistence unavailable")

    async def revoke_api_key(self, **_kwargs: Any) -> bool:
        raise MandatoryAuditWriteError("Mandatory audit persistence unavailable")


def _principal() -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        subject="unit-user",
        token_type="access",
        jti=None,
        roles=["user"],
        permissions=[],
        is_admin=False,
        org_ids=[],
        team_ids=[],
    )


@pytest.fixture
def _patched_user_api_keys_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_require_principal_active_verified(*_args, **_kwargs):
        return {
            "id": 1,
            "username": "api-key-user",
            "email": "api-key-user@example.com",
            "role": "user",
            "is_active": True,
            "is_verified": True,
            "storage_quota_mb": 5120,
            "storage_used_mb": 0.0,
            "created_at": datetime.utcnow(),
            "last_login": None,
        }

    async def _fake_get_api_key_manager():
        return _FailingAPIKeyManager()

    monkeypatch.setattr(
        users_endpoints,
        "_require_principal_active_verified",
        _fake_require_principal_active_verified,
    )
    monkeypatch.setattr(users_endpoints, "get_api_key_manager", _fake_get_api_key_manager)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_users_create_api_key_returns_503_on_mandatory_audit_failure(
    _patched_user_api_keys_deps,
) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await users_endpoints.create_api_key(
            payload=APIKeyCreateRequest(name="strict-audit-key", scope="read"),
            request=object(),
            principal=_principal(),
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Mandatory audit persistence unavailable"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_users_create_virtual_api_key_returns_503_on_mandatory_audit_failure(
    _patched_user_api_keys_deps,
) -> None:
    payload = users_endpoints.SelfVirtualAPIKeyRequest(name="strict-audit-virtual")

    with pytest.raises(HTTPException) as exc_info:
        await users_endpoints.create_virtual_api_key(
            payload=payload,
            request=object(),
            principal=_principal(),
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Mandatory audit persistence unavailable"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_users_rotate_api_key_returns_503_on_mandatory_audit_failure(
    _patched_user_api_keys_deps,
) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await users_endpoints.rotate_api_key(
            key_id=123,
            payload=APIKeyRotateRequest(expires_in_days=90),
            request=object(),
            principal=_principal(),
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Mandatory audit persistence unavailable"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_users_revoke_api_key_returns_503_on_mandatory_audit_failure(
    _patched_user_api_keys_deps,
) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await users_endpoints.revoke_api_key(
            key_id=321,
            request=object(),
            principal=_principal(),
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Mandatory audit persistence unavailable"
