from __future__ import annotations

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.schemas.api_key_schemas import (
    APIKeyCreateRequest,
    APIKeyRotateRequest,
    APIKeyUpdateRequest,
)
from tldw_Server_API.app.api.v1.schemas.org_team_schemas import VirtualKeyCreateRequest
from tldw_Server_API.app.core.Audit.unified_audit_service import MandatoryAuditWriteError
from tldw_Server_API.app.services import admin_api_keys_service as svc


@pytest.mark.asyncio
@pytest.mark.unit
async def test_update_user_api_key_passes_backend_mode_to_admin_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def _fake_enforce_admin_user_scope(*args, **kwargs) -> None:  # noqa: ANN002
        return None

    async def _fake_update_api_key_metadata(
        db,
        *,
        user_id: int,
        key_id: int,
        rate_limit: int | None = None,
        allowed_ips: list[str] | None = None,
        is_postgres: bool,
    ) -> dict:
        captured["is_postgres"] = is_postgres
        captured["user_id"] = user_id
        captured["key_id"] = key_id
        captured["rate_limit"] = rate_limit
        captured["allowed_ips"] = allowed_ips
        return {
            "id": key_id,
            "scope": "read",
            "key_prefix": "sk-test",
        }

    async def _fake_is_pg() -> bool:
        return True

    monkeypatch.setattr(svc.admin_scope_service, "enforce_admin_user_scope", _fake_enforce_admin_user_scope)
    monkeypatch.setattr(svc, "update_api_key_metadata", _fake_update_api_key_metadata)

    result = await svc.update_user_api_key(
        principal=object(),
        user_id=12,
        key_id=34,
        request=APIKeyUpdateRequest(rate_limit=55, allowed_ips=["10.1.1.1"]),
        db=object(),
        is_pg_fn=_fake_is_pg,
    )

    assert result.id == 34
    assert result.scope == "read"
    assert captured == {
        "is_postgres": True,
        "user_id": 12,
        "key_id": 34,
        "rate_limit": 55,
        "allowed_ips": ["10.1.1.1"],
    }


class _FailingAdminAPIKeyManager:
    async def create_api_key(self, **_kwargs):
        raise MandatoryAuditWriteError("Mandatory audit persistence unavailable")

    async def rotate_api_key(self, **_kwargs):
        raise MandatoryAuditWriteError("Mandatory audit persistence unavailable")

    async def revoke_api_key(self, **_kwargs):
        raise MandatoryAuditWriteError("Mandatory audit persistence unavailable")

    async def create_virtual_key(self, **_kwargs):
        raise MandatoryAuditWriteError("Mandatory audit persistence unavailable")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_admin_create_user_api_key_returns_503_on_mandatory_audit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_enforce_admin_user_scope(*_args, **_kwargs):
        return None

    async def _fake_get_mgr():
        return _FailingAdminAPIKeyManager()

    monkeypatch.setattr(svc.admin_scope_service, "enforce_admin_user_scope", _fake_enforce_admin_user_scope)
    monkeypatch.setattr(svc, "get_api_key_manager", _fake_get_mgr)

    with pytest.raises(HTTPException) as exc_info:
        await svc.create_user_api_key(
            principal=object(),
            user_id=22,
            request=APIKeyCreateRequest(name="admin-key", scope="read"),
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Mandatory audit persistence unavailable"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_admin_rotate_user_api_key_returns_503_on_mandatory_audit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_enforce_admin_user_scope(*_args, **_kwargs):
        return None

    async def _fake_get_mgr():
        return _FailingAdminAPIKeyManager()

    monkeypatch.setattr(svc.admin_scope_service, "enforce_admin_user_scope", _fake_enforce_admin_user_scope)
    monkeypatch.setattr(svc, "get_api_key_manager", _fake_get_mgr)

    with pytest.raises(HTTPException) as exc_info:
        await svc.rotate_user_api_key(
            principal=object(),
            user_id=22,
            key_id=345,
            request=APIKeyRotateRequest(expires_in_days=90),
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Mandatory audit persistence unavailable"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_admin_create_user_api_key_passes_actor_metadata_to_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _CapturingAdminAPIKeyManager:
        async def create_api_key(self, **kwargs):
            captured.update(kwargs)
            return {
                "id": 44,
                "key": "tldw_test_key",
                "key_prefix": "tldw_test...",
                "name": "actor-key",
                "scope": "read",
                "expires_at": None,
                "created_at": "2026-04-09T00:00:00+00:00",
                "message": "ok",
            }

    async def _fake_enforce_admin_user_scope(*_args, **_kwargs):
        return None

    async def _fake_get_mgr():
        return _CapturingAdminAPIKeyManager()

    principal = svc.AuthPrincipal(
        kind="user",
        user_id=999,
        api_key_id=None,
        subject="admin-subject",
        token_type="access",
        jti=None,
        roles=["admin"],
        permissions=[],
        is_admin=True,
        org_ids=[],
        team_ids=[],
    )

    monkeypatch.setattr(svc.admin_scope_service, "enforce_admin_user_scope", _fake_enforce_admin_user_scope)
    monkeypatch.setattr(svc, "get_api_key_manager", _fake_get_mgr)

    await svc.create_user_api_key(
        principal=principal,
        user_id=22,
        request=APIKeyCreateRequest(name="actor-key", scope="read"),
    )

    assert captured["user_id"] == 22
    assert captured["actor_user_id"] == 999
    assert captured["actor_subject"] == "admin-subject"
    assert captured["actor_kind"] == "user"
    assert captured["actor_roles"] == ["admin"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_admin_revoke_user_api_key_returns_503_on_mandatory_audit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_enforce_admin_user_scope(*_args, **_kwargs):
        return None

    async def _fake_get_mgr():
        return _FailingAdminAPIKeyManager()

    monkeypatch.setattr(svc.admin_scope_service, "enforce_admin_user_scope", _fake_enforce_admin_user_scope)
    monkeypatch.setattr(svc, "get_api_key_manager", _fake_get_mgr)

    with pytest.raises(HTTPException) as exc_info:
        await svc.revoke_user_api_key(
            principal=object(),
            user_id=22,
            key_id=345,
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Mandatory audit persistence unavailable"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_admin_create_virtual_key_returns_503_on_mandatory_audit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_enforce_admin_user_scope(*_args, **_kwargs):
        return None

    async def _fake_get_mgr():
        return _FailingAdminAPIKeyManager()

    monkeypatch.setattr(svc.admin_scope_service, "enforce_admin_user_scope", _fake_enforce_admin_user_scope)
    monkeypatch.setattr(svc, "get_api_key_manager", _fake_get_mgr)

    with pytest.raises(HTTPException) as exc_info:
        await svc.create_virtual_key(
            principal=object(),
            user_id=22,
            payload=VirtualKeyCreateRequest(name="vkey"),
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Mandatory audit persistence unavailable"
