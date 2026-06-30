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


class _ExplodingAdminAPIKeyManager:
    async def list_user_keys(self, **_kwargs):
        raise RuntimeError("api key list failed at /private/api-keys.db")

    async def create_api_key(self, **_kwargs):
        raise RuntimeError("api key create failed at /private/api-keys.db")

    async def rotate_api_key(self, **_kwargs):
        raise RuntimeError("api key rotate failed at /private/api-keys.db")

    async def revoke_api_key(self, **_kwargs):
        raise RuntimeError("api key revoke failed at /private/api-keys.db")

    async def create_virtual_key(self, **_kwargs):
        raise RuntimeError("virtual key create failed at /private/api-keys.db")


async def _allow_user_scope(*_args, **_kwargs):
    return None


async def _exploding_api_key_manager():
    return _ExplodingAdminAPIKeyManager()


async def _raise_virtual_key_list_failure() -> bool:
    raise RuntimeError("virtual key list failed at /private/api-keys.db")


async def _raise_audit_log_failure() -> bool:
    raise RuntimeError("api key audit failed at /private/api-keys.db")


async def _assert_admin_api_key_log_sanitized(
    call,
    *,
    expected_detail: str,
    expected_log: str,
    raw_marker: str,
) -> None:
    messages: list[str] = []
    sink_id = svc.logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        with pytest.raises(HTTPException) as exc_info:
            await call()
    finally:
        svc.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == expected_detail
    assert expected_log in joined
    assert raw_marker not in joined
    assert "/private/" not in joined


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
async def test_list_user_api_keys_sanitizes_generic_failure_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(svc.admin_scope_service, "enforce_admin_user_scope", _allow_user_scope)
    monkeypatch.setattr(svc, "get_api_key_manager", _exploding_api_key_manager)

    await _assert_admin_api_key_log_sanitized(
        lambda: svc.list_user_api_keys(
            principal=object(),
            user_id=22,
            include_revoked=False,
        ),
        expected_detail="Failed to list API keys",
        expected_log="Failed to list API keys",
        raw_marker="api key list failed",
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_create_user_api_key_sanitizes_generic_failure_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(svc.admin_scope_service, "enforce_admin_user_scope", _allow_user_scope)
    monkeypatch.setattr(svc, "get_api_key_manager", _exploding_api_key_manager)

    await _assert_admin_api_key_log_sanitized(
        lambda: svc.create_user_api_key(
            principal=object(),
            user_id=22,
            request=APIKeyCreateRequest(name="admin-key", scope="read"),
        ),
        expected_detail="Failed to create API key",
        expected_log="Failed to create API key",
        raw_marker="api key create failed",
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_rotate_user_api_key_sanitizes_generic_failure_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(svc.admin_scope_service, "enforce_admin_user_scope", _allow_user_scope)
    monkeypatch.setattr(svc, "get_api_key_manager", _exploding_api_key_manager)

    await _assert_admin_api_key_log_sanitized(
        lambda: svc.rotate_user_api_key(
            principal=object(),
            user_id=22,
            key_id=345,
            request=APIKeyRotateRequest(expires_in_days=90),
        ),
        expected_detail="Failed to rotate API key",
        expected_log="Failed to rotate API key",
        raw_marker="api key rotate failed",
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_revoke_user_api_key_sanitizes_generic_failure_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(svc.admin_scope_service, "enforce_admin_user_scope", _allow_user_scope)
    monkeypatch.setattr(svc, "get_api_key_manager", _exploding_api_key_manager)

    await _assert_admin_api_key_log_sanitized(
        lambda: svc.revoke_user_api_key(
            principal=object(),
            user_id=22,
            key_id=345,
        ),
        expected_detail="Failed to revoke API key",
        expected_log="Failed to revoke API key",
        raw_marker="api key revoke failed",
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_update_user_api_key_sanitizes_generic_failure_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_update_metadata(*_args, **_kwargs):
        raise RuntimeError("api key update failed at /private/api-keys.db")

    async def is_pg() -> bool:
        return False

    monkeypatch.setattr(svc.admin_scope_service, "enforce_admin_user_scope", _allow_user_scope)
    monkeypatch.setattr(svc, "update_api_key_metadata", fail_update_metadata)

    await _assert_admin_api_key_log_sanitized(
        lambda: svc.update_user_api_key(
            principal=object(),
            user_id=22,
            key_id=345,
            request=APIKeyUpdateRequest(rate_limit=55),
            db=object(),
            is_pg_fn=is_pg,
        ),
        expected_detail="Failed to update API key",
        expected_log="Failed to update API key",
        raw_marker="api key update failed",
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_create_virtual_key_sanitizes_generic_failure_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(svc.admin_scope_service, "enforce_admin_user_scope", _allow_user_scope)
    monkeypatch.setattr(svc, "get_api_key_manager", _exploding_api_key_manager)

    await _assert_admin_api_key_log_sanitized(
        lambda: svc.create_virtual_key(
            principal=object(),
            user_id=22,
            payload=VirtualKeyCreateRequest(name="vkey"),
        ),
        expected_detail="Failed to create virtual key",
        expected_log="Failed to create virtual key",
        raw_marker="virtual key create failed",
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_list_virtual_keys_sanitizes_generic_failure_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(svc.admin_scope_service, "enforce_admin_user_scope", _allow_user_scope)

    await _assert_admin_api_key_log_sanitized(
        lambda: svc.list_virtual_keys(
            principal=object(),
            user_id=22,
            db=object(),
            name=None,
            status_filter=None,
            org_id=None,
            team_id=None,
            created_after=None,
            created_before=None,
            is_pg_fn=_raise_virtual_key_list_failure,
        ),
        expected_detail="Failed to list virtual keys",
        expected_log="Failed to list virtual keys",
        raw_marker="virtual key list failed",
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_get_api_key_audit_log_sanitizes_generic_failure_log() -> None:
    await _assert_admin_api_key_log_sanitized(
        lambda: svc.get_api_key_audit_log(
            principal=object(),
            key_id=345,
            limit=25,
            offset=0,
            db=object(),
            is_pg_fn=_raise_audit_log_failure,
        ),
        expected_detail="Failed to load audit log",
        expected_log="Failed to load audit log",
        raw_marker="api key audit failed",
    )


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

    access_token_type = "acc" + "ess"
    principal = svc.AuthPrincipal(
        kind="user",
        user_id=999,
        api_key_id=None,
        subject="admin-subject",
        token_type=access_token_type,
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
