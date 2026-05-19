from __future__ import annotations

from typing import Any

import pytest

import tldw_Server_API.app.core.AuthNZ.api_key_audit as api_key_audit
from tldw_Server_API.app.core.Audit.unified_audit_service import (
    AuditEventCategory,
    AuditEventType,
    MandatoryAuditWriteError,
)


class _StubAuditService:
    def __init__(self) -> None:
        self.log_calls: list[dict[str, Any]] = []
        self.flush_calls: list[dict[str, Any]] = []

    async def log_event(self, **kwargs: Any) -> None:
        self.log_calls.append(kwargs)

    async def flush(self, **kwargs: Any) -> None:
        self.flush_calls.append(kwargs)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_emit_mandatory_api_key_audit_writes_unified_event_and_flushes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    stub = _StubAuditService()

    async def _fake_get_or_create(user_id: int):
        captured["user_id"] = user_id
        return stub

    monkeypatch.setattr(api_key_audit, "_create_isolated_audit_service", _fake_get_or_create)

    await api_key_audit.emit_mandatory_api_key_management_audit(
        user_id=42,
        event_type=AuditEventType.DATA_UPDATE,
        category=AuditEventCategory.DATA_MODIFICATION,
        action="api_key.rotate",
        resource_id="101",
        metadata={"old_key_id": 10, "new_key_id": 101},
    )

    assert captured["user_id"] == 42
    assert len(stub.log_calls) == 1
    assert stub.log_calls[0]["event_type"] == AuditEventType.DATA_UPDATE
    assert stub.log_calls[0]["category"] == AuditEventCategory.DATA_MODIFICATION
    assert stub.log_calls[0]["action"] == "api_key.rotate"
    assert stub.log_calls[0]["resource_type"] == "api_key"
    assert stub.log_calls[0]["resource_id"] == "101"
    assert stub.log_calls[0]["metadata"] == {
        "old_key_id": 10,
        "new_key_id": 101,
        "target_user_id": 42,
    }
    assert stub.log_calls[0]["context"].user_id == "42"
    assert stub.flush_calls == [{"raise_on_failure": True}]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_emit_mandatory_api_key_audit_wraps_service_creation_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fail_create(_user_id: int) -> None:
        raise RuntimeError("bootstrap boom")

    monkeypatch.setattr(api_key_audit, "_create_isolated_audit_service", _fail_create)

    with pytest.raises(MandatoryAuditWriteError) as exc_info:
        await api_key_audit.emit_mandatory_api_key_management_audit(
            user_id=9,
            event_type=AuditEventType.DATA_WRITE,
            category=AuditEventCategory.DATA_MODIFICATION,
            action="api_key.create",
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "bootstrap boom" in str(exc_info.value.__cause__)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_emit_mandatory_api_key_audit_wraps_log_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingAuditService:
        async def log_event(self, **_kwargs: Any) -> None:
            raise RuntimeError("boom")

        async def flush(self, **_kwargs: Any) -> None:
            raise AssertionError("flush should not run when log_event fails")

    async def _fake_get_or_create(_user_id: int):
        return _FailingAuditService()

    monkeypatch.setattr(api_key_audit, "_create_isolated_audit_service", _fake_get_or_create)

    with pytest.raises(MandatoryAuditWriteError, match="Mandatory audit persistence unavailable"):
        await api_key_audit.emit_mandatory_api_key_management_audit(
            user_id=7,
            event_type=AuditEventType.DATA_WRITE,
            category=AuditEventCategory.DATA_MODIFICATION,
            action="api_key.create",
            resource_id="7",
        )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_emit_mandatory_api_key_audit_wraps_flush_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingFlushAuditService:
        async def log_event(self, **_kwargs: Any) -> None:
            return None

        async def flush(self, **_kwargs: Any) -> None:
            raise RuntimeError("flush boom")

    async def _fake_get_or_create(_user_id: int):
        return _FailingFlushAuditService()

    monkeypatch.setattr(api_key_audit, "_create_isolated_audit_service", _fake_get_or_create)

    with pytest.raises(MandatoryAuditWriteError, match="Mandatory audit persistence unavailable"):
        await api_key_audit.emit_mandatory_api_key_management_audit(
            user_id=8,
            event_type=AuditEventType.DATA_UPDATE,
            category=AuditEventCategory.SECURITY,
            action="api_key.revoke",
            resource_id="18",
        )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_emit_mandatory_api_key_audit_uses_isolated_service_and_preserves_actor_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cached = _StubAuditService()
    isolated = _StubAuditService()

    async def _fake_get_or_create(_user_id: int):
        return cached

    async def _fake_create_isolated(_user_id: int):
        return isolated

    async def _fail_cached_flush(**_kwargs: Any) -> None:
        raise AssertionError("cached service should not be flushed for strict API key writes")

    cached.flush = _fail_cached_flush  # type: ignore[method-assign]
    monkeypatch.setattr(api_key_audit, "_get_or_create_audit_service", _fake_get_or_create)
    monkeypatch.setattr(
        api_key_audit,
        "_create_isolated_audit_service",
        _fake_create_isolated,
        raising=False,
    )

    await api_key_audit.emit_mandatory_api_key_management_audit(
        user_id=42,
        event_type=AuditEventType.DATA_WRITE,
        category=AuditEventCategory.DATA_MODIFICATION,
        action="api_key.create",
        resource_id="101",
        metadata={"scope": "read"},
        actor_user_id=7,
        actor_subject="admin-user",
        actor_kind="user",
        actor_roles=["admin"],
    )

    assert cached.log_calls == []
    assert len(isolated.log_calls) == 1
    assert isolated.log_calls[0]["metadata"]["scope"] == "read"
    assert isolated.log_calls[0]["metadata"]["actor_user_id"] == 7
    assert isolated.log_calls[0]["metadata"]["actor_subject"] == "admin-user"
    assert isolated.log_calls[0]["metadata"]["actor_kind"] == "user"
    assert isolated.log_calls[0]["metadata"]["actor_roles"] == ["admin"]
    assert isolated.log_calls[0]["metadata"]["target_user_id"] == 42
