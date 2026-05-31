"""Unit tests for ShareAuditService."""
from __future__ import annotations

import pytest

from tldw_Server_API.app.core.exceptions import AuditLogError
from tldw_Server_API.app.core.Sharing import share_audit_service as share_audit_service_module
from tldw_Server_API.app.core.Sharing.share_audit_service import (
    SHARE_CREATED,
    SHARE_REVOKED,
    TOKEN_CREATED,
    ShareAuditService,
)

pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.error_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def error(self, *args: object, **kwargs: object) -> None:
        self.error_calls.append((args, dict(kwargs)))


@pytest.fixture
def audit_service(repo):
    return ShareAuditService(repo)


@pytest.mark.asyncio
async def test_log_event(audit_service):
    await audit_service.log(
        SHARE_CREATED,
        resource_type="workspace",
        resource_id="ws-1",
        owner_user_id=1,
        actor_user_id=1,
        share_id=42,
        metadata={"scope_type": "team", "scope_id": 10},
    )
    events = await audit_service.query(owner_user_id=1)
    assert len(events) == 1
    assert events[0]["event_type"] == SHARE_CREATED
    assert events[0]["share_id"] == 42
    assert events[0]["metadata"]["scope_type"] == "team"


@pytest.mark.asyncio
async def test_log_multiple_events(audit_service):
    await audit_service.log(
        SHARE_CREATED, resource_type="workspace", resource_id="ws-1",
        owner_user_id=1, actor_user_id=1,
    )
    await audit_service.log(
        SHARE_REVOKED, resource_type="workspace", resource_id="ws-1",
        owner_user_id=1, actor_user_id=1,
    )
    await audit_service.log(
        TOKEN_CREATED, resource_type="chatbook", resource_id="cb-1",
        owner_user_id=1, actor_user_id=1,
    )
    all_events = await audit_service.query(owner_user_id=1)
    assert len(all_events) == 3


@pytest.mark.asyncio
async def test_query_filter_by_resource(audit_service):
    await audit_service.log(
        SHARE_CREATED, resource_type="workspace", resource_id="ws-1",
        owner_user_id=1,
    )
    await audit_service.log(
        TOKEN_CREATED, resource_type="chatbook", resource_id="cb-1",
        owner_user_id=1,
    )
    ws_events = await audit_service.query(resource_type="workspace")
    assert len(ws_events) == 1
    assert ws_events[0]["resource_type"] == "workspace"


@pytest.mark.asyncio
async def test_query_pagination(audit_service):
    for i in range(5):
        await audit_service.log(
            SHARE_CREATED, resource_type="workspace", resource_id=f"ws-{i}",
            owner_user_id=1,
        )
    page1 = await audit_service.query(limit=2, offset=0)
    page2 = await audit_service.query(limit=2, offset=2)
    assert len(page1) == 2
    assert len(page2) == 2
    # Different events
    assert page1[0]["resource_id"] != page2[0]["resource_id"]


@pytest.mark.asyncio
async def test_log_with_ip_and_user_agent(audit_service):
    await audit_service.log(
        SHARE_CREATED, resource_type="workspace", resource_id="ws-1",
        owner_user_id=1, ip_address="192.168.1.1", user_agent="TestAgent/1.0",
    )
    events = await audit_service.query(owner_user_id=1)
    assert events[0]["ip_address"] == "192.168.1.1"
    assert events[0]["user_agent"] == "TestAgent/1.0"


@pytest.mark.asyncio
async def test_share_audit_service_uses_unified_writer_when_provided(tmp_path):
    from tldw_Server_API.app.core.Sharing.unified_share_audit import UnifiedShareAuditWriter

    writer = UnifiedShareAuditWriter(db_path=str(tmp_path / "audit_shared.db"))
    # Pass no repo so the service *must* use the writer
    service = ShareAuditService(writer=writer)

    try:
        await service.log(
            SHARE_CREATED,
            resource_type="workspace",
            resource_id="ws-1",
            owner_user_id=1,
            actor_user_id=2,
            share_id=42,
            metadata={"scope_type": "team"},
        )
        events = await service.query(owner_user_id=1)
    finally:
        await writer.stop()

    assert len(events) == 1
    assert events[0]["owner_user_id"] == 1
    assert events[0]["actor_user_id"] == 2
    assert events[0]["share_id"] == 42



@pytest.mark.asyncio
async def test_log_raises_audit_log_error_when_repo_write_fails(audit_service, monkeypatch):
    """Audit logging failures should propagate as AuditLogError."""
    async def _fail(*args, **kwargs):
        raise RuntimeError("DB is down")

    monkeypatch.setattr(audit_service._repo, "log_audit_event", _fail)
    with pytest.raises(AuditLogError) as excinfo:
        await audit_service.log(
            SHARE_CREATED, resource_type="workspace", resource_id="ws-1",
            owner_user_id=1,
        )
    assert "Failed to log share audit event" in str(excinfo.value)


@pytest.mark.asyncio
async def test_log_failure_sanitizes_fallback_log(audit_service, monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(share_audit_service_module, "logger", logger_stub, raising=True)

    raw_event_type = f"{SHARE_CREATED}.raw-marker./private/tmp/share-audit.db?token=secret-token"

    async def _fail(*args, **kwargs):
        raise RuntimeError("backend exploded /private/tmp/share-audit.db token=secret-token")

    monkeypatch.setattr(audit_service._repo, "log_audit_event", _fail)

    with pytest.raises(AuditLogError):
        await audit_service.log(
            raw_event_type,
            resource_type="workspace",
            resource_id="ws-1",
            owner_user_id=1,
        )

    rendered_log = repr(logger_stub.error_calls)
    for marker in (
        "raw-marker",
        "/private/tmp/share-audit.db",
        "token=secret-token",
        "backend exploded",
    ):
        assert marker not in rendered_log
    assert logger_stub.error_calls == [
        (("ShareAuditService.log failed; exception_type=RuntimeError",), {})
    ]
