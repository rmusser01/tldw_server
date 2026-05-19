"""Unit tests for the unified Sharing audit boundary."""
from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Sharing import unified_share_audit as audit_module
from tldw_Server_API.app.core.Sharing.unified_share_audit import UnifiedShareAuditWriter

pytestmark = pytest.mark.unit


def _capture_warning_messages():
    messages: list[str] = []
    sink_id = audit_module.logger.add(
        lambda message: messages.append(str(message.record.get("message") or "")),
        level="WARNING",
    )
    return messages, sink_id


@pytest.mark.asyncio
async def test_share_audit_writer_keeps_owner_and_actor_distinct(tmp_path):
    writer = UnifiedShareAuditWriter(db_path=str(tmp_path / "audit_shared.db"))
    await writer.initialize()
    try:
        await writer.log_event(
            event_type="share.created",
            resource_type="workspace",
            resource_id="ws-1",
            owner_user_id=7,
            actor_user_id=11,
            share_id=42,
            metadata={"scope_type": "team"},
        )
        rows = await writer.query_events(owner_user_id=7)
    finally:
        await writer.stop()

    assert len(rows) == 1
    assert rows[0]["id"] == 1
    assert rows[0]["event_type"] == "share.created"
    assert rows[0]["owner_user_id"] == 7
    assert rows[0]["actor_user_id"] == 11
    assert rows[0]["share_id"] == 42
    assert rows[0]["metadata"] == {"scope_type": "team"}


def test_project_row_skips_missing_compatibility_id_without_raw_event_log_values():
    writer = UnifiedShareAuditWriter(db_path=":memory:")
    private_event_id = "/private/share/audit.db?token=raw-event-id-secret"
    private_event_type = "share.created./private/type?token=raw-event-type-secret"
    messages, sink_id = _capture_warning_messages()
    try:
        result = writer._project_row(
            {
                "event_id": private_event_id,
                "event_type": private_event_type,
                "tenant_user_id": "7",
                "metadata": {"owner_user_id": 7},
            }
        )
    finally:
        audit_module.logger.remove(sink_id)

    assert result is None
    rendered = "\n".join(messages)
    assert "Skipping sharing audit row without compatibility_id" in rendered
    assert private_event_id not in rendered
    assert private_event_type not in rendered
    assert "/private/share/audit.db" not in rendered
    assert "raw-event-id-secret" not in rendered
    assert "raw-event-type-secret" not in rendered


def test_project_row_skips_missing_owner_user_id_without_raw_event_log_values():
    writer = UnifiedShareAuditWriter(db_path=":memory:")
    private_event_id = "/private/share/owner.db?token=raw-owner-event-secret"
    private_event_type = "token.used./private/type?token=raw-owner-type-secret"
    messages, sink_id = _capture_warning_messages()
    try:
        result = writer._project_row(
            {
                "event_id": private_event_id,
                "event_type": private_event_type,
                "tenant_user_id": None,
                "context_user_id": "11",
                "metadata": {"compatibility_id": 3},
            }
        )
    finally:
        audit_module.logger.remove(sink_id)

    assert result is None
    rendered = "\n".join(messages)
    assert "Skipping sharing audit row without owner_user_id" in rendered
    assert private_event_id not in rendered
    assert private_event_type not in rendered
    assert "/private/share/owner.db" not in rendered
    assert "raw-owner-event-secret" not in rendered
    assert "raw-owner-type-secret" not in rendered
