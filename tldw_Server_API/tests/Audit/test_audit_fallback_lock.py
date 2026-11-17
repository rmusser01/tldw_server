import asyncio
import threading
import time

import pytest

from tldw_Server_API.app.core.Audit.unified_audit_service import (
    AuditContext,
    AuditEvent,
    AuditEventType,
    UnifiedAuditService,
)


@pytest.mark.asyncio
async def test_fallback_queue_lock_prevents_overlap(tmp_path, monkeypatch):
    db_path = tmp_path / "audit.db"
    svc1 = UnifiedAuditService(
        db_path=str(db_path),
        enable_pii_detection=False,
        enable_risk_scoring=False,
        buffer_size=1,
        flush_interval=60.0,
        inline_io=False,
    )
    svc2 = UnifiedAuditService(
        db_path=str(db_path),
        enable_pii_detection=False,
        enable_risk_scoring=False,
        buffer_size=1,
        flush_interval=60.0,
        inline_io=False,
    )

    await svc1.initialize(start_background_tasks=False)
    await svc2.initialize(start_background_tasks=False)

    async with svc1.buffer_lock:
        svc1.event_buffer.extend(
            [AuditEvent(event_type=AuditEventType.API_REQUEST, context=AuditContext(user_id="u1")) for _ in range(3)]
        )
    async with svc2.buffer_lock:
        svc2.event_buffer.extend(
            [AuditEvent(event_type=AuditEventType.API_REQUEST, context=AuditContext(user_id="u2")) for _ in range(3)]
        )

    state = {"in_progress": False, "overlap": False}
    state_lock = threading.Lock()
    original_append = UnifiedAuditService._append_events_to_fallback

    def _wrapped_append(self, fb_path, events):
        with state_lock:
            if state["in_progress"]:
                state["overlap"] = True
            state["in_progress"] = True
        time.sleep(0.05)
        original_append(self, fb_path, events)
        with state_lock:
            state["in_progress"] = False

    monkeypatch.setattr(UnifiedAuditService, "_append_events_to_fallback", _wrapped_append)

    orig_ensure_1 = svc1._ensure_db_pool
    orig_ensure_2 = svc2._ensure_db_pool

    async def _boom():
        raise RuntimeError("forced flush failure")

    svc1._ensure_db_pool = _boom
    svc2._ensure_db_pool = _boom

    try:
        await asyncio.gather(svc1.flush(), svc2.flush())
        fb_path = db_path.parent / "audit_fallback_queue.jsonl"
        assert fb_path.exists(), "Expected fallback queue to be created"
        lines = fb_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 2, "Expected fallback queue lines from both services"
        assert state["overlap"] is False, "Fallback writes should be serialized across instances"
    finally:
        svc1._ensure_db_pool = orig_ensure_1
        svc2._ensure_db_pool = orig_ensure_2
        await svc1.stop()
        await svc2.stop()
