"""One ACP action keeps one identity across memory and durable audit history."""

from collections import deque
from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.api.v1.endpoints import agent_client_protocol as acp
from tldw_Server_API.app.core.DB_Management import ACP_Audit_DB as audit

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("actions", [("close",), ("close", "close")])
def test_recorded_actions_are_not_duplicated_when_history_merges_layers(tmp_path, monkeypatch, actions):
    now = datetime.now(timezone.utc)
    instants = iter(now + timedelta(microseconds=index) for index in range(len(actions)))

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return next(instants)

    db = audit.ACPAuditDB(db_path=str(tmp_path / "acp_audit.db"))
    monkeypatch.setattr(audit, "get_acp_audit_db", lambda: db)
    monkeypatch.setattr(audit, "datetime", Clock)
    monkeypatch.setattr(acp, "_ACP_AUDIT_EVENTS", deque(maxlen=100))
    monkeypatch.setattr(acp, "_now_iso", lambda: (now - timedelta(seconds=1)).isoformat())
    try:
        recorded = [
            acp._acp_record_audit_event(
                action=action,
                user_id=7,
                session_id="owned-audit-session",
                metadata={"task_id": 1, "token": "synthetic-private-value"},  # nosec B105 # Synthetic redaction input, never a usable credential.
            )
            for action in actions
        ]
        history = acp._acp_list_audit_events(session_id="owned-audit-session")
        assert history == recorded  # nosec B101 # One logical action must have one merged history identity.
        assert db.query_events(session_id="owned-audit-session")[::-1] == recorded  # nosec B101 # Durable and returned events must agree.
        assert all(row["metadata"]["token"] == "[redacted]" for row in history)  # nosec B101 # Identity reuse must retain sanitization.
    finally:
        db.close()


def test_record_failure_keeps_the_sanitized_in_memory_audit(monkeypatch):
    class UnavailableAudit:
        def record_event(self, **kwargs):
            raise RuntimeError("synthetic unavailable store")

    monkeypatch.setattr(audit, "get_acp_audit_db", lambda: UnavailableAudit())
    monkeypatch.setattr(acp, "_ACP_AUDIT_EVENTS", deque(maxlen=100))
    event = acp._acp_record_audit_event(
        action="close",
        user_id=7,
        session_id="owned-audit-session",
        metadata={"authorization": "synthetic-private-value", "task_id": 1},
    )
    assert list(acp._ACP_AUDIT_EVENTS) == [event]  # nosec B101 # Best-effort persistence must not discard the memory event.
    assert event["metadata"] == {"authorization": "[redacted]", "task_id": 1}  # nosec B101 # Fallback retains exact sanitization.


def test_flush_failure_keeps_the_same_event_in_memory_and_durable_history(tmp_path, monkeypatch):
    db = audit.ACPAuditDB(db_path=str(tmp_path / "acp_audit.db"))
    monkeypatch.setattr(audit, "get_acp_audit_db", lambda: db)
    monkeypatch.setattr(acp, "_ACP_AUDIT_EVENTS", deque(maxlen=100))
    monkeypatch.setattr(acp, "_now_iso", lambda: (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat())

    def fail_flush(*, threshold):
        raise RuntimeError("synthetic unavailable flush")

    monkeypatch.setattr(db, "flush_if_needed", fail_flush)
    try:
        event = acp._acp_record_audit_event(action="close", user_id=7, session_id="owned-audit-session")
        assert acp._acp_list_audit_events(session_id="owned-audit-session") == [event]  # nosec B101 # A failed flush must retain a single event identity.
    finally:
        db.close()
