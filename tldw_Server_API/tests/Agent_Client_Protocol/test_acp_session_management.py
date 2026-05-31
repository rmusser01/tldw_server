"""Tests for ACP session TTL, quotas, and audit persistence (Phase 1)."""
from __future__ import annotations

import asyncio
import importlib.machinery
import os
import sys
import tempfile
import time
import types
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

# Stub heavyweight audio deps before app import.
if "torch" not in sys.modules:
    _fake_torch = types.ModuleType("torch")
    _fake_torch.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    _fake_torch.Tensor = object
    _fake_torch.nn = types.SimpleNamespace(Module=object)
    sys.modules["torch"] = _fake_torch

if "faster_whisper" not in sys.modules:
    _fake_fw = types.ModuleType("faster_whisper")
    _fake_fw.__spec__ = importlib.machinery.ModuleSpec("faster_whisper", loader=None)

    class _StubWhisperModel:
        def __init__(self, *args, **kwargs):
            pass

    _fake_fw.WhisperModel = _StubWhisperModel
    _fake_fw.BatchedInferencePipeline = _StubWhisperModel
    sys.modules["faster_whisper"] = _fake_fw

if "transformers" not in sys.modules:
    _fake_tf = types.ModuleType("transformers")
    _fake_tf.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)

    class _StubProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class _StubModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    _fake_tf.AutoProcessor = _StubProcessor
    _fake_tf.Qwen2AudioForConditionalGeneration = _StubModel
    sys.modules["transformers"] = _fake_tf


# ---- Session Store quota/TTL tests ----

@pytest.fixture
def session_store(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB
    from tldw_Server_API.app.services.admin_acp_sessions_service import ACPSessionStore
    _db = ACPSessionsDB(db_path=str(tmp_path / "session_mgmt_test.db"))
    return ACPSessionStore(db=_db)


@pytest.mark.asyncio
async def test_session_quota_enforcement(session_store):
    """Max concurrent sessions quota is enforced."""
    session_store.configure_quotas(max_concurrent_per_user=2)

    # Create 2 sessions — should be fine
    await session_store.register_session("s1", user_id=1, name="Session 1")
    await session_store.register_session("s2", user_id=1, name="Session 2")

    # Third session should hit quota
    error = await session_store.check_session_quota(user_id=1)
    assert error is not None
    assert error["code"] == "quota_exceeded"
    assert error["current"] == 2
    assert error["limit"] == 2


@pytest.mark.asyncio
async def test_session_quota_allows_after_close(session_store):
    """Closing a session frees up quota."""
    session_store.configure_quotas(max_concurrent_per_user=1)

    await session_store.register_session("s1", user_id=1, name="Session 1")
    error = await session_store.check_session_quota(user_id=1)
    assert error is not None

    await session_store.close_session("s1")
    error = await session_store.check_session_quota(user_id=1)
    assert error is None


@pytest.mark.asyncio
async def test_session_quota_per_user(session_store):
    """Quota is per-user, not global."""
    session_store.configure_quotas(max_concurrent_per_user=1)

    await session_store.register_session("s1", user_id=1, name="User 1 Session")

    # User 2 should not be blocked by user 1's sessions
    error = await session_store.check_session_quota(user_id=2)
    assert error is None


@pytest.mark.asyncio
async def test_token_quota_enforcement(session_store):
    """Token quota is enforced per session."""
    session_store.configure_quotas(max_tokens_per_session=100)

    await session_store.register_session("s1", user_id=1, name="Session 1")

    # Record a prompt that pushes tokens over limit
    await session_store.record_prompt(
        "s1",
        [{"role": "user", "content": "hi"}],
        {"stopReason": "end", "usage": {"prompt_tokens": 50, "completion_tokens": 60}},
    )

    error = await session_store.check_token_quota("s1")
    assert error is not None
    assert error["code"] == "token_quota_exceeded"
    assert error["current"] == 110


@pytest.mark.asyncio
async def test_token_quota_ok_under_limit(session_store):
    """Token quota check passes when under limit."""
    session_store.configure_quotas(max_tokens_per_session=1000)

    await session_store.register_session("s1", user_id=1, name="Session 1")
    await session_store.record_prompt(
        "s1",
        [{"role": "user", "content": "hi"}],
        {"stopReason": "end", "usage": {"prompt_tokens": 10, "completion_tokens": 20}},
    )

    error = await session_store.check_token_quota("s1")
    assert error is None


@pytest.mark.asyncio
async def test_session_eviction(session_store):
    """Expired sessions are evicted by cleanup."""
    session_store.configure_quotas(session_ttl_seconds=0)  # Immediate expiry

    await session_store.register_session("s1", user_id=1, name="Session 1")
    rec = await session_store.get_session("s1")
    assert rec is not None
    assert rec.status == "active"

    # Run eviction
    evicted = await session_store._evict_expired_sessions()
    assert evicted == 1

    rec = await session_store.get_session("s1")
    assert rec.status == "closed"


@pytest.mark.asyncio
async def test_retention_maintenance_purges_expired_sessions_and_audit_events(session_store, tmp_path):
    """Store-level retention maintenance enforces session and audit retention."""
    from tldw_Server_API.app.core.DB_Management.ACP_Audit_DB import ACPAuditDB

    session_store.configure_retention(session_retention_days=0, audit_retention_days=0)
    await session_store.register_session("s1", user_id=1, name="Session 1")
    await session_store.record_prompt(
        "s1",
        [{"role": "user", "content": "secret prompt"}],
        {"content": "secret response", "usage": {}},
    )
    await session_store.close_session("s1")

    audit_db = ACPAuditDB(db_path=str(tmp_path / "audit.db"), retention_days=0)
    audit_db.record_event(action="session_close", user_id=1, session_id="s1")
    audit_db.flush()

    result = await session_store.run_retention_maintenance(audit_db=audit_db)

    assert result["sessions_evicted"] == 0
    assert result["sessions_purged"] == 1
    assert result["audit_events_purged"] == 1
    assert await session_store.get_session("s1") is None
    assert audit_db.query_events(session_id="s1") == []


@pytest.mark.asyncio
async def test_quota_status(session_store):
    """Quota status returns current usage."""
    session_store.configure_quotas(max_concurrent_per_user=5, max_tokens_per_session=1000)

    await session_store.register_session("s1", user_id=1, name="Session 1")
    status = await session_store.get_quota_status(user_id=1, session_id="s1")
    assert status["concurrent_sessions"]["current"] == 1
    assert status["concurrent_sessions"]["limit"] == 5
    assert status["session_tokens"]["current"] == 0
    assert status["session_tokens"]["limit"] == 1000


# ---- Audit DB tests ----

def test_audit_db_record_and_flush():
    """Audit events are recorded and flushed to SQLite."""
    from tldw_Server_API.app.core.DB_Management.ACP_Audit_DB import ACPAuditDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = ACPAuditDB(db_path=os.path.join(tmpdir, "audit.db"))
        db.record_event(action="session_new", user_id=1, session_id="s1", metadata={"test": True})
        db.record_event(action="prompt", user_id=1, session_id="s1")

        # Hot cache should have 2 events
        cache = db.get_hot_cache()
        assert len(cache) == 2

        # Flush to disk
        flushed = db.flush()
        assert flushed == 2

        # Query from SQLite
        events = db.query_events(session_id="s1")
        assert len(events) == 2
        assert events[0]["action"] == "prompt"  # DESC order
        assert events[1]["action"] == "session_new"

        db.close()


def test_audit_db_query_filters():
    """Audit events can be filtered by user_id and action."""
    from tldw_Server_API.app.core.DB_Management.ACP_Audit_DB import ACPAuditDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = ACPAuditDB(db_path=os.path.join(tmpdir, "audit.db"))
        db.record_event(action="session_new", user_id=1, session_id="s1")
        db.record_event(action="prompt", user_id=2, session_id="s2")
        db.record_event(action="session_close", user_id=1, session_id="s1")
        db.flush()

        # Filter by user_id
        events = db.query_events(user_id=1)
        assert len(events) == 2

        # Filter by action
        events = db.query_events(action="prompt")
        assert len(events) == 1
        assert events[0]["user_id"] == 2

        db.close()


def test_audit_db_hot_cache_filter():
    """Hot cache can be filtered by session_id."""
    from tldw_Server_API.app.core.DB_Management.ACP_Audit_DB import ACPAuditDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = ACPAuditDB(db_path=os.path.join(tmpdir, "audit.db"))
        db.record_event(action="a", user_id=1, session_id="s1")
        db.record_event(action="b", user_id=1, session_id="s2")

        filtered = db.get_hot_cache(session_id="s1")
        assert len(filtered) == 1
        assert filtered[0]["session_id"] == "s1"

        db.close()


def test_audit_db_purge():
    """Old events are purged by retention policy."""
    from tldw_Server_API.app.core.DB_Management.ACP_Audit_DB import ACPAuditDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = ACPAuditDB(db_path=os.path.join(tmpdir, "audit.db"), retention_days=0)
        db.record_event(action="old", user_id=1, session_id="s1")
        db.flush()

        # Purge with 0-day retention should delete everything
        deleted = db.purge_old_events()
        assert deleted >= 1

        events = db.query_events()
        assert len(events) == 0

        db.close()


def test_audit_db_purge_trims_hot_cache():
    """Retention maintenance removes stale events from SQLite and hot cache."""
    from tldw_Server_API.app.core.DB_Management.ACP_Audit_DB import ACPAuditDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = ACPAuditDB(db_path=os.path.join(tmpdir, "audit.db"), retention_days=0)
        db.record_event(action="old", user_id=1, session_id="s1")
        db.flush()

        deleted = db.purge_old_events(now=time.time() + 5)

        assert deleted >= 1
        assert db.query_events(session_id="s1") == []
        assert db.get_hot_cache(session_id="s1") == []

        db.close()


def test_audit_db_negative_retention_does_not_use_future_cutoff():
    """Negative retention values are normalized before cutoff calculation."""
    from tldw_Server_API.app.core.DB_Management.ACP_Audit_DB import ACPAuditDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = ACPAuditDB(db_path=os.path.join(tmpdir, "audit.db"), retention_days=-10)
        db.record_event(action="future-safe", user_id=1, session_id="s1")
        db.flush()
        conn = db._get_conn()
        conn.execute("UPDATE acp_audit_events SET created_at = ?", (2000.0,))
        conn.commit()

        deleted = db.purge_old_events(now=1000.0)

        assert deleted == 0
        assert len(db.query_events(session_id="s1")) == 1

        db.close()


def test_audit_db_singleton_updates_explicit_retention(monkeypatch):
    """Explicit retention overrides are applied to an existing audit singleton."""
    import tldw_Server_API.app.core.DB_Management.ACP_Audit_DB as audit_db_module

    monkeypatch.setattr(audit_db_module, "_audit_db", None)
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "audit.db")
        first = audit_db_module.get_acp_audit_db(db_path=db_path, retention_days=30)
        second = audit_db_module.get_acp_audit_db(db_path=db_path, retention_days=0)

        assert second is first
        second.record_event(action="retention_override", user_id=1, session_id="s1")
        second.flush()

        deleted = second.purge_old_events(now=time.time() + 5)
        assert deleted >= 1
        assert second.query_events() == []

        second.close()
        monkeypatch.setattr(audit_db_module, "_audit_db", None)


def test_audit_db_singleton_clamps_negative_explicit_retention(monkeypatch):
    """Explicit singleton retention overrides cannot remain negative."""
    import tldw_Server_API.app.core.DB_Management.ACP_Audit_DB as audit_db_module

    monkeypatch.setattr(audit_db_module, "_audit_db", None)
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "audit.db")
        first = audit_db_module.get_acp_audit_db(db_path=db_path, retention_days=30)
        second = audit_db_module.get_acp_audit_db(db_path=db_path, retention_days=-5)

        assert second is first
        assert second._retention_days == 0

        second.close()
        monkeypatch.setattr(audit_db_module, "_audit_db", None)


def test_audit_db_double_flush_is_idempotent():
    """Flushing with no new events returns 0."""
    from tldw_Server_API.app.core.DB_Management.ACP_Audit_DB import ACPAuditDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = ACPAuditDB(db_path=os.path.join(tmpdir, "audit.db"))
        db.record_event(action="test", user_id=1, session_id="s1")
        assert db.flush() == 1
        assert db.flush() == 0  # No new events
        db.close()


def test_acp_record_audit_event_sanitizes_persistence_failure(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints
    import tldw_Server_API.app.core.DB_Management.ACP_Audit_DB as audit_db_module

    warnings: list[str] = []

    def _fail_audit_db():
        raise RuntimeError("audit backend exploded at /private/audit.db")

    monkeypatch.setattr(audit_db_module, "get_acp_audit_db", _fail_audit_db)
    monkeypatch.setattr(
        acp_endpoints.logger,
        "warning",
        lambda message, *args, **kwargs: warnings.append(str(message).format(*args)),
    )
    monkeypatch.setattr(acp_endpoints.logger, "info", lambda *args, **kwargs: None)

    event = acp_endpoints._acp_record_audit_event(
        action="session_new",
        user_id=1,
        session_id="audit-sanitizer-session",
        metadata={"source": "unit"},
    )

    assert event["action"] == "session_new"
    assert event["session_id"] == "audit-sanitizer-session"
    assert warnings == ["ACP audit persistence failed"]


@pytest.mark.asyncio
async def test_permission_response_audit_records_policy_snapshot_fingerprint(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints
    from tldw_Server_API.app.core.Agent_Client_Protocol.runner_client import (
        ACPRunnerClient,
        PendingPermission,
        SessionWebSocketRegistry,
    )

    cfg = SimpleNamespace(
        command="echo",
        args=[],
        env={},
        cwd=None,
        startup_timeout_sec=0,
    )
    client = ACPRunnerClient(cfg)
    session_id = "audit-session"
    request_id = "perm-1"
    future = asyncio.get_running_loop().create_future()

    registry = SessionWebSocketRegistry(session_id=session_id)
    registry.pending_permissions[request_id] = PendingPermission(
        request_id=request_id,
        session_id=session_id,
        tool_name="fs.write",
        tool_arguments={"path": "README.md"},
        acp_message_id="acp-1",
        future=future,
        policy_snapshot_fingerprint="snapshot-audit-123",
        approval_requirement="approval_required",
        governance_reason="policy_approval_required",
    )
    client._ws_registry[session_id] = registry

    recorded: list[dict[str, object]] = []

    monkeypatch.setattr(
        acp_endpoints,
        "_acp_record_audit_event",
        lambda *, action, user_id, session_id, metadata=None: recorded.append(
            {
                "action": action,
                "user_id": user_id,
                "session_id": session_id,
                "metadata": metadata or {},
            }
        ),
    )

    class _Stream:
        async def send_json(self, payload):
            raise AssertionError(f"unexpected error payload: {payload}")

    await acp_endpoints._handle_client_message(
        client,
        session_id,
        {
            "type": "permission_response",
            "request_id": request_id,
            "approved": True,
        },
        _Stream(),
        user_id=7,
    )

    assert recorded
    event = recorded[-1]
    assert event["action"] == "permission_response"
    assert event["metadata"]["approved"] is True
    assert event["metadata"]["policy_snapshot_fingerprint"] == "snapshot-audit-123"
    assert event["metadata"]["approval_requirement"] == "approval_required"


@pytest.mark.asyncio
async def test_permission_response_audit_uses_client_metadata_accessor(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints

    recorded: list[dict[str, object]] = []

    monkeypatch.setattr(
        acp_endpoints,
        "_acp_record_audit_event",
        lambda *, action, user_id, session_id, metadata=None: recorded.append(
            {
                "action": action,
                "user_id": user_id,
                "session_id": session_id,
                "metadata": metadata or {},
            }
        ),
    )

    class _Client:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, bool, str | None]] = []

        def get_pending_permission_metadata(self, session_id: str, request_id: str) -> dict[str, object]:
            assert session_id == "audit-accessor-session"
            assert request_id == "perm-accessor"
            return {
                "tool_name": "fs.read",
                "approval_requirement": "approval_required",
                "policy_snapshot_fingerprint": "snapshot-accessor",
            }

        async def respond_to_permission(
            self,
            session_id: str,
            request_id: str,
            approved: bool,
            batch_approve_tier: str | None = None,
        ) -> bool:
            self.calls.append((session_id, request_id, approved, batch_approve_tier))
            return True

    class _Stream:
        async def send_json(self, payload):
            raise AssertionError(f"unexpected error payload: {payload}")

    client = _Client()
    await acp_endpoints._handle_client_message(
        client,
        "audit-accessor-session",
        {
            "type": "permission_response",
            "request_id": "perm-accessor",
            "approved": True,
        },
        _Stream(),
        user_id=11,
    )

    assert client.calls == [("audit-accessor-session", "perm-accessor", True, None)]
    assert recorded[-1]["metadata"]["tool_name"] == "fs.read"
    assert recorded[-1]["metadata"]["policy_snapshot_fingerprint"] == "snapshot-accessor"


@pytest.mark.asyncio
async def test_permission_response_audit_logs_metadata_accessor_failure(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints

    warnings: list[str] = []
    recorded: list[dict[str, object]] = []

    monkeypatch.setattr(
        acp_endpoints,
        "_acp_record_audit_event",
        lambda *, action, user_id, session_id, metadata=None: recorded.append(
            {
                "action": action,
                "user_id": user_id,
                "session_id": session_id,
                "metadata": metadata or {},
            }
        ),
    )
    monkeypatch.setattr(
        acp_endpoints.logger,
        "warning",
        lambda message, *args: warnings.append(str(message).format(*args)),
    )

    class _Client:
        def get_pending_permission_metadata(self, session_id: str, request_id: str) -> dict[str, object]:
            del session_id, request_id
            raise RuntimeError("metadata lookup failed")

        async def respond_to_permission(
            self,
            session_id: str,
            request_id: str,
            approved: bool,
            batch_approve_tier: str | None = None,
        ) -> bool:
            del session_id, request_id, approved, batch_approve_tier
            return True

    class _Stream:
        async def send_json(self, payload):
            raise AssertionError(f"unexpected error payload: {payload}")

    await acp_endpoints._handle_client_message(
        _Client(),
        "audit-accessor-failure",
        {
            "type": "permission_response",
            "request_id": "perm-accessor-failure",
            "approved": False,
        },
        _Stream(),
        user_id=13,
    )

    assert recorded[-1]["metadata"]["approved"] is False
    assert "policy_snapshot_fingerprint" not in recorded[-1]["metadata"]
    assert any("Failed to load ACP pending permission metadata" in warning for warning in warnings)
