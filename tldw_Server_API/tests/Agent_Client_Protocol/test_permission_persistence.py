"""Tests for permission decision persistence ("remember" pattern).

Covers the PermissionDecisionService, GovernanceFilter integration,
and DB CRUD operations for the permission_decisions table.
"""
from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.event_bus import SessionEventBus
from tldw_Server_API.app.core.Agent_Client_Protocol.events import AgentEvent, AgentEventKind
from tldw_Server_API.app.core.Agent_Client_Protocol.governance_filter import GovernanceFilter
from tldw_Server_API.app.core.Agent_Client_Protocol.permission_decision_service import (
    PermissionDecisionService,
)
from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db():
    """Create a temporary ACP sessions DB for each test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = ACPSessionsDB(db_path=path)
    yield db
    db.close()
    os.unlink(path)


@pytest.fixture
def svc(tmp_db):
    """PermissionDecisionService backed by a temp DB."""
    return PermissionDecisionService(tmp_db)


@pytest.fixture
def bus():
    return SessionEventBus(session_id="s1")


def _make_tool_event(
    tool_name: str,
    session_id: str = "s1",
    tool_call_id: str = "tc1",
    arguments: dict | None = None,
    metadata: dict | None = None,
) -> AgentEvent:
    return AgentEvent(
        session_id=session_id,
        kind=AgentEventKind.TOOL_CALL,
        payload={
            "tool_id": tool_call_id,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "arguments": arguments or {},
        },
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Service unit tests
# ---------------------------------------------------------------------------

class TestPermissionDecisionService:
    def test_persist_and_check_global(self, svc):
        svc.persist(user_id=1, tool_pattern="bash", decision="allow", scope="global")
        assert svc.check(user_id=1, tool_name="bash") == "allow"

    def test_persist_and_check_deny(self, svc):
        svc.persist(user_id=1, tool_pattern="rm_*", decision="deny", scope="global")
        assert svc.check(user_id=1, tool_name="rm_file") == "deny"
        assert svc.check(user_id=1, tool_name="read_file") is None

    def test_session_scope_only_applies_to_matching_session(self, svc):
        svc.persist(
            user_id=1,
            tool_pattern="bash",
            decision="allow",
            scope="session",
            session_id="s1",
        )
        assert svc.check(user_id=1, tool_name="bash", session_id="s1") == "allow"
        assert svc.check(user_id=1, tool_name="bash", session_id="s2") is None
        # Without session_id, session-scoped decisions don't match
        assert svc.check(user_id=1, tool_name="bash") is None

    def test_global_scope_applies_across_sessions(self, svc):
        svc.persist(user_id=1, tool_pattern="bash", decision="allow", scope="global")
        assert svc.check(user_id=1, tool_name="bash", session_id="s1") == "allow"
        assert svc.check(user_id=1, tool_name="bash", session_id="s2") == "allow"
        assert svc.check(user_id=1, tool_name="bash") == "allow"

    def test_global_overrides_session(self, svc):
        """Global scope is checked first and takes priority."""
        svc.persist(
            user_id=1,
            tool_pattern="bash",
            decision="deny",
            scope="session",
            session_id="s1",
        )
        svc.persist(user_id=1, tool_pattern="bash", decision="allow", scope="global")
        # Global "allow" should win
        assert svc.check(user_id=1, tool_name="bash", session_id="s1") == "allow"

    def test_list_and_revoke(self, svc):
        did = svc.persist(user_id=1, tool_pattern="bash", decision="allow", scope="global")
        items = svc.list_for_user(user_id=1)
        assert len(items) == 1
        assert items[0]["id"] == did

        assert svc.revoke(did) is True
        assert svc.list_for_user(user_id=1) == []

    def test_revoke_nonexistent_returns_false(self, svc):
        assert svc.revoke("nonexistent-id") is False

    def test_different_users_isolated(self, svc):
        svc.persist(user_id=1, tool_pattern="bash", decision="allow", scope="global")
        assert svc.check(user_id=2, tool_name="bash") is None

    def test_fnmatch_wildcard(self, svc):
        svc.persist(user_id=1, tool_pattern="file_*", decision="allow", scope="global")
        assert svc.check(user_id=1, tool_name="file_read") == "allow"
        assert svc.check(user_id=1, tool_name="file_write") == "allow"
        assert svc.check(user_id=1, tool_name="bash") is None


# ---------------------------------------------------------------------------
# GovernanceFilter integration tests
# ---------------------------------------------------------------------------

class TestGovernanceFilterPersistence:
    @pytest.mark.asyncio
    async def test_remembered_allow_skips_prompt(self, bus, svc):
        """Persisted 'allow' decision auto-approves the tool call."""
        svc.persist(user_id=42, tool_pattern="bash", decision="allow", scope="global")
        gov = GovernanceFilter(
            bus=bus,
            permission_decision_service=svc,
            session_metadata={"user_id": 42},
        )

        q = bus.subscribe("test")
        ev = _make_tool_event("bash")
        await gov.process(ev)

        # The original event should be published directly (auto-approved)
        got = await asyncio.wait_for(q.get(), timeout=1.0)
        assert got is ev
        assert got.kind == AgentEventKind.TOOL_CALL
        # No pending entries
        assert gov.pending_count == 0

    @pytest.mark.asyncio
    async def test_remembered_deny_blocks_immediately(self, bus, svc):
        """Persisted 'deny' decision blocks the tool call without prompting."""
        svc.persist(user_id=42, tool_pattern="bash", decision="deny", scope="global")
        gov = GovernanceFilter(
            bus=bus,
            permission_decision_service=svc,
            session_metadata={"user_id": 42},
        )

        q = bus.subscribe("test")
        ev = _make_tool_event("bash")
        await gov.process(ev)

        # Should get a TOOL_RESULT with error, not the original event
        got = await asyncio.wait_for(q.get(), timeout=1.0)
        assert got.kind == AgentEventKind.TOOL_RESULT
        assert "denied by remembered decision" in got.payload.get("error", "")
        assert gov.pending_count == 0

    @pytest.mark.asyncio
    async def test_no_persisted_decision_falls_through(self, bus, svc):
        """When no decision is persisted, normal tier logic applies."""
        gov = GovernanceFilter(
            bus=bus,
            permission_decision_service=svc,
            session_metadata={"user_id": 42},
        )

        q = bus.subscribe("test")
        # bash is typically "individual" tier, so should be held
        ev = _make_tool_event("bash")
        await gov.process(ev)

        got = await asyncio.wait_for(q.get(), timeout=1.0)
        assert got.kind == AgentEventKind.PERMISSION_REQUEST
        assert gov.pending_count == 1

    @pytest.mark.asyncio
    async def test_session_scope_only_applies_to_matching_session(self, bus, svc):
        """Session-scoped decision only applies when session_id matches."""
        svc.persist(
            user_id=42,
            tool_pattern="bash",
            decision="allow",
            scope="session",
            session_id="s1",
        )
        gov = GovernanceFilter(
            bus=bus,
            permission_decision_service=svc,
            session_metadata={"user_id": 42},
        )

        # Same session -- auto-approved
        q = bus.subscribe("test")
        ev = _make_tool_event("bash", session_id="s1")
        await gov.process(ev)
        got = await asyncio.wait_for(q.get(), timeout=1.0)
        assert got.kind == AgentEventKind.TOOL_CALL
        assert gov.pending_count == 0

        # Different session -- falls through to tier logic
        bus2 = SessionEventBus(session_id="s2")
        gov2 = GovernanceFilter(
            bus=bus2,
            permission_decision_service=svc,
            session_metadata={"user_id": 42},
        )
        q2 = bus2.subscribe("test")
        ev2 = _make_tool_event("bash", session_id="s2")
        await gov2.process(ev2)
        got2 = await asyncio.wait_for(q2.get(), timeout=1.0)
        assert got2.kind == AgentEventKind.PERMISSION_REQUEST

    @pytest.mark.asyncio
    async def test_global_scope_applies_across_sessions(self, bus, svc):
        """Global-scoped decision applies regardless of session_id."""
        svc.persist(user_id=42, tool_pattern="bash", decision="allow", scope="global")
        gov = GovernanceFilter(
            bus=bus,
            permission_decision_service=svc,
            session_metadata={"user_id": 42},
        )

        for sid in ("s1", "s2", "s3"):
            b = SessionEventBus(session_id=sid)
            g = GovernanceFilter(
                bus=b,
                permission_decision_service=svc,
                session_metadata={"user_id": 42},
            )
            q = b.subscribe("test")
            ev = _make_tool_event("bash", session_id=sid)
            await g.process(ev)
            got = await asyncio.wait_for(q.get(), timeout=1.0)
            assert got.kind == AgentEventKind.TOOL_CALL

    @pytest.mark.asyncio
    async def test_on_permission_response_with_remember_persists(self, bus, svc):
        """on_permission_response with remember=True persists the decision."""
        gov = GovernanceFilter(
            bus=bus,
            permission_decision_service=svc,
            session_metadata={"user_id": 42},
        )

        # First, hold a tool call
        q = bus.subscribe("test")
        ev = _make_tool_event("bash", session_id="s1")
        await gov.process(ev)
        perm_req = await asyncio.wait_for(q.get(), timeout=1.0)
        assert perm_req.kind == AgentEventKind.PERMISSION_REQUEST
        request_id = perm_req.payload["request_id"]

        # Approve with remember=True
        await gov.on_permission_response(
            request_id, "approve", remember=True, scope="global"
        )

        # The tool event should now be published
        approved_ev = await asyncio.wait_for(q.get(), timeout=1.0)
        assert approved_ev.kind == AgentEventKind.TOOL_CALL

        # Verify the decision was persisted
        decisions = svc.list_for_user(user_id=42)
        assert len(decisions) == 1
        assert decisions[0]["tool_pattern"] == "bash"
        assert decisions[0]["decision"] == "allow"
        assert decisions[0]["scope"] == "global"

    @pytest.mark.asyncio
    async def test_on_permission_response_without_remember_does_not_persist(self, bus, svc):
        """on_permission_response without remember does NOT persist."""
        gov = GovernanceFilter(
            bus=bus,
            permission_decision_service=svc,
            session_metadata={"user_id": 42},
        )

        q = bus.subscribe("test")
        ev = _make_tool_event("bash", session_id="s1")
        await gov.process(ev)
        perm_req = await asyncio.wait_for(q.get(), timeout=1.0)
        request_id = perm_req.payload["request_id"]

        await gov.on_permission_response(request_id, "approve")

        # No decision persisted
        assert svc.list_for_user(user_id=42) == []

    @pytest.mark.asyncio
    async def test_on_permission_response_deny_with_remember(self, bus, svc):
        """Deny with remember=True persists a 'deny' decision."""
        gov = GovernanceFilter(
            bus=bus,
            permission_decision_service=svc,
            session_metadata={"user_id": 42},
        )

        q = bus.subscribe("test")
        ev = _make_tool_event("bash", session_id="s1")
        await gov.process(ev)
        perm_req = await asyncio.wait_for(q.get(), timeout=1.0)
        request_id = perm_req.payload["request_id"]

        await gov.on_permission_response(
            request_id,
            "deny",
            reason="too dangerous",
            remember=True,
            scope="session",
        )

        # Verify deny error was published
        error_ev = await asyncio.wait_for(q.get(), timeout=1.0)
        assert error_ev.kind == AgentEventKind.TOOL_RESULT
        assert error_ev.payload.get("is_error") is True

        # Verify the decision was persisted as "deny"
        decisions = svc.list_for_user(user_id=42)
        assert len(decisions) == 1
        assert decisions[0]["decision"] == "deny"
        assert decisions[0]["scope"] == "session"
        assert decisions[0]["session_id"] == "s1"

    @pytest.mark.asyncio
    async def test_no_perm_service_remember_is_noop(self, bus):
        """When no perm service is set, remember=True does nothing."""
        gov = GovernanceFilter(
            bus=bus,
            permission_decision_service=None,
            session_metadata={"user_id": 42},
        )

        q = bus.subscribe("test")
        ev = _make_tool_event("bash", session_id="s1")
        await gov.process(ev)
        perm_req = await asyncio.wait_for(q.get(), timeout=1.0)
        request_id = perm_req.payload["request_id"]

        # Should not raise even though no perm service
        await gov.on_permission_response(
            request_id, "approve", remember=True, scope="global"
        )
        approved_ev = await asyncio.wait_for(q.get(), timeout=1.0)
        assert approved_ev.kind == AgentEventKind.TOOL_CALL

    @pytest.mark.asyncio
    async def test_snapshot_deny_takes_precedence_over_persisted_allow(self, bus, svc):
        """MCPHub snapshot deny (step 1-3) takes precedence over persisted allow (step 4)."""
        from unittest.mock import MagicMock

        svc.persist(user_id=42, tool_pattern="bash", decision="allow", scope="global")

        snapshot = MagicMock()
        snapshot.resolved_policy_document = {"denied_tools": ["bash"]}

        gov = GovernanceFilter(
            bus=bus,
            policy_snapshot=snapshot,
            permission_decision_service=svc,
            session_metadata={"user_id": 42},
        )

        q = bus.subscribe("test")
        ev = _make_tool_event("bash")
        await gov.process(ev)

        got = await asyncio.wait_for(q.get(), timeout=1.0)
        # Snapshot deny wins
        assert got.kind == AgentEventKind.TOOL_RESULT
        assert "denied by policy" in got.payload.get("error", "")


# ---------------------------------------------------------------------------
# DB CRUD tests
# ---------------------------------------------------------------------------

class TestPermissionDecisionsDB:
    def test_insert_and_list(self, tmp_db):
        tmp_db.insert_permission_decision(
            id="d1",
            user_id=1,
            tool_pattern="bash",
            decision="allow",
            scope="global",
        )
        decisions = tmp_db.list_permission_decisions(user_id=1)
        assert len(decisions) == 1
        assert decisions[0]["id"] == "d1"
        assert decisions[0]["tool_pattern"] == "bash"
        assert decisions[0]["decision"] == "allow"

    def test_delete(self, tmp_db):
        tmp_db.insert_permission_decision(
            id="d1",
            user_id=1,
            tool_pattern="bash",
            decision="allow",
        )
        assert tmp_db.delete_permission_decision("d1") is True
        assert tmp_db.delete_permission_decision("d1") is False
        assert tmp_db.list_permission_decisions(user_id=1) == []

    def test_get_by_id(self, tmp_db):
        tmp_db.insert_permission_decision(
            id="d1",
            user_id=1,
            tool_pattern="bash",
            decision="deny",
        )
        d = tmp_db.get_permission_decision("d1")
        assert d is not None
        assert d["decision"] == "deny"
        assert tmp_db.get_permission_decision("nonexistent") is None

    def test_check_permission_decision(self, tmp_db):
        tmp_db.insert_permission_decision(
            id="d1",
            user_id=1,
            tool_pattern="bash",
            decision="allow",
            scope="global",
        )
        assert tmp_db.check_permission_decision(1, "bash") == "allow"
        assert tmp_db.check_permission_decision(1, "read_file") is None
        assert tmp_db.check_permission_decision(2, "bash") is None

    def test_expired_decisions_filtered(self, tmp_db):
        """Expired decisions should not appear in list_permission_decisions."""
        tmp_db.insert_permission_decision(
            id="d1",
            user_id=1,
            tool_pattern="bash",
            decision="allow",
            scope="global",
            expires_at="2020-01-01T00:00:00+00:00",
        )
        assert tmp_db.list_permission_decisions(user_id=1) == []

    def test_schema_version_is_9(self, tmp_db):
        from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import _SCHEMA_VERSION

        conn = tmp_db._get_conn()
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == _SCHEMA_VERSION
