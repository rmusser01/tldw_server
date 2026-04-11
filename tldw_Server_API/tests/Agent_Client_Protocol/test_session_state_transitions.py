"""Tests for phase/activity/state_version wiring in SessionRecord and service layer."""
from __future__ import annotations

import os
import tempfile

import pytest

from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB
from tldw_Server_API.app.services.admin_acp_sessions_service import (
    ACPSessionStore,
    SessionRecord,
)


@pytest.fixture
def store():
    """Create a fresh ACPSessionStore backed by a temp DB."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "acp_sessions.db")
        db = ACPSessionsDB(db_path=path)
        yield ACPSessionStore(db=db)
        db.close()


# -------------------------------------------------------------------
# 1. SessionRecord field existence
# -------------------------------------------------------------------


def test_session_record_has_phase_activity():
    """SessionRecord includes phase and activity fields."""
    rec = SessionRecord(session_id="s1", user_id=1)
    assert rec.phase == "running"
    assert rec.activity is None
    assert rec.activity_detail is None
    assert rec.state_version == 1
    assert rec.stalled_from_activity is None


def test_session_record_custom_values():
    """SessionRecord can be constructed with custom state-machine values."""
    rec = SessionRecord(
        session_id="s2",
        user_id=1,
        phase="planning",
        activity="tool_call",
        activity_detail={"tool": "grep"},
        state_version=5,
        stalled_from_activity="thinking",
    )
    assert rec.phase == "planning"
    assert rec.activity == "tool_call"
    assert rec.activity_detail == {"tool": "grep"}
    assert rec.state_version == 5
    assert rec.stalled_from_activity == "thinking"


# -------------------------------------------------------------------
# 2. to_info_dict includes state fields
# -------------------------------------------------------------------


def test_to_info_dict_includes_state_fields():
    """to_info_dict() returns phase, activity, state_version."""
    rec = SessionRecord(
        session_id="s1",
        user_id=1,
        phase="planning",
        activity="tool_call",
        activity_detail={"tool": "bash"},
        state_version=3,
        stalled_from_activity="thinking",
    )
    info = rec.to_info_dict()
    assert info["phase"] == "planning"
    assert info["activity"] == "tool_call"
    assert info["activity_detail"] == {"tool": "bash"}
    assert info["state_version"] == 3
    assert info["stalled_from_activity"] == "thinking"


def test_to_info_dict_defaults():
    """to_info_dict() defaults for state fields."""
    rec = SessionRecord(session_id="s1", user_id=1)
    info = rec.to_info_dict()
    assert info["phase"] == "running"
    assert info["activity"] is None
    assert info["activity_detail"] is None
    assert info["state_version"] == 1
    assert info["stalled_from_activity"] is None


# -------------------------------------------------------------------
# 3. update_session_state changes phase
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_session_state_changes_phase(store: ACPSessionStore):
    """update_session_state() changes phase in DB and returns updated record."""
    await store.register_session(session_id="s1", user_id=1)

    updated = await store.update_session_state("s1", phase="planning")
    assert updated is not None
    assert updated.phase == "planning"
    assert updated.state_version == 2  # incremented from 1


@pytest.mark.asyncio
async def test_update_session_state_changes_activity(store: ACPSessionStore):
    """update_session_state() changes activity and activity_detail."""
    await store.register_session(session_id="s1", user_id=1)

    updated = await store.update_session_state(
        "s1",
        activity="tool_call",
        activity_detail={"tool": "bash", "command": "ls"},
    )
    assert updated is not None
    assert updated.activity == "tool_call"
    assert updated.activity_detail == {"tool": "bash", "command": "ls"}
    assert updated.state_version == 2


@pytest.mark.asyncio
async def test_update_session_state_round_trips_via_get(store: ACPSessionStore):
    """State changes persist and are visible via get_session."""
    await store.register_session(session_id="s1", user_id=1)
    await store.update_session_state("s1", phase="stalled", activity="waiting")

    rec = await store.get_session("s1")
    assert rec is not None
    assert rec.phase == "stalled"
    assert rec.activity == "waiting"
    assert rec.state_version == 2


# -------------------------------------------------------------------
# 4. Optimistic locking
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_session_state_version_conflict(store: ACPSessionStore):
    """update_session_state() with wrong version returns None."""
    await store.register_session(session_id="s1", user_id=1)

    result = await store.update_session_state(
        "s1",
        phase="planning",
        expected_state_version=99,  # wrong
    )
    assert result is None

    # Session should be unchanged
    rec = await store.get_session("s1")
    assert rec is not None
    assert rec.phase == "running"
    assert rec.state_version == 1


@pytest.mark.asyncio
async def test_update_session_state_correct_version(store: ACPSessionStore):
    """update_session_state() with correct version succeeds."""
    await store.register_session(session_id="s1", user_id=1)

    result = await store.update_session_state(
        "s1",
        phase="planning",
        expected_state_version=1,
    )
    assert result is not None
    assert result.phase == "planning"
    assert result.state_version == 2
