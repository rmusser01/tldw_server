"""Tests for state-machine columns in ACP_Sessions_DB (schema v11)."""
from __future__ import annotations

import os
import sqlite3
import tempfile

import pytest

from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import (
    ACPSessionsDB,
    _SCHEMA_VERSION,
)


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "acp_sessions.db")
        instance = ACPSessionsDB(db_path=path)
        yield instance
        instance.close()


def test_schema_version_is_11(tmp_path):
    """DB schema version is 11."""
    assert _SCHEMA_VERSION == 11


def test_session_persists_phase_activity(db):
    """Phase/activity columns persist in DB."""
    row = db.register_session(
        session_id="s1",
        user_id=1,
        phase="planning",
        activity="tool_call",
        activity_detail_json='{"tool": "grep"}',
        state_version=1,
    )
    assert row["phase"] == "planning"
    assert row["activity"] == "tool_call"
    assert row["activity_detail_json"] == '{"tool": "grep"}'
    assert row["state_version"] == 1

    # Defaults when not specified
    row2 = db.register_session(session_id="s2", user_id=1)
    assert row2["phase"] == "running"
    assert row2["activity"] is None
    assert row2["activity_detail_json"] is None
    assert row2["state_version"] == 1


def test_state_version_increments(db):
    """Each update increments state_version."""
    db.register_session(session_id="s1", user_id=1)
    sess = db.get_session("s1")
    assert sess["state_version"] == 1

    db.update_session_state("s1", phase="planning")
    sess = db.get_session("s1")
    assert sess["state_version"] == 2
    assert sess["phase"] == "planning"

    db.update_session_state("s1", activity="tool_call")
    sess = db.get_session("s1")
    assert sess["state_version"] == 3
    assert sess["activity"] == "tool_call"


def test_optimistic_locking_rejects_stale_version(db):
    """Update with wrong expected_state_version returns False."""
    db.register_session(session_id="s1", user_id=1)
    # state_version is 1 now

    result = db.update_session_state(
        "s1",
        phase="planning",
        expected_state_version=99,  # wrong
    )
    assert result is False

    # Session should be unchanged
    sess = db.get_session("s1")
    assert sess["state_version"] == 1
    assert sess["phase"] == "running"


def test_optimistic_locking_allows_correct_version(db):
    """Update with correct expected_state_version succeeds."""
    db.register_session(session_id="s1", user_id=1)
    # state_version is 1 now

    result = db.update_session_state(
        "s1",
        phase="planning",
        expected_state_version=1,
    )
    assert result is True

    sess = db.get_session("s1")
    assert sess["state_version"] == 2
    assert sess["phase"] == "planning"


def test_update_session_state_updates_last_activity(db):
    """update_session_state touches last_activity_at."""
    row = db.register_session(session_id="s1", user_id=1)
    old_activity = row["last_activity_at"]

    db.update_session_state("s1", phase="planning")
    sess = db.get_session("s1")
    assert sess["last_activity_at"] >= old_activity


def test_update_session_state_stalled_from_activity(db):
    """stalled_from_activity is persisted."""
    db.register_session(session_id="s1", user_id=1)
    db.update_session_state(
        "s1",
        phase="stalled",
        stalled_from_activity="tool_call",
    )
    sess = db.get_session("s1")
    assert sess["phase"] == "stalled"
    assert sess["stalled_from_activity"] == "tool_call"


def test_migration_adds_columns(tmp_path):
    """Existing v10 DB gains new columns on migration."""
    db_path = str(tmp_path / "migrate.db")

    # Create a bare v10 DB with sessions table missing new columns
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            agent_type TEXT NOT NULL DEFAULT 'custom',
            name TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active',
            cwd TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            last_activity_at TEXT,
            message_count INTEGER NOT NULL DEFAULT 0,
            prompt_tokens INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            total_tokens INTEGER NOT NULL DEFAULT 0,
            bootstrap_ready INTEGER NOT NULL DEFAULT 1,
            needs_bootstrap INTEGER NOT NULL DEFAULT 0,
            forked_from TEXT,
            tags TEXT NOT NULL DEFAULT '[]',
            mcp_servers TEXT NOT NULL DEFAULT '[]',
            persona_id TEXT,
            workspace_id TEXT,
            workspace_group_id TEXT,
            scope_snapshot_id TEXT,
            policy_snapshot_version TEXT,
            policy_snapshot_fingerprint TEXT,
            policy_snapshot_refreshed_at TEXT,
            policy_summary TEXT,
            policy_provenance_summary TEXT,
            policy_refresh_error TEXT,
            model TEXT,
            token_budget INTEGER DEFAULT NULL,
            auto_terminate_at_budget INTEGER NOT NULL DEFAULT 0,
            budget_exhausted INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.execute("PRAGMA user_version=10")
    conn.commit()
    conn.close()

    # Open via ACPSessionsDB -- migration should fire
    db = ACPSessionsDB(db_path=db_path)
    _ = db._get_conn()  # triggers schema init

    # Verify columns exist by inserting and reading
    row = db.register_session(session_id="mig1", user_id=1, phase="planning")
    assert row["phase"] == "planning"
    assert row["state_version"] == 1
    assert row["activity"] is None
    assert row["stalled_from_activity"] is None

    db.close()
