"""Tests for ACP Sessions SQLite persistence."""
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone

import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import AgentRegistry
from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB, _ensure_column


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "acp_sessions.db")
        instance = ACPSessionsDB(db_path=path)
        yield instance
        instance.close()


class TestSessionCRUD:
    def test_register_and_get_session(self, db):
        row = db.register_session(
            session_id="s1",
            user_id=1,
            agent_type="claude_code",
            name="Test Session",
            cwd="/tmp/work",
        )
        assert row is not None
        assert row["session_id"] == "s1"
        assert row["user_id"] == 1
        assert row["agent_type"] == "claude_code"
        assert row["name"] == "Test Session"
        assert row["status"] == "active"

        fetched = db.get_session("s1")
        assert fetched is not None
        assert fetched["session_id"] == "s1"

    def test_get_session_not_found(self, db):
        assert db.get_session("nonexistent") is None

    def test_close_session(self, db):
        db.register_session(session_id="s1", user_id=1)
        db.close_session("s1")
        rec = db.get_session("s1")
        assert rec["status"] == "closed"

    def test_list_sessions_filters(self, db):
        db.register_session(session_id="s1", user_id=1, agent_type="claude_code")
        db.register_session(session_id="s2", user_id=2, agent_type="codex")
        db.register_session(session_id="s3", user_id=1, agent_type="claude_code")
        db.close_session("s3")

        # Filter by user
        sessions, total = db.list_sessions(user_id=1)
        assert total == 2
        assert len(sessions) == 2

        # Filter by status
        sessions, total = db.list_sessions(user_id=1, status="active")
        assert total == 1
        assert sessions[0]["session_id"] == "s1"

    def test_register_session_with_tags_and_mcp(self, db):
        db.register_session(
            session_id="s1",
            user_id=1,
            tags=["workflow", "test"],
            mcp_servers=[{"name": "fs", "type": "stdio"}],
        )
        rec = db.get_session("s1")
        assert rec["tags"] == ["workflow", "test"]
        assert rec["mcp_servers"] == [{"name": "fs", "type": "stdio"}]

    def test_register_session_persists_sandbox_context_and_filters_by_workspace(self, db):
        db.register_session(
            session_id="s-workspace",
            user_id=1,
            agent_type="codex",
            workspace_id="workspace-1",
            sandbox_session_id="sandbox-session-1",
            sandbox_run_id="sandbox-run-1",
        )
        db.register_session(
            session_id="s-other",
            user_id=1,
            agent_type="codex",
            workspace_id="workspace-2",
        )

        row = db.get_session("s-workspace")
        assert row["sandbox_session_id"] == "sandbox-session-1"
        assert row["sandbox_run_id"] == "sandbox-run-1"

        rows, total = db.list_sessions(user_id=1, workspace_id="workspace-1")
        assert total == 1
        assert rows[0]["session_id"] == "s-workspace"

    def test_list_sessions_workspace_filter_uses_direct_predicate(self, db):
        db.register_session(
            session_id="s-workspace",
            user_id=1,
            agent_type="codex",
            workspace_id="workspace-1",
        )

        statements: list[str] = []
        conn = db._get_conn()
        conn.set_trace_callback(statements.append)
        try:
            db.list_sessions(workspace_id="workspace-1")
        finally:
            conn.set_trace_callback(None)

        session_queries = [
            statement
            for statement in statements
            if "FROM sessions" in statement and "workspace_id" in statement
        ]
        assert session_queries
        assert all("IS NULL OR" not in statement for statement in session_queries)
        assert all("workspace_id =" in statement for statement in session_queries)

    def test_register_session_with_policy_snapshot_fields(self, db):
        row = db.register_session(
            session_id="s1",
            user_id=1,
            policy_snapshot_version="v1",
            policy_snapshot_fingerprint="fingerprint-1",
            policy_snapshot_refreshed_at="2026-03-14T12:00:00+00:00",
            policy_summary={"allowed_tools": 2, "denied_tools": 1},
            policy_provenance_summary={"sources": ["mcp_hub"]},
            policy_refresh_error="temporary mismatch",
        )
        assert row["policy_snapshot_version"] == "v1"
        assert row["policy_snapshot_fingerprint"] == "fingerprint-1"
        assert row["policy_snapshot_refreshed_at"] == "2026-03-14T12:00:00+00:00"
        assert row["policy_summary"] == {"allowed_tools": 2, "denied_tools": 1}
        assert row["policy_provenance_summary"] == {"sources": ["mcp_hub"]}
        assert row["policy_refresh_error"] == "temporary mismatch"

    def test_update_policy_snapshot_fields_can_set_and_clear_values(self, db):
        db.register_session(session_id="s1", user_id=1)

        db.update_policy_snapshot_state(
            "s1",
            policy_snapshot_version="v2",
            policy_snapshot_fingerprint="fingerprint-2",
            policy_snapshot_refreshed_at="2026-03-14T13:00:00+00:00",
            policy_summary={"approval_required": True},
            policy_provenance_summary={"resolved_capabilities": ["tool.invoke.research"]},
            policy_refresh_error="refresh failed",
        )

        updated = db.get_session("s1")
        assert updated["policy_snapshot_version"] == "v2"
        assert updated["policy_snapshot_fingerprint"] == "fingerprint-2"
        assert updated["policy_snapshot_refreshed_at"] == "2026-03-14T13:00:00+00:00"
        assert updated["policy_summary"] == {"approval_required": True}
        assert updated["policy_provenance_summary"] == {
            "resolved_capabilities": ["tool.invoke.research"]
        }
        assert updated["policy_refresh_error"] == "refresh failed"

        db.update_policy_snapshot_state(
            "s1",
            policy_snapshot_version=None,
            policy_snapshot_fingerprint=None,
            policy_snapshot_refreshed_at=None,
            policy_summary=None,
            policy_provenance_summary=None,
            policy_refresh_error=None,
        )

        cleared = db.get_session("s1")
        assert cleared["policy_snapshot_version"] is None
        assert cleared["policy_snapshot_fingerprint"] is None
        assert cleared["policy_snapshot_refreshed_at"] is None
        assert cleared["policy_summary"] is None
        assert cleared["policy_provenance_summary"] is None
        assert cleared["policy_refresh_error"] is None

    def test_update_session_activity(self, db):
        db.register_session(session_id="s1", user_id=1)
        original = db.get_session("s1")
        db.update_activity("s1")
        updated = db.get_session("s1")
        # last_activity_at should be updated (or at least not None)
        assert updated["last_activity_at"] is not None

    def test_set_session_error(self, db):
        db.register_session(session_id="s1", user_id=1)
        db.set_session_status("s1", "error")
        rec = db.get_session("s1")
        assert rec["status"] == "error"

    def test_list_sessions_pagination(self, db):
        for i in range(10):
            db.register_session(session_id=f"s{i}", user_id=1)
        sessions, total = db.list_sessions(user_id=1, limit=3, offset=0)
        assert total == 10
        assert len(sessions) == 3
        sessions2, _ = db.list_sessions(user_id=1, limit=3, offset=3)
        assert len(sessions2) == 3
        # No overlap
        ids1 = {s["session_id"] for s in sessions}
        ids2 = {s["session_id"] for s in sessions2}
        assert ids1.isdisjoint(ids2)

    def test_delete_session(self, db):
        db.register_session(session_id="s1", user_id=1)
        assert db.delete_session("s1") is True
        assert db.get_session("s1") is None
        assert db.delete_session("s1") is False  # already deleted

    def test_register_defaults(self, db):
        """Verify default values for optional fields."""
        row = db.register_session(session_id="s1", user_id=1)
        assert row["agent_type"] == "custom"
        assert row["name"] == ""
        assert row["cwd"] == ""
        assert row["tags"] == []
        assert row["mcp_servers"] == []
        assert row["message_count"] == 0
        assert row["prompt_tokens"] == 0
        assert row["completion_tokens"] == 0
        assert row["total_tokens"] == 0
        assert row["bootstrap_ready"] is True


def test_ensure_column_rejects_unknown_table_or_definition(db):
    conn = db._get_conn()

    with pytest.raises(ValueError, match="Unsupported ACP session migration target"):
        _ensure_column(conn, "unknown_table", "policy_snapshot_version", "policy_snapshot_version TEXT")

    with pytest.raises(ValueError, match="Unsupported ACP session migration target"):
        _ensure_column(conn, "sessions", "policy_snapshot_version", "policy_snapshot_version INTEGER")
        assert row["needs_bootstrap"] is False
        assert row["forked_from"] is None
        assert row["persona_id"] is None
        assert row["workspace_id"] is None

    def test_boolean_fields_conversion(self, db):
        """Ensure integer booleans in SQLite are returned as Python bools."""
        db.register_session(session_id="s1", user_id=1)
        rec = db.get_session("s1")
        assert isinstance(rec["bootstrap_ready"], bool)
        assert isinstance(rec["needs_bootstrap"], bool)

    def test_list_sessions_filter_by_agent_type(self, db):
        db.register_session(session_id="s1", user_id=1, agent_type="claude_code")
        db.register_session(session_id="s2", user_id=1, agent_type="codex")
        sessions, total = db.list_sessions(agent_type="codex")
        assert total == 1
        assert sessions[0]["session_id"] == "s2"

    def test_created_at_is_populated(self, db):
        row = db.register_session(session_id="s1", user_id=1)
        assert row["created_at"] is not None
        assert len(row["created_at"]) > 0  # ISO timestamp string


class TestSessionMessages:
    def test_record_prompt_stores_messages(self, db):
        db.register_session(session_id="s1", user_id=1)
        prompt = [{"role": "user", "content": "Hello"}]
        result = {"content": [{"text": "Hi there"}], "usage": {"input_tokens": 10, "output_tokens": 5}}
        usage = db.record_prompt("s1", prompt, result)
        assert usage is not None
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 5
        assert usage["total_tokens"] == 15

        rec = db.get_session("s1")
        assert rec["message_count"] == 2
        assert rec["total_tokens"] == 15

    def test_record_prompt_nonexistent_session(self, db):
        assert db.record_prompt("nope", [], {}) is None

    def test_get_messages(self, db):
        db.register_session(session_id="s1", user_id=1)
        db.record_prompt(
            "s1",
            [{"role": "user", "content": "Hello"}],
            {"content": [{"text": "Hi"}], "usage": {}},
        )
        messages = db.get_messages("s1")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_get_messages_with_limit(self, db):
        db.register_session(session_id="s1", user_id=1)
        for i in range(5):
            db.record_prompt(
                "s1",
                [{"role": "user", "content": f"msg {i}"}],
                {"content": [{"text": f"reply {i}"}], "usage": {}},
            )
        messages = db.get_messages("s1", limit=4)
        assert len(messages) == 4

    def test_record_prompt_accumulates_tokens(self, db):
        db.register_session(session_id="s1", user_id=1)
        db.record_prompt("s1", [{"role": "user", "content": "a"}],
                         {"content": "r1", "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
        db.record_prompt("s1", [{"role": "user", "content": "b"}],
                         {"content": "r2", "usage": {"prompt_tokens": 20, "completion_tokens": 10}})
        rec = db.get_session("s1")
        assert rec["prompt_tokens"] == 30
        assert rec["completion_tokens"] == 15
        assert rec["total_tokens"] == 45
        assert rec["message_count"] == 4

    def test_record_prompt_handles_missing_usage(self, db):
        db.register_session(session_id="s1", user_id=1)
        usage = db.record_prompt("s1", [{"role": "user", "content": "hello"}],
                                 {"content": "world"})
        assert usage["prompt_tokens"] == 0
        assert usage["total_tokens"] == 0

    def test_update_token_usage_directly(self, db):
        db.register_session(session_id="s1", user_id=1)
        db.update_token_usage("s1", prompt_tokens=50, completion_tokens=25)
        rec = db.get_session("s1")
        assert rec["prompt_tokens"] == 50
        assert rec["completion_tokens"] == 25
        assert rec["total_tokens"] == 75
        # Second call accumulates
        db.update_token_usage("s1", prompt_tokens=10, completion_tokens=5)
        rec = db.get_session("s1")
        assert rec["prompt_tokens"] == 60
        assert rec["completion_tokens"] == 30
        assert rec["total_tokens"] == 90


class TestForkSession:
    def test_fork_copies_messages(self, db):
        db.register_session(session_id="s1", user_id=1, agent_type="claude_code")
        db.record_prompt("s1", [{"role": "user", "content": "Hello"}],
                         {"content": [{"text": "Hi"}], "usage": {}})
        db.record_prompt("s1", [{"role": "user", "content": "Next"}],
                         {"content": [{"text": "OK"}], "usage": {}})
        forked = db.fork_session("s1", "s2", message_index=1, user_id=1)
        assert forked is not None
        assert forked["forked_from"] == "s1"
        assert forked["agent_type"] == "claude_code"
        assert forked["needs_bootstrap"] is True
        messages = db.get_messages("s2")
        assert len(messages) == 2  # messages 0 and 1

    def test_fork_nonexistent_source(self, db):
        assert db.fork_session("nope", "s2", message_index=0, user_id=1) is None

    def test_fork_wrong_user(self, db):
        db.register_session(session_id="s1", user_id=1)
        assert db.fork_session("s1", "s2", message_index=-1, user_id=999) is None

    def test_get_fork_lineage(self, db):
        db.register_session(session_id="s1", user_id=1)
        db.fork_session("s1", "s2", message_index=-1, user_id=1)
        db.fork_session("s2", "s3", message_index=-1, user_id=1)
        lineage = db.get_fork_lineage("s3")
        assert lineage == ["s1", "s2"]

    def test_get_fork_lineage_no_fork(self, db):
        db.register_session(session_id="s1", user_id=1)
        assert db.get_fork_lineage("s1") == []

    def test_fork_all_messages(self, db):
        db.register_session(session_id="s1", user_id=1)
        db.record_prompt("s1", [{"role": "user", "content": "a"}],
                         {"content": "b", "usage": {}})
        # Copy all messages with a large index
        forked = db.fork_session("s1", "s2", message_index=999, user_id=1)
        messages = db.get_messages("s2")
        assert len(messages) == 2  # all messages copied


class TestQuotasAndCleanup:
    def test_check_session_quota_under_limit(self, db):
        db.configure_quotas(max_concurrent_per_user=3)
        db.register_session(session_id="s1", user_id=1)
        assert db.check_session_quota(1) is None

    def test_check_session_quota_exceeded(self, db):
        db.configure_quotas(max_concurrent_per_user=1)
        db.register_session(session_id="s1", user_id=1)
        error = db.check_session_quota(1)
        assert error is not None
        assert error["code"] == "quota_exceeded"
        assert error["current"] == 1
        assert error["limit"] == 1

    def test_check_session_quota_ignores_closed(self, db):
        db.configure_quotas(max_concurrent_per_user=1)
        db.register_session(session_id="s1", user_id=1)
        db.close_session("s1")
        # Closed session shouldn't count
        assert db.check_session_quota(1) is None

    def test_check_token_quota_under_limit(self, db):
        db.configure_quotas(max_tokens_per_session=1000)
        db.register_session(session_id="s1", user_id=1)
        assert db.check_token_quota("s1") is None

    def test_check_token_quota_exceeded(self, db):
        db.configure_quotas(max_tokens_per_session=100)
        db.register_session(session_id="s1", user_id=1)
        db.update_token_usage("s1", prompt_tokens=80, completion_tokens=30)
        error = db.check_token_quota("s1")
        assert error is not None
        assert error["code"] == "token_quota_exceeded"

    def test_check_token_quota_nonexistent_session(self, db):
        db.configure_quotas(max_tokens_per_session=100)
        assert db.check_token_quota("nope") is None

    def test_get_quota_status(self, db):
        db.configure_quotas(max_concurrent_per_user=5, max_tokens_per_session=1000)
        db.register_session(session_id="s1", user_id=1)
        status = db.get_quota_status(1, session_id="s1")
        assert status["concurrent_sessions"]["current"] == 1
        assert status["concurrent_sessions"]["limit"] == 5
        assert "session_tokens" in status

    def test_get_quota_status_without_session(self, db):
        db.configure_quotas(max_concurrent_per_user=5)
        db.register_session(session_id="s1", user_id=1)
        status = db.get_quota_status(1)
        assert status["concurrent_sessions"]["current"] == 1
        assert "session_tokens" not in status

    def test_evict_expired_sessions(self, db):
        db.configure_quotas(session_ttl_seconds=0)  # Immediate expiry
        db.register_session(session_id="s1", user_id=1)
        evicted = db.evict_expired_sessions()
        assert evicted == 1
        rec = db.get_session("s1")
        assert rec["status"] == "closed"

    def test_evict_skips_already_closed(self, db):
        db.configure_quotas(session_ttl_seconds=0)
        db.register_session(session_id="s1", user_id=1)
        db.close_session("s1")
        evicted = db.evict_expired_sessions()
        assert evicted == 0

    def test_evict_preserves_fresh_sessions(self, db):
        db.configure_quotas(session_ttl_seconds=86400)  # 24h
        db.register_session(session_id="s1", user_id=1)
        evicted = db.evict_expired_sessions()
        assert evicted == 0
        rec = db.get_session("s1")
        assert rec["status"] == "active"

    def test_purge_retained_sessions_deletes_old_closed_sessions_and_messages(self, db):
        db.register_session(session_id="old-closed", user_id=1)
        db.record_prompt(
            "old-closed",
            [{"role": "user", "content": "Retained prompt"}],
            {"content": "Retained response", "usage": {}},
        )
        db.close_session("old-closed")
        db.register_session(session_id="old-active", user_id=1)
        db.register_session(session_id="fresh-closed", user_id=1)
        db.close_session("fresh-closed")

        conn = db._get_conn()
        conn.execute(
            "UPDATE sessions SET created_at = ?, last_activity_at = ? WHERE session_id IN (?, ?)",
            (
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
                "old-closed",
                "old-active",
            ),
        )
        conn.execute(
            "UPDATE sessions SET created_at = ?, last_activity_at = ? WHERE session_id = ?",
            ("2026-01-12T00:00:00+00:00", "2026-01-12T00:00:00+00:00", "fresh-closed"),
        )
        conn.commit()

        deleted = db.purge_retained_sessions(
            retention_days=1,
            now=datetime(2026, 1, 12, tzinfo=timezone.utc),
        )

        assert deleted == 1
        assert db.get_session("old-closed") is None
        assert db.get_messages("old-closed") == []
        assert db.get_session("old-active") is not None
        assert db.get_session("fresh-closed") is not None


class TestCascadeDelete:
    def test_delete_session_cascades_messages(self, db):
        db.register_session(session_id="s1", user_id=1)
        db.record_prompt("s1", [{"role": "user", "content": "Hello"}],
                         {"content": "Hi", "usage": {}})
        assert len(db.get_messages("s1")) == 2
        db.delete_session("s1")
        # Messages should be gone too (CASCADE)
        assert db.get_messages("s1") == []


class TestAgentRegistry:
    def test_save_and_get_agent_entry(self, db):
        entry = db.save_agent_entry({
            "agent_type": "test_agent",
            "name": "Test Agent",
            "command": "test-cmd",
            "source": "api",
        })
        assert entry is not None
        assert entry["agent_type"] == "test_agent"
        fetched = db.get_agent_entry("test_agent")
        assert fetched is not None
        assert fetched["name"] == "Test Agent"

    def test_delete_agent_entry(self, db):
        db.save_agent_entry({"agent_type": "tmp", "name": "Tmp", "source": "api"})
        assert db.delete_agent_entry("tmp") is True
        assert db.get_agent_entry("tmp") is None
        assert db.delete_agent_entry("tmp") is False

    def test_list_agent_entries(self, db):
        db.save_agent_entry({"agent_type": "a1", "name": "A1", "source": "api"})
        db.save_agent_entry({"agent_type": "a2", "name": "A2", "source": "yaml"})
        all_entries = db.list_agent_entries()
        assert len(all_entries) == 2
        api_entries = db.list_agent_entries(source="api")
        assert len(api_entries) == 1
        assert api_entries[0]["agent_type"] == "a1"

    def test_save_agent_upsert(self, db):
        db.save_agent_entry({"agent_type": "a1", "name": "Original", "source": "api"})
        db.save_agent_entry({"agent_type": "a1", "name": "Updated", "source": "api"})
        entry = db.get_agent_entry("a1")
        assert entry["name"] == "Updated"

    def test_agent_entrypoint_strategy_fields_round_trip(self, db):
        saved = db.save_agent_entry({
            "agent_type": "adapter",
            "name": "Adapter",
            "entrypoint_strategy": "external_acp_adapter",
            "acp_command": "adapter-acp",
            "acp_args": '["--stdio"]',
            "adapter_source": "example/adapter",
            "adapter_docs_url": "https://example.test/adapter",
            "certification_blocker": "adapter_missing",
            "source": "api",
        })

        assert saved["entrypoint_strategy"] == "external_acp_adapter"
        assert saved["acp_command"] == "adapter-acp"
        assert saved["acp_args"] == '["--stdio"]'
        assert saved["adapter_source"] == "example/adapter"
        assert saved["certification_blocker"] == "adapter_missing"

    def test_agent_registry_adapter_metadata_round_trips(self, db):
        saved = db.save_agent_entry({
            "agent_type": "codex",
            "name": "Codex",
            "command": "codex",
            "entrypoint_strategy": "external_acp_adapter",
            "acp_command": "codex-acp",
            "adapter_source": "zed-industries/codex-acp",
            "adapter_package": "@zed-industries/codex-acp",
            "adapter_version": "0.15.0",
            "adapter_version_policy": "exact_pin_required",
            "adapter_install_source": "github_release_preferred",
            "credential_policy": "delegated_to_adapter",
            "runtime_backend": "acp_downstream",
            "source": "api",
        })

        assert saved["entrypoint_strategy"] == "external_acp_adapter"
        assert saved["adapter_source"] == "zed-industries/codex-acp"
        assert saved["adapter_package"] == "@zed-industries/codex-acp"
        assert saved["adapter_version"] == "0.15.0"
        assert saved["adapter_version_policy"] == "exact_pin_required"
        assert saved["adapter_install_source"] == "github_release_preferred"
        assert saved["credential_policy"] == "delegated_to_adapter"
        assert saved["runtime_backend"] == "acp_downstream"

        listed = db.list_agent_entries(source="api")
        codex = next(entry for entry in listed if entry["agent_type"] == "codex")
        assert codex["entrypoint_strategy"] == "external_acp_adapter"
        assert codex["adapter_version"] == "0.15.0"
        assert codex["credential_policy"] == "delegated_to_adapter"
        assert codex["runtime_backend"] == "acp_downstream"


def test_legacy_agent_registry_rows_get_entrypoint_defaults(tmp_path):
    db_path = tmp_path / "legacy_acp_sessions.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE agent_registry (
            agent_type TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            command TEXT NOT NULL DEFAULT '',
            args TEXT NOT NULL DEFAULT '[]',
            env TEXT NOT NULL DEFAULT '{}',
            requires_api_key TEXT,
            is_default INTEGER NOT NULL DEFAULT 0,
            install_instructions TEXT NOT NULL DEFAULT '[]',
            docs_url TEXT,
            mcp_orchestration TEXT NOT NULL DEFAULT 'agent_driven',
            mcp_entry_tool TEXT NOT NULL DEFAULT 'execute',
            mcp_structured_response INTEGER NOT NULL DEFAULT 0,
            mcp_llm_provider TEXT,
            mcp_llm_model TEXT,
            mcp_max_iterations INTEGER NOT NULL DEFAULT 20,
            mcp_refresh_tools INTEGER NOT NULL DEFAULT 0,
            source TEXT NOT NULL DEFAULT 'api',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        INSERT INTO agent_registry (
            agent_type, name, command, source, created_at, updated_at
        ) VALUES ('legacy_api', 'Legacy API', 'legacy-cli', 'api', '2026-01-01', '2026-01-01');
        PRAGMA user_version=13;
        """
    )
    conn.commit()
    conn.close()

    db = ACPSessionsDB(db_path=str(db_path))
    try:
        row = db.get_agent_entry("legacy_api")
        assert row["entrypoint_strategy"] == "documented_candidate"
        assert row["acp_command"] == ""
        assert row["acp_args"] == "[]"

        registry = AgentRegistry(yaml_path="/missing.yaml", db=db)
        registry._load_api_entries()
        entry = registry.get_entry("legacy_api")
        assert entry is not None
        assert entry.entrypoint_strategy == "documented_candidate"
        assert entry.acp_command == ""
        assert entry.acp_args == []
    finally:
        db.close()


def test_legacy_agent_registry_rows_get_adapter_metadata_defaults(tmp_path):
    db_path = tmp_path / "legacy_acp_sessions.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE agent_registry (
            agent_type TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            command TEXT NOT NULL DEFAULT '',
            args TEXT NOT NULL DEFAULT '[]',
            env TEXT NOT NULL DEFAULT '{}',
            requires_api_key TEXT,
            is_default INTEGER NOT NULL DEFAULT 0,
            install_instructions TEXT NOT NULL DEFAULT '[]',
            docs_url TEXT,
            mcp_orchestration TEXT NOT NULL DEFAULT 'agent_driven',
            mcp_entry_tool TEXT NOT NULL DEFAULT 'execute',
            mcp_structured_response INTEGER NOT NULL DEFAULT 0,
            mcp_llm_provider TEXT,
            mcp_llm_model TEXT,
            mcp_max_iterations INTEGER NOT NULL DEFAULT 20,
            mcp_refresh_tools INTEGER NOT NULL DEFAULT 0,
            entrypoint_strategy TEXT NOT NULL DEFAULT 'documented_candidate',
            acp_command TEXT NOT NULL DEFAULT '',
            acp_args TEXT NOT NULL DEFAULT '[]',
            adapter_source TEXT,
            adapter_docs_url TEXT,
            certification_blocker TEXT,
            source TEXT NOT NULL DEFAULT 'api',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        INSERT INTO agent_registry (
            agent_type, name, command, source, created_at, updated_at
        ) VALUES ('legacy_api', 'Legacy API', 'legacy-cli', 'api', '2026-01-01', '2026-01-01');
        PRAGMA user_version=14;
        """
    )
    conn.commit()
    conn.close()

    db = ACPSessionsDB(db_path=str(db_path))
    try:
        columns = {
            str(row[1])
            for row in db._get_conn().execute('PRAGMA table_info("agent_registry")').fetchall()
        }
        assert {
            "adapter_package",
            "adapter_version",
            "adapter_version_policy",
            "adapter_install_source",
            "credential_policy",
            "runtime_backend",
        }.issubset(columns)

        row = db.get_agent_entry("legacy_api")
        assert row is not None
        assert row["adapter_package"] is None
        assert row["adapter_version"] is None
        assert row["adapter_version_policy"] == "unknown"
        assert row["adapter_install_source"] == "unknown"
        assert row["credential_policy"] == "unknown"
        assert row["runtime_backend"] == "acp_downstream"

        listed = db.list_agent_entries(source="api")
        assert listed[0]["agent_type"] == "legacy_api"
        assert listed[0]["adapter_version_policy"] == "unknown"
        assert listed[0]["runtime_backend"] == "acp_downstream"
    finally:
        db.close()


class TestHealthHistory:
    def test_record_and_get_health_check(self, db):
        db.record_health_check("claude_code", "healthy", 0, '{"status": "available"}')
        history = db.get_health_history("claude_code")
        assert len(history) == 1
        assert history[0]["health"] == "healthy"
        assert history[0]["agent_type"] == "claude_code"
        assert history[0]["details"] == '{"status": "available"}'
        assert history[0]["checked_at"] is not None

    def test_health_history_limit(self, db):
        for i in range(10):
            db.record_health_check("claude_code", "healthy", 0)
        history = db.get_health_history("claude_code", limit=3)
        assert len(history) == 3

    def test_health_history_empty(self, db):
        assert db.get_health_history("nonexistent") == []

    def test_health_history_ordered_by_checked_at_desc(self, db):
        db.record_health_check("agent_a", "healthy", 0)
        db.record_health_check("agent_a", "degraded", 1)
        db.record_health_check("agent_a", "unavailable", 3)
        history = db.get_health_history("agent_a")
        assert len(history) == 3
        # Most recent first
        assert history[0]["health"] == "unavailable"
        assert history[2]["health"] == "healthy"

    def test_health_history_filters_by_agent_type(self, db):
        db.record_health_check("agent_a", "healthy", 0)
        db.record_health_check("agent_b", "degraded", 1)
        history_a = db.get_health_history("agent_a")
        assert len(history_a) == 1
        assert history_a[0]["agent_type"] == "agent_a"

    def test_health_history_consecutive_failures_stored(self, db):
        db.record_health_check("agent_a", "degraded", 2, '{"error": "timeout"}')
        history = db.get_health_history("agent_a")
        assert history[0]["consecutive_failures"] == 2
        assert history[0]["details"] == '{"error": "timeout"}'
