from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ACP_Audit_DB import ACPAuditDB
from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB
from tldw_Server_API.app.core.DB_Management.Circuit_Breaker_Registry_DB import (
    CircuitBreakerRegistryDB,
)
from tldw_Server_API.app.core.DB_Management.ChatWorkflows_DB import (
    ChatWorkflowsDatabase,
)
from tldw_Server_API.app.core.DB_Management.Guardian_DB import GuardianDB
from tldw_Server_API.app.core.DB_Management.Kanban_DB import KanbanDB
from tldw_Server_API.app.core.DB_Management.Meetings_DB import MeetingsDatabase
from tldw_Server_API.app.core.DB_Management.Orchestration_DB import OrchestrationDB
from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
from tldw_Server_API.app.core.DB_Management.Prompts_DB import PromptsDatabase
from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
from tldw_Server_API.app.core.DB_Management.TopicMonitoring_DB import (
    TopicMonitoringDB,
    WatchlistRuleRecord,
)
from tldw_Server_API.app.core.DB_Management.Voice_Registry_DB import VoiceRegistryDB
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Evaluations.connection_pool import ConnectionPool
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Sandbox.store import SQLiteStore


def _sqlite_pragma_map(conn: sqlite3.Connection) -> dict[str, int | str]:
    return {
        "journal_mode": str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower(),
        "synchronous": int(conn.execute("PRAGMA synchronous").fetchone()[0]),
        "foreign_keys": int(conn.execute("PRAGMA foreign_keys").fetchone()[0]),
        "busy_timeout": int(conn.execute("PRAGMA busy_timeout").fetchone()[0]),
        "temp_store": int(conn.execute("PRAGMA temp_store").fetchone()[0]),
    }


def _sqlite_int_pragma(conn: sqlite3.Connection, pragma: str) -> int:
    return int(conn.execute(f"PRAGMA {pragma}").fetchone()[0])


class _FakeCursor:
    def __init__(self, *, rows: list[dict] | None = None, rowcount: int = 0) -> None:
        self._rows = rows or []
        self.rowcount = rowcount

    def fetchall(self):
        return self._rows


class _RecordingConnection:
    def __init__(self) -> None:
        self.row_factory = None
        self.in_transaction = False
        self.statements: list[str] = []
        self.closed = False

    def execute(self, sql: str, params=None):
        self.statements.append(sql)
        normalized = sql.strip().upper()
        if normalized.startswith("BEGIN"):
            self.in_transaction = True
        if normalized.startswith("SELECT ID FROM GUARDIAN_RELATIONSHIPS"):
            return _FakeCursor(rows=[])
        return _FakeCursor()

    def commit(self) -> None:
        self.statements.append("COMMIT()")

    def rollback(self) -> None:
        self.statements.append("ROLLBACK()")

    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


@pytest.mark.unit
def test_voice_registry_runtime_connection_uses_standard_sqlite_pragmas(tmp_path, monkeypatch):
    monkeypatch.setattr(
        DatabasePaths,
        "get_user_db_base_dir",
        lambda *args, **kwargs: tmp_path,
        raising=True,
    )
    db = VoiceRegistryDB(tmp_path / "voice_registry.db")
    conn = db._connect()
    try:
        pragmas = _sqlite_pragma_map(conn)
    finally:
        conn.close()

    assert pragmas == {
        "journal_mode": "wal",
        "synchronous": 1,
        "foreign_keys": 1,
        "busy_timeout": 5000,
        "temp_store": 2,
    }


@pytest.mark.unit
def test_topic_monitoring_runtime_connection_uses_standard_sqlite_pragmas(tmp_path):
    db = TopicMonitoringDB(str(tmp_path / "monitoring" / "alerts.db"))
    conn = db._connect()
    try:
        pragmas = _sqlite_pragma_map(conn)
    finally:
        conn.close()

    assert pragmas == {
        "journal_mode": "wal",
        "synchronous": 1,
        "foreign_keys": 1,
        "busy_timeout": 5000,
        "temp_store": 2,
    }


@pytest.mark.unit
def test_guardian_runtime_connection_uses_standard_sqlite_pragmas(tmp_path):
    db = GuardianDB(str(tmp_path / "guardian.db"))
    conn = db._connect()
    try:
        pragmas = _sqlite_pragma_map(conn)
    finally:
        conn.close()

    assert pragmas == {
        "journal_mode": "wal",
        "synchronous": 1,
        "foreign_keys": 1,
        "busy_timeout": 5000,
        "temp_store": 2,
    }


@pytest.mark.unit
def test_guardian_db_requires_existing_parent_directory(tmp_path):
    missing_parent_path = tmp_path / "missing" / "guardian.db"

    with pytest.raises(ValueError, match="parent directory must already exist"):
        GuardianDB(str(missing_parent_path))


@pytest.mark.unit
def test_personalization_runtime_connection_uses_standard_sqlite_pragmas(tmp_path):
    db = PersonalizationDB(str(tmp_path / "personalization.db"))
    conn = db._connect()
    try:
        pragmas = _sqlite_pragma_map(conn)
    finally:
        conn.close()

    assert pragmas == {
        "journal_mode": "wal",
        "synchronous": 1,
        "foreign_keys": 1,
        "busy_timeout": 5000,
        "temp_store": 2,
    }


@pytest.mark.unit
def test_research_sessions_runtime_connection_uses_standard_sqlite_pragmas(tmp_path):
    db = ResearchSessionsDB(tmp_path / "research.db")
    conn = db._connect()
    try:
        pragmas = _sqlite_pragma_map(conn)
    finally:
        conn.close()

    assert pragmas == {
        "journal_mode": "wal",
        "synchronous": 1,
        "foreign_keys": 1,
        "busy_timeout": 5000,
        "temp_store": 2,
    }


@pytest.mark.unit
def test_meetings_runtime_connection_uses_standard_sqlite_pragmas(tmp_path):
    db = MeetingsDatabase(tmp_path / "meetings.db", client_id="test", user_id="1")
    conn = db.get_connection()
    try:
        pragmas = _sqlite_pragma_map(conn)
    finally:
        db.close_connection()

    assert pragmas == {
        "journal_mode": "wal",
        "synchronous": 1,
        "foreign_keys": 1,
        "busy_timeout": 5000,
        "temp_store": 2,
    }


@pytest.mark.unit
def test_orchestration_runtime_connection_uses_standard_sqlite_pragmas(tmp_path):
    db = OrchestrationDB(user_id=1, db_dir=str(tmp_path))
    conn = db._get_conn()
    try:
        pragmas = _sqlite_pragma_map(conn)
    finally:
        db.close()

    assert pragmas == {
        "journal_mode": "wal",
        "synchronous": 1,
        "foreign_keys": 1,
        "busy_timeout": 5000,
        "temp_store": 2,
    }


@pytest.mark.unit
def test_circuit_breaker_runtime_connection_uses_standard_sqlite_pragmas(tmp_path):
    db = CircuitBreakerRegistryDB(tmp_path / "circuit_breaker.db")
    conn = db._connect()
    try:
        pragmas = _sqlite_pragma_map(conn)
    finally:
        conn.close()

    assert pragmas == {
        "journal_mode": "wal",
        "synchronous": 1,
        "foreign_keys": 1,
        "busy_timeout": 5000,
        "temp_store": 2,
    }


@pytest.mark.unit
def test_workflows_runtime_connection_uses_standard_sqlite_pragmas(tmp_path):
    db = WorkflowsDatabase(str(tmp_path / "workflows.db"))
    try:
        pragmas = _sqlite_pragma_map(db._conn)
    finally:
        db.close()

    assert pragmas == {
        "journal_mode": "wal",
        "synchronous": 1,
        "foreign_keys": 1,
        "busy_timeout": 5000,
        "temp_store": 2,
    }


@pytest.mark.unit
def test_prompts_runtime_connection_uses_standard_sqlite_pragmas(tmp_path):
    db = PromptsDatabase(str(tmp_path / "prompts.db"), client_id="test-client")
    conn = db.get_connection()
    try:
        pragmas = _sqlite_pragma_map(conn)
    finally:
        db.close()

    assert pragmas == {
        "journal_mode": "wal",
        "synchronous": 1,
        "foreign_keys": 1,
        "busy_timeout": 1000,
        "temp_store": 2,
    }


@pytest.mark.unit
def test_jobs_manager_runtime_connection_uses_standard_sqlite_pragmas(tmp_path):
    manager = JobManager(tmp_path / "jobs.db")
    conn = manager._connect()
    try:
        pragmas = _sqlite_pragma_map(conn)
    finally:
        conn.close()

    assert pragmas == {
        "journal_mode": "wal",
        "synchronous": 1,
        "foreign_keys": 1,
        "busy_timeout": 5000,
        "temp_store": 2,
    }


@pytest.mark.unit
def test_sqlite_store_runtime_connection_uses_standard_sqlite_pragmas(tmp_path):
    store = SQLiteStore(db_path=str(tmp_path / "sandbox_store.db"))
    conn = store._conn()
    try:
        pragmas = _sqlite_pragma_map(conn)
    finally:
        conn.close()

    assert pragmas == {
        "journal_mode": "wal",
        "synchronous": 1,
        "foreign_keys": 1,
        "busy_timeout": 5000,
        "temp_store": 2,
    }


@pytest.mark.unit
def test_kanban_runtime_connection_uses_standard_sqlite_pragmas_and_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    db = KanbanDB(db_path=str(tmp_path / "user_dbs" / "1" / "kanban.db"), user_id="1")
    conn = db._connect()
    try:
        pragmas = _sqlite_pragma_map(conn)
        cache_size = _sqlite_int_pragma(conn, "cache_size")
    finally:
        conn.close()
        db.close()

    assert pragmas == {
        "journal_mode": "wal",
        "synchronous": 1,
        "foreign_keys": 1,
        "busy_timeout": 30000,
        "temp_store": 2,
    }
    assert cache_size == -64000


@pytest.mark.unit
def test_acp_sessions_runtime_connection_uses_standard_sqlite_pragmas(tmp_path):
    db = ACPSessionsDB(db_path=str(tmp_path / "acp_sessions.db"))
    try:
        conn = db._get_conn()
        pragmas = _sqlite_pragma_map(conn)
    finally:
        db.close()

    assert pragmas == {
        "journal_mode": "wal",
        "synchronous": 1,
        "foreign_keys": 1,
        "busy_timeout": 5000,
        "temp_store": 2,
    }


@pytest.mark.unit
def test_acp_audit_runtime_connection_uses_standard_sqlite_pragmas(tmp_path):
    db = ACPAuditDB(db_path=str(tmp_path / "acp_audit.db"))
    try:
        conn = db._get_conn()
        pragmas = _sqlite_pragma_map(conn)
    finally:
        db.close()

    assert pragmas == {
        "journal_mode": "wal",
        "synchronous": 1,
        "foreign_keys": 1,
        "busy_timeout": 5000,
        "temp_store": 2,
    }


@pytest.mark.unit
def test_chat_workflows_runtime_connection_uses_standard_sqlite_pragmas(tmp_path):
    db = ChatWorkflowsDatabase(db_path=tmp_path / "chat_workflows.db", client_id="test")
    try:
        pragmas = _sqlite_pragma_map(db._conn)
    finally:
        db.close()

    assert pragmas == {
        "journal_mode": "wal",
        "synchronous": 1,
        "foreign_keys": 1,
        "busy_timeout": 5000,
        "temp_store": 2,
    }


@pytest.mark.unit
def test_evaluations_connection_pool_runtime_connection_uses_standard_sqlite_pragmas(tmp_path):
    pool = ConnectionPool(
        db_path=str(tmp_path / "evaluations.db"),
        pool_size=1,
        max_overflow=0,
        enable_monitoring=False,
    )
    try:
        conn = pool._pool[0].connection
        pragmas = _sqlite_pragma_map(conn)
        mmap_size = _sqlite_int_pragma(conn, "mmap_size")
    finally:
        pool.shutdown()

    assert pragmas == {
        "journal_mode": "wal",
        "synchronous": 1,
        "foreign_keys": 1,
        "busy_timeout": 30000,
        "temp_store": 2,
    }
    assert mmap_size == 268435456


@pytest.mark.unit
def test_voice_registry_replace_user_voices_uses_begin_immediate(tmp_path, monkeypatch):
    monkeypatch.setattr(
        DatabasePaths,
        "get_user_db_base_dir",
        lambda *args, **kwargs: tmp_path,
        raising=True,
    )
    db = VoiceRegistryDB(tmp_path / "voice_registry.db")
    conn = _RecordingConnection()
    db._connect = lambda: conn

    db.replace_user_voices(1, [])

    assert conn.statements[0] == "BEGIN IMMEDIATE"


@pytest.mark.unit
def test_topic_monitoring_replace_watchlist_rules_uses_begin_immediate(tmp_path):
    db = TopicMonitoringDB(str(tmp_path / "monitoring" / "alerts.db"))
    conn = _RecordingConnection()
    db._connect = lambda: conn

    db.replace_watchlist_rules("wl-1", [])

    assert conn.statements[0] == "BEGIN IMMEDIATE"


@pytest.mark.unit
def test_topic_monitoring_schema_migration_uses_begin_immediate(tmp_path):
    db = TopicMonitoringDB(str(tmp_path / "monitoring" / "alerts.db"))
    conn = _RecordingConnection()

    db._migrate_watchlist_rules_schema(conn)

    assert conn.statements[0] == "BEGIN IMMEDIATE"


@pytest.mark.unit
def test_guardian_delete_all_for_user_uses_begin_immediate(tmp_path):
    db = GuardianDB(str(tmp_path / "guardian.db"))
    conn = _RecordingConnection()
    db._connect = lambda: conn

    counts = db.delete_all_for_user("user-1")

    assert counts["relationships"] == 0
    assert conn.statements[0] == "BEGIN IMMEDIATE"


@pytest.mark.unit
def test_prompts_database_transaction_uses_begin_immediate():
    db = PromptsDatabase.__new__(PromptsDatabase)
    conn = _RecordingConnection()
    db.get_connection = lambda: conn

    with db.transaction():
        pass

    assert conn.statements[0] == "BEGIN IMMEDIATE"


@pytest.mark.unit
def test_meetings_database_transaction_uses_begin_immediate():
    db = MeetingsDatabase.__new__(MeetingsDatabase)
    conn = _RecordingConnection()
    db.get_connection = lambda: conn

    with db.transaction():
        pass

    assert conn.statements[0] == "BEGIN IMMEDIATE"
