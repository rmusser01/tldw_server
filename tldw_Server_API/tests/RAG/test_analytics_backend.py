from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3
import threading
from typing import Any, List, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.RAG.rag_service.analytics_db import AnalyticsDatabase
from tldw_Server_API.app.core.RAG.rag_service.analytics_system import UnifiedFeedbackSystem, UserFeedbackStore


class _StubConnection:
    def __init__(self) -> None:
        self.statements: List[Tuple[str, Any]] = []

    def execute(self, statement: str, params: Any = None) -> None:
        self.statements.append((statement, params))


class _StubChaChaDb:
    def __init__(self, backend_type: BackendType) -> None:
        self.backend_type = backend_type
        self._conn = _StubConnection()

    @contextmanager
    def transaction(self):  # type: ignore[override]
        yield self._conn

    @property
    def executed(self) -> List[Tuple[str, Any]]:
        return self._conn.statements


class _SQLiteBackfillDb:
    def __init__(self, db_path: Path) -> None:
        self.backend_type = BackendType.SQLITE
        self._conn = sqlite3.connect(str(db_path))

    @contextmanager
    def transaction(self):  # type: ignore[override]
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise


def test_user_feedback_schema_sqlite_executes_expected_statements() -> None:

    db = _StubChaChaDb(BackendType.SQLITE)
    UserFeedbackStore(db)
    statements = [stmt for stmt, _ in db.executed]
    assert any("CREATE TABLE IF NOT EXISTS conversation_feedback" in stmt for stmt in statements)
    assert any("idx_feedback_conv" in stmt for stmt in statements)
    assert any("idx_feedback_created" in stmt for stmt in statements)
    assert any("CREATE TABLE IF NOT EXISTS feedback_idempotency" in stmt for stmt in statements)
    assert any("idx_feedback_idempotency_created" in stmt for stmt in statements)


def test_user_feedback_schema_sqlite_includes_required_fields() -> None:

    db = _StubChaChaDb(BackendType.SQLITE)
    UserFeedbackStore(db)
    statements = [stmt for stmt, _ in db.executed]
    assert any("helpful INTEGER" in stmt for stmt in statements)
    assert any("issues" in stmt and "conversation_feedback" in stmt for stmt in statements)
    assert any("pending INTEGER" in stmt and "feedback_idempotency" in stmt for stmt in statements)


def test_user_feedback_schema_postgres_uses_boolean_and_timestamp() -> None:

    db = _StubChaChaDb(BackendType.POSTGRESQL)
    UserFeedbackStore(db)
    statements = [stmt for stmt, _ in db.executed]
    assert any("helpful BOOLEAN" in stmt for stmt in statements)
    assert any("issues" in stmt and "conversation_feedback" in stmt for stmt in statements)
    assert any("pending BOOLEAN" in stmt and "feedback_idempotency" in stmt for stmt in statements)
    assert any("TIMESTAMPTZ" in stmt for stmt in statements)


def test_user_feedback_schema_backfills_missing_issues_column_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "feedback-backfill.sqlite"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE conversations (id TEXT PRIMARY KEY)")
        conn.execute(
            """
            CREATE TABLE conversation_feedback (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                message_id TEXT,
                query TEXT,
                document_ids TEXT,
                chunk_ids TEXT,
                relevance_score INTEGER,
                helpful INTEGER,
                user_notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    db = _SQLiteBackfillDb(db_path)
    try:
        UserFeedbackStore(db)

        cursor = db._conn.execute("PRAGMA table_info(conversation_feedback)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "issues" in columns
    finally:
        db._conn.close()


@pytest.mark.asyncio
async def test_feedback_idempotency_claim_finalize_roundtrip_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "feedback-idempotency.sqlite"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE conversations (id TEXT PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()

    db = _SQLiteBackfillDb(db_path)
    try:
        store = UserFeedbackStore(db)
        reserved, record = await store.claim_idempotency(
            "idem:test-key",
            ["missing_details"],
            "Initial note",
            ttl_seconds=300,
        )
        assert reserved is True
        assert record["feedback_id"] is None
        assert record["pending"] is True

        reserved_again, record_again = await store.claim_idempotency(
            "idem:test-key",
            [],
            None,
            ttl_seconds=300,
        )
        assert reserved_again is False
        assert record_again["pending"] is True

        await store.update_idempotency(
            "idem:test-key",
            ["missing_details", "incorrect_information"],
            "Merged note",
        )
        issues, notes, has_pending_merge = await store.finalize_idempotency(
            "idem:test-key",
            "fb_123",
            ["missing_details"],
            "Initial note",
        )
        assert set(issues) == {"missing_details", "incorrect_information"}
        assert notes == "Merged note"
        assert has_pending_merge is True

        reserved_final, record_final = await store.claim_idempotency(
            "idem:test-key",
            [],
            None,
            ttl_seconds=300,
        )
        assert reserved_final is False
        assert record_final["feedback_id"] == "fb_123"
        assert record_final["pending"] is False
    finally:
        db._conn.close()


def test_record_search_sqlite_writes_row(tmp_path: Path) -> None:
    db_path = tmp_path / "analytics.sqlite"
    analytics = AnalyticsDatabase(str(db_path))
    try:
        analytics.record_search(
            {
                "query": "backend coverage",
                "results_count": 3,
                "response_time_ms": 42,
                "cache_hit": False,
            }
        )
        conn = analytics.backend.connect()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT query_hash, results_count FROM search_analytics")
            row = cursor.fetchone()
            assert row is not None
            assert row[1] == 3
        finally:
            conn.close()
    finally:
        analytics.close()


def test_record_search_postgres_calls_backend_execute() -> None:

    analytics = AnalyticsDatabase.__new__(AnalyticsDatabase)
    analytics._local = threading.local()
    analytics._backend = MagicMock()
    analytics._backend.backend_type = BackendType.POSTGRESQL

    calls: List[Tuple[str, Any]] = []

    def transaction_factory():

        @contextmanager
        def _ctx():
            yield object()

        return _ctx()

    analytics.transaction = transaction_factory  # type: ignore[assignment]

    def record_execute(_conn: Any, query: str, params: Any = None):
        calls.append((query, params))
        return MagicMock()

    analytics._execute = MagicMock(side_effect=record_execute)  # type: ignore[attr-defined]

    analytics.record_search(
        {
            "query": "backend coverage",
            "results_count": 3,
            "response_time_ms": 42,
            "cache_hit": True,
        }
    )

    analytics._execute.assert_called_once()
    inserted_query = calls[0][0]
    assert "INSERT INTO search_analytics" in inserted_query
    assert "%s" in inserted_query


def test_record_event_sqlite_writes_row(tmp_path: Path) -> None:
    db_path = tmp_path / "analytics.sqlite"
    analytics = AnalyticsDatabase(str(db_path))
    try:
        analytics.record_event(
            {
                "event_type": "feedback",
                "query_hash": "abc123",
                "metrics": {"rank": 2, "dwell_ms": 3000},
            }
        )
        conn = analytics.backend.connect()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT event_type, query_hash, metrics FROM analytics_events")
            row = cursor.fetchone()
            assert row is not None
            assert row[0] == "feedback"
            assert row[1] == "abc123"
        finally:
            conn.close()
    finally:
        analytics.close()


@pytest.mark.asyncio
async def test_submit_feedback_maps_issues_to_analytics() -> None:
    system = UnifiedFeedbackSystem.__new__(UnifiedFeedbackSystem)
    system.enable_analytics = True
    system.analytics = MagicMock()
    system.user_feedback = None
    system.analytics.record_search_quality = AsyncMock(return_value=True)
    system.analytics.record_document_performance = AsyncMock(return_value=True)
    system.analytics.record_feedback = AsyncMock(return_value=True)
    system.analytics.record_event = AsyncMock(return_value=True)

    await system.submit_feedback(
        conversation_id="",
        query="reset auth",
        document_ids=[],
        chunk_ids=[],
        feedback_type="report",
        issues=["missing_details"],
        user_notes="Need the new flow",
        session_id="sess-1",
    )

    system.analytics.record_feedback.assert_awaited()
    payload = system.analytics.record_feedback.await_args.args[0]
    assert payload["feedback_type"] == "report"
    assert payload["categories"] == ["missing_details"]
    assert payload["improvement_areas"] == ["missing_details"]
