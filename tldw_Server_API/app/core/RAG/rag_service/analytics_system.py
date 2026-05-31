# analytics_system.py
"""
Analytics and feedback system for RAG with dual storage:
1. Databases/Analytics.db - Server-side QA and anonymized metrics
2. ChaChaNotes_DB - User-specific feedback linked to conversations

This module replaces the old feedback_system.py with proper separation
of concerns between analytics and user data.
"""

import asyncio
import hashlib
import json
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
)
from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)

try:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError
except (ImportError, AttributeError):
    CharactersRAGDBError = RuntimeError  # type: ignore[assignment]

from .analytics_db import DEFAULT_ANALYTICS_DB_PATH, get_analytics_db


class FeedbackType(Enum):
    """Types of feedback that can be collected."""
    RELEVANCE = "relevance"  # 1-5 star rating
    HELPFUL = "helpful"  # Yes/No
    CLICK = "click"  # User clicked on result
    DWELL_TIME = "dwell_time"  # Time spent on result
    COPY = "copy"  # User copied text from result
    REPORT = "report"  # User reported an issue
    CITATION_USED = "citation_used"  # Citation was used in answer


class AnalyticsEventType(Enum):
    """Types of analytics events."""
    SEARCH = "search"
    FEEDBACK = "feedback"
    GENERATION = "generation"
    CITATION = "citation"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    ERROR = "error"
    PERFORMANCE = "performance"


@dataclass
class AnalyticsEvent:
    """An analytics event for server-side QA."""
    event_type: AnalyticsEventType
    timestamp: datetime = field(default_factory=datetime.now)
    query_hash: Optional[str] = None  # Hashed for privacy
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "query_hash": self.query_hash,
            "metrics": json.dumps(self.metrics)
        }


@dataclass
class UserFeedback:
    """User feedback linked to conversation."""
    conversation_id: str
    message_id: Optional[str]
    query: str
    document_ids: list[str]
    chunk_ids: list[str]
    relevance_score: Optional[int]  # 1-5
    helpful: Optional[bool]
    issues: list[str]
    user_notes: Optional[str]
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
            "query": self.query,
            "document_ids": json.dumps(self.document_ids),
            "chunk_ids": json.dumps(self.chunk_ids),
            "relevance_score": self.relevance_score,
            "helpful": self.helpful,
            "issues": json.dumps(self.issues),
            "user_notes": self.user_notes,
            "created_at": self.created_at.isoformat()
        }


class AnalyticsStore:
    """
    Server-side analytics storage for QA and system improvement.
    No PII is stored - only anonymized metrics.
    Uses the new AnalyticsDatabase for storage.
    """

    def __init__(self, db_path: str = DEFAULT_ANALYTICS_DB_PATH):
        """
        Initialize analytics store.

        Args:
            db_path: Path to Analytics database
        """
        self.db = get_analytics_db(db_path)

    async def record_search(self, search_data: dict[str, Any]) -> bool:
        """
        Record search analytics.

        Args:
            search_data: Dictionary containing search metrics

        Returns:
            Success status
        """
        try:
            # Use sync method from AnalyticsDatabase
            await asyncio.get_event_loop().run_in_executor(
                None, self.db.record_search, search_data
            )
        except (BackendDatabaseError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            logger.error("Failed to record search analytics")
        else:
            return True
        return False

    async def record_feedback(self, feedback_data: dict[str, Any]) -> bool:
        """
        Record anonymized feedback analytics.

        Args:
            feedback_data: Dictionary containing feedback data

        Returns:
            Success status
        """
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self.db.record_feedback, feedback_data
            )
        except (BackendDatabaseError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            logger.error("Failed to record feedback")
        else:
            return True
        return False

    async def record_event(self, event: "AnalyticsEvent | dict[str, Any]") -> bool:
        """
        Record a generic analytics event.

        Args:
            event: AnalyticsEvent or dict payload

        Returns:
            Success status
        """
        try:
            payload = event.to_dict() if hasattr(event, "to_dict") else dict(event)
            await asyncio.get_event_loop().run_in_executor(
                None, self.db.record_event, payload
            )
        except (BackendDatabaseError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            logger.error("Failed to record analytics event")
        else:
            return True
        return False

    async def record_performance_metric(
        self,
        *,
        metric_type: str,
        value: float,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """
        Record a performance metric as an analytics event.

        Args:
            metric_type: Metric name (e.g., "search_latency")
            value: Metric value
            metadata: Optional additional fields

        Returns:
            Success status
        """
        metrics: dict[str, Any] = {"metric_type": metric_type, "value": value}
        if metadata:
            metrics.update(metadata)
        event = AnalyticsEvent(
            event_type=AnalyticsEventType.PERFORMANCE,
            metrics=metrics,
        )
        return await self.record_event(event)

    async def record_search_quality(self, *, query_hash: str, relevance_score: float, clicked: bool) -> bool:
        """
        Record search quality metrics as a feedback event.

        Args:
            query_hash: Hashed query value
            relevance_score: Normalized relevance score (0-1)
            clicked: Whether any chunks were clicked

        Returns:
            Success status
        """
        event = AnalyticsEvent(
            event_type=AnalyticsEventType.FEEDBACK,
            query_hash=query_hash,
            metrics={
                "quality_score": relevance_score,
                "clicked": clicked,
            },
        )
        return await self.record_event(event)

    async def record_document_performance(
        self,
        doc_data: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> bool:
        """
        Record document performance metrics.

        Args:
            doc_data: Dictionary containing document metrics
            kwargs: Alternate keyword payload when doc_data is omitted

        Returns:
            Success status
        """
        payload = doc_data or kwargs
        if not payload:
            return False
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self.db.record_document_performance, payload
            )
        except (BackendDatabaseError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            logger.error("Failed to record document performance")
        else:
            return True
        return False

    async def record_error(self, error_data: Optional[dict[str, Any]] = None, **kwargs: Any) -> bool:
        """
        Record error tracking information.

        Args:
            error_data: Dictionary containing error information
            kwargs: Alternate keyword payload when error_data is omitted

        Returns:
            Success status
        """
        payload = error_data or kwargs
        if not payload:
            return False
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self.db.record_error, payload
            )
        except (BackendDatabaseError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            logger.error("Failed to record error")
        else:
            return True
        return False

    async def record_feature_usage(self, feature_data: dict[str, Any]) -> bool:
        """
        Record feature usage statistics.

        Args:
            feature_data: Dictionary containing feature usage data

        Returns:
            Success status
        """
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self.db.record_feature_usage, feature_data
            )
        except (BackendDatabaseError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            logger.error("Failed to record feature usage")
        else:
            return True
        return False

    async def get_analytics_summary(self, days: int = 7) -> dict[str, Any]:
        """
        Get analytics summary for the specified period.

        Args:
            days: Number of days to include in summary

        Returns:
            Dictionary containing analytics summary
        """
        try:
            summary = await asyncio.get_event_loop().run_in_executor(
                None, self.db.get_analytics_summary, days
            )
        except (BackendDatabaseError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            logger.error("Failed to get analytics summary")
        else:
            return summary
        return {}

    async def cleanup_old_data(self, days_to_keep: int = 90) -> int:
        """
        Clean up old analytics data.

        Args:
            days_to_keep: Number of days of data to retain

        Returns:
            Number of records deleted
        """
        try:
            deleted = await asyncio.get_event_loop().run_in_executor(
                None, self.db.cleanup_old_data, days_to_keep
            )
        except (BackendDatabaseError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            logger.error("Failed to cleanup old data")
        else:
            return deleted
        return 0


class UserFeedbackStore:
    """
    Stores user-specific feedback in ChaChaNotes_DB.
    Links feedback to conversations for context.
    """

    def __init__(self, chacha_db):
        """
        Initialize user feedback store.

        Args:
            chacha_db: ChaChaNotes database instance
        """
        self.db = chacha_db
        self._init_schema()

    def _init_schema(self):
        """Ensure feedback tables exist in ChaChaNotes_DB."""
        statements_sqlite = (
            """
            CREATE TABLE IF NOT EXISTS conversation_feedback (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                message_id TEXT,
                query TEXT,
                document_ids TEXT,
                chunk_ids TEXT,
                relevance_score INTEGER CHECK(relevance_score BETWEEN 1 AND 5),
                helpful INTEGER,
                issues TEXT,
                user_notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_feedback_conv ON conversation_feedback(conversation_id)",
            "CREATE INDEX IF NOT EXISTS idx_feedback_created ON conversation_feedback(created_at)",
            """
            CREATE TABLE IF NOT EXISTS feedback_idempotency (
                dedupe_key TEXT PRIMARY KEY,
                feedback_id TEXT,
                issues TEXT NOT NULL DEFAULT '[]',
                user_notes TEXT,
                pending INTEGER NOT NULL DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_feedback_idempotency_created ON feedback_idempotency(created_at)",
        )

        statements_postgres = (
            """
            CREATE TABLE IF NOT EXISTS conversation_feedback (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                message_id TEXT,
                query TEXT,
                document_ids TEXT,
                chunk_ids TEXT,
                relevance_score INTEGER CHECK(relevance_score BETWEEN 1 AND 5),
                helpful BOOLEAN,
                issues TEXT,
                user_notes TEXT,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT fk_conversation
                    FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id)
                    ON DELETE CASCADE
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_feedback_conv ON conversation_feedback(conversation_id)",
            "CREATE INDEX IF NOT EXISTS idx_feedback_created ON conversation_feedback(created_at)",
            """
            CREATE TABLE IF NOT EXISTS feedback_idempotency (
                dedupe_key TEXT PRIMARY KEY,
                feedback_id TEXT,
                issues TEXT NOT NULL DEFAULT '[]',
                user_notes TEXT,
                pending BOOLEAN NOT NULL DEFAULT TRUE,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_feedback_idempotency_created ON feedback_idempotency(created_at)",
        )

        statements = statements_sqlite if self.db.backend_type == BackendType.SQLITE else statements_postgres

        try:
            with self.db.transaction() as conn:
                for statement in statements:
                    conn.execute(statement)
                self._ensure_issues_column(conn)
        except (BackendDatabaseError, CharactersRAGDBError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            logger.error("Failed to initialize feedback schema")

    def _ensure_issues_column(self, conn: Any) -> None:
        """Backfill `issues` on pre-existing schemas created before this field existed."""
        if self.db.backend_type == BackendType.POSTGRESQL:
            conn.execute(
                "ALTER TABLE conversation_feedback "
                "ADD COLUMN IF NOT EXISTS issues TEXT"
            )
            return

        try:
            conn.execute("ALTER TABLE conversation_feedback ADD COLUMN issues TEXT")
        except Exception as exc:  # noqa: BLE001 - backend adapters expose varied DB exceptions
            message = str(exc).lower()
            if "duplicate column name" in message or "already exists" in message:
                return
            raise

    async def add_feedback(
        self,
        conversation_id: str,
        query: str,
        document_ids: list[str],
        chunk_ids: list[str],
        relevance_score: Optional[int] = None,
        helpful: Optional[bool] = None,
        issues: Optional[list[str]] = None,
        user_notes: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> str:
        """
        Add feedback for a conversation.

        Returns:
            Feedback ID
        """
        feedback_id = f"fb_{int(time.time() * 1000)}_{hashlib.md5(query.encode(), usedforsecurity=False).hexdigest()[:8]}"

        helpful_value: Optional[bool]
        helpful_value = None if helpful is None else bool(helpful)

        insert_sql = """
            INSERT INTO conversation_feedback
                (id, conversation_id, message_id, query, document_ids, chunk_ids,
                 relevance_score, helpful, issues, user_notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        if self.db.backend_type == BackendType.SQLITE:
            insert_sql = insert_sql.replace('%s', '?')

        issues_payload = json.dumps(issues or [])

        try:
            with self.db.transaction() as conn:
                conn.execute(
                    insert_sql,
                    (
                        feedback_id,
                        conversation_id,
                        message_id,
                        query,
                        json.dumps(document_ids),
                        json.dumps(chunk_ids),
                        relevance_score,
                        helpful_value,
                        issues_payload,
                        user_notes,
                    ),
                )
        except (BackendDatabaseError, CharactersRAGDBError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            logger.error("Failed to add feedback")
            raise
        else:
            logger.info(f"Added feedback {feedback_id} for conversation {conversation_id}")
            return feedback_id

    def _idempotency_row_to_dict(self, row: Any, cursor: Any) -> dict[str, Any]:
        if isinstance(row, dict):
            return row
        if hasattr(row, "keys"):
            return dict(row)
        columns = [col[0] for col in getattr(cursor, "description", [])] if cursor else []
        return {columns[idx]: row[idx] for idx in range(min(len(columns), len(row)))}

    def _decode_issues(self, raw: Any) -> list[str]:
        if raw is None:
            return []
        if isinstance(raw, list):
            return [str(item) for item in raw if str(item).strip()]
        if isinstance(raw, str):
            try:
                decoded = json.loads(raw)
            except (json.JSONDecodeError, TypeError, ValueError):
                return []
            if isinstance(decoded, list):
                return [str(item) for item in decoded if str(item).strip()]
            return []
        return []

    @staticmethod
    def _merge_issue_lists(existing: list[str], incoming: list[str]) -> list[str]:
        merged: list[str] = []
        seen = set()
        for item in existing + incoming:
            text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            merged.append(text)
        return merged

    async def claim_idempotency(
        self,
        dedupe_key: str,
        issues: Optional[list[str]],
        user_notes: Optional[str],
        *,
        ttl_seconds: int = 300,
    ) -> tuple[bool, dict[str, Any]]:
        """Atomically claim an idempotency key, returning existing record on conflict."""
        normalized_issues = self._merge_issue_lists([], issues or [])
        issues_payload = json.dumps(normalized_issues)
        ttl = max(int(ttl_seconds), 1)

        if self.db.backend_type == BackendType.SQLITE:
            cleanup_sql = (
                "DELETE FROM feedback_idempotency "
                "WHERE datetime(created_at) < datetime('now', ?)"
            )
            cleanup_params: tuple[Any, ...] = (f"-{ttl} seconds",)
            insert_sql = (
                "INSERT OR IGNORE INTO feedback_idempotency "
                "(dedupe_key, feedback_id, issues, user_notes, pending, created_at, updated_at) "
                "VALUES (?, NULL, ?, ?, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
            select_sql = (
                "SELECT feedback_id, issues, user_notes, pending, created_at "
                "FROM feedback_idempotency WHERE dedupe_key = ?"
            )
        else:
            cleanup_sql = (
                "DELETE FROM feedback_idempotency "
                "WHERE created_at < (CURRENT_TIMESTAMP - (INTERVAL '1 second' * %s))"
            )
            cleanup_params = (ttl,)
            insert_sql = (
                "INSERT INTO feedback_idempotency "
                "(dedupe_key, feedback_id, issues, user_notes, pending, created_at, updated_at) "
                "VALUES (%s, NULL, %s, %s, TRUE, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) "
                "ON CONFLICT (dedupe_key) DO NOTHING"
            )
            select_sql = (
                "SELECT feedback_id, issues, user_notes, pending, created_at "
                "FROM feedback_idempotency WHERE dedupe_key = %s"
            )

        with self.db.transaction() as conn:
            conn.execute(cleanup_sql, cleanup_params)
            cursor = conn.execute(insert_sql, (dedupe_key, issues_payload, user_notes))
            inserted = bool(getattr(cursor, "rowcount", 0))

            row_cursor = conn.execute(select_sql, (dedupe_key,))
            row = row_cursor.fetchone()

        if row:
            record = self._idempotency_row_to_dict(row, row_cursor)
            return inserted, {
                "feedback_id": record.get("feedback_id"),
                "issues": self._decode_issues(record.get("issues")),
                "user_notes": record.get("user_notes"),
                "pending": bool(record.get("pending")),
            }

        # Fallback for very unusual adapter behavior where SELECT returns no row.
        return inserted, {
            "feedback_id": None,
            "issues": normalized_issues,
            "user_notes": user_notes,
            "pending": True,
        }

    async def update_idempotency(
        self,
        dedupe_key: str,
        issues: list[str],
        user_notes: Optional[str],
    ) -> None:
        normalized_issues = self._merge_issue_lists([], issues)
        issues_payload = json.dumps(normalized_issues)
        if self.db.backend_type == BackendType.SQLITE:
            update_sql = (
                "UPDATE feedback_idempotency "
                "SET issues = ?, user_notes = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE dedupe_key = ?"
            )
        else:
            update_sql = (
                "UPDATE feedback_idempotency "
                "SET issues = %s, user_notes = %s, updated_at = CURRENT_TIMESTAMP "
                "WHERE dedupe_key = %s"
            )

        with self.db.transaction() as conn:
            conn.execute(update_sql, (issues_payload, user_notes, dedupe_key))

    async def clear_idempotency(self, dedupe_key: str) -> None:
        if self.db.backend_type == BackendType.SQLITE:
            delete_sql = "DELETE FROM feedback_idempotency WHERE dedupe_key = ?"
        else:
            delete_sql = "DELETE FROM feedback_idempotency WHERE dedupe_key = %s"

        with self.db.transaction() as conn:
            conn.execute(delete_sql, (dedupe_key,))

    async def finalize_idempotency(
        self,
        dedupe_key: str,
        feedback_id: Optional[str],
        fallback_issues: list[str],
        fallback_user_notes: Optional[str],
    ) -> tuple[list[str], Optional[str], bool]:
        """Finalize an idempotency record and return merged payload state."""
        if self.db.backend_type == BackendType.SQLITE:
            select_sql = (
                "SELECT feedback_id, issues, user_notes, pending "
                "FROM feedback_idempotency WHERE dedupe_key = ?"
            )
            update_sql = (
                "UPDATE feedback_idempotency "
                "SET feedback_id = ?, issues = ?, user_notes = ?, pending = 0, updated_at = CURRENT_TIMESTAMP "
                "WHERE dedupe_key = ?"
            )
        else:
            select_sql = (
                "SELECT feedback_id, issues, user_notes, pending "
                "FROM feedback_idempotency WHERE dedupe_key = %s FOR UPDATE"
            )
            update_sql = (
                "UPDATE feedback_idempotency "
                "SET feedback_id = %s, issues = %s, user_notes = %s, pending = FALSE, updated_at = CURRENT_TIMESTAMP "
                "WHERE dedupe_key = %s"
            )

        with self.db.transaction() as conn:
            cursor = conn.execute(select_sql, (dedupe_key,))
            row = cursor.fetchone()
            if not row:
                return fallback_issues, fallback_user_notes, False

            record = self._idempotency_row_to_dict(row, cursor)
            record_issues = self._decode_issues(record.get("issues"))
            merged_issues = self._merge_issue_lists(fallback_issues, record_issues)
            merged_user_notes = (
                record.get("user_notes")
                if record.get("user_notes") is not None
                else fallback_user_notes
            )
            has_pending_merge = (
                merged_issues != fallback_issues or merged_user_notes != fallback_user_notes
            )

            conn.execute(
                update_sql,
                (
                    feedback_id,
                    json.dumps(merged_issues),
                    merged_user_notes,
                    dedupe_key,
                ),
            )
            return merged_issues, merged_user_notes, has_pending_merge

    async def merge_feedback_update(
        self,
        feedback_id: str,
        issues: Optional[list[str]] = None,
        user_notes: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Merge issues (union) and overwrite user_notes for an existing feedback row."""
        if issues is None and user_notes is None:
            return None

        select_sql = "SELECT issues, user_notes FROM conversation_feedback WHERE id = %s"
        if self.db.backend_type == BackendType.SQLITE:
            select_sql = select_sql.replace('%s', '?')

        result: Optional[dict[str, Any]] = None
        try:
            cursor = self.db.execute_query(select_sql, (feedback_id,))
            row = cursor.fetchone()
            if row:
                if isinstance(row, dict):
                    record = dict(row)
                else:
                    # Handle sqlite3.Row or tuple-like results
                    if hasattr(row, "keys"):
                        record = dict(row)
                    elif cursor.description:
                        columns = [col[0] for col in cursor.description]
                        record = dict(zip(columns, row))
                    else:
                        record = {}

                raw_issues = record.get("issues")
                existing_issues: list[str] = []
                if raw_issues:
                    try:
                        decoded = json.loads(raw_issues)
                        if isinstance(decoded, list):
                            existing_issues = [str(item) for item in decoded if str(item).strip()]
                    except (json.JSONDecodeError, TypeError, ValueError):
                        existing_issues = []

                def _merge_issues(existing: list[str], incoming: Optional[list[str]]) -> list[str]:
                    if not incoming:
                        return list(existing)
                    merged: list[str] = []
                    seen = set()
                    for item in existing + [str(item) for item in incoming if str(item).strip()]:
                        if item in seen:
                            continue
                        seen.add(item)
                        merged.append(item)
                    return merged

                updated_issues = _merge_issues(existing_issues, issues)
                existing_notes = record.get("user_notes")
                updated_notes = existing_notes if user_notes is None else user_notes

                if updated_issues == existing_issues and updated_notes == existing_notes:
                    result = {"issues": existing_issues, "user_notes": existing_notes}
                else:
                    update_sql = "UPDATE conversation_feedback SET issues = %s, user_notes = %s WHERE id = %s"
                    if self.db.backend_type == BackendType.SQLITE:
                        update_sql = update_sql.replace('%s', '?')

                    with self.db.transaction() as conn:
                        conn.execute(update_sql, (json.dumps(updated_issues), updated_notes, feedback_id))

                    result = {"issues": updated_issues, "user_notes": updated_notes}
        except (BackendDatabaseError, CharactersRAGDBError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            logger.error("Failed to merge feedback update")
        else:
            return result
        return None

    async def get_conversation_feedback(
        self,
        conversation_id: str
    ) -> list[dict[str, Any]]:
        """Get all feedback for a conversation."""
        select_sql = """
            SELECT *
            FROM conversation_feedback
            WHERE conversation_id = %s
            ORDER BY created_at DESC
        """
        if self.db.backend_type == BackendType.SQLITE:
            select_sql = select_sql.replace('%s', '?')

        try:
            cursor = self.db.execute_query(select_sql, (conversation_id,))
            feedback: list[dict[str, Any]] = []
            for row in cursor.fetchall():
                fb = dict(row)
                fb["document_ids"] = json.loads(fb["document_ids"]) if fb.get("document_ids") else []
                fb["chunk_ids"] = json.loads(fb["chunk_ids"]) if fb.get("chunk_ids") else []
                fb["issues"] = json.loads(fb["issues"]) if fb.get("issues") else []
                helpful_value = fb.get("helpful")
                fb["helpful"] = None if helpful_value is None else bool(helpful_value)
                feedback.append(fb)
        except (BackendDatabaseError, CharactersRAGDBError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            logger.error("Failed to get conversation feedback")
        else:
            return feedback
        return []

    async def get_feedback_by_id(self, feedback_id: str) -> Optional[dict[str, Any]]:
        """Get a single feedback record by its id."""
        select_sql = "SELECT * FROM conversation_feedback WHERE id = %s"
        if self.db.backend_type == BackendType.SQLITE:
            select_sql = select_sql.replace('%s', '?')

        try:
            cursor = self.db.execute_query(select_sql, (feedback_id,))
            row = cursor.fetchone()
            if not row:
                return None
            fb = dict(row)
            fb["document_ids"] = json.loads(fb["document_ids"]) if fb.get("document_ids") else []
            fb["chunk_ids"] = json.loads(fb["chunk_ids"]) if fb.get("chunk_ids") else []
            fb["issues"] = json.loads(fb["issues"]) if fb.get("issues") else []
            helpful_value = fb.get("helpful")
            fb["helpful"] = None if helpful_value is None else bool(helpful_value)
            return fb
        except (BackendDatabaseError, CharactersRAGDBError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            logger.error("Failed to get feedback by id")
            return None

    async def delete_feedback(self, feedback_id: str) -> bool:
        """Delete a feedback record by id. Returns True if a row was deleted."""
        delete_sql = "DELETE FROM conversation_feedback WHERE id = %s"
        if self.db.backend_type == BackendType.SQLITE:
            delete_sql = delete_sql.replace('%s', '?')

        try:
            with self.db.transaction() as conn:
                cursor = conn.execute(delete_sql, (feedback_id,))
                deleted = getattr(cursor, "rowcount", 0) or 0
                return deleted > 0
        except (BackendDatabaseError, CharactersRAGDBError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            logger.error("Failed to delete feedback")
            raise


class UnifiedFeedbackSystem:
    """
    Unified feedback system that manages both:
    1. Databases/Analytics.db for server-side QA
    2. ChaChaNotes_DB for user-specific feedback
    """

    def __init__(
        self,
        analytics_db_path: str = DEFAULT_ANALYTICS_DB_PATH,
        chacha_db = None,
        enable_analytics: bool = True
    ):
        """
        Initialize unified feedback system.

        Args:
            analytics_db_path: Path to analytics database
            chacha_db: ChaChaNotes database instance
            enable_analytics: Whether to enable server-side analytics
        """
        self.enable_analytics = enable_analytics

        if enable_analytics:
            self.analytics: Optional[AnalyticsStore] = AnalyticsStore(analytics_db_path)
        else:
            self.analytics = None

        if chacha_db:
            self.user_feedback: Optional[UserFeedbackStore] = UserFeedbackStore(chacha_db)
        else:
            self.user_feedback = None

        # Performance tracking
        self._performance_buffer: deque[dict[str, Any]] = deque(maxlen=100)
        self._error_buffer: deque[dict[str, Any]] = deque(maxlen=50)

    async def submit_feedback(
        self,
        conversation_id: str,
        query: str,
        document_ids: list[str],
        chunk_ids: list[str],
        feedback_type: Optional[str] = None,
        relevance_score: Optional[int] = None,
        helpful: Optional[bool] = None,
        issues: Optional[list[str]] = None,
        user_notes: Optional[str] = None,
        session_id: Optional[str] = None,
        _user_id: Optional[str] = None,  # Reserved for future use
        message_id: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Submit feedback to both stores.

        Returns:
            Result with feedback ID and status
        """
        result: dict[str, Any] = {"success": False, "feedback_id": None, "errors": []}

        # Store in user's conversation DB
        if self.user_feedback and conversation_id:
            try:
                feedback_id = await self.user_feedback.add_feedback(
                    conversation_id=conversation_id,
                    query=query,
                    document_ids=document_ids,
                    chunk_ids=chunk_ids,
                    relevance_score=relevance_score,
                    helpful=helpful,
                    issues=issues,
                    user_notes=user_notes,
                    message_id=message_id
                )
                result["feedback_id"] = feedback_id
                result["success"] = True

            except (BackendDatabaseError, CharactersRAGDBError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as e:
                result["errors"].append(f"User feedback error: {str(e)}")

        # Store anonymized metrics in Analytics DB
        if self.enable_analytics and self.analytics:
            try:
                # Hash query for privacy
                query_hash = hashlib.sha256(query.encode()).hexdigest()
                resolved_type = feedback_type
                if not resolved_type:
                    if relevance_score is not None:
                        resolved_type = "relevance"
                    elif helpful is not None:
                        resolved_type = "helpful"
                    elif issues or user_notes:
                        resolved_type = "report"

                # Record search quality
                if relevance_score:
                    await self.analytics.record_search_quality(
                        query_hash=query_hash,
                        relevance_score=relevance_score / 5.0,  # Normalize to 0-1
                        clicked=len(chunk_ids) > 0
                    )

                # Record document performance
                for doc_id in document_ids:
                    await self.analytics.record_document_performance(
                        {
                            "document_id": doc_id,
                            "relevance_score": relevance_score / 5.0 if relevance_score else None,
                            "feedback": "positive" if helpful is True else "negative" if helpful is False else None,
                        }
                    )

                categories = issues or []
                response_quality = None
                if helpful is True:
                    response_quality = "helpful"
                elif helpful is False:
                    response_quality = "not_helpful"

                rating = relevance_score
                if rating is None and helpful is not None:
                    rating = 1 if helpful else 0

                await self.analytics.record_feedback(
                    {
                        "session_id": session_id,
                        "query": query,
                        "feedback_type": resolved_type,
                        "rating": rating,
                        "response_quality": response_quality,
                        "retrieval_accuracy": None,
                        "response_time_acceptable": None,
                        "categories": categories,
                        "improvement_areas": categories,
                    }
                )

                # Record feedback event
                event = AnalyticsEvent(
                    event_type=AnalyticsEventType.FEEDBACK,
                    query_hash=query_hash,
                    metrics={
                        "relevance": relevance_score,
                        "helpful": helpful,
                        "chunks_used": len(chunk_ids),
                        "has_notes": bool(user_notes),
                        "feedback_type": resolved_type,
                    }
                )
                await self.analytics.record_event(event)

            except (BackendDatabaseError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as e:
                result["errors"].append(f"Analytics error: {str(e)}")

        return result

    async def record_implicit_interaction(
        self,
        *,
        user_id: Optional[str],
        query: Optional[str],
        doc_id: Optional[str],
        event_type: str,
        impression: Optional[list[str]] = None,
        corpus: Optional[str] = None,
        chunk_ids: Optional[list[str]] = None,
        rank: Optional[int] = None,
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        message_id: Optional[str] = None,
        dwell_ms: Optional[int] = None,
    ) -> None:
        """Record a lightweight implicit signal (click/expand/copy).

        Updates per-user personalization priors and pairwise preferences.
        Also emits anonymized analytics event when enabled.
        """
        try:
            # Update per-user store (isolated per tenant)
            try:
                from .user_personalization_store import UserPersonalizationStore  # lazy import
                store = UserPersonalizationStore(user_id)
                store.record_event(
                    event_type=event_type,
                    doc_id=doc_id,
                    corpus=corpus,
                    impression=impression or [],
                    chunk_ids=chunk_ids,
                    rank=rank,
                    session_id=session_id,
                    conversation_id=conversation_id,
                    message_id=message_id,
                    dwell_ms=dwell_ms,
                    query=query,
                )
            except ValueError:
                logger.debug("Personalization store update skipped")
            except (AttributeError, OSError, RuntimeError, TypeError):
                logger.debug("Personalization store update failed")

            # Emit anonymized analytics
            if self.enable_analytics and self.analytics:
                def _hash_identifier(value: Optional[str]) -> Optional[str]:
                    if not value:
                        return None
                    return hashlib.sha256(str(value).encode()).hexdigest()[:16]

                qh = None
                if query:
                    qh = hashlib.sha256(query.encode()).hexdigest()
                impression_list = impression or []
                if len(impression_list) > 50:
                    impression_list = impression_list[:50]
                chunk_list = chunk_ids or []
                if len(chunk_list) > 50:
                    chunk_list = chunk_list[:50]
                metrics = {
                    "implicit": True,
                    "type": event_type,
                    "doc_id": doc_id,
                    "chunk_ids": chunk_list,
                    "rank": rank,
                    "dwell_ms": dwell_ms,
                    "corpus": corpus,
                    "impression_list": impression_list,
                    "session_hash": _hash_identifier(session_id),
                    "conversation_hash": _hash_identifier(conversation_id),
                    "message_hash": _hash_identifier(message_id),
                }
                metrics = {k: v for k, v in metrics.items() if v is not None}
                evt = AnalyticsEvent(
                    event_type=AnalyticsEventType.FEEDBACK,
                    query_hash=qh,
                    metrics=metrics,
                )
                await self.analytics.record_event(evt)
        except (BackendDatabaseError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            logger.debug("Implicit interaction recording failed")

    async def record_search(
        self,
        query: str,
        results_count: int,
        cache_hit: bool,
        latency_ms: float
    ):
        """Record a search event."""
        if self.enable_analytics and self.analytics:
            query_hash = hashlib.sha256(query.encode()).hexdigest()

            event = AnalyticsEvent(
                event_type=AnalyticsEventType.SEARCH,
                query_hash=query_hash,
                metrics={
                    "results_count": results_count,
                    "cache_hit": cache_hit,
                    "latency_ms": latency_ms
                }
            )
            await self.analytics.record_event(event)

            # Record performance metric
            await self.analytics.record_performance_metric(
                metric_type="search_latency",
                value=latency_ms,
                metadata={"cache_hit": cache_hit}
            )

    async def record_citation_usage(
        self,
        document_ids: list[str],
        chunk_ids: list[str],
        citation_style: str
    ):
        """Record citation usage."""
        if self.enable_analytics and self.analytics:
            # Record citation event
            event = AnalyticsEvent(
                event_type=AnalyticsEventType.CITATION,
                metrics={
                    "documents_cited": len(document_ids),
                    "chunks_cited": len(chunk_ids),
                    "style": citation_style
                }
            )
            await self.analytics.record_event(event)

            # Update document performance
            for doc_id in document_ids:
                await self.analytics.record_document_performance(
                    document_id=doc_id,
                    cited=True
                )

    async def record_error(
        self,
        error_type: str,
        error_message: str,
        context: Optional[dict[str, Any]] = None
    ):
        """Record an error occurrence."""
        if self.enable_analytics and self.analytics:
            await self.analytics.record_error(
                error_type=error_type,
                error_message=error_message,
                metadata=context
            )

    async def get_analytics_dashboard(self) -> dict[str, Any]:
        """Get analytics dashboard data."""
        if not self.enable_analytics or not self.analytics:
            return {"error": "Analytics not enabled"}

        return await self.analytics.get_analytics_summary()

    async def get_conversation_insights(
        self,
        conversation_id: str
    ) -> dict[str, Any]:
        """Get insights for a specific conversation."""
        insights = {
            "feedback_count": 0,
            "average_relevance": 0,
            "helpful_percentage": 0,
            "feedback_history": []
        }

        if self.user_feedback:
            feedback = await self.user_feedback.get_conversation_feedback(conversation_id)
            insights["feedback_count"] = len(feedback)
            insights["feedback_history"] = feedback

            if feedback:
                relevance_scores = [f["relevance_score"] for f in feedback if f["relevance_score"]]
                if relevance_scores:
                    insights["average_relevance"] = statistics.mean(relevance_scores)

                helpful_votes = [f["helpful"] for f in feedback if f["helpful"] is not None]
                if helpful_votes:
                    insights["helpful_percentage"] = sum(helpful_votes) / len(helpful_votes) * 100

        return insights


# Global instance management
_feedback_system: Optional[UnifiedFeedbackSystem] = None


def get_feedback_system(
    analytics_db_path: str = DEFAULT_ANALYTICS_DB_PATH,
    chacha_db = None,
    enable_analytics: bool = True
) -> UnifiedFeedbackSystem:
    """Get or create global feedback system instance."""
    global _feedback_system
    if _feedback_system is None:
        _feedback_system = UnifiedFeedbackSystem(
            analytics_db_path=analytics_db_path,
            chacha_db=chacha_db,
            enable_analytics=enable_analytics
        )
    return _feedback_system


# Pipeline integration functions

async def collect_feedback(context: Any, **kwargs) -> Any:
    """Collect feedback in RAG pipeline."""
    if not context.config.get("feedback", {}).get("enabled", False):
        return context

    feedback_system = get_feedback_system(
        chacha_db=context.config.get("chacha_db"),
        enable_analytics=context.config.get("enable_analytics", True)
    )

    # Submit feedback if provided
    if "feedback_data" in context.metadata:
        fb = context.metadata["feedback_data"]

        result = await feedback_system.submit_feedback(
            conversation_id=fb.get("conversation_id", ""),
            query=context.query,
            document_ids=[doc.id for doc in context.documents],
            chunk_ids=[doc.id for doc in context.documents],  # Assuming doc.id is chunk_id
            feedback_type=fb.get("feedback_type"),
            relevance_score=fb.get("relevance_score"),
            helpful=fb.get("helpful"),
            user_notes=fb.get("user_notes"),
            session_id=fb.get("session_id"),
            _user_id=fb.get("user_id")
        )

        context.metadata["feedback_result"] = result

    # Record search metrics
    await feedback_system.record_search(
        query=context.query,
        results_count=len(context.documents),
        cache_hit=context.cache_hit,
        latency_ms=context.timings.get("total", 0) * 1000
    )

    return context


async def apply_feedback_boost(context: Any, **kwargs) -> Any:
    """Apply feedback-based boosting to search results using per-user priors."""
    if not context.config.get("feedback", {}).get("apply_boost", False):
        return context
    try:
        user_id = context.config.get("user_id")
        from .user_personalization_store import UserPersonalizationStore
        store = UserPersonalizationStore(user_id)
        context.documents = store.boost_documents(context.documents, corpus=context.config.get("index_namespace"))
    except ValueError:
        logger.debug("Feedback boost skipped")
    except (AttributeError, OSError, RuntimeError, TypeError):
        logger.debug("Feedback boost failed")
    else:
        return context
    return context
