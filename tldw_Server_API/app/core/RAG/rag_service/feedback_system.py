# feedback_system.py
"""
Feedback collection and analysis system for RAG.

This module provides functionality to collect user feedback on search results,
track relevance scores, and use feedback to improve future searches.
"""

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np
from loguru import logger

from tldw_Server_API.app.core.Metrics import get_metrics_registry


class InvalidRelevanceScoreError(ValueError):
    """Raised when a relevance score is out of the 1-5 range."""

    def __init__(self) -> None:
        super().__init__("Relevance score must be between 1 and 5")


class FeedbackType(Enum):
    """Types of feedback that can be collected."""
    RELEVANCE = "relevance"  # 1-5 star rating
    HELPFUL = "helpful"  # Yes/No
    CLICK = "click"  # User clicked on result
    DWELL_TIME = "dwell_time"  # Time spent on result
    COPY = "copy"  # User copied text from result
    REPORT = "report"  # User reported an issue


class RelevanceScore(Enum):
    """Relevance scoring scale."""
    VERY_POOR = 1
    POOR = 2
    FAIR = 3
    GOOD = 4
    EXCELLENT = 5


@dataclass
class FeedbackEntry:
    """A single feedback entry."""
    id: str
    query: str
    document_id: str
    user_id: str
    feedback_type: FeedbackType
    value: Any
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "query": self.query,
            "document_id": self.document_id,
            "user_id": self.user_id,
            "feedback_type": self.feedback_type.value,
            "value": json.dumps(self.value) if not isinstance(self.value, (str, int, float)) else self.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": json.dumps(self.metadata)
        }


@dataclass
class QueryPerformance:
    """Performance metrics for a specific query."""
    query: str
    total_results: int = 0
    clicked_results: int = 0
    avg_relevance: float = 0.0
    helpful_count: int = 0
    unhelpful_count: int = 0
    avg_dwell_time: float = 0.0

    @property
    def click_through_rate(self) -> float:
        """Calculate click-through rate."""
        return self.clicked_results / self.total_results if self.total_results > 0 else 0.0

    @property
    def helpfulness_score(self) -> float:
        """Calculate helpfulness score."""
        total = self.helpful_count + self.unhelpful_count
        return self.helpful_count / total if total > 0 else 0.0


class FeedbackStore:
    """SQLite-based storage for feedback data."""

    def __init__(self, db_path: str = "feedback.db"):
        """
        Initialize feedback store.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    value TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    created_at REAL DEFAULT (unixepoch())
                )
            """)

            # Create indexes for efficient queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_query ON feedback(query)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_document ON feedback(document_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user ON feedback(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON feedback(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON feedback(feedback_type)")

            conn.commit()

    def add_feedback(self, entry: FeedbackEntry) -> bool:
        """
        Add feedback entry to store.

        Args:
            entry: Feedback entry to store

        Returns:
            Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                data = entry.to_dict()
                conn.execute(
                    """
                    INSERT OR REPLACE INTO feedback
                    (id, query, document_id, user_id, feedback_type, value, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        data["id"],
                        data["query"],
                        data["document_id"],
                        data["user_id"],
                        data["feedback_type"],
                        data["value"],
                        data["timestamp"],
                        data["metadata"]
                    )
                )
                conn.commit()
                return True

        except (sqlite3.Error, OSError, RuntimeError, TypeError, ValueError):
            logger.error("Failed to store feedback")
            return False

    def get_feedback_for_query(
        self,
        query: str,
        feedback_type: Optional[FeedbackType] = None,
        limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get feedback entries for a specific query."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if feedback_type:
                cursor = conn.execute(
                    """
                    SELECT * FROM feedback
                    WHERE query = ? AND feedback_type = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (query, feedback_type.value, limit)
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM feedback
                    WHERE query = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (query, limit)
                )

            return [dict(row) for row in cursor.fetchall()]

    def get_document_feedback(
        self,
        document_id: str,
        limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get all feedback for a specific document."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM feedback
                WHERE document_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (document_id, limit)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_aggregated_stats(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> dict[str, Any]:
        """Get aggregated feedback statistics."""
        with sqlite3.connect(self.db_path) as conn:
            # Build time filter
            time_filter = ""
            params = []

            if start_time:
                time_filter = "WHERE timestamp >= ?"
                params.append(start_time.isoformat())

            if end_time:
                if time_filter:
                    time_filter += " AND timestamp <= ?"
                else:
                    time_filter = "WHERE timestamp <= ?"
                params.append(end_time.isoformat())

            # Get overall stats
            overall_stats_sql_template = """
                SELECT
                    COUNT(*) as total_feedback,
                    COUNT(DISTINCT query) as unique_queries,
                    COUNT(DISTINCT document_id) as unique_documents,
                    COUNT(DISTINCT user_id) as unique_users
                FROM feedback
                {time_filter}
                """
            overall_stats_sql = overall_stats_sql_template.format_map(locals())  # nosec B608
            cursor = conn.execute(
                overall_stats_sql,
                params
            )
            overall = dict(cursor.fetchone())

            # Get feedback type breakdown
            by_type_sql_template = """
                SELECT
                    feedback_type,
                    COUNT(*) as count
                FROM feedback
                {time_filter}
                GROUP BY feedback_type
                """
            by_type_sql = by_type_sql_template.format_map(locals())  # nosec B608
            cursor = conn.execute(
                by_type_sql,
                params
            )
            by_type = {row[0]: row[1] for row in cursor.fetchall()}

            # Get average relevance scores
            relevance_filter = (" AND " + time_filter) if time_filter else ""
            avg_relevance_sql_template = """
                SELECT AVG(CAST(value AS REAL)) as avg_relevance
                FROM feedback
                WHERE feedback_type = 'relevance'{relevance_filter}
                """
            avg_relevance_sql = avg_relevance_sql_template.format_map(locals())  # nosec B608
            cursor = conn.execute(
                avg_relevance_sql,
                params if time_filter else []
            )
            avg_relevance = cursor.fetchone()[0] or 0.0

            return {
                "overall": overall,
                "by_type": by_type,
                "avg_relevance": avg_relevance
            }


class FeedbackAnalyzer:
    """Analyzes feedback to improve search quality."""

    def __init__(self, store: FeedbackStore):
        """
        Initialize feedback analyzer.

        Args:
            store: Feedback storage backend
        """
        self.store = store
        self.reranking_weights: dict[str, float] = {}
        self.document_scores: dict[str, float] = {}

    def calculate_document_score(self, document_id: str) -> float:
        """
        Calculate quality score for a document based on feedback.

        Args:
            document_id: Document identifier

        Returns:
            Quality score (0-1)
        """
        feedback = self.store.get_document_feedback(document_id)

        if not feedback:
            return 0.5  # Neutral score for no feedback

        # Calculate weighted score
        score = 0.0
        weight_sum = 0.0

        for entry in feedback:
            feedback_type = entry["feedback_type"]
            value = entry["value"]

            if feedback_type == "relevance":
                # Relevance scores (1-5) have high weight
                try:
                    relevance = float(json.loads(value) if isinstance(value, str) else value)
                    normalized = (relevance - 1) / 4  # Normalize to 0-1
                    score += normalized * 3.0  # Weight of 3
                    weight_sum += 3.0
                except (TypeError, ValueError, json.JSONDecodeError):
                    logger.debug("Failed to parse relevance feedback value")
                    try:
                        get_metrics_registry().increment(
                            "app_warning_events_total",
                            labels={"component": "rag", "event": "feedback_parse_relevance_failed"},
                        )
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        logger.debug("metrics increment failed for rag feedback_parse_relevance_failed")

            elif feedback_type == "helpful":
                # Helpful yes/no
                try:
                    helpful = json.loads(value) if isinstance(value, str) else value
                    score += (1.0 if helpful else 0.0) * 2.0  # Weight of 2
                    weight_sum += 2.0
                except (TypeError, ValueError, json.JSONDecodeError):
                    logger.debug("Failed to parse helpful feedback value")
                    try:
                        get_metrics_registry().increment(
                            "app_warning_events_total",
                            labels={"component": "rag", "event": "feedback_parse_helpful_failed"},
                        )
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        logger.debug("metrics increment failed for rag feedback_parse_helpful_failed")

            elif feedback_type == "click":
                # Click indicates interest
                score += 0.7 * 1.0  # Weight of 1
                weight_sum += 1.0

            elif feedback_type == "dwell_time":
                # Longer dwell time is better
                try:
                    dwell = float(json.loads(value) if isinstance(value, str) else value)
                    # Normalize dwell time (assume 30s is good)
                    normalized = min(dwell / 30.0, 1.0)
                    score += normalized * 1.5  # Weight of 1.5
                    weight_sum += 1.5
                except (TypeError, ValueError, json.JSONDecodeError):
                    logger.debug("Failed to parse dwell_time feedback value")
                    try:
                        get_metrics_registry().increment(
                            "app_warning_events_total",
                            labels={"component": "rag", "event": "feedback_parse_dwell_failed"},
                        )
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        logger.debug("metrics increment failed for rag feedback_parse_dwell_failed")

        return score / weight_sum if weight_sum > 0 else 0.5

    def get_query_performance(self, query: str) -> QueryPerformance:
        """
        Analyze performance metrics for a query.

        Args:
            query: Query string

        Returns:
            Performance metrics
        """
        feedback = self.store.get_feedback_for_query(query)

        perf = QueryPerformance(query=query)
        relevance_scores = []
        dwell_times = []
        unique_docs = set()

        for entry in feedback:
            unique_docs.add(entry["document_id"])
            feedback_type = entry["feedback_type"]
            value = entry["value"]

            if feedback_type == "relevance":
                try:
                    score = float(json.loads(value) if isinstance(value, str) else value)
                    relevance_scores.append(score)
                except (TypeError, ValueError, json.JSONDecodeError):
                    logger.debug("Failed to parse relevance score for query document")

            elif feedback_type == "helpful":
                try:
                    helpful = json.loads(value) if isinstance(value, str) else value
                    if helpful:
                        perf.helpful_count += 1
                    else:
                        perf.unhelpful_count += 1
                except (TypeError, ValueError, json.JSONDecodeError):
                    logger.debug("Failed to parse helpful feedback value")

            elif feedback_type == "click":
                perf.clicked_results += 1

            elif feedback_type == "dwell_time":
                try:
                    dwell = float(json.loads(value) if isinstance(value, str) else value)
                    dwell_times.append(dwell)
                except (TypeError, ValueError, json.JSONDecodeError):
                    logger.debug("Failed to parse dwell_time feedback value")

        perf.total_results = len(unique_docs)
        perf.avg_relevance = float(np.mean(relevance_scores)) if relevance_scores else 0.0
        perf.avg_dwell_time = float(np.mean(dwell_times)) if dwell_times else 0.0

        return perf

    def get_reranking_weights(
        self,
        query: str,
        document_ids: list[str]
    ) -> dict[str, float]:
        """
        Get reranking weights for documents based on feedback.

        Args:
            query: Query string
            document_ids: List of document IDs to rerank

        Returns:
            Dictionary of document_id -> weight multiplier
        """
        weights = {}

        for doc_id in document_ids:
            # Get document quality score
            doc_score = self.calculate_document_score(doc_id)

            # Get query-specific feedback
            query_feedback = self.store.get_feedback_for_query(query)
            query_doc_score = 0.5  # Default

            for entry in query_feedback:
                if entry["document_id"] == doc_id and entry["feedback_type"] == "relevance":
                    try:
                        score = float(json.loads(entry["value"]) if isinstance(entry["value"], str) else entry["value"])
                        query_doc_score = (score - 1) / 4  # Normalize to 0-1
                        break
                    except (TypeError, ValueError, json.JSONDecodeError):
                        logger.debug("Failed to parse relevance score for query document")

            # Combine document and query-specific scores
            combined_score = 0.7 * doc_score + 0.3 * query_doc_score

            # Convert to weight multiplier (0.5 to 1.5)
            weights[doc_id] = 0.5 + combined_score

        return weights

    def identify_poor_performers(
        self,
        threshold: float = 0.3,
        min_feedback: int = 5
    ) -> list[str]:
        """
        Identify documents with poor feedback scores.

        Args:
            threshold: Score threshold below which documents are considered poor
            min_feedback: Minimum feedback entries required

        Returns:
            List of poor-performing document IDs
        """
        # Get all documents with feedback
        with sqlite3.connect(self.store.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT document_id, COUNT(*) as count
                FROM feedback
                GROUP BY document_id
                HAVING count >= ?
                """,
                (min_feedback,)
            )

            poor_performers = []

            for row in cursor.fetchall():
                doc_id = row[0]
                score = self.calculate_document_score(doc_id)

                if score < threshold:
                    poor_performers.append(doc_id)

        return poor_performers


class FeedbackSystem:
    """Main feedback system coordinating collection and analysis."""

    def __init__(self, db_path: str = "feedback.db"):
        """
        Initialize feedback system.

        Args:
            db_path: Path to feedback database
        """
        self.store = FeedbackStore(db_path)
        self.analyzer = FeedbackAnalyzer(self.store)
        self.active_sessions: dict[str, Any] = {}

    def generate_feedback_id(self, query: str, document_id: str, user_id: str) -> str:
        """Generate unique feedback ID."""
        content = f"{query}:{document_id}:{user_id}:{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def submit_feedback(
        self,
        query: str,
        document_id: str,
        user_id: str,
        feedback_type: FeedbackType,
        value: Any,
        metadata: Optional[dict[str, Any]] = None
    ) -> bool:
        """
        Submit feedback entry.

        Args:
            query: Query that generated the result
            document_id: ID of the document being rated
            user_id: User providing feedback
            feedback_type: Type of feedback
            value: Feedback value
            metadata: Additional metadata

        Returns:
            Success status
        """
        try:
            entry = FeedbackEntry(
                id=self.generate_feedback_id(query, document_id, user_id),
                query=query,
                document_id=document_id,
                user_id=user_id,
                feedback_type=feedback_type,
                value=value,
                metadata=metadata or {}
            )

            success = self.store.add_feedback(entry)

            if success:
                logger.info(
                    f"Feedback submitted: {feedback_type.value} for doc {document_id} "
                    f"by user {user_id}"
                )
        except (TypeError, ValueError, RuntimeError):
            logger.error("Failed to submit feedback")
            try:
                get_metrics_registry().increment(
                    "app_exception_events_total",
                    labels={"component": "rag", "event": "feedback_submit_failed"},
                )
            except (AttributeError, RuntimeError, TypeError, ValueError):
                logger.debug("metrics increment failed for rag feedback_submit_failed")
        else:
            return success
        return False

    async def submit_relevance_score(
        self,
        query: str,
        document_id: str,
        user_id: str,
        score: int,
        metadata: Optional[dict[str, Any]] = None
    ) -> bool:
        """Submit relevance score (1-5)."""
        if not 1 <= score <= 5:
            raise InvalidRelevanceScoreError()

        return await self.submit_feedback(
            query=query,
            document_id=document_id,
            user_id=user_id,
            feedback_type=FeedbackType.RELEVANCE,
            value=score,
            metadata=metadata
        )

    async def submit_helpful_vote(
        self,
        query: str,
        document_id: str,
        user_id: str,
        helpful: bool,
        metadata: Optional[dict[str, Any]] = None
    ) -> bool:
        """Submit helpful/unhelpful vote."""
        return await self.submit_feedback(
            query=query,
            document_id=document_id,
            user_id=user_id,
            feedback_type=FeedbackType.HELPFUL,
            value=helpful,
            metadata=metadata
        )

    async def track_click(
        self,
        query: str,
        document_id: str,
        user_id: str,
        position: int,
        metadata: Optional[dict[str, Any]] = None
    ) -> bool:
        """Track document click."""
        meta = metadata or {}
        meta["position"] = position

        return await self.submit_feedback(
            query=query,
            document_id=document_id,
            user_id=user_id,
            feedback_type=FeedbackType.CLICK,
            value=True,
            metadata=meta
        )

    async def track_dwell_time(
        self,
        query: str,
        document_id: str,
        user_id: str,
        dwell_seconds: float,
        metadata: Optional[dict[str, Any]] = None
    ) -> bool:
        """Track time spent on document."""
        return await self.submit_feedback(
            query=query,
            document_id=document_id,
            user_id=user_id,
            feedback_type=FeedbackType.DWELL_TIME,
            value=dwell_seconds,
            metadata=metadata
        )

    def get_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> dict[str, Any]:
        """
        Get feedback statistics.

        Args:
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Statistics dictionary
        """
        stats = self.store.get_aggregated_stats(start_time, end_time)

        # Add performance insights
        if stats["overall"]["unique_queries"] > 0:
            # Get top performing queries
            with sqlite3.connect(self.store.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT query, AVG(CAST(value AS REAL)) as avg_score
                    FROM feedback
                    WHERE feedback_type = 'relevance'
                    GROUP BY query
                    ORDER BY avg_score DESC
                    LIMIT 10
                    """
                )
                stats["top_queries"] = [
                    {"query": row[0], "avg_score": row[1]}
                    for row in cursor.fetchall()
                ]

        return stats

    def apply_feedback_reranking(
        self,
        query: str,
        documents: list[dict[str, Any]],
        score_field: str = "score"
    ) -> list[dict[str, Any]]:
        """
        Apply feedback-based reranking to search results.

        Args:
            query: Query string
            documents: List of documents with scores
            score_field: Name of the score field in documents

        Returns:
            Reranked documents
        """
        # Get document IDs
        doc_ids = [doc.get("id", str(i)) for i, doc in enumerate(documents)]

        # Get reranking weights
        weights = self.analyzer.get_reranking_weights(query, doc_ids)

        # Apply weights to scores
        for i, doc in enumerate(documents):
            doc_id = doc.get("id", str(i))
            original_score = doc.get(score_field, 0.0)
            weight = weights.get(doc_id, 1.0)

            # Apply weight and store both scores
            doc[score_field] = original_score * weight
            doc["original_score"] = original_score
            doc["feedback_weight"] = weight

        # Resort by new scores
        documents.sort(key=lambda x: x[score_field], reverse=True)

        return documents


# Global instance
_feedback_system: Optional[FeedbackSystem] = None


def get_feedback_system(db_path: str = "feedback.db") -> FeedbackSystem:
    """Get or create global feedback system instance."""
    global _feedback_system
    if _feedback_system is None:
        _feedback_system = FeedbackSystem(db_path)
    return _feedback_system


# Pipeline integration functions

async def collect_feedback(context: Any, **kwargs) -> Any:
    """Collect feedback in RAG pipeline."""
    if not context.config.get("feedback", {}).get("enabled", False):
        return context

    feedback = get_feedback_system()

    # Check if we have feedback data in context
    if "feedback" in context.metadata:
        feedback_data = context.metadata["feedback"]

        # Submit feedback
        await feedback.submit_feedback(
            query=context.query,
            document_id=feedback_data.get("document_id"),
            user_id=feedback_data.get("user_id", "anonymous"),
            feedback_type=FeedbackType(feedback_data.get("type", "click")),
            value=feedback_data.get("value"),
            metadata=feedback_data.get("metadata", {})
        )

    return context


async def apply_feedback_scores(context: Any, **kwargs) -> Any:
    """Apply feedback-based reranking in pipeline."""
    if not context.config.get("feedback", {}).get("reranking", False):
        return context

    feedback = get_feedback_system()

    # Apply reranking if we have documents
    if hasattr(context, "documents") and context.documents:
        context.documents = feedback.apply_feedback_reranking(
            query=context.query,
            documents=context.documents
        )

        logger.debug(f"Applied feedback reranking to {len(context.documents)} documents")

    return context
