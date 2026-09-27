"""Owner-scoped Chat evidence queries using a caller-owned query executor.

The executor retains its adapter or path fallback, error policy, and connection
lifecycle. These helpers own only query construction and row selection.
"""

from collections.abc import Callable
from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType

QueryExecutor = Callable[[str, tuple[Any, ...]], list[dict[str, Any]]]


def search_chat_history(
    execute_query: QueryExecutor,
    query: str,
    *,
    db_adapter: Any = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return recent active Chat evidence, excluding saved Knowledge QA history.

    PostgreSQL shares tables across users and requires the adapter's owner
    predicate. SQLite uses the caller's per-user database file.
    """
    is_postgres = getattr(db_adapter, "backend_type", None) == BackendType.POSTGRESQL
    owner_clause = "AND conv.client_id = ?" if is_postgres else ""
    owner_params = (str(db_adapter.client_id),) if is_postgres else ()
    sql = f"""
        SELECT
            m.id,
            m.conversation_id,
            m.content,
            m.sender,
            m.timestamp,
            conv.character_id,
            conv.source AS conversation_source,
            conv.title AS conversation_title,
            cc.name AS character_name
        FROM messages m
        JOIN conversations conv ON m.conversation_id = conv.id
        LEFT JOIN character_cards cc ON conv.character_id = cc.id
        WHERE m.deleted = 0
          AND conv.deleted = 0
          {owner_clause}
          AND m.content LIKE ?
          AND COALESCE(conv.source, '') != ?
        ORDER BY m.timestamp DESC
        LIMIT ?
    """  # nosec B608 - fixed owner predicate; every value remains bound.
    return execute_query(sql, (*owner_params, f"%{query}%", "knowledge_qa", limit))


def get_chat_history_metadata(
    execute_query: QueryExecutor,
    message_id: str,
    *,
    db_adapter: Any = None,
) -> dict[str, Any]:
    """Return an active message's metadata within the caller's database scope."""
    is_postgres = getattr(db_adapter, "backend_type", None) == BackendType.POSTGRESQL
    owner_clause = "AND conv.client_id = ?" if is_postgres else ""
    owner_params = (str(db_adapter.client_id),) if is_postgres else ()
    results = execute_query(
        f"""
        SELECT m.*, conv.character_id
        FROM messages m
        JOIN conversations conv ON m.conversation_id = conv.id
        WHERE m.id = ?
          AND m.deleted = 0 AND conv.deleted = 0
          {owner_clause}
        """,  # nosec B608 - fixed owner predicate; every value remains bound.
        (message_id, *owner_params),
    )
    return dict(results[0]) if results else {}
