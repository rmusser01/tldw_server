"""Receipt-aware SQLite privacy operations without schema bootstrap or key erasure."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from tldw_Server_API.app.core.DB_Management.sqlite_policy import configure_sqlite_connection


def has_workspace_chat_startup_receipts(path: Path) -> bool:
    """Probe an existing file; missing files return false and storage errors propagate."""
    if not path.exists():
        return False
    with closing(sqlite3.connect(path)) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'workspace_chat_startup_receipts'",
        ).fetchone()
    return bool(row[0] if row else 0)


def count_workspace_chat_startup_messages(path: Path, owner_id: str) -> int:
    """Count live chat messages attributed to the writer or immutable receipt owner."""
    if not path.exists():
        raise FileNotFoundError(path)
    with closing(sqlite3.connect(path)) as conn:
        row = conn.execute(
            """
            SELECT COUNT(1)
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE (c.client_id = ? OR EXISTS (
                SELECT 1 FROM workspace_chat_startup_receipts r
                WHERE r.conversation_id = c.id AND r.owner_user_id = ?
            )) AND c.deleted = 0 AND m.deleted = 0
            """,
            (owner_id, owner_id),
        ).fetchone()
    return int(row[0] if row else 0)


def erase_workspace_chat_startup_chats(path: Path, owner_id: str) -> int:
    """Atomically delete owned chats/messages, retaining FK-nullified receipt tombstones."""
    if not path.exists():
        return 0
    with closing(sqlite3.connect(path)) as conn, conn:
        configure_sqlite_connection(conn, use_wal=False, synchronous=None)
        messages = conn.execute(
            "DELETE FROM messages WHERE conversation_id IN ("
            "SELECT c.id FROM conversations c WHERE c.client_id = ? OR EXISTS ("
            "SELECT 1 FROM workspace_chat_startup_receipts r "
            "WHERE r.conversation_id = c.id AND r.owner_user_id = ?))",
            (owner_id, owner_id),
        ).rowcount
        conversations = conn.execute(
            "DELETE FROM conversations WHERE client_id = ? OR EXISTS ("
            "SELECT 1 FROM workspace_chat_startup_receipts r "
            "WHERE r.conversation_id = conversations.id AND r.owner_user_id = ?)",
            (owner_id, owner_id),
        ).rowcount
    return messages + conversations
