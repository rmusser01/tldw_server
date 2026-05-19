"""Sandbox run queue database abstraction.

Provides the SQL layer for the durable run queue used by the Sandbox
orchestrator.  Keeps raw SQL out of ``Sandbox/store.py`` and follows
the project convention of placing all DB operations under
``/app/core/DB_Management/``.

Two concrete implementations:

* :class:`SQLiteRunQueueDB` -- for the SQLite sandbox store.
* :class:`PostgresRunQueueDB` -- for the PostgreSQL sandbox store.
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Generator

from loguru import logger


# ---------------------------------------------------------------------------
# SQLite implementation
# ---------------------------------------------------------------------------


class SQLiteRunQueueDB:
    """SQLite-backed run queue operations.

    Accepts either a ``sqlite3.Connection`` factory (callable returning a
    context-managed connection) or a raw connection.
    """

    _INIT_SQL = """\
    CREATE TABLE IF NOT EXISTS run_queue (
        run_id   TEXT PRIMARY KEY,
        user_id  TEXT NOT NULL,
        priority INTEGER DEFAULT 0,
        enqueued_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """

    def __init__(self, conn_factory: Any, lock: threading.Lock | None = None) -> None:
        self._conn = conn_factory
        self._lock = lock or threading.RLock()

    def ensure_table(self, con: Any) -> None:
        """Create the ``run_queue`` table if it does not exist."""
        con.executescript(self._INIT_SQL)

    def enqueue(self, run_id: str, user_id: str, priority: int = 0) -> None:
        """Insert or replace a run in the queue."""
        with self._lock, self._conn() as con:
            con.execute(
                "INSERT OR REPLACE INTO run_queue"
                "(run_id, user_id, priority, enqueued_at) "
                "VALUES (?, ?, ?, datetime('now'))",
                (str(run_id), str(user_id), int(priority)),
            )

    def dequeue(self, worker_id: str) -> dict[str, Any] | None:
        """Atomically remove and return the highest-priority queued run.

        Returns ``None`` when the queue is empty.
        """
        with self._lock, self._conn() as con:
            cur = con.execute(
                "SELECT run_id, user_id, priority, enqueued_at "
                "FROM run_queue "
                "ORDER BY priority DESC, enqueued_at ASC "
                "LIMIT 1",
            )
            row = cur.fetchone()
            if not row:
                return None
            run_id = row["run_id"] if isinstance(row, sqlite3.Row) else row[0]
            result = {
                "run_id": row["run_id"] if isinstance(row, sqlite3.Row) else row[0],
                "user_id": row["user_id"] if isinstance(row, sqlite3.Row) else row[1],
                "priority": int(row["priority"] if isinstance(row, sqlite3.Row) else row[2]),
                "enqueued_at": row["enqueued_at"] if isinstance(row, sqlite3.Row) else row[3],
            }
            con.execute("DELETE FROM run_queue WHERE run_id = ?", (run_id,))
            return result

    def remove(self, run_id: str) -> bool:
        """Remove a specific run from the queue. Returns ``True`` if found."""
        with self._lock, self._conn() as con:
            cur = con.execute(
                "DELETE FROM run_queue WHERE run_id = ?",
                (str(run_id),),
            )
            return bool(cur.rowcount and cur.rowcount > 0)


# ---------------------------------------------------------------------------
# PostgreSQL implementation
# ---------------------------------------------------------------------------


class PostgresRunQueueDB:
    """PostgreSQL-backed run queue operations.

    Accepts a connection factory (callable returning a context-managed
    connection) and an optional lock.
    """

    _INIT_SQL = """\
    CREATE TABLE IF NOT EXISTS run_queue (
        run_id      TEXT PRIMARY KEY,
        user_id     TEXT NOT NULL,
        priority    INTEGER DEFAULT 0,
        enqueued_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
    );
    """

    def __init__(self, conn_factory: Any, lock: threading.Lock | None = None) -> None:
        self._conn = conn_factory
        self._lock = lock or threading.RLock()

    def ensure_table(self, con: Any, cur: Any) -> None:
        """Create the ``run_queue`` table if it does not exist."""
        cur.execute(self._INIT_SQL)

    def enqueue(self, run_id: str, user_id: str, priority: int = 0) -> None:
        """Insert or update a run in the queue."""
        with self._lock, self._conn() as con, con.cursor() as cur:
            cur.execute(
                "INSERT INTO run_queue(run_id, user_id, priority, enqueued_at) "
                "VALUES (%s, %s, %s, NOW()) "
                "ON CONFLICT (run_id) DO UPDATE SET priority = EXCLUDED.priority, "
                "enqueued_at = NOW()",
                (str(run_id), str(user_id), int(priority)),
            )

    def dequeue(self, worker_id: str) -> dict[str, Any] | None:
        """Atomically dequeue the highest-priority run using FOR UPDATE SKIP LOCKED."""
        with self._lock, self._conn() as con, con.cursor() as cur:
            cur.execute(
                "DELETE FROM run_queue "
                "WHERE run_id = ("
                "  SELECT run_id FROM run_queue "
                "  ORDER BY priority DESC, enqueued_at ASC "
                "  LIMIT 1 "
                "  FOR UPDATE SKIP LOCKED"
                ") RETURNING run_id, user_id, priority, enqueued_at",
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "run_id": row[0],
                "user_id": row[1],
                "priority": int(row[2]),
                "enqueued_at": str(row[3]),
            }

    def remove(self, run_id: str) -> bool:
        """Remove a specific run from the queue. Returns ``True`` if found."""
        with self._lock, self._conn() as con, con.cursor() as cur:
            cur.execute(
                "DELETE FROM run_queue WHERE run_id = %s",
                (str(run_id),),
            )
            return bool(cur.rowcount and cur.rowcount > 0)
