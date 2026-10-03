import contextlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from tldw_Server_API.app.core.DB_Management.sqlite_policy import (
    configure_sqlite_connection,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths


def _project_root_from(file_path: Path) -> Path:
    # file_path: .../tldw_Server_API/app/core/Embeddings/vector_store_batches_db.py
    return file_path.parent.parent.parent.parent


def get_db_path(user_id: Optional[str]) -> Path:
    user_dir = DatabasePaths.get_user_vector_store_dir(user_id)
    return user_dir / 'vector_store_batches.db'


def _prime(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Apply recommended SQLite PRAGMAs for concurrency and resilience."""
    with contextlib.suppress(Exception):
        configure_sqlite_connection(
            conn,
            busy_timeout_ms=3000,
            temp_store=None,
            synchronous=None,
            foreign_keys=False,
        )
    return conn


def _connect(user_id: Optional[str]) -> sqlite3.Connection:
    return _prime(sqlite3.connect(get_db_path(user_id), check_same_thread=False))


def _ensure_initialized(user_id: Optional[str]) -> None:
    """Ensure the batches table exists for the given user.

    This guards against cases where the base directory changes during tests
    after module import time, so the original init_db path no longer applies.
    """
    try:
        init_db(user_id)
    except Exception as init_error:
        # Best effort; callers will raise if operations still fail
        _ = init_error


def init_db(user_id: Optional[str]) -> None:
    with _connect(user_id) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vector_store_batches (
                id TEXT PRIMARY KEY,
                store_id TEXT NOT NULL,
                user_id TEXT,
                status TEXT NOT NULL,
                upserted INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                meta_json TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
            """
        )
        conn.commit()


def create_batch(batch_id: str, store_id: str, user_id: Optional[str], status: str = 'processing',
                 upserted: int = 0, error: Optional[str] = None, meta: Optional[dict[str, Any]] = None) -> None:
    ts = int(time.time())
    _ensure_initialized(user_id)
    with _connect(user_id) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO vector_store_batches
            (id, store_id, user_id, status, upserted, error, meta_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM vector_store_batches WHERE id = ?), ?), ?)
            """,
            (
                batch_id, store_id, user_id, status, upserted, error or None,
                json.dumps(meta or {}), batch_id, ts, ts
            )
        )
        conn.commit()


def update_batch(batch_id: str, user_id: Optional[str], status: Optional[str] = None, upserted: Optional[int] = None,
                 error: Optional[str] = None, meta: Optional[dict[str, Any]] = None) -> None:
    _ensure_initialized(user_id)
    fields = []
    values = []
    if status is not None:
        fields.append('status = ?')
        values.append(status)
    if upserted is not None:
        fields.append('upserted = ?')
        values.append(upserted)
    if error is not None:
        fields.append('error = ?')
        values.append(error)
    if meta is not None:
        fields.append('meta_json = ?')
        values.append(json.dumps(meta))
    # Always update updated_at
    fields.append('updated_at = ?')
    values.append(int(time.time()))
    values.append(batch_id)

    if not fields:
        return

    with _connect(user_id) as conn:
        set_clause = ", ".join(fields)
        update_batch_sql_template = "UPDATE vector_store_batches SET {set_clause} WHERE id = ?"
        update_batch_sql = update_batch_sql_template.format_map(locals())  # nosec B608
        conn.execute(update_batch_sql, values)
        conn.commit()


def get_batch(batch_id: str, user_id: Optional[str]) -> Optional[dict[str, Any]]:
    _ensure_initialized(user_id)
    with _connect(user_id) as conn:
        cur = conn.execute(
            "SELECT id, store_id, user_id, status, upserted, error, meta_json, created_at, updated_at\n             FROM vector_store_batches WHERE id = ?",
            (batch_id,)
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            'id': row[0],
            'store_id': row[1],
            'user_id': row[2],
            'status': row[3],
            'upserted': row[4],
            'error': row[5],
            'meta': json.loads(row[6] or '{}'),
            'created_at': row[7],
            'updated_at': row[8],
        }


def count_batches(user_id: Optional[str]) -> int:
    _ensure_initialized(user_id)
    with _connect(user_id) as conn:
        row = conn.execute("SELECT COUNT(1) FROM vector_store_batches").fetchone()
        return int(row[0]) if row and row[0] is not None else 0


def list_batches(user_id: Optional[str], status: Optional[str] = None, limit: int = 50, offset: int = 0):
    _ensure_initialized(user_id)
    query = "SELECT id, store_id, user_id, status, upserted, error, meta_json, created_at, updated_at FROM vector_store_batches"
    params = []
    if status:
        query += " WHERE status = ?"
        params.append(status)
    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    with _connect(user_id) as conn:
        cur = conn.execute(query, params)
        rows = cur.fetchall()
        return [
            {
                'id': r[0],
                'store_id': r[1],
                'user_id': r[2],
                'status': r[3],
                'upserted': r[4],
                'error': r[5],
                'meta': json.loads(r[6] or '{}'),
                'created_at': r[7],
                'updated_at': r[8],
            }
            for r in rows
        ]
