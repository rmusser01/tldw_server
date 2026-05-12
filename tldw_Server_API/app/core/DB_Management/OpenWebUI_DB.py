"""Read-only helpers for uploaded OpenWebUI SQLite databases."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


SQLITE_MAGIC = b"SQLite format 3\x00"

REQUIRED_SCHEMA: dict[str, set[str]] = {
    "user": {"id", "name", "email", "created_at", "updated_at"},
    "folder": {
        "id",
        "parent_id",
        "user_id",
        "name",
        "items",
        "meta",
        "is_expanded",
        "created_at",
        "updated_at",
    },
    "chat": {
        "id",
        "user_id",
        "title",
        "chat",
        "created_at",
        "updated_at",
        "share_id",
        "archived",
        "pinned",
        "meta",
        "folder_id",
    },
}
HYDRATION_FILE_SCHEMA: dict[str, set[str]] = {
    "file": {
        "id",
        "user_id",
        "hash",
        "filename",
        "path",
        "data",
        "meta",
        "created_at",
        "updated_at",
    },
}
HYDRATION_CHAT_FILE_SCHEMA: dict[str, set[str]] = {
    "chat_file": {
        "id",
        "chat_id",
        "file_id",
        "message_id",
        "user_id",
        "created_at",
        "updated_at",
    },
}
TABLE_INFO_QUERIES = {
    "user": "PRAGMA table_info(user)",
    "folder": "PRAGMA table_info(folder)",
    "chat": "PRAGMA table_info(chat)",
    "file": "PRAGMA table_info(file)",
    "chat_file": "PRAGMA table_info(chat_file)",
}


@contextmanager
def open_validated_openwebui_db(file_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Open an uploaded OpenWebUI SQLite database in read-only mode after schema validation."""
    path = Path(file_path)
    try:
        with path.open("rb") as handle:
            if handle.read(len(SQLITE_MAGIC)) != SQLITE_MAGIC:
                raise ValueError("Invalid OpenWebUI SQLite database")
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("Unable to read OpenWebUI SQLite database") from exc

    resolved = path.resolve()
    uri = f"{resolved.as_uri()}?mode=ro"
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            conn.enable_load_extension(False)
        except (AttributeError, sqlite3.OperationalError):
            pass
        validate_openwebui_schema(conn)
        yield conn
    except ValueError:
        raise
    except sqlite3.Error as exc:
        raise ValueError("Invalid OpenWebUI SQLite database") from exc
    finally:
        if conn is not None:
            conn.close()


def validate_openwebui_schema(conn: sqlite3.Connection) -> None:
    """Validate the minimal OpenWebUI source tables required for chat import."""
    _validate_required_schema(conn, REQUIRED_SCHEMA)


def validate_openwebui_file_schema(conn: sqlite3.Connection) -> None:
    """Validate OpenWebUI source tables required for attachment hydration."""
    _validate_required_schema(conn, HYDRATION_FILE_SCHEMA | HYDRATION_CHAT_FILE_SCHEMA)


def _validate_required_schema(conn: sqlite3.Connection, required_schema: dict[str, set[str]]) -> None:
    """Validate that a source database contains the required tables and columns."""
    table_rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'",
    ).fetchall()
    tables = {str(row["name"]) for row in table_rows}
    missing_tables = sorted(set(required_schema) - tables)
    if missing_tables:
        missing = ", ".join(missing_tables)
        raise ValueError(f"missing required OpenWebUI table: {missing}")

    for table, required_columns in required_schema.items():
        column_rows = conn.execute(TABLE_INFO_QUERIES[table]).fetchall()
        columns = {str(row["name"]) for row in column_rows}
        missing_columns = sorted(required_columns - columns)
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"missing required OpenWebUI column in {table}: {missing}")


def _normalize_ids(ids: list[str] | tuple[str, ...]) -> list[str]:
    """Return string ids for query binding and drop empty batches early."""
    return [str(row_id) for row_id in ids]


def load_openwebui_users(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return source users ordered by display label."""
    return list(
        conn.execute(
            """
            SELECT id, name, email, created_at, updated_at
            FROM user
            ORDER BY COALESCE(name, email, id), id
            """
        ).fetchall()
    )


def load_openwebui_user(conn: sqlite3.Connection, user_id: str) -> sqlite3.Row | None:
    """Return one source user by OpenWebUI user id."""
    return conn.execute(
        """
        SELECT id, name, email, created_at, updated_at
        FROM user
        WHERE id = ?
        """,
        (user_id,),
    ).fetchone()


def load_openwebui_chats_for_user(conn: sqlite3.Connection, user_id: str) -> list[sqlite3.Row]:
    """Return all chat rows for a source user."""
    return list(iter_openwebui_chats_for_user(conn, user_id))


def iter_openwebui_chats_for_user(conn: sqlite3.Connection, user_id: str) -> Iterator[sqlite3.Row]:
    """Iterate chat rows for a source user without materializing the full result set."""
    return iter(
        conn.execute(
            """
            SELECT id, user_id, title, chat, created_at, updated_at, share_id, archived, pinned, meta, folder_id
            FROM chat
            WHERE user_id = ?
            ORDER BY COALESCE(updated_at, created_at, 0), id
            """,
            (user_id,),
        )
    )


def load_openwebui_folders_for_user(conn: sqlite3.Connection, user_id: str) -> dict[str, sqlite3.Row]:
    """Return source folders for one OpenWebUI user keyed by folder id."""
    rows = conn.execute(
        """
        SELECT id, parent_id, user_id, name, items, meta, is_expanded, created_at, updated_at
        FROM folder
        WHERE user_id = ?
        """,
        (user_id,),
    ).fetchall()
    return {str(row["id"]): row for row in rows}


def load_openwebui_file_rows_for_ids(
    conn: sqlite3.Connection,
    file_ids: list[str] | tuple[str, ...],
    user_id: str | None = None,
) -> list[sqlite3.Row]:
    """Return OpenWebUI file rows for the supplied file ids."""
    normalized_file_ids = _normalize_ids(file_ids)
    if not normalized_file_ids:
        return []

    return list(
        conn.execute(
            """
            SELECT id, user_id, hash, filename, path, data, meta, created_at, updated_at
            FROM file
            WHERE id IN (SELECT CAST(value AS TEXT) FROM json_each(?))
              AND (? IS NULL OR user_id = ?)
            ORDER BY id
            """,
            (json.dumps(normalized_file_ids), user_id, user_id),
        ).fetchall()
    )


def load_openwebui_chat_file_rows_for_chats(
    conn: sqlite3.Connection,
    chat_ids: list[str] | tuple[str, ...],
    user_id: str | None = None,
) -> list[sqlite3.Row]:
    """Return OpenWebUI chat-file link rows for the supplied chat ids."""
    normalized_chat_ids = _normalize_ids(chat_ids)
    if not normalized_chat_ids:
        return []

    return list(
        conn.execute(
            """
            SELECT id, chat_id, file_id, message_id, user_id, created_at, updated_at
            FROM chat_file
            WHERE chat_id IN (SELECT CAST(value AS TEXT) FROM json_each(?))
              AND (? IS NULL OR user_id = ?)
            ORDER BY chat_id, id
            """,
            (json.dumps(normalized_chat_ids), user_id, user_id),
        ).fetchall()
    )


def iter_openwebui_files_for_user(conn: sqlite3.Connection, user_id: str) -> Iterator[sqlite3.Row]:
    """Iterate all OpenWebUI file rows for a source user."""
    return iter(
        conn.execute(
            """
            SELECT id, user_id, hash, filename, path, data, meta, created_at, updated_at
            FROM file
            WHERE user_id = ?
            ORDER BY COALESCE(updated_at, created_at, 0), id
            """,
            (user_id,),
        )
    )
