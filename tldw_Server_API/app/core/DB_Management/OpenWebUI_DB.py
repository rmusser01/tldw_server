"""Read-only helpers for uploaded OpenWebUI SQLite databases."""

from __future__ import annotations

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
TABLE_INFO_QUERIES = {
    "user": "PRAGMA table_info(user)",
    "folder": "PRAGMA table_info(folder)",
    "chat": "PRAGMA table_info(chat)",
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
    table_rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN (?, ?, ?)",
        tuple(REQUIRED_SCHEMA.keys()),
    ).fetchall()
    tables = {str(row["name"]) for row in table_rows}
    missing_tables = sorted(set(REQUIRED_SCHEMA) - tables)
    if missing_tables:
        missing = ", ".join(missing_tables)
        raise ValueError(f"missing required OpenWebUI table: {missing}")

    for table, required_columns in REQUIRED_SCHEMA.items():
        column_rows = conn.execute(TABLE_INFO_QUERIES[table]).fetchall()
        columns = {str(row["name"]) for row in column_rows}
        missing_columns = sorted(required_columns - columns)
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"missing required OpenWebUI column in {table}: {missing}")


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
