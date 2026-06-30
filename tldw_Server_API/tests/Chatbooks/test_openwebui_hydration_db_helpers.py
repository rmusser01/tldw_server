import sqlite3
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management import OpenWebUI_DB as openwebui_db_reader


pytestmark = pytest.mark.unit


OPENWEBUI_BASE_TABLES = {
    "user": [
        "id TEXT PRIMARY KEY",
        "name TEXT",
        "email TEXT",
        "created_at INTEGER",
        "updated_at INTEGER",
    ],
    "folder": [
        "id TEXT PRIMARY KEY",
        "parent_id TEXT",
        "user_id TEXT",
        "name TEXT",
        "items TEXT",
        "meta TEXT",
        "is_expanded INTEGER",
        "created_at INTEGER",
        "updated_at INTEGER",
    ],
    "chat": [
        "id TEXT PRIMARY KEY",
        "user_id TEXT",
        "title TEXT",
        "chat TEXT",
        "created_at INTEGER",
        "updated_at INTEGER",
        "share_id TEXT",
        "archived INTEGER",
        "pinned INTEGER",
        "meta TEXT",
        "folder_id TEXT",
    ],
}

OPENWEBUI_FILE_COLUMNS = [
    "id TEXT PRIMARY KEY",
    "user_id TEXT",
    "hash TEXT",
    "filename TEXT",
    "path TEXT",
    "data TEXT",
    "meta TEXT",
    "created_at INTEGER",
    "updated_at INTEGER",
]

OPENWEBUI_CHAT_FILE_COLUMNS = [
    "id TEXT PRIMARY KEY",
    "chat_id TEXT",
    "file_id TEXT",
    "message_id TEXT",
    "user_id TEXT",
    "created_at INTEGER",
    "updated_at INTEGER",
]


def _column_name(definition: str) -> str:
    return definition.split(" ", 1)[0]


def _create_table(conn: sqlite3.Connection, name: str, columns: list[str]) -> None:
    conn.execute(f"CREATE TABLE {name} ({', '.join(columns)})")


def write_openwebui_hydration_db(
    path: Path,
    *,
    include_file: bool = True,
    include_chat_file: bool = True,
    omit_file_columns: set[str] | None = None,
    omit_chat_file_columns: set[str] | None = None,
) -> Path:
    conn = sqlite3.connect(path)
    try:
        for table, columns in OPENWEBUI_BASE_TABLES.items():
            _create_table(conn, table, columns)

        if include_file:
            omitted = omit_file_columns or set()
            file_columns = [
                column for column in OPENWEBUI_FILE_COLUMNS if _column_name(column) not in omitted
            ]
            _create_table(conn, "file", file_columns)

        if include_chat_file:
            omitted = omit_chat_file_columns or set()
            chat_file_columns = [
                column
                for column in OPENWEBUI_CHAT_FILE_COLUMNS
                if _column_name(column) not in omitted
            ]
            _create_table(conn, "chat_file", chat_file_columns)

        conn.commit()
    finally:
        conn.close()
    return path


def open_row_connection(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def test_validate_openwebui_file_schema_rejects_missing_file_table(tmp_path):
    db_path = write_openwebui_hydration_db(tmp_path / "webui.db", include_file=False)

    with open_row_connection(db_path) as conn:
        with pytest.raises(ValueError, match="missing required OpenWebUI table: file"):
            openwebui_db_reader.validate_openwebui_file_schema(conn)


@pytest.mark.parametrize("column", ["id", "user_id", "filename"])
def test_validate_openwebui_file_schema_rejects_missing_required_file_columns(
    tmp_path,
    column,
):
    db_path = write_openwebui_hydration_db(
        tmp_path / f"missing-{column}.db",
        omit_file_columns={column},
    )

    with open_row_connection(db_path) as conn:
        with pytest.raises(ValueError, match=f"missing required OpenWebUI column in file: {column}"):
            openwebui_db_reader.validate_openwebui_file_schema(conn)


def test_validate_openwebui_file_schema_rejects_missing_chat_file_table(tmp_path):
    db_path = write_openwebui_hydration_db(tmp_path / "webui.db", include_chat_file=False)

    with open_row_connection(db_path) as conn:
        with pytest.raises(ValueError, match="missing required OpenWebUI table: chat_file"):
            openwebui_db_reader.validate_openwebui_file_schema(conn)


def test_validate_openwebui_schema_still_accepts_text_only_import_schema(tmp_path):
    db_path = write_openwebui_hydration_db(
        tmp_path / "webui.db",
        include_file=False,
        include_chat_file=False,
    )

    with open_row_connection(db_path) as conn:
        openwebui_db_reader.validate_openwebui_schema(conn)


def test_load_openwebui_file_rows_for_ids_filters_by_user(tmp_path):
    db_path = write_openwebui_hydration_db(tmp_path / "webui.db")

    with open_row_connection(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO file (id, user_id, hash, filename, path, data, meta, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("file-a", "owui-user", "hash-a", "notes.pdf", "uploads/file-a_notes.pdf", "{}", "{}", 1, 2),
                ("file-b", "other-user", "hash-b", "private.pdf", "uploads/file-b_private.pdf", "{}", "{}", 3, 4),
                ("file-c", "owui-user", "hash-c", "image.png", "uploads/file-c_image.png", "{}", "{}", 5, 6),
            ],
        )
        conn.commit()

        rows = openwebui_db_reader.load_openwebui_file_rows_for_ids(
            conn,
            ["file-a", "file-b"],
            user_id="owui-user",
        )

    assert [row["id"] for row in rows] == ["file-a"]
    assert rows[0]["filename"] == "notes.pdf"


def test_load_openwebui_chat_file_rows_for_chats_filters_by_chat_and_user(tmp_path):
    db_path = write_openwebui_hydration_db(tmp_path / "webui.db")

    with open_row_connection(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO chat_file (id, chat_id, file_id, message_id, user_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("link-a", "chat-a", "file-a", "message-a", "owui-user", 1, 2),
                ("link-b", "chat-b", "file-b", "message-b", "owui-user", 3, 4),
                ("link-c", "chat-a", "file-c", "message-c", "other-user", 5, 6),
            ],
        )
        conn.commit()

        rows = openwebui_db_reader.load_openwebui_chat_file_rows_for_chats(
            conn,
            ["chat-a"],
            user_id="owui-user",
        )

    assert [row["id"] for row in rows] == ["link-a"]
    assert rows[0]["file_id"] == "file-a"


def test_hydration_row_helpers_treat_quoted_ids_as_literal_values(tmp_path):
    db_path = write_openwebui_hydration_db(tmp_path / "webui.db")
    quoted_file_id = "file-a' OR 1=1 --"
    quoted_chat_id = "chat-a' OR 1=1 --"

    with open_row_connection(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO file (id, user_id, hash, filename, path, data, meta, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (quoted_file_id, "owui-user", "hash-a", "quoted.txt", "uploads/quoted.txt", "{}", "{}", 1, 2),
                ("file-b", "owui-user", "hash-b", "other.txt", "uploads/other.txt", "{}", "{}", 3, 4),
            ],
        )
        conn.executemany(
            """
            INSERT INTO chat_file (id, chat_id, file_id, message_id, user_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("link-a", quoted_chat_id, quoted_file_id, "message-a", "owui-user", 1, 2),
                ("link-b", "chat-b", "file-b", "message-b", "owui-user", 3, 4),
            ],
        )
        conn.commit()

        file_rows = openwebui_db_reader.load_openwebui_file_rows_for_ids(
            conn,
            [quoted_file_id],
            user_id="owui-user",
        )
        chat_file_rows = openwebui_db_reader.load_openwebui_chat_file_rows_for_chats(
            conn,
            [quoted_chat_id],
            user_id="owui-user",
        )

    assert [row["id"] for row in file_rows] == [quoted_file_id]
    assert [row["id"] for row in chat_file_rows] == ["link-a"]
