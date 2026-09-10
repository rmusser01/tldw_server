"""Required import tables cannot execute uploaded virtual-table/view queries."""

import sqlite3
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management import OpenWebUI_DB as reader

pytestmark = pytest.mark.unit
SCHEMAS = reader.REQUIRED_SCHEMA | reader.HYDRATION_FILE_SCHEMA | reader.HYDRATION_CHAT_FILE_SCHEMA


def _database(path: Path, replaced: str | None = None, prefix: str = "", suffix: str = "") -> None:
    """Create ordinary fixture tables, optionally replacing one with FTS5."""
    with sqlite3.connect(path) as conn:
        for name, columns in SCHEMAS.items():
            if name == replaced:
                # FTS reserves the table name as a hidden column (notably chat).
                fts_columns = sorted(columns - {name})
                sql = f'{prefix} "{name}" USING fts5({", ".join(fts_columns)})'
            else:
                definitions = [
                    f'"{column}" TEXT' + (" PRIMARY KEY" if column == "id" else "") for column in sorted(columns)
                ]
                sql = f'CREATE TABLE "{name}" ({", ".join(definitions)}) {suffix}'
            conn.execute(sql)


@pytest.mark.parametrize("table", list(SCHEMAS))
@pytest.mark.parametrize("prefix", ["CREATE VIRTUAL TABLE", "cReAtE /* ordinary-looking */ vIrTuAl\nTaBlE"])
def test_required_table_cannot_be_virtual(tmp_path: Path, table: str, prefix: str) -> None:
    path = tmp_path / "source.db"
    _database(path, table, prefix)
    with pytest.raises(ValueError, match="missing required OpenWebUI table"):
        with reader.open_validated_openwebui_db(path) as conn:
            reader.validate_openwebui_file_schema(conn)


def test_external_content_view_cannot_run_match_during_user_import(tmp_path: Path) -> None:
    path = tmp_path / "source.db"
    _database(path)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            ALTER TABLE user RENAME TO actual_user;
            INSERT INTO actual_user VALUES('1', 'example.invalid', 'known-user', 'Example', '1');
            CREATE VIRTUAL TABLE indexed_text USING fts5(body);
            INSERT INTO indexed_text VALUES('ordinary text');
            CREATE VIEW user_content AS
                SELECT rowid, id, name, email, created_at, updated_at FROM actual_user
                WHERE EXISTS(SELECT 1 FROM indexed_text WHERE indexed_text MATCH 'ordinary');
            CREATE VIRTUAL TABLE user USING fts5(
                id, name, email, created_at, updated_at,
                content='user_content', content_rowid='rowid'
            );
            """
        )
    with pytest.raises(ValueError, match="missing required OpenWebUI table: user"):
        with reader.open_validated_openwebui_db(path) as conn:
            reader.load_openwebui_users(conn)


@pytest.mark.parametrize("suffix", ["", "WITHOUT ROWID", "STRICT"])
def test_ordinary_table_variants_remain_accepted(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / "source.db"
    _database(path, suffix=suffix)
    with reader.open_validated_openwebui_db(path) as conn:
        reader.validate_openwebui_file_schema(conn)
        assert reader.load_openwebui_users(conn) == []


def test_missing_native_table_metadata_fails_closed(tmp_path: Path) -> None:
    class OldMetadataConnection(sqlite3.Connection):
        def execute(self, sql: str, parameters: tuple = ()) -> sqlite3.Cursor:
            if sql == "PRAGMA main.table_list":
                return super().execute("SELECT 1 WHERE 0")
            return super().execute(sql, parameters)

    path = tmp_path / "source.db"
    _database(path)
    with sqlite3.connect(path, factory=OldMetadataConnection) as conn:
        conn.row_factory = sqlite3.Row
        with pytest.raises(ValueError, match="SQLite 3.37 or newer"):
            reader.validate_openwebui_schema(conn)
