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


@pytest.mark.parametrize("suffix", ["", "WITHOUT ROWID", "STRICT"])
def test_metadata_cannot_evaluate_unrelated_fts_config_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    """Missing connection read restrictions lets metadata execute the source view."""
    path = tmp_path / "source.db"
    _database(path, suffix=suffix)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            INSERT INTO user(id, name, email) VALUES('u1', 'Example', 'example.invalid');
            CREATE VIRTUAL TABLE indexed_text USING fts5(body);
            INSERT INTO indexed_text VALUES('ordinary text');
            CREATE VIRTUAL TABLE auxiliary USING fts5(body);
            ALTER TABLE auxiliary_config RENAME TO saved_config;
            CREATE VIEW auxiliary_config AS
                SELECT k, observed_config(v) AS v FROM saved_config
                WHERE EXISTS(SELECT 1 FROM indexed_text WHERE indexed_text MATCH 'ordinary');
            """
        )
    evaluated: list[int] = []
    original_connect = sqlite3.connect

    def observed_connect(*args, **kwargs):
        conn = original_connect(*args, **kwargs)
        conn.create_function("observed_config", 1, lambda value: evaluated.append(value) or value)
        return conn

    monkeypatch.setattr(sqlite3, "connect", observed_connect)
    with reader.open_validated_openwebui_db(path) as conn:
        reader.validate_openwebui_file_schema(conn)
        users = reader.load_openwebui_users(conn)
    assert evaluated == []
    assert [row["id"] for row in users] == ["u1"]


def test_unrelated_fts_does_not_prevent_ordinary_hydration_reads(tmp_path: Path) -> None:
    """A metadata guard must preserve the reader's ordinary and JSON helper reads."""
    path = tmp_path / "source.db"
    _database(path)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE VIRTUAL TABLE unrelated USING fts5(body);
            INSERT INTO unrelated VALUES('ordinary text');
            INSERT INTO file(id, user_id, filename) VALUES('f1', 'u1', 'example.txt');
            INSERT INTO chat_file(id, chat_id, file_id, user_id) VALUES('cf1', 'c1', 'f1', 'u1');
            """
        )
    with reader.open_validated_openwebui_db(path) as conn:
        reader.validate_openwebui_file_schema(conn)
        files = reader.load_openwebui_file_rows_for_ids(conn, ["f1"], "u1")
        links = reader.load_openwebui_chat_file_rows_for_chats(conn, ["c1"], "u1")
    assert [row["filename"] for row in files] == ["example.txt"]
    assert [row["file_id"] for row in links] == ["f1"]


def test_persisted_json_each_cannot_read_fts_pages(tmp_path: Path) -> None:
    """Allowing built-in JSON reads must not allow a source FTS object of that name."""
    path = tmp_path / "source.db"
    _database(path)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE VIRTUAL TABLE json_each USING fts5(value);
            INSERT INTO json_each VALUES('ordinary');
            INSERT INTO file(id, user_id) VALUES('ordinary', 'u1');
            """
        )
    with reader.open_validated_openwebui_db(path) as conn:
        reader.validate_openwebui_file_schema(conn)
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("SELECT rowid FROM json_each WHERE json_each MATCH 'ordinary'").fetchall()


def test_nested_config_views_cannot_search_fts_during_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Nested constructors can bypass the read authorizer while preparing a view."""
    path = tmp_path / "source.db"
    _database(path)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            INSERT INTO user(id, name) VALUES('u1', 'Example');
            CREATE VIRTUAL TABLE seed USING fts5(body);
            INSERT INTO seed VALUES('ordinary text');
            CREATE VIRTUAL TABLE indexed_text USING fts5(body);
            INSERT INTO indexed_text VALUES('ordinary text');
            ALTER TABLE indexed_text_config RENAME TO saved_indexed_config;
            CREATE VIEW indexed_text_config AS SELECT k, v FROM saved_indexed_config
                WHERE EXISTS(SELECT 1 FROM seed WHERE seed MATCH 'ordinary');
            CREATE VIRTUAL TABLE auxiliary USING fts5(body);
            ALTER TABLE auxiliary_config RENAME TO saved_config;
            CREATE VIEW auxiliary_config AS SELECT k, v FROM saved_config
                WHERE EXISTS(SELECT 1 FROM indexed_text WHERE indexed_text MATCH 'ordinary');
            """
        )
    statements: list[str] = []
    original_connect = sqlite3.connect

    def traced_connect(*args, **kwargs):
        conn = original_connect(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(sqlite3, "connect", traced_connect)
    with reader.open_validated_openwebui_db(path) as conn:
        reader.validate_openwebui_file_schema(conn)
        users = reader.load_openwebui_users(conn)
    assert not any("seed_idx" in statement for statement in statements)
    assert [row["id"] for row in users] == ["u1"]


@pytest.mark.parametrize("failure", ["error", "ignored", "unavailable"])
def test_source_closes_if_schema_trust_cannot_be_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """A rejected or unsupported safety setting must not permit an import."""
    path = tmp_path / "source.db"
    _database(path)
    original_connect = sqlite3.connect
    connections: list[sqlite3.Connection] = []

    class UntrustedSchemaUnavailableConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if sql == "PRAGMA trusted_schema = OFF":
                if failure == "error":
                    raise sqlite3.DatabaseError("schema trust setting unavailable")
                if failure == "ignored":
                    return super().execute("SELECT 1 WHERE 0")
            if sql == "PRAGMA trusted_schema" and failure == "unavailable":
                return super().execute("SELECT 1 WHERE 0")
            return super().execute(sql, parameters)

    def unavailable_connect(*args, **kwargs):
        conn = original_connect(*args, **kwargs, factory=UntrustedSchemaUnavailableConnection)
        connections.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", unavailable_connect)
    with pytest.raises(ValueError, match="Invalid OpenWebUI SQLite database"):
        with reader.open_validated_openwebui_db(path):
            pytest.fail("Import must not expose a connection with trusted schema enabled")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")
