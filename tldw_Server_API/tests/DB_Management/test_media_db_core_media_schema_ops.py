import importlib
import importlib.util
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError, SchemaError


pytestmark = pytest.mark.unit


EXPECTED_MEDIA_COLUMNS = {
    "id",
    "url",
    "title",
    "type",
    "content",
    "author",
    "ingestion_date",
    "transcription_model",
    "is_trash",
    "trash_date",
    "vector_embedding",
    "chunking_status",
    "vector_processing",
    "content_hash",
    "source_hash",
    "uuid",
    "last_modified",
    "version",
    "org_id",
    "team_id",
    "visibility",
    "owner_user_id",
    "latest_transcription_run_id",
    "next_transcription_run_id",
    "client_id",
    "deleted",
    "prev_version",
    "merge_parent_uuid",
}


def _load_core_media_module():
    module_name = (
        "tldw_Server_API.app.core.DB_Management.media_db.schema.features.core_media"
    )
    spec = importlib.util.find_spec(module_name)
    assert spec is not None, f"Expected schema helper module spec for {module_name}"
    return importlib.import_module(module_name)


class _FakeCursor:
    def __init__(self, *, rows=None, row=None):
        self._rows = rows if rows is not None else []
        self._row = row

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._row


def test_core_media_helper_rebinds_on_media_database() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.media_database import MediaDatabase

    helper_module = _load_core_media_module()

    assert MediaDatabase.__dict__["_apply_schema_v1_sqlite"].__globals__["__name__"] == (
        helper_module.__name__
    )
    assert MediaDatabase.__dict__["_apply_schema_v1_postgres"].__globals__["__name__"] == (
        helper_module.__name__
    )


def test_apply_sqlite_core_media_schema_runs_email_ensure_after_schema_script() -> None:
    helper_module = _load_core_media_module()

    calls: list[tuple[str, object]] = []
    queries: list[str] = []

    class FakeConn:
        def executescript(self, script: str):
            calls.append(("executescript", script))

        def execute(self, query: str):
            queries.append(query)
            if query == "PRAGMA table_info(Media)":
                return _FakeCursor(rows=[{"name": name} for name in EXPECTED_MEDIA_COLUMNS])
            if query == "SELECT version FROM schema_version LIMIT 1":
                return _FakeCursor(row={"version": 22})
            raise AssertionError(f"unexpected query: {query}")

    conn = FakeConn()
    db = SimpleNamespace(
        db_path_str="core-media.sqlite",
        _TABLES_SQL_V1="tables sql",
        _INDICES_SQL_V1="indices sql",
        _TRIGGERS_SQL_V1="triggers sql",
        _SCHEMA_UPDATE_VERSION_SQL_V1="schema update sql",
        _CLAIMS_TABLE_SQL="claims sql",
        _MEDIA_FILES_TABLE_SQL="media files sql",
        _TTS_HISTORY_TABLE_SQL="tts history sql",
        _DATA_TABLES_SQL="data tables sql",
        _CURRENT_SCHEMA_VERSION=22,
        _ensure_sqlite_email_schema=lambda value: calls.append(("email", value)),
        _ensure_fts_structures=lambda value: calls.append(("fts", value)),
    )

    helper_module.apply_sqlite_core_media_schema(db, conn)

    assert calls[0][0] == "executescript"
    assert calls[1:] == [("email", conn), ("fts", conn)]
    assert queries == [
        "PRAGMA table_info(Media)",
        "SELECT version FROM schema_version LIMIT 1",
    ]


def test_apply_sqlite_core_media_schema_raises_on_media_validation_failure() -> None:
    helper_module = _load_core_media_module()

    class FakeConn:
        def executescript(self, script: str):
            return None

        def execute(self, query: str):
            if query == "PRAGMA table_info(Media)":
                return _FakeCursor(rows=[{"name": "id"}])
            if query == "SELECT version FROM schema_version LIMIT 1":
                return _FakeCursor(row={"version": 22})
            raise AssertionError(f"unexpected query: {query}")

    db = SimpleNamespace(
        db_path_str="core-media.sqlite",
        _TABLES_SQL_V1="tables sql",
        _INDICES_SQL_V1="indices sql",
        _TRIGGERS_SQL_V1="triggers sql",
        _SCHEMA_UPDATE_VERSION_SQL_V1="schema update sql",
        _CLAIMS_TABLE_SQL="claims sql",
        _MEDIA_FILES_TABLE_SQL="media files sql",
        _TTS_HISTORY_TABLE_SQL="tts history sql",
        _DATA_TABLES_SQL="data tables sql",
        _CURRENT_SCHEMA_VERSION=22,
        _ensure_sqlite_email_schema=lambda conn: None,
        _ensure_fts_structures=lambda conn: None,
    )

    with pytest.raises(SchemaError, match="Media table is missing columns"):
        helper_module.apply_sqlite_core_media_schema(db, FakeConn())


def test_apply_sqlite_core_media_schema_fts_failure_is_warning_only() -> None:
    helper_module = _load_core_media_module()

    calls: list[tuple[str, object]] = []

    class FakeConn:
        def executescript(self, script: str):
            calls.append(("executescript", script))

        def execute(self, query: str):
            if query == "PRAGMA table_info(Media)":
                return _FakeCursor(rows=[{"name": name} for name in EXPECTED_MEDIA_COLUMNS])
            if query == "SELECT version FROM schema_version LIMIT 1":
                return _FakeCursor(row={"version": 22})
            raise AssertionError(f"unexpected query: {query}")

    conn = FakeConn()
    db = SimpleNamespace(
        db_path_str="core-media.sqlite",
        _TABLES_SQL_V1="tables sql",
        _INDICES_SQL_V1="indices sql",
        _TRIGGERS_SQL_V1="triggers sql",
        _SCHEMA_UPDATE_VERSION_SQL_V1="schema update sql",
        _CLAIMS_TABLE_SQL="claims sql",
        _MEDIA_FILES_TABLE_SQL="media files sql",
        _TTS_HISTORY_TABLE_SQL="tts history sql",
        _DATA_TABLES_SQL="data tables sql",
        _CURRENT_SCHEMA_VERSION=22,
        _ensure_sqlite_email_schema=lambda value: calls.append(("email", value)),
        _ensure_fts_structures=lambda value: (
            calls.append(("fts-attempt", value)),
            (_ for _ in ()).throw(DatabaseError("fts boom")),
        )[1],
    )

    helper_module.apply_sqlite_core_media_schema(db, conn)

    assert calls[0][0] == "executescript"
    assert calls[1:] == [("email", conn), ("fts-attempt", conn)]


def test_apply_postgres_core_media_schema_orders_base_tables_then_initializers_then_email_and_schema_updates() -> None:
    helper_module = _load_core_media_module()

    conn = object()
    calls: list[tuple[object, ...]] = []
    table_checks: list[str] = []

    class FakeBackend:
        def execute(self, query: str, params=None, *, connection) -> None:
            calls.append(("execute", query, params))

        def table_exists(self, table: str, *, connection) -> bool:
            table_checks.append(table)
            return True

    def convert(sql: str) -> list[str]:
        mapping = {
            "tables": ["CREATE TABLE media (...)", "INSERT INTO schema_version VALUES (0)"],
            "claims": ["CREATE TABLE claims (...)"],
            "mediafiles": ["CREATE TABLE mediafiles (...)"],
            "tts": ["CREATE TABLE tts_history (...)"],
            "datatable": ["CREATE TABLE data_tables (...)"],
            "indices": ["CREATE INDEX idx_media_title ON media(title)"],
        }
        return mapping[sql]

    db = SimpleNamespace(
        _TABLES_SQL_V1="tables",
        _CLAIMS_TABLE_SQL="claims",
        _MEDIA_FILES_TABLE_SQL="mediafiles",
        _TTS_HISTORY_TABLE_SQL="tts",
        _DATA_TABLES_SQL="datatable",
        _INDICES_SQL_V1="indices",
        _CURRENT_SCHEMA_VERSION=22,
        backend=FakeBackend(),
        _convert_sqlite_sql_to_postgres_statements=convert,
        _ensure_postgres_email_schema=lambda value: calls.append(("email", value)),
    )

    helper_module.apply_postgres_core_media_schema(db, conn)

    assert calls == [
        ("execute", "CREATE TABLE media (...)", None),
        ("execute", "CREATE TABLE claims (...)", None),
        ("execute", "CREATE TABLE mediafiles (...)", None),
        ("execute", "CREATE TABLE tts_history (...)", None),
        ("execute", "CREATE TABLE data_tables (...)", None),
        ("execute", "INSERT INTO schema_version VALUES (0)", None),
        ("execute", "CREATE INDEX idx_media_title ON media(title)", None),
        ("email", conn),
        ("execute", "DELETE FROM schema_version WHERE version <> %s", (0,)),
        (
            "execute",
            "INSERT INTO schema_version (version) VALUES (%s) ON CONFLICT (version) DO NOTHING",
            (0,),
        ),
        ("execute", "UPDATE schema_version SET version = %s", (db._CURRENT_SCHEMA_VERSION,)),
    ]
    assert table_checks == [
        "media",
        "keywords",
        "mediakeywords",
        "transcripts",
        "mediachunks",
        "unvectorizedmediachunks",
        "documentversions",
        "documentversionidentifiers",
        "documentstructureindex",
        "sync_log",
        "chunkingtemplates",
        "claims",
    ]


def test_apply_postgres_core_media_schema_raises_when_critical_table_missing() -> None:
    helper_module = _load_core_media_module()

    conn = object()
    calls: list[tuple[object, ...]] = []
    table_checks: list[str] = []

    class FakeBackend:
        def execute(self, query: str, params=None, *, connection) -> None:
            calls.append(("execute", query, params))

        def table_exists(self, table: str, *, connection) -> bool:
            table_checks.append(table)
            return table != "claims"

    db = SimpleNamespace(
        _TABLES_SQL_V1="tables",
        _CLAIMS_TABLE_SQL="claims",
        _MEDIA_FILES_TABLE_SQL="mediafiles",
        _TTS_HISTORY_TABLE_SQL="tts",
        _DATA_TABLES_SQL="datatable",
        _INDICES_SQL_V1="indices",
        _CURRENT_SCHEMA_VERSION=22,
        backend=FakeBackend(),
        _convert_sqlite_sql_to_postgres_statements=lambda sql: ["CREATE TABLE x (...)"],
        _ensure_postgres_email_schema=lambda value: calls.append(("email", value)),
    )

    with pytest.raises(SchemaError, match="Postgres schema init missing table: claims"):
        helper_module.apply_postgres_core_media_schema(db, conn)

    assert table_checks[-1] == "claims"
    assert ("email", conn) not in calls
    assert not any(
        query in {
            "CREATE INDEX idx_media_title ON media(title)",
            "DELETE FROM schema_version WHERE version <> %s",
            "INSERT INTO schema_version (version) VALUES (%s) ON CONFLICT (version) DO NOTHING",
            "UPDATE schema_version SET version = %s",
        }
        for _, query, _ in calls
    )
