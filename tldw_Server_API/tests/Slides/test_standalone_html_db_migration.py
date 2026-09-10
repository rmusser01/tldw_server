import json
import sqlite3
import threading

import pytest

from tldw_Server_API.app.core.Slides import slides_migrations
from tldw_Server_API.app.core.Slides.slides_db import SchemaError, SlidesDatabase


def _create_legacy_database(
    db_path,
    *,
    version_rows: tuple[int, ...] = (1,),
    with_presentation: bool = False,
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY NOT NULL)")
        conn.executemany(
            "INSERT INTO schema_version (version) VALUES (?)",
            ((version,) for version in version_rows),
        )
        conn.execute(
            """
            CREATE TABLE presentations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                theme TEXT DEFAULT 'black',
                marp_theme TEXT,
                template_id TEXT,
                visual_style_id TEXT,
                visual_style_scope TEXT,
                visual_style_name TEXT,
                visual_style_version INTEGER,
                visual_style_snapshot TEXT,
                settings TEXT,
                studio_data TEXT,
                slides TEXT NOT NULL,
                slides_text TEXT NOT NULL,
                source_type TEXT,
                source_ref TEXT,
                source_query TEXT,
                custom_css TEXT,
                created_at DATETIME NOT NULL,
                last_modified DATETIME NOT NULL,
                deleted INTEGER DEFAULT 0,
                client_id TEXT NOT NULL,
                version INTEGER DEFAULT 1
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE presentations_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                presentation_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                client_id TEXT NOT NULL
            )
            """
        )
        if with_presentation:
            conn.execute(
                """
                INSERT INTO presentations (
                    id, title, slides, slides_text, created_at, last_modified,
                    client_id
                ) VALUES ('legacy', 'Legacy', '[]', '', '2026-01-01',
                          '2026-01-01', 'legacy-client')
                """
            )


def _schema_snapshot(db_path) -> tuple[list[int], set[str], set[str], set[str]]:
    with sqlite3.connect(db_path) as conn:
        versions = [row[0] for row in conn.execute("SELECT version FROM schema_version")]
        columns = {row[1] for row in conn.execute("PRAGMA table_info(presentations)")}
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        indexes = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
    return versions, columns, tables, indexes


def _structural_snapshot(db_path) -> tuple[int, tuple[tuple, ...], tuple[int, ...]]:
    with sqlite3.connect(db_path) as conn:
        schema_version = int(conn.execute("PRAGMA schema_version").fetchone()[0])
        objects = tuple(
            conn.execute(
                """
                SELECT type, name, tbl_name, COALESCE(sql, '')
                FROM sqlite_master
                WHERE name NOT LIKE 'sqlite_%'
                ORDER BY type, name
                """
            )
        )
        versions = tuple(row[0] for row in conn.execute("SELECT version FROM schema_version ORDER BY version"))
    return schema_version, objects, versions


def _version_columns(db_path) -> set[str]:
    with sqlite3.connect(db_path) as conn:
        return {row[1] for row in conn.execute("PRAGMA table_info(presentations_versions)")}


def _create_incomplete_v2_version_metadata_database(db_path) -> None:
    db = SlidesDatabase(db_path=db_path, client_id="legacy-v2")
    created = db.create_presentation(
        presentation_id="legacy-v2",
        title="Original title",
        description=None,
        theme="black",
        marp_theme=None,
        settings=None,
        studio_data=None,
        slides="[]",
        slides_text="",
        source_type="manual",
        source_ref=None,
        source_query=None,
        custom_css=None,
    )
    renamed = db.update_presentation(
        presentation_id=created.id,
        update_fields={"title": "Renamed title"},
        expected_version=created.version,
    )
    db.soft_delete_presentation(created.id, renamed.version)
    db.close_connection()
    with sqlite3.connect(db_path) as conn:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(presentations_versions)")}
        for column in ("title", "deleted"):
            if column in existing:
                conn.execute(f"ALTER TABLE presentations_versions DROP COLUMN {column}")


def test_new_database_is_created_at_schema_v2(tmp_path):
    db_path = tmp_path / "Slides.db"
    db = SlidesDatabase(db_path=db_path, client_id="tester")
    db.close_connection()

    versions, columns, tables, indexes = _schema_snapshot(db_path)

    assert versions == [2]
    assert {
        "content_kind",
        "html_document",
        "html_sha256",
        "html_bytes",
        "html_slide_count",
        "generation_job_uuid",
        "generation_provenance_json",
    }.issubset(columns)
    assert {"slides_generation_receipts", "slides_generation_inputs"}.issubset(tables)
    assert "idx_presentations_generation_job_uuid" in indexes
    assert {"title", "deleted"}.issubset(_version_columns(db_path))
    with sqlite3.connect(db_path) as conn:
        triggers = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")}
    assert {"presentations_ai", "presentations_ad", "presentations_au"}.issubset(triggers)


def test_future_schema_is_rejected_without_structural_mutation(tmp_path):
    db_path = tmp_path / "Slides.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY NOT NULL)")
        conn.execute("INSERT INTO schema_version (version) VALUES (3)")
        conn.execute(
            """
            CREATE TABLE presentations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                slides_text TEXT NOT NULL,
                deleted INTEGER NOT NULL DEFAULT 0,
                created_at DATETIME NOT NULL
            )
            """
        )
        conn.execute("CREATE TABLE future_only (sentinel TEXT NOT NULL)")
    before = _structural_snapshot(db_path)

    with pytest.raises(SchemaError, match="Unsupported Slides schema versions"):
        SlidesDatabase(db_path=db_path, client_id="tester")

    assert _structural_snapshot(db_path) == before


@pytest.mark.parametrize("schema_version", [0, 1])
def test_legacy_schema_upgrades_and_backfills_structured_rows(tmp_path, schema_version):
    db_path = tmp_path / "Slides.db"
    _create_legacy_database(
        db_path,
        version_rows=(schema_version,),
        with_presentation=True,
    )

    db = SlidesDatabase(db_path=db_path, client_id="tester")
    row = db.get_presentation_by_id("legacy")
    db.close_connection()

    with sqlite3.connect(db_path) as conn:
        fts_counts = (
            conn.execute("SELECT COUNT(*) FROM presentations").fetchone()[0],
            conn.execute("SELECT COUNT(*) FROM presentations_fts_docsize").fetchone()[0],
        )

    assert row.content_kind == "structured_slides"
    assert row.html_document is None
    assert row.generation_job_uuid is None
    assert fts_counts == (1, 1)
    assert _schema_snapshot(db_path)[0] == [2]


@pytest.mark.parametrize("version_rows", [(), (0, 1)])
def test_migration_normalizes_empty_or_multiple_version_rows(tmp_path, version_rows):
    db_path = tmp_path / "Slides.db"
    _create_legacy_database(db_path, version_rows=version_rows)

    db = SlidesDatabase(db_path=db_path, client_id="tester")
    db.close_connection()

    assert _schema_snapshot(db_path)[0] == [2]


def test_schema_v2_reopen_is_idempotent(tmp_path):
    db_path = tmp_path / "Slides.db"
    first = SlidesDatabase(db_path=db_path, client_id="first")
    first.close_connection()

    second = SlidesDatabase(db_path=db_path, client_id="second")
    second.close_connection()

    versions, columns, tables, _ = _schema_snapshot(db_path)
    assert versions == [2]
    assert len(columns) == len(set(columns))
    assert {"slides_generation_receipts", "slides_generation_inputs"}.issubset(tables)
    assert {"title", "deleted"}.issubset(_version_columns(db_path))


def test_incomplete_v2_backfills_version_title_and_deleted_idempotently(tmp_path):
    db_path = tmp_path / "Slides.db"
    _create_incomplete_v2_version_metadata_database(db_path)

    migrated = SlidesDatabase(db_path=db_path, client_id="migrated")
    first_rows, first_total = migrated.list_presentation_version_metadata(
        presentation_id="legacy-v2",
        limit=10,
        offset=0,
    )
    migrated.close_connection()
    reopened = SlidesDatabase(db_path=db_path, client_id="reopened")
    second_rows, second_total = reopened.list_presentation_version_metadata(
        presentation_id="legacy-v2",
        limit=10,
        offset=0,
    )
    reopened.close_connection()

    expected = [
        (3, "Renamed title", 1),
        (2, "Renamed title", 0),
        (1, "Original title", 0),
    ]
    assert first_total == second_total == 3
    assert [(row.version, row.title, row.deleted) for row in first_rows] == expected
    assert [(row.version, row.title, row.deleted) for row in second_rows] == expected
    assert {"title", "deleted"}.issubset(_version_columns(db_path))


def test_incomplete_v2_malformed_snapshot_backfills_null_safe_metadata(tmp_path):
    db_path = tmp_path / "Slides.db"
    _create_incomplete_v2_version_metadata_database(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE presentations_versions SET payload_json = '{' "
            "WHERE presentation_id = 'legacy-v2' AND version = 1"
        )
        conn.execute(
            "UPDATE presentations_versions SET payload_json = ? " "WHERE presentation_id = 'legacy-v2' AND version = 2",
            (
                json.dumps(
                    {
                        "title": {"html_document": "SECRET-NESTED-SOURCE"},
                        "deleted": 2,
                    }
                ),
            ),
        )
        conn.execute(
            "UPDATE presentations_versions SET payload_json = ? " "WHERE presentation_id = 'legacy-v2' AND version = 3",
            (json.dumps({"title": ["SECRET-ARRAY"], "deleted": "0"}),),
        )

    db = SlidesDatabase(db_path=db_path, client_id="migrated")
    rows, total = db.list_presentation_version_metadata(
        presentation_id="legacy-v2",
        limit=10,
        offset=0,
    )
    db.close_connection()

    assert total == 3
    assert {row.version: (row.title, row.deleted) for row in rows} == {
        1: (None, None),
        2: (None, None),
        3: (None, None),
    }


def test_incomplete_v2_version_metadata_migration_rolls_back_atomically(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "Slides.db"
    _create_incomplete_v2_version_metadata_database(db_path)
    original_execute = slides_migrations._execute_migration_statement

    def fail_on_deleted_column(conn, statement, parameters=()):
        if "presentations_versions add column deleted" in statement.lower():
            raise sqlite3.OperationalError("injected version metadata failure")
        return original_execute(conn, statement, parameters)

    monkeypatch.setattr(
        slides_migrations,
        "_execute_migration_statement",
        fail_on_deleted_column,
    )

    with pytest.raises(SchemaError, match="injected version metadata failure"):
        SlidesDatabase(db_path=db_path, client_id="migrated")

    assert not {"title", "deleted"}.intersection(_version_columns(db_path))


def test_concurrent_connections_complete_incomplete_v2_version_metadata(tmp_path):
    db_path = tmp_path / "Slides.db"
    _create_incomplete_v2_version_metadata_database(db_path)
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def migrate() -> None:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            barrier.wait(timeout=5)
            slides_migrations.migrate_slides_schema(conn)
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - thread reports failures
            errors.append(exc)
        finally:
            conn.close()

    threads = [threading.Thread(target=migrate) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert {"title", "deleted"}.issubset(_version_columns(db_path))


def test_complete_v2_reopen_is_read_only_under_competing_writer(tmp_path):
    db_path = tmp_path / "Slides.db"
    first = SlidesDatabase(db_path=db_path, client_id="first")
    first.close_connection()

    observer = sqlite3.connect(db_path)
    writer = sqlite3.connect(db_path)
    before_data_version = int(observer.execute("PRAGMA data_version").fetchone()[0])
    writer.execute("BEGIN IMMEDIATE")
    try:
        try:
            reopened = SlidesDatabase(db_path=db_path, client_id="second")
        except SchemaError as exc:
            pytest.fail(f"normalized v2 reopen attempted a write lock: {exc}")
        reopened.close_connection()
        after_data_version = int(observer.execute("PRAGMA data_version").fetchone()[0])
    finally:
        writer.rollback()
        writer.close()
        observer.close()

    assert after_data_version == before_data_version


def test_statement_failure_rolls_back_entire_v2_migration(tmp_path, monkeypatch):
    db_path = tmp_path / "Slides.db"
    _create_legacy_database(db_path, version_rows=(1,), with_presentation=True)
    original_execute = slides_migrations._execute_migration_statement
    calls = 0

    def fail_after_first_statement(conn, statement, parameters=()):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise sqlite3.OperationalError("injected migration failure")
        return original_execute(conn, statement, parameters)

    monkeypatch.setattr(
        slides_migrations,
        "_execute_migration_statement",
        fail_after_first_statement,
    )

    with pytest.raises(SchemaError, match="injected migration failure"):
        SlidesDatabase(db_path=db_path, client_id="tester")

    versions, columns, tables, indexes = _schema_snapshot(db_path)
    assert versions == [1]
    assert "content_kind" not in columns
    assert "slides_generation_receipts" not in tables
    assert "slides_generation_inputs" not in tables
    assert "idx_presentations_generation_job_uuid" not in indexes


def test_concurrent_connections_can_migrate_the_same_database(tmp_path):
    db_path = tmp_path / "Slides.db"
    _create_legacy_database(db_path, version_rows=(1,), with_presentation=True)
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def migrate() -> None:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            barrier.wait(timeout=5)
            slides_migrations.migrate_slides_schema(conn)
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - thread reports failures
            errors.append(exc)
        finally:
            conn.close()

    threads = [threading.Thread(target=migrate) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert _schema_snapshot(db_path)[0] == [2]


def test_migration_runner_never_uses_executescript(tmp_path):
    db_path = tmp_path / "Slides.db"
    _create_legacy_database(db_path, version_rows=(1,))

    class NoScriptConnection(sqlite3.Connection):
        def executescript(self, _script):  # pragma: no cover - failure is the assertion
            raise AssertionError("migration must execute statements individually")

    conn = sqlite3.connect(db_path, factory=NoScriptConnection)
    conn.row_factory = sqlite3.Row
    try:
        slides_migrations.migrate_slides_schema(conn)
    finally:
        conn.close()

    assert _schema_snapshot(db_path)[0] == [2]
