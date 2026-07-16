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
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
    return versions, columns, tables, indexes


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
    SlidesDatabase._schema_init_paths.discard(str(db_path.resolve()))

    second = SlidesDatabase(db_path=db_path, client_id="second")
    second.close_connection()

    versions, columns, tables, _ = _schema_snapshot(db_path)
    assert versions == [2]
    assert len(columns) == len(set(columns))
    assert {"slides_generation_receipts", "slides_generation_inputs"}.issubset(tables)


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
        except BaseException as exc:  # pragma: no cover - assertion reports details
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
