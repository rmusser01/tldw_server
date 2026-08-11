"""Schema v58 coverage for canonical Notes links and graph projections."""

from __future__ import annotations

import inspect
import json
import sqlite3
from pathlib import Path
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, QueryResult
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    SchemaError,
)

pytestmark = pytest.mark.unit


def test_graph_update_triggers_name_only_graph_relevant_columns() -> None:
    sqlite_source = " ".join(inspect.getsource(CharactersRAGDB._notes_graph_schema_sqlite).split())
    postgres_statements: list[str] = []

    class _RecordingBackend:
        def execute(self, query: str, *, connection: object) -> None:
            assert connection is postgres_connection
            postgres_statements.append(query)

    class _RecordingDatabase:
        backend = _RecordingBackend()

    postgres_connection = object()
    CharactersRAGDB._notes_graph_schema_postgres(
        _RecordingDatabase(),  # type: ignore[arg-type]
        postgres_connection,
    )
    postgres_sql = " ".join(" ".join(query.split()) for query in postgres_statements)

    note_columns = "title, content, conversation_id, created_at, last_modified, deleted"
    assert f"AFTER UPDATE OF {note_columns} ON notes" in sqlite_source
    assert f"UPDATE OF {note_columns} ON notes" in postgres_sql
    assert "AFTER UPDATE OF keyword, deleted ON keywords" in sqlite_source
    assert "UPDATE OF keyword, deleted ON chacha_keywords" in postgres_sql


def test_postgres_v58_canonical_rows_are_updated_in_one_set_based_statement() -> None:
    source = inspect.getsource(CharactersRAGDB._migrate_from_v57_to_v58_postgres)

    assert "jsonb_to_recordset" in source
    assert "for row in canonical_rows:\n            backend.execute" not in source


OWNER = "owner-notes-link"
CREATED_AT = "2026-08-10T12:00:00+00:00"


def _replace_with_v57_edge_table(
    db_path: Path,
    *,
    edge_rows: list[tuple[object, ...]],
    extra_notes: list[tuple[str, str]] | None = None,
) -> None:
    """Create a genuine legacy edge table while retaining the current base schema."""

    source_id = "11111111-1111-4111-8111-111111111111"
    target_id = "22222222-2222-4222-8222-222222222222"
    db = CharactersRAGDB(str(db_path), client_id=OWNER)
    try:
        for note_id, title in [(source_id, "Source"), (target_id, "Target")]:
            if db.get_note_by_id(note_id) is None:
                db.add_note(title=title, content="Body", note_id=note_id)
        for note_id, note_owner in extra_notes or []:
            if db.get_note_by_id(note_id) is None:
                db.add_note(title=note_id, content="Body", note_id=note_id)
            with db.transaction() as conn:
                conn.execute("UPDATE notes SET client_id = ? WHERE id = ?", (note_owner, note_id))
    finally:
        db.close_connection()

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("DROP TABLE note_edges")
        conn.execute(
            """
            CREATE TABLE note_edges(
              edge_id TEXT PRIMARY KEY,
              user_id TEXT NOT NULL,
              from_note_id TEXT NOT NULL,
              to_note_id TEXT NOT NULL,
              type TEXT NOT NULL,
              directed INTEGER NOT NULL DEFAULT 0,
              weight REAL DEFAULT 1.0,
              created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              created_by TEXT NOT NULL,
              metadata JSON
            )
            """
        )
        conn.execute(
            "CREATE UNIQUE INDEX uniq_note_edges_undirected "
            "ON note_edges(user_id, type, directed, from_note_id, to_note_id)"
        )
        conn.executemany(
            "INSERT INTO note_edges VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            edge_rows,
        )
        conn.execute(
            "UPDATE db_schema_version SET version = 57 WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        )


def _valid_edge(**overrides: object) -> tuple[object, ...]:
    row: dict[str, object] = {
        "edge_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "user_id": OWNER,
        "from_note_id": "11111111-1111-4111-8111-111111111111",
        "to_note_id": "22222222-2222-4222-8222-222222222222",
        "type": "manual",
        "directed": 0,
        "weight": 2.5,
        "created_at": CREATED_AT,
        "created_by": "device-legacy",
        "metadata": json.dumps({"label": "Related", "context": {"kind": "research"}}),
    }
    row.update(overrides)
    return tuple(row[key] for key in row)


def test_sqlite_v57_to_v58_normalizes_legacy_link_and_installs_graph_state(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "notes-link-v57.sqlite"
    _replace_with_v57_edge_table(db_path, edge_rows=[_valid_edge()])

    migrated = CharactersRAGDB(str(db_path), client_id=OWNER)
    try:
        with migrated.transaction() as conn:
            assert migrated._get_db_version(conn) == 58
            row = conn.execute("SELECT * FROM note_edges").fetchone()
            assert row["label"] == "Related"
            assert json.loads(row["properties"]) == {"context": {"kind": "research"}}
            assert row["version"] == 1
            assert row["deleted"] == 0
            assert row["deleted_at"] is None
            assert row["last_modified"] == CREATED_AT
            assert row["created_at"] == CREATED_AT
            assert row["created_by"] == "device-legacy"
            assert conn.execute("PRAGMA foreign_key_check").fetchall() == []

            tables = {
                value[0] for value in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            }
            assert {
                "note_wikilink_edges",
                "note_graph_note_state",
                "note_graph_dirty",
                "note_graph_projection_state",
                "note_graph_revisions",
            } <= tables
            for derived_table in ("note_wikilink_edges", "note_graph_dirty"):
                table_sql = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                    (derived_table,),
                ).fetchone()[0]
                assert "ON DELETE CASCADE" in table_sql

            unresolved_target = str(uuid4())
            conn.execute(
                "INSERT INTO note_wikilink_edges (source_note_id, target_note_id, "
                "source_version, parser_version) VALUES (?, ?, 1, 1)",
                (str(row["from_note_id"]), unresolved_target),
            )
            assert (
                conn.execute(
                    "SELECT target_note_id FROM note_wikilink_edges WHERE target_note_id = ?",
                    (unresolved_target,),
                ).fetchone()[0]
                == unresolved_target
            )

            revision_before = conn.execute(
                "SELECT revision FROM note_graph_revisions WHERE singleton_id = 1"
            ).fetchone()[0]
            conn.execute(
                "UPDATE notes SET content = content || ' changed' WHERE id = ?",
                (row["from_note_id"],),
            )
            revision_after = conn.execute(
                "SELECT revision FROM note_graph_revisions WHERE singleton_id = 1"
            ).fetchone()[0]
            assert revision_after == revision_before + 1
            assert (
                conn.execute(
                    "SELECT generation FROM note_graph_dirty WHERE note_id = ?",
                    (row["from_note_id"],),
                ).fetchone()[0]
                >= 1
            )

            conn.execute("UPDATE note_edges SET label = 'Changed' WHERE edge_id = ?", (row["edge_id"],))
            keyword_revision = conn.execute(
                "SELECT revision FROM note_graph_revisions WHERE singleton_id = 1"
            ).fetchone()[0]
            assert keyword_revision == revision_after + 1
            conn.execute(
                "INSERT INTO keywords(sync_id, keyword, client_id) VALUES (?, ?, ?)",
                (str(uuid4()), "Graph tag", OWNER),
            )
            keyword_id = conn.execute("SELECT id FROM keywords WHERE keyword = 'Graph tag'").fetchone()[0]
            tagged_revision = conn.execute(
                "SELECT revision FROM note_graph_revisions WHERE singleton_id = 1"
            ).fetchone()[0]
            assert tagged_revision == keyword_revision + 1
            conn.execute(
                "INSERT INTO note_keywords(note_id, keyword_id) VALUES (?, ?)",
                (row["from_note_id"], keyword_id),
            )
            assert (
                conn.execute("SELECT revision FROM note_graph_revisions WHERE singleton_id = 1").fetchone()[0]
                == tagged_revision + 1
            )
    finally:
        migrated.close_connection()

    reopened = CharactersRAGDB(str(db_path), client_id=OWNER)
    try:
        with reopened.transaction() as conn:
            assert reopened._get_db_version(conn) == 58
            assert conn.execute("SELECT COUNT(*) FROM note_edges").fetchone()[0] == 1
    finally:
        reopened.close_connection()


def test_sqlite_v57_to_v58_queues_existing_notes_for_projection_rebuild(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "notes-link-v57-existing-notes.sqlite"
    source_id = "11111111-1111-4111-8111-111111111111"
    target_id = "22222222-2222-4222-8222-222222222222"
    db = CharactersRAGDB(str(db_path), client_id=OWNER)
    try:
        db.add_note("Source", f"[[id:{target_id}]]", note_id=source_id)
        db.add_note("Target", "Body", note_id=target_id)
    finally:
        db.close_connection()

    with sqlite3.connect(db_path) as conn:
        for trigger_name in (
            "notes_graph_notes_ai",
            "notes_graph_notes_au",
            "notes_graph_notes_ad",
            "notes_graph_edges_ai",
            "notes_graph_edges_au",
            "notes_graph_edges_ad",
            "notes_graph_keywords_ai",
            "notes_graph_keywords_au",
            "notes_graph_keywords_ad",
            "notes_graph_note_keywords_ai",
            "notes_graph_note_keywords_au",
            "notes_graph_note_keywords_ad",
            "notes_graph_conversations_ai",
            "notes_graph_conversations_au",
            "notes_graph_conversations_ad",
        ):
            conn.execute(f"DROP TRIGGER IF EXISTS {trigger_name}")  # nosec B608
        for table_name in (
            "note_wikilink_edges",
            "note_graph_note_state",
            "note_graph_dirty",
            "note_graph_projection_state",
            "note_graph_revisions",
        ):
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")  # nosec B608
        conn.execute(
            "UPDATE db_schema_version SET version = 57 WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        )

    migrated = CharactersRAGDB(str(db_path), client_id=OWNER)
    try:
        with migrated.transaction() as conn:
            dirty_ids = {
                str(row[0]) for row in conn.execute("SELECT note_id FROM note_graph_dirty ORDER BY note_id").fetchall()
            }
            status = conn.execute(
                "SELECT parser_version, rebuild_state, rebuild_cursor "
                "FROM note_graph_projection_state WHERE singleton_id = 1"
            ).fetchone()
        assert dirty_ids == {source_id, target_id}
        assert tuple(status) == (1, "pending", None)
    finally:
        migrated.close_connection()


@pytest.mark.parametrize(
    ("overrides", "extra_notes", "match"),
    [
        ({"edge_id": "not-a-uuid"}, None, "canonical UUIDv4"),
        ({"to_note_id": "11111111-1111-4111-8111-111111111111"}, None, "differ"),
        ({"to_note_id": "33333333-3333-4333-8333-333333333333"}, None, "endpoint"),
        (
            {"to_note_id": "33333333-3333-4333-8333-333333333333"},
            [("33333333-3333-4333-8333-333333333333", "different-owner")],
            "owner",
        ),
        ({"weight": -1}, None, "weight"),
        ({"type": "wikilink"}, None, "manual"),
        ({"directed": 2}, None, "boolean"),
        (
            {
                "from_note_id": "22222222-2222-4222-8222-222222222222",
                "to_note_id": "11111111-1111-4111-8111-111111111111",
            },
            None,
            "canonical order",
        ),
        ({"metadata": "[]"}, None, "metadata"),
        ({"metadata": "{broken"}, None, "metadata"),
        ({"metadata": json.dumps({"label": 7})}, None, "label"),
        ({"metadata": json.dumps({"label": "x" * 257})}, None, "label"),
        ({"metadata": json.dumps({"a": {"b": {"c": {"d": {"e": True}}}}})}, None, "properties"),
        ({"metadata": json.dumps({str(index): index for index in range(65)})}, None, "properties"),
        ({"metadata": json.dumps({"payload": "x" * (16 * 1024)})}, None, "properties"),
    ],
)
def test_sqlite_v57_to_v58_rolls_back_invalid_legacy_link(
    tmp_path: Path,
    overrides: dict[str, object],
    extra_notes: list[tuple[str, str]] | None,
    match: str,
) -> None:
    db_path = tmp_path / f"invalid-{uuid4()}.sqlite"
    _replace_with_v57_edge_table(
        db_path,
        edge_rows=[_valid_edge(**overrides)],
        extra_notes=extra_notes,
    )

    with pytest.raises(CharactersRAGDBError, match=match):
        CharactersRAGDB(str(db_path), client_id=OWNER)

    with sqlite3.connect(db_path) as conn:
        assert (
            conn.execute(
                "SELECT version FROM db_schema_version WHERE schema_name = ?",
                (CharactersRAGDB._SCHEMA_NAME,),
            ).fetchone()[0]
            == 57
        )
        assert "properties" not in {row[1] for row in conn.execute("PRAGMA table_info(note_edges)").fetchall()}


def test_sqlite_v57_to_v58_rejects_duplicate_logical_links_before_rebuild(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "duplicate-links.sqlite"
    second = list(_valid_edge(edge_id="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"))
    _replace_with_v57_edge_table(db_path, edge_rows=[])
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP INDEX uniq_note_edges_undirected")
        conn.executemany(
            "INSERT INTO note_edges VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [_valid_edge(), tuple(second)],
        )

    with pytest.raises(CharactersRAGDBError, match="duplicate"):
        CharactersRAGDB(str(db_path), client_id=OWNER)

    with sqlite3.connect(db_path) as conn:
        assert (
            conn.execute(
                "SELECT version FROM db_schema_version WHERE schema_name = ?",
                (CharactersRAGDB._SCHEMA_NAME,),
            ).fetchone()[0]
            == 57
        )


class _PostgresMigrationBackend:
    backend_type = BackendType.POSTGRESQL

    def __init__(self, *, schema_owner: bool = True, force_notes: bool = True) -> None:
        self.schema_owner = schema_owner
        self.force_notes = force_notes
        self.calls: list[tuple[str, object]] = []

    def execute(self, statement: str, params=None, connection=None) -> QueryResult:
        normalized = " ".join(statement.split())
        self.calls.append((normalized, params))
        if "FROM pg_class AS table_row" in normalized:
            tables = (
                "notes",
                "note_edges",
                "chacha_keywords",
                "note_keywords",
                "conversations",
            )
            return QueryResult(
                rows=[
                    {
                        "table_name": table,
                        "relrowsecurity": table == "notes" and self.force_notes,
                        "relforcerowsecurity": table == "notes" and self.force_notes,
                        "is_schema_owner": self.schema_owner,
                    }
                    for table in tables
                ],
                rowcount=len(tables),
            )
        if normalized.startswith("SELECT") and "FROM note_edges AS edge" in normalized:
            return QueryResult(rows=[], rowcount=0)
        if normalized.startswith("SELECT COUNT(*)"):
            return QueryResult(rows=[{"count": 0}], rowcount=1)
        return QueryResult(rows=[], rowcount=0)


def _postgres_db(backend: _PostgresMigrationBackend) -> CharactersRAGDB:
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db._local = type("Local", (), {})()
    db._backend = backend
    db._uses_shared_content_backend = False
    return db


def test_postgres_v57_to_v58_locks_and_restores_verified_notes_rls() -> None:
    backend = _PostgresMigrationBackend()
    db = _postgres_db(backend)

    assert hasattr(db, "_migrate_from_v57_to_v58_postgres")
    db._migrate_from_v57_to_v58_postgres(object())

    statements = [statement for statement, _ in backend.calls]
    lock_index = next(index for index, statement in enumerate(statements) if statement.startswith("LOCK TABLE"))
    catalog_index = next(
        index for index, statement in enumerate(statements) if "FROM pg_class AS table_row" in statement
    )
    no_force_index = statements.index("ALTER TABLE notes NO FORCE ROW LEVEL SECURITY")
    scan_index = next(index for index, statement in enumerate(statements) if "FROM note_edges AS edge" in statement)
    force_index = statements.index("ALTER TABLE notes FORCE ROW LEVEL SECURITY")
    version_index = next(
        index for index, statement in enumerate(statements) if "INSERT INTO db_schema_version" in statement
    )

    assert statements[lock_index].startswith(
        "LOCK TABLE notes, note_edges, chacha_keywords, note_keywords, conversations IN ACCESS EXCLUSIVE MODE"
    )
    assert lock_index < catalog_index < no_force_index < scan_index < force_index < version_index
    sql = "\n".join(statements)
    assert "ADD COLUMN IF NOT EXISTS properties" in sql
    assert "CREATE TABLE IF NOT EXISTS note_graph_note_state" in sql
    assert "CREATE TABLE IF NOT EXISTS note_graph_revisions" in sql
    assert "target_note_id TEXT NOT NULL REFERENCES notes" not in sql
    assert "CREATE TRIGGER" in sql
    assert "notes_graph_conversations_changed_trigger" in sql
    assert "INSERT INTO note_graph_dirty" in sql
    assert "SELECT client_id, id, 1, CURRENT_TIMESTAMP FROM notes" in sql
    assert "INSERT INTO note_graph_projection_state" in sql
    assert "SELECT DISTINCT client_id, 1, 'pending', NULL, CURRENT_TIMESTAMP FROM notes" in sql


def test_postgres_v57_to_v58_rejects_unverified_schema_owner_before_scans() -> None:
    backend = _PostgresMigrationBackend(schema_owner=False)
    db = _postgres_db(backend)

    with pytest.raises(SchemaError, match="schema-owner"):
        db._migrate_from_v57_to_v58_postgres(object())

    statements = [statement for statement, _ in backend.calls]
    assert not any("FROM note_edges AS edge" in statement for statement in statements)
    assert not any("INSERT INTO db_schema_version" in statement for statement in statements)


def test_postgres_v58_installs_durable_link_constraints() -> None:
    backend = _PostgresMigrationBackend()
    db = _postgres_db(backend)

    db._migrate_from_v57_to_v58_postgres(object())

    sql = "\n".join(statement for statement, _ in backend.calls)
    for constraint in (
        "note_edges_source_note_fkey",
        "note_edges_target_note_fkey",
        "note_edges_manual_type_check",
        "note_edges_directed_check",
        "note_edges_weight_check",
        "note_edges_label_check",
        "note_edges_self_link_check",
        "note_edges_version_check",
    ):
        assert constraint in sql
    assert (
        "note_edges_source_note_fkey FOREIGN KEY (from_note_id) "
        "REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE"
    ) in sql
    assert (
        "note_edges_target_note_fkey FOREIGN KEY (to_note_id) REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE"
    ) in sql


def test_postgres_graph_trigger_functions_use_one_valid_declare_block() -> None:
    backend = _PostgresMigrationBackend()
    db = _postgres_db(backend)

    db._migrate_from_v57_to_v58_postgres(object())

    statements = [statement for statement, _ in backend.calls]
    notes_changed = next(statement for statement in statements if "FUNCTION notes_graph_notes_changed" in statement)
    note_keywords_changed = next(
        statement for statement in statements if "FUNCTION notes_graph_note_keywords_changed" in statement
    )
    assert notes_changed.count("DECLARE") == 1
    assert note_keywords_changed.count("DECLARE") == 1


def test_sqlite_migration_map_contains_v57_to_v58_step(tmp_path: Path) -> None:
    db = CharactersRAGDB(str(tmp_path / "migration-map.sqlite"), client_id=OWNER)
    try:
        assert db._sqlite_linear_migration_steps()[57].__name__ == "_migrate_from_v57_to_v58"
    finally:
        db.close_connection()


def test_sqlite_fresh_schema_is_v58_and_canonical(tmp_path: Path) -> None:
    db = CharactersRAGDB(str(tmp_path / "fresh-v58.sqlite"), client_id=OWNER)
    try:
        with db.transaction() as conn:
            assert db._get_db_version(conn) == 58
            columns = {row[1] for row in conn.execute("PRAGMA table_info(note_edges)").fetchall()}
            assert {
                "label",
                "properties",
                "last_modified",
                "version",
                "deleted",
                "deleted_at",
            } <= columns
            assert (
                conn.execute("SELECT revision FROM note_graph_revisions WHERE singleton_id = 1").fetchone() is not None
            )
    finally:
        db.close_connection()
