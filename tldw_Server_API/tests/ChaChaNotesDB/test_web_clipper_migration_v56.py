"""Schema v56 coverage for external Web Clipper identity mappings."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from uuid import UUID, uuid1, uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import QueryResult
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    BackendType,
    CharactersRAGDB,
    CharactersRAGDBError,
    SchemaError,
)


def _assert_uuid4(value: str) -> None:
    parsed = UUID(value)
    assert parsed.version == 4
    assert str(parsed) == value


def _replacement_note_id(clip_id: str) -> str:
    digest = hashlib.sha256(f"web-clipper-migration:notes.note:{clip_id}".encode()).hexdigest()
    return str(UUID(digest[:32], version=4))


def _build_v55_fixture(db_path: Path) -> tuple[str, str]:
    db = CharactersRAGDB(str(db_path), client_id="web-clipper-v55-fixture")
    canonical_id = str(uuid4())
    try:
        db.upsert_workspace("ws-1", "Workspace")
        for note_id in ("clip-legacy", canonical_id):
            db.add_note(title=note_id, content="Body", note_id=note_id)
    finally:
        db.close_connection()

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("DROP TABLE note_attachments")
        conn.execute("DROP TABLE note_clipper_workspace_placements")
        conn.execute("DROP TABLE note_clipper_documents")
        conn.execute(
            """
            CREATE TABLE note_clipper_documents(
              clip_id TEXT PRIMARY KEY REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE,
              note_id TEXT NOT NULL UNIQUE REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE,
              clip_type TEXT NOT NULL,
              source_url TEXT,
              source_title TEXT,
              capture_metadata_json TEXT NOT NULL DEFAULT '{}',
              analysis_json TEXT NOT NULL DEFAULT '{}',
              content_budget_json TEXT NOT NULL DEFAULT '{}',
              source_note_version INTEGER,
              created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              deleted BOOLEAN NOT NULL DEFAULT 0,
              CHECK(clip_id = note_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE note_clipper_workspace_placements(
              clip_id TEXT NOT NULL REFERENCES note_clipper_documents(clip_id) ON DELETE CASCADE ON UPDATE CASCADE,
              workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE ON UPDATE CASCADE,
              workspace_note_id INTEGER,
              source_note_id TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE,
              source_note_version INTEGER,
              created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              deleted BOOLEAN NOT NULL DEFAULT 0,
              PRIMARY KEY (clip_id, workspace_id)
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO note_clipper_documents(
              clip_id, note_id, clip_type, source_note_version
            ) VALUES (?, ?, 'article', 1)
            """,
            [("clip-legacy", "clip-legacy"), (canonical_id, canonical_id)],
        )
        conn.execute(
            """
            INSERT INTO note_clipper_workspace_placements(
              clip_id, workspace_id, workspace_note_id, source_note_id, source_note_version
            ) VALUES ('clip-legacy', 'ws-1', 7, 'clip-legacy', 1)
            """
        )
        conn.execute(
            "UPDATE db_schema_version SET version = 55 WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        )
    return "clip-legacy", canonical_id


def test_v55_to_v56_rekeys_only_non_uuid_note_and_preserves_public_mapping(tmp_path: Path) -> None:
    db_path = tmp_path / "web-clipper-v55.sqlite"
    legacy_clip_id, canonical_id = _build_v55_fixture(db_path)

    migrated = CharactersRAGDB(str(db_path), client_id="web-clipper-v56")
    try:
        with migrated.transaction() as conn:
            version = conn.execute(
                "SELECT version FROM db_schema_version WHERE schema_name = ?",
                (CharactersRAGDB._SCHEMA_NAME,),
            ).fetchone()["version"]
            legacy_document = migrated._fetch_note_clipper_document_row(
                column="clip_id",
                value=legacy_clip_id,
                conn=conn,
            )
            canonical_document = migrated._fetch_note_clipper_document_row(
                column="clip_id",
                value=canonical_id,
                conn=conn,
            )
            assert version == CharactersRAGDB._CURRENT_SCHEMA_VERSION
            assert legacy_document is not None
            assert legacy_document["clip_id"] == legacy_clip_id
            assert legacy_document["note_id"] != legacy_clip_id
            _assert_uuid4(str(legacy_document["note_id"]))
            assert canonical_document is not None
            assert canonical_document["note_id"] == canonical_id
            placement = migrated.list_note_clipper_workspace_placements(legacy_clip_id)[0]
            assert placement["source_note_id"] == legacy_document["note_id"]
            assert migrated.get_note_by_id(str(legacy_document["note_id"])) is not None
            assert migrated.get_note_by_id(legacy_clip_id) is None
            assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM sync_log WHERE entity = 'notes' AND entity_id = ?",
                    (legacy_clip_id,),
                ).fetchone()[0]
                == 0
            )

            table_sql = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'note_clipper_documents'"
            ).fetchone()["sql"]
            assert "CHECK(clip_id = note_id)" not in table_sql
            foreign_keys = conn.execute("PRAGMA foreign_key_list('note_clipper_documents')").fetchall()
            assert {(row["from"], row["table"]) for row in foreign_keys} == {("note_id", "notes")}
    finally:
        migrated.close_connection()

    reopened = CharactersRAGDB(str(db_path), client_id="web-clipper-v56-reopen")
    try:
        reopened_document = reopened.get_note_clipper_document_by_clip_id(legacy_clip_id)
        assert reopened_document is not None
        assert reopened_document["note_id"] == legacy_document["note_id"]
    finally:
        reopened.close_connection()


def test_v55_to_v56_fails_closed_for_noncanonical_uuid_note(tmp_path: Path) -> None:
    db_path = tmp_path / "web-clipper-v55-noncanonical-uuid.sqlite"
    _build_v55_fixture(db_path)
    unsafe_uuid = str(uuid1())
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute(
            "UPDATE notes SET id = ? WHERE id = 'clip-legacy'",
            (unsafe_uuid,),
        )
        conn.execute(
            "UPDATE note_clipper_documents SET clip_id = ?, note_id = ? WHERE clip_id = 'clip-legacy'",
            (unsafe_uuid, unsafe_uuid),
        )
        conn.execute(
            "UPDATE note_clipper_workspace_placements SET clip_id = ?, source_note_id = ? "
            "WHERE clip_id = 'clip-legacy'",
            (unsafe_uuid, unsafe_uuid),
        )

    with pytest.raises(CharactersRAGDBError, match="noncanonical UUID"):
        CharactersRAGDB(str(db_path), client_id="web-clipper-v56-fail-closed")

    with sqlite3.connect(db_path) as conn:
        assert (
            conn.execute(
                "SELECT version FROM db_schema_version WHERE schema_name = ?",
                (CharactersRAGDB._SCHEMA_NAME,),
            ).fetchone()[0]
            == 55
        )
        assert conn.execute("SELECT id FROM notes WHERE id = ?", (unsafe_uuid,)).fetchone()
        assert (
            conn.execute(
                "SELECT note_id FROM note_clipper_documents WHERE clip_id = ?",
                (unsafe_uuid,),
            ).fetchone()[0]
            == unsafe_uuid
        )


@pytest.mark.parametrize("payload", ["1", "[]", "null", '"scalar"'])
def test_v55_to_v56_rolls_back_for_non_object_sync_payload(
    tmp_path: Path,
    payload: str,
) -> None:
    db_path = tmp_path / "web-clipper-v55-non-object-sync-payload.sqlite"
    _build_v55_fixture(db_path)
    with sqlite3.connect(db_path) as conn:
        updated = conn.execute(
            "UPDATE sync_log SET payload = ? WHERE entity = 'notes' AND entity_id = 'clip-legacy'",
            (payload,),
        )
        assert updated.rowcount > 0

    with pytest.raises(CharactersRAGDBError, match="sync_log payload"):
        CharactersRAGDB(str(db_path), client_id="web-clipper-v56-invalid-sync-payload")

    with sqlite3.connect(db_path) as conn:
        assert (
            conn.execute(
                "SELECT version FROM db_schema_version WHERE schema_name = ?",
                (CharactersRAGDB._SCHEMA_NAME,),
            ).fetchone()[0]
            == 55
        )
        assert conn.execute("SELECT id FROM notes WHERE id = 'clip-legacy'").fetchone()
        assert {
            row[0]
            for row in conn.execute(
                "SELECT payload FROM sync_log WHERE entity = 'notes' AND entity_id = 'clip-legacy'"
            ).fetchall()
        } == {payload}


def test_v55_to_v56_rolls_back_for_duplicate_root_sync_payload_id(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "web-clipper-v55-duplicate-sync-payload-id.sqlite"
    _build_v55_fixture(db_path)
    payload = '{"id":"clip-legacy","id":"forged","version":1}'
    with sqlite3.connect(db_path) as conn:
        updated = conn.execute(
            "UPDATE sync_log SET payload = ? WHERE entity = 'notes' AND entity_id = 'clip-legacy'",
            (payload,),
        )
        assert updated.rowcount > 0

    with pytest.raises(CharactersRAGDBError, match="sync_log payload"):
        CharactersRAGDB(str(db_path), client_id="web-clipper-v56-duplicate-sync-id")

    with sqlite3.connect(db_path) as conn:
        assert (
            conn.execute(
                "SELECT version FROM db_schema_version WHERE schema_name = ?",
                (CharactersRAGDB._SCHEMA_NAME,),
            ).fetchone()[0]
            == 55
        )
        assert conn.execute("SELECT id FROM notes WHERE id = 'clip-legacy'").fetchone()
        assert (
            conn.execute(
                "SELECT payload FROM sync_log WHERE entity = 'notes' AND entity_id = 'clip-legacy'"
            ).fetchone()[0]
            == payload
        )


@pytest.mark.parametrize(
    "payload",
    [
        '{"version":1}',
        '{"id":123,"version":1}',
        '{"id":"forged","version":1}',
    ],
)
def test_v55_to_v56_rolls_back_for_unattributed_sync_payload_id(
    tmp_path: Path,
    payload: str,
) -> None:
    db_path = tmp_path / "web-clipper-v55-unattributed-sync-payload-id.sqlite"
    _build_v55_fixture(db_path)
    with sqlite3.connect(db_path) as conn:
        updated = conn.execute(
            "UPDATE sync_log SET payload = ? WHERE entity = 'notes' AND entity_id = 'clip-legacy'",
            (payload,),
        )
        assert updated.rowcount > 0

    with pytest.raises(CharactersRAGDBError, match="sync_log payload"):
        CharactersRAGDB(str(db_path), client_id="web-clipper-v56-unattributed-sync-id")

    with sqlite3.connect(db_path) as conn:
        assert (
            conn.execute(
                "SELECT version FROM db_schema_version WHERE schema_name = ?",
                (CharactersRAGDB._SCHEMA_NAME,),
            ).fetchone()[0]
            == 55
        )
        assert conn.execute("SELECT id FROM notes WHERE id = 'clip-legacy'").fetchone()
        assert (
            conn.execute(
                "SELECT payload FROM sync_log WHERE entity = 'notes' AND entity_id = 'clip-legacy'"
            ).fetchone()[0]
            == payload
        )


@pytest.mark.parametrize(
    ("table_name", "id_column"),
    [("sync_envelopes", "entity_id"), ("sync_object_state", "object_id")],
)
def test_v55_to_v56_rolls_back_when_local_sync_history_references_legacy_id(
    tmp_path: Path,
    table_name: str,
    id_column: str,
) -> None:
    db_path = tmp_path / f"web-clipper-v55-{table_name}.sqlite"
    _build_v55_fixture(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            f"CREATE TABLE {table_name}(domain TEXT NOT NULL, {id_column} TEXT NOT NULL)"  # nosec B608
        )
        conn.execute(
            f"INSERT INTO {table_name}(domain, {id_column}) VALUES ('notes.note', 'clip-legacy')"  # nosec B608
        )

    with pytest.raises(CharactersRAGDBError, match="canonical Sync history"):
        CharactersRAGDB(str(db_path), client_id="web-clipper-v56-sync-history")

    with sqlite3.connect(db_path) as conn:
        assert (
            conn.execute(
                "SELECT version FROM db_schema_version WHERE schema_name = ?",
                (CharactersRAGDB._SCHEMA_NAME,),
            ).fetchone()[0]
            == 55
        )
        assert conn.execute("SELECT id FROM notes WHERE id = 'clip-legacy'").fetchone()


@pytest.mark.parametrize(
    ("table_name", "id_column"),
    [
        ("sync_log", "entity_id"),
        ("sync_envelopes", "entity_id"),
        ("sync_object_state", "object_id"),
    ],
)
def test_v55_to_v56_rolls_back_when_replacement_has_prior_sync_history(
    tmp_path: Path,
    table_name: str,
    id_column: str,
) -> None:
    db_path = tmp_path / f"web-clipper-v55-replacement-{table_name}.sqlite"
    _build_v55_fixture(db_path)
    replacement = _replacement_note_id("clip-legacy")
    with sqlite3.connect(db_path) as conn:
        if table_name == "sync_log":
            conn.execute(
                "INSERT INTO sync_log(entity, entity_id, operation, timestamp, "
                "client_id, version, payload) "
                "VALUES ('notes', ?, 'delete', CURRENT_TIMESTAMP, 'prior-owner', 1, ?)",
                (replacement, f'{{"id":"{replacement}","deleted":1}}'),
            )
        else:
            conn.execute(
                f"CREATE TABLE {table_name}(domain TEXT NOT NULL, {id_column} TEXT NOT NULL)"  # nosec B608
            )
            conn.execute(
                f"INSERT INTO {table_name}(domain, {id_column}) VALUES ('notes.note', ?) ",  # nosec B608
                (replacement,),
            )

    with pytest.raises(CharactersRAGDBError, match="Sync history"):
        CharactersRAGDB(str(db_path), client_id="web-clipper-v56-replacement-history")

    with sqlite3.connect(db_path) as conn:
        assert (
            conn.execute(
                "SELECT version FROM db_schema_version WHERE schema_name = ?",
                (CharactersRAGDB._SCHEMA_NAME,),
            ).fetchone()[0]
            == 55
        )
        assert conn.execute("SELECT id FROM notes WHERE id = 'clip-legacy'").fetchone()
        assert conn.execute(
            f"SELECT 1 FROM {table_name} WHERE {id_column} = ?",  # nosec B608
            (replacement,),
        ).fetchone()


def test_v55_to_v56_rolls_back_for_unverifiable_non_cascading_reference(tmp_path: Path) -> None:
    db_path = tmp_path / "web-clipper-v55-external-reference.sqlite"
    _build_v55_fixture(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """
            CREATE TABLE external_note_refs(
              note_id TEXT NOT NULL REFERENCES notes(id) ON UPDATE NO ACTION
            )
            """
        )
        conn.execute("INSERT INTO external_note_refs(note_id) VALUES ('clip-legacy')")

    with pytest.raises(CharactersRAGDBError, match="non-cascading reference"):
        CharactersRAGDB(str(db_path), client_id="web-clipper-v56-external-reference")

    with sqlite3.connect(db_path) as conn:
        assert (
            conn.execute(
                "SELECT version FROM db_schema_version WHERE schema_name = ?",
                (CharactersRAGDB._SCHEMA_NAME,),
            ).fetchone()[0]
            == 55
        )
        assert conn.execute("SELECT id FROM notes WHERE id = 'clip-legacy'").fetchone()


def test_sqlite_migration_map_contains_v55_to_v56_step(tmp_path: Path) -> None:
    db = CharactersRAGDB(str(tmp_path / "migration-map.sqlite"), client_id="migration-map")
    try:
        assert db._sqlite_linear_migration_steps()[55].__name__ == "_migrate_from_v55_to_v56"
    finally:
        db.close_connection()


def test_postgres_fresh_schema_decouples_public_clip_id_from_note_fk() -> None:
    class _FakePostgresBackend:
        backend_type = BackendType.POSTGRESQL

        def __init__(self) -> None:
            self.statements: list[str] = []

        def execute(self, statement: str, params=None, connection=None) -> None:
            self.statements.append(statement)

    backend = _FakePostgresBackend()
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db._local = type("Local", (), {})()
    db._backend = backend
    db._uses_shared_content_backend = False

    db._ensure_web_clipper_schema_postgres(object())

    document_ddl = next(
        statement
        for statement in backend.statements
        if "CREATE TABLE IF NOT EXISTS note_clipper_documents" in statement
    )
    clip_line = next(line for line in document_ddl.splitlines() if "clip_id" in line)
    note_line = next(line for line in document_ddl.splitlines() if "note_id" in line)
    placement_ddl = next(
        statement
        for statement in backend.statements
        if "CREATE TABLE IF NOT EXISTS note_clipper_workspace_placements" in statement
    )
    assert "REFERENCES notes" not in clip_line
    assert "REFERENCES notes" in note_line
    assert "CHECK(clip_id = note_id)" not in document_ddl
    assert "client_id" in document_ddl
    assert "PRIMARY KEY (client_id, clip_id)" in document_ddl
    assert "UNIQUE (client_id, note_id)" in document_ddl
    assert "client_id" in placement_ddl
    assert "PRIMARY KEY (client_id, clip_id, workspace_id)" in placement_ddl
    assert ("FOREIGN KEY (client_id, clip_id) REFERENCES note_clipper_documents(client_id, clip_id)") in " ".join(
        placement_ddl.split()
    )


class _OwnerAwareMigrationBackend:
    backend_type = BackendType.POSTGRESQL

    def __init__(
        self,
        *,
        invalid_owner_count: int = 0,
        forced_rls: bool = True,
        legacy_row: bool = False,
        sync_tables: bool = False,
    ) -> None:
        self.invalid_owner_count = invalid_owner_count
        self.forced_rls = forced_rls
        self.legacy_row = legacy_row
        self.sync_tables = sync_tables
        self.replacement_note_id: str | None = None
        self.statements: list[tuple[str, object]] = []

    def table_exists(self, table_name: str, connection=None) -> bool:
        tables = {
            "note_clipper_documents",
            "note_clipper_workspace_placements",
            "notes",
            "sync_log",
        }
        if self.sync_tables:
            tables.update(("sync_envelopes", "sync_object_state"))
        return table_name in tables

    def execute(self, statement: str, params=None, connection=None) -> QueryResult:
        self.statements.append((statement, params))
        normalized = " ".join(statement.split())
        if "relrowsecurity" in statement and "relforcerowsecurity" in statement:
            return QueryResult(
                rows=[
                    {
                        "relrowsecurity": True,
                        "relforcerowsecurity": self.forced_rls,
                        "is_schema_owner": True,
                    }
                ],
                rowcount=1,
            )
        if "AS invalid_owner_count" in statement:
            return QueryResult(rows=[{"invalid_owner_count": self.invalid_owner_count}], rowcount=1)
        if "FROM note_clipper_documents d" in statement:
            rows = (
                [
                    {
                        "clip_id": "clip-legacy",
                        "note_id": "clip-legacy",
                        "existing_note_id": "clip-legacy",
                        "owner_client_id": "910001",
                    }
                ]
                if self.legacy_row
                else []
            )
            return QueryResult(rows=rows, rowcount=len(rows))
        if "c.confrelid = 'notes'::regclass" in statement:
            return QueryResult(rows=[], rowcount=0)
        if "FROM pg_constraint c" in statement and "note_clipper_workspace_placements" in statement:
            return QueryResult(
                rows=[
                    {
                        "conname": "note_clipper_workspace_placements_pkey",
                        "contype": "p",
                        "definition": "PRIMARY KEY (clip_id, workspace_id)",
                        "columns": ["clip_id", "workspace_id"],
                    },
                    {
                        "conname": "note_clipper_workspace_placements_clip_id_fkey",
                        "contype": "f",
                        "definition": (
                            "FOREIGN KEY (clip_id) REFERENCES note_clipper_documents(clip_id) ON UPDATE CASCADE"
                        ),
                        "columns": ["clip_id"],
                    },
                ],
                rowcount=2,
            )
        if "FROM pg_constraint c" in statement and "note_clipper_documents" in statement:
            return QueryResult(
                rows=[
                    {
                        "conname": "note_clipper_documents_clip_id_fkey",
                        "contype": "f",
                        "definition": "FOREIGN KEY (clip_id) REFERENCES notes(id) ON UPDATE CASCADE",
                        "columns": ["clip_id"],
                    },
                    {
                        "conname": "note_clipper_documents_check",
                        "contype": "c",
                        "definition": "CHECK ((clip_id = note_id))",
                        "columns": ["clip_id", "note_id"],
                    },
                    {
                        "conname": "note_clipper_documents_pkey",
                        "contype": "p",
                        "definition": "PRIMARY KEY (clip_id)",
                        "columns": ["clip_id"],
                    },
                    {
                        "conname": "note_clipper_documents_note_id_key",
                        "contype": "u",
                        "definition": "UNIQUE (note_id)",
                        "columns": ["note_id"],
                    },
                    {
                        "conname": "note_clipper_documents_note_id_fkey",
                        "contype": "f",
                        "definition": "FOREIGN KEY (note_id) REFERENCES notes(id) ON UPDATE CASCADE",
                        "columns": ["note_id"],
                    },
                ],
                rowcount=5,
            )
        if "COUNT(*)" in normalized:
            return QueryResult(rows=[{"count": 0}], rowcount=1)
        if normalized.startswith("UPDATE notes SET id ="):
            self.replacement_note_id = str(params[0])
        if normalized.startswith("SELECT note_id FROM note_clipper_documents WHERE"):
            return QueryResult(
                rows=[{"note_id": self.replacement_note_id}],
                rowcount=1,
            )
        return QueryResult(rows=[], rowcount=0)


def _migration_statement_index(backend: _OwnerAwareMigrationBackend, fragment: str) -> int:
    return next(
        index for index, (statement, _) in enumerate(backend.statements) if fragment in " ".join(statement.split())
    )


def test_postgres_v55_to_v56_locks_before_preflight_and_restores_forced_rls() -> None:
    backend = _OwnerAwareMigrationBackend(forced_rls=True)
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db._local = type("Local", (), {})()
    db._backend = backend
    db._uses_shared_content_backend = False

    db._migrate_from_v55_to_v56_postgres(object())

    lock_index = _migration_statement_index(backend, "LOCK TABLE")
    rls_state_index = _migration_statement_index(backend, "relforcerowsecurity")
    no_force_index = _migration_statement_index(backend, "ALTER TABLE notes NO FORCE ROW LEVEL SECURITY")
    mapping_read_index = _migration_statement_index(backend, "FROM note_clipper_documents d")
    force_index = _migration_statement_index(backend, "ALTER TABLE notes FORCE ROW LEVEL SECURITY")
    version_index = _migration_statement_index(backend, "INSERT INTO db_schema_version")

    lock_sql = " ".join(backend.statements[lock_index][0].split())
    for table_name in (
        "notes",
        "workspaces",
        "sync_log",
        "note_clipper_documents",
        "note_clipper_workspace_placements",
    ):
        assert table_name in lock_sql
    assert "ACCESS EXCLUSIVE MODE" in lock_sql
    assert lock_index < rls_state_index < no_force_index < mapping_read_index
    assert mapping_read_index < force_index < version_index

    sql = "\n".join(statement for statement, _ in backend.statements)
    placement_preflight = next(
        statement
        for statement, _ in backend.statements
        if "FROM note_clipper_workspace_placements placement" in statement and "invalid_owner_count" in statement
    )
    placement_preflight_index = _migration_statement_index(
        backend,
        "FROM note_clipper_workspace_placements placement",
    )
    assert lock_index < placement_preflight_index
    assert "JOIN notes owner_note ON owner_note.id = document.note_id" in " ".join(placement_preflight.split())
    assert "source_note.client_id IS DISTINCT FROM owner_note.client_id" in placement_preflight
    assert "workspace.client_id IS DISTINCT FROM owner_note.client_id" in placement_preflight
    assert "document.client_id IS DISTINCT FROM owner_note.client_id" in placement_preflight
    assert "VALIDATE CONSTRAINT note_clipper_documents_client_note_key" not in sql


def test_postgres_v55_to_v56_verifies_sync_payload_is_an_object_and_rewritten() -> None:
    backend = _OwnerAwareMigrationBackend(legacy_row=True)
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db._local = type("Local", (), {})()
    db._backend = backend
    db._uses_shared_content_backend = False

    db._migrate_from_v55_to_v56_postgres(object())

    statements = [" ".join(statement.split()) for statement, _ in backend.statements]
    preflight_index = next(
        index
        for index, statement in enumerate(statements)
        if "jsonb_typeof(payload::jsonb) IS DISTINCT FROM 'object'" in statement
    )
    preflight = statements[preflight_index]
    assert "jsonb_typeof(payload::jsonb -> 'id') IS DISTINCT FROM 'string'" in preflight
    assert "(payload::jsonb ->> 'id') IS DISTINCT FROM %s" in preflight
    update_index = next(
        index for index, statement in enumerate(statements) if statement.startswith("UPDATE sync_log SET entity_id")
    )
    verification_index = next(
        index
        for index, statement in enumerate(statements)
        if index > update_index and "payload::jsonb ->> 'id'" in statement
    )
    assert preflight_index < update_index < verification_index


def test_postgres_v55_to_v56_checks_replacement_history_after_locks_before_rekey() -> None:
    backend = _OwnerAwareMigrationBackend(legacy_row=True, sync_tables=True)
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db._local = type("Local", (), {})()
    db._backend = backend
    db._uses_shared_content_backend = False

    db._migrate_from_v55_to_v56_postgres(object())

    replacement = _replacement_note_id("clip-legacy")
    normalized = [" ".join(statement.split()) for statement, _ in backend.statements]
    lock_index = next(index for index, statement in enumerate(normalized) if statement.startswith("LOCK TABLE"))
    rekey_index = next(
        index for index, statement in enumerate(normalized) if statement.startswith("UPDATE notes SET id")
    )
    replacement_checks = [
        index
        for index, ((statement, params), normalized_statement) in enumerate(
            zip(backend.statements, normalized, strict=True)
        )
        if (
            ("FROM sync_log" in normalized_statement and params == ("910001", "notes", replacement))
            or ("FROM sync_envelopes" in normalized_statement and params == ("notes.note", replacement))
            or ("FROM sync_object_state" in normalized_statement and params == ("notes.note", replacement))
        )
    ]
    assert len(replacement_checks) == 3
    assert all(lock_index < index < rekey_index for index in replacement_checks)


def test_postgres_v55_to_v56_fails_closed_for_invalid_authoritative_owner() -> None:
    backend = _OwnerAwareMigrationBackend(invalid_owner_count=1)
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db._local = type("Local", (), {})()
    db._backend = backend
    db._uses_shared_content_backend = False

    with pytest.raises(SchemaError, match="owner"):
        db._migrate_from_v55_to_v56_postgres(object())

    assert not any("INSERT INTO db_schema_version" in statement for statement, _ in backend.statements)


def test_get_postgres_schema_version_can_lock_initializer_row() -> None:
    class _VersionBackend:
        backend_type = BackendType.POSTGRESQL

        def __init__(self) -> None:
            self.statement = ""

        def execute(self, statement: str, params=None, connection=None) -> QueryResult:
            self.statement = statement
            return QueryResult(rows=[{"version": 55}], rowcount=1)

    backend = _VersionBackend()
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db._local = type("Local", (), {})()
    db._backend = backend
    db._uses_shared_content_backend = False

    assert db._get_schema_version_postgres(object(), lock=True) == 55
    assert "FOR UPDATE" in backend.statement


def test_postgres_v55_to_v56_removes_legacy_identity_constraints_and_advances_version() -> None:
    class _FakePostgresBackend:
        backend_type = BackendType.POSTGRESQL

        def __init__(self) -> None:
            self.statements: list[tuple[str, object]] = []

        def table_exists(self, table_name: str, connection=None) -> bool:
            return table_name in {
                "note_clipper_documents",
                "note_clipper_workspace_placements",
                "notes",
                "sync_log",
            }

        def execute(self, statement: str, params=None, connection=None) -> QueryResult:
            self.statements.append((statement, params))
            if "relrowsecurity" in statement and "relforcerowsecurity" in statement:
                return QueryResult(
                    rows=[
                        {
                            "relrowsecurity": True,
                            "relforcerowsecurity": False,
                            "is_schema_owner": True,
                        }
                    ],
                    rowcount=1,
                )
            if "FROM note_clipper_documents d" in statement:
                return QueryResult(rows=[], rowcount=0)
            if "FROM pg_constraint c" in statement and "note_clipper_workspace_placements" in statement:
                return QueryResult(
                    rows=[
                        {
                            "conname": "note_clipper_workspace_placements_pkey",
                            "contype": "p",
                            "definition": "PRIMARY KEY (clip_id, workspace_id)",
                            "columns": ["clip_id", "workspace_id"],
                        },
                        {
                            "conname": "note_clipper_workspace_placements_clip_id_fkey",
                            "contype": "f",
                            "definition": (
                                "FOREIGN KEY (clip_id) REFERENCES note_clipper_documents(clip_id) ON UPDATE CASCADE"
                            ),
                            "columns": ["clip_id"],
                        },
                    ],
                    rowcount=2,
                )
            if "FROM pg_constraint c" in statement and "note_clipper_documents" in statement:
                return QueryResult(
                    rows=[
                        {
                            "conname": "note_clipper_documents_clip_id_fkey",
                            "contype": "f",
                            "definition": "FOREIGN KEY (clip_id) REFERENCES notes(id) ON UPDATE CASCADE",
                            "columns": ["clip_id"],
                        },
                        {
                            "conname": "note_clipper_documents_check",
                            "contype": "c",
                            "definition": "CHECK ((clip_id = note_id))",
                            "columns": ["clip_id", "note_id"],
                        },
                        {
                            "conname": "note_clipper_documents_pkey",
                            "contype": "p",
                            "definition": "PRIMARY KEY (clip_id)",
                            "columns": ["clip_id"],
                        },
                        {
                            "conname": "note_clipper_documents_note_id_key",
                            "contype": "u",
                            "definition": "UNIQUE (note_id)",
                            "columns": ["note_id"],
                        },
                        {
                            "conname": "note_clipper_documents_note_id_fkey",
                            "contype": "f",
                            "definition": "FOREIGN KEY (note_id) REFERENCES notes(id) ON UPDATE CASCADE",
                            "columns": ["note_id"],
                        },
                    ],
                    rowcount=5,
                )
            return QueryResult(rows=[], rowcount=0)

    backend = _FakePostgresBackend()
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db._local = type("Local", (), {})()
    db._backend = backend
    db._uses_shared_content_backend = False

    db._migrate_from_v55_to_v56_postgres(object())

    sql = "\n".join(statement for statement, _ in backend.statements)
    assert 'DROP CONSTRAINT "note_clipper_documents_clip_id_fkey"' in sql
    assert 'DROP CONSTRAINT "note_clipper_documents_check"' in sql
    assert any(
        "INSERT INTO db_schema_version" in statement and params == (CharactersRAGDB._SCHEMA_NAME, 56)
        for statement, params in backend.statements
    )


def test_postgres_v55_to_v56_fails_closed_without_durable_note_mapping_constraints() -> None:
    class _IncompletePostgresBackend:
        backend_type = BackendType.POSTGRESQL

        def table_exists(self, table_name: str, connection=None) -> bool:
            return True

        def execute(self, statement: str, params=None, connection=None) -> QueryResult:
            if "relrowsecurity" in statement and "relforcerowsecurity" in statement:
                return QueryResult(
                    rows=[
                        {
                            "relrowsecurity": True,
                            "relforcerowsecurity": False,
                            "is_schema_owner": True,
                        }
                    ],
                    rowcount=1,
                )
            if "FROM note_clipper_documents d" in statement:
                return QueryResult(rows=[], rowcount=0)
            if "FROM pg_constraint c" in statement and "note_clipper_documents" in statement:
                return QueryResult(
                    rows=[
                        {
                            "conname": "note_clipper_documents_clip_id_fkey",
                            "contype": "f",
                            "definition": "FOREIGN KEY (clip_id) REFERENCES notes(id) ON UPDATE CASCADE",
                            "columns": ["clip_id"],
                        }
                    ],
                    rowcount=1,
                )
            return QueryResult(rows=[], rowcount=0)

    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db._local = type("Local", (), {})()
    db._backend = _IncompletePostgresBackend()
    db._uses_shared_content_backend = False

    with pytest.raises(SchemaError, match="durable note mapping constraints"):
        db._migrate_from_v55_to_v56_postgres(object())
