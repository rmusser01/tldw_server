from __future__ import annotations

import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, QueryResult
from tldw_Server_API.app.core.DB_Management.chacha.organization_sync_store import (
    NotesOrganizationSyncStore,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Sync.v2.materializers.notes_organization import (
    NotesOrganizationMaterializer,
)
from tldw_Server_API.app.core.Sync.v2.models import SyncEnvelope
from tldw_Server_API.app.core.Sync.v2.notes_organization import organization_link_id

pytestmark = pytest.mark.unit


class _PostgresMigrationBackend:
    backend_type = BackendType.POSTGRESQL

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...] | None, Any]] = []

    def execute(
        self,
        statement: str,
        params: tuple[Any, ...] | None = None,
        connection: Any = None,
    ) -> QueryResult:
        normalized = " ".join(statement.split())
        self.calls.append((normalized, params, connection))
        if normalized.startswith("SELECT id FROM"):
            return QueryResult(rows=[{"id": 11}, {"id": 12}], rowcount=2)
        if normalized.startswith("SELECT COUNT(*)"):
            return QueryResult(rows=[{"count": 0}], rowcount=1)
        return QueryResult(rows=[], rowcount=1)


class _CursorResult:
    def __init__(self, row: dict[str, object] | None = None) -> None:
        self._row = row

    def fetchone(self) -> dict[str, object] | None:
        return self._row


class _PostgresProjectionConnection:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.inserted = False

    def execute(self, statement: str, params: tuple[object, ...] = ()) -> _CursorResult:
        normalized = " ".join(statement.split())
        self.events.append(f"sql:{normalized}")
        if normalized.startswith("SELECT * FROM chacha_keywords WHERE sync_id"):
            if not self.inserted:
                return _CursorResult()
            return _CursorResult(
                {
                    "id": 41,
                    "sync_id": str(params[0]),
                    "keyword": "Portable",
                    "parent_id": None,
                    "deleted": False,
                    "version": 1,
                }
            )
        if normalized.startswith("SELECT id, sync_id FROM chacha_keywords"):
            return _CursorResult()
        if normalized.startswith("INSERT INTO chacha_keywords"):
            self.inserted = True
        return _CursorResult()


class _PostgresProjectionDB:
    backend_type = BackendType.POSTGRESQL
    client_id = "owner-1"

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.connection = _PostgresProjectionConnection(events)

    @contextmanager
    def transaction(self):
        self.events.append("product:begin")
        yield self.connection
        self.events.append("product:commit")

    @staticmethod
    def _map_table_for_backend(table: str) -> str:
        return "chacha_keywords" if table == "keywords" else table

    @staticmethod
    def _get_current_utc_timestamp_iso() -> str:
        return "2026-08-08T00:00:00+00:00"


class _SyncApplyStore:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.state = None

    def get_object_state(self, *args):
        self.events.append("sync:read-state")
        return self.state

    def upsert_object_state(self, state):
        self.events.append("sync:write-state")
        self.state = state
        return state

    def mark_envelope_apply_status(self, server_cursor: int, *, apply_status: str, **kwargs):
        self.events.append(f"sync:mark-{apply_status}")


def test_postgres_v55_migration_uses_transactional_nullable_backfill_validation_and_constraints() -> None:
    backend = _PostgresMigrationBackend()
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db._local = type("Local", (), {})()
    db._backend = backend
    db._uses_shared_content_backend = False
    connection = object()

    db._migrate_from_v54_to_v55_postgres(connection)

    statements = [statement for statement, _, _ in backend.calls]
    assert all(call_connection is connection for _, _, call_connection in backend.calls)
    expected_tables = ("chacha_keywords", "keyword_collections", "note_folders")
    expected_indexes = (
        "idx_keywords_sync_id_unique",
        "idx_keyword_collections_sync_id_unique",
        "idx_note_folders_sync_id_unique",
    )
    for table, index_name in zip(expected_tables, expected_indexes, strict=True):
        add_index = next(
            index
            for index, statement in enumerate(statements)
            if statement == f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS sync_id TEXT"
        )
        updates = [
            (index, params)
            for index, (statement, params, _) in enumerate(backend.calls)
            if statement == f"UPDATE {table} SET sync_id = %s WHERE id = %s"
        ]
        validation_indexes = [
            index
            for index, statement in enumerate(statements)
            if statement.startswith("SELECT COUNT(*)") and f"FROM {table}" in statement
        ]
        not_null_index = statements.index(f"ALTER TABLE {table} ALTER COLUMN sync_id SET NOT NULL")
        unique_index = statements.index(
            f"CREATE UNIQUE INDEX IF NOT EXISTS {index_name} ON {table}(sync_id)"
        )

        assert len(updates) == 2
        assert add_index < min(index for index, _ in updates)
        assert max(index for index, _ in updates) < min(validation_indexes)
        assert max(validation_indexes) < not_null_index < unique_index
        generated = [str(params[0]) for _, params in updates if params is not None]
        assert len(set(generated)) == 2
        assert all(str(uuid.UUID(value)) == value and uuid.UUID(value).version == 4 for value in generated)
        assert [params[1] for _, params in updates if params is not None] == [11, 12]

    assert statements[-1].startswith("INSERT INTO db_schema_version")
    assert backend.calls[-1][1] == (CharactersRAGDB._SCHEMA_NAME, 55)
    assert any(
        "CREATE TABLE IF NOT EXISTS note_folder_sync_suppressions" in statement
        and "PRIMARY KEY(note_id, folder_id)" in statement
        for statement in statements
    )


def test_projection_apply_methods_use_one_transaction_and_snapshot_all_domains(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = CharactersRAGDB(str(tmp_path / "organization.sqlite"), client_id="projection-contract")
    store = NotesOrganizationSyncStore(db)
    original_transaction = db.transaction
    transaction_count = 0

    @contextmanager
    def counted_transaction():
        nonlocal transaction_count
        transaction_count += 1
        with original_transaction() as conn:
            yield conn

    monkeypatch.setattr(db, "transaction", counted_transaction)
    try:
        keyword_sync_id = str(uuid.uuid4())
        collection_sync_id = str(uuid.uuid4())
        folder_sync_id = str(uuid.uuid4())
        keyword = store.apply_resource(
            domain="notes.keyword",
            object_id=keyword_sync_id,
            operation="upsert",
            payload={"keyword": "portable"},
        )
        assert transaction_count == 1
        collection = store.apply_resource(
            domain="notes.keyword_collection",
            object_id=collection_sync_id,
            operation="upsert",
            payload={"name": "Portable", "parent_sync_id": None},
        )
        assert transaction_count == 2
        folder = store.apply_resource(
            domain="notes.folder",
            object_id=folder_sync_id,
            operation="upsert",
            payload={"name": "Portable", "parent_sync_id": None},
        )
        assert transaction_count == 3
        note_id = db.add_note(title="Portable snapshot", content="relationships")
        keyword_link_payload = {
            "subject_type": "note",
            "subject_id": note_id,
            "keyword_sync_id": keyword_sync_id,
        }
        keyword_link_id = organization_link_id(
            "notes.keyword_link", ["note", note_id, keyword_sync_id]
        )
        store.apply_relationship(
            domain="notes.keyword_link",
            object_id=keyword_link_id,
            operation="upsert",
            payload=keyword_link_payload,
            routing_metadata={"bootstrap_capture": True},
        )
        assert transaction_count == 5  # add_note and relationship each own one transaction

        snapshot = store.snapshot()
        assert {(resource.domain, resource.sync_id) for resource in snapshot.resources} >= {
            ("notes.keyword", keyword.sync_id),
            ("notes.keyword_collection", collection.sync_id),
            ("notes.folder", folder.sync_id),
        }
        assert any(
            relationship.domain == "notes.keyword_link"
            and relationship.object_id == keyword_link_id
            and relationship.payload == keyword_link_payload
            for relationship in snapshot.relationships
        )

        store.apply_relationship(
            domain="notes.keyword_link",
            object_id=keyword_link_id,
            operation="tombstone",
            payload=keyword_link_payload,
            routing_metadata={"bootstrap_capture": True},
        )
        assert all(
            relationship.object_id != keyword_link_id
            for relationship in store.snapshot().relationships
        )
    finally:
        db.close_connection()


def test_postgres_materializer_commits_product_sql_before_sync_apply_state() -> None:
    events: list[str] = []
    note_db = _PostgresProjectionDB(events)
    sync_store = _SyncApplyStore(events)
    envelope = SyncEnvelope(
        dataset_id="dataset-1",
        client_envelope_id="env-keyword",
        domain="notes.keyword",
        operation="upsert",
        object_id="11111111-1111-4111-8111-111111111111",
        server_cursor=7,
        object_revision=1,
        payload={"keyword": "Portable"},
        payload_hash="sha256:portable",
        status="accepted",
    )

    result = NotesOrganizationMaterializer(
        note_db,  # type: ignore[arg-type] - server-free PostgreSQL persistence contract.
        "notes.keyword",
    ).apply(
        envelope,
        store=sync_store,  # type: ignore[arg-type] - focused Sync ordering recorder.
    )

    assert result.status == "applied"
    insert_index = next(
        index
        for index, event in enumerate(events)
        if event.startswith("sql:INSERT INTO chacha_keywords")
    )
    assert events.index("product:begin") < insert_index < events.index("product:commit")
    assert events.index("product:commit") < events.index("sync:write-state")
    assert events.index("sync:write-state") < events.index("sync:mark-applied")
    insert_event = events[insert_index]
    assert "sync_id, keyword" in insert_event
    assert "deleted" in insert_event
