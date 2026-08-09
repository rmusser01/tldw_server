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
