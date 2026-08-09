from __future__ import annotations

import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseConfig,
    QueryResult,
)
from tldw_Server_API.app.core.DB_Management.chacha.organization_sync_store import (
    NotesOrganizationSyncStore,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.materializers.notes_organization import (
    NotesOrganizationMaterializer,
)
from tldw_Server_API.app.core.Sync.v2.models import SyncDomain, SyncEnvelope, SyncObjectState
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


class _PostgresDatasetLockBackend:
    config = DatabaseConfig(backend_type=BackendType.POSTGRESQL)

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
        return QueryResult(
            rows=[
                {
                    "dataset_id": "dataset-1",
                    "domain_set_json": '["notes.keyword"]',
                }
            ],
            rowcount=1,
        )


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


class _PostgresSqlIntentConnection:
    def __init__(self, connection: Any, events: list[str]) -> None:
        self.connection = connection
        self.events = events

    def execute(self, statement: str, params: tuple[object, ...] = ()):
        normalized = " ".join(statement.split())
        self.events.append(f"sql:{normalized}")
        translated = statement.replace("chacha_keywords", "keywords")
        return self.connection.execute(translated, params)


class _PostgresSqlIntentDB:
    """Run PostgreSQL projection branches on SQLite while recording SQL intent."""

    backend_type = BackendType.POSTGRESQL

    def __init__(self, db: CharactersRAGDB, events: list[str]) -> None:
        self.db = db
        self.events = events
        self.client_id = db.client_id

    @contextmanager
    def transaction(self):
        self.events.append("product:begin")
        try:
            with self.db.transaction() as connection:
                yield _PostgresSqlIntentConnection(connection, self.events)
        except Exception:
            self.events.append("product:rollback")
            raise
        self.events.append("product:commit")

    @staticmethod
    def _map_table_for_backend(table: str) -> str:
        return "chacha_keywords" if table == "keywords" else table

    def _get_current_utc_timestamp_iso(self) -> str:
        return self.db._get_current_utc_timestamp_iso()


class _LifecycleSyncStore:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.states: dict[tuple[str, str, str], SyncObjectState] = {}
        self.statuses: dict[int, tuple[str, str | None, str | None]] = {}

    def get_object_state(self, dataset_id: str, domain: str, object_id: str):
        self.events.append("sync:read-state")
        return self.states.get((dataset_id, domain, object_id))

    def upsert_object_state(self, state: SyncObjectState):
        self.events.append("sync:write-state")
        self.states[(state.dataset_id, state.domain, state.object_id)] = state
        return state

    def mark_envelope_apply_status(
        self,
        server_cursor: int,
        *,
        apply_status: str,
        apply_error_code: str | None = None,
        apply_error_message: str | None = None,
    ):
        self.events.append(f"sync:mark-{apply_status}")
        self.statuses[server_cursor] = (
            apply_status,
            apply_error_code,
            apply_error_message,
        )


_PG_KEYWORD_ID = "11111111-1111-4111-8111-111111111111"
_PG_COLLECTION_ID = "22222222-2222-4222-8222-222222222222"
_PG_PARENT_COLLECTION_ID = "33333333-3333-4333-8333-333333333333"
_PG_FOLDER_ID = "44444444-4444-4444-8444-444444444444"
_PG_PARENT_FOLDER_ID = "55555555-5555-4555-8555-555555555555"
_PG_NOTE_ID = "66666666-6666-4666-8666-666666666666"


def _pg_payload(domain: SyncDomain) -> dict[str, object]:
    return {
        "notes.keyword": {"keyword": "Portable"},
        "notes.keyword_link": {
            "subject_type": "note",
            "subject_id": _PG_NOTE_ID,
            "keyword_sync_id": _PG_KEYWORD_ID,
        },
        "notes.keyword_collection": {
            "name": "Projects",
            "parent_sync_id": _PG_PARENT_COLLECTION_ID,
        },
        "notes.keyword_collection_link": {
            "collection_sync_id": _PG_COLLECTION_ID,
            "keyword_sync_id": _PG_KEYWORD_ID,
        },
        "notes.folder": {"name": "Work", "parent_sync_id": _PG_PARENT_FOLDER_ID},
        "notes.folder_link": {
            "note_id": _PG_NOTE_ID,
            "folder_sync_id": _PG_FOLDER_ID,
        },
    }[domain]


def _pg_object_id(domain: SyncDomain, payload: dict[str, object]) -> str:
    if domain == "notes.keyword":
        return _PG_KEYWORD_ID
    if domain == "notes.keyword_collection":
        return _PG_COLLECTION_ID
    if domain == "notes.folder":
        return _PG_FOLDER_ID
    fields = {
        "notes.keyword_link": ("subject_type", "subject_id", "keyword_sync_id"),
        "notes.keyword_collection_link": ("collection_sync_id", "keyword_sync_id"),
        "notes.folder_link": ("note_id", "folder_sync_id"),
    }[domain]
    return organization_link_id(
        domain, [cast(str, payload[field_name]) for field_name in fields]
    )


def _pg_envelope(
    domain: SyncDomain,
    *,
    cursor: int,
    operation: str = "upsert",
    base: SyncObjectState | None = None,
    restore: bool = False,
) -> SyncEnvelope:
    payload = _pg_payload(domain)
    if operation == "tombstone" and domain in {
        "notes.keyword",
        "notes.keyword_collection",
        "notes.folder",
    }:
        payload = {}
    return SyncEnvelope(
        dataset_id="dataset-1",
        client_envelope_id=f"env-{domain}-{cursor}",
        domain=domain,
        operation=operation,
        object_id=_pg_object_id(domain, _pg_payload(domain)),
        server_cursor=cursor,
        base_server_cursor=base.latest_server_cursor if base else None,
        base_object_revision=base.object_revision if base else None,
        base_object_hash=base.object_hash if base else None,
        object_revision=cursor,
        payload=payload,
        payload_hash=f"sha256:{domain}:{cursor}",
        routing_metadata={"restore_intent": True} if restore else {},
        status="accepted",
    )


def _seed_pg_dependencies(db: _PostgresSqlIntentDB, domain: SyncDomain) -> None:
    projection = NotesOrganizationSyncStore(db)  # type: ignore[arg-type]
    if domain in {"notes.keyword_link", "notes.folder_link"}:
        db.db.add_note("Linked", "Body", note_id=_PG_NOTE_ID)
    if domain in {"notes.keyword_link", "notes.keyword_collection_link"}:
        projection.apply_resource(
            domain="notes.keyword",
            object_id=_PG_KEYWORD_ID,
            operation="upsert",
            payload={"keyword": "Dependency"},
        )
    if domain in {"notes.keyword_collection", "notes.keyword_collection_link"}:
        projection.apply_resource(
            domain="notes.keyword_collection",
            object_id=_PG_PARENT_COLLECTION_ID,
            operation="upsert",
            payload={"name": "Root", "parent_sync_id": None},
        )
    if domain == "notes.keyword_collection_link":
        projection.apply_resource(
            domain="notes.keyword_collection",
            object_id=_PG_COLLECTION_ID,
            operation="upsert",
            payload={"name": "Projects", "parent_sync_id": _PG_PARENT_COLLECTION_ID},
        )
    if domain in {"notes.folder", "notes.folder_link"}:
        projection.apply_resource(
            domain="notes.folder",
            object_id=_PG_PARENT_FOLDER_ID,
            operation="upsert",
            payload={"name": "Root", "parent_sync_id": None},
        )
    if domain == "notes.folder_link":
        projection.apply_resource(
            domain="notes.folder",
            object_id=_PG_FOLDER_ID,
            operation="upsert",
            payload={"name": "Work", "parent_sync_id": _PG_PARENT_FOLDER_ID},
        )


def _pg_projection_is_active(db: CharactersRAGDB, domain: SyncDomain) -> bool:
    sql = {
        "notes.keyword": "SELECT deleted AS active FROM keywords WHERE sync_id = ?",
        "notes.keyword_link": (
            "SELECT 0 AS active FROM note_keywords l "
            "JOIN keywords k ON k.id = l.keyword_id WHERE l.note_id = ? AND k.sync_id = ?"
        ),
        "notes.keyword_collection": (
            "SELECT deleted AS active FROM keyword_collections WHERE sync_id = ?"
        ),
        "notes.keyword_collection_link": (
            "SELECT 0 AS active FROM collection_keywords l "
            "JOIN keyword_collections c ON c.id = l.collection_id "
            "JOIN keywords k ON k.id = l.keyword_id "
            "WHERE c.sync_id = ? AND k.sync_id = ?"
        ),
        "notes.folder": "SELECT deleted AS active FROM note_folders WHERE sync_id = ?",
        "notes.folder_link": (
            "SELECT 0 AS active FROM note_folder_memberships l "
            "JOIN note_folders f ON f.id = l.folder_id WHERE l.note_id = ? AND f.sync_id = ?"
        ),
    }[domain]
    params = {
        "notes.keyword": (_PG_KEYWORD_ID,),
        "notes.keyword_link": (_PG_NOTE_ID, _PG_KEYWORD_ID),
        "notes.keyword_collection": (_PG_COLLECTION_ID,),
        "notes.keyword_collection_link": (_PG_COLLECTION_ID, _PG_KEYWORD_ID),
        "notes.folder": (_PG_FOLDER_ID,),
        "notes.folder_link": (_PG_NOTE_ID, _PG_FOLDER_ID),
    }[domain]
    with db.transaction() as connection:
        row = connection.execute(sql, params).fetchone()
    return row is not None and not bool(row["active"])


def _assert_product_commit_precedes_sync_state(events: list[str]) -> None:
    assert events.index("product:commit") < events.index("sync:write-state")
    assert events.index("sync:write-state") < events.index("sync:mark-applied")


def test_postgres_append_gate_uses_dataset_row_for_update_sql() -> None:
    backend = _PostgresDatasetLockBackend()
    db = SyncDatabase.__new__(SyncDatabase)
    db.backend = cast(Any, backend)
    connection = object()

    row = db._require_dataset_domain_for_update(
        "dataset-1",
        "notes.keyword",
        connection=connection,
    )

    assert row["dataset_id"] == "dataset-1"
    assert backend.calls == [
        (
            "SELECT * FROM sync_datasets WHERE dataset_id = ? FOR UPDATE",
            ("dataset-1",),
            connection,
        )
    ]


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


@pytest.mark.parametrize(
    ("domain", "table"),
    [
        ("notes.keyword", "chacha_keywords"),
        ("notes.keyword_link", "note_keywords"),
        ("notes.keyword_collection", "keyword_collections"),
        ("notes.keyword_collection_link", "collection_keywords"),
        ("notes.folder", "note_folders"),
        ("notes.folder_link", "note_folder_memberships"),
    ],
)
def test_postgres_materializer_lifecycle_has_equivalent_sql_intent_and_commit_order(
    domain: SyncDomain,
    table: str,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    sqlite_db = CharactersRAGDB(
        str(tmp_path / f"postgres-{domain}.sqlite"),
        client_id="postgres-contract",
    )
    note_db = _PostgresSqlIntentDB(sqlite_db, events)
    sync_store = _LifecycleSyncStore(events)
    materializer = NotesOrganizationMaterializer(
        note_db,  # type: ignore[arg-type] - server-free PostgreSQL SQL contract.
        domain,
    )
    try:
        _seed_pg_dependencies(note_db, domain)
        events.clear()

        created = _pg_envelope(domain, cursor=1)
        assert materializer.apply(
            created,
            store=sync_store,  # type: ignore[arg-type] - ordering recorder.
        ).status == "applied"
        assert any(f"INSERT INTO {table}" in event for event in events)
        if domain not in {"notes.keyword", "notes.keyword_collection", "notes.folder"}:
            assert any(
                f"INSERT INTO {table}" in event and "ON CONFLICT DO NOTHING" in event
                for event in events
            )
        _assert_product_commit_precedes_sync_state(events)
        assert _pg_projection_is_active(sqlite_db, domain)

        created_state = sync_store.get_object_state(
            created.dataset_id, created.domain, created.object_id
        )
        assert created_state is not None
        events.clear()
        tombstone = _pg_envelope(
            domain,
            cursor=2,
            operation="tombstone",
            base=created_state,
        )
        assert materializer.apply(
            tombstone,
            store=sync_store,  # type: ignore[arg-type] - ordering recorder.
        ).status == "applied"
        expected_tombstone_table = (
            "note_folder_sync_suppressions"
            if domain == "notes.folder_link"
            else table
        )
        assert any(expected_tombstone_table in event for event in events)
        _assert_product_commit_precedes_sync_state(events)
        assert not _pg_projection_is_active(sqlite_db, domain)

        tombstone_state = sync_store.get_object_state(
            tombstone.dataset_id, tombstone.domain, tombstone.object_id
        )
        assert tombstone_state is not None
        events.clear()
        restored = _pg_envelope(
            domain,
            cursor=3,
            base=tombstone_state,
            restore=True,
        )
        assert materializer.apply(
            restored,
            store=sync_store,  # type: ignore[arg-type] - ordering recorder.
        ).status == "applied"
        assert any(table in event for event in events)
        if domain not in {"notes.keyword", "notes.keyword_collection", "notes.folder"}:
            assert any(
                f"INSERT INTO {table}" in event and "ON CONFLICT DO NOTHING" in event
                for event in events
            )
        _assert_product_commit_precedes_sync_state(events)
        assert _pg_projection_is_active(sqlite_db, domain)

        events.clear()
        assert materializer.apply(
            restored,
            store=sync_store,  # type: ignore[arg-type] - ordering recorder.
        ).status == "applied"
        assert "product:begin" not in events
        assert events[-1] == "sync:mark-applied"
    finally:
        sqlite_db.close_connection()


@pytest.mark.parametrize(
    "domain",
    (
        "notes.keyword_link",
        "notes.keyword_collection",
        "notes.keyword_collection_link",
        "notes.folder",
        "notes.folder_link",
    ),
)
def test_postgres_materializer_dependency_failure_rolls_back_before_retryable_status(
    domain: SyncDomain,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    sqlite_db = CharactersRAGDB(
        str(tmp_path / f"postgres-missing-{domain}.sqlite"),
        client_id="postgres-contract",
    )
    note_db = _PostgresSqlIntentDB(sqlite_db, events)
    sync_store = _LifecycleSyncStore(events)
    try:
        result = NotesOrganizationMaterializer(
            note_db,  # type: ignore[arg-type] - server-free PostgreSQL SQL contract.
            domain,
        ).apply(
            _pg_envelope(domain, cursor=10),
            store=sync_store,  # type: ignore[arg-type] - ordering recorder.
        )

        assert result.status == "failed"
        assert result.error_code == "notes_organization_projection_failed"
        assert result.message == (
            "Notes organization dependency or hierarchy validation failed"
        )
        assert events.index("product:rollback") < events.index("sync:mark-failed")
        assert "product:commit" not in events
        assert not sync_store.states
    finally:
        sqlite_db.close_connection()


@pytest.mark.parametrize(
    ("domain", "table"),
    [
        ("notes.keyword", "keywords"),
        ("notes.keyword_link", "note_keywords"),
        ("notes.keyword_collection", "keyword_collections"),
        ("notes.keyword_collection_link", "collection_keywords"),
        ("notes.folder", "note_folders"),
        ("notes.folder_link", "note_folder_memberships"),
    ],
)
def test_postgres_materializer_product_failure_rolls_back_before_retryable_status(
    domain: SyncDomain,
    table: str,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    sqlite_db = CharactersRAGDB(
        str(tmp_path / f"postgres-failure-{domain}.sqlite"),
        client_id="postgres-contract",
    )
    note_db = _PostgresSqlIntentDB(sqlite_db, events)
    sync_store = _LifecycleSyncStore(events)
    try:
        _seed_pg_dependencies(note_db, domain)
        trigger = f"fail_{table}_insert"
        with sqlite_db.transaction() as connection:
            connection.execute(
                f"CREATE TRIGGER {trigger} BEFORE INSERT ON {table} "  # nosec B608 - fixed test table matrix
                "BEGIN SELECT RAISE(ABORT, 'LEAK-SENTINEL-PG-BACKEND'); END"
            )
        events.clear()

        result = NotesOrganizationMaterializer(
            note_db,  # type: ignore[arg-type] - server-free PostgreSQL SQL contract.
            domain,
        ).apply(
            _pg_envelope(domain, cursor=20),
            store=sync_store,  # type: ignore[arg-type] - ordering recorder.
        )

        assert result.status == "failed"
        assert result.error_code == "notes_organization_projection_failed"
        assert result.message == "Notes organization product database operation failed"
        assert "LEAK-SENTINEL-PG-BACKEND" not in result.message
        assert events.index("product:rollback") < events.index("sync:mark-failed")
        assert "product:commit" not in events
        assert not sync_store.states
        assert not _pg_projection_is_active(sqlite_db, domain)
    finally:
        sqlite_db.close_connection()
