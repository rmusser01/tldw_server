from __future__ import annotations

import inspect

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import (
    SYNC_POSTGRES_SCHEMA,
    SyncDatabase,
)
from tldw_Server_API.app.core.Sync.v2.adapters import SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.domain_adapters.notes import NotesDomainAdapter
from tldw_Server_API.app.core.Sync.v2.errors import SyncDatasetNotFoundError
from tldw_Server_API.app.core.Sync.v2.materializers.notes import NotesMaterializer
from tldw_Server_API.app.core.Sync.v2.models import (
    SyncAttachmentRevisionBindingCreate,
    SyncDatasetCreate,
    SyncEnvelopeCreate,
)
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

pytestmark = pytest.mark.integration


def test_postgres_attachment_binding_and_storage_namespace_sql_plan_contracts() -> None:
    compact_schema = " ".join(SYNC_POSTGRES_SCHEMA.split())
    read_owner_guard = getattr(
        SyncDatabase,
        "_require_attachment_binding_dataset_owner",
        None,
    )
    mutation_owner_guard = getattr(
        SyncDatabase,
        "_require_attachment_binding_dataset_owner_for_update",
        None,
    )
    assert read_owner_guard is not None
    assert mutation_owner_guard is not None
    read_owner_source = " ".join(inspect.getsource(read_owner_guard).split())
    mutation_owner_source = " ".join(inspect.getsource(mutation_owner_guard).split())
    ensure_source = inspect.getsource(SyncDatabase._ensure_attachment_binding_tables)
    lookup_source = inspect.getsource(SyncDatabase.get_attachment_revision_binding)
    unresolved_source = inspect.getsource(
        SyncDatabase.list_unresolved_attachment_revision_bindings
    )
    resolve_source = inspect.getsource(SyncDatabase.resolve_attachment_revision_binding)
    blob_lookup_source = inspect.getsource(
        SyncDatabase._require_exact_available_blob_for_binding
    )
    namespace_source = inspect.getsource(SyncDatabase.get_or_create_storage_namespace)
    acceptance_source = inspect.getsource(
        SyncDatabase._create_attachment_binding_for_envelope
    )
    completion_source = inspect.getsource(SyncDatabase.complete_blob_upload)

    assert "CREATE TABLE IF NOT EXISTS sync_attachment_revision_bindings" in compact_schema
    assert "PRIMARY KEY (dataset_id, attachment_id, attachment_revision)" in compact_schema
    assert "attachment_revision > 0" in compact_schema
    assert "size_bytes > 0" in compact_schema
    assert "blob_hash ~ '^sha256:[0-9a-f]{64}$'" in compact_schema
    assert "availability_at_acceptance IN ('available', 'metadata_only')" in compact_schema
    assert (
        "CREATE INDEX IF NOT EXISTS idx_sync_attachment_bindings_unresolved ON "
        "sync_attachment_revision_bindings(dataset_id, establishing_server_cursor, "
        "attachment_id, attachment_revision) WHERE resolved_blob_id IS NULL AND "
        "retention_released_at IS NULL"
    ) in compact_schema
    assert (
        "CREATE INDEX IF NOT EXISTS idx_sync_attachment_bindings_blob ON "
        "sync_attachment_revision_bindings(dataset_id, resolved_blob_id)"
    ) in compact_schema
    assert (
        "CREATE INDEX IF NOT EXISTS idx_sync_attachment_bindings_pending_digest ON "
        "sync_attachment_revision_bindings(dataset_id, blob_hash, size_bytes, "
        "establishing_server_cursor, attachment_id, attachment_revision) WHERE "
        "resolved_blob_id IS NULL AND retention_released_at IS NULL"
    ) in compact_schema
    assert "CREATE TABLE IF NOT EXISTS sync_dataset_storage_namespaces" in compact_schema
    assert "storage_namespace_id ~ '^[0-9a-f]{32}$'" in compact_schema
    assert (
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_sync_dataset_storage_namespace_id ON "
        "sync_dataset_storage_namespaces(storage_namespace_id)"
    ) in compact_schema
    assert (
        "CREATE INDEX IF NOT EXISTS idx_sync_dataset_storage_namespaces_owner ON "
        "sync_dataset_storage_namespaces(owner_user_id, dataset_id)"
    ) in compact_schema
    assert "sync_attachment_revision_bindings" in ensure_source
    assert "sync_dataset_storage_namespaces" in ensure_source
    assert "WHERE dataset_id = ? AND owner_user_id = ?" in read_owner_source
    assert "FOR UPDATE" not in read_owner_source
    assert "WHERE dataset_id = ? AND owner_user_id = ?" in mutation_owner_source
    assert "FOR UPDATE" in mutation_owner_source
    assert "dataset_id = ? AND attachment_id = ? AND attachment_revision = ?" in " ".join(
        lookup_source.split()
    )
    assert "LIMIT ?" in unresolved_source
    assert "ORDER BY establishing_server_cursor, attachment_id, attachment_revision" in " ".join(
        unresolved_source.split()
    )
    assert "resolved_blob_id IS NULL" in unresolved_source
    assert "retention_released_at IS NULL" in unresolved_source
    assert "FOR UPDATE" in resolve_source
    assert "WHERE dataset_id = ? AND blob_id = ?" in " ".join(
        blob_lookup_source.split()
    )
    assert "attachment_id = ?" not in blob_lookup_source
    assert "owner_user_id" in namespace_source
    assert "FOR UPDATE" in namespace_source
    assert "owner_user_id" in lookup_source
    assert "owner_user_id" in unresolved_source
    assert "FOR UPDATE" in acceptance_source
    assert "_get_dataset_row_for_update" in completion_source
    assert "_resolve_pending_bindings_for_blob" in completion_source


def test_postgres_attachment_binding_lookup_and_unresolved_page_use_declared_indexes(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = SyncDatabase(backend=backend)
    store = SyncV2Store(db)
    try:
        store.enroll_dataset(
            SyncDatasetCreate(
                dataset_id="dataset-binding-pg",
                owner_user_id="owner-binding-pg",
                scope_type="personal",
                encryption_policy="server_trusted_v1",
                domains=["notes.note", "attachment.ref"],
            )
        )
        store.create_attachment_revision_binding(
            SyncAttachmentRevisionBindingCreate(
                dataset_id="dataset-binding-pg",
                attachment_id="11111111-1111-4111-8111-111111111111",
                attachment_revision=1,
                blob_hash="sha256:" + "a" * 64,
                size_bytes=1,
                establishing_server_cursor=1,
                availability_at_acceptance="metadata_only",
            ),
            owner_user_id="owner-binding-pg",
        )
        with pytest.raises(SyncDatasetNotFoundError):
            store.get_attachment_revision_binding(
                "dataset-binding-pg",
                "11111111-1111-4111-8111-111111111111",
                1,
                owner_user_id="other-owner-pg",
            )
        with pytest.raises(SyncDatasetNotFoundError):
            store.list_unresolved_attachment_revision_bindings(
                "dataset-binding-pg",
                owner_user_id="other-owner-pg",
            )
        with pytest.raises(SyncDatasetNotFoundError):
            store.create_attachment_revision_binding(
                SyncAttachmentRevisionBindingCreate(
                    dataset_id="dataset-binding-pg",
                    attachment_id="22222222-2222-4222-8222-222222222222",
                    attachment_revision=1,
                    blob_hash="sha256:" + "b" * 64,
                    size_bytes=1,
                    establishing_server_cursor=2,
                    availability_at_acceptance="metadata_only",
                ),
                owner_user_id="other-owner-pg",
            )
        with pytest.raises(SyncDatasetNotFoundError):
            store.resolve_attachment_revision_binding(
                "dataset-binding-pg",
                "11111111-1111-4111-8111-111111111111",
                1,
                blob_id="other-owner-blob",
                owner_user_id="other-owner-pg",
            )
        with pytest.raises(SyncDatasetNotFoundError):
            store.release_attachment_revision_binding(
                "dataset-binding-pg",
                "11111111-1111-4111-8111-111111111111",
                1,
                released_at="2026-08-11T21:00:00+00:00",
                owner_user_id="other-owner-pg",
            )
        with backend.transaction() as conn:
            db.execute("SET LOCAL enable_seqscan = off", connection=conn)
            lookup_rows = db.execute(
                "EXPLAIN SELECT attachment_id FROM sync_attachment_revision_bindings "
                "WHERE dataset_id = ? AND attachment_id = ? AND attachment_revision = ?",
                (
                    "dataset-binding-pg",
                    "11111111-1111-4111-8111-111111111111",
                    1,
                ),
                connection=conn,
            ).rows
            page_rows = db.execute(
                "EXPLAIN SELECT attachment_id FROM sync_attachment_revision_bindings "
                "WHERE dataset_id = ? AND resolved_blob_id IS NULL "
                "AND retention_released_at IS NULL AND establishing_server_cursor > ? "
                "ORDER BY establishing_server_cursor, attachment_id, attachment_revision "
                "LIMIT ?",
                ("dataset-binding-pg", 0, 1000),
                connection=conn,
            ).rows
            namespace_rows = db.execute(
                "EXPLAIN SELECT storage_namespace_id FROM sync_dataset_storage_namespaces "
                "WHERE owner_user_id = ? AND dataset_id = ?",
                ("owner-binding-pg", "dataset-binding-pg"),
                connection=conn,
            ).rows
            digest_rows = db.execute(
                "EXPLAIN SELECT attachment_id FROM sync_attachment_revision_bindings "
                "WHERE dataset_id = ? AND blob_hash = ? AND size_bytes = ? "
                "AND resolved_blob_id IS NULL AND retention_released_at IS NULL "
                "ORDER BY establishing_server_cursor, attachment_id, attachment_revision "
                "LIMIT 1000",
                ("dataset-binding-pg", "sha256:" + "a" * 64, 1),
                connection=conn,
            ).rows
        lookup_plan = " ".join(
            str(next(iter(row.values()))) for row in lookup_rows
        )
        page_plan = " ".join(str(next(iter(row.values()))) for row in page_rows)
        namespace_plan = " ".join(
            str(next(iter(row.values()))) for row in namespace_rows
        )
        digest_plan = " ".join(str(next(iter(row.values()))) for row in digest_rows)
        assert "sync_attachment_revision_bindings_pkey" in lookup_plan
        assert "idx_sync_attachment_bindings_unresolved" in page_plan
        assert "idx_sync_dataset_storage_namespaces_owner" in namespace_plan
        assert "idx_sync_attachment_bindings_pending_digest" in digest_plan
    finally:
        if db.backend_type == BackendType.POSTGRESQL:
            backend.get_pool().close_all()


def _ready_encryption():
    return server_trusted_encryption_status_from_config(
        mode="managed_storage",
        server_trusted_enabled=True,
        auth_mode="multi_user",
    )


def _envelope(**overrides) -> SyncEnvelopeCreate:
    values = {
        "dataset_id": "dataset-postgres",
        "client_envelope_id": "env-create",
        "domain": "notes.note",
        "operation": "upsert",
        "object_id": "note-postgres",
        "device_id": "device-postgres",
        "client_sequence": 1,
        "schema_version": 1,
        "object_revision": 1,
        "payload": {
            "title": "  PostgreSQL π note  ",
            "content": "# Exact Markdown\n\n[[Linked note]] & <source> 🧠\n",
            "conversation_id": None,
            "message_id": None,
        },
        "payload_hash": "sha256:pg-note-v1",
        "created_at_client": "2026-05-23T18:12:44+00:00",
        "deleted": False,
        "encryption_metadata": {"policy": "server_trusted_v1"},
    }
    values.update(overrides)
    return SyncEnvelopeCreate(**values)


def _push(service: SyncV2Service, envelope: SyncEnvelopeCreate):
    return service.push(
        user_id="user-postgres",
        dataset_id="dataset-postgres",
        device_id="device-postgres",
        envelopes=[envelope],
    )


def test_postgresql_notes_sync_contract_round_trip(
    tmp_path,
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    note_db = CharactersRAGDB(
        db_path=":memory:",
        client_id="user-postgres",
        backend=backend,
    )
    try:
        conversation_id = note_db.add_conversation(
            {"id": "conversation-postgres", "title": "Source conversation"}
        )
        message_id = note_db.add_message(
            {
                "id": "message-postgres",
                "conversation_id": conversation_id,
                "sender": "user",
                "content": "Source message",
            }
        )
        service = SyncV2Service(
            store=SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "Sync_v2.db")),
            adapters=SyncAdapterRegistry([NotesDomainAdapter()]),
            materializers={"notes.note": NotesMaterializer(note_db)},
            clock=lambda: "2026-05-23T18:12:00+00:00",
            settings=SyncV2Settings(
                supported_domains=["notes.note"],
                operations={"notes.note": ["upsert", "tombstone"]},
                server_trusted_encryption=_ready_encryption(),
            ),
        )
        service.register_device(
            user_id="user-postgres",
            display_name="PostgreSQL client",
            client_type="chatbook",
            device_id="device-postgres",
        )
        service.enroll_dataset(
            user_id="user-postgres",
            dataset_id="dataset-postgres",
            domains=["notes.note"],
        )

        create = _envelope(
            payload={
                "title": "  PostgreSQL π note  ",
                "content": "# Exact Markdown\n\n[[Linked note]] & <source> 🧠\n",
                "conversation_id": conversation_id,
                "message_id": message_id,
            }
        )
        assert [item.client_envelope_id for item in _push(service, create).accepted] == [
            "env-create"
        ]
        created = service.store.get_object_state(
            "dataset-postgres", "notes.note", "note-postgres"
        )
        assert created is not None

        update = _envelope(
            client_envelope_id="env-update",
            client_sequence=2,
            base_server_cursor=created.latest_server_cursor,
            base_object_revision=created.object_revision,
            base_object_hash=created.object_hash,
            object_revision=2,
            payload={
                "title": "  PostgreSQL π note revised  ",
                "content": "# Revised exactly\n\n- one\n- two\n",
                "conversation_id": conversation_id,
                "message_id": message_id,
            },
            payload_hash="sha256:pg-note-v2",
        )
        assert [item.client_envelope_id for item in _push(service, update).accepted] == [
            "env-update"
        ]
        updated = service.store.get_object_state(
            "dataset-postgres", "notes.note", "note-postgres"
        )
        assert updated is not None

        tombstone = _envelope(
            client_envelope_id="env-delete",
            client_sequence=3,
            operation="tombstone",
            base_server_cursor=updated.latest_server_cursor,
            base_object_revision=updated.object_revision,
            base_object_hash=updated.object_hash,
            object_revision=3,
            payload={"deleted_at": "2026-05-23T18:35:00+00:00", "reason": "user_deleted"},
            payload_hash="sha256:pg-note-delete",
            deleted=True,
        )
        assert [item.client_envelope_id for item in _push(service, tombstone).accepted] == [
            "env-delete"
        ]
        deleted = service.store.get_object_state(
            "dataset-postgres", "notes.note", "note-postgres"
        )
        assert deleted is not None and deleted.deleted is True

        stale = _envelope(
            client_envelope_id="env-stale",
            client_sequence=4,
            base_server_cursor=updated.latest_server_cursor,
            base_object_revision=updated.object_revision,
            base_object_hash=updated.object_hash,
            object_revision=3,
            payload={"title": "Stale", "content": "Must not resurrect."},
            payload_hash="sha256:pg-note-stale",
        )
        stale_result = _push(service, stale)
        assert stale_result.accepted == []
        assert len(stale_result.conflicts) == 1

        restore = _envelope(
            client_envelope_id="env-restore",
            client_sequence=5,
            base_server_cursor=deleted.latest_server_cursor,
            base_object_revision=deleted.object_revision,
            base_object_hash=deleted.object_hash,
            object_revision=4,
            payload={
                "title": "  PostgreSQL π note revised  ",
                "content": "# Revised exactly\n\n- one\n- two\n",
                "conversation_id": conversation_id,
                "message_id": message_id,
            },
            payload_hash="sha256:pg-note-restored",
            routing_metadata={"restore_intent": True},
        )
        assert [item.client_envelope_id for item in _push(service, restore).accepted] == [
            "env-restore"
        ]
        envelope_count = len(
            service.store.list_envelopes_after(
                "dataset-postgres", 0, domains=["notes.note"], limit=10
            )
        )
        assert [item.client_envelope_id for item in _push(service, restore).accepted] == [
            "env-restore"
        ]
        assert len(
            service.store.list_envelopes_after(
                "dataset-postgres", 0, domains=["notes.note"], limit=10
            )
        ) == envelope_count

        note = note_db.get_note_by_id("note-postgres")
        assert note is not None
        assert note["title"] == "  PostgreSQL π note revised  "
        assert note["content"] == "# Revised exactly\n\n- one\n- two\n"
        assert note["conversation_id"] == conversation_id
        assert note["message_id"] == message_id
        state = service.store.get_object_state(
            "dataset-postgres", "notes.note", "note-postgres"
        )
        assert state is not None
        assert state.deleted is False
        assert state.object_revision == 4
    finally:
        note_db.close_connection()
        if note_db.backend_type == BackendType.POSTGRESQL:
            backend.get_pool().close_all()
