from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.domain_adapters.notes import NotesDomainAdapter
from tldw_Server_API.app.core.Sync.v2.materializers.notes import NotesMaterializer
from tldw_Server_API.app.core.Sync.v2.models import SyncEnvelopeCreate
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

pytestmark = pytest.mark.integration


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
