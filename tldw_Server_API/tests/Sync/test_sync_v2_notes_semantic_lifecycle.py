"""Sync v2 Notes semantic lifecycle authority contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.chacha.note_semantic_models import (
    SemanticDimensionState,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import StaticSyncAdapter, SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.domain_adapters.notes import NotesDomainAdapter
from tldw_Server_API.app.core.Sync.v2.materializers.notes import NotesMaterializer
from tldw_Server_API.app.core.Sync.v2.models import SyncEnvelopeCreate
from tldw_Server_API.app.core.Sync.v2.security import server_trusted_encryption_status_from_config
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

DATASET_ID = "dataset-1"
OWNER_ID = "server-user-1"
NOW = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)


def _ready_encryption():
    return server_trusted_encryption_status_from_config(
        mode="managed_storage", server_trusted_enabled=True, auth_mode="multi_user"
    )


@pytest.fixture()
def chacha_db(tmp_path: Path) -> CharactersRAGDB:
    database = CharactersRAGDB(str(tmp_path / "ChaChaNotes.db"), client_id=OWNER_ID)
    yield database
    database.close_all_connections()


@pytest.fixture()
def sync_service(tmp_path: Path, chacha_db: CharactersRAGDB) -> SyncV2Service:
    service = SyncV2Service(
        store=SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "Sync_v2.db")),
        adapters=SyncAdapterRegistry(
            [StaticSyncAdapter(domain="notes.note", supported_adapter_versions={1})]
        ),
        materializers={"notes.note": NotesMaterializer(chacha_db)},
        clock=lambda: "2026-08-29T12:00:00+00:00",
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(
            supported_domains=["notes.note"],
            operations={"notes.note": ["upsert", "tombstone"]},
            server_trusted_encryption=_ready_encryption(),
        ),
    )
    service.register_device(
        user_id="user-1", display_name="Laptop", client_type="chatbook", device_id="device-1"
    )
    service.enroll_dataset(user_id="user-1", dataset_id=DATASET_ID, domains=["notes.note"])
    return service


def _activate_semantic_generation(db: CharactersRAGDB) -> str:
    config = db.note_semantic_store.create_configuration(
        dataset_id=DATASET_ID,
        capability_revision="capability-v1",
        disclosure_hash="disclosure-v1",
        provider="provider-a",
        model="model-a",
        endpoint_origin_revision="origin-v1",
        endpoint_origin_display="https://api.example.test",
        data_boundary="provider",
        vector_backend="chromadb",
        storage_boundary="server_local",
        storage_label="local semantic vectors",
        normalization_version="normalization-v1",
        chunker_version="chunker-v1",
        now=NOW,
    )
    enabled = db.note_semantic_store.enable_configuration(
        dataset_id=DATASET_ID,
        expected_configuration_revision=config.configuration_revision,
        capability_revision="capability-v1",
        now=NOW,
    )
    assert enabled is not None
    pending = db.note_semantic_store.create_generation(
        dataset_id=DATASET_ID,
        configuration_revision=enabled.configuration_revision,
        compatibility_hash=None,
        dimension_state=SemanticDimensionState.PENDING,
        dimensions=None,
        root_job_id="job-1",
        now=NOW,
    )
    resolved = db.note_semantic_store.resolve_generation_dimensions(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        expected_configuration_revision=enabled.configuration_revision,
        dimensions=768,
        compatibility_hash="compatibility-v1",
        now=NOW,
    )
    assert resolved is not None
    assert db.note_semantic_store.activate_generation(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        expected_configuration_revision=resolved.configuration_revision,
        publication_receipt="receipt-1",
        now=NOW,
    ) is not None
    return pending.id


def _envelope(**overrides: object) -> SyncEnvelopeCreate:
    values: dict[str, object] = {
        "dataset_id": DATASET_ID,
        "client_envelope_id": "env-create",
        "domain": "notes.note",
        "operation": "upsert",
        "object_id": "note-1",
        "device_id": "device-1",
        "client_sequence": 1,
        "schema_version": 1,
        "object_revision": 1,
        "payload": {"title": "Trip notes", "content": "Body"},
        "payload_hash": "sha256:note-v1",
        "created_at_client": "2026-08-29T12:00:00+00:00",
        "deleted": False,
        "encryption_metadata": {"policy": "server_trusted_v1"},
    }
    values.update(overrides)
    return SyncEnvelopeCreate(**values)


def _push(service: SyncV2Service, envelope: SyncEnvelopeCreate):
    return service.push(
        user_id="user-1", dataset_id=DATASET_ID, device_id="device-1", envelopes=[envelope]
    )


def test_sync_note_mutations_use_envelope_dataset_for_semantic_ledger(
    sync_service: SyncV2Service, chacha_db: CharactersRAGDB
) -> None:
    generation_id = _activate_semantic_generation(chacha_db)

    assert [item.client_envelope_id for item in _push(sync_service, _envelope()).accepted] == ["env-create"]
    with chacha_db.transaction() as conn:
        state = conn.execute(
            "SELECT content_version,dirty_generation,state FROM note_semantic_note_state "
            "WHERE owner_user_id=? AND dataset_id=? AND note_id=?",
            (OWNER_ID, DATASET_ID, "note-1"),
        ).fetchone()
        work = conn.execute(
            "SELECT kind,generation_id FROM note_semantic_work "
            "WHERE owner_user_id=? AND dataset_id=? AND note_id=?",
            (OWNER_ID, DATASET_ID, "note-1"),
        ).fetchone()
    assert tuple(state) == (1, 1, "pending")
    assert tuple(work) == ("index_note", generation_id)

    head = sync_service.store.get_object_state(DATASET_ID, "notes.note", "note-1")
    assert head is not None
    tombstone = _envelope(
        client_envelope_id="env-delete",
        client_sequence=2,
        operation="tombstone",
        object_revision=2,
        payload={"deleted_at": "2026-08-29T12:01:00+00:00", "reason": "user_deleted"},
        payload_hash="sha256:note-v2",
        base_server_cursor=head.latest_server_cursor,
        base_object_revision=head.object_revision,
        base_object_hash=head.object_hash,
        deleted=True,
    )
    assert [item.client_envelope_id for item in _push(sync_service, tombstone).accepted] == ["env-delete"]
    with chacha_db.transaction() as conn:
        state = conn.execute(
            "SELECT content_version,dirty_generation,state FROM note_semantic_note_state "
            "WHERE owner_user_id=? AND dataset_id=? AND note_id=?",
            (OWNER_ID, DATASET_ID, "note-1"),
        ).fetchone()
        work = conn.execute(
            "SELECT kind,generation_id,dirty_generation FROM note_semantic_work "
            "WHERE owner_user_id=? AND dataset_id=? AND note_id=?",
            (OWNER_ID, DATASET_ID, "note-1"),
        ).fetchone()
    assert tuple(state) == (2, 2, "tombstoned")
    assert tuple(work) == ("delete_note_vectors", generation_id, 2)

    sync_service.adapters = SyncAdapterRegistry([NotesDomainAdapter()])
    tombstone_head = sync_service.store.get_object_state(DATASET_ID, "notes.note", "note-1")
    assert tombstone_head is not None
    restore = _envelope(
        client_envelope_id="env-restore",
        client_sequence=3,
        object_revision=3,
        payload={"title": "Trip notes", "content": "Body"},
        payload_hash="sha256:note-v3",
        base_server_cursor=tombstone_head.latest_server_cursor,
        base_object_revision=tombstone_head.object_revision,
        base_object_hash=tombstone_head.object_hash,
        routing_metadata={"restore_intent": True},
    )
    assert [item.client_envelope_id for item in _push(sync_service, restore).accepted] == ["env-restore"]
    with chacha_db.transaction() as conn:
        state = conn.execute(
            "SELECT content_version,dirty_generation,state FROM note_semantic_note_state "
            "WHERE owner_user_id=? AND dataset_id=? AND note_id=?",
            (OWNER_ID, DATASET_ID, "note-1"),
        ).fetchone()
        work = conn.execute(
            "SELECT kind,generation_id,dirty_generation FROM note_semantic_work "
            "WHERE owner_user_id=? AND dataset_id=? AND note_id=?",
            (OWNER_ID, DATASET_ID, "note-1"),
        ).fetchone()
    assert tuple(state) == (3, 3, "pending")
    assert tuple(work) == ("index_note", generation_id, 3)


def test_notes_import_style_fresh_database_has_no_server_local_semantic_state(tmp_path: Path) -> None:
    database = CharactersRAGDB(str(tmp_path / "imported-notes.sqlite"), client_id=OWNER_ID)
    try:
        database.add_note("Imported", "Body", note_id="note-imported")
        with database.transaction() as conn:
            assert conn.execute("SELECT COUNT(*) FROM note_semantic_index_configs").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM note_semantic_note_state").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM note_semantic_chunks").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM note_semantic_work").fetchone()[0] == 0
    finally:
        database.close_all_connections()
