"""Sync v2 Notes semantic lifecycle authority contracts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import notes as notes_endpoint
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.DB_Management.chacha.note_semantic_models import (
    SemanticDimensionState,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Notes_Graph.suggestion_content import content_fingerprint
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


class _NoopRateLimiter:
    async def check_user_rate_limit(self, *_args: object, **_kwargs: object):
        return True, {}


def _notes_api_app(db: CharactersRAGDB, *, user_id: int) -> FastAPI:
    async def override_user() -> User:
        return User(
            id=user_id,
            username=f"user-{user_id}",
            email=f"user-{user_id}@example.test",
            is_active=True,
            is_admin=True,
        )

    def override_db() -> CharactersRAGDB:
        return db

    app = FastAPI()
    app.include_router(notes_endpoint.router, prefix="/api/v1/notes")
    app.dependency_overrides[notes_endpoint.get_request_user] = override_user
    app.dependency_overrides[notes_endpoint.get_chacha_db_for_user] = override_db
    app.dependency_overrides[notes_endpoint.get_rate_limiter_dep] = lambda: _NoopRateLimiter()
    return app


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


def test_notes_json_api_round_trip_omits_server_local_semantic_state(tmp_path: Path) -> None:
    source_owner = "91001"
    target_owner = "91002"
    source = CharactersRAGDB(str(tmp_path / "source-notes.sqlite"), client_id=source_owner)
    target = CharactersRAGDB(str(tmp_path / "target-notes.sqlite"), client_id=target_owner)
    try:
        generation_id = _activate_semantic_generation(source)
        note_id = source.note_store.add_note(
            "Exported Note",
            "Exported body",
            note_id="note-exported",
            semantic_dataset_id=DATASET_ID,
        )
        assert note_id == "note-exported"
        with source.transaction() as conn:
            conn.execute(
                "INSERT INTO note_semantic_chunks("
                "chunk_id,owner_user_id,dataset_id,generation_id,note_id,content_version,"
                "ordinal,field,start_offset,end_offset,chunk_fingerprint,normalization_version,chunker_version"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "chunk-exported",
                    source_owner,
                    DATASET_ID,
                    generation_id,
                    note_id,
                    1,
                    0,
                    "content",
                    0,
                    13,
                    content_fingerprint("", "Exported body"),
                    "normalization-v1",
                    "chunker-v1",
                ),
            )
        assert source.note_semantic_store.publish_note_manifest(
            owner_user_id=source_owner,
            dataset_id=DATASET_ID,
            generation_id=generation_id,
            note_id=note_id,
            claimed_dirty_generation=1,
            content_version=1,
            manifest={"chunk_count": 1, "manifest_hash": "manifest-exported"},
            now=NOW,
        )

        with source.transaction() as conn:
            source_counts = {
                table: conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE owner_user_id=? AND dataset_id=?",  # nosec B608
                    (source_owner, DATASET_ID),
                ).fetchone()[0]
                for table in (
                    "note_semantic_index_configs",
                    "note_semantic_generations",
                    "note_semantic_note_state",
                    "note_semantic_chunks",
                    "note_semantic_work",
                )
            }
        assert source_counts == {
            "note_semantic_index_configs": 1,
            "note_semantic_generations": 1,
            "note_semantic_note_state": 1,
            "note_semantic_chunks": 1,
            "note_semantic_work": 1,
        }

        with TestClient(_notes_api_app(source, user_id=int(source_owner))) as source_client:
            export_response = source_client.get("/api/v1/notes/export")
        assert export_response.status_code == 200, export_response.text
        export_payload = export_response.json()
        assert set(export_payload) == {
            "notes",
            "data",
            "items",
            "results",
            "count",
            "total",
            "limit",
            "offset",
            "pagination",
            "exported_at",
        }
        assert export_payload["count"] == 1
        exported_note = export_payload["notes"][0]
        assert set(exported_note) == {
            "title",
            "content",
            "conversation_id",
            "message_id",
            "id",
            "created_at",
            "last_modified",
            "version",
            "client_id",
            "deleted",
            "studio",
            "keywords",
            "folders",
            "keyword_sync",
        }
        assert exported_note["title"] == "Exported Note"
        assert exported_note["content"] == "Exported body"
        assert export_payload["data"] == export_payload["notes"]
        assert export_payload["items"] == export_payload["notes"]
        assert export_payload["results"] == export_payload["notes"]

        with TestClient(_notes_api_app(target, user_id=int(target_owner))) as target_client:
            import_response = target_client.post(
                "/api/v1/notes/import",
                json={
                    "duplicate_strategy": "create_copy",
                    "items": [
                        {
                            "file_name": "notes-export.json",
                            "format": "json",
                            "content": json.dumps(export_payload),
                        }
                    ],
                },
            )
        assert import_response.status_code == 200, import_response.text
        assert import_response.json() == {
            "files": [
                {
                    "file_name": "notes-export.json",
                    "source_format": "json",
                    "detected_notes": 1,
                    "created_count": 1,
                    "updated_count": 0,
                    "skipped_count": 0,
                    "failed_count": 0,
                    "errors": [],
                }
            ],
            "detected_notes": 1,
            "created_count": 1,
            "updated_count": 0,
            "skipped_count": 0,
            "failed_count": 0,
        }
        imported_notes = target.list_notes(limit=10)
        assert [(note["title"], note["content"]) for note in imported_notes] == [
            ("Exported Note", "Exported body")
        ]
        assert target.note_semantic_store.get_configuration(DATASET_ID) is None
        with target.transaction() as conn:
            for table in (
                "note_semantic_index_configs",
                "note_semantic_generations",
                "note_semantic_note_state",
                "note_semantic_chunks",
                "note_semantic_work",
            ):
                assert conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE owner_user_id=? AND dataset_id=?",  # nosec B608
                    (target_owner, DATASET_ID),
                ).fetchone()[0] == 0
            assert conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' "
                "AND name LIKE 'note_semantic_vectors_%'"
            ).fetchone()[0] == 0
    finally:
        source.close_all_connections()
        target.close_all_connections()
