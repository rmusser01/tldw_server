"""Real-store projection tests for Notes attachment reference mutations."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.integration

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.attachment_refs_v2 import (
    attachment_ref_v2_object_hash,
)
from tldw_Server_API.app.core.Sync.v2.errors import SyncHeadConflictError
from tldw_Server_API.app.core.Sync.v2.materializers import AttachmentRefMaterializer
from tldw_Server_API.app.core.Sync.v2.models import (
    SyncBlobObjectCreate,
    SyncDatasetCreate,
    SyncEnvelope,
    SyncEnvelopeCreate,
)
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

OWNER = "user-1"
DATASET = "dataset-1"
DEVICE = "device-1"
NOTE_ID = "b2222222-2222-4222-8222-222222222222"
ATTACHMENT_ID = "a1111111-1111-4111-8111-111111111111"
OTHER_ATTACHMENT_ID = "c3333333-3333-4333-8333-333333333333"
CREATED_AT = "2026-08-11T20:30:00+00:00"
BLOB_HASH = "sha256:" + "a" * 64


@pytest.fixture
def projection(tmp_path: Path):
    note_db = CharactersRAGDB(str(tmp_path / "notes.sqlite"), client_id=OWNER)
    note_db.add_note("Parent", "Body", note_id=NOTE_ID)
    sync_store = SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync.sqlite"))
    sync_store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id=DATASET,
            owner_user_id=OWNER,
            domains=["notes.note", "attachment.ref"],
            metadata={"notes_attachment_v2": {"state": "ready"}},
        )
    )
    materializer = AttachmentRefMaterializer(note_db)
    yield note_db, sync_store, materializer
    note_db.close_all_connections()


def _payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "attachment_id": ATTACHMENT_ID,
        "parent_domain": "notes.note",
        "parent_object_id": NOTE_ID,
        "file_name": "diagram.png",
        "original_file_name": "diagram.png",
        "content_type": "image/png",
        "size_bytes": 512,
        "blob_hash": BLOB_HASH,
        "created_at": CREATED_AT,
        "last_modified": CREATED_AT,
        "created_by": DEVICE,
    }
    payload.update(overrides)
    return payload


def _envelope(
    *,
    revision: int = 1,
    operation: str = "upsert",
    payload: dict[str, Any] | None = None,
    base: SyncEnvelope | None = None,
    envelope_id: str | None = None,
    attachment_id: str = ATTACHMENT_ID,
    routing_metadata: dict[str, object] | None = None,
    client_sequence: int | None = None,
) -> SyncEnvelopeCreate:
    payload = payload or _payload(attachment_id=attachment_id)
    values: dict[str, Any] = {
        "dataset_id": DATASET,
        "client_envelope_id": envelope_id or f"env-{attachment_id}-{revision}",
        "domain": "attachment.ref",
        "operation": operation,
        "object_id": attachment_id,
        "device_id": DEVICE,
        "client_sequence": client_sequence or revision,
        "schema_version": 2,
        "adapter_version": 2,
        "object_revision": revision,
        "payload": payload,
        "payload_hash": attachment_ref_v2_object_hash(
            operation,
            payload,
            object_revision=revision,
        ),
        "created_at_client": payload["last_modified"],
        "encryption_metadata": {"policy": "server_trusted_v1"},
        "routing_metadata": routing_metadata or {},
    }
    if base is not None:
        values.update(
            {
                "base_server_cursor": base.server_cursor,
                "base_object_revision": base.object_revision,
                "base_object_hash": base.payload_hash,
            }
        )
    return SyncEnvelopeCreate(**values)


def _accept_and_apply(
    store: SyncV2Store,
    materializer: AttachmentRefMaterializer,
    envelope: SyncEnvelopeCreate,
):
    accepted = store.insert_envelope(envelope)
    result = materializer.apply(accepted, store=store)
    return accepted, result


def test_create_projects_into_registry_and_preserves_pending_binding(projection) -> None:
    note_db, store, materializer = projection

    accepted, result = _accept_and_apply(store, materializer, _envelope())

    row = note_db.note_attachment_store.get(DATASET, ATTACHMENT_ID)
    binding = store.get_attachment_revision_binding(
        DATASET,
        ATTACHMENT_ID,
        1,
        owner_user_id=OWNER,
    )
    assert result.status == "applied"
    assert row is not None and row.object_hash == accepted.payload_hash
    assert row.version == 1 and row.source_kind == "sync"
    assert binding is not None
    assert binding.availability_at_acceptance == "metadata_only"
    assert binding.resolved_blob_id is None


def test_create_observes_present_blob_without_changing_envelope_identity(projection) -> None:
    note_db, store, materializer = projection
    store.complete_blob_upload(
        SyncBlobObjectCreate(
            blob_id="blob-present",
            dataset_id=DATASET,
            owner_user_id=OWNER,
            attachment_id="unrelated-provenance",
            payload_hash=BLOB_HASH,
            content_type="image/png",
            size_bytes=512,
            storage_backend="local_fs",
            storage_key="blobs/v2/" + "1" * 32 + "/" + "a" * 64 + ".blob",
        )
    )
    submitted = _envelope()

    accepted, result = _accept_and_apply(store, materializer, submitted)

    binding = store.get_attachment_revision_binding(
        DATASET,
        ATTACHMENT_ID,
        1,
        owner_user_id=OWNER,
    )
    assert result.status == "applied"
    assert accepted.payload == submitted.payload
    assert accepted.payload_hash == submitted.payload_hash
    assert binding is not None
    assert binding.availability_at_acceptance == "available"
    assert binding.resolved_blob_id == "blob-present"
    assert note_db.note_attachment_store.get(DATASET, ATTACHMENT_ID) is not None


def test_late_blob_resolution_never_rehashes_or_enriches_accepted_envelope(projection) -> None:
    _, store, materializer = projection
    accepted, result = _accept_and_apply(store, materializer, _envelope())
    assert result.status == "applied"
    before = (accepted.payload, accepted.payload_hash, accepted.server_cursor)

    store.complete_blob_upload(
        SyncBlobObjectCreate(
            blob_id="blob-late",
            dataset_id=DATASET,
            owner_user_id=OWNER,
            attachment_id="unrelated-provenance",
            payload_hash=BLOB_HASH,
            content_type="image/png",
            size_bytes=512,
            storage_backend="local_fs",
            storage_key="blobs/v2/" + "2" * 32 + "/" + "a" * 64 + ".blob",
        )
    )

    replayed = store.insert_envelope(_envelope())
    binding = store.get_attachment_revision_binding(
        DATASET,
        ATTACHMENT_ID,
        1,
        owner_user_id=OWNER,
    )
    assert (replayed.payload, replayed.payload_hash, replayed.server_cursor) == before
    assert binding is not None
    assert binding.availability_at_acceptance == "metadata_only"
    assert binding.resolved_blob_id == "blob-late"


def test_rename_then_replace_advance_registry_exactly_once(projection) -> None:
    note_db, store, materializer = projection
    first, _ = _accept_and_apply(store, materializer, _envelope())
    rename_payload = _payload(
        file_name="renamed.png",
        last_modified="2026-08-11T20:31:00+00:00",
    )
    second, renamed = _accept_and_apply(
        store,
        materializer,
        _envelope(revision=2, payload=rename_payload, base=first),
    )
    replacement_hash = "sha256:" + "b" * 64
    replace_payload = _payload(
        file_name="renamed.png",
        size_bytes=1024,
        blob_hash=replacement_hash,
        last_modified="2026-08-11T20:32:00+00:00",
    )
    third, replaced = _accept_and_apply(
        store,
        materializer,
        _envelope(revision=3, payload=replace_payload, base=second),
    )

    row = note_db.note_attachment_store.get(DATASET, ATTACHMENT_ID)
    assert renamed.status == replaced.status == "applied"
    assert row is not None
    assert row.file_name == "renamed.png"
    assert row.size_bytes == 1024 and row.blob_hash == replacement_hash
    assert row.version == 3 and row.object_hash == third.payload_hash


def test_tombstone_and_routing_only_restore_preserve_stable_identity(projection) -> None:
    note_db, store, materializer = projection
    created, _ = _accept_and_apply(store, materializer, _envelope())
    note_db.soft_delete_note(NOTE_ID, expected_version=1)
    tombstone_payload = _payload(
        last_modified="2026-08-11T20:31:00+00:00",
        deleted_at="2026-08-11T20:31:00+00:00",
        reason="removed",
    )
    tombstone, deleted = _accept_and_apply(
        store,
        materializer,
        _envelope(
            revision=2,
            operation="tombstone",
            payload=tombstone_payload,
            base=created,
        ),
    )
    restore_payload = _payload(last_modified="2026-08-11T20:32:00+00:00")
    restored_envelope, restored = _accept_and_apply(
        store,
        materializer,
        _envelope(
            revision=3,
            payload=restore_payload,
            base=tombstone,
            routing_metadata={"restore_intent": True},
        ),
    )

    row = note_db.note_attachment_store.get(DATASET, ATTACHMENT_ID)
    assert deleted.status == restored.status == "applied"
    assert row is not None and row.deleted is False
    assert row.version == 3 and row.object_hash == restored_envelope.payload_hash
    assert row.attachment_id == ATTACHMENT_ID and row.note_id == NOTE_ID


def test_exact_and_postcondition_replay_do_not_advance_product_revision(projection) -> None:
    note_db, store, materializer = projection
    submitted = _envelope()
    accepted = store.insert_envelope(submitted)
    note_db.note_attachment_store.create(
        dataset_id=DATASET,
        attachment_id=ATTACHMENT_ID,
        note_id=NOTE_ID,
        file_name="diagram.png",
        original_file_name="diagram.png",
        content_type="image/png",
        size_bytes=512,
        blob_hash=BLOB_HASH,
        object_hash=accepted.payload_hash or "",
        created_at=CREATED_AT,
        last_modified=CREATED_AT,
        created_by=DEVICE,
        source_kind="sync",
    )

    first = materializer.apply(accepted, store=store)
    replay = materializer.apply(store.insert_envelope(submitted), store=store)

    row = note_db.note_attachment_store.get(DATASET, ATTACHMENT_ID)
    assert first.status == replay.status == "applied"
    assert row is not None and row.version == 1
    assert row.created_at == CREATED_AT and row.last_modified == CREATED_AT


def test_stale_base_is_rejected_before_product_write(projection) -> None:
    note_db, store, materializer = projection
    first, _ = _accept_and_apply(store, materializer, _envelope())
    stale = _envelope(
        revision=2,
        payload=_payload(
            file_name="stale.png",
            last_modified="2026-08-11T20:31:00+00:00",
        ),
        base=first,
    )
    stale = replace(stale, base_object_hash="sha256:" + "f" * 64)
    with pytest.raises(SyncHeadConflictError, match="sync_head_changed"):
        store.insert_envelope(stale)

    row = note_db.note_attachment_store.get(DATASET, ATTACHMENT_ID)
    assert row is not None and row.version == 1 and row.file_name == "diagram.png"


def test_first_tombstone_is_a_conflict_without_creating_a_live_product_row(
    projection,
) -> None:
    note_db, store, materializer = projection
    payload = _payload(
        deleted_at=CREATED_AT,
        reason="invalid first tombstone",
    )
    accepted = store.insert_envelope(
        _envelope(operation="tombstone", payload=payload)
    )

    result = materializer.apply(accepted, store=store)

    assert result.status == "conflict"
    assert note_db.note_attachment_store.get(DATASET, ATTACHMENT_ID) is None


def test_name_collision_and_hidden_parent_fail_without_partial_projection(projection) -> None:
    note_db, store, materializer = projection
    _accept_and_apply(store, materializer, _envelope())
    conflicting_payload = _payload(
        attachment_id=OTHER_ATTACHMENT_ID,
        last_modified="2026-08-11T20:31:00+00:00",
    )
    collision = _accept_and_apply(
        store,
        materializer,
        _envelope(
            attachment_id=OTHER_ATTACHMENT_ID,
            payload=conflicting_payload,
            envelope_id="env-name-collision",
            client_sequence=2,
        ),
    )[1]
    note_db.soft_delete_note(NOTE_ID, expected_version=1)
    hidden_id = "d4444444-4444-4444-8444-444444444444"
    hidden = _accept_and_apply(
        store,
        materializer,
        _envelope(
            attachment_id=hidden_id,
            payload=_payload(
                attachment_id=hidden_id,
                file_name="hidden.png",
                original_file_name="hidden.png",
                last_modified="2026-08-11T20:32:00+00:00",
                created_at="2026-08-11T20:32:00+00:00",
            ),
            envelope_id="env-hidden-parent",
            client_sequence=3,
        ),
    )[1]

    assert collision.status == "conflict"
    assert hidden.status == "failed"
    assert note_db.note_attachment_store.get(DATASET, OTHER_ATTACHMENT_ID) is None
    assert note_db.note_attachment_store.get(DATASET, hidden_id) is None
