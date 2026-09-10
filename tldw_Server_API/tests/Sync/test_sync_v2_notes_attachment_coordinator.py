"""Owner, readiness, replay, and crash tests for attachment mutation capture."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.attachment_refs_v2 import (
    AttachmentRefV2TombstonePayload,
)
from tldw_Server_API.app.core.Sync.v2.domain_adapters.attachment_refs import (
    AttachmentRefDomainAdapter,
)
from tldw_Server_API.app.core.Sync.v2.materializers import (
    AttachmentRefMaterializer,
    MaterializationResult,
)
from tldw_Server_API.app.core.Sync.v2.models import (
    SyncBlobObjectCreate,
    SyncDatasetCreate,
)
from tldw_Server_API.app.core.Sync.v2.notes_attachment_coordinator import (
    NotesAttachmentCoordinator,
    NotesAttachmentMutationError,
    NotesAttachmentMutationPlan,
    NotesAttachmentSyncNotReadyError,
    _is_allocated_name_for_request,
)
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

OWNER = "user-1"
OTHER_OWNER = "user-2"
DATASET = "dataset-1"
NOTE_ID = "b2222222-2222-4222-8222-222222222222"
ATTACHMENT_ID = "a1111111-1111-4111-8111-111111111111"
CREATED_AT = "2026-08-11T20:30:00+00:00"
BLOB_HASH = "sha256:" + "a" * 64


class _FailOnceMaterializer:
    domain = "attachment.ref"

    def __init__(self, delegate: AttachmentRefMaterializer) -> None:
        self.delegate = delegate
        self.failed = False

    def apply(self, envelope, *, store):
        if not self.failed:
            self.failed = True
            store.mark_envelope_apply_status(
                envelope.server_cursor,
                apply_status="failed",
                apply_error_code="injected_failure",
                apply_error_message="injected failure",
            )
            return MaterializationResult(status="failed", error_code="injected_failure")
        return self.delegate.apply(envelope, store=store)


@pytest.fixture
def coordinator_fixture(tmp_path: Path):
    note_db = CharactersRAGDB(str(tmp_path / "notes.sqlite"), client_id=OWNER)
    note_db.add_note("Parent", "Body", note_id=NOTE_ID)
    store = SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync.sqlite"))
    store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id=DATASET,
            owner_user_id=OWNER,
            domains=["notes.note", "attachment.ref"],
            metadata={
                "default_personal": True,
                "client_family": "chatbook",
                "notes_attachment_v2": {"state": "ready"},
            },
        )
    )
    service = SyncV2Service(
        store=store,
        adapters=SyncAdapterRegistry(
            [AttachmentRefDomainAdapter(v2_writes_enabled=True)]
        ),
        materializers={"attachment.ref": AttachmentRefMaterializer(note_db)},
        settings=SyncV2Settings(
            supports_attachments=True,
            server_trusted_encryption=server_trusted_encryption_status_from_config(
                mode="managed_storage",
                server_trusted_enabled=True,
                auth_mode="multi_user",
            ),
        ),
        clock=lambda: CREATED_AT,
    )
    coordinator = NotesAttachmentCoordinator(service=service, note_db=note_db)
    yield note_db, service, coordinator
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
        "created_by": "server-origin",
    }
    payload.update(overrides)
    return payload


def _plan(**overrides: Any) -> NotesAttachmentMutationPlan:
    values: dict[str, Any] = {
        "owner_id": OWNER,
        "dataset_id": None,
        "operation": "upsert",
        "attachment_id": ATTACHMENT_ID,
        "payload": _payload(),
        "idempotency_key": "request-1",
        "source": "notes_attachment_test",
        "require_available_blob": True,
    }
    values.update(overrides)
    return NotesAttachmentMutationPlan(**values)


def _store_blob(service: SyncV2Service, *, blob_hash: str = BLOB_HASH) -> None:
    service.store.complete_blob_upload(
        SyncBlobObjectCreate(
            blob_id="blob-1",
            dataset_id=DATASET,
            owner_user_id=OWNER,
            attachment_id="unrelated-provenance",
            payload_hash=blob_hash,
            content_type="image/png",
            size_bytes=512,
            storage_backend="local_fs",
            storage_key="blobs/v2/" + "1" * 32 + "/" + "a" * 64 + ".blob",
        )
    )


@pytest.mark.integration
def test_optional_dataset_resolves_only_the_canonical_default(coordinator_fixture) -> None:
    _, service, coordinator = coordinator_fixture

    ready = coordinator.resolve_mutation_ready(owner_id=OWNER, dataset_id=None)

    assert ready is not None and ready.dataset.dataset_id == DATASET
    assert coordinator.require_mutation_ready(
        owner_id=OWNER,
        dataset_id=DATASET,
    ) == ready
    with pytest.raises(NotesAttachmentSyncNotReadyError):
        coordinator.require_mutation_ready(
            owner_id=OWNER,
            dataset_id="another-dataset",
        )
    assert service.store.get_dataset(DATASET) is not None


@pytest.mark.integration
def test_inactive_or_wrong_owner_resolution_is_none_and_never_falls_back(
    coordinator_fixture,
) -> None:
    _, service, coordinator = coordinator_fixture
    dataset = service.store.get_dataset(DATASET)
    assert dataset is not None
    service.store.db.execute(
        "UPDATE sync_datasets SET metadata_json = ? WHERE dataset_id = ?",
        ('{"client_family":"other","default_personal":true}', DATASET),
    )

    assert coordinator.resolve_mutation_ready(owner_id=OWNER, dataset_id=None) is None
    assert coordinator.resolve_mutation_ready(owner_id=OTHER_OWNER, dataset_id=DATASET) is None


@pytest.mark.parametrize(
    ("gate", "supports_blobs", "state"),
    [(False, True, "ready"), (True, False, "ready"), (True, True, "failed")],
)
@pytest.mark.integration
def test_readiness_failures_happen_before_product_writes(
    coordinator_fixture,
    gate: bool,
    supports_blobs: bool,
    state: str,
) -> None:
    note_db, service, coordinator = coordinator_fixture
    service.adapters.register(AttachmentRefDomainAdapter(v2_writes_enabled=gate))
    service.settings = replace(service.settings, supports_attachments=supports_blobs)
    service.store.db.execute(
        "UPDATE sync_datasets SET metadata_json = ? WHERE dataset_id = ?",
        (
            '{"client_family":"chatbook","default_personal":true,'
            f'"notes_attachment_v2":{{"state":"{state}"}}}}',
            DATASET,
        ),
    )

    with pytest.raises(NotesAttachmentSyncNotReadyError):
        coordinator.capture(_plan(require_available_blob=False))

    assert note_db.note_attachment_store.get(DATASET, ATTACHMENT_ID) is None


@pytest.mark.integration
def test_capture_rechecks_note_and_blob_then_returns_durable_response(
    coordinator_fixture,
) -> None:
    note_db, service, coordinator = coordinator_fixture
    _store_blob(service)

    result = coordinator.capture(_plan())

    binding = service.store.get_attachment_revision_binding(
        DATASET,
        ATTACHMENT_ID,
        1,
        owner_user_id=OWNER,
    )
    assert result.idempotent_replay is False
    assert result.envelope.apply_status == "applied"
    assert result.attachment == note_db.note_attachment_store.get(DATASET, ATTACHMENT_ID)
    assert binding is not None and binding.resolved_blob_id == "blob-1"
    assert binding.availability_at_acceptance == "available"


@pytest.mark.integration
def test_exact_retry_reuses_manifest_without_advancing_revision(coordinator_fixture) -> None:
    _, service, coordinator = coordinator_fixture
    _store_blob(service)

    first = coordinator.capture(_plan())
    replay = coordinator.capture(_plan())

    assert replay.idempotent_replay is True
    assert replay.envelope.server_cursor == first.envelope.server_cursor
    assert replay.attachment.version == first.attachment.version == 1
    assert len(
        service.store.list_envelopes_for_entity(
            DATASET,
            "attachment.ref",
            entity_id=ATTACHMENT_ID,
            limit=100,
        )
    ) == 1


@pytest.mark.integration
def test_exact_retry_uses_the_indexed_stable_key_lookup(
    coordinator_fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, service, coordinator = coordinator_fixture
    _store_blob(service)
    first = coordinator.capture(_plan())
    original = service.store.list_envelopes_for_entity

    def indexed_lookup(dataset_id, domain, **kwargs):
        assert kwargs.get("stable_key") == first.envelope.stable_key
        assert kwargs.get("entity_id") is None
        assert kwargs.get("limit") == 1
        return original(dataset_id, domain, **kwargs)

    monkeypatch.setattr(service.store, "list_envelopes_for_entity", indexed_lookup)

    replay = coordinator.capture(_plan())

    assert replay.idempotent_replay is True
    assert replay.envelope.server_cursor == first.envelope.server_cursor


@pytest.mark.integration
def test_idempotency_drift_is_rejected_without_a_product_write(coordinator_fixture) -> None:
    _, service, coordinator = coordinator_fixture
    _store_blob(service)
    coordinator.capture(_plan())

    with pytest.raises(NotesAttachmentMutationError, match="idempotency"):
        coordinator.capture(
            _plan(payload=_payload(file_name="different.png"))
        )

    attachment = coordinator.note_db.note_attachment_store.get(DATASET, ATTACHMENT_ID)
    assert attachment is not None and attachment.file_name == "diagram.png"


@pytest.mark.integration
def test_allocated_filename_replay_requires_the_exact_requested_stem() -> None:
    assert _is_allocated_name_for_request("Report.pdf", "Report-1.pdf") is True
    assert _is_allocated_name_for_request("ReportLong.pdf", "Report-1.pdf") is False


@pytest.mark.integration
def test_filename_allocation_fails_closed_beyond_its_bounded_search(
    coordinator_fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, coordinator = coordinator_fixture
    attachments = [
        SimpleNamespace(
            attachment_id=str(index),
            normalized_file_name=f"existing-{index}.pdf",
        )
        for index in range(1001)
    ]

    def list_page(
        dataset_id,
        note_id,
        *,
        after_attachment_id=None,
        limit=200,
        state="live",
    ):
        del dataset_id, note_id, state
        start = 0 if after_attachment_id is None else int(after_attachment_id) + 1
        return attachments[start : start + limit]

    monkeypatch.setattr(
        coordinator.note_db.note_attachment_store,
        "list_page",
        list_page,
    )

    with pytest.raises(NotesAttachmentMutationError, match="bounded search"):
        coordinator._allocate_unique_file_name(
            dataset_id=DATASET,
            note_id=NOTE_ID,
            requested_file_name="report.pdf",
        )


@pytest.mark.integration
def test_note_read_set_race_and_blob_mismatch_reject_before_append(
    coordinator_fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    note_db, service, coordinator = coordinator_fixture
    _store_blob(service, blob_hash="sha256:" + "b" * 64)
    original = note_db.note_store.get_note_by_id
    calls = 0

    def racing_read(note_id: str, *, include_deleted: bool = False):
        nonlocal calls
        calls += 1
        result = original(note_id, include_deleted=include_deleted)
        if calls == 1:
            note_db.update_note(
                note_id,
                {"title": "Changed", "content": "Body"},
                expected_version=1,
            )
        return result

    monkeypatch.setattr(note_db.note_store, "get_note_by_id", racing_read)

    with pytest.raises(NotesAttachmentMutationError, match="note.*changed"):
        coordinator.capture(_plan())
    assert service.store.list_envelopes_for_entity(
        DATASET,
        "attachment.ref",
        entity_id=ATTACHMENT_ID,
        limit=100,
    ) == []
    assert note_db.note_attachment_store.get(DATASET, ATTACHMENT_ID) is None


@pytest.mark.integration
def test_blob_mismatch_rejects_before_append(coordinator_fixture) -> None:
    note_db, service, coordinator = coordinator_fixture
    _store_blob(service, blob_hash="sha256:" + "b" * 64)

    with pytest.raises(NotesAttachmentMutationError, match="available blob"):
        coordinator.capture(_plan())

    assert service.store.list_envelopes_for_entity(
        DATASET,
        "attachment.ref",
        entity_id=ATTACHMENT_ID,
        limit=100,
    ) == []
    assert note_db.note_attachment_store.get(DATASET, ATTACHMENT_ID) is None


@pytest.mark.integration
def test_failed_projection_manifest_is_resumed_to_exact_postcondition(
    coordinator_fixture,
) -> None:
    _, service, coordinator = coordinator_fixture
    _store_blob(service)
    delegate = AttachmentRefMaterializer(coordinator.note_db)
    service.materializers["attachment.ref"] = _FailOnceMaterializer(delegate)

    with pytest.raises(NotesAttachmentMutationError, match="projection"):
        coordinator.capture(_plan())

    replay = coordinator.capture(_plan())
    assert replay.idempotent_replay is True
    assert replay.envelope.apply_status == "applied"
    assert replay.attachment.version == 1


@pytest.mark.integration
def test_tombstone_plan_uses_exact_base_and_routing_restore_remains_separate(
    coordinator_fixture,
) -> None:
    _, service, coordinator = coordinator_fixture
    _store_blob(service)
    created = coordinator.capture(_plan())
    tombstone_payload = AttachmentRefV2TombstonePayload.model_validate(
        {
            **_payload(),
            "last_modified": "2026-08-11T20:31:00+00:00",
            "deleted_at": "2026-08-11T20:31:00+00:00",
            "reason": "removed",
        }
    ).model_dump(mode="json")
    deleted = coordinator.capture(
        _plan(
            operation="tombstone",
            payload=tombstone_payload,
            idempotency_key="request-delete",
            base_server_cursor=created.envelope.server_cursor,
            base_object_revision=1,
            base_object_hash=created.envelope.payload_hash,
            require_available_blob=False,
        )
    )

    assert deleted.attachment.deleted is True
    assert deleted.envelope.routing_metadata.get("restore_intent") is None
