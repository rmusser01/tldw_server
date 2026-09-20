from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.endpoints import sync as sync_endpoint
from tldw_Server_API.app.api.v1.schemas.sync_v2_models import SyncV2Envelope
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2 import adapters as sync_adapters
from tldw_Server_API.app.core.Sync.v2.factory import default_sync_v2_registry
from tldw_Server_API.app.core.Sync.v2.materializers import AttachmentRefMaterializer
from tldw_Server_API.app.core.Sync.v2.models import (
    M1_SYNC_DOMAINS,
    SyncDataset,
    SyncEnvelopeCreate,
)
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store


def _ready_encryption():
    return server_trusted_encryption_status_from_config(
        mode="managed_storage",
        server_trusted_enabled=True,
        auth_mode="multi_user",
    )


def _test_user() -> User:
    return User(id="user-1", username="user-1")


class _LegacyAttachmentRefAdapter:
    """Frozen v1 writer used only to preserve legacy integration coverage."""

    domain = "attachment.ref"
    supported_adapter_versions = {1}

    def evaluate_envelope(
        self,
        envelope: SyncEnvelopeCreate,
        *,
        dataset: SyncDataset,
        context: sync_adapters.SyncAdapterContext | None = None,
    ):
        del dataset
        try:
            metadata = sync_adapters.extract_attachment_ref_metadata(envelope)
        except sync_adapters.AttachmentRefValidationError as exc:
            return sync_adapters.AdapterRejected(
                client_envelope_id=envelope.client_envelope_id,
                error_code=exc.error_code,
                message=str(exc),
            )
        prior = context.prior_envelopes if context is not None else ()
        conflicting = next(
            (
                item
                for item in prior
                if item.operation != "tombstone"
                and _legacy_attachment_identity_matches(item, envelope, metadata)
                and _legacy_attachment_hash(item) != metadata.payload_hash
            ),
            None,
        )
        if conflicting is None:
            return sync_adapters.AdapterAccepted(
                client_envelope_id=envelope.client_envelope_id
            )
        return sync_adapters.AdapterConflict(
            client_envelope_id=envelope.client_envelope_id,
            domain="attachment.ref",
            entity_id=envelope.entity_id,
            conflict_type="attachment_ref_hash_mismatch",
            message=(
                "attachment.ref stable attachment ID was reused with a different "
                "payload hash"
            ),
            metadata={
                "attachment_id": metadata.attachment_id,
                "incoming_payload_hash": metadata.payload_hash,
                "conflicting_payload_hash": _legacy_attachment_hash(conflicting),
                "conflicting_envelope_id": conflicting.client_envelope_id,
            },
        )


def _legacy_attachment_identity_matches(
    prior: Any,
    incoming: SyncEnvelopeCreate,
    incoming_metadata: Any,
) -> bool:
    prior_payload = prior.payload or prior.payload_clear
    prior_attachment_id = prior_payload.get("attachment_id")
    if isinstance(prior_attachment_id, str) and prior_attachment_id.strip():
        return prior_attachment_id.strip() == incoming_metadata.attachment_id
    if prior.stable_key and incoming.stable_key:
        return prior.stable_key == incoming.stable_key
    return prior.entity_id == incoming.entity_id


def _legacy_attachment_hash(envelope: Any) -> str:
    payload = envelope.payload or envelope.payload_clear
    payload_hash = payload.get("payload_hash")
    if isinstance(payload_hash, str) and payload_hash.strip():
        return payload_hash.strip()
    return envelope.payload_hash or ""


def test_attachment_ref_v1_compatibility_evaluator_is_test_only() -> None:
    assert not hasattr(sync_adapters, "_evaluate_attachment_ref_v1")


@pytest.fixture()
def sync_service(tmp_path: Path) -> Iterator[SyncV2Service]:
    default_sync_v2_registry.cache_clear()
    registry = default_sync_v2_registry()
    registry.register(_LegacyAttachmentRefAdapter())
    service = SyncV2Service(
        store=SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync_v2_attachment_refs.db")),
        adapters=registry,
        materializers={"attachment.ref": AttachmentRefMaterializer()},
        clock=lambda: "2026-05-23T18:12:00+00:00",
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(
            server_trusted_encryption=_ready_encryption(),
            restore_manifest_scan_limit=100,
        ),
    )
    for device_id in ("device-1", "device-2"):
        service.register_device(
            user_id="user-1",
            display_name=device_id,
            client_type="chatbook",
            device_id=device_id,
        )
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=list(M1_SYNC_DOMAINS),
    )
    yield service
    default_sync_v2_registry.cache_clear()


@pytest.fixture()
def client(sync_service: SyncV2Service) -> TestClient:
    app = FastAPI()
    app.include_router(sync_endpoint.router, prefix="/api/v1/sync")
    app.dependency_overrides[get_request_user] = _test_user
    app.dependency_overrides[sync_endpoint.get_sync_v2_service] = lambda: sync_service
    if hasattr(sync_endpoint, "get_sync_v2_profile_service"):
        app.dependency_overrides[sync_endpoint.get_sync_v2_profile_service] = lambda: sync_service
    return TestClient(app)


def _attachment_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "attachment_id": "att-1",
        "parent_domain": "notes.note",
        "parent_object_id": "note-1",
        "content_type": "image/png",
        "size_bytes": 512,
        "payload_hash": "sha256:blob-v1",
        "availability": "client_local",
    }
    payload.update(overrides)
    return payload


def _attachment_ref(**overrides: Any) -> SyncEnvelopeCreate:
    payload = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "env-attachment-1",
        "domain": "attachment.ref",
        "operation": "upsert",
        "object_id": "att-1",
        "device_id": "device-1",
        "client_sequence": 1,
        "schema_version": 1,
        "object_revision": 1,
        "payload": _attachment_payload(),
        "payload_hash": "sha256:blob-v1",
        "payload_size_bytes": 128,
        "created_at_client": "2026-05-23T18:12:44+00:00",
        "encryption_metadata": {"policy": "server_trusted_v1"},
        "stable_key": "attachment:att-1",
    }
    payload.update(overrides)
    return SyncEnvelopeCreate(**payload)


def _attachment_ref_after(
    service: SyncV2Service,
    **overrides: Any,
) -> SyncEnvelopeCreate:
    state = service.store.get_object_state(
        "dataset-1",
        "attachment.ref",
        overrides.get("object_id", "att-1"),
    )
    assert state is not None
    payload = {
        "base_server_cursor": state.latest_server_cursor,
        "base_object_revision": state.object_revision,
        "base_object_hash": state.object_hash,
        "object_revision": state.object_revision + 1,
    }
    payload.update(overrides)
    return _attachment_ref(**payload)


def _push_one(service: SyncV2Service, envelope: SyncEnvelopeCreate):
    return service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[envelope],
    )


ATTACHMENT_V2_ID = "a1111111-1111-4111-8111-111111111111"
NOTE_V2_ID = "b2222222-2222-4222-8222-222222222222"
ATTACHMENT_V2_TIMESTAMP = "2026-08-11T20:30:00+00:00"
ATTACHMENT_V2_BLOB_HASH = "sha256:" + "a" * 64


def _attachment_ref_v2_module():
    from tldw_Server_API.app.core.Sync.v2 import attachment_refs_v2

    return attachment_refs_v2


def _attachment_v2_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "attachment_id": ATTACHMENT_V2_ID,
        "parent_domain": "notes.note",
        "parent_object_id": NOTE_V2_ID,
        "file_name": "diagram.png",
        "original_file_name": "diagram.png",
        "content_type": "image/png",
        "size_bytes": 512,
        "blob_hash": ATTACHMENT_V2_BLOB_HASH,
        "created_at": ATTACHMENT_V2_TIMESTAMP,
        "last_modified": ATTACHMENT_V2_TIMESTAMP,
        "created_by": "device-1",
    }
    payload.update(overrides)
    return payload


def _attachment_v2_tombstone_payload(**overrides: Any) -> dict[str, Any]:
    payload = _attachment_v2_payload(
        last_modified="2026-08-11T20:31:00+00:00",
    )
    payload.update(
        {
            "deleted_at": "2026-08-11T20:31:00+00:00",
            "reason": "removed",
        }
    )
    payload.update(overrides)
    return payload


def _attachment_v2_envelope(**overrides: Any) -> SyncEnvelopeCreate:
    contract = _attachment_ref_v2_module()
    payload = overrides.pop("payload", _attachment_v2_payload())
    values: dict[str, Any] = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "attachment-v2-create",
        "domain": "attachment.ref",
        "operation": "upsert",
        "object_id": ATTACHMENT_V2_ID,
        "device_id": "device-1",
        "client_sequence": 1,
        "schema_version": 2,
        "adapter_version": 2,
        "object_revision": 1,
        "payload": payload,
        "created_at_client": ATTACHMENT_V2_TIMESTAMP,
        "encryption_metadata": {"policy": "server_trusted_v1"},
        "routing_metadata": {},
    }
    values.update(overrides)
    values.setdefault(
        "payload_hash",
        contract.attachment_ref_v2_object_hash(
            values["operation"],
            payload,
            object_revision=values["object_revision"],
        ),
    )
    return SyncEnvelopeCreate(**values)


def _attachment_v2_dataset(
    *,
    state: str = "ready",
    bootstrap_id: str | None = None,
    encryption_policy: str = "server_trusted_v1",
) -> SyncDataset:
    metadata: dict[str, Any] = {"notes_attachment_v2": {"state": state}}
    if bootstrap_id is not None:
        metadata["notes_attachment_v2"]["bootstrap_id"] = bootstrap_id
    return SyncDataset(
        dataset_id="dataset-1",
        owner_user_id="user-1",
        scope_type="personal",
        encryption_policy=encryption_policy,
        domains=["notes.note", "attachment.ref"],
        workspace_id=None,
        metadata=metadata,
        created_at=ATTACHMENT_V2_TIMESTAMP,
        updated_at=ATTACHMENT_V2_TIMESTAMP,
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"attachment_id": ATTACHMENT_V2_ID.upper()}, "canonical lowercase UUIDv4"),
        ({"attachment_id": "11111111-1111-1111-8111-111111111111"}, "UUIDv4"),
        ({"parent_object_id": NOTE_V2_ID.upper()}, "canonical lowercase UUIDv4"),
        ({"blob_hash": "sha256:" + "A" * 64}, "lowercase SHA-256"),
        ({"blob_hash": "sha256:abc"}, "lowercase SHA-256"),
        ({"size_bytes": 0}, "greater than or equal to 1"),
        ({"size_bytes": -1}, "greater than or equal to 1"),
        ({"size_bytes": True}, "valid integer"),
        ({"availability": "available"}, "extra inputs are not permitted"),
        ({"resolved_blob_id": "blob-1"}, "extra inputs are not permitted"),
        ({"storage_status": "available"}, "extra inputs are not permitted"),
        ({"retention_released_at": ATTACHMENT_V2_TIMESTAMP}, "extra inputs are not permitted"),
        ({"restore_intent": True}, "extra inputs are not permitted"),
    ],
)
def test_attachment_ref_v2_payload_is_exact_and_strict(
    overrides: dict[str, Any],
    message: str,
) -> None:
    contract = _attachment_ref_v2_module()

    with pytest.raises(contract.AttachmentRefV2ValidationError, match=message):
        contract.parse_attachment_ref_v2_payload(
            "upsert",
            _attachment_v2_payload(**overrides),
        )


@pytest.mark.parametrize(
    "content_type",
    [
        " Image/PNG ",
        "Image/PNG",
        "image/png; charset=utf-8",
    ],
)
def test_attachment_ref_v2_rejects_noncanonical_content_type_before_hashing(
    content_type: str,
) -> None:
    contract = _attachment_ref_v2_module()
    payload = _attachment_v2_payload(content_type=content_type)

    with pytest.raises(
        contract.AttachmentRefV2ValidationError,
        match="canonical normalized media type",
    ):
        contract.attachment_ref_v2_object_hash(
            "upsert",
            payload,
            object_revision=1,
        )


def test_attachment_ref_v2_filename_uses_canonical_notes_policy() -> None:
    contract = _attachment_ref_v2_module()

    parsed = contract.parse_attachment_ref_v2_payload(
        "upsert",
        _attachment_v2_payload(file_name="Résumé Draft?.PDF"),
    )
    assert parsed.file_name == "Résumé_Draft.pdf"

    with pytest.raises(
        contract.AttachmentRefV2ValidationError,
        match="Unsupported attachment type",
    ):
        contract.parse_attachment_ref_v2_payload(
            "upsert",
            _attachment_v2_payload(file_name="payload.exe"),
        )


def test_attachment_ref_v2_filename_rejects_control_characters_before_hashing() -> None:
    contract = _attachment_ref_v2_module()

    with pytest.raises(
        contract.AttachmentRefV2ValidationError,
        match="Invalid attachment filename",
    ):
        contract.attachment_ref_v2_object_hash(
            "upsert",
            _attachment_v2_payload(file_name="bad\x00.pdf"),
            object_revision=1,
        )


def test_attachment_ref_v2_rejects_filename_whose_casefolded_key_expands_past_limit() -> None:
    contract = _attachment_ref_v2_module()

    with pytest.raises(
        contract.AttachmentRefV2ValidationError,
        match="normalized attachment filename exceeds",
    ):
        contract.attachment_ref_v2_object_hash(
            "upsert",
            _attachment_v2_payload(file_name=("\u0130" * 176) + ".pdf"),
            object_revision=1,
        )


@pytest.mark.parametrize(
    "original_file_name",
    [
        "../Report.pdf",
        "folder/Report.pdf",
        "folder\\Report.pdf",
        "folder\uff0fReport.pdf",
        "\uff0e\uff0e",
        "bad\x00.pdf",
        "x" * 256,
        ("\U0001f4ce" * 256),
    ],
)
def test_attachment_ref_v2_rejects_unsafe_or_oversized_original_filename_before_hashing(
    original_file_name: str,
) -> None:
    contract = _attachment_ref_v2_module()

    with pytest.raises(
        contract.AttachmentRefV2ValidationError,
        match="original_file_name",
    ):
        contract.attachment_ref_v2_object_hash(
            "upsert",
            _attachment_v2_payload(original_file_name=original_file_name),
            object_revision=1,
        )


def test_attachment_ref_v2_accepts_original_filename_at_utf8_boundary() -> None:
    contract = _attachment_ref_v2_module()
    original_file_name = "\U0001f4ce" * 255

    parsed = contract.parse_attachment_ref_v2_payload(
        "upsert",
        _attachment_v2_payload(original_file_name=original_file_name),
    )

    assert len(parsed.original_file_name) == 255
    assert len(parsed.original_file_name.encode("utf-8")) == 1020


def test_attachment_ref_v2_restore_intent_is_routing_metadata_only() -> None:
    contract = _attachment_ref_v2_module()
    parsed = contract.parse_attachment_ref_v2_payload(
        "upsert",
        _attachment_v2_payload(),
    )

    assert "restore_intent" not in parsed.model_dump(mode="json")
    assert contract.validate_attachment_ref_v2_routing_metadata(
        "upsert", {"restore_intent": True}
    ) == {"restore_intent": True}


@pytest.mark.parametrize("restore_intent", [False, 1, "true"])
def test_attachment_ref_v2_restore_intent_requires_literal_true(
    restore_intent: object,
) -> None:
    contract = _attachment_ref_v2_module()

    with pytest.raises(
        contract.AttachmentRefV2ValidationError,
        match="boolean true",
    ):
        contract.validate_attachment_ref_v2_routing_metadata(
            "upsert",
            {"restore_intent": restore_intent},
        )


@pytest.mark.parametrize(
    "routing_metadata",
    [
        {"created_by": "forged-device"},
        {"available": True},
        {"resolved_blob_id": "private-blob-id"},
        {"storage_status": "available"},
        {"retention_released_at": ATTACHMENT_V2_TIMESTAMP},
        {"arbitrary": "private-routing-value"},
    ],
)
def test_attachment_ref_v2_routing_metadata_rejects_nonrouting_fields_safely(
    routing_metadata: dict[str, object],
) -> None:
    contract = _attachment_ref_v2_module()

    with pytest.raises(contract.AttachmentRefV2ValidationError) as exc_info:
        contract.validate_attachment_ref_v2_routing_metadata(
            "upsert",
            routing_metadata,
        )

    assert str(exc_info.value) == (
        "attachment.ref v2 routing metadata contains unsupported fields"
    )
    assert not any(str(value) in str(exc_info.value) for value in routing_metadata.values())


def test_attachment_ref_v2_routing_metadata_allows_verified_bootstrap_keys() -> None:
    contract = _attachment_ref_v2_module()

    assert contract.validate_attachment_ref_v2_routing_metadata(
        "upsert",
        {"bootstrap_capture": True, "bootstrap_id": "bootstrap-1"},
    ) == {"bootstrap_capture": True, "bootstrap_id": "bootstrap-1"}


def test_attachment_ref_v2_canonical_object_hash_has_exact_vector() -> None:
    contract = _attachment_ref_v2_module()

    assert contract.attachment_ref_v2_object_hash(
        "upsert",
        _attachment_v2_payload(),
        object_revision=1,
    ) == (
        "sha256:04d1fb35d8be65f0a23b73ab8365ff156e0b06f9d174a484d1cc040af360eac6"
    )


def test_attachment_ref_v2_tombstone_is_a_strict_complete_snapshot() -> None:
    contract = _attachment_ref_v2_module()
    tombstone = _attachment_v2_tombstone_payload()

    parsed = contract.parse_attachment_ref_v2_payload("tombstone", tombstone)
    assert parsed.deleted_at == "2026-08-11T20:31:00+00:00"
    assert parsed.reason == "removed"

    with pytest.raises(contract.AttachmentRefV2ValidationError):
        contract.parse_attachment_ref_v2_payload("tombstone", _attachment_v2_payload())
    with pytest.raises(contract.AttachmentRefV2ValidationError):
        contract.parse_attachment_ref_v2_payload("upsert", tombstone)
    with pytest.raises(contract.AttachmentRefV2ValidationError):
        contract.parse_attachment_ref_v2_payload(
            "tombstone",
            _attachment_v2_tombstone_payload(reason="x" * 257),
        )


def test_attachment_ref_v2_tombstone_provenance_binds_deleted_at() -> None:
    contract = _attachment_ref_v2_module()
    prior = contract.parse_attachment_ref_v2_payload(
        "upsert",
        _attachment_v2_payload(),
    )

    accepted = contract.validate_attachment_ref_v2(
        "tombstone",
        _attachment_v2_tombstone_payload(),
        envelope_created_at_client="2026-08-11T20:31:00+00:00",
        authenticated_device_id="device-2",
        prior_payload=prior,
        prior_operation="upsert",
    )
    assert accepted.deleted_at == accepted.last_modified

    with pytest.raises(contract.AttachmentRefV2ValidationError, match="deleted_at"):
        contract.validate_attachment_ref_v2(
            "tombstone",
            _attachment_v2_tombstone_payload(
                deleted_at="2026-08-11T20:32:00+00:00"
            ),
            envelope_created_at_client="2026-08-11T20:31:00+00:00",
            authenticated_device_id="device-2",
            prior_payload=prior,
            prior_operation="upsert",
        )


def test_attachment_ref_v2_object_hash_binds_revision_and_lifecycle() -> None:
    contract = _attachment_ref_v2_module()
    live_v1 = contract.attachment_ref_v2_object_hash(
        "upsert",
        _attachment_v2_payload(),
        object_revision=1,
    )
    live_v2 = contract.attachment_ref_v2_object_hash(
        "upsert",
        _attachment_v2_payload(),
        object_revision=2,
    )
    tombstone_v2 = contract.attachment_ref_v2_object_hash(
        "tombstone",
        _attachment_v2_tombstone_payload(),
        object_revision=2,
    )

    assert len({live_v1, live_v2, tombstone_v2}) == 3
    assert tombstone_v2 == (
        "sha256:a1233681df93f6d194f35e518938646e91d66172557ee0f138a06f75d23d6eae"
    )


def test_attachment_ref_v2_provenance_acceptance_vectors() -> None:
    contract = _attachment_ref_v2_module()
    normalized_client_time = "2026-08-11T20:30:00+00:00"

    client = contract.validate_attachment_ref_v2(
        "upsert",
        _attachment_v2_payload(),
        envelope_created_at_client="2026-08-11T13:30:00-07:00",
        authenticated_device_id="device-1",
    )
    server = contract.validate_attachment_ref_v2(
        "upsert",
        _attachment_v2_payload(created_by="server-origin"),
        envelope_created_at_client=normalized_client_time,
        authenticated_device_id="server-origin",
        trusted_server_origin=True,
    )

    assert client.created_at == normalized_client_time
    assert client.last_modified == normalized_client_time
    assert client.created_by == "device-1"
    assert server.created_at == normalized_client_time
    assert server.last_modified == normalized_client_time
    assert server.created_by == "server-origin"


def test_attachment_ref_v2_legacy_provenance_requires_verified_bootstrap() -> None:
    contract = _attachment_ref_v2_module()
    legacy = _attachment_v2_payload(
        created_at="2020-01-02T03:04:05+00:00",
        created_by="legacy-device",
    )

    with pytest.raises(contract.AttachmentRefV2ValidationError, match="provenance"):
        contract.validate_attachment_ref_v2(
            "upsert",
            legacy,
            envelope_created_at_client=ATTACHMENT_V2_TIMESTAMP,
            authenticated_device_id="server-origin",
            trusted_server_origin=True,
            verified_bootstrap=False,
        )

    accepted = contract.validate_attachment_ref_v2(
        "upsert",
        legacy,
        envelope_created_at_client=ATTACHMENT_V2_TIMESTAMP,
        authenticated_device_id="server-origin",
        trusted_server_origin=True,
        verified_bootstrap=True,
    )
    assert accepted.created_at == "2020-01-02T03:04:05+00:00"
    assert accepted.created_by == "legacy-device"
    assert accepted.last_modified == ATTACHMENT_V2_TIMESTAMP


def test_attachment_ref_v2_updates_preserve_creation_fields_and_mutation_time() -> None:
    contract = _attachment_ref_v2_module()
    prior = contract.parse_attachment_ref_v2_payload(
        "upsert",
        _attachment_v2_payload(),
    )
    modified = "2026-08-11T20:31:00+00:00"

    accepted = contract.validate_attachment_ref_v2(
        "upsert",
        _attachment_v2_payload(file_name="renamed.png", last_modified=modified),
        envelope_created_at_client=modified,
        authenticated_device_id="device-2",
        prior_payload=prior,
        prior_operation="upsert",
    )
    assert accepted.created_at == prior.created_at
    assert accepted.created_by == prior.created_by
    assert accepted.original_file_name == prior.original_file_name
    assert accepted.last_modified == modified

    with pytest.raises(contract.AttachmentRefV2ValidationError, match="immutable"):
        contract.validate_attachment_ref_v2(
            "upsert",
            _attachment_v2_payload(
                created_by="device-2",
                last_modified=modified,
            ),
            envelope_created_at_client=modified,
            authenticated_device_id="device-2",
            prior_payload=prior,
            prior_operation="upsert",
        )


def test_attachment_ref_v2_exact_replay_never_enriches_or_rewrites_payload() -> None:
    contract = _attachment_ref_v2_module()
    payload = _attachment_v2_payload()
    before = dict(payload)

    first = contract.validate_attachment_ref_v2(
        "upsert",
        payload,
        envelope_created_at_client=ATTACHMENT_V2_TIMESTAMP,
        authenticated_device_id="device-1",
    )
    replay = contract.validate_attachment_ref_v2(
        "upsert",
        payload,
        envelope_created_at_client=ATTACHMENT_V2_TIMESTAMP,
        authenticated_device_id="device-1",
        prior_payload=first,
        prior_operation="upsert",
    )

    assert payload == before
    assert replay == first
    assert replay.model_dump(mode="json") == before


def test_attachment_ref_v2_adapter_rejects_version_one_writes() -> None:
    from tldw_Server_API.app.core.Sync.v2.adapters import (
        AdapterRejected,
        AttachmentRefAdapter,
    )

    outcome = AttachmentRefAdapter(v2_writes_enabled=True).evaluate_envelope(
        _attachment_ref(),
        dataset=_attachment_v2_dataset(),
    )

    assert isinstance(outcome, AdapterRejected)
    assert outcome.error_code == "attachment_ref_v1_immutable"


def test_attachment_ref_v2_uses_the_dedicated_domain_adapter() -> None:
    from tldw_Server_API.app.core.Sync.v2.domain_adapters.attachment_refs import (
        AttachmentRefDomainAdapter,
    )

    adapter = default_sync_v2_registry().get("attachment.ref")

    assert isinstance(adapter, AttachmentRefDomainAdapter)


def test_attachment_ref_v2_adapter_rejects_v1_v2_object_id_collision() -> None:
    from tldw_Server_API.app.core.Sync.v2.adapters import (
        AdapterConflict,
        AttachmentRefAdapter,
        SyncAdapterContext,
    )

    legacy = _attachment_ref(
        object_id=ATTACHMENT_V2_ID,
        payload=_attachment_payload(attachment_id=ATTACHMENT_V2_ID),
    )
    outcome = AttachmentRefAdapter(v2_writes_enabled=True).evaluate_envelope(
        _attachment_v2_envelope(),
        dataset=_attachment_v2_dataset(),
        context=SyncAdapterContext(
            prior_envelopes=(legacy,),
            supports_attachments=True,
        ),
    )

    assert isinstance(outcome, AdapterConflict)
    assert outcome.conflict_type == "attachment_ref_immutable_version_collision"


@pytest.mark.parametrize(
    ("state", "enabled"),
    [("initializing", True), ("ready", False), ("failed", True)],
)
def test_attachment_ref_v2_writes_require_gate_and_ready_dataset(
    state: str,
    enabled: bool,
) -> None:
    from tldw_Server_API.app.core.Sync.v2.adapters import (
        AdapterRejected,
        AttachmentRefAdapter,
    )

    outcome = AttachmentRefAdapter(v2_writes_enabled=enabled).evaluate_envelope(
        _attachment_v2_envelope(),
        dataset=_attachment_v2_dataset(state=state),
    )

    assert isinstance(outcome, AdapterRejected)
    assert outcome.error_code == "attachment_ref_v2_not_writable"


def test_attachment_ref_v2_writes_require_generic_blob_transfer_gate() -> None:
    from tldw_Server_API.app.core.Sync.v2.adapters import (
        AdapterRejected,
        AttachmentRefAdapter,
        SyncAdapterContext,
    )

    outcome = AttachmentRefAdapter(v2_writes_enabled=True).evaluate_envelope(
        _attachment_v2_envelope(),
        dataset=_attachment_v2_dataset(),
        context=SyncAdapterContext(supports_attachments=False),
    )

    assert isinstance(outcome, AdapterRejected)
    assert outcome.error_code == "attachment_ref_v2_not_writable"


def test_attachment_ref_v2_writes_require_server_trusted_encryption_policy() -> None:
    from tldw_Server_API.app.core.Sync.v2.adapters import (
        AdapterRejected,
        AttachmentRefAdapter,
    )

    outcome = AttachmentRefAdapter(v2_writes_enabled=True).evaluate_envelope(
        _attachment_v2_envelope(),
        dataset=_attachment_v2_dataset(encryption_policy="client_private_v1"),
    )

    assert isinstance(outcome, AdapterRejected)
    assert outcome.error_code == "attachment_ref_v2_not_writable"


def test_attachment_ref_v2_verified_bootstrap_preserves_legacy_provenance() -> None:
    from tldw_Server_API.app.core.Sync.v2.adapters import (
        AdapterAccepted,
        AttachmentRefAdapter,
        SyncAdapterContext,
    )

    bootstrap_id = "bootstrap-1"
    payload = _attachment_v2_payload(
        created_at="2020-01-02T03:04:05+00:00",
        created_by="legacy-device",
    )
    envelope = _attachment_v2_envelope(
        device_id="server-origin",
        payload=payload,
        routing_metadata={
            "bootstrap_capture": True,
            "bootstrap_id": bootstrap_id,
        },
    )
    outcome = AttachmentRefAdapter(v2_writes_enabled=True).evaluate_envelope(
        envelope,
        dataset=_attachment_v2_dataset(
            state="initializing",
            bootstrap_id=bootstrap_id,
        ),
        context=SyncAdapterContext(
            trusted_server_origin=True,
            attachment_ref_bootstrap_id=bootstrap_id,
        ),
    )

    assert isinstance(outcome, AdapterAccepted)


@pytest.mark.parametrize(
    "missing_key",
    [
        "attachment_id",
        "parent_domain",
        "parent_object_id",
        "content_type",
        "size_bytes",
        "payload_hash",
        "availability",
    ],
)
def test_attachment_ref_schema_requires_metadata_fields(missing_key: str) -> None:
    payload = _attachment_payload()
    payload.pop(missing_key)

    with pytest.raises(ValidationError, match="attachment.ref envelopes require payload metadata fields"):
        SyncV2Envelope(
            dataset_id="dataset-1",
            client_envelope_id=f"env-missing-{missing_key}",
            domain="attachment.ref",
            operation="upsert",
            object_id="att-1",
            payload=payload,
            payload_hash="sha256:blob-v1",
            encryption_metadata={"policy": "server_trusted_v1"},
        )


def test_attachment_ref_schema_rejects_mismatched_object_id_and_attachment_id() -> None:
    with pytest.raises(ValidationError, match="attachment.ref object_id must match payload attachment_id"):
        SyncV2Envelope(
            dataset_id="dataset-1",
            client_envelope_id="env-mismatched-object",
            domain="attachment.ref",
            operation="upsert",
            object_id="att-alias",
            payload=_attachment_payload(attachment_id="att-1"),
            payload_hash="sha256:blob-v1",
            encryption_metadata={"policy": "server_trusted_v1"},
        )


def test_attachment_ref_is_accepted_and_visible_through_pull(
    sync_service: SyncV2Service,
) -> None:
    result = _push_one(sync_service, _attachment_ref())

    assert [item.client_envelope_id for item in result.accepted] == ["env-attachment-1"]
    assert result.rejected == []
    assert result.conflicts == []
    state = sync_service.store.get_object_state("dataset-1", "attachment.ref", "att-1")
    assert state is not None
    assert state.object_hash == "sha256:blob-v1"
    assert state.deleted is False

    pulled = sync_service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-2",
        domains=["attachment.ref"],
    )

    assert [item.client_envelope_id for item in pulled.envelopes] == ["env-attachment-1"]
    assert pulled.envelopes[0].payload["attachment_id"] == "att-1"
    assert pulled.envelopes[0].payload["parent_domain"] == "notes.note"


def test_duplicate_attachment_ref_same_payload_is_idempotent(
    sync_service: SyncV2Service,
) -> None:
    first = _push_one(sync_service, _attachment_ref())
    duplicate = _push_one(
        sync_service,
        _attachment_ref_after(
            sync_service,
            client_envelope_id="env-attachment-duplicate",
            client_sequence=2,
            object_revision=1,
        ),
    )

    assert [item.client_envelope_id for item in first.accepted] == ["env-attachment-1"]
    assert [item.client_envelope_id for item in duplicate.accepted] == ["env-attachment-duplicate"]
    assert duplicate.rejected == []
    assert duplicate.conflicts == []
    assert sync_service.store.list_conflicts("dataset-1") == []
    stored = sync_service.store.list_envelopes_for_entity(
        "dataset-1",
        "attachment.ref",
        entity_id="att-1",
        limit=10,
    )
    assert sorted((item.client_envelope_id, item.apply_status) for item in stored) == [
        ("env-attachment-1", "applied"),
        ("env-attachment-duplicate", "applied"),
    ]


def test_duplicate_attachment_ref_different_payload_hash_conflicts_without_overwrite(
    sync_service: SyncV2Service,
) -> None:
    _push_one(sync_service, _attachment_ref())

    divergent = _push_one(
        sync_service,
        _attachment_ref(
            client_envelope_id="env-attachment-divergent",
            client_sequence=2,
            payload=_attachment_payload(
                content_type="image/jpeg",
                payload_hash="sha256:blob-v2",
            ),
            payload_hash="sha256:blob-v2",
        ),
    )

    assert divergent.accepted == []
    assert [item.client_envelope_id for item in divergent.conflicts] == ["env-attachment-divergent"]
    conflicts = sync_service.store.list_conflicts("dataset-1")
    assert len(conflicts) == 1
    assert conflicts[0].domain == "attachment.ref"
    assert conflicts[0].conflict_type == "attachment_ref_hash_mismatch"

    history = sync_service.store.list_envelopes_after(
        "dataset-1",
        0,
        domains=["attachment.ref"],
        status=None,
    )
    assert [(item.client_envelope_id, item.status, item.payload_hash) for item in history] == [
        ("env-attachment-1", "accepted", "sha256:blob-v1"),
        ("env-attachment-divergent", "conflict", "sha256:blob-v2"),
    ]


def test_attachment_ref_mismatched_object_id_cannot_bypass_hash_guard(
    sync_service: SyncV2Service,
) -> None:
    _push_one(sync_service, _attachment_ref())

    bypass_attempt = _push_one(
        sync_service,
        _attachment_ref(
            client_envelope_id="env-attachment-alias-divergent",
            client_sequence=2,
            object_id="att-alias",
            stable_key="attachment:att-alias",
            payload=_attachment_payload(
                attachment_id="att-1",
                content_type="image/jpeg",
                payload_hash="sha256:blob-v2",
            ),
            payload_hash="sha256:blob-v2",
        ),
    )

    assert bypass_attempt.accepted == []
    assert bypass_attempt.conflicts == []
    assert [(item.client_envelope_id, item.error_code) for item in bypass_attempt.rejected] == [
        ("env-attachment-alias-divergent", "attachment_ref_object_id_mismatch")
    ]
    assert sync_service.store.get_object_state("dataset-1", "attachment.ref", "att-alias") is None
    history = sync_service.store.list_envelopes_after(
        "dataset-1",
        0,
        domains=["attachment.ref"],
        status=None,
    )
    assert [(item.client_envelope_id, item.status, item.object_id) for item in history] == [
        ("env-attachment-1", "accepted", "att-1"),
    ]


def test_stale_upsert_after_tombstone_cannot_resurrect_attachment_ref(
    sync_service: SyncV2Service,
) -> None:
    _push_one(sync_service, _attachment_ref())
    tombstone = _push_one(
        sync_service,
        _attachment_ref_after(
            sync_service,
            client_envelope_id="env-attachment-tombstone",
            client_sequence=2,
            operation="tombstone",
        ),
    )

    deleted_state = sync_service.store.get_object_state("dataset-1", "attachment.ref", "att-1")
    assert [item.client_envelope_id for item in tombstone.accepted] == ["env-attachment-tombstone"]
    assert deleted_state is not None
    assert deleted_state.deleted is True

    stale_upsert = _push_one(
        sync_service,
        _attachment_ref_after(
            sync_service,
            client_envelope_id="env-attachment-stale-upsert",
            client_sequence=3,
        ),
    )

    current_state = sync_service.store.get_object_state("dataset-1", "attachment.ref", "att-1")
    assert stale_upsert.accepted == []
    assert stale_upsert.rejected == []
    assert [item.client_envelope_id for item in stale_upsert.conflicts] == ["env-attachment-stale-upsert"]
    assert current_state is not None
    assert current_state.deleted is True
    assert current_state.latest_server_cursor == deleted_state.latest_server_cursor
    conflicts = sync_service.store.list_conflicts("dataset-1")
    assert [(item.local_envelope_id, item.conflict_type) for item in conflicts] == [
        ("env-attachment-stale-upsert", "attachment_ref_tombstoned")
    ]


def test_restore_preview_reports_attachment_refs_and_missing_blobs(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    _push_one(sync_service, _attachment_ref())
    _push_one(
        sync_service,
        _attachment_ref(
            client_envelope_id="env-attachment-server",
            object_id="att-server",
            stable_key="attachment:att-server",
            client_sequence=2,
            payload=_attachment_payload(
                attachment_id="att-server",
                payload_hash="sha256:server-blob",
                availability="server",
            ),
            payload_hash="sha256:server-blob",
        ),
    )

    response = client.post(
        "/api/v1/sync/restore/preview",
        json={"dataset_ids": ["dataset-1"], "local_inventory": []},
    )

    assert response.status_code == 200
    body = response.json()
    assert {item["attachment_id"] for item in body["attachment_refs"]} == {
        "att-1",
        "att-server",
    }
    assert body["attachment_refs"][0]["parent_domain"] == "notes.note"
    assert [item["attachment_id"] for item in body["missing_blobs"]] == ["att-1"]
    assert body["warnings"] == [
        {
            "code": "sync_key_recovery_missing",
            "message": "No active Sync v2 key recovery bundle is available for this dataset.",
            "dataset_id": "dataset-1",
            "attachment_id": None,
            "object_id": None,
            "payload_hash": None,
        },
        {
            "code": "sync_attachment_blob_missing",
            "message": "Attachment blob is not available from the Sync v2 M1 server.",
            "dataset_id": "dataset-1",
            "attachment_id": "att-1",
            "object_id": "att-1",
            "payload_hash": "sha256:blob-v1",
        },
    ]


def test_restore_preview_omits_tombstoned_attachment_refs(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    _push_one(sync_service, _attachment_ref())
    _push_one(
        sync_service,
        _attachment_ref_after(
            sync_service,
            client_envelope_id="env-attachment-tombstone",
            client_sequence=2,
            operation="tombstone",
        ),
    )

    response = client.post(
        "/api/v1/sync/restore/preview",
        json={"dataset_ids": ["dataset-1"], "local_inventory": []},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["attachment_refs"] == []
    assert body["missing_blobs"] == []
    assert body["warnings"] == [
        {
            "code": "sync_key_recovery_missing",
            "message": "No active Sync v2 key recovery bundle is available for this dataset.",
            "dataset_id": "dataset-1",
            "attachment_id": None,
            "object_id": None,
            "payload_hash": None,
        }
    ]


def test_public_attachment_revision_round_trips_and_is_valid_for_followup_cas(
    client: TestClient,
) -> None:
    initial = client.post(
        "/api/v1/sync/push",
        json={
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "envelopes": [
                {
                    "dataset_id": "dataset-1",
                    "client_envelope_id": "env-public-attachment-1",
                    "device_id": "device-1",
                    "client_sequence": 10,
                    "domain": "attachment.ref",
                    "operation": "upsert",
                    "object_id": "att-public",
                    "object_revision": 1,
                    "payload": _attachment_payload(attachment_id="att-public"),
                    "payload_hash": "sha256:blob-v1",
                    "encryption_metadata": {"policy": "server_trusted_v1"},
                }
            ],
        },
    )

    assert initial.status_code == 200
    assert initial.json()["accepted"][0]["object_revision"] == 1
    pulled = client.get(
        "/api/v1/sync/pull",
        params={
            "dataset_id": "dataset-1",
            "device_id": "device-2",
            "cursor": "0",
            "domain": "attachment.ref",
        },
    )
    assert pulled.status_code == 200
    head = pulled.json()["envelopes"][0]
    assert head["object_revision"] == 1

    tombstone = client.post(
        "/api/v1/sync/push",
        json={
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "envelopes": [
                {
                    "dataset_id": "dataset-1",
                    "client_envelope_id": "env-public-attachment-2",
                    "device_id": "device-1",
                    "client_sequence": 11,
                    "domain": "attachment.ref",
                    "operation": "tombstone",
                    "object_id": "att-public",
                    "object_revision": 2,
                    "base_server_cursor": head["server_cursor"],
                    "base_object_revision": head["object_revision"],
                    "base_object_hash": head["payload_hash"],
                    "payload": _attachment_payload(attachment_id="att-public"),
                    "payload_hash": "sha256:blob-v1",
                    "encryption_metadata": {"policy": "server_trusted_v1"},
                }
            ],
        },
    )

    assert tombstone.status_code == 200
    assert tombstone.json()["conflicts"] == []
    assert tombstone.json()["accepted"][0]["object_revision"] == 2


def test_revisionless_attachment_head_accepts_its_projected_state_as_base(
    sync_service: SyncV2Service,
) -> None:
    initial = _push_one(sync_service, _attachment_ref(object_revision=None))
    assert [item.object_revision for item in initial.accepted] == [1]

    tombstone = _push_one(
        sync_service,
        _attachment_ref_after(
            sync_service,
            client_envelope_id="env-legacy-attachment-tombstone",
            client_sequence=2,
            operation="tombstone",
        ),
    )

    assert tombstone.conflicts == []
    assert [item.client_envelope_id for item in tombstone.accepted] == ["env-legacy-attachment-tombstone"]


def test_blob_upload_and_download_are_explicitly_unsupported_in_m1(
    client: TestClient,
) -> None:
    upload = client.post(
        "/api/v1/sync/attachments",
        json={
            "dataset_id": "dataset-1",
            "domain": "attachment.ref",
            "object_id": "att-1",
            "attachment_id": "att-1",
            "content_type": "image/png",
            "size_bytes": 512,
            "payload_ciphertext": "ciphertext",
            "payload_hash": "sha256:blob-v1",
            "encryption_policy": "server_trusted_v1",
        },
    )
    download = client.get(
        "/api/v1/sync/attachments/att-1",
        params={"dataset_id": "dataset-1"},
    )

    assert upload.status_code == 501
    assert upload.json()["detail"]["error_code"] == "sync_blob_transfer_not_supported"
    assert download.status_code == 501
    assert download.json()["detail"]["error_code"] == "sync_blob_transfer_not_supported"


def test_invalid_blob_upload_is_unsupported_before_m2_schema_validation(
    client: TestClient,
) -> None:
    upload = client.post(
        "/api/v1/sync/attachments",
        content=b"\x00not-json-attachment-bytes",
        headers={"content-type": "application/octet-stream"},
    )

    assert upload.status_code == 501
    assert upload.json()["detail"]["error_code"] == "sync_blob_transfer_not_supported"
