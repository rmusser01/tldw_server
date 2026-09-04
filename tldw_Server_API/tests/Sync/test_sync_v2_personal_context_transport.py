from __future__ import annotations

import hashlib
import hmac
import sqlite3
import uuid
from dataclasses import replace
from pathlib import Path

import pytest
from tldw_profile_core.canonical import canonical_json_bytes

from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Personalization.personal_context_publication import (
    CanonicalApplyReceipt,
)
from tldw_Server_API.app.core.Sync.v2.adapters import (
    AdapterAccepted,
    SyncAdapterRegistry,
)
from tldw_Server_API.app.core.Sync.v2.domain_adapters.personal_context import (
    PersonalContextDomainAdapter,
)
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.materializers.personal_context import (
    PersonalContextMaterializer,
)
from tldw_Server_API.app.core.Sync.v2.models import (
    PERSONAL_CONTEXT_SYNC_DOMAINS,
    SyncDatasetCreate,
    SyncDomain,
    SyncEnvelopeCreate,
)
from tldw_Server_API.app.core.Sync.v2.personal_context_ongoing_contract import (
    PersonalContextAuthorityMetadata,
    PersonalContextExchangeProof,
)
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.server_origin import (
    insert_personal_context_authority,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store
from tldw_Server_API.tests.Personalization.personal_context_test_support import (
    preference_record,
)

pytestmark = pytest.mark.unit

DOMAIN = "personal_context.record"
DATASET_ID = "dataset-a"
PROFILE_ID = "profile-a"
INTEGRITY_KEY = b"i" * 32
ENCRYPTION_KEY = b"e" * 32
INTEGRITY_KEY_ID = "personal-context-integrity-v1"
CANARY = "SERVER-SYNC-PLAINTEXT-CANARY-21e8"
EXCHANGE = PersonalContextExchangeProof(
    ongoing_sync_version=1,
    activation_epoch="epoch_0123456789abcdef",
    continuity_token="continuity_0123456789abcdef",
)


class _RecordingService:
    def __init__(self) -> None:
        self.values = []

    def apply_sync_object(self, **values):
        self.values.append(values["value"])
        return values["value"]

    def apply_sync_ingress(self, **values):
        self.values.append(values["value"])
        identity = values["identity"]
        value = values["value"]
        return CanonicalApplyReceipt(
            resulting_object_id=value.record_id,
            resulting_version_id=value.version_id,
            manifest_revision=1,
            manifest_version_id="manifest-v1",
            purge_generation=identity.purge_generation,
            publication_batch_id="batch-1",
            profile_publication_sequence=1,
            receipt_id=str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    "tldw:personal-context:ingress:"
                    f"{identity.dataset_id}:{identity.device_id}:{identity.client_envelope_id}",
                )
            ),
            dataset_id=identity.dataset_id,
            device_id=identity.device_id,
            client_envelope_id=identity.client_envelope_id,
            canonical_payload_digest=identity.canonical_payload_digest,
            wire_entity_version=identity.wire_entity_version,
        )


class _LegacyNotesAdapter:
    domain: SyncDomain = "notes.note"
    supported_adapter_versions = {1}

    def evaluate_envelope(self, envelope, *, dataset):
        del dataset
        return AdapterAccepted(client_envelope_id=envelope.client_envelope_id)


def _tag(payload: dict[str, object]) -> str:
    digest = hmac.new(INTEGRITY_KEY, canonical_json_bytes(payload), hashlib.sha256)
    return f"hmac-sha256-v1:{digest.hexdigest()}"


def _envelope(
    payload: dict[str, object],
    *,
    client_envelope_id: str,
    base_server_cursor: int | None = None,
    base_object_revision: int | None = None,
    base_object_hash: str | None = None,
) -> SyncEnvelopeCreate:
    return SyncEnvelopeCreate(
        dataset_id=DATASET_ID,
        client_envelope_id=client_envelope_id,
        device_id="device-a",
        domain=DOMAIN,
        operation="upsert",
        object_id=str(payload["record_id"]),
        parent_id=str(payload["scope_id"]),
        adapter_version=1,
        schema_version=1,
        payload=payload,
        payload_hash=_tag(payload),
        payload_size_bytes=len(canonical_json_bytes(payload)),
        base_server_cursor=base_server_cursor,
        base_object_revision=base_object_revision,
        base_object_hash=base_object_hash,
        base_version=payload.get("parent_version_id"),
        entity_version=str(payload["version_id"]),
        encryption_metadata={"policy": "server_trusted_v1"},
        routing_metadata={
            "integrity_key_id": INTEGRITY_KEY_ID,
            "profile_id": PROFILE_ID,
            "purge_generation": None,
        },
    )


def _service(
    tmp_path: Path, *, active_exchange: bool = True
) -> tuple[SyncV2Service, _RecordingService, Path]:
    sqlite_path = tmp_path / "sync.db"
    store = SyncV2Store(SyncDatabase(sqlite_path=sqlite_path))
    adapters = SyncAdapterRegistry(
        [
            PersonalContextDomainAdapter(
                domain=domain,
                integrity_key_resolver=lambda _dataset, _key_id: INTEGRITY_KEY,
                encryption_key_resolver=lambda _dataset: (ENCRYPTION_KEY, 1),
            )
            for domain in PERSONAL_CONTEXT_SYNC_DOMAINS
        ]
        + [_LegacyNotesAdapter()]
    )
    target = _RecordingService()
    service = SyncV2Service(
        store=store,
        adapters=adapters,
        materializers={
            domain: PersonalContextMaterializer(
                domain=domain,
                service_resolver=lambda _user_id: target,
            )
            for domain in PERSONAL_CONTEXT_SYNC_DOMAINS
        },
        settings=SyncV2Settings(
            pull_token_signing_secret="test-only-pull-secret",
            server_trusted_encryption=server_trusted_encryption_status_from_config(
                mode="managed_storage",
                server_trusted_enabled=True,
                auth_mode="multi_user",
            )
        ),
    )
    for device_id in ("device-a", "device-b"):
        service.register_device(
            user_id="user-a",
            display_name=device_id,
            client_type="chatbook",
            device_id=device_id,
            capabilities={
                "supported_adapter_versions": {
                    **{domain: [1] for domain in PERSONAL_CONTEXT_SYNC_DOMAINS},
                    "notes.note": [1],
                }
            },
        )
    store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id=DATASET_ID,
            owner_user_id="user-a",
            encryption_policy="server_trusted_v1",
            domains=[*PERSONAL_CONTEXT_SYNC_DOMAINS, "notes.note"],
            metadata={
                "personal_context": {
                    "profile_id": PROFILE_ID,
                    "integrity_key_id": INTEGRITY_KEY_ID,
                    "purge_generation": 0,
                    "link_state": "complete",
                    **(
                        {
                            "ongoing_sync_version": 1,
                            "activation_epoch": EXCHANGE.activation_epoch,
                            "continuity_token": EXCHANGE.continuity_token,
                        }
                        if active_exchange
                        else {"ongoing_sync_version": 0}
                    ),
                    "link_receipts": {
                        "device-a": {
                            "profile_id": PROFILE_ID,
                            "integrity_key_id": INTEGRITY_KEY_ID,
                            "purge_generation": 0,
                            "bootstrap_cursor": "fixture-cursor",
                        }
                    },
                }
            },
        )
    )
    store.complete_personal_context_link_receipt(
        user_id="user-a",
        dataset_id=DATASET_ID,
        device_id="device-a",
        profile_id=PROFILE_ID,
        integrity_key_id=INTEGRITY_KEY_ID,
        purge_generation=0,
        bootstrap_cursor="fixture-cursor",
    )
    store.complete_personal_context_link_receipt(
        user_id="user-a",
        dataset_id=DATASET_ID,
        device_id="device-b",
        profile_id=PROFILE_ID,
        integrity_key_id=INTEGRITY_KEY_ID,
        purge_generation=0,
        bootstrap_cursor="fixture-cursor",
    )
    return service, target, sqlite_path


def _insert_authority(
    service: SyncV2Service,
    *,
    record_id: str,
    sequence: int,
    applied: bool = True,
) -> int:
    payload = preference_record(
        record_id=record_id,
        version_id=f"{record_id}-version",
        value=f"clear-{record_id}",
    ).model_dump(mode="json")
    envelope = replace(
        _envelope(
            payload,
            client_envelope_id=f"authority-{record_id}",
        ),
        routing_metadata={
            "integrity_key_id": INTEGRITY_KEY_ID,
            "profile_id": PROFILE_ID,
            "purge_generation": 0,
        },
    )
    dataset = service.store.get_dataset(DATASET_ID)
    assert dataset is not None
    protected = service._protect_personal_context_for_storage(dataset, envelope)
    stored = insert_personal_context_authority(
        service,
        envelope=protected,
        authority=PersonalContextAuthorityMetadata(
            role="home_authority",
            publication_batch_id=f"publication-batch-{sequence:04d}",
            profile_publication_sequence=sequence,
            batch_ordinal=0,
            batch_size=1,
        ),
    )
    assert stored.server_cursor is not None
    if applied:
        service.store.mark_envelope_apply_status(
            stored.server_cursor,
            apply_status="applied",
        )
    return stored.server_cursor


def _insert_hidden_ingress(service: SyncV2Service, *, ordinal: int) -> int:
    record_id = f"hidden-{ordinal:03d}"
    payload = preference_record(
        record_id=record_id,
        version_id=f"{record_id}-version",
    ).model_dump(mode="json")
    envelope = replace(
        _envelope(payload, client_envelope_id=f"ingress-{record_id}"),
        apply_status="applied",
        routing_metadata={
            "integrity_key_id": INTEGRITY_KEY_ID,
            "profile_id": PROFILE_ID,
            "purge_generation": 0,
            "personal_context_authority": {"role": "client_ingress"},
        },
    )
    dataset = service.store.get_dataset(DATASET_ID)
    assert dataset is not None
    stored = service.store.insert_envelope(
        service._protect_personal_context_for_storage(dataset, envelope)
    )
    assert stored.server_cursor is not None
    return stored.server_cursor


def _insert_note(service: SyncV2Service, *, object_id: str) -> int:
    payload = {"note_id": object_id, "title": object_id}
    canonical = canonical_json_bytes(payload)
    stored = service.store.insert_envelope(
        SyncEnvelopeCreate(
            dataset_id=DATASET_ID,
            client_envelope_id=f"note-{object_id}",
            device_id="device-a",
            domain="notes.note",
            operation="upsert",
            object_id=object_id,
            adapter_version=1,
            schema_version=1,
            payload=payload,
            payload_hash="sha256:" + hashlib.sha256(canonical).hexdigest(),
            payload_size_bytes=len(canonical),
            apply_status="applied",
        )
    )
    assert stored.server_cursor is not None
    return stored.server_cursor


def test_two_version_push_satisfies_sync_cas_and_encrypts_transport_history(
    tmp_path: Path,
) -> None:
    service, target, sqlite_path = _service(tmp_path)
    first_payload = preference_record(value=CANARY).model_dump(mode="json")
    first = _envelope(first_payload, client_envelope_id="device-a:record:1")

    first_result = service.push(
        user_id="user-a",
        dataset_id=DATASET_ID,
        device_id="device-a",
        envelopes=[first],
        personal_context_exchange=EXCHANGE,
    )

    assert first_result.rejected == []
    assert first_result.conflicts == []
    assert len(first_result.accepted) == 1
    assert first_result.personal_context_exchange == EXCHANGE
    first_cursor = first_result.accepted[0].server_sequence
    replayed = service.push(
        user_id="user-a",
        dataset_id=DATASET_ID,
        device_id="device-a",
        envelopes=[first],
        personal_context_exchange=EXCHANGE,
    )
    assert replayed.rejected == []
    assert replayed.conflicts == []
    assert replayed.accepted[0].server_sequence == first_cursor
    assert len(target.values) == 1
    with sqlite3.connect(sqlite_path) as connection:
        receipt = connection.execute(
            "SELECT * FROM sync_personal_context_ingress_receipts "
            "WHERE server_sequence = ?",
            (first_cursor,),
        ).fetchone()
    assert receipt is not None
    assert receipt[3] == "device-a:record:1"
    assert receipt[7] == str(first_payload["version_id"])
    assert receipt[13] == str(first_payload["version_id"])
    second_payload = preference_record(
        version_id="record-v2",
        parent_version_id=str(first_payload["version_id"]),
        value="structured",
    ).model_dump(mode="json")
    second = _envelope(
        second_payload,
        client_envelope_id="device-a:record:2",
        base_server_cursor=first_cursor,
        base_object_revision=first_result.accepted[0].object_revision,
        base_object_hash=first.payload_hash,
    )

    second_result = service.push(
        user_id="user-a",
        dataset_id=DATASET_ID,
        device_id="device-a",
        envelopes=[second],
        personal_context_exchange=EXCHANGE,
    )

    assert second_result.rejected == []
    assert second_result.conflicts == []
    assert len(second_result.accepted) == 1
    assert [value.version_id for value in target.values] == [
        first_payload["version_id"],
        "record-v2",
    ]
    pulled = service.pull(
        user_id="user-a",
        dataset_id=DATASET_ID,
        device_id="device-b",
        domains=[DOMAIN],
        personal_context_exchange=EXCHANGE,
    )
    assert pulled.envelopes == []
    assert pulled.personal_context_exchange == EXCHANGE
    durable = b"".join(
        path.read_bytes()
        for path in tmp_path.iterdir()
        if path.name.startswith(sqlite_path.name)
    )
    assert CANARY.encode() not in durable


def test_signed_personal_context_pull_never_exposes_client_ingress(
    tmp_path: Path,
) -> None:
    """A bootstrap-style signed cursor must retain the ingress egress gate."""

    service, _target, _sqlite_path = _service(tmp_path)
    payload = preference_record(value="private").model_dump(mode="json")
    pushed = service.push(
        user_id="user-a",
        dataset_id=DATASET_ID,
        device_id="device-a",
        envelopes=[_envelope(payload, client_envelope_id="device-a:record:signed")],
        personal_context_exchange=EXCHANGE,
    )
    assert pushed.accepted
    device = service._require_registered_device("user-a", "device-b")
    signed_cursor = service._encode_pull_token(
        dataset_id=DATASET_ID,
        device_id="device-b",
        version_set=service._pull_version_set(device),
        watermarks={(DOMAIN, 1): 0},
    )

    pulled = service.pull(
        user_id="user-a",
        dataset_id=DATASET_ID,
        device_id="device-b",
        domains=[DOMAIN],
        cursor=signed_cursor,
        include_own_changes=True,
        personal_context_exchange=EXCHANGE,
    )

    assert pulled.envelopes == []


def test_signed_authority_pull_restores_clear_payload_and_retries_lookahead(
    tmp_path: Path,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    first_cursor = _insert_authority(
        service,
        record_id="authority-first",
        sequence=1,
    )
    second_cursor = _insert_authority(
        service,
        record_id="authority-second",
        sequence=2,
    )
    device = service._require_registered_device("user-a", "device-b")
    signed_cursor = service._encode_pull_token(
        dataset_id=DATASET_ID,
        device_id="device-b",
        version_set=service._pull_version_set(device),
        watermarks={(DOMAIN, 1): 0},
    )

    first = service.pull(
        user_id="user-a",
        dataset_id=DATASET_ID,
        device_id="device-b",
        domains=[DOMAIN],
        cursor=signed_cursor,
        page_size=1,
        include_own_changes=True,
        personal_context_exchange=EXCHANGE,
    )

    assert [item.server_cursor for item in first.envelopes] == [first_cursor]
    assert first.envelopes[0].payload["payload"]["value"] == "clear-authority-first"
    assert first.has_more is True
    assert first.next_cursor is not None

    second = service.pull(
        user_id="user-a",
        dataset_id=DATASET_ID,
        device_id="device-b",
        domains=[DOMAIN],
        cursor=first.next_cursor,
        page_size=1,
        include_own_changes=True,
        personal_context_exchange=EXCHANGE,
    )

    assert [item.server_cursor for item in second.envelopes] == [second_cursor]
    assert second.envelopes[0].payload["payload"]["value"] == "clear-authority-second"


def test_recovery_scan_stops_at_100_hidden_rows_without_skipping_101(
    tmp_path: Path,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    cursors = [_insert_hidden_ingress(service, ordinal=index) for index in range(101)]

    first = service.store.scan_personal_context_authority(
        DATASET_ID,
        after_server_cursor=0,
        limit=1,
        row_budget=100,
        deadline_ns=2**63 - 1,
        domains=[DOMAIN],
        adapter_versions=[1],
    )

    assert first.raw_rows_scanned == 100
    assert first.raw_scan_watermark == cursors[99]
    assert first.visible_envelopes == []
    assert first.source_exhausted is False

    second = service.store.scan_personal_context_authority(
        DATASET_ID,
        after_server_cursor=first.raw_scan_watermark,
        limit=1,
        row_budget=100,
        deadline_ns=2**63 - 1,
        domains=[DOMAIN],
        adapter_versions=[1],
    )

    assert second.raw_rows_scanned == 1
    assert second.raw_scan_watermark == cursors[100]
    assert second.source_exhausted is True


def test_legacy_authority_pull_stops_before_pending_barrier(
    tmp_path: Path,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    first_cursor = _insert_authority(
        service,
        record_id="before-barrier",
        sequence=1,
    )
    pending_cursor = _insert_authority(
        service,
        record_id="pending-barrier",
        sequence=2,
        applied=False,
    )
    _insert_authority(
        service,
        record_id="after-barrier",
        sequence=3,
    )

    pulled = service.pull(
        user_id="user-a",
        dataset_id=DATASET_ID,
        device_id="device-b",
        domains=[DOMAIN],
        cursor=0,
        page_size=10,
        include_own_changes=True,
        personal_context_exchange=EXCHANGE,
    )

    assert [item.server_cursor for item in pulled.envelopes] == [first_cursor]
    assert pulled.next_cursor == str(first_cursor)
    assert pulled.has_more is True
    assert pending_cursor > first_cursor


def test_old_generation_authority_is_invisible_and_cannot_barrier_active_scan(
    tmp_path: Path,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    _insert_authority(service, record_id="old-applied", sequence=1)
    pending_cursor = _insert_authority(
        service,
        record_id="old-pending",
        sequence=2,
        applied=False,
    )

    scan = service.store.scan_personal_context_authority(
        DATASET_ID,
        after_server_cursor=0,
        limit=10,
        deadline_ns=2**63 - 1,
        domains=[DOMAIN],
        adapter_versions=[1],
        profile_id=PROFILE_ID,
        integrity_key_id=INTEGRITY_KEY_ID,
        purge_generation=1,
    )

    assert scan.visible_envelopes == []
    assert scan.raw_scan_watermark == pending_cursor
    assert scan.source_exhausted is True


def test_signed_mixed_pull_preserves_each_stream_watermark_without_duplicates(
    tmp_path: Path,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    note_cursor = _insert_note(service, object_id="note-first")
    _insert_hidden_ingress(service, ordinal=1)
    authority_cursor = _insert_authority(
        service,
        record_id="mixed-authority",
        sequence=1,
    )
    device = service._require_registered_device("user-a", "device-b")
    streams = [("notes.note", 1), (DOMAIN, 1)]
    token = service._encode_pull_token(
        dataset_id=DATASET_ID,
        device_id="device-b",
        version_set=service._pull_version_set(device),
        watermarks=dict.fromkeys(streams, 0),
    )

    first = service.pull(
        user_id="user-a",
        dataset_id=DATASET_ID,
        device_id="device-b",
        domains=["notes.note", DOMAIN],
        cursor=token,
        page_size=1,
        include_own_changes=True,
        personal_context_exchange=EXCHANGE,
    )
    assert [item.server_cursor for item in first.envelopes] == [note_cursor]

    second = service.pull(
        user_id="user-a",
        dataset_id=DATASET_ID,
        device_id="device-b",
        domains=["notes.note", DOMAIN],
        cursor=first.next_cursor,
        page_size=1,
        include_own_changes=True,
        personal_context_exchange=EXCHANGE,
    )
    assert [item.server_cursor for item in second.envelopes] == [authority_cursor]

    third = service.pull(
        user_id="user-a",
        dataset_id=DATASET_ID,
        device_id="device-b",
        domains=["notes.note", DOMAIN],
        cursor=second.next_cursor,
        page_size=1,
        include_own_changes=True,
        personal_context_exchange=EXCHANGE,
    )
    assert third.envelopes == []


def test_legacy_mixed_pull_retries_hidden_prefix_without_duplicate_delivery(
    tmp_path: Path,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    note_cursor = _insert_note(service, object_id="legacy-note")
    _insert_hidden_ingress(service, ordinal=1)
    authority_cursor = _insert_authority(
        service,
        record_id="legacy-authority",
        sequence=1,
    )

    first = service.pull(
        user_id="user-a",
        dataset_id=DATASET_ID,
        device_id="device-b",
        domains=["notes.note", DOMAIN],
        cursor=0,
        page_size=1,
        include_own_changes=True,
        personal_context_exchange=EXCHANGE,
    )
    assert [item.server_cursor for item in first.envelopes] == [note_cursor]

    second = service.pull(
        user_id="user-a",
        dataset_id=DATASET_ID,
        device_id="device-b",
        domains=["notes.note", DOMAIN],
        cursor=first.next_cursor,
        page_size=1,
        include_own_changes=True,
        personal_context_exchange=EXCHANGE,
    )
    assert [item.server_cursor for item in second.envelopes] == [authority_cursor]

    third = service.pull(
        user_id="user-a",
        dataset_id=DATASET_ID,
        device_id="device-b",
        domains=["notes.note", DOMAIN],
        cursor=second.next_cursor,
        page_size=1,
        include_own_changes=True,
        personal_context_exchange=EXCHANGE,
    )
    assert third.envelopes == []


@pytest.mark.parametrize("signed", [False, True])
def test_personal_context_domain_subset_does_not_advance_unrequested_stream(
    tmp_path: Path,
    signed: bool,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    authority_cursor = _insert_authority(
        service,
        record_id="subset-authority",
        sequence=1,
    )
    cursor: str | int = 0
    if signed:
        device = service._require_registered_device("user-a", "device-b")
        cursor = service._encode_pull_token(
            dataset_id=DATASET_ID,
            device_id="device-b",
            version_set=service._pull_version_set(device),
            watermarks={("personal_context.scope", 1): 0},
        )

    excluded = service.pull(
        user_id="user-a",
        dataset_id=DATASET_ID,
        device_id="device-b",
        domains=["personal_context.scope"],
        cursor=cursor,
        include_own_changes=True,
        personal_context_exchange=EXCHANGE,
    )
    assert excluded.envelopes == []

    included = service.pull(
        user_id="user-a",
        dataset_id=DATASET_ID,
        device_id="device-b",
        domains=[DOMAIN],
        cursor=0,
        include_own_changes=True,
        personal_context_exchange=EXCHANGE,
    )
    assert [item.server_cursor for item in included.envelopes] == [authority_cursor]


def test_expired_absolute_recovery_deadline_does_not_advance_cursor(
    tmp_path: Path,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    _insert_authority(service, record_id="deadline-authority", sequence=1)
    ticks = iter((0, 100_000_000))
    service._recovery_clock_ns = lambda: next(ticks, 100_000_000)

    pulled = service.pull(
        user_id="user-a",
        dataset_id=DATASET_ID,
        device_id="device-b",
        domains=[DOMAIN],
        cursor=0,
        include_own_changes=True,
        personal_context_exchange=EXCHANGE,
    )

    assert pulled.envelopes == []
    assert pulled.next_cursor == "0"
    assert pulled.has_more is True


def test_conflict_fails_closed_when_profile_storage_key_is_unavailable(
    tmp_path: Path,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    for domain in PERSONAL_CONTEXT_SYNC_DOMAINS:
        service.adapters.get(domain).encryption_key_resolver = None
    payload = preference_record().model_dump(mode="json")
    envelope = _envelope(payload, client_envelope_id="device-a:record:conflict")
    envelope = replace(envelope, object_id="wrong-record-id")

    result = service.push(
        user_id="user-a",
        dataset_id=DATASET_ID,
        device_id="device-a",
        envelopes=[envelope],
        personal_context_exchange=EXCHANGE,
    )

    assert result.conflicts == []
    assert result.accepted == []
    assert len(result.rejected) == 1
    assert result.rejected[0].error_code == "personal_context_storage_unavailable"


def test_generic_envelope_store_failure_is_not_mislabeled_as_key_unavailability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    payload = preference_record().model_dump(mode="json")
    envelope = _envelope(payload, client_envelope_id="device-a:record:store-failure")

    def fail_insert(_envelope: object) -> object:
        raise SyncStoreError("injected generic store failure")

    monkeypatch.setattr(service.store, "insert_envelope", fail_insert)

    with pytest.raises(SyncStoreError, match="injected generic store failure"):
        service.push(
            user_id="user-a",
            dataset_id=DATASET_ID,
            device_id="device-a",
            envelopes=[envelope],
            personal_context_exchange=EXCHANGE,
        )


@pytest.mark.parametrize(
    "exchange",
    [
        None,
        PersonalContextExchangeProof(
            ongoing_sync_version=1,
            activation_epoch=EXCHANGE.activation_epoch,
            continuity_token="tampered_0123456789abcdef",
        ),
        type(
            "IncompleteProof",
            (),
            {
                "ongoing_sync_version": 1,
                "activation_epoch": EXCHANGE.activation_epoch,
            },
        )(),
    ],
)
def test_personal_context_push_requires_exact_persisted_active_exchange(
    tmp_path: Path,
    exchange: object | None,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    payload = preference_record().model_dump(mode="json")
    envelope = _envelope(payload, client_envelope_id="device-a:activation-required")

    with pytest.raises(SyncStoreError, match="personal_context_activation_required"):
        service.push(
            user_id="user-a",
            dataset_id=DATASET_ID,
            device_id="device-a",
            envelopes=[envelope],
            personal_context_exchange=exchange,
        )

    assert service.store.get_envelope_by_client_id(
        DATASET_ID, envelope.client_envelope_id
    ) is None


def test_version_zero_rejects_ongoing_exchange_even_with_matching_tokens(
    tmp_path: Path,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path, active_exchange=False)
    payload = preference_record().model_dump(mode="json")

    with pytest.raises(SyncStoreError, match="personal_context_activation_required"):
        service.push(
            user_id="user-a",
            dataset_id=DATASET_ID,
            device_id="device-a",
            envelopes=[
                _envelope(payload, client_envelope_id="device-a:version-zero")
            ],
            personal_context_exchange=EXCHANGE,
        )
