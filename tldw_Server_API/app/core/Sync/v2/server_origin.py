from __future__ import annotations

"""Server-origin Sync v2 capture helpers for normal Notes/Chat API writes."""

import hashlib
import json
from dataclasses import dataclass
from uuid import uuid4

from .adapters import AdapterAccepted, AdapterConflict, AdapterDeferred, AdapterRejected
from .errors import SyncStoreError
from .models import (
    DEFAULT_M1_ENCRYPTION_POLICY,
    SyncDataset,
    SyncDomain,
    SyncEnvelope,
    SyncEnvelopeCreate,
    SyncOperation,
)
from .service import SyncV2Service

SERVER_ORIGIN_DEVICE_ID = "server-origin"


class SyncServerOriginMaterializationError(SyncStoreError):
    """Raised when an accepted server-origin envelope is not applied cleanly."""

    def __init__(self, envelope: SyncEnvelope) -> None:
        super().__init__("sync_server_origin_materialization_failed")
        self.envelope = envelope


class SyncServerOriginIdempotencyConflictError(SyncStoreError):
    """Raised when an idempotency key is reused for a different mutation."""

    def __init__(self, envelope: SyncEnvelope) -> None:
        super().__init__("sync_server_origin_idempotency_conflict")
        self.envelope = envelope


@dataclass(frozen=True, slots=True)
class ServerOriginCaptureResult:
    """Result of accepting and materializing a server-origin mutation."""

    dataset: SyncDataset
    envelope: SyncEnvelope


def capture_server_origin_mutation(
    service: SyncV2Service,
    *,
    user_id: str,
    domain: SyncDomain,
    operation: SyncOperation,
    object_id: str,
    payload: dict[str, object],
    source: str,
    parent_id: str | None = None,
    stable_key: str | None = None,
) -> ServerOriginCaptureResult:
    """Append and materialize one trusted server-origin Sync v2 mutation."""

    dataset = _active_default_personal_dataset(service, user_id)
    if domain not in dataset.domains:
        raise SyncStoreError(f"Sync domain is not enrolled for this dataset: {domain}")

    payload_hash, payload_size = canonical_payload_hash(payload)
    if stable_key:
        existing = service.store.list_envelopes_for_entity(
            dataset.dataset_id,
            domain,
            stable_key=stable_key,
            limit=1,
        )
        if existing:
            accepted = existing[0]
            if (
                accepted.operation != operation
                or accepted.object_id != object_id
                or accepted.parent_id != parent_id
                or not _payload_matches_idempotent_replay(accepted, payload, payload_hash)
            ):
                raise SyncServerOriginIdempotencyConflictError(accepted)
            if accepted.apply_status in {"failed", "conflict"}:
                raise SyncServerOriginMaterializationError(accepted)
            return ServerOriginCaptureResult(dataset=dataset, envelope=accepted)

    state = service.store.get_object_state(dataset.dataset_id, domain, object_id)
    object_revision = 1 if state is None else state.object_revision + 1
    now = service.clock() or None
    client_envelope_id = (
        stable_server_origin_envelope_id(dataset.dataset_id, domain, stable_key)
        if stable_key
        else f"server-origin-{uuid4().hex}"
    )
    envelope = SyncEnvelopeCreate(
        dataset_id=dataset.dataset_id,
        client_envelope_id=client_envelope_id,
        domain=domain,
        operation=operation,
        object_id=object_id,
        device_id=SERVER_ORIGIN_DEVICE_ID,
        client_sequence=None,
        base_server_cursor=state.latest_server_cursor if state is not None else None,
        base_object_revision=state.object_revision if state is not None else None,
        base_object_hash=state.object_hash if state is not None else None,
        object_revision=object_revision,
        parent_id=parent_id,
        schema_version=1,
        payload=dict(payload),
        payload_hash=payload_hash,
        payload_size_bytes=payload_size,
        created_at_client=now,
        deleted=operation == "tombstone",
        encryption_metadata={"policy": DEFAULT_M1_ENCRYPTION_POLICY},
        routing_metadata={
            "source": source,
            "origin": "server",
            "server_device_id": SERVER_ORIGIN_DEVICE_ID,
        },
        stable_key=stable_key,
    )

    outcome = service._evaluate_envelope(dataset, envelope)
    if isinstance(outcome, AdapterRejected):
        raise SyncStoreError(outcome.message)
    if isinstance(outcome, AdapterDeferred):
        raise SyncStoreError(outcome.message)
    if isinstance(outcome, AdapterConflict):
        raise SyncStoreError(outcome.message or "Sync server-origin mutation conflicted")
    if not isinstance(outcome, AdapterAccepted):
        raise SyncStoreError("Sync server-origin mutation was not accepted")

    inserted = service.store.insert_envelope(envelope)
    materialization = service._materialize_envelope(inserted)
    inserted = service._envelope_snapshot(inserted)
    if materialization.status in {"failed", "conflict"}:
        raise SyncServerOriginMaterializationError(inserted)
    return ServerOriginCaptureResult(dataset=dataset, envelope=inserted)


def server_origin_stable_key(
    *,
    source: str,
    domain: SyncDomain,
    operation: SyncOperation,
    idempotency_key: str | None,
) -> str | None:
    """Return a privacy-preserving stable key for API idempotency."""

    if idempotency_key is None:
        return None
    normalized = idempotency_key.strip()
    if not normalized:
        return None
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{source}:{domain}:{operation}:{digest}"


def server_origin_object_id(domain: SyncDomain, idempotency_key: str | None) -> str | None:
    """Return a deterministic object id for active-Sync create retries."""

    if idempotency_key is None:
        return None
    normalized = idempotency_key.strip()
    if not normalized:
        return None
    digest = hashlib.sha256(f"{domain}:{normalized}".encode("utf-8")).hexdigest()
    return f"{domain.replace('.', '-')}-{digest[:32]}"


def stable_server_origin_envelope_id(
    dataset_id: str,
    domain: SyncDomain,
    stable_key: str | None,
) -> str:
    """Return a deterministic client envelope id for active-Sync API retries."""

    digest = hashlib.sha256(f"{dataset_id}:{domain}:{stable_key}".encode("utf-8")).hexdigest()
    return f"server-origin-{digest[:32]}"


def get_active_server_origin_sync_service_for_user(user_id: str) -> SyncV2Service | None:
    """Return a Sync v2 service only when the user has an active personal profile."""

    from .factory import sync_v2_service_for_user, sync_v2_storage_exists_for_user

    if not sync_v2_storage_exists_for_user(user_id):
        return None
    service = sync_v2_service_for_user(user_id)
    try:
        _active_default_personal_dataset(service, user_id)
    except SyncStoreError:
        return None
    return service


def canonical_payload_hash(payload: dict[str, object]) -> tuple[str, int]:
    """Return the canonical server-trusted payload hash and encoded size."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}", len(encoded)


def _payload_matches_idempotent_replay(
    accepted: SyncEnvelope,
    payload: dict[str, object],
    payload_hash: str,
) -> bool:
    if accepted.payload_hash == payload_hash:
        return True
    if accepted.domain != "chat.message" or accepted.operation != "append":
        return False

    accepted_payload = dict(accepted.payload)
    replay_payload = dict(payload)
    accepted_payload.pop("timestamp", None)
    replay_payload.pop("timestamp", None)
    accepted_without_timestamp, _ = canonical_payload_hash(accepted_payload)
    replay_without_timestamp, _ = canonical_payload_hash(replay_payload)
    return accepted_without_timestamp == replay_without_timestamp


def _active_default_personal_dataset(service: SyncV2Service, user_id: str) -> SyncDataset:
    for dataset in service.store.list_datasets_for_user(user_id):
        if (
            dataset.scope_type == "personal"
            and dataset.metadata.get("default_personal") is True
            and dataset.metadata.get("client_family") == "chatbook"
        ):
            return dataset
    raise SyncStoreError("Sync default personal dataset was not found or is not accessible")


__all__ = [
    "SERVER_ORIGIN_DEVICE_ID",
    "ServerOriginCaptureResult",
    "SyncServerOriginMaterializationError",
    "canonical_payload_hash",
    "capture_server_origin_mutation",
    "get_active_server_origin_sync_service_for_user",
]
