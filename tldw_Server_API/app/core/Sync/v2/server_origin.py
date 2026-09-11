from __future__ import annotations

"""Server-origin Sync v2 capture helpers for normal Notes/Chat API writes."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from uuid import uuid4

from .adapters import AdapterAccepted, AdapterConflict, AdapterDeferred, AdapterRejected
from .errors import SyncStoreError
from .models import (
    CLIENT_PRIVATE_SERVER_FRONTEND_LIMITATION_CODE,
    CLIENT_PRIVATE_SERVER_FRONTEND_LIMITATION_MESSAGE,
    DEFAULT_M1_ENCRYPTION_POLICY,
    SyncDataset,
    SyncDomain,
    SyncEnvelope,
    SyncEnvelopeCreate,
    SyncOperation,
    server_frontend_mutation_enabled_for_policy,
)
from .personal_context_ongoing_contract import PersonalContextAuthorityMetadata
from .service import SyncV2Service
from .store import SyncV2Store

SERVER_ORIGIN_DEVICE_ID = "server-origin"


def insert_personal_context_authority(
    service: SyncV2Service,
    *,
    envelope: SyncEnvelopeCreate,
    authority: PersonalContextAuthorityMetadata,
    sync_store: SyncV2Store | None = None,
) -> SyncEnvelope:
    """Insert one internal-only already-canonical Personal Context egress row."""

    if authority.role != "home_authority":
        raise SyncStoreError("Personal Context authority role is required")
    store = service.store if sync_store is None else sync_store
    stored = store.insert_envelope(
        replace(
            envelope,
            device_id=SERVER_ORIGIN_DEVICE_ID,
            status="accepted",
            apply_status="pending",
            routing_metadata={
                **envelope.routing_metadata,
                "personal_context_authority": authority.model_dump(mode="json"),
            },
        )
    )
    if stored.server_cursor is None:
        raise SyncStoreError("Personal Context authority receipt is unavailable")
    return stored


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


class SyncServerOriginMutationNotSupportedError(SyncStoreError):
    """Raised when a dataset policy cannot support trusted server-origin writes."""

    def __init__(self, dataset: SyncDataset, domain: SyncDomain) -> None:
        super().__init__(CLIENT_PRIVATE_SERVER_FRONTEND_LIMITATION_MESSAGE)
        self.dataset = dataset
        self.domain = domain
        self.error_code = CLIENT_PRIVATE_SERVER_FRONTEND_LIMITATION_CODE


class SyncServerOriginRestoreConflictError(SyncStoreError):
    """Raised when a server-origin restore does not target the current tombstone."""

    def __init__(self, object_id: str) -> None:
        super().__init__("Note restore requires the current deleted note version.")
        self.object_id = object_id
        self.error_code = "sync_server_origin_restore_conflict"


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
    routing_metadata: Mapping[str, object] | None = None,
) -> ServerOriginCaptureResult:
    """Append and materialize one trusted server-origin Sync v2 mutation."""

    dataset = _active_default_personal_dataset(service, user_id)
    if domain not in dataset.domains:
        raise SyncStoreError(f"Sync domain is not enrolled for this dataset: {domain}")
    if not server_frontend_mutation_enabled_for_policy(dataset.encryption_policy):
        raise SyncServerOriginMutationNotSupportedError(dataset, domain)

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
            accepted = _require_capture_applied(service, accepted)
            return ServerOriginCaptureResult(dataset=dataset, envelope=accepted)

    state = service.store.get_object_state(dataset.dataset_id, domain, object_id)
    object_revision = 1 if state is None else state.object_revision + 1
    now = service.clock() or None
    client_envelope_id = (
        stable_server_origin_envelope_id(dataset.dataset_id, domain, stable_key)
        if stable_key
        else f"server-origin-{uuid4().hex}"
    )
    canonical_routing_metadata = dict(routing_metadata or {})
    canonical_routing_metadata.update(
        {
            "source": source,
            "origin": "server",
            "server_device_id": SERVER_ORIGIN_DEVICE_ID,
            "server_owner_user_id": user_id,
        }
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
        routing_metadata=canonical_routing_metadata,
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
    inserted = _require_capture_applied(service, inserted)
    return ServerOriginCaptureResult(dataset=dataset, envelope=inserted)


def _require_capture_applied(
    service: SyncV2Service,
    envelope: SyncEnvelope,
) -> SyncEnvelope:
    """Retry replayable capture debt and return only a durably applied envelope."""

    if envelope.apply_status in {"conflict", "superseded"}:
        raise SyncServerOriginMaterializationError(envelope)
    if envelope.apply_status != "applied":
        service._materialize_envelope(envelope)
        envelope = service._envelope_snapshot(envelope)
    if envelope.apply_status != "applied":
        raise SyncServerOriginMaterializationError(envelope)
    return envelope


def capture_server_origin_note_restore(
    service: SyncV2Service,
    *,
    user_id: str,
    object_id: str,
    note: Mapping[str, object],
    expected_version: int,
    source: str,
) -> ServerOriginCaptureResult:
    """Validate and capture a restore of the current Notes tombstone."""

    current_version = note.get("version")
    try:
        version_matches = int(current_version) == int(expected_version)
    except (TypeError, ValueError):
        version_matches = False
    if not bool(note.get("deleted")) or not version_matches:
        raise SyncServerOriginRestoreConflictError(object_id)

    return capture_server_origin_mutation(
        service,
        user_id=user_id,
        domain="notes.note",
        operation="upsert",
        object_id=object_id,
        payload={
            "title": str(note.get("title") or ""),
            "content": str(note.get("content") or ""),
            "conversation_id": note.get("conversation_id"),
            "message_id": note.get("message_id"),
        },
        source=source,
        routing_metadata={"restore_intent": True},
    )


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
    digest = hashlib.sha256(f"{domain}:{normalized}".encode()).hexdigest()
    return f"{domain.replace('.', '-')}-{digest[:32]}"


def stable_server_origin_envelope_id(
    dataset_id: str,
    domain: SyncDomain,
    stable_key: str | None,
) -> str:
    """Return a deterministic client envelope id for active-Sync API retries."""

    digest = hashlib.sha256(f"{dataset_id}:{domain}:{stable_key}".encode()).hexdigest()
    return f"server-origin-{digest[:32]}"


def get_active_server_origin_sync_service_for_user(user_id: str) -> SyncV2Service | None:
    """Return the active personal service, preserving lookup failures."""

    from .factory import sync_v2_service_for_user, sync_v2_storage_exists_for_user

    if not sync_v2_storage_exists_for_user(user_id):
        return None
    service = sync_v2_service_for_user(user_id)
    for dataset in service.store.list_datasets_for_user(user_id):
        if (
            dataset.scope_type == "personal"
            and dataset.metadata.get("default_personal") is True
            and dataset.metadata.get("client_family") == "chatbook"
        ):
            return service
    return None


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
    "CLIENT_PRIVATE_SERVER_FRONTEND_LIMITATION_CODE",
    "SyncServerOriginMaterializationError",
    "SyncServerOriginMutationNotSupportedError",
    "SyncServerOriginRestoreConflictError",
    "canonical_payload_hash",
    "insert_personal_context_authority",
    "capture_server_origin_note_restore",
    "capture_server_origin_mutation",
    "get_active_server_origin_sync_service_for_user",
]
