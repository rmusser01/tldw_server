"""Strict Sync v2 adapter for durable explicit Notes links."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..adapters import (
    AdapterAccepted,
    AdapterConflict,
    AdapterDeferred,
    AdapterRejected,
    SyncAdapterContext,
    SyncAdapterOutcome,
    SyncHead,
)
from ..models import SyncDataset, SyncEnvelopeCreate
from ..notes_link import (
    NotesLinkValidationError,
    parse_notes_link_payload,
    validate_notes_link_object_id,
    validate_notes_link_provenance,
)
from ._lineage import current_head, prior_envelopes

_IMMUTABLE_FIELDS = (
    "source_note_id",
    "target_note_id",
    "type",
    "directed",
    "created_at",
    "created_by",
)


@dataclass(slots=True)
class NotesLinkDomainAdapter:
    """Validate one canonical ``notes.link`` envelope without product writes."""

    domain: str = "notes.link"
    supported_adapter_versions: set[int] = field(default_factory=lambda: {1})

    def evaluate_envelope(
        self,
        envelope: SyncEnvelopeCreate,
        *,
        dataset: SyncDataset,
        context: SyncAdapterContext | None = None,
    ) -> SyncAdapterOutcome:
        """Return a stable validation, dependency, conflict, or acceptance result."""

        if envelope.domain != self.domain or envelope.schema_version != 1:
            return _rejected(
                envelope,
                "notes_link_payload_invalid",
                "notes.link envelope domain or schema version is invalid",
            )
        trusted_bootstrap = _trusted_bootstrap(dataset, envelope, context)
        readiness = _readiness_error(dataset, trusted_bootstrap=trusted_bootstrap)
        if readiness is not None:
            return _rejected(envelope, "notes_link_domain_not_ready", readiness)

        try:
            payload = parse_notes_link_payload(envelope.operation, envelope.payload)
            validate_notes_link_object_id(envelope.object_id)
        except NotesLinkValidationError as exc:
            return _rejected(envelope, exc.error_code, "notes.link payload validation failed")

        head = _get_head(envelope, context)
        if head is not None and _literal_replay(head, envelope):
            return AdapterAccepted(client_envelope_id=envelope.client_envelope_id)
        if head is None and _has_base(envelope):
            return _base_conflict(envelope, "The referenced notes.link base does not exist")
        if head is not None and not _exact_base(envelope, head):
            return _base_conflict(envelope, "The notes.link base does not match the current head")

        restore_intent = envelope.routing_metadata.get("restore_intent")
        if restore_intent not in {None, True} or (restore_intent is True and envelope.operation != "upsert"):
            return _rejected(
                envelope,
                "notes_link_payload_invalid",
                "restore_intent must be the boolean true on an upsert",
            )
        if head is not None and envelope.operation == "upsert" and _is_deleted(head):
            if restore_intent is not True:
                return _base_conflict(envelope, "Restore requires explicit restore intent")
        if restore_intent is True and (head is None or not _is_deleted(head) or not _exact_base(envelope, head)):
            return _base_conflict(envelope, "Restore requires the exact current tombstone head")

        if head is not None and any(
            payload.get(field_name) != head.payload.get(field_name) for field_name in _IMMUTABLE_FIELDS
        ):
            return _conflict(
                envelope,
                "notes_link_identity_conflict",
                "notes.link identity and creation provenance are immutable",
            )

        try:
            validate_notes_link_provenance(
                payload,
                envelope_created_at_client=envelope.created_at_client,
                authenticated_device_id=envelope.device_id,
                prior_payload=head.payload if head is not None else None,
                trusted_bootstrap=trusted_bootstrap,
            )
        except NotesLinkValidationError:
            return _rejected(
                envelope,
                "notes_link_payload_invalid",
                "notes.link provenance validation failed",
            )

        dependency = _validate_note_dependencies(envelope, payload, dataset, context)
        if dependency is not None:
            return dependency

        duplicate = _logical_duplicate(envelope, payload, context)
        if duplicate is not None:
            return duplicate
        return AdapterAccepted(client_envelope_id=envelope.client_envelope_id)


def _readiness_error(dataset: SyncDataset, *, trusted_bootstrap: bool) -> str | None:
    if "notes.note" not in dataset.domains or "notes.link" not in dataset.domains:
        return "notes.note and notes.link must both be enrolled"
    metadata = dataset.metadata.get("notes_link_v1")
    if isinstance(metadata, dict) and (
        metadata.get("state") == "ready" or (metadata.get("state") == "initializing" and trusted_bootstrap)
    ):
        return None
    return "The notes.link domain is not ready"


def _trusted_bootstrap(
    dataset: SyncDataset,
    envelope: SyncEnvelopeCreate,
    context: SyncAdapterContext | None,
) -> bool:
    metadata = dataset.metadata.get("notes_link_v1")
    bootstrap_id = metadata.get("bootstrap_id") if isinstance(metadata, dict) else None
    return bool(
        context is not None
        and context.trusted_server_origin
        and isinstance(bootstrap_id, str)
        and context.notes_link_bootstrap_id == bootstrap_id
        and envelope.routing_metadata.get("bootstrap_capture") is True
        and envelope.routing_metadata.get("bootstrap_id") == bootstrap_id
    )


def _get_head(
    envelope: SyncEnvelopeCreate,
    context: SyncAdapterContext | None,
) -> SyncHead | None:
    if context is not None and context.get_head is not None:
        return context.get_head(envelope.domain, envelope.object_id)
    matching = [
        item
        for item in prior_envelopes(envelope, context)
        if item.domain == envelope.domain and item.object_id == envelope.object_id
    ]
    return current_head(matching)


def _domain_heads(context: SyncAdapterContext | None) -> tuple[SyncHead, ...]:
    candidates: list[SyncHead] = []
    if context is not None:
        if context.list_heads is not None:
            candidates.extend(context.list_heads("notes.link"))
        candidates.extend(item for item in context.prior_envelopes if item.domain == "notes.link")
    by_id: dict[str, list[SyncHead]] = {}
    for item in candidates:
        by_id.setdefault(item.object_id, []).append(item)
    return tuple(head for items in by_id.values() if (head := current_head(items)) is not None)


def _validate_note_dependencies(
    envelope: SyncEnvelopeCreate,
    payload: dict[str, object],
    dataset: SyncDataset,
    context: SyncAdapterContext | None,
) -> AdapterDeferred | AdapterConflict | None:
    for note_id in (str(payload["source_note_id"]), str(payload["target_note_id"])):
        head = context.get_head("notes.note", note_id) if context and context.get_head else None
        if head is None:
            return AdapterDeferred(
                client_envelope_id=envelope.client_envelope_id,
                message="A required notes.note endpoint is not available yet",
            )
        if head.dataset_id != dataset.dataset_id:
            return _conflict(
                envelope,
                "notes_link_ownership_conflict",
                "notes.link endpoints must belong to the same dataset",
            )
    return None


def _logical_duplicate(
    envelope: SyncEnvelopeCreate,
    payload: dict[str, object],
    context: SyncAdapterContext | None,
) -> AdapterConflict | None:
    identity = tuple(payload[field_name] for field_name in _IMMUTABLE_FIELDS[:4])
    for head in _domain_heads(context):
        if head.object_id == envelope.object_id:
            continue
        candidate = tuple(head.payload.get(field_name) for field_name in _IMMUTABLE_FIELDS[:4])
        if candidate == identity:
            return _conflict(
                envelope,
                "notes_link_logical_identity_conflict",
                "The notes.link logical identity already exists",
            )
    return None


def _has_base(envelope: SyncEnvelopeCreate) -> bool:
    return any(
        value is not None
        for value in (
            envelope.base_server_cursor,
            envelope.base_object_revision,
            envelope.base_object_hash,
            envelope.base_version,
        )
    )


def _exact_base(envelope: SyncEnvelopeCreate, head: SyncHead) -> bool:
    head_version = head.entity_version if head.entity_version is not None else head.object_revision
    cursor_matches = (
        envelope.base_server_cursor in {None, 0}
        if head.server_cursor is None
        else envelope.base_server_cursor == head.server_cursor
    )
    return bool(
        cursor_matches
        and envelope.base_object_hash == head.payload_hash
        and envelope.base_object_revision == head.object_revision
        and str(envelope.base_version) == str(head_version)
    )


def _literal_replay(head: SyncHead, envelope: SyncEnvelopeCreate) -> bool:
    fields = (
        "dataset_id",
        "client_envelope_id",
        "domain",
        "operation",
        "object_id",
        "device_id",
        "base_server_cursor",
        "base_object_revision",
        "base_object_hash",
        "base_version",
        "object_revision",
        "entity_version",
        "schema_version",
        "payload",
        "payload_hash",
        "created_at_client",
        "routing_metadata",
    )
    return all(getattr(head, field_name) == getattr(envelope, field_name) for field_name in fields)


def _is_deleted(head: SyncHead) -> bool:
    return head.operation == "tombstone" or head.deleted


def _rejected(
    envelope: SyncEnvelopeCreate,
    error_code: str,
    message: str,
) -> AdapterRejected:
    return AdapterRejected(
        client_envelope_id=envelope.client_envelope_id,
        error_code=error_code,
        message=message,
    )


def _conflict(
    envelope: SyncEnvelopeCreate,
    conflict_type: str,
    message: str,
) -> AdapterConflict:
    return AdapterConflict(
        client_envelope_id=envelope.client_envelope_id,
        domain=envelope.domain,
        entity_id=envelope.object_id,
        conflict_type=conflict_type,
        message=message,
    )


def _base_conflict(envelope: SyncEnvelopeCreate, message: str) -> AdapterConflict:
    return _conflict(envelope, "notes_link_base_conflict", message)


__all__ = ["NotesLinkDomainAdapter"]
