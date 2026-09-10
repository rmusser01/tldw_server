from __future__ import annotations

"""Sync v2 domain adapter for encrypted notes envelopes."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..adapters import (
    AdapterAccepted,
    AdapterConflict,
    AdapterRejected,
    SyncAdapterContext,
    SyncAdapterOutcome,
)
from ..models import (
    SyncDataset,
    SyncDomain,
    SyncEnvelope,
    SyncEnvelopeCreate,
    validate_notes_note_upsert_payload,
)
from ._lineage import (
    current_head,
    delete_update_conflict,
    incoming_references_head,
    prior_envelopes,
)

_CONTENT_UPDATE_KINDS = {
    "body",
    "content",
    "encrypted_content",
    "note_body",
    "note_content",
    "title",
    "title_body",
}
_CONTENT_FIELD_NAMES = {"body", "content", "title"}


@dataclass(slots=True)
class NotesDomainAdapter:
    """Evaluate notes envelopes using private-content conflict rules."""

    domain: SyncDomain = "notes.note"
    supported_adapter_versions: set[int] = field(default_factory=lambda: {1})

    def evaluate_envelope(
        self,
        envelope: SyncEnvelopeCreate,
        *,
        dataset: SyncDataset,
        context: SyncAdapterContext | None = None,
    ) -> SyncAdapterOutcome:
        """Accept metadata-only changes and conflict concurrent encrypted edits."""

        del dataset
        if envelope.domain == "notes.note" and envelope.operation == "upsert":
            try:
                validate_notes_note_upsert_payload(envelope.payload)
            except ValueError as exc:
                return AdapterRejected(
                    client_envelope_id=envelope.client_envelope_id,
                    error_code="notes_note_payload_invalid",
                    message=str(exc),
                )
        restore_intent = (
            envelope.routing_metadata.get("restore_intent")
            if envelope.domain == "notes.note"
            else None
        )
        if restore_intent is not None and (
            restore_intent is not True or envelope.operation != "upsert"
        ):
            return AdapterRejected(
                client_envelope_id=envelope.client_envelope_id,
                error_code="notes_note_restore_intent_invalid",
                message=(
                    "notes.note restore_intent must be the boolean true on an upsert "
                    "when supplied"
                ),
            )
        prior = prior_envelopes(envelope, context)
        delete_conflict = delete_update_conflict(
            envelope,
            prior,
            is_delete=_is_delete,
            conflict_factory=_manual_delete_conflict,
        )
        if delete_conflict is not None:
            return delete_conflict
        if restore_intent is True:
            head = current_head(prior)
            if (
                head is None
                or not _is_delete(head)
                or not incoming_references_head(envelope, head)
            ):
                return _restore_target_conflict(envelope)
            return AdapterAccepted(client_envelope_id=envelope.client_envelope_id)
        if _is_content_bearing(envelope) and not incoming_references_head(
            envelope, _current_content_head(prior)
        ):
            conflicting = _current_content_head(
                item
                for item in prior
                if item.operation == "upsert" and item.payload_hash != envelope.payload_hash
            )
            if conflicting is not None:
                return AdapterConflict(
                    client_envelope_id=envelope.client_envelope_id,
                    domain=self.domain,
                    entity_id=envelope.entity_id,
                    conflict_type="encrypted_content_edit",
                    message="Concurrent encrypted note title/body edits require manual resolution.",
                    metadata={"conflicting_envelope_id": conflicting.client_envelope_id},
                )
        return AdapterAccepted(client_envelope_id=envelope.client_envelope_id)


def _manual_delete_conflict(envelope: SyncEnvelopeCreate) -> AdapterConflict:
    return AdapterConflict(
        client_envelope_id=envelope.client_envelope_id,
        domain=envelope.domain,
        entity_id=envelope.entity_id,
        conflict_type="delete_update_conflict",
        message="Delete-vs-update note changes require manual resolution.",
    )


def _restore_target_conflict(envelope: SyncEnvelopeCreate) -> AdapterConflict:
    return AdapterConflict(
        client_envelope_id=envelope.client_envelope_id,
        domain=envelope.domain,
        entity_id=envelope.entity_id,
        conflict_type="restore_target_conflict",
        message="Note restore requires the exact current tombstone head.",
    )


def _is_delete(envelope: SyncEnvelope | SyncEnvelopeCreate) -> bool:
    return envelope.operation == "tombstone" or bool(
        envelope.payload_clear.get("deleted")
        or envelope.payload_clear.get("soft_deleted")
        or envelope.payload_clear.get("tombstone")
    )


def _is_content_bearing(envelope: SyncEnvelope | SyncEnvelopeCreate) -> bool:
    if envelope.domain == "notes.note" and envelope.operation == "upsert":
        return True
    update_kind = _metadata_string(envelope, "update_kind") or _metadata_string(
        envelope, "change_kind"
    )
    if update_kind and update_kind in _CONTENT_UPDATE_KINDS:
        return True
    content_fields = envelope.routing_metadata.get("content_fields") or envelope.routing_metadata.get(
        "encrypted_fields"
    )
    return bool(_CONTENT_FIELD_NAMES.intersection(_string_set(content_fields)))


def _current_content_head(prior: Iterable[SyncEnvelope]) -> SyncEnvelope | None:
    return current_head(item for item in prior if _is_content_bearing(item))


def _metadata_string(envelope: SyncEnvelope | SyncEnvelopeCreate, key: str) -> str | None:
    value = envelope.routing_metadata.get(key) or envelope.payload_clear.get(key)
    if value is None:
        return None
    return str(value).strip().lower()


def _string_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value.strip().lower()}
    if isinstance(value, list | tuple | set):
        return {str(item).strip().lower() for item in value}
    return {str(value).strip().lower()}


__all__ = ["NotesDomainAdapter"]
