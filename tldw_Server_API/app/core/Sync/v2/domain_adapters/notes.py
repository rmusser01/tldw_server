from __future__ import annotations

"""Sync v2 domain adapter for encrypted notes envelopes."""

from dataclasses import dataclass, field
from typing import Any

from ..adapters import AdapterAccepted, AdapterConflict, SyncAdapterContext, SyncAdapterOutcome
from ..models import SyncDataset, SyncDomain, SyncEnvelope, SyncEnvelopeCreate

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

    domain: SyncDomain = "notes"
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
        prior = _prior_envelopes(envelope, context)
        delete_conflict = _delete_update_conflict(envelope, prior)
        if delete_conflict is not None:
            return delete_conflict
        if _is_content_bearing(envelope):
            conflicting = next(
                (
                    item
                    for item in prior
                    if _is_content_bearing(item)
                    and item.operation == "upsert"
                    and item.payload_hash != envelope.payload_hash
                ),
                None,
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


def _prior_envelopes(
    envelope: SyncEnvelopeCreate,
    context: SyncAdapterContext | None,
) -> list[SyncEnvelope]:
    return [
        item
        for item in (context.prior_envelopes if context is not None else [])
        if item.client_envelope_id != envelope.client_envelope_id
    ]


def _delete_update_conflict(
    envelope: SyncEnvelopeCreate,
    prior: list[SyncEnvelope],
) -> AdapterConflict | None:
    incoming_delete = _is_delete(envelope)
    if incoming_delete and any(not _is_delete(item) for item in prior):
        return _manual_delete_conflict(envelope)
    if not incoming_delete and any(_is_delete(item) for item in prior):
        return _manual_delete_conflict(envelope)
    return None


def _manual_delete_conflict(envelope: SyncEnvelopeCreate) -> AdapterConflict:
    return AdapterConflict(
        client_envelope_id=envelope.client_envelope_id,
        domain=envelope.domain,
        entity_id=envelope.entity_id,
        conflict_type="delete_update_conflict",
        message="Delete-vs-update note changes require manual resolution.",
    )


def _is_delete(envelope: SyncEnvelope | SyncEnvelopeCreate) -> bool:
    return envelope.operation == "delete" or bool(
        envelope.payload_clear.get("deleted")
        or envelope.payload_clear.get("soft_deleted")
        or envelope.payload_clear.get("tombstone")
    )


def _is_content_bearing(envelope: SyncEnvelope | SyncEnvelopeCreate) -> bool:
    update_kind = _metadata_string(envelope, "update_kind") or _metadata_string(
        envelope, "change_kind"
    )
    if update_kind and update_kind in _CONTENT_UPDATE_KINDS:
        return True
    content_fields = envelope.routing_metadata.get("content_fields") or envelope.routing_metadata.get(
        "encrypted_fields"
    )
    return bool(_CONTENT_FIELD_NAMES.intersection(_string_set(content_fields)))


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
