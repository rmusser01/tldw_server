from __future__ import annotations

"""Sync v2 domain adapter for chat threads and messages."""

from dataclasses import dataclass, field

from ..adapters import AdapterAccepted, AdapterConflict, SyncAdapterContext, SyncAdapterOutcome
from ..models import SyncDataset, SyncDomain, SyncEnvelope, SyncEnvelopeCreate
from ._lineage import delete_update_conflict, prior_envelopes

_MESSAGE_KINDS = {"chat_message", "message"}


@dataclass(slots=True)
class ChatDomainAdapter:
    """Evaluate chat envelopes using append-only message conflict rules."""

    domain: SyncDomain = "chat"
    supported_adapter_versions: set[int] = field(default_factory=lambda: {1})

    def evaluate_envelope(
        self,
        envelope: SyncEnvelopeCreate,
        *,
        dataset: SyncDataset,
        context: SyncAdapterContext | None = None,
    ) -> SyncAdapterOutcome:
        """Accept independent messages and conflict divergent stable message IDs."""

        del dataset
        prior = prior_envelopes(envelope, context)
        delete_conflict = delete_update_conflict(
            envelope,
            prior,
            is_delete=_is_delete,
            conflict_factory=_manual_delete_conflict,
        )
        if delete_conflict is not None:
            return delete_conflict
        if _is_message(envelope):
            incoming_id = _message_key(envelope)
            conflicting = next(
                (
                    item
                    for item in prior
                    if _is_message(item)
                    and _message_key(item) == incoming_id
                    and item.payload_hash != envelope.payload_hash
                ),
                None,
            )
            if conflicting is not None:
                return AdapterConflict(
                    client_envelope_id=envelope.client_envelope_id,
                    domain=self.domain,
                    entity_id=envelope.entity_id,
                    conflict_type="message_hash_mismatch",
                    message="Chat message stable ID was reused with different content hash.",
                    metadata={"conflicting_envelope_id": conflicting.client_envelope_id},
                )
        return AdapterAccepted(client_envelope_id=envelope.client_envelope_id)


def _manual_delete_conflict(envelope: SyncEnvelopeCreate) -> AdapterConflict:
    return AdapterConflict(
        client_envelope_id=envelope.client_envelope_id,
        domain=envelope.domain,
        entity_id=envelope.entity_id,
        conflict_type="delete_update_conflict",
        message="Delete-vs-update chat changes require manual resolution.",
    )


def _is_delete(envelope: SyncEnvelope | SyncEnvelopeCreate) -> bool:
    return envelope.operation == "delete" or bool(
        envelope.payload_clear.get("deleted")
        or envelope.payload_clear.get("soft_deleted")
        or envelope.payload_clear.get("tombstone")
    )


def _is_message(envelope: SyncEnvelope | SyncEnvelopeCreate) -> bool:
    kind = _entity_kind(envelope)
    return kind in _MESSAGE_KINDS or bool(_metadata_value(envelope, "message_id"))


def _message_key(envelope: SyncEnvelope | SyncEnvelopeCreate) -> str:
    value = _metadata_value(envelope, "message_id")
    if value is not None:
        return str(value)
    return envelope.stable_key or envelope.entity_id


def _entity_kind(envelope: SyncEnvelope | SyncEnvelopeCreate) -> str:
    value = (
        _metadata_value(envelope, "entity_kind")
        or _metadata_value(envelope, "entity_type")
        or _metadata_value(envelope, "record_type")
        or ""
    )
    return str(value).strip().lower()


def _metadata_value(envelope: SyncEnvelope | SyncEnvelopeCreate, key: str) -> object | None:
    return envelope.routing_metadata.get(key) or envelope.payload_clear.get(key)


__all__ = ["ChatDomainAdapter"]
