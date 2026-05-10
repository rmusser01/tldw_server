from __future__ import annotations

"""Sync v2 domain adapter for source cache entries."""

from dataclasses import dataclass, field

from ..adapters import (
    AdapterAccepted,
    AdapterConflict,
    AdapterRejected,
    SyncAdapterContext,
    SyncAdapterOutcome,
)
from ..models import SyncDataset, SyncDomain, SyncEnvelope, SyncEnvelopeCreate


@dataclass(slots=True)
class SourceCacheAdapter:
    """Evaluate source-cache envelopes using source ID plus content hash."""

    domain: SyncDomain = "source_cache"
    supported_adapter_versions: set[int] = field(default_factory=lambda: {1})

    def evaluate_envelope(
        self,
        envelope: SyncEnvelopeCreate,
        *,
        dataset: SyncDataset,
        context: SyncAdapterContext | None = None,
    ) -> SyncAdapterOutcome:
        """Allow cache versions to coexist unless the same version diverges."""

        del dataset
        if not (_source_id(envelope) and _content_hash(envelope)):
            return AdapterRejected(
                client_envelope_id=envelope.client_envelope_id,
                error_code="missing_source_cache_identity",
                message="Source cache envelopes require source_id and content_hash metadata.",
            )
        prior = [
            item
            for item in (context.prior_envelopes if context is not None else [])
            if item.client_envelope_id != envelope.client_envelope_id
        ]
        delete_conflict = _delete_update_conflict(envelope, prior)
        if delete_conflict is not None:
            return delete_conflict
        conflicting = next(
            (
                item
                for item in prior
                if _source_id(item) == _source_id(envelope)
                and _content_hash(item) == _content_hash(envelope)
                and item.payload_hash != envelope.payload_hash
            ),
            None,
        )
        if conflicting is not None:
            return AdapterConflict(
                client_envelope_id=envelope.client_envelope_id,
                domain=self.domain,
                entity_id=envelope.entity_id,
                conflict_type="source_cache_hash_mismatch",
                message="Source cache source ID and content hash matched but payload hash differed.",
                metadata={"conflicting_envelope_id": conflicting.client_envelope_id},
            )
        return AdapterAccepted(client_envelope_id=envelope.client_envelope_id)


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
        message="Delete-vs-update source cache changes require manual resolution.",
    )


def _is_delete(envelope: SyncEnvelope | SyncEnvelopeCreate) -> bool:
    return envelope.operation == "delete" or bool(
        envelope.payload_clear.get("deleted")
        or envelope.payload_clear.get("soft_deleted")
        or envelope.payload_clear.get("tombstone")
    )


def _source_id(envelope: SyncEnvelope | SyncEnvelopeCreate) -> object | None:
    return envelope.routing_metadata.get("source_id") or envelope.payload_clear.get("source_id")


def _content_hash(envelope: SyncEnvelope | SyncEnvelopeCreate) -> object | None:
    return (
        envelope.routing_metadata.get("content_hash")
        or envelope.payload_clear.get("content_hash")
        or envelope.payload_clear.get("payload_hash")
    )


__all__ = ["SourceCacheAdapter"]
