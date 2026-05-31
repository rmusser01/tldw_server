from __future__ import annotations

"""Sync v2 domain adapter for workspaces and workspace source refs."""

from dataclasses import dataclass, field

from ..adapters import (
    AdapterAccepted,
    AdapterConflict,
    AdapterRejected,
    SyncAdapterContext,
    SyncAdapterOutcome,
)
from ..models import SyncDataset, SyncDomain, SyncEnvelope, SyncEnvelopeCreate
from ._lineage import delete_update_conflict, prior_envelopes

_SOURCE_REF_KINDS = {
    "source_ref",
    "workspace_source",
    "workspace_source_ref",
    "workspace_source_membership",
}
_SOURCE_REF_OPERATIONS = {"tombstone", "upsert"}


@dataclass(slots=True)
class WorkspacesDomainAdapter:
    """Evaluate workspace envelopes using V1 server-side metadata rules."""

    domain: SyncDomain = "workspaces.workspace"
    supported_adapter_versions: set[int] = field(default_factory=lambda: {1})

    def evaluate_envelope(
        self,
        envelope: SyncEnvelopeCreate,
        *,
        dataset: SyncDataset,
        context: SyncAdapterContext | None = None,
    ) -> SyncAdapterOutcome:
        """Accept set-like source refs and flag manual workspace conflicts."""

        del dataset
        explicit_conflict = _explicit_metadata_conflict(envelope)
        if explicit_conflict is not None:
            return explicit_conflict
        prior = prior_envelopes(envelope, context)
        delete_conflict = delete_update_conflict(
            envelope,
            prior,
            is_delete=_is_delete,
            conflict_factory=_manual_delete_conflict,
        )
        if delete_conflict is not None:
            return delete_conflict
        if envelope.domain == "workspaces.source_ref" or _is_source_ref(envelope):
            if envelope.operation not in _SOURCE_REF_OPERATIONS:
                return AdapterRejected(
                    client_envelope_id=envelope.client_envelope_id,
                    error_code="invalid_workspace_source_ref_operation",
                    message="Workspace source refs only support upsert or tombstone.",
                )
            if not (_metadata_value(envelope, "workspace_id") and _source_id(envelope)):
                return AdapterRejected(
                    client_envelope_id=envelope.client_envelope_id,
                    error_code="missing_workspace_source_ref_metadata",
                    message="Workspace source refs require workspace_id and source_id metadata.",
                )
        return AdapterAccepted(client_envelope_id=envelope.client_envelope_id)


def _explicit_metadata_conflict(envelope: SyncEnvelopeCreate) -> AdapterConflict | None:
    conflict_kind = str(_metadata_value(envelope, "conflict_kind") or "").strip().lower()
    if conflict_kind in {"ordered", "ordered_field", "order"}:
        return AdapterConflict(
            client_envelope_id=envelope.client_envelope_id,
            domain=envelope.domain,
            entity_id=envelope.entity_id,
            conflict_type="ordered_field_conflict",
            message="Ordered workspace field changes require manual resolution.",
        )
    if conflict_kind in {"name", "rename", "title"}:
        return AdapterConflict(
            client_envelope_id=envelope.client_envelope_id,
            domain=envelope.domain,
            entity_id=envelope.entity_id,
            conflict_type="rename_conflict",
            message="Workspace rename changes require manual resolution.",
        )
    return None


def _manual_delete_conflict(envelope: SyncEnvelopeCreate) -> AdapterConflict:
    return AdapterConflict(
        client_envelope_id=envelope.client_envelope_id,
        domain=envelope.domain,
        entity_id=envelope.entity_id,
        conflict_type="delete_update_conflict",
        message="Delete-vs-update workspace changes require manual resolution.",
    )


def _is_delete(envelope: SyncEnvelope | SyncEnvelopeCreate) -> bool:
    return envelope.operation == "delete" or bool(
        envelope.payload_clear.get("deleted")
        or envelope.payload_clear.get("soft_deleted")
        or envelope.payload_clear.get("tombstone")
    )


def _is_source_ref(envelope: SyncEnvelope | SyncEnvelopeCreate) -> bool:
    kind = str(
        _metadata_value(envelope, "entity_kind")
        or _metadata_value(envelope, "entity_type")
        or _metadata_value(envelope, "record_type")
        or ""
    ).strip().lower()
    link_type = str(
        _metadata_value(envelope, "link_type")
        or _metadata_value(envelope, "relation_type")
        or _metadata_value(envelope, "relationship")
        or ""
    ).strip().lower()
    link_tokens = {
        token
        for token in link_type.replace("-", "_").replace(" ", "_").split("_")
        if token
    }
    return kind in _SOURCE_REF_KINDS or "source" in link_tokens


def _source_id(envelope: SyncEnvelope | SyncEnvelopeCreate) -> object | None:
    return _metadata_value(envelope, "source_id") or _metadata_value(envelope, "target_entity_id")


def _metadata_value(envelope: SyncEnvelope | SyncEnvelopeCreate, key: str) -> object | None:
    return envelope.routing_metadata.get(key) or envelope.payload_clear.get(key)


__all__ = ["WorkspacesDomainAdapter"]
