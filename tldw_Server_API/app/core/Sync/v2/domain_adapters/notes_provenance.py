"""Independent exact-head Knowledge history under an owned canonical note."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..adapters import AdapterAccepted, AdapterConflict, AdapterRejected, SyncAdapterContext, SyncAdapterOutcome
from ..models import SyncDataset, SyncEnvelopeCreate
from ..notes_provenance_contract import notes_provenance_object_hash, validate_notes_provenance_payload


@dataclass(slots=True)
class NotesProvenanceDomainAdapter:
    """Validate provenance bytes, independent lineage, and active-parent ownership."""

    domain: str = "notes.provenance"
    supported_adapter_versions: set[int] = field(default_factory=lambda: {1})

    def evaluate_envelope(
        self, envelope: SyncEnvelopeCreate, *, dataset: SyncDataset, context: SyncAdapterContext | None = None
    ) -> SyncAdapterOutcome:
        def reject(code: str) -> AdapterRejected:
            return AdapterRejected(envelope.client_envelope_id, code, code)

        if (
            dataset.encryption_policy != "server_trusted_v1"
            or envelope.encryption_metadata.get("policy", "server_trusted_v1") != "server_trusted_v1"
        ):
            return reject("notes_provenance_encryption_unsupported")
        if type(envelope.object_revision) is not int or not 1 <= envelope.object_revision <= 9_007_199_254_740_991:
            return reject("notes_provenance_revision_invalid")
        try:
            validate_notes_provenance_payload(envelope.payload)
            if envelope.payload_hash != notes_provenance_object_hash(
                envelope.payload, deleted=envelope.operation == "tombstone"
            ):
                raise ValueError("hash mismatch")
            if (
                envelope.parent_id != envelope.object_id
                or envelope.schema_version != 1
                or envelope.operation not in {"upsert", "tombstone"}
            ):
                raise ValueError("invalid provenance identity")
        except (TypeError, ValueError):
            return reject("notes_provenance_payload_invalid")
        readiness = dataset.metadata.get("notes_provenance_v1", {})
        bootstrap = bool(context and context.trusted_server_origin and readiness.get("state") == "initializing")
        if readiness.get("state") != "ready" and not bootstrap:
            return reject("notes_provenance_sync_not_ready")
        if context is None or context.get_head is None:
            return reject("notes_provenance_parent_missing")
        parent = context.get_head("notes.note", envelope.object_id)
        if parent is None or (parent.operation == "tombstone" and envelope.operation != "tombstone"):
            return reject("notes_provenance_parent_missing")
        head = context.get_head("notes.provenance", envelope.object_id)
        restore = envelope.routing_metadata.get("restore_intent")
        if restore is not None and (restore is not True or envelope.operation != "upsert"):
            return reject("notes_provenance_restore_intent_invalid")
        base = (envelope.base_server_cursor, envelope.base_object_revision, envelope.base_object_hash)
        conflict = (
            (
                head is None
                and (
                    any(value is not None for value in base)
                    or restore is True
                    or (envelope.operation == "tombstone" and not bootstrap)
                )
            )
            or (
                head is not None
                and base
                != (
                    head.server_cursor if head.server_cursor is not None else 0,
                    head.object_revision,
                    head.payload_hash,
                )
            )
            or (
                head is not None
                and envelope.operation == "upsert"
                and (head.operation == "tombstone") != (restore is True)
            )
            or (
                head is not None
                and (restore is True or envelope.operation == "tombstone")
                and envelope.payload != head.payload
            )
        )
        if conflict:
            return AdapterConflict(
                envelope.client_envelope_id,
                envelope.domain,
                envelope.object_id,
                "whole_object_conflict",
                "Provenance requires its exact independent retained head",
            )
        if not bootstrap and envelope.object_revision != (1 if head is None else (head.object_revision or 0) + 1):
            return reject("notes_provenance_revision_invalid")
        return AdapterAccepted(envelope.client_envelope_id)
