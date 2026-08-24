"""Strict dormant Sync v1 adapter for canonical Notes tasks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from tldw_Server_API.app.core.exceptions import NotesTaskContractError

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
from ..notes_task_contract import (
    notes_task_object_hash,
    parse_notes_task_tombstone_v1,
    parse_notes_task_v1,
)
from ._lineage import (
    current_head,
    incoming_references_exact_head,
    prior_envelopes,
)

_REPLAY_FIELDS = (
    "dataset_id",
    "client_envelope_id",
    "domain",
    "operation",
    "object_id",
    "parent_id",
    "device_id",
    "base_server_cursor",
    "base_object_revision",
    "base_object_hash",
    "object_revision",
    "entity_version",
    "adapter_version",
    "schema_version",
    "payload",
    "payload_hash",
    "created_at_client",
    "routing_metadata",
)


@dataclass(slots=True)
class NotesTaskDomainAdapter:
    """Validate exact task lineage without activating a public Sync domain."""

    domain: str = "notes.task"
    supported_adapter_versions: set[int] = field(default_factory=lambda: {1})

    def evaluate_envelope(
        self,
        envelope: SyncEnvelopeCreate,
        *,
        dataset: SyncDataset,
        context: SyncAdapterContext | None = None,
    ) -> SyncAdapterOutcome:
        """Return a stable validation, dependency, conflict, or acceptance result."""

        if (
            envelope.domain != self.domain
            or envelope.operation not in {"upsert", "tombstone"}
            or envelope.adapter_version != 1
            or envelope.schema_version != 1
        ):
            return _rejected(envelope, "notes_task_payload_invalid")

        trusted_bootstrap = _trusted_task_bootstrap(envelope, context)
        trusted_server_mutation = _trusted_server_task_mutation(
            envelope,
            dataset=dataset,
            context=context,
        )
        restore_intent = envelope.routing_metadata.get("restore_intent")
        allowed_routing = (
            {
                "bootstrap_capture",
                "bootstrap_id",
                "source",
                "origin",
                "server_device_id",
                "server_owner_user_id",
            }
            if trusted_bootstrap
            else (
                {
                    "source",
                    "origin",
                    "server_device_id",
                    "server_owner_user_id",
                    "restore_intent",
                    "task_projection",
                }
                if trusted_server_mutation
                else {"restore_intent"}
            )
        )
        if set(envelope.routing_metadata) - allowed_routing or restore_intent not in {
            None,
            True,
        }:
            return _rejected(envelope, "notes_task_payload_invalid")
        if restore_intent is True and envelope.operation != "upsert":
            return _rejected(envelope, "notes_task_payload_invalid")

        try:
            parser = (
                parse_notes_task_tombstone_v1
                if envelope.operation == "tombstone"
                else parse_notes_task_v1
            )
            payload = parser(
                envelope.payload,
                owner_user_id=dataset.owner_user_id,
            )
            expected_revision = envelope.object_revision
            if (
                expected_revision is None
                or envelope.entity_version != expected_revision
                or envelope.payload_hash
                != notes_task_object_hash(
                    payload,
                    revision=expected_revision,
                    deleted=envelope.operation == "tombstone",
                )
            ):
                raise NotesTaskContractError("notes.task lineage is invalid")
        except NotesTaskContractError:
            return _rejected(envelope, "notes_task_payload_invalid")

        if envelope.object_id != payload.task_id or envelope.parent_id != payload.note_id:
            return _conflict(envelope, "notes_task_identity_conflict")

        get_authorized_note = context.get_authorized_note if context is not None else None
        if not trusted_bootstrap and get_authorized_note is None:
            return _rejected(envelope, "notes_task_authorization_unavailable")
        if not trusted_bootstrap:
            if get_authorized_note is None:
                return _rejected(envelope, "notes_task_authorization_unavailable")
            note = get_authorized_note(payload.note_id)
            if note is None or note.dataset_id != dataset.dataset_id:
                return AdapterDeferred(
                    client_envelope_id=envelope.client_envelope_id,
                    message="The authorized parent note is not available yet",
                )
            if _is_deleted(note):
                return _conflict(envelope, "notes_task_parent_conflict")

        head = _get_head(envelope, context)
        if head is not None and _literal_replay(head, envelope):
            return AdapterAccepted(client_envelope_id=envelope.client_envelope_id)

        if head is None:
            if trusted_bootstrap:
                if _has_base(envelope) or restore_intent is True:
                    return _conflict(envelope, "notes_task_base_conflict")
                return AdapterAccepted(client_envelope_id=envelope.client_envelope_id)
            if _has_base(envelope) or expected_revision != 1:
                return _conflict(envelope, "notes_task_base_conflict")
            if restore_intent is True:
                return _conflict(envelope, "notes_task_restore_conflict")
            return AdapterAccepted(client_envelope_id=envelope.client_envelope_id)

        if not _has_base(envelope):
            return _conflict(envelope, "notes_task_create_conflict")
        if not _same_identity(head, payload.task_id, payload.note_id):
            return _conflict(envelope, "notes_task_identity_conflict")
        if not incoming_references_exact_head(envelope, head):
            return _conflict(envelope, "notes_task_base_conflict")
        if expected_revision != int(head.object_revision or 0) + 1:
            return _rejected(envelope, "notes_task_payload_invalid")

        head_deleted = _is_deleted(head)
        if restore_intent is True:
            if not head_deleted:
                return _conflict(envelope, "notes_task_restore_conflict")
        elif envelope.operation == "upsert" and head_deleted:
            return _conflict(envelope, "notes_task_restore_conflict")
        elif envelope.operation == "tombstone" and head_deleted:
            return _conflict(envelope, "notes_task_base_conflict")

        return AdapterAccepted(client_envelope_id=envelope.client_envelope_id)


def _get_head(
    envelope: SyncEnvelopeCreate,
    context: SyncAdapterContext,
) -> SyncHead | None:
    """Return the current object head from the context's preferred lookup."""

    if context.get_head is not None:
        return context.get_head(envelope.domain, envelope.object_id)
    matches = (
        item
        for item in prior_envelopes(envelope, context)
        if item.domain == envelope.domain and item.object_id == envelope.object_id
    )
    return current_head(matches)


def _literal_replay(head: SyncHead, envelope: SyncEnvelopeCreate) -> bool:
    """Return whether an incoming envelope exactly replays the stored head."""

    return all(
        getattr(head, field_name) == getattr(envelope, field_name)
        for field_name in _REPLAY_FIELDS
    )


def _has_base(envelope: SyncEnvelopeCreate) -> bool:
    """Return whether any optimistic lineage token was supplied."""

    return any(
        value is not None
        for value in (
            envelope.base_server_cursor,
            envelope.base_object_revision,
            envelope.base_object_hash,
        )
    )


def _same_identity(head: SyncHead, task_id: str, note_id: str) -> bool:
    """Return whether head metadata and payload preserve task-note identity."""

    return bool(
        head.object_id == task_id
        and head.parent_id == note_id
        and head.payload.get("task_id") == task_id
        and head.payload.get("note_id") == note_id
    )


def _is_deleted(head: SyncHead) -> bool:
    """Return whether a head represents a tombstoned task."""

    return head.operation == "tombstone" or head.deleted


def _trusted_task_bootstrap(
    envelope: SyncEnvelopeCreate,
    context: SyncAdapterContext | None,
) -> bool:
    """Return whether routing and context authorize private bootstrap capture."""

    bootstrap_id = envelope.routing_metadata.get("bootstrap_id")
    return bool(
        context is not None
        and context.trusted_server_origin
        and isinstance(bootstrap_id, str)
        and bootstrap_id
        and context.notes_task_bootstrap_id == bootstrap_id
        and envelope.routing_metadata.get("bootstrap_capture") is True
        and envelope.routing_metadata.get("source") == "notes-task-bootstrap"
        and envelope.routing_metadata.get("origin") == "server"
    )


def _trusted_server_task_mutation(
    envelope: SyncEnvelopeCreate,
    *,
    dataset: SyncDataset,
    context: SyncAdapterContext | None,
) -> bool:
    """Validate closed server provenance and optional projection evidence."""

    routing = envelope.routing_metadata
    required = {"source", "origin", "server_device_id", "server_owner_user_id"}
    if not (
        context is not None
        and context.trusted_server_origin
        and required.issubset(routing)
        and routing.get("origin") == "server"
        and routing.get("server_device_id") == "server-origin"
        and routing.get("server_owner_user_id") == dataset.owner_user_id
        and envelope.device_id == "server-origin"
        and isinstance(routing.get("source"), str)
        and 1 <= len(str(routing["source"])) <= 128
    ):
        return False
    projection = routing.get("task_projection")
    if projection is None:
        return True
    if not isinstance(projection, Mapping):
        return False
    try:
        from ..notes_task_coordinator import (  # Local import avoids adapter cycles.
            _validate_task_projection_group_metadata,
        )

        anchor = _validate_task_projection_group_metadata(projection)
    except (ImportError, ValueError):
        return False
    return (
        anchor.task_id == envelope.object_id
        and anchor.task_envelope_id == envelope.client_envelope_id
        and anchor.task_revision == envelope.object_revision
        and anchor.task_hash == envelope.payload_hash
    )


def _rejected(
    envelope: SyncEnvelopeCreate,
    error_code: str,
) -> AdapterRejected:
    """Build a bounded task validation rejection."""

    return AdapterRejected(
        client_envelope_id=envelope.client_envelope_id,
        error_code=error_code,
        message="notes.task envelope validation failed",
    )


def _conflict(
    envelope: SyncEnvelopeCreate,
    conflict_type: str,
) -> AdapterConflict:
    """Build a bounded task lineage conflict."""

    return AdapterConflict(
        client_envelope_id=envelope.client_envelope_id,
        domain=envelope.domain,
        entity_id=envelope.object_id,
        conflict_type=conflict_type,
        message="notes.task lineage requires a current authorized base",
    )


__all__ = ["NotesTaskDomainAdapter"]
