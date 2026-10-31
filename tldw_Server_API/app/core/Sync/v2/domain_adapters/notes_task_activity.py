"""Strict dormant Sync v1 adapter for immutable Notes task activity."""

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
    NotesTaskActivityV1,
    notes_task_activity_object_hash,
    parse_notes_task_activity_tombstone_v1,
    parse_notes_task_activity_v1,
)
from ._lineage import incoming_references_exact_head


@dataclass(slots=True)
class NotesTaskActivityDomainAdapter:
    """Validate immutable activity lineage without public domain activation."""

    domain: str = "notes.task_activity"
    supported_adapter_versions: set[int] = field(default_factory=lambda: {1})

    def evaluate_envelope(
        self,
        envelope: SyncEnvelopeCreate,
        *,
        dataset: SyncDataset,
        context: SyncAdapterContext | None = None,
    ) -> SyncAdapterOutcome:
        """Return a stable authorization, lineage, or acceptance outcome."""

        if (
            envelope.domain != self.domain
            or envelope.operation not in {"upsert", "tombstone"}
            or envelope.adapter_version != 1
            or envelope.schema_version != 1
        ):
            return _rejected(envelope, "notes_task_activity_payload_invalid")
        trusted_bootstrap = _trusted_activity_bootstrap(envelope, context)
        trusted_server_mutation = _trusted_server_activity_mutation(
            envelope,
            dataset=dataset,
            context=context,
        )
        coordinator_derived = bool(
            context is not None and context.coordinator_derived_task_activity
        )
        routing_fields = set(envelope.routing_metadata)
        bootstrap_routing = {
            "bootstrap_capture",
            "bootstrap_id",
            "source",
            "origin",
            "server_device_id",
            "server_owner_user_id",
        }
        server_routing = {
            "source",
            "origin",
            "server_device_id",
            "server_owner_user_id",
        }
        valid_routing_shape = (
            routing_fields == bootstrap_routing
            if trusted_bootstrap
            else (
                frozenset(routing_fields)
                in {
                    frozenset(server_routing),
                    frozenset(server_routing | {"task_projection"}),
                }
                if trusted_server_mutation
                else (
                    routing_fields in (set(), {"task_projection"})
                    if coordinator_derived
                    else not routing_fields
                )
            )
        )
        if not valid_routing_shape:
            return _rejected(envelope, "notes_task_activity_payload_invalid")

        head = _get_head(envelope, context)
        exact_replay = head is not None and _exact_semantic_replay(head, envelope)
        if head is not None and _is_deleted(head) and not exact_replay:
            return _conflict(envelope, "notes_task_activity_immutable")

        if context is None or context.authenticated_actor_type is None:
            return _rejected(envelope, "notes_task_activity_authorization_unavailable")

        if exact_replay and envelope.operation == "tombstone":
            note_id = envelope.payload.get("note_id")
            task_id = envelope.payload.get("task_id")
            if not isinstance(note_id, str) or not (
                task_id is None or isinstance(task_id, str)
            ):
                return _rejected(envelope, "notes_task_activity_payload_invalid")
            parent_outcome = None if trusted_bootstrap else _authorize_parents(
                envelope,
                dataset=dataset,
                context=context,
                note_id=note_id,
                task_id=task_id,
            )
            return parent_outcome or AdapterAccepted(
                client_envelope_id=envelope.client_envelope_id
            )

        original: NotesTaskActivityV1 | None = None
        try:
            if envelope.operation == "upsert":
                payload = parse_notes_task_activity_v1(
                    envelope.payload,
                    owner_user_id=dataset.owner_user_id,
                    bound_actor_type=context.authenticated_actor_type,
                    bound_actor_id=context.authenticated_actor_id,
                    authenticated_device_id=context.authenticated_device_id,
                    trusted_server_origin=context.trusted_server_origin,
                )
                expected_hash = notes_task_activity_object_hash(
                    payload,
                    revision=1,
                    deleted=False,
                )
            else:
                if head is None:
                    return _conflict(envelope, "notes_task_activity_base_conflict")
                original = _parse_stored_create(head, dataset.owner_user_id)
                payload = parse_notes_task_activity_tombstone_v1(
                    envelope.payload,
                    envelope_created_at_client=envelope.created_at_client or "",
                    original_activity=original,
                )
                expected_hash = notes_task_activity_object_hash(
                    payload,
                    revision=2,
                    deleted=True,
                    activity_id=envelope.object_id,
                    original_create_hash=head.payload_hash,
                )
            expected_revision = 1 if envelope.operation == "upsert" else 2
            if (
                envelope.object_revision != expected_revision
                or envelope.entity_version != expected_revision
                or envelope.payload_hash != expected_hash
            ):
                raise NotesTaskContractError("activity lineage is invalid")
        except NotesTaskContractError:
            return _rejected(envelope, "notes_task_activity_payload_invalid")

        note_id = payload.note_id
        task_id = payload.task_id
        activity_id = payload.activity_id if isinstance(payload, NotesTaskActivityV1) else envelope.object_id
        if envelope.object_id != activity_id or envelope.parent_id != note_id:
            return _conflict(envelope, "notes_task_activity_identity_conflict")

        parent_outcome = None if trusted_bootstrap else _authorize_parents(
            envelope,
            dataset=dataset,
            context=context,
            note_id=note_id,
            task_id=task_id,
            allow_deleted_task=(
                (trusted_server_mutation or coordinator_derived)
                and isinstance(payload, NotesTaskActivityV1)
                and payload.event_type == "deleted"
            ),
        )
        if parent_outcome is not None:
            return parent_outcome

        if isinstance(payload, NotesTaskActivityV1):
            coordinator_derived = coordinator_derived and payload.source_kind == "client"
            if coordinator_derived and routing_fields and not _valid_coordinator_projection(
                envelope,
                task_id=payload.task_id,
            ):
                return _rejected(envelope, "notes_task_activity_payload_invalid")
            if (
                not context.trusted_server_origin
                and not coordinator_derived
                and payload.event_type != "corrected"
            ):
                return _rejected(envelope, "notes_task_activity_origin_invalid")
            if payload.corrects_activity_id is not None:
                target = _lookup_head(context, payload.corrects_activity_id)
                if not _same_activity_scope(target, dataset.dataset_id, note_id, task_id):
                    return _parent_deferred(envelope)

        if exact_replay:
            return AdapterAccepted(client_envelope_id=envelope.client_envelope_id)

        if head is None:
            if envelope.operation != "upsert" or _has_base(envelope):
                return _conflict(envelope, "notes_task_activity_base_conflict")
            return AdapterAccepted(client_envelope_id=envelope.client_envelope_id)

        if envelope.operation == "upsert":
            return _conflict(envelope, "notes_task_activity_identity_reused")
        if not _same_activity_scope(head, dataset.dataset_id, note_id, task_id):
            return _conflict(envelope, "notes_task_activity_identity_conflict")
        if not incoming_references_exact_head(envelope, head):
            return _conflict(envelope, "notes_task_activity_base_conflict")
        return AdapterAccepted(client_envelope_id=envelope.client_envelope_id)


def _get_head(
    envelope: SyncEnvelopeCreate,
    context: SyncAdapterContext | None,
) -> SyncHead | None:
    """Load the current activity head when a lookup is available."""

    return _lookup_head(context, envelope.object_id)


def _lookup_head(context: SyncAdapterContext | None, object_id: str) -> SyncHead | None:
    """Load one activity head without exposing other scopes."""

    if context is None:
        return None
    if context.get_head is not None:
        return context.get_head("notes.task_activity", object_id)
    return next(
        (
            item
            for item in reversed(tuple(context.prior_envelopes))
            if item.domain == "notes.task_activity" and item.object_id == object_id
        ),
        None,
    )


def _parse_stored_create(head: SyncHead, owner_user_id: str) -> NotesTaskActivityV1:
    """Re-validate one accepted live create using its stored provenance."""

    payload = head.payload
    source_kind = payload.get("source_kind")
    trusted = source_kind != "client"
    return parse_notes_task_activity_v1(
        payload,
        owner_user_id=owner_user_id,
        bound_actor_type=str(payload.get("actor_type")),
        bound_actor_id=payload.get("actor_id"),
        authenticated_device_id=(None if trusted else head.device_id),
        trusted_server_origin=trusted,
    )


def _authorize_parents(
    envelope: SyncEnvelopeCreate,
    *,
    dataset: SyncDataset,
    context: SyncAdapterContext,
    note_id: str,
    task_id: str | None,
    allow_deleted_task: bool = False,
) -> SyncAdapterOutcome | None:
    """Authorize the required note and optional same-note live task."""

    if context.get_authorized_note is None:
        return _rejected(envelope, "notes_task_activity_authorization_unavailable")
    note = context.get_authorized_note(note_id)
    if note is None or note.dataset_id != dataset.dataset_id:
        return _parent_deferred(envelope)
    if _is_deleted(note):
        return _conflict(envelope, "notes_task_activity_parent_conflict")
    if task_id is None:
        return None
    if context.get_authorized_task is None:
        return _rejected(envelope, "notes_task_activity_authorization_unavailable")
    task = context.get_authorized_task(task_id)
    if (
        task is None
        or task.dataset_id != dataset.dataset_id
        or task.parent_id != note_id
        or task.payload.get("note_id") != note_id
    ):
        return _parent_deferred(envelope)
    if _is_deleted(task) and not allow_deleted_task:
        return _conflict(envelope, "notes_task_activity_parent_conflict")
    return None


def _valid_coordinator_projection(
    envelope: SyncEnvelopeCreate,
    *,
    task_id: str | None,
) -> bool:
    """Validate client-derived projection evidence without trusting client routing."""

    projection = envelope.routing_metadata.get("task_projection")
    if not isinstance(projection, Mapping) or task_id is None:
        return False
    try:
        from ..notes_task_coordinator import _validate_task_projection_group_metadata

        anchor = _validate_task_projection_group_metadata(projection)
    except (ImportError, ValueError):
        return False
    return anchor.task_id == task_id


def _same_activity_scope(
    head: SyncHead | None,
    dataset_id: str,
    note_id: str,
    task_id: str | None,
) -> bool:
    """Return whether one activity head has the exact authorized parents."""

    if head is None or head.dataset_id != dataset_id or head.parent_id != note_id:
        return False
    payload = head.payload
    return payload.get("note_id") == note_id and payload.get("task_id") == task_id


def _trusted_activity_bootstrap(
    envelope: SyncEnvelopeCreate,
    context: SyncAdapterContext | None,
) -> bool:
    """Return whether context and routing authorize legacy activity capture."""

    bootstrap_id = envelope.routing_metadata.get("bootstrap_id")
    metadata = envelope.payload.get("metadata")
    return bool(
        context is not None
        and context.trusted_server_origin
        and isinstance(bootstrap_id, str)
        and bootstrap_id
        and context.notes_task_activity_bootstrap_id == bootstrap_id
        and envelope.operation == "upsert"
        and envelope.payload.get("source_kind") == "trusted_bootstrap_v1"
        and isinstance(metadata, dict)
        and metadata.get("legacy_source_verified") is True
        and envelope.routing_metadata.get("bootstrap_capture") is True
        and envelope.routing_metadata.get("source")
        == "notes-task-activity-bootstrap"
        and envelope.routing_metadata.get("origin") == "server"
    )


def _trusted_server_activity_mutation(
    envelope: SyncEnvelopeCreate,
    *,
    dataset: SyncDataset,
    context: SyncAdapterContext | None,
) -> bool:
    """Validate closed server provenance and optional task projection anchor."""

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
        envelope.payload.get("task_id") == anchor.task_id
        and envelope.parent_id == envelope.payload.get("note_id")
    )


def _exact_semantic_replay(head: SyncHead, envelope: SyncEnvelopeCreate) -> bool:
    """Return whether an immutable ID and canonical fingerprint are unchanged."""

    return bool(
        head.dataset_id == envelope.dataset_id
        and head.operation == envelope.operation
        and head.object_id == envelope.object_id
        and head.parent_id == envelope.parent_id
        and head.object_revision == envelope.object_revision
        and head.entity_version == envelope.entity_version
        and head.payload_hash == envelope.payload_hash
        and head.payload == envelope.payload
        and head.deleted == envelope.deleted
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


def _is_deleted(head: SyncHead) -> bool:
    """Return whether an activity or parent head is tombstoned."""

    return head.operation == "tombstone" or head.deleted


def _parent_deferred(envelope: SyncEnvelopeCreate) -> AdapterDeferred:
    """Return a sanitized parent/target dependency outcome."""

    return AdapterDeferred(
        client_envelope_id=envelope.client_envelope_id,
        message="The authorized activity parent is not available yet",
    )


def _rejected(envelope: SyncEnvelopeCreate, error_code: str) -> AdapterRejected:
    """Build a bounded activity validation rejection."""

    return AdapterRejected(
        client_envelope_id=envelope.client_envelope_id,
        error_code=error_code,
        message="notes.task_activity envelope validation failed",
    )


def _conflict(envelope: SyncEnvelopeCreate, conflict_type: str) -> AdapterConflict:
    """Build a bounded immutable activity conflict."""

    return AdapterConflict(
        client_envelope_id=envelope.client_envelope_id,
        domain=envelope.domain,
        entity_id=envelope.object_id,
        conflict_type=conflict_type,
        message="notes.task_activity immutable lineage conflicted",
    )


__all__ = ["NotesTaskActivityDomainAdapter"]
