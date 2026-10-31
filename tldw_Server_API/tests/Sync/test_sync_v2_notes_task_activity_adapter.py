"""Contracts for immutable dormant ``notes.task_activity`` lineage."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tldw_Server_API.app.core.Sync.v2 import domain_adapters
from tldw_Server_API.app.core.Sync.v2.adapters import (
    AdapterAccepted,
    AdapterConflict,
    AdapterDeferred,
    AdapterRejected,
    SyncAdapterContext,
)
from tldw_Server_API.app.core.Sync.v2.models import (
    DEFAULT_M1_ENCRYPTION_POLICY,
    SyncDataset,
    SyncEnvelope,
    SyncEnvelopeCreate,
)
from tldw_Server_API.app.core.Sync.v2.notes_task_contract import (
    NotesTaskActivityV1,
    notes_task_activity_object_hash,
    parse_notes_task_activity_tombstone_v1,
    parse_notes_task_activity_v1,
)
from tldw_Server_API.app.core.Sync.v2.server_origin import SERVER_ORIGIN_DEVICE_ID

pytestmark = pytest.mark.unit

OWNER_ID = "owner-user-1"
DATASET_ID = "dataset-1"
ACTIVITY_ID = "11111111-1111-4111-8111-111111111111"
TARGET_ACTIVITY_ID = "22222222-2222-4222-8222-222222222222"
NOTE_ID = "33333333-3333-4333-8333-333333333333"
TASK_ID = "44444444-4444-4444-8444-444444444444"
OTHER_NOTE_ID = "55555555-5555-4555-8555-555555555555"
OTHER_TASK_ID = "66666666-6666-4666-8666-666666666666"
DEVICE_ID = "77777777-7777-4777-8777-777777777777"
NOW = "2026-08-13T10:00:00+00:00"


def _adapter() -> domain_adapters.NotesTaskActivityDomainAdapter:
    adapter_type = getattr(domain_adapters, "NotesTaskActivityDomainAdapter", None)
    assert adapter_type is not None, "NotesTaskActivityDomainAdapter is not implemented"
    return adapter_type()


def _dataset() -> SyncDataset:
    return SyncDataset(
        dataset_id=DATASET_ID,
        owner_user_id=OWNER_ID,
        scope_type="personal",
        encryption_policy=DEFAULT_M1_ENCRYPTION_POLICY,
        domains=["notes.note"],
        workspace_id=None,
        metadata={},
        created_at=NOW,
        updated_at=NOW,
    )


def _values(event_type: str) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    if event_type in {"completed", "corrected"}:
        return {"status": "open"}, {"status": "done"}
    if event_type == "created":
        return None, {
            "title": "Prepare launch notes",
            "status": "open",
            "completed_at": None,
            "metadata": {
                "description": None,
                "priority": None,
                "due_date": None,
                "estimate": None,
                "recurrence": None,
                "assignee_id": None,
                "tags": [],
                "custom": {},
            },
        }
    raise AssertionError(f"unsupported test event type: {event_type}")


def _payload(
    *,
    activity_id: str = ACTIVITY_ID,
    note_id: str = NOTE_ID,
    task_id: str | None = TASK_ID,
    event_type: str = "corrected",
    trusted: bool = False,
    actor_id: str | None = OWNER_ID,
) -> dict[str, object]:
    old_value, new_value = _values(event_type)
    return {
        "activity_id": activity_id,
        "note_id": note_id,
        "task_id": task_id,
        "event_type": event_type,
        "actor_type": "user",
        "actor_id": actor_id,
        "source_device_id": None if trusted else DEVICE_ID,
        "client_occurred_at": NOW,
        "source_kind": "rest" if trusted else "client",
        "corrects_activity_id": TARGET_ACTIVITY_ID if event_type == "corrected" else None,
        "old_value": old_value,
        "new_value": new_value,
        "metadata": {},
    }


def _parse_create(payload: dict[str, object], *, trusted: bool) -> NotesTaskActivityV1:
    return parse_notes_task_activity_v1(
        payload,
        owner_user_id=OWNER_ID,
        bound_actor_type="user",
        bound_actor_id=OWNER_ID,
        authenticated_device_id=None if trusted else DEVICE_ID,
        trusted_server_origin=trusted,
    )


def _incoming_create(
    *,
    activity_id: str = ACTIVITY_ID,
    note_id: str = NOTE_ID,
    task_id: str | None = TASK_ID,
    event_type: str = "corrected",
    trusted: bool = False,
    suffix: str = "create",
) -> SyncEnvelopeCreate:
    parsed = _parse_create(
        _payload(
            activity_id=activity_id,
            note_id=note_id,
            task_id=task_id,
            event_type=event_type,
            trusted=trusted,
        ),
        trusted=trusted,
    )
    return SyncEnvelopeCreate(
        dataset_id=DATASET_ID,
        client_envelope_id=f"env-{suffix}",
        domain="notes.task_activity",
        operation="upsert",
        object_id=activity_id,
        parent_id=note_id,
        device_id=None if trusted else DEVICE_ID,
        object_revision=1,
        entity_version=1,
        adapter_version=1,
        schema_version=1,
        payload=parsed.model_dump(mode="json"),
        payload_hash=notes_task_activity_object_hash(parsed, revision=1, deleted=False),
        created_at_client=NOW,
        routing_metadata={},
    )


def _stored(incoming: SyncEnvelopeCreate, *, cursor: int = 10) -> SyncEnvelope:
    excluded = {"server_cursor", "server_sequence"}
    return SyncEnvelope(
        **{
            field_name: getattr(incoming, field_name)
            for field_name in incoming.__dataclass_fields__
            if field_name not in excluded
        },
        server_cursor=cursor,
    )


def _incoming_tombstone(
    head: SyncEnvelope,
    *,
    suffix: str = "tombstone",
    reason: str = "user_request",
) -> SyncEnvelopeCreate:
    original = _parse_create(dict(head.payload), trusted=head.device_id is None)
    raw = {
        "note_id": original.note_id,
        "task_id": original.task_id,
        "deleted_at": NOW,
        "delete_reason": reason,
    }
    parsed = parse_notes_task_activity_tombstone_v1(
        raw,
        envelope_created_at_client=NOW,
        original_activity=original,
    )
    return SyncEnvelopeCreate(
        dataset_id=DATASET_ID,
        client_envelope_id=f"env-{suffix}",
        domain="notes.task_activity",
        operation="tombstone",
        object_id=head.object_id,
        parent_id=head.parent_id,
        device_id=DEVICE_ID,
        base_server_cursor=head.server_cursor,
        base_object_revision=head.object_revision,
        base_object_hash=head.payload_hash,
        object_revision=2,
        entity_version=2,
        adapter_version=1,
        schema_version=1,
        payload=parsed.model_dump(mode="json"),
        payload_hash=notes_task_activity_object_hash(
            parsed,
            revision=2,
            deleted=True,
            activity_id=str(head.object_id),
            original_create_hash=head.payload_hash,
        ),
        created_at_client=NOW,
        deleted=True,
        routing_metadata={},
    )


def _note_head(
    *,
    note_id: str = NOTE_ID,
    dataset_id: str = DATASET_ID,
    deleted: bool = False,
) -> SyncEnvelope:
    return SyncEnvelope(
        dataset_id=dataset_id,
        client_envelope_id=f"note-{note_id}",
        domain="notes.note",
        operation="tombstone" if deleted else "upsert",
        server_cursor=1,
        object_id=note_id,
        object_revision=1,
        entity_version=1,
        payload={"title": "Parent", "content": "body"},
        payload_hash="sha256:" + "1" * 64,
        created_at_client=NOW,
        deleted=deleted,
    )


def _task_head(
    *,
    task_id: str = TASK_ID,
    note_id: str = NOTE_ID,
    dataset_id: str = DATASET_ID,
    deleted: bool = False,
) -> SyncEnvelope:
    return SyncEnvelope(
        dataset_id=dataset_id,
        client_envelope_id=f"task-{task_id}",
        domain="notes.task",
        operation="tombstone" if deleted else "upsert",
        server_cursor=2,
        object_id=task_id,
        parent_id=note_id,
        object_revision=1,
        entity_version=1,
        payload={"task_id": task_id, "note_id": note_id},
        payload_hash="sha256:" + "2" * 64,
        created_at_client=NOW,
        deleted=deleted,
    )


def _correction_target(
    *,
    note_id: str = NOTE_ID,
    task_id: str | None = TASK_ID,
    dataset_id: str = DATASET_ID,
) -> SyncEnvelope:
    incoming = _incoming_create(
        activity_id=TARGET_ACTIVITY_ID,
        note_id=note_id,
        task_id=task_id,
        event_type="completed",
        trusted=True,
        suffix="target",
    )
    return replace(_stored(incoming, cursor=3), dataset_id=dataset_id)


def _context(
    activity_head: SyncEnvelope | None = None,
    *,
    trusted: bool = False,
    note: SyncEnvelope | None = None,
    task: SyncEnvelope | None = None,
    target: SyncEnvelope | None = None,
    note_available: bool = True,
    task_available: bool = True,
) -> SyncAdapterContext:
    authorized_note = _note_head() if note is None else note
    authorized_task = _task_head() if task is None else task
    correction = _correction_target() if target is None else target

    def get_head(domain: str, object_id: str):
        if domain != "notes.task_activity":
            return None
        if activity_head is not None and object_id == activity_head.object_id:
            return activity_head
        if object_id == correction.object_id:
            return correction
        return None

    def get_authorized_note(note_id: str):
        if not note_available or note_id != authorized_note.object_id:
            return None
        return authorized_note

    def get_authorized_task(task_id: str):
        if not task_available or task_id != authorized_task.object_id:
            return None
        return authorized_task

    return SyncAdapterContext(
        get_head=get_head,
        get_authorized_note=get_authorized_note,
        get_authorized_task=get_authorized_task,
        trusted_server_origin=trusted,
        authenticated_actor_type="user",
        authenticated_actor_id=OWNER_ID,
        authenticated_device_id=None if trusted else DEVICE_ID,
    )


def test_activity_adapter_accepts_closed_server_projection_routing_only() -> None:
    incoming = _incoming_create(event_type="created", trusted=True)
    routing = {
        "source": "notes.tasks.mcp",
        "origin": "server",
        "server_device_id": SERVER_ORIGIN_DEVICE_ID,
        "server_owner_user_id": OWNER_ID,
        "task_projection": {
            "projection_version": 1,
            "task_id": TASK_ID,
            "task_envelope_id": "task-envelope-1",
            "task_revision": 1,
            "task_hash": "sha256:" + "a" * 64,
            "note_envelope_id": "note-envelope-1",
            "note_hash": "sha256:" + "b" * 64,
            "linked": True,
            "marker_hash": "sha256:" + "c" * 64,
        },
    }
    routed = replace(
        incoming,
        device_id=SERVER_ORIGIN_DEVICE_ID,
        routing_metadata=routing,
    )

    assert isinstance(
        _adapter().evaluate_envelope(
            routed,
            dataset=_dataset(),
            context=_context(trusted=True),
        ),
        AdapterAccepted,
    )
    injected = _adapter().evaluate_envelope(
        routed,
        dataset=_dataset(),
        context=_context(),
    )
    malformed = _adapter().evaluate_envelope(
        replace(routed, routing_metadata={**routing, "markdown": "secret"}),
        dataset=_dataset(),
        context=_context(trusted=True),
    )
    assert isinstance(injected, AdapterRejected)
    assert isinstance(malformed, AdapterRejected)


@pytest.mark.parametrize("task_id", [TASK_ID, None])
def test_activity_adapter_accepts_trusted_lifecycle_create_with_optional_task(
    task_id: str | None,
) -> None:
    incoming = _incoming_create(
        task_id=task_id,
        event_type="created",
        trusted=True,
    )

    outcome = _adapter().evaluate_envelope(
        incoming,
        dataset=_dataset(),
        context=_context(trusted=True),
    )

    assert outcome == AdapterAccepted(client_envelope_id=incoming.client_envelope_id)


def test_activity_adapter_allows_only_authorized_direct_corrections() -> None:
    correction = _incoming_create()
    accepted = _adapter().evaluate_envelope(
        correction,
        dataset=_dataset(),
        context=_context(),
    )
    assert isinstance(accepted, AdapterAccepted)

    lifecycle = _incoming_create(event_type="completed", trusted=True)
    lifecycle = replace(
        lifecycle,
        device_id=DEVICE_ID,
        payload=_payload(event_type="completed", trusted=False),
    )
    parsed = _parse_create(dict(lifecycle.payload), trusted=False)
    lifecycle = replace(
        lifecycle,
        payload_hash=notes_task_activity_object_hash(
            parsed, revision=1, deleted=False
        ),
    )
    rejected = _adapter().evaluate_envelope(
        lifecycle,
        dataset=_dataset(),
        context=_context(),
    )
    assert isinstance(rejected, AdapterRejected)
    assert rejected.error_code == "notes_task_activity_origin_invalid"


def test_activity_adapter_accepts_exact_fingerprint_replay_without_duplication() -> None:
    incoming = _incoming_create(event_type="created", trusted=True)
    stored = _stored(incoming)
    replay = replace(incoming, client_envelope_id="env-replay")

    outcome = _adapter().evaluate_envelope(
        replay,
        dataset=_dataset(),
        context=_context(stored, trusted=True),
    )

    assert outcome == AdapterAccepted(client_envelope_id="env-replay")


def test_activity_adapter_replay_still_requires_authorized_scope() -> None:
    incoming = _incoming_create(event_type="created", trusted=True)
    stored = _stored(incoming)
    replay = replace(incoming, client_envelope_id="env-replay")

    unavailable = _adapter().evaluate_envelope(
        replay,
        dataset=_dataset(),
        context=None,
    )
    foreign = _adapter().evaluate_envelope(
        replay,
        dataset=_dataset(),
        context=_context(
            stored,
            trusted=True,
            note=_note_head(dataset_id="other-dataset"),
        ),
    )

    assert isinstance(unavailable, AdapterRejected)
    assert unavailable.error_code == "notes_task_activity_authorization_unavailable"
    assert isinstance(foreign, AdapterDeferred)


def test_activity_adapter_conflicts_changed_stable_id_reuse() -> None:
    stored = _stored(_incoming_create(event_type="created", trusted=True))
    changed = _incoming_create(
        event_type="completed",
        trusted=True,
        suffix="changed",
    )

    outcome = _adapter().evaluate_envelope(
        changed,
        dataset=_dataset(),
        context=_context(stored, trusted=True),
    )

    assert isinstance(outcome, AdapterConflict)
    assert outcome.conflict_type == "notes_task_activity_identity_reused"


def test_activity_adapter_defers_missing_or_foreign_note_without_disclosure() -> None:
    incoming = _incoming_create(event_type="created", trusted=True)
    missing = _adapter().evaluate_envelope(
        incoming,
        dataset=_dataset(),
        context=_context(trusted=True, note_available=False),
    )
    foreign = _adapter().evaluate_envelope(
        incoming,
        dataset=_dataset(),
        context=_context(
            trusted=True,
            note=_note_head(dataset_id="other-dataset"),
        ),
    )

    assert isinstance(missing, AdapterDeferred)
    assert isinstance(foreign, AdapterDeferred)
    assert missing.message == foreign.message
    assert NOTE_ID not in missing.message


def test_activity_adapter_requires_same_note_live_task() -> None:
    incoming = _incoming_create(event_type="created", trusted=True)
    missing = _adapter().evaluate_envelope(
        incoming,
        dataset=_dataset(),
        context=_context(trusted=True, task_available=False),
    )
    wrong_note = _adapter().evaluate_envelope(
        incoming,
        dataset=_dataset(),
        context=_context(
            trusted=True,
            task=_task_head(note_id=OTHER_NOTE_ID),
        ),
    )
    deleted = _adapter().evaluate_envelope(
        incoming,
        dataset=_dataset(),
        context=_context(trusted=True, task=_task_head(deleted=True)),
    )

    assert isinstance(missing, AdapterDeferred)
    assert isinstance(wrong_note, AdapterDeferred)
    assert missing.message == wrong_note.message
    assert isinstance(deleted, AdapterConflict)
    assert deleted.conflict_type == "notes_task_activity_parent_conflict"


def test_activity_adapter_accepts_exact_one_way_tombstone_and_replay() -> None:
    live = _stored(_incoming_create(event_type="created", trusted=True))
    tombstone = _incoming_tombstone(live)
    accepted = _adapter().evaluate_envelope(
        tombstone,
        dataset=_dataset(),
        context=_context(live),
    )
    assert isinstance(accepted, AdapterAccepted)

    deleted = _stored(tombstone, cursor=11)
    replay = replace(tombstone, client_envelope_id="env-delete-replay")
    replayed = _adapter().evaluate_envelope(
        replay,
        dataset=_dataset(),
        context=_context(deleted),
    )
    assert isinstance(replayed, AdapterAccepted)


def test_activity_adapter_rejects_restore_update_and_second_tombstone() -> None:
    live = _stored(_incoming_create(event_type="created", trusted=True))
    deleted = _stored(_incoming_tombstone(live), cursor=11)

    recreate = _incoming_create(event_type="created", trusted=True, suffix="recreate")
    recreate_outcome = _adapter().evaluate_envelope(
        recreate,
        dataset=_dataset(),
        context=_context(deleted, trusted=True),
    )
    changed_delete = _incoming_tombstone(live, reason="correction")
    changed_delete = replace(
        changed_delete,
        base_server_cursor=deleted.server_cursor,
        base_object_revision=deleted.object_revision,
        base_object_hash=deleted.payload_hash,
    )
    delete_outcome = _adapter().evaluate_envelope(
        changed_delete,
        dataset=_dataset(),
        context=_context(deleted),
    )

    for outcome in (recreate_outcome, delete_outcome):
        assert isinstance(outcome, AdapterConflict)
        assert outcome.conflict_type == "notes_task_activity_immutable"


def test_activity_adapter_rejects_invalid_lineage_and_contract() -> None:
    incoming = _incoming_create(event_type="created", trusted=True)
    invalid = (
        replace(incoming, object_revision=2, entity_version=2),
        replace(incoming, payload_hash="sha256:" + "0" * 64),
        replace(incoming, parent_id=OTHER_NOTE_ID),
        replace(incoming, adapter_version=2),
    )

    outcomes = [
        _adapter().evaluate_envelope(
            envelope,
            dataset=_dataset(),
            context=_context(trusted=True),
        )
        for envelope in invalid
    ]

    assert isinstance(outcomes[0], AdapterRejected)
    assert isinstance(outcomes[1], AdapterRejected)
    assert isinstance(outcomes[2], AdapterConflict)
    assert isinstance(outcomes[3], AdapterRejected)


def test_activity_adapter_requires_correction_target_in_exact_scope() -> None:
    incoming = _incoming_create()
    missing = _adapter().evaluate_envelope(
        incoming,
        dataset=_dataset(),
        context=_context(target=replace(_correction_target(), object_id=OTHER_TASK_ID)),
    )
    wrong_note = _adapter().evaluate_envelope(
        incoming,
        dataset=_dataset(),
        context=_context(target=_correction_target(note_id=OTHER_NOTE_ID)),
    )

    assert isinstance(missing, AdapterDeferred)
    assert isinstance(wrong_note, AdapterDeferred)
    assert missing.message == wrong_note.message
