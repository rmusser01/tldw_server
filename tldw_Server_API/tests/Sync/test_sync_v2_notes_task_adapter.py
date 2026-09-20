"""Contracts for strict dormant ``notes.task`` adapter lineage."""

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
    notes_task_object_hash,
    parse_notes_task_v1,
)
from tldw_Server_API.app.core.Sync.v2.server_origin import SERVER_ORIGIN_DEVICE_ID

pytestmark = pytest.mark.unit

OWNER_ID = "owner-user-1"
DATASET_ID = "dataset-1"
TASK_ID = "11111111-1111-4111-8111-111111111111"
NOTE_ID = "22222222-2222-4222-8222-222222222222"
OTHER_NOTE_ID = "33333333-3333-4333-8333-333333333333"
DEVICE_ID = "44444444-4444-4444-8444-444444444444"
NOW = "2026-08-13T10:00:00+00:00"


def _adapter() -> domain_adapters.NotesTaskDomainAdapter:
    adapter_type = getattr(domain_adapters, "NotesTaskDomainAdapter", None)
    assert adapter_type is not None, "NotesTaskDomainAdapter is not implemented"
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


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "task_id": TASK_ID,
        "note_id": NOTE_ID,
        "title": "Prepare launch notes",
        "description": "Confirm the final release checklist.",
        "status": "open",
        "completed_at": None,
        "priority": "high",
        "due_date": "2026-08-31",
        "estimate": "90m",
        "recurrence": {
            "frequency": "weekly",
            "interval": 2,
            "by_weekday": ["mo", "we", "fr"],
            "until": "2026-12-31",
            "state": "active",
            "occurrence_index": 7,
        },
        "assignee_id": OWNER_ID,
        "tags": ["alpha", "Zulu"],
        "custom": {"board.column": "Next"},
    }
    payload.update(overrides)
    return payload


def _incoming(
    *,
    operation: str = "upsert",
    payload: dict[str, object] | None = None,
    base: SyncEnvelope | None = None,
    restore: object | None = None,
    suffix: str = "incoming",
    object_id: str = TASK_ID,
    parent_id: str | None = NOTE_ID,
    adapter_version: int = 1,
    schema_version: int = 1,
) -> SyncEnvelopeCreate:
    raw_payload = _payload() if payload is None else payload
    revision = 1 if base is None else int(base.object_revision or 0) + 1
    parsed = parse_notes_task_v1(raw_payload, owner_user_id=OWNER_ID)
    routing = {} if restore is None else {"restore_intent": restore}
    return SyncEnvelopeCreate(
        dataset_id=DATASET_ID,
        client_envelope_id=f"env-{suffix}",
        domain="notes.task",
        operation=operation,
        object_id=object_id,
        parent_id=parent_id,
        device_id=DEVICE_ID,
        base_server_cursor=base.server_cursor if base is not None else None,
        base_object_revision=base.object_revision if base is not None else None,
        base_object_hash=base.payload_hash if base is not None else None,
        object_revision=revision,
        entity_version=revision,
        adapter_version=adapter_version,
        schema_version=schema_version,
        payload=parsed.model_dump(mode="json"),
        payload_hash=notes_task_object_hash(
            parsed,
            revision=revision,
            deleted=operation == "tombstone",
        ),
        created_at_client=NOW,
        routing_metadata=routing,
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


def _note_head(
    note_id: str = NOTE_ID,
    *,
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


def _context(
    task_head: SyncEnvelope | None = None,
    *,
    note_head: SyncEnvelope | None = None,
    note_available: bool = True,
) -> SyncAdapterContext:
    note = _note_head() if note_head is None else note_head

    def get_head(domain: str, object_id: str):
        if domain == "notes.task" and task_head is not None:
            return task_head if object_id == task_head.object_id else None
        return None

    def get_authorized_note(note_id: str):
        if not note_available or note_id != note.object_id:
            return None
        return note

    return SyncAdapterContext(
        get_head=get_head,
        get_authorized_note=get_authorized_note,
    )


def _server_routing(incoming: SyncEnvelopeCreate) -> dict[str, object]:
    return {
        "source": "notes.tasks.rest",
        "origin": "server",
        "server_device_id": SERVER_ORIGIN_DEVICE_ID,
        "server_owner_user_id": OWNER_ID,
        "task_projection": {
            "projection_version": 1,
            "task_id": TASK_ID,
            "task_envelope_id": incoming.client_envelope_id,
            "task_revision": incoming.object_revision,
            "task_hash": incoming.payload_hash,
            "note_envelope_id": "note-envelope-2",
            "note_hash": "sha256:" + "a" * 64,
            "linked": True,
            "marker_hash": "sha256:" + "b" * 64,
        },
    }


def test_notes_task_adapter_accepts_closed_server_projection_routing_only() -> None:
    incoming = _incoming()
    routed = replace(
        incoming,
        device_id=SERVER_ORIGIN_DEVICE_ID,
        routing_metadata=_server_routing(incoming),
    )
    trusted_context = replace(_context(), trusted_server_origin=True)

    assert isinstance(
        _adapter().evaluate_envelope(
            routed,
            dataset=_dataset(),
            context=trusted_context,
        ),
        AdapterAccepted,
    )
    injected = _adapter().evaluate_envelope(
        routed,
        dataset=_dataset(),
        context=_context(),
    )
    malformed = _adapter().evaluate_envelope(
        replace(
            routed,
            routing_metadata={**_server_routing(incoming), "secret": "no"},
        ),
        dataset=_dataset(),
        context=trusted_context,
    )
    assert isinstance(injected, AdapterRejected)
    assert isinstance(malformed, AdapterRejected)


def test_notes_task_adapter_accepts_create_exact_update_and_completion_reopen() -> None:
    adapter = _adapter()
    create = _incoming()
    assert adapter.evaluate_envelope(
        create, dataset=_dataset(), context=_context()
    ) == AdapterAccepted(client_envelope_id=create.client_envelope_id)

    live = _stored(create)
    completion = _incoming(
        payload=_payload(status="done", completed_at="2026-08-13T11:00:00+00:00"),
        base=live,
        suffix="complete",
    )
    assert isinstance(
        adapter.evaluate_envelope(
            completion, dataset=_dataset(), context=_context(live)
        ),
        AdapterAccepted,
    )

    done = _stored(completion, cursor=11)
    reopen = _incoming(
        payload=_payload(status="open", completed_at=None),
        base=done,
        suffix="reopen",
    )
    assert isinstance(
        adapter.evaluate_envelope(reopen, dataset=_dataset(), context=_context(done)),
        AdapterAccepted,
    )


def test_notes_task_adapter_accepts_recurrence_state_mutation() -> None:
    live = _stored(_incoming())
    recurrence = dict(_payload()["recurrence"])
    recurrence["state"] = "paused"
    recurrence["occurrence_index"] = 8
    incoming = _incoming(
        payload=_payload(recurrence=recurrence),
        base=live,
        suffix="recurrence",
    )

    assert isinstance(
        _adapter().evaluate_envelope(
            incoming, dataset=_dataset(), context=_context(live)
        ),
        AdapterAccepted,
    )


def test_notes_task_adapter_accepts_exact_replay_without_new_lineage() -> None:
    incoming = _incoming()
    stored = _stored(incoming)

    assert _adapter().evaluate_envelope(
        incoming, dataset=_dataset(), context=_context(stored)
    ) == AdapterAccepted(client_envelope_id=incoming.client_envelope_id)


@pytest.mark.parametrize("operation", ["upsert", "tombstone"])
def test_notes_task_adapter_conflicts_stale_update_and_delete(operation: str) -> None:
    live = _stored(_incoming())
    stale = replace(live, server_cursor=9, payload_hash="sha256:" + "9" * 64)
    incoming = _incoming(operation=operation, base=stale, suffix=f"stale-{operation}")

    outcome = _adapter().evaluate_envelope(
        incoming, dataset=_dataset(), context=_context(live)
    )
    assert isinstance(outcome, AdapterConflict)
    assert outcome.conflict_type == "notes_task_base_conflict"


def test_notes_task_adapter_requires_empty_head_for_create() -> None:
    live = _stored(_incoming())
    second_create = _incoming(suffix="duplicate-create")

    outcome = _adapter().evaluate_envelope(
        second_create, dataset=_dataset(), context=_context(live)
    )
    assert isinstance(outcome, AdapterConflict)
    assert outcome.conflict_type == "notes_task_create_conflict"


def test_notes_task_adapter_requires_exact_live_base_for_tombstone() -> None:
    live = _stored(_incoming())
    tombstone = _incoming(operation="tombstone", base=live, suffix="delete")

    assert isinstance(
        _adapter().evaluate_envelope(
            tombstone, dataset=_dataset(), context=_context(live)
        ),
        AdapterAccepted,
    )


def test_notes_task_adapter_requires_explicit_exact_restore() -> None:
    live = _stored(_incoming())
    tombstone = _stored(
        _incoming(operation="tombstone", base=live, suffix="delete"),
        cursor=11,
    )

    ordinary = _incoming(base=tombstone, suffix="ordinary-upsert")
    outcome = _adapter().evaluate_envelope(
        ordinary, dataset=_dataset(), context=_context(tombstone)
    )
    assert isinstance(outcome, AdapterConflict)
    assert outcome.conflict_type == "notes_task_restore_conflict"

    restore = _incoming(base=tombstone, restore=True, suffix="restore")
    assert isinstance(
        _adapter().evaluate_envelope(
            restore, dataset=_dataset(), context=_context(tombstone)
        ),
        AdapterAccepted,
    )

    restore_live = _incoming(base=live, restore=True, suffix="restore-live")
    outcome = _adapter().evaluate_envelope(
        restore_live, dataset=_dataset(), context=_context(live)
    )
    assert isinstance(outcome, AdapterConflict)
    assert outcome.conflict_type == "notes_task_restore_conflict"


def test_notes_task_adapter_rejects_changed_task_or_parent_identity() -> None:
    live = _stored(_incoming())
    changed_parent = _incoming(
        payload=_payload(note_id=OTHER_NOTE_ID),
        base=live,
        parent_id=OTHER_NOTE_ID,
        suffix="changed-parent",
    )
    outcome = _adapter().evaluate_envelope(
        changed_parent,
        dataset=_dataset(),
        context=_context(live, note_head=_note_head(OTHER_NOTE_ID)),
    )
    assert isinstance(outcome, AdapterConflict)
    assert outcome.conflict_type == "notes_task_identity_conflict"

    changed_object = _incoming(object_id=OTHER_NOTE_ID, suffix="changed-object")
    outcome = _adapter().evaluate_envelope(
        changed_object, dataset=_dataset(), context=_context()
    )
    assert isinstance(outcome, AdapterConflict)
    assert outcome.conflict_type == "notes_task_identity_conflict"


def test_notes_task_adapter_defers_missing_or_foreign_parent_without_disclosure() -> None:
    missing = _adapter().evaluate_envelope(
        _incoming(), dataset=_dataset(), context=_context(note_available=False)
    )
    foreign = _adapter().evaluate_envelope(
        _incoming(),
        dataset=_dataset(),
        context=_context(note_head=_note_head(dataset_id="other-dataset")),
    )

    assert isinstance(missing, AdapterDeferred)
    assert isinstance(foreign, AdapterDeferred)
    assert missing.message == foreign.message
    assert TASK_ID not in missing.message
    assert NOTE_ID not in missing.message


def test_notes_task_adapter_conflicts_deleted_parent_and_requires_authorization_context() -> None:
    deleted = _adapter().evaluate_envelope(
        _incoming(),
        dataset=_dataset(),
        context=_context(note_head=_note_head(deleted=True)),
    )
    assert isinstance(deleted, AdapterConflict)
    assert deleted.conflict_type == "notes_task_parent_conflict"

    unavailable = _adapter().evaluate_envelope(
        _incoming(), dataset=_dataset(), context=None
    )
    assert isinstance(unavailable, AdapterRejected)
    assert unavailable.error_code == "notes_task_authorization_unavailable"


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ({"operation": "append"}, "notes_task_payload_invalid"),
        ({"adapter_version": 2}, "notes_task_payload_invalid"),
        ({"schema_version": 2}, "notes_task_payload_invalid"),
        ({"parent_id": OTHER_NOTE_ID}, "notes_task_identity_conflict"),
        ({"restore": False}, "notes_task_payload_invalid"),
    ],
)
def test_notes_task_adapter_rejects_invalid_envelope_and_routing_shapes(
    mutation: dict[str, object], expected_code: str
) -> None:
    incoming = _incoming(**mutation)
    outcome = _adapter().evaluate_envelope(
        incoming, dataset=_dataset(), context=_context()
    )

    if isinstance(outcome, AdapterRejected):
        assert outcome.error_code == expected_code
    else:
        assert isinstance(outcome, AdapterConflict)
        assert outcome.conflict_type == expected_code


def test_notes_task_adapter_rejects_wrong_hash_revision_and_noncanonical_payload() -> None:
    wrong_hash = replace(_incoming(), payload_hash="sha256:" + "0" * 64)
    wrong_revision = replace(_incoming(), object_revision=2, entity_version=2)
    malformed = replace(_incoming(), payload={**_payload(), "title": " spaced "})

    for incoming in (wrong_hash, wrong_revision, malformed):
        outcome = _adapter().evaluate_envelope(
            incoming, dataset=_dataset(), context=_context()
        )
        assert isinstance(outcome, AdapterRejected)
        assert outcome.error_code == "notes_task_payload_invalid"
