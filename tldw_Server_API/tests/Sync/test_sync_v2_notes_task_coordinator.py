"""Deterministic compound coordination for Notes task mutations."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Notes_Tasks.models import TaskActor
from tldw_Server_API.app.core.Notes_Tasks.projection_markers import (
    TaskMarker,
    task_marker_hash,
)
from tldw_Server_API.app.core.Notes_Tasks.service import (
    NotesTaskActivityCapture,
    NotesTaskCaptureMutation,
    NotesTaskService,
)
from tldw_Server_API.app.core.Sync.v2.adapters import SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.domain_adapters import (
    NotesTaskActivityDomainAdapter,
    NotesTaskDomainAdapter,
)
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.materializers import (
    MaterializationResult,
    NotesTaskActivityMaterializer,
    NotesTaskMaterializer,
)
from tldw_Server_API.app.core.Sync.v2.models import SyncEnvelopeCreate
from tldw_Server_API.app.core.Sync.v2.notes_task_activity_bootstrap import (
    NotesTaskActivityBootstrapper,
)
from tldw_Server_API.app.core.Sync.v2.notes_task_bootstrap import NotesTaskBootstrapper
from tldw_Server_API.app.core.Sync.v2.notes_task_contract import (
    notes_task_object_hash,
    parse_notes_task_v1,
)
from tldw_Server_API.app.core.Sync.v2.notes_task_coordinator import (
    TASK_PROJECTION_ROUTING_KEY,
    NotesTaskCoordinator,
    _validate_task_projection_group_metadata,
)
from tldw_Server_API.app.core.Sync.v2.server_origin import canonical_payload_hash
from tldw_Server_API.app.core.Sync.v2.server_origin_batch import (
    ServerOriginMutationStep,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

pytestmark = pytest.mark.unit

OWNER_ID = "coordinator-owner"
DATASET_ID = "coordinator-dataset"
NOTE_ID = "11111111-1111-4111-8111-111111111111"


def _uuid(index: int) -> str:
    return str(UUID(int=index, version=4))


def _mutation(index: int = 2, *, projection_status: str = "live") -> NotesTaskCaptureMutation:
    task_id = _uuid(index)
    activity_id = _uuid(index + 10_000)
    task_hash = "sha256:" + f"{index:064x}"
    task_envelope_id = f"notes-task-server-{index:032x}"
    activity_envelope_id = f"notes-task-activity-server-{index:032x}"
    task_step = ServerOriginMutationStep(
        domain="notes.task",
        operation="upsert",
        object_id=task_id,
        parent_id=NOTE_ID,
        payload={"task_id": task_id, "note_id": NOTE_ID, "title": f"Task {index}"},
        client_envelope_id=task_envelope_id,
        object_revision=2,
    )
    activity_step = ServerOriginMutationStep(
        domain="notes.task_activity",
        operation="upsert",
        object_id=activity_id,
        parent_id=NOTE_ID,
        payload={"activity_id": activity_id, "task_id": task_id, "note_id": NOTE_ID},
        client_envelope_id=activity_envelope_id,
        object_revision=1,
    )
    actor = TaskActor(actor_type="user", actor_id=OWNER_ID)
    return NotesTaskCaptureMutation(
        owner_user_id=OWNER_ID,
        dataset_id=DATASET_ID,
        actor=actor,
        operation="upsert",
        before={"canonical_revision": 1, "canonical_hash": "sha256:" + "0" * 64},
        after={
            "id": task_id,
            "note_id": NOTE_ID,
            "canonical_revision": 2,
            "canonical_hash": task_hash,
            "projection_status": projection_status,
            "deleted": False,
        },
        base_revision=1,
        base_hash="sha256:" + "0" * 64,
        restore_intent=False,
        idempotency_key=f"task-mutation-{index}",
        step=task_step,
        activity=NotesTaskActivityCapture(
            payload=cast(Any, object()),
            step=activity_step,
        ),
    )


def _note_step(*, envelope_id: str | None = None) -> ServerOriginMutationStep:
    return ServerOriginMutationStep(
        domain="notes.note",
        operation="upsert",
        object_id=NOTE_ID,
        payload={"title": "Tasks", "content": "- [ ] Task 2\n"},
        client_envelope_id=envelope_id,
        object_revision=4,
    )


def test_task_only_plan_is_stable_and_keeps_task_activity_atomic() -> None:
    coordinator = NotesTaskCoordinator()
    mutation = _mutation()

    first = coordinator.plan_task_mutation(mutation)
    retry = coordinator.plan_task_mutation(mutation)

    assert first == retry
    assert first.idempotency_key == mutation.idempotency_key
    assert [step.domain for step in first.steps] == [
        "notes.task",
        "notes.task_activity",
    ]
    assert [step.parent_id for step in first.steps] == [NOTE_ID, NOTE_ID]
    assert TASK_PROJECTION_ROUTING_KEY not in first.steps[0].routing_metadata


def test_projection_plan_binds_exact_task_and_note_envelope_evidence() -> None:
    coordinator = NotesTaskCoordinator()
    mutation = _mutation()
    note_step = _note_step()

    plan = coordinator.plan_task_mutation(mutation, note_step=note_step)
    retry = coordinator.plan_task_mutation(mutation, note_step=note_step)

    assert plan == retry
    assert [step.domain for step in plan.steps] == [
        "notes.task",
        "notes.task_activity",
        "notes.note",
    ]
    planned_note = plan.steps[-1]
    assert planned_note.client_envelope_id is not None
    note_hash, _ = canonical_payload_hash(dict(planned_note.payload))
    raw_anchor = plan.steps[0].routing_metadata[TASK_PROJECTION_ROUTING_KEY]
    anchor = _validate_task_projection_group_metadata(cast(dict[str, object], raw_anchor))
    assert anchor.task_id == mutation.step.object_id
    assert anchor.task_envelope_id == mutation.step.client_envelope_id
    assert anchor.task_revision == mutation.after["canonical_revision"]
    assert anchor.task_hash == mutation.after["canonical_hash"]
    assert anchor.note_envelope_id == planned_note.client_envelope_id
    assert anchor.note_hash == note_hash
    assert anchor.marker_hash == task_marker_hash(
        TaskMarker(
            task_id=mutation.step.object_id,
            revision=2,
            object_hash=cast(str, mutation.after["canonical_hash"]),
        )
    )
    assert plan.steps[1].routing_metadata[TASK_PROJECTION_ROUTING_KEY] == raw_anchor
    assert TASK_PROJECTION_ROUTING_KEY not in planned_note.routing_metadata


def test_note_reconciliation_accepts_499_task_transitions() -> None:
    coordinator = NotesTaskCoordinator()

    plan = coordinator.plan_note_reconciliation(
        tuple(_mutation(index) for index in range(2, 501)),
        note_step=_note_step(envelope_id="notes-task-note-reconcile-1"),
        idempotency_key="note-reconcile-499",
    )

    assert len(plan.steps) == 999
    assert plan.steps[-1].domain == "notes.note"
    assert len({step.client_envelope_id for step in plan.steps}) == 999


def test_note_reconciliation_rejects_500_before_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = False

    def fail_if_called(**_kwargs: object) -> None:
        nonlocal captured
        captured = True

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Sync.v2.notes_task_coordinator.capture_server_origin_mutation_batch",
        fail_if_called,
    )
    coordinator = NotesTaskCoordinator(service=cast(Any, object()), user_id=OWNER_ID)

    with pytest.raises(SyncStoreError, match="notes_task_mutation_group_limit_exceeded"):
        coordinator.plan_note_reconciliation(
            tuple(_mutation(index) for index in range(2, 502)),
            note_step=_note_step(envelope_id="notes-task-note-reconcile-2"),
            idempotency_key="note-reconcile-500",
        )

    assert captured is False


def test_capture_submits_the_complete_plan_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[ServerOriginMutationStep, ...]] = []
    expected = object()

    def capture(**kwargs: object) -> object:
        calls.append(tuple(cast(tuple[ServerOriginMutationStep, ...], kwargs["steps"])))
        return expected

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Sync.v2.notes_task_coordinator.capture_server_origin_mutation_batch",
        capture,
    )
    coordinator = NotesTaskCoordinator(service=cast(Any, object()), user_id=OWNER_ID)
    plan = coordinator.plan_task_mutation(_mutation(), note_step=_note_step())

    result = coordinator.capture(plan, source="notes.tasks.rest")

    assert result is expected
    assert calls == [plan.steps]


def test_ready_server_capture_appends_task_and_activity_as_one_group(tmp_path: Path) -> None:
    note_db = CharactersRAGDB(tmp_path / "product.db", client_id=OWNER_ID)
    sync_store = SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync.db"))
    try:
        dataset = sync_store.get_or_create_default_personal_dataset(OWNER_ID)
        note_db.note_store.add_note("Tasks", "body", note_id=NOTE_ID)
        note_db.bind_local_task_graph_to_dataset(
            owner_user_id=OWNER_ID,
            target_dataset_id=dataset.dataset_id,
        )
        task = note_db.task_store.create_task(
            owner_user_id=OWNER_ID,
            dataset_id=dataset.dataset_id,
            task_id=_uuid(2),
            note_id=NOTE_ID,
            text="Task 2",
            projection_status="unlinked",
        )
        note_payload = {"title": "Tasks", "content": "body"}
        note_hash, note_size = canonical_payload_hash(note_payload)
        sync_store.insert_envelope(
            SyncEnvelopeCreate(
                dataset_id=dataset.dataset_id,
                client_envelope_id="note-existing-1",
                domain="notes.note",
                operation="upsert",
                object_id=NOTE_ID,
                device_id="server-origin",
                object_revision=1,
                payload=note_payload,
                payload_hash=note_hash,
                payload_size_bytes=note_size,
                created_at_client="2026-08-24T10:00:00+00:00",
                status="accepted",
                apply_status="applied",
                applied_at="2026-08-24T10:00:00+00:00",
            )
        )
        service = SyncV2Service(
            store=sync_store,
            adapters=SyncAdapterRegistry(
                [NotesTaskDomainAdapter(), NotesTaskActivityDomainAdapter()]
            ),
            materializers={
                "notes.task": NotesTaskMaterializer(note_db),
                "notes.task_activity": NotesTaskActivityMaterializer(note_db),
            },
            clock=lambda: "2026-08-24T10:01:00+00:00",
        )
        dataset = NotesTaskBootstrapper(note_db).bootstrap(
            service=service,
            dataset=dataset,
        )
        dataset = NotesTaskActivityBootstrapper(note_db).bootstrap(
            service=service,
            dataset=dataset,
        )
        sync_store.db.execute(
            "UPDATE sync_datasets SET domain_set_json = ? WHERE dataset_id = ?",
            (
                json.dumps([*dataset.domains, "notes.task", "notes.task_activity"]),
                dataset.dataset_id,
            ),
        )
        coordinator = NotesTaskCoordinator(service=service, user_id=OWNER_ID)
        task_service = NotesTaskService(
            task_coordinator_resolver=lambda **_kwargs: coordinator
        )
        updated = task_service.update_task(
            db=note_db,
            task_id=str(task["id"]),
            expected_task_version=int(task["version"]),
            expected_note_version=None,
            actor=TaskActor(actor_type="user", actor_id=OWNER_ID),
            metadata={"priority": "high"},
            record_only=True,
        )
        task_head = sync_store.get_current_head(
            dataset.dataset_id,
            "notes.task",
            str(task["id"]),
        )
        assert task_head is not None
        assert task_head.mutation_group_id is not None
        server_group = sync_store.list_mutation_group(
            dataset.dataset_id,
            task_head.mutation_group_id,
        )
        assert [item.domain for item in server_group] == [
            "notes.task",
            "notes.task_activity",
        ]
        assert len({item.mutation_group_id for item in server_group}) == 1
        assert all(item.apply_status == "applied" for item in server_group)
        assert server_group[1].payload["source_kind"] == "rest"
        parsed_after = parse_notes_task_v1(
            server_group[0].payload,
            owner_user_id=OWNER_ID,
        )
        assert note_db.task_store.verify_sync_task_postcondition(
            owner_user_id=OWNER_ID,
            dataset_id=dataset.dataset_id,
            payload=parsed_after,
            canonical_revision=int(updated["canonical_revision"]),
            canonical_hash=str(updated["canonical_hash"]),
            deleted=False,
        )

        updated = task_service.update_task(
            db=note_db,
            task_id=str(task["id"]),
            expected_task_version=int(updated["version"]),
            expected_note_version=None,
            actor=TaskActor(
                actor_type="user",
                actor_id=OWNER_ID,
                tool_name="notes.tasks.update",
            ),
            metadata={"priority": "medium"},
            record_only=True,
        )
        mcp_head = sync_store.get_current_head(
            dataset.dataset_id,
            "notes.task",
            str(task["id"]),
        )
        assert mcp_head is not None
        assert mcp_head.mutation_group_id is not None
        mcp_group = sync_store.list_mutation_group(
            dataset.dataset_id,
            mcp_head.mutation_group_id,
        )
        assert mcp_group[1].payload["source_kind"] == "mcp"

        device_id = "22222222-2222-4222-8222-222222222222"
        service.register_device(
            user_id=OWNER_ID,
            display_name="Task client",
            client_type="chatbook",
            device_id=device_id,
        )
        task_head = sync_store.get_current_head(
            dataset.dataset_id,
            "notes.task",
            str(task["id"]),
        )
        assert task_head is not None
        client_payload = parse_notes_task_v1(
            {**dict(task_head.payload), "priority": "low"},
            owner_user_id=OWNER_ID,
        )
        client_revision = int(task_head.object_revision or 0) + 1
        client_envelope = SyncEnvelopeCreate(
            dataset_id=dataset.dataset_id,
            client_envelope_id="client-task-transition-1",
            domain="notes.task",
            operation="upsert",
            object_id=str(task["id"]),
            parent_id=NOTE_ID,
            device_id=device_id,
            base_server_cursor=task_head.server_cursor,
            base_object_revision=task_head.object_revision,
            base_object_hash=task_head.payload_hash,
            object_revision=client_revision,
            entity_version=client_revision,
            payload=client_payload.model_dump(mode="json"),
            payload_hash=notes_task_object_hash(
                client_payload,
                revision=client_revision,
                deleted=False,
            ),
            created_at_client="2026-08-24T10:02:00+00:00",
            encryption_metadata={"policy": "server_trusted_v1"},
        )

        active_dataset = sync_store.get_dataset(dataset.dataset_id)
        client_device = sync_store.get_device(OWNER_ID, device_id)
        assert active_dataset is not None
        assert client_device is not None
        expanded = service._expand_task_client_push(
            dataset=active_dataset,
            device=client_device,
            envelope=client_envelope,
        )
        assert [item.domain for item in expanded] == [
            "notes.task",
            "notes.task_activity",
        ]

        activity_materializer = service.materializers["notes.task_activity"]

        class _FailActivityOnce:
            failed = False

            def apply(self, envelope: Any, *, store: SyncV2Store) -> MaterializationResult:
                if not self.failed:
                    self.failed = True
                    store.mark_envelope_apply_status(
                        envelope.server_cursor,
                        apply_status="failed",
                        apply_error_code="injected_split",
                        apply_error_message="retry same group",
                    )
                    return MaterializationResult(
                        status="failed",
                        error_code="injected_split",
                    )
                return activity_materializer.apply(envelope, store=store)

        service.materializers["notes.task_activity"] = _FailActivityOnce()
        split_result = service.push(
            user_id=OWNER_ID,
            dataset_id=dataset.dataset_id,
            device_id=device_id,
            envelopes=[client_envelope],
        )

        assert split_result.accepted == []
        assert [item.error_code for item in split_result.rejected] == [
            "sync_projection_failed"
        ]
        assert split_result.rejected[0].retryable is True
        client_task_head = sync_store.get_current_head(
            dataset.dataset_id,
            "notes.task",
            str(task["id"]),
        )
        assert client_task_head is not None
        assert client_task_head.mutation_group_id is not None
        client_group = sync_store.list_mutation_group(
            dataset.dataset_id,
            client_task_head.mutation_group_id,
        )
        assert [item.domain for item in client_group] == [
            "notes.task",
            "notes.task_activity",
        ]
        assert [item.apply_status for item in client_group] == ["applied", "failed"]

        replay = service.push(
            user_id=OWNER_ID,
            dataset_id=dataset.dataset_id,
            device_id=device_id,
            envelopes=[client_envelope],
        )
        assert replay.rejected == []
        assert replay.conflicts == []
        assert len(replay.accepted) == 1
        assert len(
            sync_store.list_mutation_group(
                dataset.dataset_id,
                client_task_head.mutation_group_id,
            )
        ) == 2
        assert all(
            item.apply_status == "applied"
            for item in sync_store.list_mutation_group(
                dataset.dataset_id,
                client_task_head.mutation_group_id,
            )
        )

        changed_reuse = service.push(
            user_id=OWNER_ID,
            dataset_id=dataset.dataset_id,
            device_id=device_id,
            envelopes=[replace(client_envelope, payload_hash="sha256:" + "f" * 64)],
        )
        assert changed_reuse.accepted == []
        assert [item.error_code for item in changed_reuse.rejected] == [
            "idempotency_conflict"
        ]
    finally:
        note_db.close_connection()


@pytest.mark.parametrize(
    "broken_steps",
    [
        lambda plan: (plan.steps[0],),
        lambda plan: (plan.steps[1], plan.steps[0]),
        lambda plan: (plan.steps[0], replace(plan.steps[1], parent_id=_uuid(999))),
        lambda plan: (plan.steps[0], plan.steps[1], plan.steps[1]),
    ],
)
def test_compound_plan_validation_rejects_partial_or_misordered_groups(
    broken_steps: Any,
) -> None:
    coordinator = NotesTaskCoordinator()
    valid = coordinator.plan_task_mutation(_mutation(), note_step=_note_step())

    with pytest.raises(SyncStoreError, match="notes_task_mutation_group_invalid"):
        coordinator.validate_plan(broken_steps(valid))
