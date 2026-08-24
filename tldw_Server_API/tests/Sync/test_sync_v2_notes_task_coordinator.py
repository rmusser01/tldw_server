"""Deterministic compound coordination for Notes task mutations."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
)
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Notes_Tasks.markdown_parser import parse_note_checklists
from tldw_Server_API.app.core.Notes_Tasks.models import TaskActor
from tldw_Server_API.app.core.Notes_Tasks.projection_markers import (
    TaskMarker,
    render_task_marker,
    task_marker_hash,
)
from tldw_Server_API.app.core.Notes_Tasks.service import (
    NotesTaskActivityCapture,
    NotesTaskCaptureMutation,
    NotesTaskService,
)
from tldw_Server_API.app.core.Sync.v2.adapters import SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.domain_adapters import (
    NotesDomainAdapter,
    NotesTaskActivityDomainAdapter,
    NotesTaskDomainAdapter,
)
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.materializers import (
    MaterializationResult,
    NotesMaterializer,
    NotesTaskActivityMaterializer,
    NotesTaskMaterializer,
)
from tldw_Server_API.app.core.Sync.v2.models import SyncEnvelopeCreate, SyncObjectState
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


def test_ready_server_capture_appends_task_and_activity_as_one_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        note_envelope = sync_store.insert_envelope(
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
        assert note_envelope.server_cursor is not None
        sync_store.upsert_object_state(
            SyncObjectState(
                dataset_id=dataset.dataset_id,
                domain="notes.note",
                object_id=NOTE_ID,
                object_revision=1,
                object_hash=note_hash,
                latest_server_cursor=note_envelope.server_cursor,
                deleted=False,
            )
        )
        service = SyncV2Service(
            store=sync_store,
            adapters=SyncAdapterRegistry(
                [
                    NotesDomainAdapter(),
                    NotesTaskDomainAdapter(),
                    NotesTaskActivityDomainAdapter(),
                ]
            ),
            materializers={
                "notes.note": NotesMaterializer(note_db),
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
        coordinator = NotesTaskCoordinator(
            service=service,
            user_id=OWNER_ID,
            dataset_id=dataset.dataset_id,
        )
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

        marker = render_task_marker(
            str(updated["id"]),
            revision=int(updated["canonical_revision"]),
            object_hash=str(updated["canonical_hash"]),
        )
        note_db.update_note(
            note_id=NOTE_ID,
            update_data={"content": f"- [ ] Task 2 @priority(medium) {marker}\n"},
            expected_version=1,
        )
        projected_note = note_db.get_note_by_id(NOTE_ID)
        assert projected_note is not None
        projected_item = parse_note_checklists(
            note_id=NOTE_ID,
            note_version=int(projected_note["version"]),
            content=str(projected_note["content"]),
        ).items[0]
        note_db.set_task_projection(
            owner_user_id=OWNER_ID,
            dataset_id=dataset.dataset_id,
            task_id=str(updated["id"]),
            note_id=NOTE_ID,
            note_version=int(projected_note["version"]),
            line_number=projected_item.locator.line_number,
            start_offset=projected_item.locator.start_offset,
            end_offset=projected_item.locator.end_offset,
            normalized_text_hash=projected_item.locator.normalized_text_hash,
            occurrence_index=projected_item.locator.occurrence_index,
            block_fingerprint=projected_item.locator.block_fingerprint,
            raw_line=projected_item.raw_line,
            has_child_content=projected_item.has_child_content,
        )
        projected_task = note_db.get_task(
            owner_user_id=OWNER_ID,
            dataset_id=dataset.dataset_id,
            task_id=str(updated["id"]),
        )
        assert projected_task is not None
        active_dataset = sync_store.get_dataset(dataset.dataset_id)
        assert active_dataset is not None
        active_metadata = dict(active_dataset.metadata)
        active_metadata["notes_organization_v1"] = {"state": "ready"}
        sync_store.db.execute(
            "UPDATE sync_datasets SET metadata_json = ? WHERE dataset_id = ?",
            (json.dumps(active_metadata), dataset.dataset_id),
        )

        completed = task_service.update_task(
            db=note_db,
            task_id=str(projected_task["id"]),
            expected_task_version=int(projected_task["version"]),
            expected_note_version=int(projected_note["version"]),
            actor=TaskActor(actor_type="user", actor_id=OWNER_ID),
            status="done",
        )
        completed_note = note_db.get_note_by_id(NOTE_ID)
        assert completed_note is not None
        assert completed["status"] == "done"
        assert "- [x] Task 2 @priority(medium)" in completed_note["content"]
        completed_marker = parse_note_checklists(
            note_id=NOTE_ID,
            note_version=int(completed_note["version"]),
            content=str(completed_note["content"]),
        ).items[0].marker
        assert completed_marker is not None
        assert completed_marker.revision == completed["canonical_revision"]
        assert completed_marker.object_hash == completed["canonical_hash"]

        device_id = "22222222-2222-4222-8222-222222222222"
        service.register_device(
            user_id=OWNER_ID,
            display_name="Task client",
            client_type="chatbook",
            device_id=device_id,
        )
        unrelated_content = f"Context preserved\n\n{completed_note['content']}"
        unrelated_capture = coordinator.capture_note_projection(
            task_service._note_projection_step(
                coordinator=coordinator,
                note=completed_note,
                content=unrelated_content,
            ),
            idempotency_key="unrelated-note-edit-before-client-task-push",
        )
        assert unrelated_capture.fully_applied is True
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
        unrelated_note = note_db.get_note_by_id(NOTE_ID)
        assert unrelated_note is not None
        duplicate_content = (
            f"{unrelated_note['content']}"
            f"- [ ] Stale duplicate {render_task_marker(str(task['id']), revision=999, object_hash='sha256:' + 'f' * 64)}\n"
        )
        duplicate_capture = coordinator.capture_note_projection(
            task_service._note_projection_step(
                coordinator=coordinator,
                note=unrelated_note,
                content=duplicate_content,
            ),
            idempotency_key="duplicate-task-marker-before-client-push",
        )
        assert duplicate_capture.fully_applied is True
        with pytest.raises(
            SyncStoreError,
            match="notes_task_projection_base_invalid",
        ):
            service._expand_task_client_push(
                dataset=active_dataset,
                device=client_device,
                envelope=client_envelope,
            )
        duplicate_note = note_db.get_note_by_id(NOTE_ID)
        assert duplicate_note is not None
        corrected_capture = coordinator.capture_note_projection(
            task_service._note_projection_step(
                coordinator=coordinator,
                note=duplicate_note,
                content=unrelated_content,
            ),
            idempotency_key="remove-duplicate-task-marker-before-client-push",
        )
        assert corrected_capture.fully_applied is True
        expanded = service._expand_task_client_push(
            dataset=active_dataset,
            device=client_device,
            envelope=client_envelope,
        )
        assert [item.domain for item in expanded] == [
            "notes.task",
            "notes.task_activity",
            "notes.note",
        ]

        activity_materializer = service.materializers["notes.task_activity"]

        class _FailActivityOnce:
            failed = False
            note_db = activity_materializer.note_db

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
            "notes.note",
        ]
        assert [item.apply_status for item in client_group] == [
            "applied",
            "failed",
            "pending",
        ]

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
        ) == 3
        assert all(
            item.apply_status == "applied"
            for item in sync_store.list_mutation_group(
                dataset.dataset_id,
                client_task_head.mutation_group_id,
            )
        )
        client_note = note_db.get_note_by_id(NOTE_ID)
        assert client_note is not None
        assert "Context preserved" in client_note["content"]
        client_item = parse_note_checklists(
            note_id=NOTE_ID,
            note_version=int(client_note["version"]),
            content=str(client_note["content"]),
        ).items[0]
        assert client_item.metadata["priority"] == "low"
        assert client_item.marker is not None
        assert client_item.marker.revision == client_revision

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

        client_live_head = sync_store.get_current_head(
            dataset.dataset_id,
            "notes.task",
            str(task["id"]),
        )
        assert client_live_head is not None
        client_tombstone_revision = int(client_live_head.object_revision or 0) + 1
        client_tombstone = SyncEnvelopeCreate(
            dataset_id=dataset.dataset_id,
            client_envelope_id="client-task-tombstone-1",
            domain="notes.task",
            operation="tombstone",
            object_id=str(task["id"]),
            parent_id=NOTE_ID,
            device_id=device_id,
            base_server_cursor=client_live_head.server_cursor,
            base_object_revision=client_live_head.object_revision,
            base_object_hash=client_live_head.payload_hash,
            object_revision=client_tombstone_revision,
            entity_version=client_tombstone_revision,
            payload=dict(client_live_head.payload),
            payload_hash=notes_task_object_hash(
                parse_notes_task_v1(
                    client_live_head.payload,
                    owner_user_id=OWNER_ID,
                ),
                revision=client_tombstone_revision,
                deleted=True,
            ),
            created_at_client="2026-08-24T10:03:00+00:00",
            encryption_metadata={"policy": "server_trusted_v1"},
            deleted=True,
        )
        client_deleted = service.push(
            user_id=OWNER_ID,
            dataset_id=dataset.dataset_id,
            device_id=device_id,
            envelopes=[client_tombstone],
        )
        assert client_deleted.rejected == []
        assert client_deleted.conflicts == []
        client_deleted_note = note_db.get_note_by_id(NOTE_ID)
        assert client_deleted_note is not None
        assert parse_note_checklists(
            note_id=NOTE_ID,
            note_version=int(client_deleted_note["version"]),
            content=str(client_deleted_note["content"]),
        ).items == []

        client_deleted_head = sync_store.get_current_head(
            dataset.dataset_id,
            "notes.task",
            str(task["id"]),
        )
        assert client_deleted_head is not None
        client_restore_revision = int(client_deleted_head.object_revision or 0) + 1
        client_restore = SyncEnvelopeCreate(
            dataset_id=dataset.dataset_id,
            client_envelope_id="client-task-restore-1",
            domain="notes.task",
            operation="upsert",
            object_id=str(task["id"]),
            parent_id=NOTE_ID,
            device_id=device_id,
            base_server_cursor=client_deleted_head.server_cursor,
            base_object_revision=client_deleted_head.object_revision,
            base_object_hash=client_deleted_head.payload_hash,
            object_revision=client_restore_revision,
            entity_version=client_restore_revision,
            payload=dict(client_deleted_head.payload),
            payload_hash=notes_task_object_hash(
                parse_notes_task_v1(
                    client_deleted_head.payload,
                    owner_user_id=OWNER_ID,
                ),
                revision=client_restore_revision,
                deleted=False,
            ),
            created_at_client="2026-08-24T10:04:00+00:00",
            encryption_metadata={"policy": "server_trusted_v1"},
            routing_metadata={"restore_intent": True},
        )
        client_restored = service.push(
            user_id=OWNER_ID,
            dataset_id=dataset.dataset_id,
            device_id=device_id,
            envelopes=[client_restore],
        )
        assert client_restored.rejected == []
        client_note = note_db.get_note_by_id(NOTE_ID)
        assert client_note is not None
        client_item = parse_note_checklists(
            note_id=NOTE_ID,
            note_version=int(client_note["version"]),
            content=str(client_note["content"]),
        ).items[0]
        assert client_item.marker is not None
        assert client_item.marker.revision == client_restore_revision

        note_edit_marker = client_item.marker
        assert note_edit_marker is not None
        note_db.update_note(
            note_id=NOTE_ID,
            update_data={
                "content": (
                    "- [ ] Markdown edit @priority(low) "
                    + render_task_marker(
                        note_edit_marker.task_id,
                        revision=note_edit_marker.revision,
                        object_hash=note_edit_marker.object_hash,
                    )
                    + "\n"
                )
            },
            expected_version=int(client_note["version"]),
        )
        markdown_note = note_db.get_note_by_id(NOTE_ID)
        assert markdown_note is not None
        markdown_result = task_service.reconcile_note(
            db=note_db,
            note_id=NOTE_ID,
            note_version=int(markdown_note["version"]),
            content=str(markdown_note["content"]),
            actor=TaskActor(actor_type="user", actor_id=OWNER_ID),
            owner_user_id=OWNER_ID,
        )
        markdown_task = note_db.get_task(
            owner_user_id=OWNER_ID,
            dataset_id=dataset.dataset_id,
            task_id=str(task["id"]),
        )
        assert markdown_result.updated_count == 1
        assert markdown_task is not None
        assert markdown_task["text"] == "Markdown edit"
        reconciled_note = note_db.get_note_by_id(NOTE_ID)
        assert reconciled_note is not None
        reconciled_item = parse_note_checklists(
            note_id=NOTE_ID,
            note_version=int(reconciled_note["version"]),
            content=str(reconciled_note["content"]),
        ).items[0]
        assert reconciled_item.marker is not None
        assert reconciled_item.marker.revision == markdown_task["canonical_revision"]

        current_task = note_db.get_task(
            owner_user_id=OWNER_ID,
            dataset_id=dataset.dataset_id,
            task_id=str(task["id"]),
        )
        current_note = note_db.get_note_by_id(NOTE_ID)
        assert current_task is not None
        assert current_note is not None
        deleted = task_service.delete_task(
            db=note_db,
            task_id=str(task["id"]),
            expected_task_version=int(current_task["version"]),
            expected_note_version=int(current_note["version"]),
            record_only=False,
            actor=TaskActor(actor_type="user", actor_id=OWNER_ID),
        )
        deleted_head = sync_store.get_current_head(
            dataset.dataset_id,
            "notes.task",
            str(task["id"]),
        )
        assert bool(deleted["deleted"]) is True
        assert deleted_head is not None
        assert deleted_head.operation == "tombstone"
        deleted_note = note_db.get_note_by_id(NOTE_ID)
        assert deleted_note is not None
        assert parse_note_checklists(
            note_id=NOTE_ID,
            note_version=int(deleted_note["version"]),
            content=str(deleted_note["content"]),
        ).items == []

        restored = task_service.restore_task(
            db=note_db,
            task_id=str(task["id"]),
            expected_task_version=int(deleted["version"]),
            expected_note_version=int(deleted_note["version"]),
            expected_base_server_cursor=int(deleted_head.server_cursor or 0),
            expected_base_revision=int(deleted_head.object_revision or 0),
            expected_base_hash=str(deleted_head.payload_hash),
            actor=TaskActor(actor_type="user", actor_id=OWNER_ID),
            owner_user_id=OWNER_ID,
        )
        restored_note = note_db.get_note_by_id(NOTE_ID)
        restored_head = sync_store.get_current_head(
            dataset.dataset_id,
            "notes.task",
            str(task["id"]),
        )
        assert restored_note is not None
        assert restored_head is not None
        assert restored_head.routing_metadata["restore_intent"] is True
        assert restored["projection_status"] == "live"
        assert len(
            [
                item
                for item in parse_note_checklists(
                    note_id=NOTE_ID,
                    note_version=int(restored_note["version"]),
                    content=str(restored_note["content"]),
                ).items
                if item.marker is not None and item.marker.task_id == str(task["id"])
            ]
        ) == 1
        with pytest.raises(ConflictError):
            task_service.restore_task(
                db=note_db,
                task_id=str(task["id"]),
                expected_task_version=int(deleted["version"]),
                expected_note_version=int(deleted_note["version"]),
                expected_base_server_cursor=int(deleted_head.server_cursor or 0),
                expected_base_revision=int(deleted_head.object_revision or 0),
                expected_base_hash=str(deleted_head.payload_hash),
                actor=TaskActor(actor_type="user", actor_id=OWNER_ID),
                owner_user_id=OWNER_ID,
            )

        note_db.update_note(
            note_id=NOTE_ID,
            update_data={"content": "\n"},
            expected_version=int(restored_note["version"]),
        )
        restore_removed_note = note_db.get_note_by_id(NOTE_ID)
        assert restore_removed_note is not None
        task_service.reconcile_note(
            db=note_db,
            note_id=NOTE_ID,
            note_version=int(restore_removed_note["version"]),
            content=str(restore_removed_note["content"]),
            actor=TaskActor(actor_type="user", actor_id=OWNER_ID),
            owner_user_id=OWNER_ID,
        )
        restore_unlinked = note_db.get_task(
            owner_user_id=OWNER_ID,
            dataset_id=dataset.dataset_id,
            task_id=str(task["id"]),
        )
        relink_note = note_db.get_note_by_id(NOTE_ID)
        assert restore_unlinked is not None
        assert relink_note is not None
        assert restore_unlinked["projection_status"] == "unlinked"
        relinked = task_service.relink_task(
            db=note_db,
            task_id=str(task["id"]),
            note_id=NOTE_ID,
            expected_task_version=int(restore_unlinked["version"]),
            expected_note_version=int(relink_note["version"]),
            actor=TaskActor(actor_type="user", actor_id=OWNER_ID),
            owner_user_id=OWNER_ID,
        )
        after_relink_note = note_db.get_note_by_id(NOTE_ID)
        assert relinked["id"] == restored["id"]
        assert relinked["projection_status"] == "live"
        assert after_relink_note is not None
        final_deleted = task_service.delete_task(
            db=note_db,
            task_id=str(task["id"]),
            expected_task_version=int(relinked["version"]),
            expected_note_version=int(after_relink_note["version"]),
            record_only=False,
            actor=TaskActor(actor_type="user", actor_id=OWNER_ID),
            owner_user_id=OWNER_ID,
        )
        deleted_note = note_db.get_note_by_id(NOTE_ID)
        assert final_deleted["id"] == restored["id"]
        assert deleted_note is not None

        created = task_service.create_task_for_note(
            db=note_db,
            note_id=NOTE_ID,
            text="Replacement",
            status="open",
            metadata={"estimate": "2h"},
            expected_note_version=int(deleted_note["version"]),
            actor=TaskActor(actor_type="user", actor_id=OWNER_ID),
        )
        created_head = sync_store.get_current_head(
            dataset.dataset_id,
            "notes.task",
            str(created["id"]),
        )
        assert created_head is not None
        assert created_head.object_revision == 1
        created_note = note_db.get_note_by_id(NOTE_ID)
        assert created_note is not None
        created_item = parse_note_checklists(
            note_id=NOTE_ID,
            note_version=int(created_note["version"]),
            content=str(created_note["content"]),
        ).items[0]
        assert created_item.text == "Replacement"
        assert created_item.marker is not None
        assert created_item.marker.task_id == created["id"]

        note_db.update_note(
            note_id=NOTE_ID,
            update_data={"content": "- [ ] Replacement renamed without marker\n"},
            expected_version=int(created_note["version"]),
        )
        missing_marker_note = note_db.get_note_by_id(NOTE_ID)
        assert missing_marker_note is not None
        missing_marker_result = task_service.reconcile_note(
            db=note_db,
            note_id=NOTE_ID,
            note_version=int(missing_marker_note["version"]),
            content=str(missing_marker_note["content"]),
            actor=TaskActor(actor_type="user", actor_id=OWNER_ID),
            owner_user_id=OWNER_ID,
        )
        missing_marker_drifts = note_db.task_store.list_task_projection_drifts(
            owner_user_id=OWNER_ID,
            dataset_id=dataset.dataset_id,
            note_id=NOTE_ID,
            task_id=str(created["id"]),
        )
        missing_marker_task = note_db.get_task(
            owner_user_id=OWNER_ID,
            dataset_id=dataset.dataset_id,
            task_id=str(created["id"]),
        )
        assert missing_marker_result.unlinked_count == 0
        assert missing_marker_result.warning_count == 1
        assert missing_marker_task is not None
        assert missing_marker_task["projection_status"] == "live"
        assert len(missing_marker_drifts) == 1
        assert missing_marker_drifts[0]["reason_code"] == "missing_marker_base"

        missing_marker_drift = missing_marker_drifts[0]
        original_drift_cas = note_db.task_store.compare_and_set_task_projection_drift
        fail_drift_cas_once = True

        def injected_drift_cas(**kwargs: Any) -> dict[str, Any]:
            nonlocal fail_drift_cas_once
            if fail_drift_cas_once:
                fail_drift_cas_once = False
                raise RuntimeError("injected crash after drift resolution capture")
            return original_drift_cas(**kwargs)

        monkeypatch.setattr(
            note_db.task_store,
            "compare_and_set_task_projection_drift",
            injected_drift_cas,
        )
        with pytest.raises(
            RuntimeError,
            match="injected crash after drift resolution capture",
        ):
            task_service.resolve_projection_drift(
                db=note_db,
                note_id=NOTE_ID,
                task_id=str(created["id"]),
                drift_id=str(missing_marker_drift["id"]),
                action="keep_task",
                expected_lifecycle_revision=1,
                expected_note_head_cursor=missing_marker_drift["note_head_cursor"],
                expected_note_head_hash=missing_marker_drift["note_head_hash"],
                expected_task_head_cursor=missing_marker_drift["task_head_cursor"],
                expected_task_head_hash=missing_marker_drift["task_head_hash"],
                actor=TaskActor(actor_type="user", actor_id=OWNER_ID),
                owner_user_id=OWNER_ID,
            )
        repaired_drift = task_service.resolve_projection_drift(
            db=note_db,
            note_id=NOTE_ID,
            task_id=str(created["id"]),
            drift_id=str(missing_marker_drift["id"]),
            action="keep_task",
            expected_lifecycle_revision=1,
            expected_note_head_cursor=missing_marker_drift["note_head_cursor"],
            expected_note_head_hash=missing_marker_drift["note_head_hash"],
            expected_task_head_cursor=missing_marker_drift["task_head_cursor"],
            expected_task_head_hash=missing_marker_drift["task_head_hash"],
            actor=TaskActor(actor_type="user", actor_id=OWNER_ID),
            owner_user_id=OWNER_ID,
        )
        assert repaired_drift["status"] == "resolved"
        repaired_missing_note = note_db.get_note_by_id(NOTE_ID)
        assert repaired_missing_note is not None
        repaired_missing_item = parse_note_checklists(
            note_id=NOTE_ID,
            note_version=int(repaired_missing_note["version"]),
            content=str(repaired_missing_note["content"]),
        ).items[0]
        assert repaired_missing_item.marker is not None
        assert repaired_missing_item.text == "Replacement"

        note_db.update_note(
            note_id=NOTE_ID,
            update_data={
                "content": (
                    "- [ ] Accepted malformed edit "
                    "<!-- tldw-task:v1:not-a-task:1:not-a-hash -->\n"
                )
            },
            expected_version=int(repaired_missing_note["version"]),
        )
        malformed_marker_note = note_db.get_note_by_id(NOTE_ID)
        assert malformed_marker_note is not None
        malformed_marker_result = task_service.reconcile_note(
            db=note_db,
            note_id=NOTE_ID,
            note_version=int(malformed_marker_note["version"]),
            content=str(malformed_marker_note["content"]),
            actor=TaskActor(actor_type="user", actor_id=OWNER_ID),
            owner_user_id=OWNER_ID,
        )
        malformed_marker_drifts = note_db.task_store.list_task_projection_drifts(
            owner_user_id=OWNER_ID,
            dataset_id=dataset.dataset_id,
            note_id=NOTE_ID,
            task_id=str(created["id"]),
        )
        assert malformed_marker_result.unlinked_count == 0
        assert malformed_marker_result.warning_count == 1
        assert len(malformed_marker_drifts) == 1
        assert malformed_marker_drifts[0]["reason_code"] == "malformed_marker"

        malformed_marker_drift = malformed_marker_drifts[0]
        task_service.resolve_projection_drift(
            db=note_db,
            note_id=NOTE_ID,
            task_id=str(created["id"]),
            drift_id=str(malformed_marker_drift["id"]),
            action="accept_markdown",
            expected_lifecycle_revision=1,
            expected_note_head_cursor=malformed_marker_drift["note_head_cursor"],
            expected_note_head_hash=malformed_marker_drift["note_head_hash"],
            expected_task_head_cursor=malformed_marker_drift["task_head_cursor"],
            expected_task_head_hash=malformed_marker_drift["task_head_hash"],
            actor=TaskActor(actor_type="user", actor_id=OWNER_ID),
            owner_user_id=OWNER_ID,
        )
        repaired_malformed_note = note_db.get_note_by_id(NOTE_ID)
        repaired_malformed_task = note_db.get_task(
            owner_user_id=OWNER_ID,
            dataset_id=dataset.dataset_id,
            task_id=str(created["id"]),
        )
        assert repaired_malformed_note is not None
        assert repaired_malformed_task is not None
        assert repaired_malformed_task["text"] == "Accepted malformed edit"

        note_db.update_note(
            note_id=NOTE_ID,
            update_data={"content": "\n"},
            expected_version=int(repaired_malformed_note["version"]),
        )
        removed_note = note_db.get_note_by_id(NOTE_ID)
        assert removed_note is not None
        unlink_result = task_service.reconcile_note(
            db=note_db,
            note_id=NOTE_ID,
            note_version=int(removed_note["version"]),
            content=str(removed_note["content"]),
            actor=TaskActor(actor_type="user", actor_id=OWNER_ID),
            owner_user_id=OWNER_ID,
        )
        unlinked = note_db.get_task(
            owner_user_id=OWNER_ID,
            dataset_id=dataset.dataset_id,
            task_id=str(created["id"]),
        )
        assert unlink_result.unlinked_task_ids == [created["id"]]
        assert unlinked is not None
        assert unlinked["projection_status"] == "unlinked"

        conflict_note = note_db.get_note_by_id(NOTE_ID)
        assert conflict_note is not None
        conflict_task = task_service.create_task_for_note(
            db=note_db,
            note_id=NOTE_ID,
            text="Conflict base",
            status="open",
            metadata={},
            expected_note_version=int(conflict_note["version"]),
            actor=TaskActor(actor_type="user", actor_id=OWNER_ID),
        )
        conflict_base_note = note_db.get_note_by_id(NOTE_ID)
        assert conflict_base_note is not None
        conflict_base_item = parse_note_checklists(
            note_id=NOTE_ID,
            note_version=int(conflict_base_note["version"]),
            content=str(conflict_base_note["content"]),
        ).items[0]
        conflict_base_marker = conflict_base_item.marker
        assert conflict_base_marker is not None
        explicit = task_service.update_task(
            db=note_db,
            task_id=str(conflict_task["id"]),
            expected_task_version=int(conflict_task["version"]),
            expected_note_version=int(conflict_base_note["version"]),
            actor=TaskActor(actor_type="user", actor_id=OWNER_ID),
            text="Explicit task edit",
        )
        explicit_note = note_db.get_note_by_id(NOTE_ID)
        assert explicit_note is not None
        note_db.update_note(
            note_id=NOTE_ID,
            update_data={
                "content": (
                    "- [ ] Markdown conflict "
                    + render_task_marker(
                        conflict_base_marker.task_id,
                        revision=conflict_base_marker.revision,
                        object_hash=conflict_base_marker.object_hash,
                    )
                    + "\n"
                )
            },
            expected_version=int(explicit_note["version"]),
        )
        drift_note = note_db.get_note_by_id(NOTE_ID)
        assert drift_note is not None
        drift_result = task_service.reconcile_note(
            db=note_db,
            note_id=NOTE_ID,
            note_version=int(drift_note["version"]),
            content=str(drift_note["content"]),
            actor=TaskActor(actor_type="user", actor_id=OWNER_ID),
            owner_user_id=OWNER_ID,
        )
        drifts = note_db.task_store.list_task_projection_drifts(
            owner_user_id=OWNER_ID,
            dataset_id=dataset.dataset_id,
            note_id=NOTE_ID,
            task_id=str(conflict_task["id"]),
        )
        preserved = note_db.get_task(
            owner_user_id=OWNER_ID,
            dataset_id=dataset.dataset_id,
            task_id=str(conflict_task["id"]),
        )
        assert drift_result.warning_count == 1
        assert len(drifts) == 1
        assert drifts[0]["reason_code"] == "both_changed"
        assert "content" not in drifts[0]
        assert "title" not in drifts[0]
        assert preserved is not None
        assert preserved["text"] == explicit["text"] == "Explicit task edit"
        assert "Markdown conflict" in drift_note["content"]

        drift = drifts[0]
        resolved = task_service.resolve_projection_drift(
            db=note_db,
            note_id=NOTE_ID,
            task_id=str(conflict_task["id"]),
            drift_id=str(drift["id"]),
            action="keep_task",
            expected_lifecycle_revision=1,
            expected_note_head_cursor=drift["note_head_cursor"],
            expected_note_head_hash=drift["note_head_hash"],
            expected_task_head_cursor=drift["task_head_cursor"],
            expected_task_head_hash=drift["task_head_hash"],
            actor=TaskActor(actor_type="user", actor_id=OWNER_ID),
            owner_user_id=OWNER_ID,
        )
        resolved_note = note_db.get_note_by_id(NOTE_ID)
        assert resolved["status"] == "resolved"
        assert resolved_note is not None
        assert "Explicit task edit" in resolved_note["content"]
        assert "Markdown conflict" not in resolved_note["content"]
        with pytest.raises(ConflictError):
            task_service.resolve_projection_drift(
                db=note_db,
                note_id=NOTE_ID,
                task_id=str(conflict_task["id"]),
                drift_id=str(drift["id"]),
                action="keep_task",
                expected_lifecycle_revision=1,
                expected_note_head_cursor=drift["note_head_cursor"],
                expected_note_head_hash=drift["note_head_hash"],
                expected_task_head_cursor=drift["task_head_cursor"],
                expected_task_head_hash=drift["task_head_hash"],
                actor=TaskActor(actor_type="user", actor_id=OWNER_ID),
                owner_user_id=OWNER_ID,
            )

        def create_conflict(
            base_task: dict[str, Any],
            *,
            explicit_text: str,
            markdown_text: str,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            base_note = note_db.get_note_by_id(NOTE_ID)
            assert base_note is not None
            base_items = [
                item
                for item in parse_note_checklists(
                    note_id=NOTE_ID,
                    note_version=int(base_note["version"]),
                    content=str(base_note["content"]),
                ).items
                if item.marker is not None
                and item.marker.task_id == str(base_task["id"])
            ]
            assert len(base_items) == 1
            base_marker = base_items[0].marker
            assert base_marker is not None
            explicit_task = task_service.update_task(
                db=note_db,
                task_id=str(base_task["id"]),
                expected_task_version=int(base_task["version"]),
                expected_note_version=int(base_note["version"]),
                actor=TaskActor(actor_type="user", actor_id=OWNER_ID),
                text=explicit_text,
            )
            after_explicit = note_db.get_note_by_id(NOTE_ID)
            assert after_explicit is not None
            note_db.update_note(
                note_id=NOTE_ID,
                update_data={
                    "content": (
                        f"- [ ] {markdown_text} "
                        + render_task_marker(
                            base_marker.task_id,
                            revision=base_marker.revision,
                            object_hash=base_marker.object_hash,
                        )
                        + "\n"
                    )
                },
                expected_version=int(after_explicit["version"]),
            )
            conflict_note = note_db.get_note_by_id(NOTE_ID)
            assert conflict_note is not None
            task_service.reconcile_note(
                db=note_db,
                note_id=NOTE_ID,
                note_version=int(conflict_note["version"]),
                content=str(conflict_note["content"]),
                actor=TaskActor(actor_type="user", actor_id=OWNER_ID),
                owner_user_id=OWNER_ID,
            )
            open_drifts = note_db.task_store.list_task_projection_drifts(
                owner_user_id=OWNER_ID,
                dataset_id=dataset.dataset_id,
                note_id=NOTE_ID,
                task_id=str(base_task["id"]),
            )
            assert len(open_drifts) == 1
            return explicit_task, open_drifts[0]

        keep_task = note_db.get_task(
            owner_user_id=OWNER_ID,
            dataset_id=dataset.dataset_id,
            task_id=str(conflict_task["id"]),
        )
        assert keep_task is not None
        _, accept_drift = create_conflict(
            keep_task,
            explicit_text="Explicit second edit",
            markdown_text="Accepted Markdown edit",
        )
        task_materializer = service.materializers["notes.task"]

        class _InjectedProcessExit(BaseException):
            pass

        class _CrashBeforeTaskMaterialization:
            note_db = task_materializer.note_db

            def apply(
                self,
                envelope: Any,
                *,
                store: SyncV2Store,
            ) -> MaterializationResult:
                raise _InjectedProcessExit

        service.materializers["notes.task"] = _CrashBeforeTaskMaterialization()
        try:
            with pytest.raises(_InjectedProcessExit):
                task_service.resolve_projection_drift(
                    db=note_db,
                    note_id=NOTE_ID,
                    task_id=str(keep_task["id"]),
                    drift_id=str(accept_drift["id"]),
                    action="accept_markdown",
                    expected_lifecycle_revision=1,
                    expected_note_head_cursor=accept_drift["note_head_cursor"],
                    expected_note_head_hash=accept_drift["note_head_hash"],
                    expected_task_head_cursor=accept_drift["task_head_cursor"],
                    expected_task_head_hash=accept_drift["task_head_hash"],
                    actor=TaskActor(actor_type="user", actor_id=OWNER_ID),
                    owner_user_id=OWNER_ID,
                )
        finally:
            service.materializers["notes.task"] = task_materializer
        accepted = task_service.resolve_projection_drift(
            db=note_db,
            note_id=NOTE_ID,
            task_id=str(keep_task["id"]),
            drift_id=str(accept_drift["id"]),
            action="accept_markdown",
            expected_lifecycle_revision=1,
            expected_note_head_cursor=accept_drift["note_head_cursor"],
            expected_note_head_hash=accept_drift["note_head_hash"],
            expected_task_head_cursor=accept_drift["task_head_cursor"],
            expected_task_head_hash=accept_drift["task_head_hash"],
            actor=TaskActor(actor_type="user", actor_id=OWNER_ID),
            owner_user_id=OWNER_ID,
        )
        accepted_task = note_db.get_task(
            owner_user_id=OWNER_ID,
            dataset_id=dataset.dataset_id,
            task_id=str(keep_task["id"]),
        )
        assert accepted["status"] == "resolved"
        assert accepted_task is not None
        assert accepted_task["text"] == "Accepted Markdown edit"

        _, unlink_drift = create_conflict(
            accepted_task,
            explicit_text="Explicit unlink edit",
            markdown_text="Unlinked Markdown",
        )
        unlinked_drift = task_service.resolve_projection_drift(
            db=note_db,
            note_id=NOTE_ID,
            task_id=str(accepted_task["id"]),
            drift_id=str(unlink_drift["id"]),
            action="unlink",
            expected_lifecycle_revision=1,
            expected_note_head_cursor=unlink_drift["note_head_cursor"],
            expected_note_head_hash=unlink_drift["note_head_hash"],
            expected_task_head_cursor=unlink_drift["task_head_cursor"],
            expected_task_head_hash=unlink_drift["task_head_hash"],
            actor=TaskActor(actor_type="user", actor_id=OWNER_ID),
            owner_user_id=OWNER_ID,
        )
        unlink_task = note_db.get_task(
            owner_user_id=OWNER_ID,
            dataset_id=dataset.dataset_id,
            task_id=str(accepted_task["id"]),
        )
        unlink_note = note_db.get_note_by_id(NOTE_ID)
        assert unlinked_drift["status"] == "resolved"
        assert unlink_task is not None
        assert unlink_task["projection_status"] == "unlinked"
        assert unlink_note is not None
        assert "tldw-task" not in unlink_note["content"]
        assert "Unlinked Markdown" in unlink_note["content"]

        dismiss_base = task_service.create_task_for_note(
            db=note_db,
            note_id=NOTE_ID,
            text="Dismiss base",
            status="open",
            metadata={},
            expected_note_version=int(unlink_note["version"]),
            actor=TaskActor(actor_type="user", actor_id=OWNER_ID),
        )
        dismiss_explicit, dismiss_drift = create_conflict(
            dismiss_base,
            explicit_text="Dismiss explicit",
            markdown_text="Dismiss Markdown",
        )
        dismissed = task_service.resolve_projection_drift(
            db=note_db,
            note_id=NOTE_ID,
            task_id=str(dismiss_base["id"]),
            drift_id=str(dismiss_drift["id"]),
            action="dismiss",
            expected_lifecycle_revision=1,
            expected_note_head_cursor=dismiss_drift["note_head_cursor"],
            expected_note_head_hash=dismiss_drift["note_head_hash"],
            expected_task_head_cursor=dismiss_drift["task_head_cursor"],
            expected_task_head_hash=dismiss_drift["task_head_hash"],
            actor=TaskActor(actor_type="user", actor_id=OWNER_ID),
            owner_user_id=OWNER_ID,
        )
        dismissed_task = note_db.get_task(
            owner_user_id=OWNER_ID,
            dataset_id=dataset.dataset_id,
            task_id=str(dismiss_base["id"]),
        )
        dismissed_note = note_db.get_note_by_id(NOTE_ID)
        assert dismissed["status"] == "dismissed"
        assert dismissed_task is not None
        assert dismissed_task["text"] == dismiss_explicit["text"]
        assert dismissed_note is not None
        assert "Dismiss Markdown" in dismissed_note["content"]
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
