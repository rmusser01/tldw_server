from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Notes_Tasks.models import TaskActor
from tldw_Server_API.app.core.Notes_Tasks.service import (
    NotesTaskCaptureMutation,
    NotesTaskService,
    build_task_capture_mutation,
)
from tldw_Server_API.app.core.Sync.v2.notes_task_contract import (
    notes_task_object_hash,
    parse_notes_task_v1,
)

pytestmark = pytest.mark.unit

OWNER_ID = "capture-owner"
DATASET_ID = "local-unbound"


@pytest.fixture()
def db(tmp_path: Path) -> CharactersRAGDB:
    database = CharactersRAGDB(tmp_path / "capture.db", client_id=OWNER_ID)
    try:
        yield database
    finally:
        database.close_connection()


def _actor() -> TaskActor:
    return TaskActor(
        actor_type="user",
        actor_id=OWNER_ID,
        tool_name="notes.tasks.test",
        idempotency_key="request-task-capture",
    )


def _add_note(db: CharactersRAGDB, content: str = "Intro\n") -> dict[str, Any]:
    note_id = db.add_note(title="Capture tasks", content=content)
    note = db.get_note_by_id(str(note_id))
    assert note is not None
    return note


def test_task_capture_callback_receives_exact_create_status_and_delete_steps(
    db: CharactersRAGDB,
) -> None:
    captured: list[NotesTaskCaptureMutation] = []

    def capture(mutation: NotesTaskCaptureMutation, *, conn: object) -> None:
        assert conn is not None
        captured.append(mutation)

    service = NotesTaskService(task_capture_callback=capture)
    note = _add_note(db)
    created = service.create_task_for_note(
        db=db,
        note_id=str(note["id"]),
        text="Ship capture seam",
        status="open",
        metadata={"priority": "high"},
        expected_note_version=int(note["version"]),
        actor=_actor(),
    )
    after_create_note = db.get_note_by_id(str(note["id"]))
    assert after_create_note is not None
    updated = service.update_task(
        db=db,
        task_id=str(created["id"]),
        expected_task_version=int(created["version"]),
        expected_note_version=int(after_create_note["version"]),
        actor=_actor(),
        status="done",
    )
    after_update_note = db.get_note_by_id(str(note["id"]))
    assert after_update_note is not None
    deleted = service.delete_task(
        db=db,
        task_id=str(created["id"]),
        expected_task_version=int(updated["version"]),
        expected_note_version=int(after_update_note["version"]),
        record_only=False,
        actor=_actor(),
    )

    assert [item.operation for item in captured] == ["upsert", "upsert", "tombstone"]
    assert captured[0].before is None
    assert captured[1].base_revision == int(created["canonical_revision"])
    assert captured[1].base_hash == created["canonical_hash"]
    assert captured[1].after["status"] == "done"
    assert bool(captured[2].after["deleted"]) is True
    assert captured[2].step.object_revision == deleted["canonical_revision"]
    assert captured[2].step.payload == captured[2].after["sync_payload"]
    assert captured[2].step.parent_id == note["id"]
    assert captured[2].step.client_envelope_id
    assert captured[2].idempotency_key


def test_task_capture_builder_is_stable_and_marks_restore_intent(
    db: CharactersRAGDB,
) -> None:
    note = _add_note(db)
    task = db.task_store.create_task(
        owner_user_id=OWNER_ID,
        dataset_id=DATASET_ID,
        note_id=str(note["id"]),
        text="Restore capture",
        projection_status="unlinked",
    )
    deleted = db.task_store.soft_delete_task(
        owner_user_id=OWNER_ID,
        dataset_id=DATASET_ID,
        task_id=str(task["id"]),
        expected_version=int(task["version"]),
        allow_record_only=True,
    )
    deleted_source = db.task_store._sync_bootstrap_task_row(deleted, OWNER_ID)
    payload = parse_notes_task_v1(
        deleted_source["sync_payload"],
        owner_user_id=OWNER_ID,
    )
    next_revision = int(deleted["canonical_revision"]) + 1
    next_hash = notes_task_object_hash(
        payload,
        revision=next_revision,
        deleted=False,
    )
    with db.transaction() as conn:
        restored = db.task_store.apply_sync_task_restore(
            owner_user_id=OWNER_ID,
            dataset_id=DATASET_ID,
            payload=payload,
            base_revision=int(deleted["canonical_revision"]),
            base_hash=str(deleted["canonical_hash"]),
            canonical_revision=next_revision,
            canonical_hash=next_hash,
            conn=conn,
        )

    first = build_task_capture_mutation(
        db=db,
        owner_user_id=OWNER_ID,
        dataset_id=DATASET_ID,
        actor=_actor(),
        before=deleted,
        after=restored,
    )
    second = build_task_capture_mutation(
        db=db,
        owner_user_id=OWNER_ID,
        dataset_id=DATASET_ID,
        actor=_actor(),
        before=deleted,
        after=restored,
    )

    assert first == second
    assert first.operation == "upsert"
    assert first.restore_intent is True
    assert first.step.routing_metadata == {
        "restore_intent": True,
        "product_transition_base": True,
    }
    assert first.step.base_object_revision == first.base_revision
    assert first.step.base_object_hash == first.base_hash
    assert first.base_revision == deleted["canonical_revision"]
    assert first.base_hash == deleted["canonical_hash"]
    assert first.step.object_revision == next_revision
    assert first.step.client_envelope_id == second.step.client_envelope_id


def test_task_capture_failure_rolls_back_projected_product_mutation(
    db: CharactersRAGDB,
) -> None:
    note = _add_note(db, "- [ ] Preserve me\n")
    setup = NotesTaskService()
    setup.reconcile_note_current(db=db, note_id=str(note["id"]), actor=_actor())
    task = db.list_tasks(
        owner_user_id=OWNER_ID,
        dataset_id=DATASET_ID,
        note_id=str(note["id"]),
    )[0]
    before_note = db.get_note_by_id(str(note["id"]))
    assert before_note is not None

    def fail_capture(_mutation: NotesTaskCaptureMutation, *, conn: object) -> None:
        assert conn is not None
        raise RuntimeError("injected task capture failure")

    with pytest.raises(RuntimeError, match="injected task capture failure"):
        NotesTaskService(task_capture_callback=fail_capture).update_task(
            db=db,
            task_id=str(task["id"]),
            expected_task_version=int(task["version"]),
            expected_note_version=int(before_note["version"]),
            actor=_actor(),
            status="done",
        )

    after_note = db.get_note_by_id(str(note["id"]))
    after_task = db.get_task(
        owner_user_id=OWNER_ID,
        dataset_id=DATASET_ID,
        task_id=str(task["id"]),
    )
    assert after_note == before_note
    assert after_task == task


def test_legacy_service_without_task_capture_remains_unchanged(
    db: CharactersRAGDB,
) -> None:
    note = _add_note(db)

    created = NotesTaskService().create_task_for_note(
        db=db,
        note_id=str(note["id"]),
        text="Legacy task",
        status="open",
        metadata={},
        expected_note_version=int(note["version"]),
        actor=_actor(),
    )

    assert created["text"] == "Legacy task"
    assert created["canonical_revision"] == 1


def test_task_capture_remains_unwired_from_public_factories() -> None:
    api_root = Path(__file__).resolve().parents[2]
    endpoint_source = (api_root / "app/api/v1/endpoints/notes_tasks.py").read_text()
    mcp_source = (
        api_root / "app/core/MCP_unified/modules/implementations/notes_module.py"
    ).read_text()
    assert "task_capture_callback=" not in endpoint_source
    assert "task_capture_callback=" not in mcp_source
