"""Unit tests for note checklist task reconciliation."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from tldw_Server_API.app.core.Notes_Tasks.models import TaskActor
from tldw_Server_API.app.core.Notes_Tasks.service import NotesTaskService

pytestmark = pytest.mark.unit


@pytest.fixture()
def notes_db(tmp_path: Path) -> Generator[CharactersRAGDB, None, None]:
    db = CharactersRAGDB(str(tmp_path / "notes_tasks_reconciler.db"), client_id="task_reconciler_test")
    try:
        yield db
    finally:
        db.close_connection()


@pytest.fixture()
def service() -> NotesTaskService:
    return NotesTaskService()


def _actor() -> TaskActor:
    return TaskActor(actor_type="system", actor_id="unit-test")


def _create_note(db: CharactersRAGDB, content: str) -> dict:
    note_id = db.add_note(title="Checklist", content=content)
    assert note_id is not None
    note = db.get_note_by_id(str(note_id))
    assert note is not None
    return note


def _update_note(db: CharactersRAGDB, note_id: str, expected_version: int, content: str) -> dict:
    db.update_note(
        note_id=note_id,
        update_data={"content": content},
        expected_version=expected_version,
    )
    note = db.get_note_by_id(note_id)
    assert note is not None
    return note


def _reconcile(service: NotesTaskService, db: CharactersRAGDB, note: dict):
    return service.reconcile_note(
        db=db,
        note_id=str(note["id"]),
        note_version=int(note["version"]),
        content=str(note["content"]),
        actor=_actor(),
    )


def _task_by_text(db: CharactersRAGDB, note_id: str) -> dict[str, dict]:
    return {task["text"]: task for task in db.list_tasks(note_id=note_id, include_deleted=True, limit=100)}


def _live_tasks_by_line(db: CharactersRAGDB, note_id: str) -> dict[int, dict]:
    tasks_by_line: dict[int, dict] = {}
    for task in db.list_tasks(note_id=note_id, include_deleted=True, limit=100):
        projection = db.task_store._fetch_projection(task["id"])
        if task["projection_status"] == "live" and projection is not None:
            tasks_by_line[int(projection["line_number"])] = task
    return tasks_by_line


def _tasks_for_text(db: CharactersRAGDB, note_id: str, text: str) -> list[dict]:
    return [
        task
        for task in db.list_tasks(note_id=note_id, include_deleted=True, limit=100)
        if task["text"] == text
    ]


def test_unchanged_content_is_idempotent(
    notes_db: CharactersRAGDB,
    service: NotesTaskService,
) -> None:
    note = _create_note(notes_db, "- [ ] Alpha\n- [x] Beta\n")

    first = _reconcile(service, notes_db, note)
    tasks_after_first = notes_db.list_tasks(note_id=note["id"], include_deleted=True, limit=100)
    versions_after_first = {task["id"]: task["version"] for task in tasks_after_first}

    second = _reconcile(service, notes_db, note)
    tasks_after_second = notes_db.list_tasks(note_id=note["id"], include_deleted=True, limit=100)

    assert first.created_count == 2
    assert second.created_count == 0
    assert second.updated_count == 0
    assert [task["id"] for task in tasks_after_second] == [task["id"] for task in tasks_after_first]
    assert {task["id"]: task["version"] for task in tasks_after_second} == versions_after_first
    assert notes_db.get_reconciliation_state(note["id"])["note_version"] == note["version"]


def test_same_locator_and_hash_preserves_task_id(
    notes_db: CharactersRAGDB,
    service: NotesTaskService,
) -> None:
    note = _create_note(notes_db, "- [ ] Alpha\n- [ ] Beta\n")
    _reconcile(service, notes_db, note)
    original_alpha = _task_by_text(notes_db, note["id"])["Alpha"]

    updated_note = _update_note(notes_db, note["id"], int(note["version"]), "- [x] Alpha\n- [ ] Beta\n")
    result = _reconcile(service, notes_db, updated_note)
    updated_alpha = _task_by_text(notes_db, note["id"])["Alpha"]

    assert result.created_count == 0
    assert updated_alpha["id"] == original_alpha["id"]
    assert updated_alpha["status"] == "done"


def test_unique_reordered_item_preserves_task_id(
    notes_db: CharactersRAGDB,
    service: NotesTaskService,
) -> None:
    note = _create_note(notes_db, "- [ ] Alpha\n- [ ] Beta\n")
    _reconcile(service, notes_db, note)
    original_tasks = _task_by_text(notes_db, note["id"])

    updated_note = _update_note(notes_db, note["id"], int(note["version"]), "- [ ] Beta\n- [ ] Alpha\n")
    result = _reconcile(service, notes_db, updated_note)
    reordered_tasks = _task_by_text(notes_db, note["id"])

    assert result.created_count == 0
    assert reordered_tasks["Alpha"]["id"] == original_tasks["Alpha"]["id"]
    assert reordered_tasks["Beta"]["id"] == original_tasks["Beta"]["id"]


def test_duplicate_text_reorder_becomes_ambiguous_or_distinct(
    notes_db: CharactersRAGDB,
    service: NotesTaskService,
) -> None:
    note = _create_note(notes_db, "- [ ] Solo\n- [ ] Same\n- [ ] Same\n")
    _reconcile(service, notes_db, note)
    original_live_ids = {
        task["id"]
        for task in notes_db.list_tasks(note_id=note["id"], include_deleted=True, limit=100)
        if task["projection_status"] == "live"
    }
    original_same_ids = {
        task["id"]
        for task in notes_db.list_tasks(note_id=note["id"], include_deleted=True, limit=100)
        if task["projection_status"] == "live" and task["text"] == "Same"
    }

    updated_note = _update_note(notes_db, note["id"], int(note["version"]), "- [ ] Same\n- [ ] Solo\n- [ ] Same\n")
    result = _reconcile(service, notes_db, updated_note)
    tasks = notes_db.list_tasks(note_id=note["id"], include_deleted=True, limit=100)
    live_ids = {task["id"] for task in tasks if task["projection_status"] == "live"}
    live_same_ids = {task["id"] for task in tasks if task["projection_status"] == "live" and task["text"] == "Same"}
    ambiguous_ids = {task["id"] for task in tasks if task["projection_status"] == "ambiguous"}

    assert len(tasks) >= 2
    assert result.ambiguous_count > 0 or live_ids != original_live_ids
    assert result.ambiguous_count > 0 or live_same_ids != original_same_ids
    assert ambiguous_ids or len(live_same_ids) == 2


def test_duplicate_stable_lines_preserve_ids_when_one_status_changes(
    notes_db: CharactersRAGDB,
    service: NotesTaskService,
) -> None:
    note = _create_note(notes_db, "- [ ] Same\n- [ ] Same\n")
    _reconcile(service, notes_db, note)
    original_by_line = _live_tasks_by_line(notes_db, note["id"])

    updated_note = _update_note(notes_db, note["id"], int(note["version"]), "- [x] Same\n- [ ] Same\n")
    result = _reconcile(service, notes_db, updated_note)
    updated_by_line = _live_tasks_by_line(notes_db, note["id"])

    assert result.created_count == 0
    assert result.unlinked_count == 0
    assert updated_by_line[1]["id"] == original_by_line[1]["id"]
    assert updated_by_line[1]["status"] == "done"
    assert updated_by_line[2]["id"] == original_by_line[2]["id"]
    assert updated_by_line[2]["status"] == "open"


def test_unique_hash_fallback_requires_same_block_fingerprint(
    notes_db: CharactersRAGDB,
    service: NotesTaskService,
) -> None:
    note = _create_note(
        notes_db,
        "- [ ] Parent A\n  - [ ] Move me\n    detail A\n- [ ] Parent B\n  - [ ] Stay put\n",
    )
    _reconcile(service, notes_db, note)
    original_move_task = _tasks_for_text(notes_db, note["id"], "Move me")[0]

    updated_note = _update_note(
        notes_db,
        note["id"],
        int(note["version"]),
        "- [ ] Parent A\n  - [ ] Stay put\n- [ ] Parent B\n  - [ ] Move me\n    detail B\n",
    )
    _reconcile(service, notes_db, updated_note)
    move_tasks = _tasks_for_text(notes_db, note["id"], "Move me")
    live_move_tasks = [task for task in move_tasks if task["projection_status"] == "live"]
    original_after = notes_db.get_task(original_move_task["id"], include_deleted=True)

    assert len(live_move_tasks) == 1
    assert live_move_tasks[0]["id"] != original_move_task["id"]
    assert original_after is not None
    assert original_after["projection_status"] == "unlinked"


def test_missing_line_marks_previous_task_unlinked(
    notes_db: CharactersRAGDB,
    service: NotesTaskService,
) -> None:
    note = _create_note(notes_db, "- [ ] Alpha\n- [ ] Beta\n")
    _reconcile(service, notes_db, note)
    beta_id = _task_by_text(notes_db, note["id"])["Beta"]["id"]

    updated_note = _update_note(notes_db, note["id"], int(note["version"]), "- [ ] Alpha\n")
    result = _reconcile(service, notes_db, updated_note)
    beta = notes_db.get_task(beta_id, include_deleted=True)

    assert result.unlinked_count == 1
    assert beta is not None
    assert beta["projection_status"] == "unlinked"


def test_manual_line_removal_does_not_hard_delete_task_history(
    notes_db: CharactersRAGDB,
    service: NotesTaskService,
) -> None:
    note = _create_note(notes_db, "- [ ] Keep\n- [ ] Remove\n")
    _reconcile(service, notes_db, note)
    removed_id = _task_by_text(notes_db, note["id"])["Remove"]["id"]

    updated_note = _update_note(notes_db, note["id"], int(note["version"]), "- [ ] Keep\n")
    _reconcile(service, notes_db, updated_note)
    removed_task = notes_db.get_task(removed_id, include_deleted=True)
    events = notes_db.list_task_activity(task_id=removed_id, limit=20)

    assert removed_task is not None
    assert not bool(removed_task["deleted"])
    assert removed_task["projection_status"] == "unlinked"
    assert any(event["event_type"] == "unlinked" for event in events)


def test_stale_note_version_does_not_update_reconciliation_state(
    notes_db: CharactersRAGDB,
    service: NotesTaskService,
) -> None:
    note = _create_note(notes_db, "- [ ] Alpha\n")
    _reconcile(service, notes_db, note)
    original_state = notes_db.get_reconciliation_state(note["id"])
    assert original_state is not None

    _update_note(notes_db, note["id"], int(note["version"]), "- [x] Alpha\n")

    with pytest.raises(ConflictError):
        service.reconcile_note(
            db=notes_db,
            note_id=str(note["id"]),
            note_version=int(note["version"]),
            content="- [x] Alpha\n",
            actor=_actor(),
        )

    assert notes_db.get_reconciliation_state(note["id"]) == original_state
