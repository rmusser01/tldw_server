"""Unit coverage for note task service mutation boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, InputError
from tldw_Server_API.app.core.Notes_Tasks.models import TaskActor
from tldw_Server_API.app.core.Notes_Tasks.service import NotesTaskService


@pytest.fixture()
def db(tmp_path: Path) -> CharactersRAGDB:
    database = CharactersRAGDB(str(tmp_path / "notes_task_service.db"), client_id="notes_task_service_test")
    try:
        yield database
    finally:
        database.close_connection()


def _actor() -> TaskActor:
    return TaskActor(actor_type="user", actor_id="tester")


def _add_note(db: CharactersRAGDB, *, title: str, content: str) -> dict[str, Any]:
    note_id = db.add_note(title=title, content=content)
    note = db.get_note_by_id(str(note_id))
    assert note is not None
    return note


def test_create_task_for_note_rejects_invalid_status_without_rewriting_note(db: CharactersRAGDB) -> None:
    service = NotesTaskService()
    note = _add_note(db, title="Tasks", content="Intro\n")

    with pytest.raises(InputError, match="Unsupported task status"):
        service.create_task_for_note(
            db=db,
            note_id=str(note["id"]),
            text="Alpha",
            status="blocked",
            metadata={},
            expected_note_version=int(note["version"]),
            actor=_actor(),
        )

    saved = db.get_note_by_id(str(note["id"]))
    assert saved is not None
    assert saved["content"] == "Intro\n"
    assert db.list_tasks(note_id=str(note["id"])) == []


def test_reconcile_stale_notes_reports_actual_remaining_count(db: CharactersRAGDB) -> None:
    service = NotesTaskService()
    for index in range(3):
        _add_note(db, title=f"Note {index}", content=f"- [ ] Task {index}\n")

    result = service.reconcile_stale_notes(db=db, limit=1, actor=_actor())

    assert result.processed_notes == 1
    assert result.remaining_stale_notes == 2
    assert result.status == "incomplete"


def test_delete_projection_line_preserves_crlf_boundaries() -> None:
    content = "Before\r\n- [ ] Alpha\r\nAfter\r\n"
    projection = {
        "start_offset": len("Before\r\n"),
        "end_offset": len("Before\r\n- [ ] Alpha"),
    }

    updated = NotesTaskService._delete_projection_line(content, projection)

    assert updated == "Before\r\nAfter\r\n"
