"""Unit coverage for note task service mutation boundaries."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, InputError
from tldw_Server_API.app.core.Notes_Tasks.models import TaskActor
from tldw_Server_API.app.core.Notes_Tasks.service import (
    NotesTaskCaptureMutation,
    NotesTaskService,
    _parse_checklist_line,
)

pytestmark = pytest.mark.unit


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


def test_projected_mutations_lock_task_scope_before_reading_task_or_note_rows() -> None:
    """Keep direct product writes in the same authority-first order as dataset bind."""
    for method in (NotesTaskService.update_task, NotesTaskService.delete_task):
        source = inspect.getsource(method)
        direct_write_path = source[source.index("coordinator = active_coordinator") :]
        fence = direct_write_path.index("lock_authorized_write_scope")
        assert fence < direct_write_path.index("_require_task_version")
        assert fence < direct_write_path.index("_write_note_content")


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
    assert db.list_tasks(
        note_id=str(note["id"]),
        owner_user_id=db.client_id,
        dataset_id="local-unbound",
    ) == []


def test_optional_capture_callback_receives_one_task_activity_plan(
    db: CharactersRAGDB,
) -> None:
    captured: list[NotesTaskCaptureMutation] = []
    service = NotesTaskService(
        task_capture_callback=lambda mutation, *, conn: captured.append(mutation)
    )
    note = _add_note(db, title="Tasks", content="Intro\n")

    service.create_task_for_note(
        db=db,
        note_id=str(note["id"]),
        text="Alpha",
        status="open",
        metadata={},
        expected_note_version=int(note["version"]),
        actor=TaskActor(actor_type="user", actor_id=db.client_id),
    )

    assert len(captured) == 1
    assert [step.domain for step in captured[0].steps] == [
        "notes.task",
        "notes.task_activity",
    ]
    assert captured[0].activity.payload.event_type == "created"


@pytest.mark.parametrize(
    "text",
    [
        "Call @priority(high) customer",
        "Ship @due(2026-06-30)",
        "Plan @estimate(2h)",
    ],
)
def test_task_text_rejects_parseable_metadata_tokens(text: str) -> None:
    with pytest.raises(InputError, match="metadata tokens"):
        NotesTaskService._validate_task_text(text)


@pytest.mark.parametrize(
    "text",
    [
        "Call @foo(bar) customer",
        "Keep @foo(@due(2026-06-30)) as plain text",
        "Call @due(not-a-date) customer",
        "Call @priority(urgent) customer",
        "Call @estimate(two-hours) customer",
    ],
)
def test_task_text_allows_unknown_or_malformed_metadata_like_plain_text(text: str) -> None:
    NotesTaskService._validate_task_text(text)


def test_task_text_metadata_scan_handles_unclosed_tokens_as_plain_text() -> None:
    NotesTaskService._validate_task_text("@due(" + "2" * 1995)


def test_task_text_rejects_parseable_metadata_after_expanding_casefold_prefix() -> None:
    with pytest.raises(InputError, match="metadata tokens"):
        NotesTaskService._validate_task_text("İ@due(2026-06-30)")


def test_parse_checklist_line_accepts_existing_projected_format() -> None:
    parsed = _parse_checklist_line("\t-   [x]\tAlpha")

    assert parsed is not None
    assert parsed.indent == "\t"
    assert parsed.bullet == "-"
    assert parsed.space == "   "
    assert parsed.body_part == "\tAlpha"


def test_ensure_note_reconciled_preserves_current_warning_state_without_reprocessing(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = NotesTaskService()
    note = _add_note(db, title="Tasks", content="- [ ] Review @due(not-a-date)\n")
    first = service.reconcile_note_current(db=db, note_id=str(note["id"]), actor=_actor())
    assert first.warning_count == 1

    monkeypatch.setattr(
        service._reconciler,
        "reconcile_note",
        lambda **_: pytest.fail("current warning state should not be reconciled again"),
    )

    cached = service.ensure_note_reconciled(db=db, note_id=str(note["id"]), actor=_actor())

    assert cached is not None
    assert cached.note_id == str(note["id"])
    assert cached.note_version == int(note["version"])
    assert cached.warning_count == 1


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
