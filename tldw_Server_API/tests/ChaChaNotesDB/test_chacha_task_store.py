"""Tests for task-backed note storage."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
    InputError,
)


pytestmark = pytest.mark.unit


@pytest.fixture()
def db(tmp_path) -> CharactersRAGDB:
    database = CharactersRAGDB(
        db_path=str(tmp_path / "task_store.sqlite"),
        client_id="task-store-user",
    )
    yield database
    database.close_connection()


def _create_note(db: CharactersRAGDB, content: str = "- [ ] Review source\n") -> str:
    note_id = db.add_note(title="Task Note", content=content)
    assert note_id is not None  # nosec B101
    return note_id


def _create_task(db: CharactersRAGDB, note_id: str, *, task_id: str = "task-1") -> dict:
    created = db.create_task(
        task_id=task_id,
        note_id=note_id,
        text="Review source",
        status="open",
        metadata={"due_date": "2026-06-10"},
        actor_type="user",
        actor_id="user-1",
    )
    assert created["id"] == task_id  # nosec B101
    return created


def _set_projection(db: CharactersRAGDB, task_id: str, note_id: str, *, note_version: int = 1) -> dict:
    return db.set_task_projection(
        task_id=task_id,
        note_id=note_id,
        note_version=note_version,
        line_number=1,
        start_offset=0,
        end_offset=20,
        normalized_text_hash="sha256:review",
        occurrence_index=0,
        block_fingerprint="block-1",
        raw_line="- [ ] Review source",
        has_child_content=False,
    )


def test_create_task_record_and_list_by_note(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)

    task = _create_task(db, note_id)

    assert task["note_id"] == note_id  # nosec B101
    assert task["text"] == "Review source"  # nosec B101
    assert task["status"] == "open"  # nosec B101
    assert task["metadata_json"] == {"due_date": "2026-06-10"}  # nosec B101
    assert task["projection_status"] == "live"  # nosec B101
    assert task["deleted"] in (0, False)  # nosec B101
    assert task["version"] == 1  # nosec B101
    assert task["client_id"] == "task-store-user"  # nosec B101
    fetched = db.get_task("task-1")
    assert fetched == task  # nosec B101
    listed = db.list_tasks(note_id=note_id)
    assert [row["id"] for row in listed] == ["task-1"]  # nosec B101


def test_update_status_uses_optimistic_locking(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)

    updated = db.update_task_record(
        task_id="task-1",
        expected_version=1,
        status="done",
        actor_type="user",
        actor_id="user-1",
    )

    assert updated["status"] == "done"  # nosec B101
    assert updated["version"] == 2  # nosec B101
    assert updated["completed_at"] is not None  # nosec B101
    with pytest.raises(ConflictError, match="version mismatch"):
        db.update_task_record(
            task_id="task-1",
            expected_version=1,
            status="open",
            actor_type="user",
            actor_id="user-1",
        )


def test_update_task_record_rejects_zero_rowcount_update(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)
    original_execute = db.task_store._execute

    class _ZeroRowcount:
        rowcount = 0

    def _simulate_concurrent_update(conn, query, params=None):
        if "UPDATE tasks" in query and "version = version + 1" in query:
            return _ZeroRowcount()
        return original_execute(conn, query, params)

    monkeypatch.setattr(db.task_store, "_execute", _simulate_concurrent_update)

    with pytest.raises(ConflictError, match="version mismatch"):
        db.update_task_record(
            task_id="task-1",
            expected_version=1,
            status="done",
            actor_type="user",
            actor_id="user-1",
        )


def test_reopen_clears_completed_at_and_records_event_history(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)

    db.update_task_record(
        task_id="task-1",
        expected_version=1,
        status="done",
        actor_type="user",
        actor_id="user-1",
    )
    reopened = db.update_task_record(
        task_id="task-1",
        expected_version=2,
        status="open",
        actor_type="user",
        actor_id="user-1",
    )

    assert reopened["status"] == "open"  # nosec B101
    assert reopened["completed_at"] is None  # nosec B101
    events = db.list_task_activity(task_id="task-1")
    status_events = [event for event in events if event["event_type"] == "status_changed"]
    assert [event["old_value_json"]["status"] for event in status_events] == ["open", "done"]  # nosec B101
    assert [event["new_value_json"]["status"] for event in status_events] == ["done", "open"]  # nosec B101


def test_mark_task_unlinked_rejects_zero_rowcount_update(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)
    _set_projection(db, "task-1", note_id)
    original_execute = db.task_store._execute
    projection_update_attempted = False

    class _ZeroRowcount:
        rowcount = 0

    def _simulate_concurrent_update(conn, query, params=None):
        nonlocal projection_update_attempted
        if "UPDATE tasks" in query and "version = version + 1" in query:
            return _ZeroRowcount()
        if "UPDATE task_note_projections" in query:
            projection_update_attempted = True
        return original_execute(conn, query, params)

    monkeypatch.setattr(db.task_store, "_execute", _simulate_concurrent_update)

    with pytest.raises(ConflictError, match="version mismatch"):
        db.mark_task_unlinked(
            task_id="task-1",
            expected_version=1,
            actor_type="system",
            actor_id="reconciler",
        )
    assert projection_update_attempted is False  # nosec B101


def test_soft_delete_projected_task_transactionally(db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)
    _set_projection(db, "task-1", note_id)

    with db.transaction() as conn:
        monkeypatch.setattr(
            db,
            "transaction",
            lambda: pytest.fail("task helper should use the explicit transaction connection"),
        )
        deleted = db.soft_delete_task(
            task_id="task-1",
            expected_version=1,
            projection_note_id=note_id,
            projection_note_version=1,
            projection_line_number=1,
            actor_type="user",
            actor_id="user-1",
            conn=conn,
        )

    assert deleted["deleted"] in (1, True)  # nosec B101
    assert deleted["projection_status"] == "deleted"  # nosec B101
    assert deleted["version"] == 2  # nosec B101
    fetched = db.get_task("task-1")
    assert fetched is None  # nosec B101
    included = db.get_task("task-1", include_deleted=True)
    assert included is not None  # nosec B101
    assert included["deleted"] in (1, True)  # nosec B101


def test_soft_delete_task_rejects_zero_rowcount_update(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)
    _set_projection(db, "task-1", note_id)
    original_execute = db.task_store._execute
    projection_update_attempted = False

    class _ZeroRowcount:
        rowcount = 0

    def _simulate_concurrent_update(conn, query, params=None):
        nonlocal projection_update_attempted
        if "UPDATE tasks" in query and "version = version + 1" in query:
            return _ZeroRowcount()
        if "UPDATE task_note_projections" in query:
            projection_update_attempted = True
        return original_execute(conn, query, params)

    monkeypatch.setattr(db.task_store, "_execute", _simulate_concurrent_update)

    with pytest.raises(ConflictError, match="version mismatch"):
        db.soft_delete_task(
            task_id="task-1",
            expected_version=1,
            projection_note_id=note_id,
            projection_note_version=1,
            projection_line_number=1,
            actor_type="user",
            actor_id="user-1",
        )
    assert projection_update_attempted is False  # nosec B101


def test_record_only_soft_delete_is_allowed_for_unlinked_task(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)
    _set_projection(db, "task-1", note_id)

    unlinked = db.mark_task_unlinked(
        task_id="task-1",
        expected_version=1,
        actor_type="system",
        actor_id="reconciler",
    )
    assert unlinked["projection_status"] == "unlinked"  # nosec B101
    assert unlinked["version"] == 2  # nosec B101
    deleted = db.soft_delete_task(
        task_id="task-1",
        expected_version=2,
        allow_record_only=True,
        actor_type="user",
        actor_id="user-1",
    )

    assert deleted["deleted"] in (1, True)  # nosec B101
    assert deleted["projection_status"] == "deleted"  # nosec B101
    assert deleted["version"] == 3  # nosec B101


def test_rejects_ambiguous_projection_deletion(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)
    _set_projection(db, "task-1", note_id)

    with pytest.raises(InputError, match="projection deletion is ambiguous"):
        db.soft_delete_task(
            task_id="task-1",
            expected_version=1,
            actor_type="user",
            actor_id="user-1",
        )


def test_reconciliation_state_is_recorded_per_note_version(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)

    first = db.set_reconciliation_state(
        note_id=note_id,
        note_version=1,
        status="clean",
        item_count=1,
        warning_count=0,
        cursor="line:1",
    )
    assert first["note_version"] == 1  # nosec B101
    assert first["status"] == "clean"  # nosec B101
    second = db.set_reconciliation_state(
        note_id=note_id,
        note_version=2,
        status="warnings",
        item_count=2,
        warning_count=1,
        cursor="line:2",
    )
    assert second["note_version"] == 2  # nosec B101
    assert second["status"] == "warnings"  # nosec B101
    fetched = db.get_reconciliation_state(note_id)
    assert fetched == second  # nosec B101


def test_activity_read_and_dismiss_state_is_per_user(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)
    event = db.record_task_event(
        task_id="task-1",
        note_id=note_id,
        event_type="status_changed",
        actor_type="agent",
        actor_id="assistant-1",
        old_value={"status": "open"},
        new_value={"status": "done"},
        tool_name="notes.tasks.update_status",
        policy_mode="autonomous",
    )

    assert db.get_task_activity_read_state(event["id"], user_id="user-1") is None  # nosec B101
    read = db.mark_task_activity_read(event["id"], user_id="user-1")
    assert read["event_id"] == event["id"]  # nosec B101
    assert read["user_id"] == "user-1"  # nosec B101
    assert read["read_at"] is not None  # nosec B101
    assert read["dismissed_at"] is None  # nosec B101
    dismissed = db.mark_task_activity_dismissed(event["id"], user_id="user-1")
    assert dismissed["read_at"] == read["read_at"]  # nosec B101
    assert dismissed["dismissed_at"] is not None  # nosec B101
    assert db.get_task_activity_read_state(event["id"], user_id="user-2") is None  # nosec B101


def test_activity_read_and_dismiss_helpers_accept_explicit_transaction_connection(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)
    event = db.record_task_event(
        task_id="task-1",
        note_id=note_id,
        event_type="status_changed",
        actor_type="agent",
        actor_id="assistant-1",
        old_value={"status": "open"},
        new_value={"status": "done"},
    )

    with db.transaction() as conn:
        monkeypatch.setattr(
            db,
            "transaction",
            lambda: pytest.fail("task activity helpers should use the explicit transaction connection"),
        )
        read = db.mark_task_activity_read(event["id"], user_id="user-1", conn=conn)
        dismissed = db.mark_task_activity_dismissed(event["id"], user_id="user-1", conn=conn)

    assert read["read_at"] is not None  # nosec B101
    assert dismissed["dismissed_at"] is not None  # nosec B101


def test_candidate_notes_for_task_discovery_excludes_currently_reconciled_notes(db: CharactersRAGDB) -> None:
    note_id = _create_note(db, content="Intro\n- [ ] Review source\n")
    plain_note_id = _create_note(db, content="No checklist here")

    candidates = db.candidate_notes_for_task_discovery(limit=10)
    assert [row["id"] for row in candidates] == [note_id]  # nosec B101
    db.set_reconciliation_state(
        note_id=note_id,
        note_version=1,
        status="clean",
        item_count=1,
        warning_count=0,
    )

    assert db.candidate_notes_for_task_discovery(limit=10) == []  # nosec B101
    assert plain_note_id not in {row["id"] for row in candidates}  # nosec B101
