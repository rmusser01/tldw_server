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


def _set_projection(
    db: CharactersRAGDB,
    task_id: str,
    note_id: str,
    *,
    note_version: int = 1,
    projection_status: str = "live",
) -> dict:
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
        projection_status=projection_status,
    )


def _force_projection_status_drift(
    db: CharactersRAGDB,
    task_id: str,
    *,
    task_status: str,
    projection_status: str,
) -> None:
    with db.transaction() as conn:
        conn.execute(
            "UPDATE note_tasks SET projection_status = ? WHERE id = ?",
            (task_status, task_id),
        )
        conn.execute(
            "UPDATE task_note_projections SET projection_status = ? WHERE task_id = ?",
            (projection_status, task_id),
        )


def _force_projection_note_drift(db: CharactersRAGDB, task_id: str, note_id: str) -> None:
    with db.transaction() as conn:
        conn.execute(
            "UPDATE task_note_projections SET note_id = ? WHERE task_id = ?",
            (note_id, task_id),
        )


def _projection_count(db: CharactersRAGDB, task_id: str) -> int:
    with db.transaction() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM task_note_projections WHERE task_id = ?",
            (task_id,),
        ).fetchone()
    return int(row["count"])


def _delete_projection(db: CharactersRAGDB, task_id: str) -> None:
    with db.transaction() as conn:
        conn.execute("DELETE FROM task_note_projections WHERE task_id = ?", (task_id,))


def _event_count(db: CharactersRAGDB, task_id: str, event_type: str) -> int:
    events = db.list_task_activity(task_id=task_id)
    return sum(1 for event in events if event["event_type"] == event_type)


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


def test_active_task_reads_exclude_tasks_for_soft_deleted_notes(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)
    task = _create_task(db, note_id)

    assert db.soft_delete_note(note_id, expected_version=1) is True  # nosec B101

    assert db.get_task("task-1") is None  # nosec B101
    assert db.get_task("task-1", include_deleted=True)["id"] == task["id"]  # nosec B101
    assert db.list_tasks() == []  # nosec B101
    assert db.list_tasks(note_id=note_id) == []  # nosec B101


def test_create_task_rejects_soft_deleted_notes(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)
    assert db.soft_delete_note(note_id, expected_version=1) is True  # nosec B101

    with pytest.raises(ConflictError, match="note.*deleted"):
        db.create_task(
            task_id="task-soft-deleted-note",
            note_id=note_id,
            text="Review source",
            actor_type="user",
            actor_id="user-1",
        )
    assert db.list_tasks(include_deleted=True) == []  # nosec B101


def test_create_task_rejects_deleted_projection_status(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)

    with pytest.raises(InputError, match="deleted"):
        db.create_task(
            task_id="task-deleted-projection",
            note_id=note_id,
            text="Review source",
            projection_status="deleted",
            actor_type="user",
            actor_id="user-1",
        )
    assert db.list_tasks(include_deleted=True) == []  # nosec B101


def test_list_helpers_clamp_negative_limits(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)
    for index in range(3):
        _create_task(db, note_id, task_id=f"task-{index}")

    assert len(db.list_tasks(limit=-10)) == 1  # nosec B101
    assert len(db.list_task_activity(limit=-10)) == 1  # nosec B101


def test_list_helpers_reject_nonnumeric_limits(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)

    with pytest.raises(InputError, match="limit"):
        db.list_tasks(limit="many")


def test_candidate_notes_for_task_discovery_clamps_oversized_limits(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for index in range(3):
        _create_note(db, content=f"- [ ] Review source {index}\n")
    monkeypatch.setattr(db.task_store, "_MAX_LIMIT", 2, raising=False)

    candidates = db.candidate_notes_for_task_discovery(limit=10_000)

    assert len(candidates) == 2  # nosec B101


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
        if "UPDATE note_tasks" in query and "version = version + 1" in query:
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


def test_update_task_record_rejects_deleted_tasks(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)
    deleted = db.soft_delete_task(
        task_id="task-1",
        expected_version=1,
        allow_record_only=True,
        actor_type="user",
        actor_id="user-1",
    )

    with pytest.raises(ConflictError, match="deleted"):
        db.update_task_record(
            task_id="task-1",
            expected_version=deleted["version"],
            status="open",
            actor_type="user",
            actor_id="user-1",
        )
    task = db.get_task("task-1", include_deleted=True)
    assert task["version"] == deleted["version"]  # nosec B101
    assert task["deleted"] in (1, True)  # nosec B101


def test_update_task_record_rejects_soft_deleted_parent_note(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)
    assert db.soft_delete_note(note_id, expected_version=1) is True  # nosec B101

    with pytest.raises(ConflictError, match="note.*deleted"):
        db.update_task_record(
            task_id="task-1",
            expected_version=1,
            status="done",
            actor_type="user",
            actor_id="user-1",
        )
    task = db.get_task("task-1", include_deleted=True)
    assert task["status"] == "open"  # nosec B101
    assert task["version"] == 1  # nosec B101


def test_update_task_record_rejects_projection_status_updates(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)
    _set_projection(db, "task-1", note_id)

    with pytest.raises(InputError, match="projection_status"):
        db.update_task_record(
            task_id="task-1",
            expected_version=1,
            projection_status="unlinked",
            actor_type="user",
            actor_id="user-1",
        )
    task = db.get_task("task-1", include_deleted=True)
    assert task["projection_status"] == "live"  # nosec B101
    assert task["version"] == 1  # nosec B101


def test_update_task_record_rejects_ambiguous_projected_tasks(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)
    _set_projection(db, "task-1", note_id, projection_status="ambiguous")

    with pytest.raises(ConflictError, match="ambiguous"):
        db.update_task_record(
            task_id="task-1",
            expected_version=2,
            status="done",
            actor_type="user",
            actor_id="user-1",
        )
    task = db.get_task("task-1", include_deleted=True)
    assert task["status"] == "open"  # nosec B101
    assert task["version"] == 2  # nosec B101


def test_update_task_record_rejects_drifted_ambiguous_projection_status(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)
    _set_projection(db, "task-1", note_id)
    _force_projection_status_drift(
        db,
        "task-1",
        task_status="live",
        projection_status="ambiguous",
    )

    with pytest.raises(ConflictError, match="projection status mismatch"):
        db.update_task_record(
            task_id="task-1",
            expected_version=1,
            status="done",
            actor_type="user",
            actor_id="user-1",
        )
    task = db.get_task("task-1", include_deleted=True)
    assert task["status"] == "open"  # nosec B101
    assert task["version"] == 1  # nosec B101


def test_update_task_record_rejects_drifted_projection_note_ownership(db: CharactersRAGDB) -> None:
    owning_note_id = _create_note(db, content="- [ ] Owning note\n")
    other_note_id = _create_note(db, content="- [ ] Other note\n")
    _create_task(db, owning_note_id)
    _set_projection(db, "task-1", owning_note_id)
    _force_projection_note_drift(db, "task-1", other_note_id)

    with pytest.raises(ConflictError, match="projection ownership mismatch"):
        db.update_task_record(
            task_id="task-1",
            expected_version=1,
            status="done",
            actor_type="user",
            actor_id="user-1",
        )
    task = db.get_task("task-1", include_deleted=True)
    events = db.list_task_activity(task_id="task-1")
    assert task["status"] == "open"  # nosec B101
    assert task["version"] == 1  # nosec B101
    assert [event["event_type"] for event in events] == ["created"]  # nosec B101


def test_update_task_record_rejects_unlinked_projected_tasks_by_default(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)
    _set_projection(db, "task-1", note_id)
    unlinked = db.mark_task_unlinked(
        task_id="task-1",
        expected_version=1,
        actor_type="system",
        actor_id="reconciler",
    )

    with pytest.raises(ConflictError, match="unlinked"):
        db.update_task_record(
            task_id="task-1",
            expected_version=unlinked["version"],
            text="Review source again",
            actor_type="user",
            actor_id="user-1",
        )
    task = db.get_task("task-1", include_deleted=True)
    assert task["text"] == "Review source"  # nosec B101
    assert task["version"] == unlinked["version"]  # nosec B101


def test_mark_task_unlinked_rejects_drifted_unlinked_projection_status(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)
    _set_projection(db, "task-1", note_id)
    _force_projection_status_drift(
        db,
        "task-1",
        task_status="live",
        projection_status="unlinked",
    )

    with pytest.raises(ConflictError, match="projection status mismatch"):
        db.mark_task_unlinked(
            task_id="task-1",
            expected_version=1,
            actor_type="system",
            actor_id="reconciler",
        )
    task = db.get_task("task-1", include_deleted=True)
    assert task["projection_status"] == "live"  # nosec B101
    assert task["version"] == 1  # nosec B101


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
        if "UPDATE note_tasks" in query and "version = version + 1" in query:
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


def test_mark_task_unlinked_rejects_deleted_tasks(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)
    _set_projection(db, "task-1", note_id)
    deleted = db.soft_delete_task(
        task_id="task-1",
        expected_version=1,
        projection_note_id=note_id,
        projection_note_version=1,
        projection_line_number=1,
        actor_type="user",
        actor_id="user-1",
    )

    with pytest.raises(ConflictError, match="deleted"):
        db.mark_task_unlinked(
            task_id="task-1",
            expected_version=deleted["version"],
            actor_type="system",
            actor_id="reconciler",
        )
    task = db.get_task("task-1", include_deleted=True)
    assert task["projection_status"] == "deleted"  # nosec B101
    assert task["version"] == deleted["version"]  # nosec B101


def test_mark_task_unlinked_rejects_soft_deleted_parent_note(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)
    _set_projection(db, "task-1", note_id)
    assert db.soft_delete_note(note_id, expected_version=1) is True  # nosec B101

    with pytest.raises(ConflictError, match="note.*deleted"):
        db.mark_task_unlinked(
            task_id="task-1",
            expected_version=1,
            actor_type="system",
            actor_id="reconciler",
        )
    task = db.get_task("task-1", include_deleted=True)
    assert task["projection_status"] == "live"  # nosec B101
    assert task["version"] == 1  # nosec B101


def test_mark_task_unlinked_rejects_missing_live_projection_row(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)
    _set_projection(db, "task-1", note_id)
    _delete_projection(db, "task-1")

    with pytest.raises(ConflictError, match="projection.*missing"):
        db.mark_task_unlinked(
            task_id="task-1",
            expected_version=1,
            actor_type="system",
            actor_id="reconciler",
        )
    task = db.get_task("task-1", include_deleted=True)
    assert task["projection_status"] == "live"  # nosec B101
    assert task["version"] == 1  # nosec B101
    assert _event_count(db, "task-1", "unlinked") == 0  # nosec B101


def test_mark_task_unlinked_rejects_projection_update_zero_rowcount(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)
    _set_projection(db, "task-1", note_id)
    original_execute = db.task_store._execute
    race_applied = False

    def _simulate_projection_delete_race(conn, query, params=None):
        nonlocal race_applied
        if not race_applied and "UPDATE task_note_projections" in query:
            race_applied = True
            original_execute(conn, "DELETE FROM task_note_projections WHERE task_id = ?", ("task-1",))
        return original_execute(conn, query, params)

    monkeypatch.setattr(db.task_store, "_execute", _simulate_projection_delete_race)

    with pytest.raises(ConflictError, match="projection.*changed"):
        db.mark_task_unlinked(
            task_id="task-1",
            expected_version=1,
            actor_type="system",
            actor_id="reconciler",
        )
    task = db.get_task("task-1", include_deleted=True)
    assert task["projection_status"] == "live"  # nosec B101
    assert task["version"] == 1  # nosec B101
    assert _projection_count(db, "task-1") == 1  # nosec B101
    assert _event_count(db, "task-1", "unlinked") == 0  # nosec B101


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


def test_soft_delete_task_rejects_soft_deleted_parent_note(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)
    _set_projection(db, "task-1", note_id)
    assert db.soft_delete_note(note_id, expected_version=1) is True  # nosec B101

    with pytest.raises(ConflictError, match="note.*deleted"):
        db.soft_delete_task(
            task_id="task-1",
            expected_version=1,
            projection_note_id=note_id,
            projection_note_version=1,
            projection_line_number=1,
            actor_type="user",
            actor_id="user-1",
        )
    task = db.get_task("task-1", include_deleted=True)
    delete_events = [event for event in db.list_task_activity(task_id="task-1") if event["event_type"] == "deleted"]
    assert task["deleted"] in (0, False)  # nosec B101
    assert task["version"] == 1  # nosec B101
    assert delete_events == []  # nosec B101


def test_soft_delete_task_rejects_repeat_delete_without_new_version_or_event(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)
    _set_projection(db, "task-1", note_id)
    deleted = db.soft_delete_task(
        task_id="task-1",
        expected_version=1,
        projection_note_id=note_id,
        projection_note_version=1,
        projection_line_number=1,
        actor_type="user",
        actor_id="user-1",
    )

    with pytest.raises(ConflictError, match="deleted"):
        db.soft_delete_task(
            task_id="task-1",
            expected_version=deleted["version"],
            allow_record_only=True,
            actor_type="user",
            actor_id="user-1",
        )
    task = db.get_task("task-1", include_deleted=True)
    delete_events = [event for event in db.list_task_activity(task_id="task-1") if event["event_type"] == "deleted"]
    assert task["version"] == deleted["version"]  # nosec B101
    assert len(delete_events) == 1  # nosec B101


def test_set_task_projection_rejects_deleted_tasks(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)
    _set_projection(db, "task-1", note_id)
    deleted = db.soft_delete_task(
        task_id="task-1",
        expected_version=1,
        projection_note_id=note_id,
        projection_note_version=1,
        projection_line_number=1,
        actor_type="user",
        actor_id="user-1",
    )

    with pytest.raises(ConflictError, match="deleted"):
        db.set_task_projection(
            task_id="task-1",
            note_id=note_id,
            note_version=2,
            line_number=1,
            start_offset=0,
            end_offset=20,
            normalized_text_hash="sha256:review-again",
            occurrence_index=0,
            block_fingerprint="block-2",
            raw_line="- [ ] Review source",
            has_child_content=False,
            projection_status="live",
        )
    task = db.get_task("task-1", include_deleted=True)
    assert task["projection_status"] == "deleted"  # nosec B101
    assert task["version"] == deleted["version"]  # nosec B101


def test_set_task_projection_rejects_soft_deleted_parent_note(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)
    assert db.soft_delete_note(note_id, expected_version=1) is True  # nosec B101

    with pytest.raises(ConflictError, match="note.*deleted"):
        db.set_task_projection(
            task_id="task-1",
            note_id=note_id,
            note_version=2,
            line_number=1,
            start_offset=0,
            end_offset=20,
            normalized_text_hash="sha256:review-again",
            occurrence_index=0,
            block_fingerprint="block-2",
            raw_line="- [ ] Review source",
            has_child_content=False,
            projection_status="live",
        )
    assert _projection_count(db, "task-1") == 0  # nosec B101


def test_set_task_projection_rejects_projection_status_race(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)
    _set_projection(db, "task-1", note_id)
    original_execute = db.task_store._execute
    race_applied = False

    def _simulate_projection_status_race(conn, query, params=None):
        nonlocal race_applied
        if not race_applied and "UPDATE note_tasks" in query and "SET projection_status" in query:
            race_applied = True
            original_execute(
                conn,
                "UPDATE note_tasks SET projection_status = ? WHERE id = ?",
                ("unlinked", "task-1"),
            )
        return original_execute(conn, query, params)

    monkeypatch.setattr(db.task_store, "_execute", _simulate_projection_status_race)

    with pytest.raises(ConflictError, match="projection.*changed"):
        db.set_task_projection(
            task_id="task-1",
            note_id=note_id,
            note_version=2,
            line_number=1,
            start_offset=0,
            end_offset=20,
            normalized_text_hash="sha256:review-race",
            occurrence_index=0,
            block_fingerprint="block-race",
            raw_line="- [ ] Review source",
            has_child_content=False,
            projection_status="live",
        )
    task = db.get_task("task-1", include_deleted=True)
    assert task["projection_status"] == "live"  # nosec B101
    assert task["version"] == 1  # nosec B101


def test_set_task_projection_bumps_version_when_projection_status_changes(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)
    _set_projection(db, "task-1", note_id)

    db.set_task_projection(
        task_id="task-1",
        note_id=note_id,
        note_version=2,
        line_number=1,
        start_offset=0,
        end_offset=20,
        normalized_text_hash="sha256:review-ambiguous",
        occurrence_index=0,
        block_fingerprint="block-ambiguous",
        raw_line="- [ ] Review source",
        has_child_content=False,
        projection_status="ambiguous",
    )

    task = db.get_task("task-1", include_deleted=True)
    assert task["projection_status"] == "ambiguous"  # nosec B101
    assert task["version"] == 2  # nosec B101


def test_set_task_projection_refresh_does_not_bump_live_task_version(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)
    _set_projection(db, "task-1", note_id)

    db.set_task_projection(
        task_id="task-1",
        note_id=note_id,
        note_version=2,
        line_number=2,
        start_offset=1,
        end_offset=21,
        normalized_text_hash="sha256:review-refresh",
        occurrence_index=0,
        block_fingerprint="block-refresh",
        raw_line="- [ ] Review source",
        has_child_content=False,
        projection_status="live",
    )

    task = db.get_task("task-1", include_deleted=True)
    assert task["projection_status"] == "live"  # nosec B101
    assert task["version"] == 1  # nosec B101


def test_soft_delete_task_rejects_ambiguous_projection_without_mutation(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)
    _set_projection(db, "task-1", note_id, projection_status="ambiguous")

    with pytest.raises(ConflictError, match="ambiguous"):
        db.soft_delete_task(
            task_id="task-1",
            expected_version=2,
            projection_note_id=note_id,
            projection_note_version=1,
            projection_line_number=1,
            actor_type="user",
            actor_id="user-1",
        )
    task = db.get_task("task-1", include_deleted=True)
    delete_events = [event for event in db.list_task_activity(task_id="task-1") if event["event_type"] == "deleted"]
    assert task["deleted"] in (0, False)  # nosec B101
    assert task["projection_status"] == "ambiguous"  # nosec B101
    assert task["version"] == 2  # nosec B101
    assert delete_events == []  # nosec B101


def test_soft_delete_task_rejects_drifted_unlinked_projection_status(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)
    _set_projection(db, "task-1", note_id)
    _force_projection_status_drift(
        db,
        "task-1",
        task_status="live",
        projection_status="unlinked",
    )

    with pytest.raises(ConflictError, match="projection status mismatch"):
        db.soft_delete_task(
            task_id="task-1",
            expected_version=1,
            projection_note_id=note_id,
            projection_note_version=1,
            projection_line_number=1,
            actor_type="user",
            actor_id="user-1",
        )
    task = db.get_task("task-1", include_deleted=True)
    delete_events = [event for event in db.list_task_activity(task_id="task-1") if event["event_type"] == "deleted"]
    assert task["deleted"] in (0, False)  # nosec B101
    assert task["version"] == 1  # nosec B101
    assert delete_events == []  # nosec B101


def test_set_task_projection_rejects_deleted_projection_status_for_active_task(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)
    _set_projection(db, "task-1", note_id)

    with pytest.raises(InputError, match="deleted"):
        db.set_task_projection(
            task_id="task-1",
            note_id=note_id,
            note_version=2,
            line_number=1,
            start_offset=0,
            end_offset=20,
            normalized_text_hash="sha256:review-deleted",
            occurrence_index=0,
            block_fingerprint="block-deleted",
            raw_line="- [ ] Review source",
            has_child_content=False,
            projection_status="deleted",
        )
    task = db.get_task("task-1", include_deleted=True)
    assert task["deleted"] in (0, False)  # nosec B101
    assert task["projection_status"] == "live"  # nosec B101


def test_set_task_projection_requires_task_note_ownership(db: CharactersRAGDB) -> None:
    owning_note_id = _create_note(db, content="- [ ] Owning note\n")
    other_note_id = _create_note(db, content="- [ ] Other note\n")
    _create_task(db, owning_note_id)

    with pytest.raises(ConflictError, match="owning note"):
        db.set_task_projection(
            task_id="task-1",
            note_id=other_note_id,
            note_version=1,
            line_number=1,
            start_offset=0,
            end_offset=16,
            normalized_text_hash="sha256:other",
            occurrence_index=0,
            block_fingerprint="block-other",
            raw_line="- [ ] Other note",
            has_child_content=False,
        )
    deleted = db.soft_delete_task(
        task_id="task-1",
        expected_version=1,
        allow_record_only=True,
        actor_type="user",
        actor_id="user-1",
    )
    assert deleted["deleted"] in (1, True)  # nosec B101


def test_set_task_projection_rejects_drifted_projection_note_ownership(db: CharactersRAGDB) -> None:
    owning_note_id = _create_note(db, content="- [ ] Owning note\n")
    other_note_id = _create_note(db, content="- [ ] Other note\n")
    _create_task(db, owning_note_id)
    _set_projection(db, "task-1", owning_note_id)
    _force_projection_note_drift(db, "task-1", other_note_id)

    with pytest.raises(ConflictError, match="projection ownership mismatch"):
        db.set_task_projection(
            task_id="task-1",
            note_id=owning_note_id,
            note_version=2,
            line_number=1,
            start_offset=0,
            end_offset=20,
            normalized_text_hash="sha256:review-again",
            occurrence_index=0,
            block_fingerprint="block-2",
            raw_line="- [ ] Review source",
            has_child_content=False,
            projection_status="live",
        )
    task = db.get_task("task-1", include_deleted=True)
    assert task["projection_status"] == "live"  # nosec B101
    assert task["version"] == 1  # nosec B101


def test_soft_delete_task_rejects_drifted_projection_note_ownership(db: CharactersRAGDB) -> None:
    owning_note_id = _create_note(db, content="- [ ] Owning note\n")
    other_note_id = _create_note(db, content="- [ ] Other note\n")
    _create_task(db, owning_note_id)
    _set_projection(db, "task-1", owning_note_id)
    _force_projection_note_drift(db, "task-1", other_note_id)

    with pytest.raises(ConflictError, match="projection ownership mismatch"):
        db.soft_delete_task(
            task_id="task-1",
            expected_version=1,
            projection_note_id=owning_note_id,
            projection_note_version=1,
            projection_line_number=1,
            actor_type="user",
            actor_id="user-1",
        )
    task = db.get_task("task-1", include_deleted=True)
    delete_events = [event for event in db.list_task_activity(task_id="task-1") if event["event_type"] == "deleted"]
    assert task["deleted"] in (0, False)  # nosec B101
    assert task["version"] == 1  # nosec B101
    assert delete_events == []  # nosec B101


def test_soft_delete_task_rejects_missing_live_projection_row(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)
    _set_projection(db, "task-1", note_id)
    _delete_projection(db, "task-1")

    with pytest.raises(ConflictError, match="projection.*missing"):
        db.soft_delete_task(
            task_id="task-1",
            expected_version=1,
            projection_note_id=note_id,
            projection_note_version=1,
            projection_line_number=1,
            actor_type="user",
            actor_id="user-1",
        )
    task = db.get_task("task-1", include_deleted=True)
    assert task["deleted"] in (0, False)  # nosec B101
    assert task["projection_status"] == "live"  # nosec B101
    assert task["version"] == 1  # nosec B101
    assert _event_count(db, "task-1", "deleted") == 0  # nosec B101


def test_record_only_soft_delete_allows_missing_live_projection_row(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)
    _set_projection(db, "task-1", note_id)
    _delete_projection(db, "task-1")

    deleted = db.soft_delete_task(
        task_id="task-1",
        expected_version=1,
        allow_record_only=True,
        actor_type="user",
        actor_id="user-1",
    )

    assert deleted["deleted"] in (1, True)  # nosec B101
    assert deleted["projection_status"] == "deleted"  # nosec B101
    assert deleted["version"] == 2  # nosec B101
    assert _projection_count(db, "task-1") == 0  # nosec B101


def test_soft_delete_task_rejects_projection_update_zero_rowcount(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)
    _set_projection(db, "task-1", note_id)
    original_execute = db.task_store._execute
    race_applied = False

    def _simulate_projection_delete_race(conn, query, params=None):
        nonlocal race_applied
        if not race_applied and "UPDATE task_note_projections" in query:
            race_applied = True
            original_execute(conn, "DELETE FROM task_note_projections WHERE task_id = ?", ("task-1",))
        return original_execute(conn, query, params)

    monkeypatch.setattr(db.task_store, "_execute", _simulate_projection_delete_race)

    with pytest.raises(ConflictError, match="projection.*changed"):
        db.soft_delete_task(
            task_id="task-1",
            expected_version=1,
            projection_note_id=note_id,
            projection_note_version=1,
            projection_line_number=1,
            actor_type="user",
            actor_id="user-1",
        )
    task = db.get_task("task-1", include_deleted=True)
    assert task["deleted"] in (0, False)  # nosec B101
    assert task["projection_status"] == "live"  # nosec B101
    assert task["version"] == 1  # nosec B101
    assert _projection_count(db, "task-1") == 1  # nosec B101
    assert _event_count(db, "task-1", "deleted") == 0  # nosec B101


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
        if "UPDATE note_tasks" in query and "version = version + 1" in query:
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


def test_soft_delete_task_rejects_unlinked_without_record_only(db: CharactersRAGDB) -> None:
    note_id = _create_note(db)
    _create_task(db, note_id)
    _set_projection(db, "task-1", note_id)
    unlinked = db.mark_task_unlinked(
        task_id="task-1",
        expected_version=1,
        actor_type="system",
        actor_id="reconciler",
    )

    with pytest.raises(ConflictError, match="unlinked"):
        db.soft_delete_task(
            task_id="task-1",
            expected_version=unlinked["version"],
            actor_type="user",
            actor_id="user-1",
        )
    task = db.get_task("task-1", include_deleted=True)
    delete_events = [event for event in db.list_task_activity(task_id="task-1") if event["event_type"] == "deleted"]
    assert task["deleted"] in (0, False)  # nosec B101
    assert task["version"] == unlinked["version"]  # nosec B101
    assert delete_events == []  # nosec B101


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


def test_set_reconciliation_state_maps_missing_note_fk_to_conflict(db: CharactersRAGDB) -> None:
    with pytest.raises(ConflictError, match="note"):
        db.set_reconciliation_state(
            note_id="missing-note",
            note_version=1,
            status="clean",
            item_count=0,
            warning_count=0,
        )


def test_record_task_event_maps_duplicate_event_id_to_conflict(db: CharactersRAGDB) -> None:
    db.record_task_event(
        event_id="event-1",
        event_type="created",
        actor_type="user",
        actor_id="user-1",
    )

    with pytest.raises(ConflictError, match="event.*already exists"):
        db.record_task_event(
            event_id="event-1",
            event_type="created",
            actor_type="user",
            actor_id="user-1",
        )


def test_record_task_event_maps_missing_task_fk_to_conflict(db: CharactersRAGDB) -> None:
    with pytest.raises(ConflictError, match="event.*reference"):
        db.record_task_event(
            task_id="missing-task",
            event_type="updated",
            actor_type="user",
            actor_id="user-1",
        )


def test_mark_task_activity_read_maps_missing_event_fk_to_conflict(db: CharactersRAGDB) -> None:
    with pytest.raises(ConflictError, match="event"):
        db.mark_task_activity_read("missing-event", user_id="user-1")


def test_mark_task_activity_dismissed_maps_missing_event_fk_to_conflict(db: CharactersRAGDB) -> None:
    with pytest.raises(ConflictError, match="event"):
        db.mark_task_activity_dismissed("missing-event", user_id="user-1")


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
