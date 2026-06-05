"""Integration coverage for the note-backed tasks REST API."""

from __future__ import annotations

import importlib
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_rate_limiter_dep
from tldw_Server_API.app.api.v1.endpoints import notes as notes_endpoint
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, InputError
from tldw_Server_API.app.core.Notes_Tasks.service import NotesTaskService

pytestmark = pytest.mark.integration


class _NoopRateLimiter:
    async def check_user_rate_limit(self, *_args: Any, **_kwargs: Any) -> tuple[bool, dict[str, Any]]:
        return True, {}


@pytest.fixture()
def notes_tasks_api_client(tmp_path: Path) -> Generator[tuple[TestClient, CharactersRAGDB], None, None]:
    db = CharactersRAGDB(str(tmp_path / "notes_tasks_rest_api.db"), client_id="notes_tasks_rest_user")

    async def override_user() -> User:
        return User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)

    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user

    def override_db_dep() -> CharactersRAGDB:
        return db

    fastapi_app = FastAPI()
    try:
        notes_tasks_endpoint = importlib.import_module("tldw_Server_API.app.api.v1.endpoints.notes_tasks")
    except ModuleNotFoundError:
        notes_tasks_endpoint = None
    if notes_tasks_endpoint is not None:
        fastapi_app.include_router(notes_tasks_endpoint.router, prefix="/api/v1/notes")
    fastapi_app.include_router(notes_endpoint.router, prefix="/api/v1/notes")
    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[get_chacha_db_for_user] = override_db_dep
    fastapi_app.dependency_overrides[get_rate_limiter_dep] = lambda: _NoopRateLimiter()

    try:
        with TestClient(fastapi_app) as client:
            yield client, db
    finally:
        fastapi_app.dependency_overrides.clear()
        db.close_connection()


def _create_note(client: TestClient, *, content: str, title: str = "Tasks") -> dict[str, Any]:
    response = client.post("/api/v1/notes/", json={"title": title, "content": content})
    assert response.status_code == 201, response.text
    return response.json()


def _task_by_text(db: CharactersRAGDB, note_id: str, text: str) -> dict[str, Any]:
    tasks = db.list_tasks(note_id=note_id, include_deleted=True, limit=100)
    matches = [task for task in tasks if task["text"] == text]
    assert len(matches) == 1
    return matches[0]


def _task_with_projection(db: CharactersRAGDB, task_id: str) -> dict[str, Any]:
    task = db.get_task(task_id)
    assert task is not None
    projection = db.task_store._fetch_projection(task_id)
    assert projection is not None
    return {"task": task, "projection": projection}


def _status_update(
    client: TestClient,
    *,
    task: dict[str, Any],
    status: str,
    expected_note_version: int | None = None,
    record_only: bool = False,
) -> Any:
    item: dict[str, Any] = {
        "task_id": task["id"],
        "status": status,
        "expected_task_version": task["version"],
        "record_only": record_only,
    }
    if expected_note_version is not None:
        item["expected_note_version"] = expected_note_version
    return client.post("/api/v1/notes/tasks/status", json={"updates": [item]})


def test_list_tasks_for_note_reconciles_stale_note(notes_tasks_api_client: tuple[TestClient, CharactersRAGDB]) -> None:
    client, db = notes_tasks_api_client
    note = _create_note(client, content="- [ ] Alpha\n")
    db.update_note(
        note_id=note["id"],
        update_data={"content": "- [x] Alpha\n- [ ] Beta\n"},
        expected_version=note["version"],
    )

    response = client.get(f"/api/v1/notes/{note['id']}/tasks")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["reconciliation"]["note_id"] == note["id"]
    assert payload["reconciliation"]["note_version"] == note["version"] + 1
    assert payload["reconciliation"]["status"] == "clean"
    assert {task["text"]: task["status"] for task in payload["tasks"]} == {
        "Alpha": "done",
        "Beta": "open",
    }


def test_broad_list_reports_incomplete_reconciliation_when_work_limit_reached(
    notes_tasks_api_client: tuple[TestClient, CharactersRAGDB],
) -> None:
    client, db = notes_tasks_api_client
    first = _create_note(client, title="First", content="- [ ] First\n")
    second = _create_note(client, title="Second", content="- [ ] Second\n")
    db.update_note(note_id=first["id"], update_data={"content": "- [x] First\n"}, expected_version=first["version"])
    db.update_note(note_id=second["id"], update_data={"content": "- [x] Second\n"}, expected_version=second["version"])

    response = client.get("/api/v1/notes/tasks", params={"reconcile_limit": 1, "limit": 50})

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["reconciliation"]["status"] == "incomplete"
    assert payload["reconciliation"]["processed_notes"] == 1
    assert payload["reconciliation"]["remaining_stale_notes"] >= 1


def test_get_task_includes_note_and_projection_details(
    notes_tasks_api_client: tuple[TestClient, CharactersRAGDB],
) -> None:
    client, db = notes_tasks_api_client
    note = _create_note(client, content="- [ ] Alpha @priority(high)\n")
    task = _task_by_text(db, note["id"], "Alpha")

    response = client.get(f"/api/v1/notes/tasks/{task['id']}")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["id"] == task["id"]
    assert payload["metadata"] == {"priority": "high"}
    assert payload["note"] == {"id": note["id"], "title": "Tasks", "version": note["version"]}
    assert payload["projection"]["note_version"] == note["version"]
    assert payload["projection"]["line_number"] == 1
    assert payload["projection"]["raw_line"] == "- [ ] Alpha @priority(high)"


def test_create_task_requires_expected_note_version_and_inserts_checklist_line(
    notes_tasks_api_client: tuple[TestClient, CharactersRAGDB],
) -> None:
    client, db = notes_tasks_api_client
    note = _create_note(client, content="Intro\n")

    missing_version = client.post(f"/api/v1/notes/{note['id']}/tasks", json={"text": "Alpha"})
    assert missing_version.status_code == 422

    response = client.post(
        f"/api/v1/notes/{note['id']}/tasks",
        json={
            "text": "Alpha",
            "metadata": {"due_date": "2026-06-30", "priority": "high"},
            "expected_note_version": note["version"],
        },
    )

    assert response.status_code == 201, response.text
    payload = response.json()
    saved = db.get_note_by_id(note["id"])
    assert saved is not None
    assert saved["version"] == note["version"] + 1
    assert saved["content"] == "Intro\n\n- [ ] Alpha @due(2026-06-30) @priority(high)\n"
    assert payload["text"] == "Alpha"
    assert payload["note"]["version"] == saved["version"]


def test_set_status_on_clean_note_rewrites_marker_and_records_event(
    notes_tasks_api_client: tuple[TestClient, CharactersRAGDB],
) -> None:
    client, db = notes_tasks_api_client
    note = _create_note(client, content="- [ ] Alpha\n")
    task = _task_by_text(db, note["id"], "Alpha")

    response = _status_update(client, task=task, status="done", expected_note_version=note["version"])

    assert response.status_code == 200, response.text
    saved = db.get_note_by_id(note["id"])
    assert saved is not None
    updated = response.json()["tasks"][0]
    assert saved["content"] == "- [x] Alpha\n"
    assert updated["status"] == "done"
    assert updated["version"] == task["version"] + 1
    events = db.list_task_activity(task_id=task["id"], limit=20)
    assert any(event["event_type"] == "status_changed" for event in events)


def test_status_update_preserves_unknown_checklist_tokens(
    notes_tasks_api_client: tuple[TestClient, CharactersRAGDB],
) -> None:
    client, db = notes_tasks_api_client
    note = _create_note(client, content="- [ ] Pay rent @estimate(2h) @external(abc) @priority(high)\n")
    task = _task_by_text(db, note["id"], "Pay rent @external(abc)")

    response = _status_update(client, task=task, status="done", expected_note_version=note["version"])

    assert response.status_code == 200, response.text
    saved = db.get_note_by_id(note["id"])
    assert saved is not None
    assert saved["content"] == "- [x] Pay rent @estimate(2h) @external(abc) @priority(high)\n"
    assert response.json()["tasks"][0]["status"] == "done"


def test_batch_status_update_rolls_back_when_later_item_conflicts(
    notes_tasks_api_client: tuple[TestClient, CharactersRAGDB],
) -> None:
    client, db = notes_tasks_api_client
    note = _create_note(client, content="- [ ] Alpha\n- [ ] Beta\n")
    alpha = _task_by_text(db, note["id"], "Alpha")
    beta = _task_by_text(db, note["id"], "Beta")

    response = client.post(
        "/api/v1/notes/tasks/status",
        json={
            "updates": [
                {
                    "task_id": alpha["id"],
                    "status": "done",
                    "expected_task_version": alpha["version"],
                    "expected_note_version": note["version"],
                },
                {
                    "task_id": beta["id"],
                    "status": "done",
                    "expected_task_version": beta["version"],
                    "expected_note_version": note["version"],
                },
            ],
        },
    )

    assert response.status_code == 409, response.text
    saved = db.get_note_by_id(note["id"])
    assert saved is not None
    assert saved["version"] == note["version"]
    assert saved["content"] == "- [ ] Alpha\n- [ ] Beta\n"
    assert _task_by_text(db, note["id"], "Alpha")["status"] == "open"
    assert _task_by_text(db, note["id"], "Beta")["status"] == "open"


def test_update_text_requires_expected_task_and_note_versions(
    notes_tasks_api_client: tuple[TestClient, CharactersRAGDB],
) -> None:
    client, db = notes_tasks_api_client
    note = _create_note(client, content="- [ ] Alpha\n")
    task = _task_by_text(db, note["id"], "Alpha")

    missing_task_version = client.patch(
        f"/api/v1/notes/tasks/{task['id']}",
        json={"text": "Beta", "expected_note_version": note["version"]},
    )
    assert missing_task_version.status_code == 422

    stale_note_version = client.patch(
        f"/api/v1/notes/tasks/{task['id']}",
        json={"text": "Beta", "expected_task_version": task["version"], "expected_note_version": note["version"] + 1},
    )
    assert stale_note_version.status_code == 409

    response = client.patch(
        f"/api/v1/notes/tasks/{task['id']}",
        json={"text": "Beta", "expected_task_version": task["version"], "expected_note_version": note["version"]},
    )

    assert response.status_code == 200, response.text
    saved = db.get_note_by_id(note["id"])
    assert saved is not None
    assert saved["content"] == "- [ ] Beta\n"
    assert response.json()["text"] == "Beta"


def test_update_projected_task_refreshes_sibling_projection(
    notes_tasks_api_client: tuple[TestClient, CharactersRAGDB],
) -> None:
    client, db = notes_tasks_api_client
    note = _create_note(client, content="- [ ] Alpha\n- [ ] Beta\n")
    alpha = _task_by_text(db, note["id"], "Alpha")
    beta = _task_by_text(db, note["id"], "Beta")

    alpha_response = client.patch(
        f"/api/v1/notes/tasks/{alpha['id']}",
        json={
            "text": "Alpha updated",
            "expected_task_version": alpha["version"],
            "expected_note_version": note["version"],
        },
    )
    assert alpha_response.status_code == 200, alpha_response.text
    saved = db.get_note_by_id(note["id"])
    assert saved is not None
    beta_after_alpha = db.get_task(beta["id"])
    assert beta_after_alpha is not None
    beta_projection = db.task_store._fetch_projection(beta["id"])
    assert beta_projection is not None
    assert beta_projection["note_version"] == saved["version"]

    beta_response = client.patch(
        f"/api/v1/notes/tasks/{beta['id']}",
        json={
            "text": "Beta updated",
            "expected_task_version": beta_after_alpha["version"],
            "expected_note_version": saved["version"],
        },
    )

    assert beta_response.status_code == 200, beta_response.text
    saved_after_beta = db.get_note_by_id(note["id"])
    assert saved_after_beta is not None
    assert saved_after_beta["content"] == "- [ ] Alpha updated\n- [ ] Beta updated\n"
    assert len(db.list_tasks(note_id=note["id"], include_deleted=True, limit=20)) == 2


def test_delete_projected_task_refreshes_sibling_projection(
    notes_tasks_api_client: tuple[TestClient, CharactersRAGDB],
) -> None:
    client, db = notes_tasks_api_client
    note = _create_note(client, content="- [ ] Alpha\n- [ ] Beta\n")
    alpha = _task_by_text(db, note["id"], "Alpha")
    beta = _task_by_text(db, note["id"], "Beta")

    delete_response = client.request(
        "DELETE",
        f"/api/v1/notes/tasks/{alpha['id']}",
        json={"expected_task_version": alpha["version"], "expected_note_version": note["version"]},
    )
    assert delete_response.status_code == 200, delete_response.text
    saved = db.get_note_by_id(note["id"])
    assert saved is not None
    beta_after_delete = db.get_task(beta["id"])
    assert beta_after_delete is not None
    beta_projection = db.task_store._fetch_projection(beta["id"])
    assert beta_projection is not None
    assert beta_projection["note_version"] == saved["version"]

    beta_response = client.patch(
        f"/api/v1/notes/tasks/{beta['id']}",
        json={
            "text": "Beta updated",
            "expected_task_version": beta_after_delete["version"],
            "expected_note_version": saved["version"],
        },
    )

    assert beta_response.status_code == 200, beta_response.text
    saved_after_beta = db.get_note_by_id(note["id"])
    assert saved_after_beta is not None
    assert saved_after_beta["content"] == "- [ ] Beta updated\n"


def test_create_task_rejects_newline_text(notes_tasks_api_client: tuple[TestClient, CharactersRAGDB]) -> None:
    client, _db = notes_tasks_api_client
    note = _create_note(client, content="Intro\n")

    response = client.post(
        f"/api/v1/notes/{note['id']}/tasks",
        json={"text": "Alpha\n- [ ] Injected", "expected_note_version": note["version"]},
    )

    assert response.status_code == 422, response.text


def test_update_task_rejects_newline_text(notes_tasks_api_client: tuple[TestClient, CharactersRAGDB]) -> None:
    client, db = notes_tasks_api_client
    note = _create_note(client, content="- [ ] Alpha\n")
    task = _task_by_text(db, note["id"], "Alpha")

    response = client.patch(
        f"/api/v1/notes/tasks/{task['id']}",
        json={
            "text": "Beta\r\n- [ ] Injected",
            "expected_task_version": task["version"],
            "expected_note_version": note["version"],
        },
    )

    assert response.status_code == 422, response.text


def test_service_validation_rejects_newline_text() -> None:
    with pytest.raises(InputError):
        NotesTaskService._validate_task_text("Alpha\n- [ ] Injected")


def test_metadata_update_preserves_unknown_tokens(notes_tasks_api_client: tuple[TestClient, CharactersRAGDB]) -> None:
    client, db = notes_tasks_api_client
    note = _create_note(client, content="- [ ] Alpha @foo(bar) @priority(low)\n")
    task = _task_by_text(db, note["id"], "Alpha @foo(bar)")

    response = client.patch(
        f"/api/v1/notes/tasks/{task['id']}",
        json={
            "metadata": {"priority": "high", "estimate": "2h"},
            "expected_task_version": task["version"],
            "expected_note_version": note["version"],
        },
    )

    assert response.status_code == 200, response.text
    saved = db.get_note_by_id(note["id"])
    assert saved is not None
    assert saved["content"] == "- [ ] Alpha @foo(bar) @priority(high) @estimate(2h)\n"
    assert response.json()["metadata"] == {"estimate": "2h", "priority": "high"}


def test_metadata_update_preserves_malformed_allowlisted_tokens(
    notes_tasks_api_client: tuple[TestClient, CharactersRAGDB],
) -> None:
    client, db = notes_tasks_api_client
    note = _create_note(client, content="- [ ] Alpha @due(not-a-date) @foo(bar) @priority(low)\n")
    task = _task_by_text(db, note["id"], "Alpha @due(not-a-date) @foo(bar)")

    response = client.patch(
        f"/api/v1/notes/tasks/{task['id']}",
        json={
            "metadata": {"priority": "high", "estimate": "2h"},
            "expected_task_version": task["version"],
            "expected_note_version": note["version"],
        },
    )

    assert response.status_code == 200, response.text
    saved = db.get_note_by_id(note["id"])
    assert saved is not None
    assert saved["content"] == "- [ ] Alpha @due(not-a-date) @foo(bar) @priority(high) @estimate(2h)\n"


def test_metadata_rejects_unknown_keys(notes_tasks_api_client: tuple[TestClient, CharactersRAGDB]) -> None:
    client, _db = notes_tasks_api_client
    note = _create_note(client, content="Intro\n")

    response = client.post(
        f"/api/v1/notes/{note['id']}/tasks",
        json={
            "text": "Alpha",
            "metadata": {"owner": "tester"},
            "expected_note_version": note["version"],
        },
    )

    assert response.status_code == 422, response.text


def test_metadata_rejects_impossible_due_date(notes_tasks_api_client: tuple[TestClient, CharactersRAGDB]) -> None:
    client, _db = notes_tasks_api_client
    note = _create_note(client, content="Intro\n")

    response = client.post(
        f"/api/v1/notes/{note['id']}/tasks",
        json={
            "text": "Alpha",
            "metadata": {"due_date": "2026-02-31"},
            "expected_note_version": note["version"],
        },
    )

    assert response.status_code == 422, response.text


def test_service_validation_rejects_invalid_metadata_values() -> None:
    with pytest.raises(InputError):
        NotesTaskService._validate_metadata({"due_date": "2026-02-31"})
    with pytest.raises(InputError):
        NotesTaskService._validate_metadata({"due_date": "20260605"})
    with pytest.raises(InputError):
        NotesTaskService._validate_metadata({"due_date": "2026-W23-5"})
    with pytest.raises(InputError):
        NotesTaskService._validate_metadata({"priority": "urgent"})
    with pytest.raises(InputError):
        NotesTaskService._validate_metadata({"estimate": "soon"})


def test_delete_projected_task_removes_line_transactionally(
    notes_tasks_api_client: tuple[TestClient, CharactersRAGDB],
) -> None:
    client, db = notes_tasks_api_client
    note = _create_note(client, content="Before\n- [ ] Alpha\nAfter\n")
    task = _task_by_text(db, note["id"], "Alpha")

    response = client.request(
        "DELETE",
        f"/api/v1/notes/tasks/{task['id']}",
        json={"expected_task_version": task["version"], "expected_note_version": note["version"]},
    )

    assert response.status_code == 200, response.text
    saved = db.get_note_by_id(note["id"])
    assert saved is not None
    assert saved["content"] == "Before\nAfter\n"
    deleted = db.get_task(task["id"], include_deleted=True)
    assert deleted is not None
    assert bool(deleted["deleted"])


def test_delete_with_nested_child_content_conflicts(
    notes_tasks_api_client: tuple[TestClient, CharactersRAGDB],
) -> None:
    client, db = notes_tasks_api_client
    note = _create_note(client, content="- [ ] Parent\n  child detail\n")
    task = _task_by_text(db, note["id"], "Parent")

    response = client.request(
        "DELETE",
        f"/api/v1/notes/tasks/{task['id']}",
        json={"expected_task_version": task["version"], "expected_note_version": note["version"]},
    )

    assert response.status_code == 409, response.text
    saved = db.get_note_by_id(note["id"])
    assert saved is not None
    assert saved["content"] == "- [ ] Parent\n  child detail\n"
    assert db.get_task(task["id"]) is not None


def test_unlinked_task_record_only_delete_succeeds(
    notes_tasks_api_client: tuple[TestClient, CharactersRAGDB],
) -> None:
    client, db = notes_tasks_api_client
    note = _create_note(client, content="- [ ] Alpha\n")
    task = _task_by_text(db, note["id"], "Alpha")
    patched = client.patch(
        f"/api/v1/notes/{note['id']}",
        json={"content": "No checklist\n"},
        headers={"expected-version": str(note["version"])},
    )
    assert patched.status_code == 200, patched.text
    unlinked = db.get_task(task["id"])
    assert unlinked is not None
    assert unlinked["projection_status"] == "unlinked"

    response = client.request(
        "DELETE",
        f"/api/v1/notes/tasks/{task['id']}",
        json={"expected_task_version": unlinked["version"], "record_only": True},
    )

    assert response.status_code == 200, response.text
    deleted = db.get_task(task["id"], include_deleted=True)
    assert deleted is not None
    assert bool(deleted["deleted"])


def test_unlinked_task_projection_updates_conflict_unless_record_only_metadata(
    notes_tasks_api_client: tuple[TestClient, CharactersRAGDB],
) -> None:
    client, db = notes_tasks_api_client
    note = _create_note(client, content="- [ ] Alpha\n")
    task = _task_by_text(db, note["id"], "Alpha")
    patched = client.patch(
        f"/api/v1/notes/{note['id']}",
        json={"content": "No checklist\n"},
        headers={"expected-version": str(note["version"])},
    )
    assert patched.status_code == 200, patched.text
    unlinked = db.get_task(task["id"])
    assert unlinked is not None

    status_response = _status_update(client, task=unlinked, status="done", record_only=True)
    assert status_response.status_code == 409

    text_response = client.patch(
        f"/api/v1/notes/tasks/{task['id']}",
        json={"text": "Beta", "expected_task_version": unlinked["version"], "record_only": True},
    )
    assert text_response.status_code == 409

    metadata_response = client.patch(
        f"/api/v1/notes/tasks/{task['id']}",
        json={
            "metadata": {"priority": "medium"},
            "expected_task_version": unlinked["version"],
            "record_only": True,
        },
    )
    assert metadata_response.status_code == 200, metadata_response.text
    assert metadata_response.json()["metadata"] == {"priority": "medium"}


def test_ambiguous_projected_mutations_conflict(
    notes_tasks_api_client: tuple[TestClient, CharactersRAGDB],
) -> None:
    client, db = notes_tasks_api_client
    note = _create_note(client, content="- [ ] Alpha\n")
    task = _task_by_text(db, note["id"], "Alpha")
    with db.transaction() as conn:
        conn.execute("UPDATE note_tasks SET projection_status = ? WHERE id = ?", ("ambiguous", task["id"]))
        conn.execute(
            "UPDATE task_note_projections SET projection_status = ? WHERE task_id = ?",
            ("ambiguous", task["id"]),
        )
    ambiguous = db.get_task(task["id"])
    assert ambiguous is not None

    status_response = _status_update(client, task=ambiguous, status="done", expected_note_version=note["version"])
    assert status_response.status_code == 409

    text_response = client.patch(
        f"/api/v1/notes/tasks/{task['id']}",
        json={"text": "Beta", "expected_task_version": ambiguous["version"], "expected_note_version": note["version"]},
    )
    assert text_response.status_code == 409

    metadata_response = client.patch(
        f"/api/v1/notes/tasks/{task['id']}",
        json={
            "metadata": {"priority": "high"},
            "expected_task_version": ambiguous["version"],
            "expected_note_version": note["version"],
        },
    )
    assert metadata_response.status_code == 409

    delete_response = client.request(
        "DELETE",
        f"/api/v1/notes/tasks/{task['id']}",
        json={"expected_task_version": ambiguous["version"], "expected_note_version": note["version"]},
    )
    assert delete_response.status_code == 409


def test_recent_activity_returns_unread_agent_events_and_supports_dismissal(
    notes_tasks_api_client: tuple[TestClient, CharactersRAGDB],
) -> None:
    client, db = notes_tasks_api_client
    note = _create_note(client, content="- [ ] Alpha\n")
    task = _task_by_text(db, note["id"], "Alpha")
    db.record_task_event(
        task_id=task["id"],
        note_id=note["id"],
        event_type="updated",
        actor_type="agent",
        actor_id="assistant",
        new_value={"text": "Agent proposal"},
    )
    db.record_task_event(
        task_id=task["id"],
        note_id=note["id"],
        event_type="updated",
        actor_type="user",
        actor_id="tester",
        new_value={"text": "User update"},
    )

    response = client.get("/api/v1/notes/tasks/activity")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(payload["events"]) == 1
    event = payload["events"][0]
    assert event["actor_type"] == "agent"
    assert event["read_at"] is None
    assert event["dismissed_at"] is None

    dismiss = client.patch(
        f"/api/v1/notes/tasks/activity/{event['id']}",
        json={"read": True, "dismissed": True},
    )
    assert dismiss.status_code == 200, dismiss.text
    assert dismiss.json()["read_at"] is not None
    assert dismiss.json()["dismissed_at"] is not None

    after = client.get("/api/v1/notes/tasks/activity")
    assert after.status_code == 200, after.text
    assert after.json()["events"] == []


def test_recent_activity_includes_latest_agent_event_after_many_older_user_events(
    notes_tasks_api_client: tuple[TestClient, CharactersRAGDB],
) -> None:
    client, db = notes_tasks_api_client
    note = _create_note(client, content="- [ ] Alpha\n")
    task = _task_by_text(db, note["id"], "Alpha")
    for index in range(201):
        db.record_task_event(
            task_id=task["id"],
            note_id=note["id"],
            event_type="updated",
            actor_type="user",
            actor_id=f"user-{index}",
            new_value={"index": index},
        )
    latest_agent_event = db.record_task_event(
        task_id=task["id"],
        note_id=note["id"],
        event_type="updated",
        actor_type="agent",
        actor_id="assistant",
        new_value={"text": "latest agent event"},
    )

    response = client.get("/api/v1/notes/tasks/activity")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert [event["id"] for event in payload["events"]] == [latest_agent_event["id"]]


def test_recent_activity_limit_skips_newer_dismissed_agent_event(
    notes_tasks_api_client: tuple[TestClient, CharactersRAGDB],
) -> None:
    client, db = notes_tasks_api_client
    note = _create_note(client, content="- [ ] Alpha\n")
    task = _task_by_text(db, note["id"], "Alpha")
    older_unread_event = db.record_task_event(
        task_id=task["id"],
        note_id=note["id"],
        event_type="updated",
        actor_type="agent",
        actor_id="assistant",
        new_value={"text": "older unread event"},
    )
    newer_dismissed_event = db.record_task_event(
        task_id=task["id"],
        note_id=note["id"],
        event_type="updated",
        actor_type="agent",
        actor_id="assistant",
        new_value={"text": "newer dismissed event"},
    )
    dismissed = client.patch(
        f"/api/v1/notes/tasks/activity/{newer_dismissed_event['id']}",
        json={"read": True, "dismissed": True},
    )
    assert dismissed.status_code == 200, dismissed.text

    response = client.get("/api/v1/notes/tasks/activity", params={"limit": 1})

    assert response.status_code == 200, response.text
    assert [event["id"] for event in response.json()["events"]] == [older_unread_event["id"]]


def test_reconcile_note_endpoint_refreshes_state(
    notes_tasks_api_client: tuple[TestClient, CharactersRAGDB],
) -> None:
    client, db = notes_tasks_api_client
    note = _create_note(client, content="- [ ] Alpha\n")
    db.update_note(
        note_id=note["id"],
        update_data={"content": "- [x] Alpha\n"},
        expected_version=note["version"],
    )

    response = client.post(f"/api/v1/notes/{note['id']}/tasks/reconcile")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "clean"
    assert payload["note_version"] == note["version"] + 1
    assert _task_by_text(db, note["id"], "Alpha")["status"] == "done"
