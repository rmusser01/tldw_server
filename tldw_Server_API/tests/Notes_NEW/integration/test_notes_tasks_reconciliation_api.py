"""Integration coverage for note-save task reconciliation."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_rate_limiter_dep
from tldw_Server_API.app.api.v1.endpoints import notes as notes_endpoint
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration


class _NoopRateLimiter:
    async def check_user_rate_limit(self, *_args, **_kwargs):
        return True, {}


class _FailingNotesTaskService:
    def reconcile_note(self, **_kwargs):
        raise RuntimeError("injected reconciliation failure")


@pytest.fixture()
def notes_tasks_client(tmp_path: Path) -> Generator[tuple[TestClient, CharactersRAGDB], None, None]:
    db = CharactersRAGDB(str(tmp_path / "notes_tasks_api.db"), client_id="notes_tasks_api_user")

    async def override_user():
        return User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)

    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user

    def override_db_dep():
        return db

    fastapi_app = FastAPI()
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


def _tasks_by_text(db: CharactersRAGDB, note_id: str) -> dict[str, dict]:
    return {task["text"]: task for task in db.list_tasks(note_id=note_id, include_deleted=True, limit=100)}


def test_note_create_returns_saved_note_when_reconciliation_raises(
    notes_tasks_client: tuple[TestClient, CharactersRAGDB],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, db = notes_tasks_client
    monkeypatch.setattr(notes_endpoint, "_NOTES_TASK_SERVICE", _FailingNotesTaskService())

    create_response = client.post(
        "/api/v1/notes/",
        json={"title": "Tasks", "content": "- [ ] Alpha\n"},
    )

    assert create_response.status_code == 201, create_response.text
    created = create_response.json()
    saved_note = db.get_note_by_id(created["id"])
    assert saved_note is not None
    assert saved_note["content"] == "- [ ] Alpha\n"
    assert db.get_reconciliation_state(created["id"]) is None
    assert db.list_tasks(note_id=created["id"], include_deleted=True, limit=100) == []


def test_note_patch_returns_saved_note_when_reconciliation_raises(
    notes_tasks_client: tuple[TestClient, CharactersRAGDB],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, db = notes_tasks_client

    create_response = client.post(
        "/api/v1/notes/",
        json={"title": "Tasks", "content": "- [ ] Alpha\n"},
    )
    assert create_response.status_code == 201, create_response.text
    created = create_response.json()
    original_state = db.get_reconciliation_state(created["id"])
    assert original_state is not None
    monkeypatch.setattr(notes_endpoint, "_NOTES_TASK_SERVICE", _FailingNotesTaskService())

    patch_response = client.patch(
        f"/api/v1/notes/{created['id']}",
        json={"content": "- [x] Alpha\n"},
        headers={"expected-version": str(created["version"])},
    )

    assert patch_response.status_code == 200, patch_response.text
    patched = patch_response.json()
    saved_note = db.get_note_by_id(created["id"])
    assert patched["content"] == "- [x] Alpha\n"
    assert saved_note is not None
    assert saved_note["content"] == "- [x] Alpha\n"
    assert db.get_reconciliation_state(created["id"]) == original_state


def test_note_create_ignores_empty_checklist_placeholder_with_warning_state(
    notes_tasks_client: tuple[TestClient, CharactersRAGDB],
) -> None:
    client, db = notes_tasks_client

    create_response = client.post(
        "/api/v1/notes/",
        json={"title": "Tasks", "content": "- [ ]\n- [ ] Alpha\n"},
    )

    assert create_response.status_code == 201, create_response.text
    created = create_response.json()
    state = db.get_reconciliation_state(created["id"])
    tasks = db.list_tasks(note_id=created["id"], include_deleted=True, limit=100)
    assert [task["text"] for task in tasks] == ["Alpha"]
    assert state is not None
    assert state["status"] == "warnings"
    assert state["warning_count"] == 1


def test_title_only_put_advances_reconciliation_state_without_changing_tasks(
    notes_tasks_client: tuple[TestClient, CharactersRAGDB],
) -> None:
    client, db = notes_tasks_client

    create_response = client.post(
        "/api/v1/notes/",
        json={"title": "Tasks", "content": "- [ ] Alpha\n- [x] Beta\n"},
    )
    assert create_response.status_code == 201, create_response.text
    created = create_response.json()
    note_id = created["id"]
    original_tasks = _tasks_by_text(db, note_id)

    update_response = client.put(
        f"/api/v1/notes/{note_id}",
        json={"title": "Renamed Tasks"},
        headers={"expected-version": str(created["version"])},
    )

    assert update_response.status_code == 200, update_response.text
    updated = update_response.json()
    updated_state = db.get_reconciliation_state(note_id)
    updated_tasks = _tasks_by_text(db, note_id)
    assert updated["version"] == created["version"] + 1
    assert updated_state is not None
    assert updated_state["note_version"] == updated["version"]
    assert updated_tasks["Alpha"]["id"] == original_tasks["Alpha"]["id"]
    assert updated_tasks["Alpha"]["status"] == original_tasks["Alpha"]["status"]
    assert updated_tasks["Beta"]["id"] == original_tasks["Beta"]["id"]
    assert updated_tasks["Beta"]["status"] == original_tasks["Beta"]["status"]


def test_title_only_patch_advances_reconciliation_state_without_changing_tasks(
    notes_tasks_client: tuple[TestClient, CharactersRAGDB],
) -> None:
    client, db = notes_tasks_client

    create_response = client.post(
        "/api/v1/notes/",
        json={"title": "Tasks", "content": "- [ ] Alpha\n- [x] Beta\n"},
    )
    assert create_response.status_code == 201, create_response.text
    created = create_response.json()
    note_id = created["id"]
    original_tasks = _tasks_by_text(db, note_id)

    patch_response = client.patch(
        f"/api/v1/notes/{note_id}",
        json={"title": "Patched Tasks"},
        headers={"expected-version": str(created["version"])},
    )

    assert patch_response.status_code == 200, patch_response.text
    patched = patch_response.json()
    patched_state = db.get_reconciliation_state(note_id)
    patched_tasks = _tasks_by_text(db, note_id)
    assert patched["version"] == created["version"] + 1
    assert patched_state is not None
    assert patched_state["note_version"] == patched["version"]
    assert patched_tasks["Alpha"]["id"] == original_tasks["Alpha"]["id"]
    assert patched_tasks["Alpha"]["status"] == original_tasks["Alpha"]["status"]
    assert patched_tasks["Beta"]["id"] == original_tasks["Beta"]["id"]
    assert patched_tasks["Beta"]["status"] == original_tasks["Beta"]["status"]


def test_note_create_update_and_conflict_reconcile_tasks_after_successful_saves(
    notes_tasks_client: tuple[TestClient, CharactersRAGDB],
) -> None:
    client, db = notes_tasks_client

    create_response = client.post(
        "/api/v1/notes/",
        json={"title": "Tasks", "content": "- [ ] Alpha\n- [x] Beta\n"},
    )
    assert create_response.status_code == 201, create_response.text
    created = create_response.json()
    note_id = created["id"]

    created_state = db.get_reconciliation_state(note_id)
    created_tasks = _tasks_by_text(db, note_id)
    assert created_state is not None
    assert created_state["note_version"] == created["version"]
    assert created_state["status"] == "clean"
    assert {text: task["status"] for text, task in created_tasks.items()} == {
        "Alpha": "open",
        "Beta": "done",
    }

    update_response = client.put(
        f"/api/v1/notes/{note_id}",
        json={"content": "- [x] Alpha\n- [ ] Beta\n"},
        headers={"expected-version": str(created["version"])},
    )
    assert update_response.status_code == 200, update_response.text
    updated = update_response.json()
    updated_tasks = _tasks_by_text(db, note_id)
    updated_state = db.get_reconciliation_state(note_id)

    assert updated["version"] == created["version"] + 1
    assert updated_state is not None
    assert updated_state["note_version"] == updated["version"]
    assert updated_tasks["Alpha"]["id"] == created_tasks["Alpha"]["id"]
    assert updated_tasks["Alpha"]["status"] == "done"
    assert updated_tasks["Beta"]["status"] == "open"

    conflict_response = client.put(
        f"/api/v1/notes/{note_id}",
        json={"content": "- [ ] Alpha\n"},
        headers={"expected-version": str(created["version"])},
    )
    assert conflict_response.status_code == 409, conflict_response.text

    assert db.get_reconciliation_state(note_id) == updated_state
    tasks_after_conflict = _tasks_by_text(db, note_id)
    assert tasks_after_conflict["Alpha"]["status"] == "done"
    assert tasks_after_conflict["Beta"]["status"] == "open"
