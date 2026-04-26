from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.endpoints.reminders import router as reminders_router
from tldw_Server_API.app.core.AuthNZ.permissions import TASKS_CONTROL, TASKS_READ
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.config import settings

pytestmark = pytest.mark.unit


@pytest.fixture()
def reminders_app(monkeypatch, tmp_path):
    base_dir = tmp_path / "test_reminders_api"
    base_dir.mkdir(parents=True, exist_ok=True)
    prev_base_dir = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(base_dir)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))

    app = FastAPI()
    app.include_router(reminders_router, prefix="/api/v1")

    async def override_user():
        return User(id=880, username="reminder-user", email=None, is_active=True)

    async def override_principal():
        return AuthPrincipal(
            kind="user",
            user_id=880,
            roles=[],
            permissions=[TASKS_READ, TASKS_CONTROL],
            is_admin=False,
        )

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_auth_principal] = override_principal
    try:
        yield app
    finally:
        app.dependency_overrides.clear()
        if prev_base_dir is not None:
            settings.USER_DB_BASE_DIR = prev_base_dir
        else:
            try:
                del settings.USER_DB_BASE_DIR
            except AttributeError:
                pass


def test_create_list_get_patch_delete_task(reminders_app):
    with TestClient(reminders_app) as client:
        create_payload = {
            "title": "Review investigation notes",
            "body": "Continue analysis",
            "schedule_kind": "one_time",
            "run_at": "2026-03-01T10:00:00+00:00",
            "enabled": True,
        }
        r = client.post("/api/v1/tasks", json=create_payload)
        assert r.status_code == 201, r.text
        task = r.json()
        task_id = task["id"]
        assert task["title"] == create_payload["title"]

        r = client.get("/api/v1/tasks")
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["total"] >= 1
        assert any(item["id"] == task_id for item in data["items"])

        r = client.get(f"/api/v1/tasks/{task_id}")
        assert r.status_code == 200, r.text
        assert r.json()["id"] == task_id

        r = client.patch(f"/api/v1/tasks/{task_id}", json={"enabled": False, "title": "Updated title"})
        assert r.status_code == 200, r.text
        patched = r.json()
        assert patched["enabled"] is False
        assert patched["title"] == "Updated title"

        r = client.delete(f"/api/v1/tasks/{task_id}")
        assert r.status_code == 200, r.text
        assert r.json()["deleted"] is True


def test_task_mutations_notify_scheduler(reminders_app, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import reminders as reminders_endpoint

    reconcile_calls: list[tuple[str, int]] = []
    unschedule_calls: list[str] = []

    class _FakeScheduler:
        async def reconcile_task(self, *, task_id: str, user_id: int) -> None:
            reconcile_calls.append((task_id, user_id))

        async def unschedule_task(self, *, task_id: str) -> None:
            unschedule_calls.append(task_id)

    scheduler = _FakeScheduler()
    monkeypatch.setattr(reminders_endpoint, "get_reminders_scheduler", lambda: scheduler, raising=False)

    with TestClient(reminders_app) as client:
        create_payload = {
            "title": "Scheduler sync",
            "body": "Ensure immediate reconcile",
            "schedule_kind": "one_time",
            "run_at": "2026-03-01T10:00:00+00:00",
            "enabled": True,
        }
        create_response = client.post("/api/v1/tasks", json=create_payload)
        assert create_response.status_code == 201, create_response.text
        task_id = create_response.json()["id"]

        patch_response = client.patch(f"/api/v1/tasks/{task_id}", json={"enabled": False})
        assert patch_response.status_code == 200, patch_response.text

        delete_response = client.delete(f"/api/v1/tasks/{task_id}")
        assert delete_response.status_code == 200, delete_response.text

    assert reconcile_calls == [(task_id, 880), (task_id, 880)]
    assert unschedule_calls == [task_id]


def test_patch_task_rejects_scheduler_managed_fields(reminders_app):
    with TestClient(reminders_app) as client:
        create_payload = {
            "title": "Invalid patch target",
            "schedule_kind": "one_time",
            "run_at": "2026-03-01T10:00:00+00:00",
            "enabled": True,
        }
        create_response = client.post("/api/v1/tasks", json=create_payload)
        assert create_response.status_code == 201, create_response.text
        task_id = create_response.json()["id"]

        patch_response = client.patch(
            f"/api/v1/tasks/{task_id}",
            json={"last_status": "queued", "next_run_at": "2026-03-01T11:00:00+00:00"},
        )
        assert patch_response.status_code == 422, patch_response.text


def test_task_mutations_log_scheduler_sync_failures(reminders_app, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import reminders as reminders_endpoint

    warning_calls: list[tuple[str, tuple[object, ...]]] = []

    class _FakeLogger:
        def warning(self, message: str, *args: object, **_kwargs: object) -> None:
            warning_calls.append((message, args))

    class _FailingScheduler:
        async def reconcile_task(self, *, task_id: str, user_id: int) -> None:
            raise RuntimeError(f"reconcile failed for {task_id}:{user_id} at /private/reminders.db")

        async def unschedule_task(self, *, task_id: str) -> None:
            raise RuntimeError(f"unschedule failed for {task_id} at /private/reminders.db")

    monkeypatch.setattr(reminders_endpoint, "logger", _FakeLogger(), raising=False)
    monkeypatch.setattr(reminders_endpoint, "get_reminders_scheduler", lambda: _FailingScheduler(), raising=False)

    with TestClient(reminders_app) as client:
        create_payload = {
            "title": "Scheduler failures are non-fatal",
            "schedule_kind": "one_time",
            "run_at": "2026-03-01T10:00:00+00:00",
            "enabled": True,
        }
        create_response = client.post("/api/v1/tasks", json=create_payload)
        assert create_response.status_code == 201, create_response.text
        task_id = create_response.json()["id"]

        patch_response = client.patch(f"/api/v1/tasks/{task_id}", json={"enabled": False})
        assert patch_response.status_code == 200, patch_response.text

        delete_response = client.delete(f"/api/v1/tasks/{task_id}")
        assert delete_response.status_code == 200, delete_response.text

    assert warning_calls == [
        ("reminders endpoint reconcile_task failed", ()),
        ("reminders endpoint reconcile_task failed", ()),
        ("reminders endpoint unschedule_task failed", ()),
    ]
    rendered_logs = repr(warning_calls)
    assert task_id not in rendered_logs
    assert "880" not in rendered_logs
    assert "/private/" not in rendered_logs
    assert "failed for" not in rendered_logs
