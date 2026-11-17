from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app


def _setup_env(monkeypatch, *, user_db_base: str) -> None:
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "unit-test-api-key")
    monkeypatch.setenv("USER_DB_BASE_DIR", user_db_base)
    auth_db_path = Path(user_db_base).parent / "users_test_incidents_api.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{auth_db_path}")
    monkeypatch.setenv("TEST_MODE", "true")


@pytest.mark.asyncio
async def test_incident_patch_resolves_assignee_and_preserves_workflow_fields(monkeypatch, tmp_path, fresh_services):
    _setup_env(monkeypatch, user_db_base=str(tmp_path / "user_dbs"))

    from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.services import admin_system_ops_service

    reset_settings()
    monkeypatch.setattr(admin_system_ops_service, "_STORE_PATH", tmp_path / "system_ops.json")

    async def _resolve_assignee(user_id: int) -> dict[str, object]:
        return {"assigned_to_user_id": user_id, "assigned_to_label": "Alice Admin"}

    monkeypatch.setattr(admin_ops, "_resolve_incident_assignee", _resolve_assignee)

    headers = {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}
    with TestClient(app, headers=headers) as client:
        create_resp = client.post(
            "/api/v1/admin/incidents",
            json={
                "title": "Queue backlog",
                "status": "investigating",
                "severity": "high",
                "summary": "Jobs delayed",
                "tags": ["queue"],
            },
        )
        assert create_resp.status_code == 200, create_resp.text
        incident_id = create_resp.json()["id"]

        patch_resp = client.patch(
            f"/api/v1/admin/incidents/{incident_id}",
            json={
                "assigned_to_user_id": 7,
                "root_cause": "Connection pool exhaustion",
                "impact": "Writes failed for 4 minutes",
                "action_items": [
                    {"id": "ai_keep", "text": " Add pool saturation alert ", "done": False},
                    {"id": "ai_blank", "text": "   ", "done": True},
                ],
                "update_message": "Post-mortem updated",
            },
        )
        assert patch_resp.status_code == 200, patch_resp.text
        payload = patch_resp.json()
        assert payload["assigned_to_user_id"] == 7
        assert payload["assigned_to_label"] == "Alice Admin"
        assert payload["root_cause"] == "Connection pool exhaustion"
        assert payload["impact"] == "Writes failed for 4 minutes"
        assert payload["action_items"] == [
            {"id": "ai_keep", "text": "Add pool saturation alert", "done": False}
        ]

        preserve_resp = client.patch(
            f"/api/v1/admin/incidents/{incident_id}",
            json={"status": "mitigating", "update_message": "Status updated"},
        )
        assert preserve_resp.status_code == 200, preserve_resp.text
        preserved = preserve_resp.json()
        assert preserved["status"] == "mitigating"
        assert preserved["assigned_to_user_id"] == 7
        assert preserved["assigned_to_label"] == "Alice Admin"
        assert preserved["root_cause"] == "Connection pool exhaustion"
        assert preserved["impact"] == "Writes failed for 4 minutes"
        assert preserved["action_items"] == [
            {"id": "ai_keep", "text": "Add pool saturation alert", "done": False}
        ]


@pytest.mark.asyncio
async def test_incident_patch_rejects_non_admin_assignee(monkeypatch, tmp_path, fresh_services):
    _setup_env(monkeypatch, user_db_base=str(tmp_path / "user_dbs"))

    from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.services import admin_system_ops_service

    reset_settings()
    monkeypatch.setattr(admin_system_ops_service, "_STORE_PATH", tmp_path / "system_ops.json")

    async def _reject_assignee(user_id: int) -> dict[str, object]:
        raise ValueError("incident_assignee_must_be_admin")

    monkeypatch.setattr(admin_ops, "_resolve_incident_assignee", _reject_assignee)

    headers = {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}
    with TestClient(app, headers=headers) as client:
        create_resp = client.post(
            "/api/v1/admin/incidents",
            json={"title": "Queue backlog", "severity": "high"},
        )
        assert create_resp.status_code == 200, create_resp.text
        incident_id = create_resp.json()["id"]

        patch_resp = client.patch(
            f"/api/v1/admin/incidents/{incident_id}",
            json={"assigned_to_user_id": 9, "update_message": "Assigning reviewer"},
        )
        assert patch_resp.status_code == 400, patch_resp.text
        assert patch_resp.json()["detail"] == "incident_assignee_must_be_admin"


@pytest.mark.asyncio
async def test_incident_patch_clears_assignment_with_null(monkeypatch, tmp_path, fresh_services):
    _setup_env(monkeypatch, user_db_base=str(tmp_path / "user_dbs"))

    from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.services import admin_system_ops_service

    reset_settings()
    monkeypatch.setattr(admin_system_ops_service, "_STORE_PATH", tmp_path / "system_ops.json")

    async def _resolve_assignee(user_id: int) -> dict[str, object]:
        return {"assigned_to_user_id": user_id, "assigned_to_label": "Alice Admin"}

    monkeypatch.setattr(admin_ops, "_resolve_incident_assignee", _resolve_assignee)

    headers = {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}
    with TestClient(app, headers=headers) as client:
        create_resp = client.post(
            "/api/v1/admin/incidents",
            json={"title": "Queue backlog", "severity": "high"},
        )
        assert create_resp.status_code == 200, create_resp.text
        incident_id = create_resp.json()["id"]

        assign_resp = client.patch(
            f"/api/v1/admin/incidents/{incident_id}",
            json={"assigned_to_user_id": 7, "update_message": "Assigned to Alice Admin"},
        )
        assert assign_resp.status_code == 200, assign_resp.text

        clear_resp = client.patch(
            f"/api/v1/admin/incidents/{incident_id}",
            json={"assigned_to_user_id": None, "update_message": "Assignment cleared"},
        )
        assert clear_resp.status_code == 200, clear_resp.text
        payload = clear_resp.json()
        assert payload["assigned_to_user_id"] is None
        assert payload["assigned_to_label"] is None
