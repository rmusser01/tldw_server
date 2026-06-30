from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from loguru import logger

from tldw_Server_API.app.main import app


def _setup_env(monkeypatch, *, user_db_base: str) -> None:
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "unit-test-api-key")
    monkeypatch.setenv("USER_DB_BASE_DIR", user_db_base)
    auth_db_path = Path(user_db_base).parent / "users_test_system_ops.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{auth_db_path}")
    monkeypatch.setenv("TEST_MODE", "true")


@pytest.mark.asyncio
async def test_admin_system_ops_endpoints(monkeypatch, tmp_path):
    _setup_env(monkeypatch, user_db_base=str(tmp_path / "user_dbs"))

    from tldw_Server_API.app.core.config import settings
    monkeypatch.setitem(settings, "AUDIT_BUFFER_SIZE", 1)
    monkeypatch.setitem(settings, "AUDIT_FLUSH_INTERVAL", 0.1)

    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.Logging.system_log_buffer import ensure_system_log_buffer
    from tldw_Server_API.app.services import admin_system_ops_service

    await reset_db_pool()
    reset_settings()
    await reset_session_manager()
    monkeypatch.setattr(admin_system_ops_service, "_STORE_PATH", tmp_path / "system_ops.json")

    headers = {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}

    with TestClient(app, headers=headers) as client:
        ensure_system_log_buffer()
        logger.bind(org_id=1, user_id=1).info("System ops log test entry")
        logs_resp = client.get("/api/v1/admin/system/logs", params={"level": "INFO"})
        if logs_resp.status_code == 200:
            payload = logs_resp.json()
            assert "items" in payload
            assert payload["pagination"]["limit"] == 100
            assert payload["pagination"]["offset"] == 0
            assert payload["pagination"]["total"] >= 1
            assert payload["has_more"] == payload["pagination"]["has_more"]
            assert payload["next_offset"] == payload["pagination"]["next_offset"]
            assert any("System ops log test entry" in (item.get("message") or "") for item in payload["items"])
        else:
            # Some deployments do not mount system logs endpoints.
            assert logs_resp.status_code == 404, logs_resp.text

        maint_resp = client.put(
            "/api/v1/admin/maintenance",
            json={
                "enabled": True,
                "message": "Planned maintenance",
                "allowlist_user_ids": [1],
                "allowlist_emails": [],
            },
        )
        if maint_resp.status_code == 404:
            pytest.skip("Admin system-ops routes are not mounted in this test profile")
        assert maint_resp.status_code == 200, maint_resp.text
        assert maint_resp.json()["enabled"] is True

        maint_get = client.get("/api/v1/admin/maintenance")
        assert maint_get.status_code == 200, maint_get.text
        assert maint_get.json()["enabled"] is True

        audit_log_resp = client.get("/api/v1/admin/audit-log")
        if audit_log_resp.status_code == 200:
            audit_payload = audit_log_resp.json()
            assert audit_payload["pagination"]["limit"] == 100
            assert audit_payload["pagination"]["offset"] == 0
            assert audit_payload["pagination"]["total"] == audit_payload["total"]
            assert audit_payload["has_more"] == audit_payload["pagination"]["has_more"]
            assert audit_payload["next_offset"] == audit_payload["pagination"]["next_offset"]
        else:
            assert audit_log_resp.status_code == 404, audit_log_resp.text

        flag_resp = client.put(
            "/api/v1/admin/feature-flags/ops-test",
            json={
                "scope": "global",
                "enabled": True,
                "description": "test flag",
                "target_user_ids": [3, 2, 3],
                "rollout_percent": 25,
                "variant_value": "blue_path",
            },
        )
        assert flag_resp.status_code == 200, flag_resp.text
        flag_payload = flag_resp.json()
        assert flag_payload["enabled"] is True
        assert flag_payload["target_user_ids"] == [2, 3]
        assert flag_payload["rollout_percent"] == 25
        assert flag_payload["variant_value"] == "blue_path"

        flag_update_resp = client.put(
            "/api/v1/admin/feature-flags/ops-test",
            json={
                "scope": "global",
                "enabled": False,
                "description": "test flag update",
                "target_user_ids": [9],
                "rollout_percent": 75,
                "variant_value": "green_path",
                "note": "Canary promotion",
            },
        )
        assert flag_update_resp.status_code == 200, flag_update_resp.text
        updated_flag_payload = flag_update_resp.json()
        assert updated_flag_payload["enabled"] is False
        assert updated_flag_payload["target_user_ids"] == [9]
        assert updated_flag_payload["rollout_percent"] == 75
        assert updated_flag_payload["variant_value"] == "green_path"
        assert len(updated_flag_payload["history"]) >= 2
        latest_history = updated_flag_payload["history"][-1]
        assert latest_history["before"]["rollout_percent"] == 25
        assert latest_history["after"]["rollout_percent"] == 75
        assert latest_history["before"]["target_user_ids"] == [2, 3]
        assert latest_history["after"]["target_user_ids"] == [9]
        assert latest_history["before"]["enabled"] is True
        assert latest_history["after"]["enabled"] is False

        flags_list = client.get("/api/v1/admin/feature-flags")
        assert flags_list.status_code == 200, flags_list.text
        listed_item = next((item for item in flags_list.json()["items"] if item["key"] == "ops-test"), None)
        assert listed_item is not None
        assert listed_item["rollout_percent"] == 75
        assert listed_item["target_user_ids"] == [9]
        assert listed_item["variant_value"] == "green_path"

        missing_org = client.get("/api/v1/admin/feature-flags", params={"scope": "org"})
        assert missing_org.status_code == 400, missing_org.text
        assert missing_org.json().get("detail") == "missing_org_id"

        missing_user = client.get("/api/v1/admin/feature-flags", params={"scope": "user"})
        assert missing_user.status_code == 400, missing_user.text
        assert missing_user.json().get("detail") == "missing_user_id"

        missing_org_delete = client.delete("/api/v1/admin/feature-flags/ops-test", params={"scope": "org"})
        assert missing_org_delete.status_code == 400, missing_org_delete.text
        assert missing_org_delete.json().get("detail") == "missing_org_id"

        del_flag = client.delete("/api/v1/admin/feature-flags/ops-test", params={"scope": "global"})
        assert del_flag.status_code == 200, del_flag.text

        incident_resp = client.post(
            "/api/v1/admin/incidents",
            json={
                "title": "Queue backlog",
                "status": "investigating",
                "severity": "high",
                "summary": "Background jobs delayed",
                "tags": ["queue", "jobs"],
            },
        )
        assert incident_resp.status_code == 200, incident_resp.text
        incident_id = incident_resp.json()["id"]

        update_resp = client.patch(
            f"/api/v1/admin/incidents/{incident_id}",
            json={"status": "resolved", "update_message": "Recovered"},
        )
        assert update_resp.status_code == 200, update_resp.text
        assert update_resp.json()["status"] == "resolved"

        event_resp = client.post(
            f"/api/v1/admin/incidents/{incident_id}/events",
            json={"message": "Post-mortem scheduled"},
        )
        assert event_resp.status_code == 200, event_resp.text

        list_resp = client.get("/api/v1/admin/incidents")
        assert list_resp.status_code == 200, list_resp.text
        payload = list_resp.json()
        assert payload["total"] >= 1
        assert payload["pagination"]["total"] >= 1
        assert payload["pagination"]["limit"] == 50
        assert payload["pagination"]["offset"] == 0
        assert payload["has_more"] == payload["pagination"]["has_more"]
        assert payload["next_offset"] == payload["pagination"]["next_offset"]

        delete_resp = client.delete(f"/api/v1/admin/incidents/{incident_id}")
        assert delete_resp.status_code == 200, delete_resp.text

        webhook_create_resp = client.post(
            "/api/v1/admin/webhooks",
            json={
                "url": "https://example.com/webhook",
                "events": ["incident.created"],
                "enabled": True,
            },
        )
        assert webhook_create_resp.status_code == 200, webhook_create_resp.text

        webhooks_resp = client.get("/api/v1/admin/webhooks")
        assert webhooks_resp.status_code == 200, webhooks_resp.text
        webhook_payload = webhooks_resp.json()
        assert webhook_payload["total"] >= 1
        assert webhook_payload["pagination"]["total"] >= 1
        assert webhook_payload["pagination"]["offset"] == 0
        assert webhook_payload["pagination"]["limit"] >= 1
        assert webhook_payload["has_more"] == webhook_payload["pagination"]["has_more"]
        assert webhook_payload["next_offset"] == webhook_payload["pagination"]["next_offset"]
        assert any(item["url"] == "https://example.com/webhook" for item in webhook_payload["items"])

        # Reset maintenance state for subsequent tests
        client.put(
            "/api/v1/admin/maintenance",
            json={"enabled": False, "message": "", "allowlist_user_ids": [], "allowlist_emails": []},
        )

    from tldw_Server_API.app.api.v1.API_Deps import Audit_DB_Deps as audit_deps
    await audit_deps.shutdown_all_audit_services()

    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    import sqlite3

    audit_db_path = DatabasePaths.get_audit_db_path(1)
    with sqlite3.connect(audit_db_path) as conn:
        cur = conn.execute(
            "SELECT COUNT(*) FROM audit_events WHERE event_type = ?",
            ("ops.incident",),
        )
        count = int(cur.fetchone()[0])
    assert count >= 4
