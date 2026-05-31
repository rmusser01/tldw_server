from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


def _setup_env(tmp_path) -> None:
    os.environ["AUTH_MODE"] = "single_user"
    os.environ["SINGLE_USER_API_KEY"] = "unit-test-api-key"
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path / 'users_test_monitoring_overlay.db'}"
    os.environ["MONITORING_ALERTS_DB"] = str(tmp_path / "monitoring_alerts.db")


@pytest.mark.asyncio
async def test_monitoring_alerts_include_backend_overlay_and_authoritative_actions(tmp_path) -> None:
    _setup_env(tmp_path)

    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.api.v1.endpoints import monitoring as monitoring_endpoints
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.repos.admin_monitoring_repo import (
        AuthnzAdminMonitoringRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.DB_Management.TopicMonitoring_DB import TopicAlert, TopicMonitoringDB

    await reset_db_pool()
    reset_settings()
    await reset_session_manager()
    monitoring_endpoints._TOPIC_MONITORING_DB = None

    headers = {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}
    monitoring_db = TopicMonitoringDB(os.environ["MONITORING_ALERTS_DB"])
    alert_id = monitoring_db.insert_alert(
        TopicAlert(
            user_id="1",
            scope_type="user",
            scope_id="1",
            source="watchlist",
            watchlist_id="watch-1",
            rule_id="rule-1",
            rule_category="system",
            rule_severity="warning",
            pattern="CPU high",
            text_snippet="CPU sustained at 92%",
            metadata={"host": "api-1"},
            created_at="2026-03-10T10:00:00Z",
        )
    )

    pool = await get_db_pool()
    repo = AuthnzAdminMonitoringRepo(pool)
    await repo.ensure_schema()
    await pool.execute(
        """
        INSERT OR IGNORE INTO users (id, uuid, username, email, password_hash, is_active)
        VALUES (?, ?, ?, ?, ?, 1)
        """,
        1,
        "single-user-admin-uuid",
        "single-user-admin",
        "single-user-admin@example.com",
        "x",
    )
    await repo.upsert_alert_state(
        alert_identity=f"alert:{alert_id}",
        assigned_to_user_id=1,
        snoozed_until="2026-03-10T11:00:00Z",
        escalated_severity="critical",
        updated_by_user_id=1,
    )

    monitoring_paths = sorted(
        route.path for route in app.routes if getattr(route, "path", "").startswith("/api/v1/monitoring")
    )
    assert "/api/v1/monitoring/alerts" in monitoring_paths, monitoring_paths

    with TestClient(app, headers=headers) as client:
        list_resp = client.get("/api/v1/monitoring/alerts")
        assert list_resp.status_code == 200, list_resp.text
        payload = list_resp.json()
        assert payload["pagination"]["total"] is None
        assert payload["pagination"]["limit"] == 100
        assert payload["pagination"]["offset"] == 0
        assert payload["pagination"]["has_more"] is False
        assert payload["pagination"]["next_offset"] is None
        assert payload["has_more"] is False
        assert payload["next_offset"] is None
        items = payload["items"]
        assert len(items) == 1
        assert items[0]["alert_identity"] == f"alert:{alert_id}"
        assert items[0]["assigned_to_user_id"] == 1
        assert items[0]["snoozed_until"] == "2026-03-10T11:00:00Z"
        assert items[0]["escalated_severity"] == "critical"

        read_resp = client.post(f"/api/v1/monitoring/alerts/{alert_id}/read")
        assert read_resp.status_code == 200, read_resp.text
        read_payload = read_resp.json()
        assert read_payload["status"] == "ok"
        assert read_payload["id"] == alert_id
        assert read_payload["item"]["id"] == alert_id
        assert read_payload["item"]["alert_identity"] == f"alert:{alert_id}"
        assert read_payload["item"]["is_read"] is True
        assert read_payload["item"]["read_at"] is not None
        assert read_payload["item"]["acknowledged_at"] is None
        assert read_payload["item"]["assigned_to_user_id"] == 1
        assert read_payload["item"]["snoozed_until"] == "2026-03-10T11:00:00Z"
        assert read_payload["item"]["escalated_severity"] == "critical"

        acknowledge_resp = client.post(f"/api/v1/monitoring/alerts/{alert_id}/acknowledge")
        assert acknowledge_resp.status_code == 200, acknowledge_resp.text
        acknowledge_payload = acknowledge_resp.json()
        assert acknowledge_payload["status"] == "ok"
        assert acknowledge_payload["id"] == alert_id
        assert acknowledge_payload["item"]["id"] == alert_id
        assert acknowledge_payload["item"]["alert_identity"] == f"alert:{alert_id}"
        assert acknowledge_payload["item"]["is_read"] is True
        assert acknowledge_payload["item"]["acknowledged_at"] is not None
        assert acknowledge_payload["item"]["assigned_to_user_id"] == 1

        dismiss_resp = client.delete(f"/api/v1/monitoring/alerts/{alert_id}")
        assert dismiss_resp.status_code == 200, dismiss_resp.text
        dismiss_payload = dismiss_resp.json()
        assert dismiss_payload["status"] == "ok"
        assert dismiss_payload["id"] == alert_id
        assert dismiss_payload["item"]["id"] == alert_id
        assert dismiss_payload["item"]["alert_identity"] == f"alert:{alert_id}"
        assert dismiss_payload["item"]["is_read"] is True
        assert dismiss_payload["item"]["acknowledged_at"] is not None
        assert dismiss_payload["item"]["dismissed_at"] is not None

        refreshed_resp = client.get("/api/v1/monitoring/alerts")
        assert refreshed_resp.status_code == 200, refreshed_resp.text
        refreshed_item = refreshed_resp.json()["items"][0]
        assert refreshed_item["is_read"] is True
        assert refreshed_item["read_at"] is not None
        assert refreshed_item["acknowledged_at"] is not None
        assert refreshed_item["dismissed_at"] is not None

        history_resp = client.get(
            "/api/v1/admin/monitoring/alerts/history",
            params={"alert_identity": f"alert:{alert_id}"},
        )
        assert history_resp.status_code == 200, history_resp.text
        actions = [item["action"] for item in history_resp.json()["items"]]
        assert "read" in actions
        assert "acknowledged" in actions
        assert "dismissed" in actions
