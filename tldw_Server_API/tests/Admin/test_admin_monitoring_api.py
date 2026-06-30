from __future__ import annotations

import asyncio
import os
import uuid

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app


def _setup_env(tmp_path) -> None:
    os.environ["AUTH_MODE"] = "single_user"
    os.environ["SINGLE_USER_API_KEY"] = "unit-test-api-key"
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path / 'users_test_admin_monitoring.db'}"


async def _seed_assignable_user() -> int:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

    pool = await get_db_pool()
    username = "monitoring_assignee"
    email = "monitoring_assignee@example.com"
    await pool.execute(
        "INSERT OR IGNORE INTO users (uuid, username, email, password_hash, is_active) VALUES (?,?,?,?,1)",
        str(uuid.uuid4()),
        username,
        email,
        "x",
    )
    user_id = await pool.fetchval("SELECT id FROM users WHERE username = ?", username)
    return int(user_id)


def _seed_runtime_alert(alerts_db_path: str) -> int:
    """Seed one runtime alert so admin overlay mutation tests target a real row."""
    from tldw_Server_API.app.core.DB_Management.TopicMonitoring_DB import (
        TopicAlert,
        TopicMonitoringDB,
    )

    monitoring_db = TopicMonitoringDB(alerts_db_path)
    return monitoring_db.insert_alert(
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


@pytest.mark.asyncio
async def test_admin_monitoring_rules_and_actions(tmp_path) -> None:
    _setup_env(tmp_path)

    from tldw_Server_API.app.api.v1.endpoints import monitoring as monitoring_endpoints
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    os.environ["MONITORING_ALERTS_DB"] = str(tmp_path / "monitoring_alerts.db")
    alert_id = await asyncio.to_thread(_seed_runtime_alert, os.environ["MONITORING_ALERTS_DB"])
    alert_identity = f"alert:{alert_id}"
    noncanonical_alert_identity = f"alert:{str(alert_id).zfill(len(str(alert_id)) + 2)}"

    await reset_db_pool()
    reset_settings()
    await reset_session_manager()
    monitoring_endpoints._TOPIC_MONITORING_DB = None

    headers = {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}

    with TestClient(app, headers=headers) as client:
        assignee_id = await _seed_assignable_user()

        create_rule_resp = client.post(
            "/api/v1/admin/monitoring/alert-rules",
            json={
                "metric": "cpu_percent",
                "operator": "gte",
                "threshold": 90.0,
                "duration_minutes": 10,
                "severity": "warning",
                "enabled": True,
            },
        )
        assert create_rule_resp.status_code == 200, create_rule_resp.text
        rule_id = create_rule_resp.json()["item"]["id"]

        list_rules_resp = client.get("/api/v1/admin/monitoring/alert-rules")
        assert list_rules_resp.status_code == 200, list_rules_resp.text
        assert any(item["id"] == rule_id for item in list_rules_resp.json()["items"])

        assign_resp = client.post(
            f"/api/v1/admin/monitoring/alerts/{noncanonical_alert_identity}/assign",
            json={"assigned_to_user_id": assignee_id},
        )
        assert assign_resp.status_code == 200, assign_resp.text
        assert assign_resp.json()["item"]["alert_identity"] == alert_identity
        assert assign_resp.json()["item"]["assigned_to_user_id"] == assignee_id

        unassign_resp = client.post(
            f"/api/v1/admin/monitoring/alerts/{noncanonical_alert_identity}/assign",
            json={"assigned_to_user_id": None},
        )
        assert unassign_resp.status_code == 200, unassign_resp.text
        assert unassign_resp.json()["item"]["alert_identity"] == alert_identity
        assert unassign_resp.json()["item"]["assigned_to_user_id"] is None

        snooze_resp = client.post(
            f"/api/v1/admin/monitoring/alerts/{noncanonical_alert_identity}/snooze",
            json={"snoozed_until": "2026-03-10T11:00:00Z"},
        )
        assert snooze_resp.status_code == 200, snooze_resp.text
        assert snooze_resp.json()["item"]["alert_identity"] == alert_identity
        assert snooze_resp.json()["item"]["snoozed_until"] == "2026-03-10T11:00:00Z"

        escalate_resp = client.post(
            f"/api/v1/admin/monitoring/alerts/{noncanonical_alert_identity}/escalate",
            json={"severity": "critical"},
        )
        assert escalate_resp.status_code == 200, escalate_resp.text
        assert escalate_resp.json()["item"]["alert_identity"] == alert_identity
        assert escalate_resp.json()["item"]["escalated_severity"] == "critical"

        public_alerts_resp = client.get("/api/v1/monitoring/alerts")
        assert public_alerts_resp.status_code == 200, public_alerts_resp.text
        public_items = public_alerts_resp.json()["items"]
        public_item = next(item for item in public_items if item["alert_identity"] == alert_identity)
        assert public_item["assigned_to_user_id"] is None
        assert public_item["snoozed_until"] == "2026-03-10T11:00:00+00:00"
        assert public_item["escalated_severity"] == "critical"

        history_resp = client.get(
            "/api/v1/admin/monitoring/alerts/history",
            params={"alert_identity": alert_identity},
        )
        assert history_resp.status_code == 200, history_resp.text
        assert [item["action"] for item in history_resp.json()["items"][:4]] == [
            "escalated",
            "snoozed",
            "unassigned",
            "assigned",
        ]

        snooze_payload = {"snoozed_until": "2026-03-10T11:00:00Z"}
        overlay_only_resp = client.post(
            "/api/v1/admin/monitoring/alerts/fingerprint:abc/snooze",
            json=snooze_payload,
        )
        assert overlay_only_resp.status_code == 422, overlay_only_resp.text
        assert overlay_only_resp.json()["detail"] == "unsupported_alert_identity"

        malformed_resp = client.post(
            "/api/v1/admin/monitoring/alerts/alert:not-an-int/snooze",
            json=snooze_payload,
        )
        assert malformed_resp.status_code == 422, malformed_resp.text
        assert malformed_resp.json()["detail"] == "malformed_alert_identity"

        missing_runtime_resp = client.post(
            "/api/v1/admin/monitoring/alerts/alert:404/snooze",
            json=snooze_payload,
        )
        assert missing_runtime_resp.status_code == 404, missing_runtime_resp.text
        assert missing_runtime_resp.json()["detail"] == "unknown_alert"

        delete_resp = client.delete(f"/api/v1/admin/monitoring/alert-rules/{rule_id}")
        assert delete_resp.status_code == 200, delete_resp.text
        assert delete_resp.json()["status"] == "deleted"
