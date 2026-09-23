from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_authnz_admin_monitoring_repo_postgres_round_trip(test_db_pool: Any) -> None:
    from tldw_Server_API.app.core.AuthNZ.repos.admin_monitoring_repo import (
        AuthnzAdminMonitoringRepo,
    )

    pool = test_db_pool
    repo = AuthnzAdminMonitoringRepo(pool)
    await repo.ensure_schema()

    users = UsersDB(pool)
    await users.initialize(ensure_schema=False)
    actor = await users.create_user(  # nosec B106 # Inert fixture hash, never used for login
        username="pg-monitor-actor",
        email="pg-monitor-actor@example.com",
        password_hash="hashed",
        role="admin",
        is_verified=True,
    )
    assignee = await users.create_user(  # nosec B106 # Inert fixture hash, never used for login
        username="pg-monitor-assignee",
        email="pg-monitor-assignee@example.com",
        password_hash="hashed",
        role="admin",
        is_verified=True,
    )
    actor_id = actor["id"]
    assignee_id = assignee["id"]

    created_rule = await repo.create_rule(
        metric="queue_depth",
        operator="gte",
        threshold=20.0,
        duration_minutes=15,
        severity="critical",
        enabled=True,
        created_by_user_id=int(actor_id),
    )
    assert created_rule["metric"] == "queue_depth"

    listed_rules = await repo.list_rules()
    assert any(row["id"] == created_rule["id"] for row in listed_rules)

    upserted_state = await repo.upsert_alert_state(
        alert_identity="alert:42",
        assigned_to_user_id=int(assignee_id),
        snoozed_until="2026-03-10T11:00:00Z",
        dismissed_at="2026-03-10T10:15:00Z",
        updated_by_user_id=int(actor_id),
    )
    assert upserted_state["assigned_to_user_id"] == int(assignee_id)

    cleared_state = await repo.upsert_alert_state(
        alert_identity="alert:42",
        assigned_to_user_id=None,
        updated_by_user_id=int(actor_id),
    )
    assert cleared_state["assigned_to_user_id"] is None

    states = await repo.list_alert_states(["alert:42"])
    assert len(states) == 1
    assert states[0]["assigned_to_user_id"] is None
    assert states[0]["dismissed_at"] == "2026-03-10T10:15:00Z"

    first_event = await repo.append_alert_event(
        alert_identity="alert:42",
        action="assigned",
        actor_user_id=int(actor_id),
        details_json='{"assigned_to_user_id": 2}',
        created_at="2026-03-10T10:05:00Z",
    )
    second_event = await repo.append_alert_event(
        alert_identity="alert:42",
        action="dismissed",
        actor_user_id=int(actor_id),
        details_json='{"dismissed_at": "2026-03-10T10:15:00Z"}',
        created_at="2026-03-10T10:15:00Z",
    )
    assert first_event["action"] == "assigned"
    assert second_event["action"] == "dismissed"

    events = await repo.list_alert_events(alert_identity="alert:42", limit=10)
    assert len(events) == 2
    assert events[0]["action"] == "dismissed"
    assert events[1]["action"] == "assigned"
