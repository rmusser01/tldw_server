from __future__ import annotations

from datetime import datetime, timezone

import pytest


def test_admin_monitoring_repo_coerces_iso_timestamp_strings_for_postgres() -> None:
    from tldw_Server_API.app.core.AuthNZ.repos.admin_monitoring_repo import (
        AuthnzAdminMonitoringRepo,
    )

    coerced = AuthnzAdminMonitoringRepo._coerce_timestamp_input("2026-03-10T11:00:00Z")

    assert coerced == datetime(2026, 3, 10, 11, 0, 0, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_admin_monitoring_repo_rule_state_and_event_round_trip_sqlite(tmp_path) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
    from tldw_Server_API.app.core.AuthNZ.profile_version import VersionedUserWriteGateway
    from tldw_Server_API.app.core.AuthNZ.repos.admin_monitoring_repo import (
        AuthnzAdminMonitoringRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import Settings

    db_path = tmp_path / "admin_monitoring_repo.sqlite"
    settings = Settings(
        AUTH_MODE="multi_user",
        DATABASE_URL=f"sqlite:///{db_path}",
        JWT_SECRET_KEY="monitoring-secret-key-32-characters-minimum!",
        ENABLE_REGISTRATION=True,
        REQUIRE_REGISTRATION_CODE=False,
    )

    pool = DatabasePool(settings)
    await pool.initialize()

    try:
        repo = AuthnzAdminMonitoringRepo(pool)
        await repo.ensure_schema()

        async with pool.transaction() as conn:
            gateway = VersionedUserWriteGateway("sqlite")
            for username, email in (
                ("admin-actor", "actor@example.com"),
                ("assignee-user", "assignee@example.com"),
            ):
                await gateway.insert_user(
                    conn,
                    values={
                        "username": username,
                        "email": email,
                        "password_hash": "hashed",  # nosec B105 # Inert fixture hash, never used for login
                        "role": "admin",
                        "is_active": 1,
                    },
                )
            actor_cursor = await conn.execute(
                "SELECT id FROM users WHERE username = ?",
                ("admin-actor",),
            )
            actor_id = await actor_cursor.fetchone()
            assignee_cursor = await conn.execute(
                "SELECT id FROM users WHERE username = ?",
                ("assignee-user",),
            )
            assignee_id = await assignee_cursor.fetchone()

        created_rule = await repo.create_rule(
            metric="cpu_percent",
            operator="gte",
            threshold=90.0,
            duration_minutes=10,
            severity="warning",
            enabled=True,
            created_by_user_id=int(actor_id[0]),
        )
        assert created_rule["metric"] == "cpu_percent"
        assert created_rule["enabled"] is True

        fetched_rule = await repo.get_rule(int(created_rule["id"]))
        assert fetched_rule is not None
        assert fetched_rule["severity"] == "warning"

        listed_rules = await repo.list_rules()
        assert len(listed_rules) == 1
        assert listed_rules[0]["id"] == created_rule["id"]

        upserted_state = await repo.upsert_alert_state(
            alert_identity="alert:7",
            assigned_to_user_id=int(assignee_id[0]),
            snoozed_until="2026-03-10T11:00:00Z",
            escalated_severity="critical",
            acknowledged_at="2026-03-10T10:05:00Z",
            updated_by_user_id=int(actor_id[0]),
        )
        assert upserted_state["assigned_to_user_id"] == int(assignee_id[0])

        cleared_state = await repo.upsert_alert_state(
            alert_identity="alert:7",
            assigned_to_user_id=None,
            updated_by_user_id=int(actor_id[0]),
        )
        assert cleared_state["assigned_to_user_id"] is None

        state_rows = await repo.list_alert_states(["alert:7"])
        assert len(state_rows) == 1
        assert state_rows[0]["alert_identity"] == "alert:7"
        assert state_rows[0]["assigned_to_user_id"] is None

        first_event = await repo.append_alert_event(
            alert_identity="alert:7",
            action="assigned",
            actor_user_id=int(actor_id[0]),
            details_json='{"assigned_to_user_id": 2}',
            created_at="2026-03-10T10:05:00Z",
        )
        second_event = await repo.append_alert_event(
            alert_identity="alert:7",
            action="snoozed",
            actor_user_id=int(actor_id[0]),
            details_json='{"snoozed_until": "2026-03-10T11:00:00Z"}',
            created_at="2026-03-10T10:06:00Z",
        )
        assert first_event["action"] == "assigned"
        assert second_event["action"] == "snoozed"

        event_rows = await repo.list_alert_events(alert_identity="alert:7", limit=10)
        assert len(event_rows) == 2
        assert event_rows[0]["action"] == "snoozed"
        assert event_rows[1]["action"] == "assigned"

        cleared = await repo.clear_state_and_events()
        assert cleared == {"deleted_states": 1, "deleted_events": 2}
        assert await repo.list_alert_states(["alert:7"]) == []
        assert await repo.list_alert_events(alert_identity="alert:7", limit=10) == []

        deleted = await repo.delete_rule(int(created_rule["id"]))
        assert deleted is True
        assert await repo.get_rule(int(created_rule["id"])) is None
    finally:
        await pool.close()
