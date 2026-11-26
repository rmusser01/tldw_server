"""The new-sign-in alert against isolated PostgreSQL (TASK-15649)."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.sign_in_alert import drain_pending_alerts, write_new_sign_in_alert
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase

pytestmark = [pytest.mark.integration, pytest.mark.asyncio, pytest.mark.sign_in_alerts]

PASSWORD = "Str0ngP@ssw0rd!"


def _pg_backend(db_name: str):
    return DatabaseBackendFactory.create_backend(
        DatabaseConfig(
            backend_type=BackendType.POSTGRESQL,
            pg_host=os.getenv("TEST_DB_HOST", "localhost"),
            pg_port=int(os.getenv("TEST_DB_PORT", "5432")),
            pg_database=db_name,
            pg_user=os.getenv("TEST_DB_USER", "tldw_user"),
            pg_password=os.getenv("TEST_DB_PASSWORD", "TestPassword123!"),
            pg_sslmode=os.getenv("TEST_DB_SSLMODE", "prefer"),
        )
    )


def _write(db, user_id: int, session_id: int, now: datetime):
    return write_new_sign_in_alert(
        db,
        session_id=session_id,
        ip_address="203.0.113.7",
        user_agent=None,
        device_id="dev-aaaaaaaa",
        user_id=user_id,
        now=now,
    )


async def test_alert_write_dedupes_and_isolates_users_postgres(
    isolated_test_environment: tuple[TestClient, str], monkeypatch, tmp_path
) -> None:
    _client, db_name = isolated_test_environment
    monkeypatch.setenv("USER_DB_BASE_DIR", str((tmp_path / "user_dbs").resolve()))
    backend = _pg_backend(db_name)
    one = CollectionsDatabase.from_backend(user_id="1", backend=backend)
    two = CollectionsDatabase.from_backend(user_id="2", backend=backend)
    now = datetime(2026, 11, 25, 14, 3, tzinfo=timezone.utc)

    first = _write(one, 1, 10, now)
    again = _write(one, 1, 11, now)
    other = _write(two, 2, 12, now)

    assert first.id == again.id
    assert [r.source_job_id for r in one.list_user_notifications()] == ["10"]
    assert [r.source_job_id for r in two.list_user_notifications()] == ["12"]
    assert (first.user_id, other.user_id) == ("1", "2")


async def test_second_sign_in_writes_an_alert_over_http_postgres(
    isolated_test_environment: tuple[TestClient, str], monkeypatch, tmp_path
) -> None:
    client, _db = isolated_test_environment
    monkeypatch.setenv("USER_DB_BASE_DIR", str((tmp_path / "user_dbs").resolve()))
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("JWT_SECRET_KEY", "X" * 40)
    reg = client.post(
        "/api/v1/auth/register",
        json={"username": "pgalice", "email": "pgalice@example.com", "password": PASSWORD},
    )
    assert reg.status_code == 200, reg.text

    def login(device: str) -> dict:
        r = client.post(
            "/api/v1/auth/login",
            data={"username": "pgalice", "password": PASSWORD},
            headers={"X-TLDW-Device-Id": device},
        )
        assert r.status_code == 200, r.text
        return r.json()

    a = login("dev-aaaaaaaa")
    login("dev-bbbbbbbb")
    await drain_pending_alerts()
    response = client.get("/api/v1/notifications", headers={"Authorization": f"Bearer {a['access_token']}"})
    assert response.status_code == 200, response.text
    items = [i for i in response.json()["items"] if i["kind"] == "new_sign_in"]
    assert len(items) == 1 and items[0]["link_url"] == "/settings/sessions"
