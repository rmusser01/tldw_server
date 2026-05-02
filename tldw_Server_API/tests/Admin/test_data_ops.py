from __future__ import annotations

import os
import uuid
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app


class _LoggerStub:
    def __init__(self) -> None:
        self.error_records: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.error_records.append((message, args, kwargs))


def _setup_env(tmp_path):
    os.environ["AUTH_MODE"] = "single_user"
    os.environ["SINGLE_USER_API_KEY"] = "unit-test-api-key"
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path / 'users_test_data_ops.db'}"
    os.environ["TLDW_DB_ALLOWED_BASE_DIRS"] = str(tmp_path)
    os.environ["TLDW_DB_BACKUP_PATH"] = str(tmp_path / "backups")
    os.environ["USER_DB_BASE_DIR"] = str(tmp_path / "user_dbs")


async def _seed_authnz_data() -> int:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, is_postgres_backend

    pool = await get_db_pool()
    username = "dataops_user"
    email = "dataops_user@example.com"
    if await is_postgres_backend():
        await pool.execute(
            """
            INSERT INTO users (uuid, username, email, password_hash, is_active)
            VALUES (?,?,?,?,1)
            ON CONFLICT (username) DO NOTHING
            """,
            str(uuid.uuid4()),
            username,
            email,
            "x",
        )
    else:
        await pool.execute(
            "INSERT OR IGNORE INTO users (uuid, username, email, password_hash, is_active) VALUES (?,?,?,?,1)",
            str(uuid.uuid4()),
            username,
            email,
            "x",
        )
    user_id = await pool.fetchval("SELECT id FROM users WHERE username = ?", username)
    await pool.execute(
        "INSERT INTO audit_logs (user_id, action, resource_type, resource_id, ip_address, details) VALUES (?,?,?,?,?,?)",
        int(user_id),
        "dataops.test",
        "backup",
        1,
        "127.0.0.1",
        '{"ok": true}',
    )
    return int(user_id)


@pytest.mark.asyncio
async def test_list_backups_sanitizes_generic_failure_log(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints.admin import admin_data_ops

    logger_stub = _LoggerStub()
    monkeypatch.setattr(admin_data_ops, "logger", logger_stub)

    def _raise_list_backup_items(**_kwargs):
        raise RuntimeError("backup list backend exploded at /private/backups.db")

    monkeypatch.setattr(admin_data_ops, "svc_list_backup_items", _raise_list_backup_items)

    with pytest.raises(HTTPException) as exc_info:
        await admin_data_ops.list_backups(
            dataset=None,
            user_id=None,
            limit=100,
            offset=0,
            principal=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to list backups"
    assert logger_stub.error_records == [("Failed to list backups", (), {})]


@pytest.mark.asyncio
async def test_create_backup_sanitizes_generic_failure_log(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints.admin import admin_data_ops

    logger_stub = _LoggerStub()
    monkeypatch.setattr(admin_data_ops, "logger", logger_stub)

    def _raise_create_backup_snapshot(**_kwargs):
        raise RuntimeError("backup create backend exploded at /private/backups.db")

    monkeypatch.setattr(admin_data_ops, "svc_create_backup_snapshot", _raise_create_backup_snapshot)

    with pytest.raises(HTTPException) as exc_info:
        await admin_data_ops.create_backup(
            admin_data_ops.BackupCreateRequest(dataset="authnz", backup_type="full"),
            request=object(),
            principal=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to create backup"
    assert logger_stub.error_records == [("Failed to create backup", (), {})]


@pytest.mark.asyncio
async def test_restore_backup_sanitizes_generic_failure_log(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints.admin import admin_data_ops

    logger_stub = _LoggerStub()
    monkeypatch.setattr(admin_data_ops, "logger", logger_stub)

    def _raise_restore_backup_snapshot(**_kwargs):
        raise RuntimeError("backup restore backend exploded at /private/backups.db")

    monkeypatch.setattr(admin_data_ops, "svc_restore_backup_snapshot", _raise_restore_backup_snapshot)

    with pytest.raises(HTTPException) as exc_info:
        await admin_data_ops.restore_backup(
            "backup.db",
            admin_data_ops.BackupRestoreRequest(dataset="authnz", confirm=True),
            request=object(),
            principal=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to restore backup"
    assert logger_stub.error_records == [("Failed to restore backup", (), {})]


@pytest.mark.asyncio
async def test_admin_data_ops_backups_and_exports(tmp_path):
    _setup_env(tmp_path)

    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    await reset_db_pool()
    reset_settings()
    await reset_session_manager()

    headers = {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}

    with TestClient(app, headers=headers) as client:
        await _seed_authnz_data()

        create_resp = client.post(
            "/api/v1/admin/backups",
            json={"dataset": "authnz", "backup_type": "full"},
        )
        assert create_resp.status_code == 200, create_resp.text
        backup_id = create_resp.json()["item"]["id"]

        list_resp = client.get("/api/v1/admin/backups", params={"dataset": "authnz"})
        assert list_resp.status_code == 200, list_resp.text
        payload = list_resp.json()
        assert payload["pagination"]["total"] >= 1
        assert payload["pagination"]["limit"] == 100
        assert payload["pagination"]["offset"] == 0
        assert payload["has_more"] == payload["pagination"]["has_more"]
        assert payload["next_offset"] == payload["pagination"]["next_offset"]
        listed = payload["items"]
        assert any(item["id"] == backup_id for item in listed)

        restore_resp = client.post(
            f"/api/v1/admin/backups/{backup_id}/restore",
            json={"dataset": "authnz", "confirm": True},
        )
        assert restore_resp.status_code == 200, restore_resp.text

        audit_export = client.get("/api/v1/admin/audit-log/export", params={"format": "csv"})
        assert audit_export.status_code == 200, audit_export.text
        assert "id,user_id,username,action" in audit_export.text.splitlines()[0]

        user_export = client.get("/api/v1/admin/users/export", params={"format": "csv"})
        assert user_export.status_code == 200, user_export.text
        assert "id,uuid,username,email,role" in user_export.text.splitlines()[0]


@pytest.mark.asyncio
async def test_admin_retention_policy_update(tmp_path):
    _setup_env(tmp_path)

    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    await reset_db_pool()
    reset_settings()
    await reset_session_manager()

    headers = {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}

    with TestClient(app, headers=headers) as client:
        list_resp = client.get("/api/v1/admin/retention-policies")
        assert list_resp.status_code == 200, list_resp.text
        policies = list_resp.json()["policies"]
        assert policies
        target = next(
            (policy["key"] for policy in policies if policy.get("key") == "audit_logs"),
            policies[0]["key"],
        )
        current_days = next((policy["days"] for policy in policies if policy.get("key") == target), None)
        assert isinstance(current_days, int)

        preview_resp = client.post(
            f"/api/v1/admin/retention-policies/{target}/preview",
            json={"current_days": current_days, "days": 180},
        )
        assert preview_resp.status_code == 200, preview_resp.text
        preview_signature = preview_resp.json()["preview_signature"]

        update_resp = client.put(
            f"/api/v1/admin/retention-policies/{target}",
            json={"days": 180, "preview_signature": preview_signature},
        )
        assert update_resp.status_code == 200, update_resp.text
        payload = update_resp.json()
        assert payload["key"] == target
        assert payload["days"] == 180

        reset_settings()
        list_resp = client.get("/api/v1/admin/retention-policies")
        assert list_resp.status_code == 200, list_resp.text
        policies = list_resp.json()["policies"]
        refreshed = next((policy for policy in policies if policy.get("key") == target), None)
        assert refreshed is not None
        assert refreshed["days"] == 180


@pytest.mark.asyncio
async def test_admin_data_ops_requires_user_id_for_per_user_datasets(tmp_path):
    _setup_env(tmp_path)

    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    await reset_db_pool()
    reset_settings()
    await reset_session_manager()

    headers = {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}

    with TestClient(app, headers=headers) as client:
        list_resp = client.get("/api/v1/admin/backups", params={"dataset": "media"})
        assert list_resp.status_code == 400, list_resp.text
        assert list_resp.json()["detail"] == "user_id_required"

        create_resp = client.post(
            "/api/v1/admin/backups",
            json={"dataset": "media", "backup_type": "full"},
        )
        assert create_resp.status_code == 400, create_resp.text
        assert create_resp.json()["detail"] == "user_id_required"

        restore_resp = client.post(
            "/api/v1/admin/backups/any.db/restore",
            json={"dataset": "media", "confirm": True},
        )
        assert restore_resp.status_code == 400, restore_resp.text
        assert restore_resp.json()["detail"] == "user_id_required"


@pytest.mark.asyncio
async def test_admin_data_ops_per_user_backup_success(tmp_path):
    _setup_env(tmp_path)

    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

    await reset_db_pool()
    reset_settings()
    await reset_session_manager()

    user_id = DatabasePaths.get_single_user_id()
    media_db_path = DatabasePaths.get_media_db_path(user_id)
    media_db_path.parent.mkdir(parents=True, exist_ok=True)

    import sqlite3

    with sqlite3.connect(media_db_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO test_table (name) VALUES (?)", ("sample",))
        conn.commit()

    headers = {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}

    with TestClient(app, headers=headers) as client:
        create_resp = client.post(
            "/api/v1/admin/backups",
            json={"dataset": "media", "user_id": user_id, "backup_type": "full"},
        )
        assert create_resp.status_code == 200, create_resp.text
        backup_id = create_resp.json()["item"]["id"]

        list_resp = client.get("/api/v1/admin/backups", params={"dataset": "media", "user_id": user_id})
        assert list_resp.status_code == 200, list_resp.text
        payload = list_resp.json()
        assert payload["pagination"]["total"] >= 1
        assert payload["pagination"]["limit"] == 100
        assert payload["pagination"]["offset"] == 0
        listed = payload["items"]
        assert any(item["id"] == backup_id for item in listed)
