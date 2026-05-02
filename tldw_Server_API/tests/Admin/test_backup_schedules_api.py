from __future__ import annotations

import os

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.main import app


def _setup_env(tmp_path) -> None:
    os.environ["AUTH_MODE"] = "single_user"
    os.environ["SINGLE_USER_API_KEY"] = "unit-test-api-key"
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path / 'users_test_backup_schedules.db'}"
    os.environ["TLDW_DB_ALLOWED_BASE_DIRS"] = str(tmp_path)
    os.environ["TLDW_DB_BACKUP_PATH"] = str(tmp_path / "backups")
    os.environ["USER_DB_BASE_DIR"] = str(tmp_path / "user_dbs")


async def _reset_authnz_state() -> None:
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    await reset_db_pool()
    reset_settings()
    await reset_session_manager()


async def _build_backup_repo(tmp_path):
    from pathlib import Path

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.backup_schedules_repo import (
        AuthnzBackupSchedulesRepo,
    )

    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(tmp_path / "users_test_backup_schedules.db")))
    repo = AuthnzBackupSchedulesRepo(pool)
    await repo.ensure_schema()
    return repo


@pytest.mark.asyncio
async def test_backup_schedule_roundtrip_crud(tmp_path) -> None:
    _setup_env(tmp_path)
    await _reset_authnz_state()

    headers = {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}

    with TestClient(app, headers=headers) as client:
        create_resp = client.post(
            "/api/v1/admin/backup-schedules",
            json={
                "dataset": "authnz",
                "frequency": "daily",
                "time_of_day": "02:00",
                "retention_count": 30,
            },
        )
        assert create_resp.status_code == 200, create_resp.text
        created = create_resp.json()["item"]
        schedule_id = created["id"]
        assert created["dataset"] == "authnz"
        assert created["is_paused"] is False

        list_resp = client.get("/api/v1/admin/backup-schedules")
        assert list_resp.status_code == 200, list_resp.text
        payload = list_resp.json()
        assert payload["total"] == 1
        assert payload["pagination"]["total"] == 1
        assert payload["pagination"]["limit"] == 100
        assert payload["pagination"]["offset"] == 0
        assert payload["has_more"] == payload["pagination"]["has_more"]
        assert payload["next_offset"] == payload["pagination"]["next_offset"]
        assert payload["items"][0]["id"] == schedule_id

        update_resp = client.patch(
            f"/api/v1/admin/backup-schedules/{schedule_id}",
            json={
                "frequency": "weekly",
                "time_of_day": "03:00",
                "retention_count": 21,
            },
        )
        assert update_resp.status_code == 200, update_resp.text
        updated = update_resp.json()["item"]
        assert updated["frequency"] == "weekly"
        assert updated["time_of_day"] == "03:00"
        assert updated["retention_count"] == 21

        pause_resp = client.post(f"/api/v1/admin/backup-schedules/{schedule_id}/pause")
        assert pause_resp.status_code == 200, pause_resp.text
        assert pause_resp.json()["item"]["is_paused"] is True

        resume_resp = client.post(f"/api/v1/admin/backup-schedules/{schedule_id}/resume")
        assert resume_resp.status_code == 200, resume_resp.text
        assert resume_resp.json()["item"]["is_paused"] is False

        delete_resp = client.delete(f"/api/v1/admin/backup-schedules/{schedule_id}")
        assert delete_resp.status_code == 200, delete_resp.text
        assert delete_resp.json()["status"] == "deleted"

        final_list = client.get("/api/v1/admin/backup-schedules")
        assert final_list.status_code == 200, final_list.text
        assert final_list.json()["total"] == 0


@pytest.mark.asyncio
async def test_create_backup_schedule_requires_target_user_for_per_user_dataset(tmp_path) -> None:
    _setup_env(tmp_path)
    await _reset_authnz_state()

    headers = {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}

    with TestClient(app, headers=headers) as client:
        response = client.post(
            "/api/v1/admin/backup-schedules",
            json={
                "dataset": "media",
                "frequency": "daily",
                "time_of_day": "02:00",
                "retention_count": 7,
            },
        )
        assert response.status_code == 400, response.text
        assert response.json()["detail"] == "target_user_required"


@pytest.mark.asyncio
async def test_create_backup_schedule_forbids_target_user_for_authnz(tmp_path) -> None:
    _setup_env(tmp_path)
    await _reset_authnz_state()

    headers = {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}

    with TestClient(app, headers=headers) as client:
        response = client.post(
            "/api/v1/admin/backup-schedules",
            json={
                "dataset": "authnz",
                "target_user_id": 9,
                "frequency": "daily",
                "time_of_day": "02:00",
                "retention_count": 14,
            },
        )
        assert response.status_code == 400, response.text
        assert response.json()["detail"] == "target_user_forbidden"


@pytest.mark.asyncio
async def test_create_backup_schedule_rejects_out_of_scope_user(tmp_path, monkeypatch) -> None:
    _setup_env(tmp_path)
    await _reset_authnz_state()

    principal = AuthPrincipal(
        kind="user",
        user_id=41,
        subject="user:41",
        roles=["admin"],
        is_admin=True,
    )

    async def _fake_scope_check(*args, **kwargs):
        raise HTTPException(status_code=403, detail="Not authorized to manage users outside your organization or team")

    app.dependency_overrides[get_auth_principal] = lambda: principal
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.admin.admin_data_ops._enforce_admin_user_scope",
        _fake_scope_check,
    )

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/admin/backup-schedules",
                json={
                    "dataset": "media",
                    "target_user_id": 99,
                    "frequency": "daily",
                    "time_of_day": "02:00",
                    "retention_count": 7,
                },
            )
            assert response.status_code == 403, response.text
            assert response.json()["detail"] == "Not authorized to manage users outside your organization or team"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_list_backup_schedules_hides_platform_rows_and_reports_visible_total(
    tmp_path,
) -> None:
    _setup_env(tmp_path)
    await _reset_authnz_state()

    repo = await _build_backup_repo(tmp_path)
    await repo.create_schedule(
        dataset="authnz",
        target_user_id=None,
        frequency="daily",
        time_of_day="02:00",
        timezone="UTC",
        anchor_day_of_week=None,
        anchor_day_of_month=None,
        retention_count=30,
        created_by_user_id=1,
        updated_by_user_id=1,
        next_run_at="2026-03-12T02:00:00+00:00",
    )
    media_schedule = await repo.create_schedule(
        dataset="media",
        target_user_id=41,
        frequency="daily",
        time_of_day="03:00",
        timezone="UTC",
        anchor_day_of_week=None,
        anchor_day_of_month=None,
        retention_count=7,
        created_by_user_id=1,
        updated_by_user_id=1,
        next_run_at="2026-03-12T03:00:00+00:00",
    )

    principal = AuthPrincipal(
        kind="user",
        user_id=77,
        subject="user:77",
        roles=["admin"],
        is_admin=True,
    )
    app.dependency_overrides[get_auth_principal] = lambda: principal

    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/admin/backup-schedules?limit=100&offset=0")
            assert response.status_code == 200, response.text
            payload = response.json()
            assert payload["total"] == 1
            assert payload["pagination"]["total"] == 1
            assert payload["pagination"]["limit"] == 100
            assert payload["pagination"]["offset"] == 0
            assert [item["id"] for item in payload["items"]] == [media_schedule["id"]]
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_create_authnz_backup_schedule_requires_platform_admin(tmp_path) -> None:
    _setup_env(tmp_path)
    await _reset_authnz_state()

    principal = AuthPrincipal(
        kind="user",
        user_id=51,
        subject="user:51",
        roles=["admin"],
        is_admin=True,
    )

    app.dependency_overrides[get_auth_principal] = lambda: principal

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/admin/backup-schedules",
                json={
                    "dataset": "authnz",
                    "frequency": "daily",
                    "time_of_day": "02:00",
                    "retention_count": 14,
                },
            )
            assert response.status_code == 403, response.text
            assert response.json()["detail"] == "platform_admin_required"
    finally:
        app.dependency_overrides.clear()
