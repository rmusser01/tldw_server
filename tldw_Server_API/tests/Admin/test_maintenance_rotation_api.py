from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.main import app


def _setup_env(monkeypatch, *, user_db_base: str) -> None:
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "unit-test-api-key")
    monkeypatch.setenv("USER_DB_BASE_DIR", user_db_base)
    auth_db_path = Path(user_db_base).parent / "users_test_maintenance_rotation.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{auth_db_path}")
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("ADMIN_MAINTENANCE_ROTATION_JOBS_WORKER_ENABLED", "true")


@dataclass
class _FakeRotationService:
    created_calls: list[dict] = field(default_factory=list)

    async def create_run(self, **kwargs):
        self.created_calls.append(dict(kwargs))
        if kwargs["mode"] == "execute" and not kwargs["confirmed"]:
            raise HTTPException(status_code=400, detail="confirmation_required")
        return {
            "id": "run-1",
            "mode": kwargs["mode"],
            "status": "queued",
            "domain": kwargs["domain"],
            "queue": kwargs["queue"],
            "job_type": kwargs["job_type"],
            "fields_json": "[\"payload\",\"result\"]",
            "limit": kwargs["limit"],
            "affected_count": None,
            "requested_by_user_id": kwargs["requested_by_user_id"],
            "requested_by_label": kwargs["requested_by_label"],
            "confirmation_recorded": kwargs["confirmed"],
            "job_id": None,
            "scope_summary": "domain=jobs, queue=default, job_type=encryption_rotation, fields=payload,result, limit=100",
            "key_source": "env:jobs_crypto_rotate",
            "error_message": None,
            "created_at": "2026-03-12T20:00:00+00:00",
            "started_at": None,
            "completed_at": None,
        }

    async def list_runs(self, *, limit: int, offset: int, allowed_domains: list[str] | None = None):
        return {
            "items": [
                {
                    "id": "run-1",
                    "mode": "dry_run",
                    "status": "complete",
                    "domain": "jobs",
                    "queue": "default",
                    "job_type": "encryption_rotation",
                    "fields_json": "[\"payload\",\"result\"]",
                    "limit": 100,
                    "affected_count": 42,
                    "requested_by_user_id": 1,
                    "requested_by_label": "ops-admin@example.com",
                    "confirmation_recorded": False,
                    "job_id": "job-1",
                    "scope_summary": "domain=jobs, queue=default, job_type=encryption_rotation, fields=payload,result, limit=100",
                    "key_source": "env:jobs_crypto_rotate",
                    "error_message": None,
                    "created_at": "2026-03-12T20:00:00+00:00",
                    "started_at": "2026-03-12T20:00:05+00:00",
                    "completed_at": "2026-03-12T20:00:15+00:00",
                }
            ],
            "total": 1,
            "limit": limit,
            "offset": offset,
        }

    async def get_run(self, run_id: str):
        if run_id != "run-1":
            return None
        return {
            "id": "run-1",
            "mode": "dry_run",
            "status": "complete",
            "domain": "jobs",
            "queue": "default",
            "job_type": "encryption_rotation",
            "fields_json": "[\"payload\",\"result\"]",
            "limit": 100,
            "affected_count": 42,
            "requested_by_user_id": 1,
            "requested_by_label": "ops-admin@example.com",
            "confirmation_recorded": False,
            "job_id": "job-1",
            "scope_summary": "domain=jobs, queue=default, job_type=encryption_rotation, fields=payload,result, limit=100",
            "key_source": "env:jobs_crypto_rotate",
            "error_message": None,
            "created_at": "2026-03-12T20:00:00+00:00",
            "started_at": "2026-03-12T20:00:05+00:00",
            "completed_at": "2026-03-12T20:00:15+00:00",
        }


async def _noop_enqueue_run(item):
    return None


@dataclass
class _DisabledWorkerService(_FakeRotationService):
    async def create_run(self, **kwargs):
        raise AssertionError("create_run should not be called when the worker is disabled")


@pytest.mark.asyncio
async def test_maintenance_rotation_create_list_and_detail_roundtrip(monkeypatch, tmp_path) -> None:
    _setup_env(monkeypatch, user_db_base=str(tmp_path / "user_dbs"))

    from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops

    service = _FakeRotationService()
    app.dependency_overrides[admin_ops.get_admin_maintenance_rotation_service] = lambda: service
    app.dependency_overrides[admin_ops.get_maintenance_rotation_job_enqueuer] = lambda: _noop_enqueue_run

    headers = {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}

    try:
        with TestClient(app, headers=headers) as client:
            create_resp = client.post(
                "/api/v1/admin/maintenance/rotation-runs",
                json={
                    "mode": "dry_run",
                    "domain": "jobs",
                    "queue": "default",
                    "job_type": "encryption_rotation",
                    "fields": ["payload", "result"],
                    "limit": 100,
                    "confirmed": False,
                },
            )
            assert create_resp.status_code == 200, create_resp.text
            created = create_resp.json()["item"]
            assert created["id"] == "run-1"
            assert service.created_calls[0]["domain"] == "jobs"

            list_resp = client.get("/api/v1/admin/maintenance/rotation-runs?limit=25&offset=0")
            assert list_resp.status_code == 200, list_resp.text
            listed = list_resp.json()
            assert listed["total"] == 1
            assert listed["pagination"]["total"] == 1
            assert listed["pagination"]["limit"] == 25
            assert listed["pagination"]["offset"] == 0
            assert listed["pagination"]["has_more"] is False
            assert listed["has_more"] is False
            assert listed["next_offset"] is None
            assert listed["items"][0]["id"] == "run-1"

            detail_resp = client.get("/api/v1/admin/maintenance/rotation-runs/run-1")
            assert detail_resp.status_code == 200, detail_resp.text
            assert detail_resp.json()["id"] == "run-1"
            assert detail_resp.json()["affected_count"] == 42
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_maintenance_rotation_create_fails_closed_when_worker_is_disabled(monkeypatch, tmp_path) -> None:
    _setup_env(monkeypatch, user_db_base=str(tmp_path / "user_dbs"))
    monkeypatch.delenv("ADMIN_MAINTENANCE_ROTATION_JOBS_WORKER_ENABLED", raising=False)

    from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops

    service = _DisabledWorkerService()
    app.dependency_overrides[admin_ops.get_admin_maintenance_rotation_service] = lambda: service
    app.dependency_overrides[admin_ops.get_maintenance_rotation_job_enqueuer] = lambda: _noop_enqueue_run

    headers = {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}

    try:
        with TestClient(app, headers=headers) as client:
            response = client.post(
                "/api/v1/admin/maintenance/rotation-runs",
                json={
                    "mode": "dry_run",
                    "domain": "jobs",
                    "queue": "default",
                    "job_type": "encryption_rotation",
                    "fields": ["payload", "result"],
                    "limit": 100,
                    "confirmed": False,
                },
            )
            assert response.status_code == 503, response.text
            assert response.json()["detail"] == "maintenance_rotation_worker_unavailable"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_maintenance_rotation_execute_create_requires_confirmation(monkeypatch, tmp_path) -> None:
    _setup_env(monkeypatch, user_db_base=str(tmp_path / "user_dbs"))

    from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops

    service = _FakeRotationService()
    app.dependency_overrides[admin_ops.get_admin_maintenance_rotation_service] = lambda: service
    app.dependency_overrides[admin_ops.get_maintenance_rotation_job_enqueuer] = lambda: _noop_enqueue_run

    headers = {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}

    try:
        with TestClient(app, headers=headers) as client:
            response = client.post(
                "/api/v1/admin/maintenance/rotation-runs",
                json={
                    "mode": "execute",
                    "domain": "jobs",
                    "queue": "default",
                    "job_type": "encryption_rotation",
                    "fields": ["payload"],
                    "limit": 100,
                    "confirmed": False,
                },
            )
            assert response.status_code == 400, response.text
            assert response.json()["detail"] == "confirmation_required"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_maintenance_rotation_create_preserves_domain_scope_enforcement(
    monkeypatch,
    tmp_path,
) -> None:
    _setup_env(monkeypatch, user_db_base=str(tmp_path / "user_dbs"))

    from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops

    principal = AuthPrincipal(
        kind="user",
        user_id=41,
        subject="user:41",
        roles=["admin"],
        is_admin=True,
    )

    def _fake_scope_check(*args, **kwargs):
        raise HTTPException(status_code=403, detail="Not authorized for this domain")

    service = _FakeRotationService()
    app.dependency_overrides[get_auth_principal] = lambda: principal
    app.dependency_overrides[admin_ops.get_admin_maintenance_rotation_service] = lambda: service
    app.dependency_overrides[admin_ops.get_maintenance_rotation_job_enqueuer] = lambda: _noop_enqueue_run
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.admin.admin_ops._enforce_domain_scope_unified",
        _fake_scope_check,
    )

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/admin/maintenance/rotation-runs",
                json={
                    "mode": "dry_run",
                    "domain": "jobs",
                    "queue": "default",
                    "job_type": "encryption_rotation",
                    "fields": ["payload"],
                    "limit": 100,
                    "confirmed": False,
                },
            )
            assert response.status_code == 403, response.text
            assert response.json()["detail"] == "Not authorized for this domain"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_maintenance_rotation_list_applies_domain_visibility_before_pagination(
    monkeypatch,
    tmp_path,
) -> None:
    _setup_env(monkeypatch, user_db_base=str(tmp_path / "user_dbs"))
    monkeypatch.setenv("JOBS_DOMAIN_SCOPED_RBAC", "true")
    monkeypatch.setenv("JOBS_DOMAIN_ALLOWLIST_41", "visible-domain")

    from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.maintenance_rotation_runs_repo import (
        AuthnzMaintenanceRotationRunsRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    principal = AuthPrincipal(
        kind="user",
        user_id=41,
        subject="user:41",
        roles=["admin"],
        is_admin=True,
    )

    reset_settings()
    await reset_db_pool()
    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(Path(os.environ["DATABASE_URL"].removeprefix("sqlite:///")))))
    repo = AuthnzMaintenanceRotationRunsRepo(pool)
    await repo.ensure_schema()

    visible = await repo.create_run(
        mode="dry_run",
        domain="visible-domain",
        queue="default",
        job_type="encryption_rotation",
        fields_json='["payload"]',
        limit=100,
        requested_by_user_id=41,
        requested_by_label="ops-admin@example.com",
        confirmation_recorded=False,
        scope_summary="domain=visible-domain, queue=default, fields=payload, limit=100",
        key_source="env:jobs_crypto_rotate",
    )
    await repo.mark_complete(str(visible["id"]), affected_count=7)

    hidden = await repo.create_run(
        mode="dry_run",
        domain="hidden-domain",
        queue="default",
        job_type="encryption_rotation",
        fields_json='["payload"]',
        limit=100,
        requested_by_user_id=41,
        requested_by_label="ops-admin@example.com",
        confirmation_recorded=False,
        scope_summary="domain=hidden-domain, queue=default, fields=payload, limit=100",
        key_source="env:jobs_crypto_rotate",
    )
    await repo.mark_complete(str(hidden["id"]), affected_count=3)

    app.dependency_overrides[get_auth_principal] = lambda: principal

    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/admin/maintenance/rotation-runs?limit=1&offset=0")
            assert response.status_code == 200, response.text
            payload = response.json()
            assert payload["total"] == 1
            assert payload["pagination"]["total"] == 1
            assert payload["pagination"]["limit"] == 1
            assert payload["pagination"]["offset"] == 0
            assert payload["pagination"]["has_more"] is False
            assert [item["id"] for item in payload["items"]] == [visible["id"]]
    finally:
        app.dependency_overrides.clear()
