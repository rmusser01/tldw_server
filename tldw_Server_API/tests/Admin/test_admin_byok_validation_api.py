from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.exceptions import (
    ByokValidationActiveRunError,
    ByokValidationRunNotFoundError,
)


def _setup_env(monkeypatch, *, user_db_base: str) -> None:
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "unit-test-api-key")
    monkeypatch.setenv("USER_DB_BASE_DIR", user_db_base)
    auth_db_path = Path(user_db_base).parent / "users_test_byok_validation_api.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{auth_db_path}")
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.delenv("ADMIN_BYOK_VALIDATION_JOBS_WORKER_ENABLED", raising=False)


def _validation_worker_enabled():
    return patch(
        "tldw_Server_API.app.services.admin_byok_validation_jobs_worker.byok_validation_worker_enabled",
        return_value=True,
    )


async def _admin_principal_override(request: Request) -> AuthPrincipal:
    principal = AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        subject="ops-admin@example.com",
        token_type="access",  # nosec B106
        jti=None,
        roles=["admin"],
        permissions=["system.configure"],
        is_admin=True,
        org_ids=[],
        team_ids=[],
        email="ops-admin@example.com",
    )
    request.state.auth = AuthContext(
        principal=principal,
        ip=None,
        user_agent=None,
        request_id=None,
    )
    return principal


def _reset_byok_api_overrides() -> None:
    app.dependency_overrides.clear()
    app.dependency_overrides[get_auth_principal] = _admin_principal_override


@dataclass
class _FakeByokValidationService:
    created_calls: list[dict] = field(default_factory=list)

    async def create_run(self, principal, *, org_id: int | None, provider: str | None):
        self.created_calls.append(
            {
                "principal": principal,
                "org_id": org_id,
                "provider": provider,
            }
        )
        return {
            "id": "run-1",
            "status": "queued",
            "org_id": org_id,
            "provider": provider,
            "keys_checked": None,
            "valid_count": None,
            "invalid_count": None,
            "error_count": None,
            "requested_by_user_id": principal.user_id,
            "requested_by_label": principal.email,
            "job_id": None,
            "scope_summary": "org=42, provider=openai",
            "error_message": None,
            "created_at": "2026-03-12T20:00:00+00:00",
            "started_at": None,
            "completed_at": None,
        }

    async def list_runs(self, *, limit: int, offset: int):
        return (
            [
                {
                    "id": "run-1",
                    "status": "complete",
                    "org_id": 42,
                    "provider": "openai",
                    "keys_checked": 5,
                    "valid_count": 4,
                    "invalid_count": 1,
                    "error_count": 0,
                    "requested_by_user_id": 1,
                    "requested_by_label": "ops-admin@example.com",
                    "job_id": "job-1",
                    "scope_summary": "org=42, provider=openai",
                    "error_message": None,
                    "created_at": "2026-03-12T20:00:00+00:00",
                    "started_at": "2026-03-12T20:00:05+00:00",
                    "completed_at": "2026-03-12T20:00:15+00:00",
                }
            ],
            1,
        )

    async def get_run(self, run_id: str):
        if run_id != "run-1":
            raise ByokValidationRunNotFoundError("byok_validation_run_not_found")
        return {
            "id": "run-1",
            "status": "complete",
            "org_id": 42,
            "provider": "openai",
            "keys_checked": 5,
            "valid_count": 4,
            "invalid_count": 1,
            "error_count": 0,
            "requested_by_user_id": 1,
            "requested_by_label": "ops-admin@example.com",
            "job_id": "job-1",
            "scope_summary": "org=42, provider=openai",
            "error_message": None,
            "created_at": "2026-03-12T20:00:00+00:00",
            "started_at": "2026-03-12T20:00:05+00:00",
            "completed_at": "2026-03-12T20:00:15+00:00",
        }


@dataclass
class _ConflictByokValidationService(_FakeByokValidationService):
    async def create_run(self, principal, *, org_id: int | None, provider: str | None):
        raise ByokValidationActiveRunError("active_validation_run_exists")


async def _noop_enqueue_run(item):
    return "job-1"


@pytest_asyncio.fixture
async def byok_api_client(tmp_path):
    monkeypatch = pytest.MonkeyPatch()
    _setup_env(monkeypatch, user_db_base=str(tmp_path / "user_dbs"))
    headers = {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}
    _reset_byok_api_overrides()
    from tldw_Server_API.app.services.app_lifecycle import reset_lifecycle_state

    reset_lifecycle_state(app)

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver", headers=headers) as client:
            yield client
    finally:
        app.dependency_overrides.clear()
        monkeypatch.undo()


@pytest.mark.asyncio
async def test_admin_byok_validation_create_list_and_detail_roundtrip(byok_api_client) -> None:
    from tldw_Server_API.app.api.v1.endpoints.admin import admin_byok

    service = _FakeByokValidationService()
    app.dependency_overrides[admin_byok.get_admin_byok_validation_service] = lambda: service
    app.dependency_overrides[admin_byok.get_byok_validation_job_enqueuer] = lambda: _noop_enqueue_run

    with _validation_worker_enabled():
        create_resp = await byok_api_client.post(
            "/api/v1/admin/byok/validation-runs",
            json={"org_id": 42, "provider": "openai"},
        )
    assert create_resp.status_code == 200, create_resp.text
    created = create_resp.json()
    assert created["id"] == "run-1"
    assert service.created_calls[0]["org_id"] == 42
    assert service.created_calls[0]["provider"] == "openai"

    list_resp = await byok_api_client.get("/api/v1/admin/byok/validation-runs?limit=25&offset=0")
    assert list_resp.status_code == 200, list_resp.text
    listed = list_resp.json()
    assert listed["total"] == 1
    assert listed["pagination"]["total"] == 1
    assert listed["pagination"]["limit"] == 25
    assert listed["pagination"]["offset"] == 0
    assert listed["pagination"]["has_more"] is False
    assert listed["pagination"]["next_offset"] is None
    assert listed["has_more"] is False
    assert listed["next_offset"] is None
    assert listed["items"][0]["id"] == "run-1"

    detail_resp = await byok_api_client.get("/api/v1/admin/byok/validation-runs/run-1")
    assert detail_resp.status_code == 200, detail_resp.text
    assert detail_resp.json()["id"] == "run-1"
    assert detail_resp.json()["keys_checked"] == 5


@pytest.mark.asyncio
async def test_admin_byok_validation_create_maps_active_run_conflict(byok_api_client) -> None:
    from tldw_Server_API.app.api.v1.endpoints.admin import admin_byok

    app.dependency_overrides[admin_byok.get_admin_byok_validation_service] = (
        lambda: _ConflictByokValidationService()
    )
    app.dependency_overrides[admin_byok.get_byok_validation_job_enqueuer] = lambda: _noop_enqueue_run

    with _validation_worker_enabled():
        response = await byok_api_client.post(
            "/api/v1/admin/byok/validation-runs",
            json={"org_id": 42, "provider": "openai"},
        )
    assert response.status_code == 409, response.text
    assert response.json()["detail"] == "active_validation_run_exists"


@pytest.mark.asyncio
async def test_admin_byok_validation_detail_returns_not_found(byok_api_client) -> None:
    from tldw_Server_API.app.api.v1.endpoints.admin import admin_byok

    app.dependency_overrides[admin_byok.get_admin_byok_validation_service] = (
        lambda: _FakeByokValidationService()
    )
    app.dependency_overrides[admin_byok.get_byok_validation_job_enqueuer] = lambda: _noop_enqueue_run

    response = await byok_api_client.get("/api/v1/admin/byok/validation-runs/missing")
    assert response.status_code == 404, response.text
    assert response.json()["detail"] == "byok_validation_run_not_found"


@pytest.mark.asyncio
async def test_admin_byok_validation_create_fails_closed_when_worker_is_disabled(byok_api_client) -> None:
    from tldw_Server_API.app.api.v1.endpoints.admin import admin_byok

    service = _FakeByokValidationService()
    app.dependency_overrides[admin_byok.get_admin_byok_validation_service] = lambda: service
    app.dependency_overrides[admin_byok.get_byok_validation_job_enqueuer] = lambda: _noop_enqueue_run

    response = await byok_api_client.post(
        "/api/v1/admin/byok/validation-runs",
        json={"org_id": 42, "provider": "openai"},
    )
    assert response.status_code == 503, response.text
    assert response.json()["detail"] == "byok_validation_worker_unavailable"
    assert service.created_calls == []
