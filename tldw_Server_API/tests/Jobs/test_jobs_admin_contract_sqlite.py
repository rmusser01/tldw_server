from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables


pytestmark = pytest.mark.jobs


def _setup_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("ROUTES_STABLE_ONLY", "0")
    monkeypatch.setenv("ROUTES_ENABLE", "jobs")
    monkeypatch.setenv("PRIVILEGE_METADATA_VALIDATE_ON_STARTUP", "0")
    monkeypatch.setenv("CHATBOOKS_CORE_WORKER_ENABLED", "false")
    monkeypatch.setenv("JOBS_METRICS_GAUGES_ENABLED", "false")
    monkeypatch.setenv("JOBS_METRICS_RECONCILE_ENABLE", "false")
    monkeypatch.setenv("JOBS_WEBHOOKS_ENABLED", "false")
    monkeypatch.delenv("JOBS_DB_URL", raising=False)
    monkeypatch.setenv("JOBS_DB_PATH", os.path.join(os.getcwd(), "Databases", "jobs.db"))


def _client_headers():
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings

    reset_settings()
    from tldw_Server_API.app.main import app

    app.dependency_overrides.clear()
    return app, {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}


def test_jobs_admin_list_and_detail_public_field_contract_sqlite(monkeypatch, tmp_path):
    _setup_env(monkeypatch, tmp_path)
    ensure_jobs_tables(Path(os.environ["JOBS_DB_PATH"]))
    app, headers = _client_headers()

    manager = JobManager()
    job = manager.create_job(
        domain="ps",
        queue="default",
        job_type="contract",
        payload={"hello": "world"},
        owner_user_id="user-1",
        request_id="request-1",
        trace_id="trace-1",
    )

    with TestClient(app, headers=headers) as client:
        list_response = client.get(
            "/api/v1/jobs/list",
            params={"domain": "ps", "queue": "default", "job_type": "contract", "limit": 10},
        )
        detail_response = client.get(f"/api/v1/jobs/{int(job['id'])}", params={"domain": "ps"})

    assert list_response.status_code == 200, list_response.text
    listed = list_response.json()
    assert isinstance(listed, list)
    assert len(listed) == 1
    list_item = listed[0]
    assert set(list_item) == {
        "id",
        "uuid",
        "domain",
        "queue",
        "job_type",
        "status",
        "priority",
        "retry_count",
        "max_retries",
        "available_at",
        "created_at",
        "acquired_at",
        "started_at",
        "leased_until",
        "completed_at",
    }
    assert list_item["id"] == int(job["id"])
    assert list_item["uuid"] == job["uuid"]
    assert list_item["domain"] == "ps"
    assert list_item["queue"] == "default"
    assert list_item["job_type"] == "contract"
    assert list_item["status"] == "queued"

    assert detail_response.status_code == 200, detail_response.text
    detail = detail_response.json()
    for key in {
        "id",
        "uuid",
        "domain",
        "queue",
        "job_type",
        "status",
        "payload",
        "result",
        "archived",
        "created_at",
        "updated_at",
    }:
        assert key in detail
    assert detail["id"] == int(job["id"])
    assert detail["uuid"] == job["uuid"]
    assert detail["payload"]["hello"] == "world"
    assert detail["archived"] is False
