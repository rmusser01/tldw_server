import os
import time

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Jobs.manager import JobManager


def _set_env(monkeypatch, tmp_path):


    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.delenv("SINGLE_USER_API_KEY", raising=False)
    monkeypatch.setenv("JOBS_DB_PATH", os.path.join(os.getcwd(), "Databases", "jobs.db"))


def test_jobs_list_sorting_sqlite(monkeypatch, tmp_path):


    _set_env(monkeypatch, tmp_path)
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
    reset_settings()
    from tldw_Server_API.app.main import app

    jm = JobManager()
    # Create a few jobs with different priority
    a = jm.create_job(domain="d", queue="default", job_type="t", payload={}, owner_user_id="1", priority=9)
    time.sleep(0.05)
    b = jm.create_job(domain="d", queue="default", job_type="t", payload={}, owner_user_id="1", priority=3)
    time.sleep(0.05)
    c = jm.create_job(domain="d", queue="default", job_type="t", payload={}, owner_user_id="1", priority=7)

    headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
    with TestClient(app, headers=headers) as client:
        # Default: created_at desc, so last created first -> c
        r = client.get("/api/v1/jobs/list", params={"domain": "d", "limit": 10})
        assert r.status_code == 200
        ids_default = [row["id"] for row in r.json()]
        assert ids_default[0] == c["id"]

        # Sort by created_at asc -> a first
        r2 = client.get("/api/v1/jobs/list", params={"domain": "d", "limit": 10, "sort_order": "asc"})
        assert r2.status_code == 200
        ids_asc = [row["id"] for row in r2.json()]
        assert ids_asc[0] == a["id"]

        # Sort by priority asc -> b (priority 3) first
        r3 = client.get("/api/v1/jobs/list", params={"domain": "d", "limit": 10, "sort_by": "priority", "sort_order": "asc"})
        assert r3.status_code == 200
        ids_prio = [row["id"] for row in r3.json()]
        assert ids_prio[0] == b["id"]


def test_jobs_list_sanitizes_generic_failure(monkeypatch, tmp_path):
    _set_env(monkeypatch, tmp_path)
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
    reset_settings()
    from tldw_Server_API.app.main import app

    def boom(self, **_kwargs):
        raise RuntimeError("jobs list backend exploded")

    monkeypatch.setattr(JobManager, "list_jobs", boom)

    headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
    with TestClient(app, headers=headers) as client:
        r = client.get("/api/v1/jobs/list", params={"domain": "d", "limit": 10})
        assert r.status_code == 500
        assert r.json()["detail"] == "List failed"
