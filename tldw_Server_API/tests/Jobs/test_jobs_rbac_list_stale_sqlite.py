import os
import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Jobs.manager import JobManager


def _set_env(monkeypatch, tmp_path):


    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    # Force domain-scoped RBAC behavior in single-user mode
    monkeypatch.setenv("JOBS_DOMAIN_SCOPED_RBAC", "true")
    monkeypatch.setenv("JOBS_RBAC_FORCE", "true")
    monkeypatch.setenv("JOBS_REQUIRE_DOMAIN_FILTER", "true")
    monkeypatch.setenv("JOBS_DB_PATH", os.path.join(os.getcwd(), "Databases", "jobs.db"))


def _client(monkeypatch):


    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
    reset_settings()
    from tldw_Server_API.app.main import app
    try:
        app.dependency_overrides.clear()
    except Exception:
        _ = None
    headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
    return app, headers


def test_rbac_for_list_and_stale_requires_domain_and_allowlist(monkeypatch, tmp_path):


    _set_env(monkeypatch, tmp_path)
    from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables
    ensure_jobs_tables(tmp_path / "jobs.db")
    app, headers = _client(monkeypatch)

    with TestClient(app, headers=headers) as client:
        # Missing domain -> 403 when filter required
        r1 = client.get("/api/v1/jobs/list")
        r2 = client.get("/api/v1/jobs/stale")
        assert r1.status_code == 403
        assert r2.status_code == 403

        # Domain provided but not in allowlist -> 403
        monkeypatch.setenv("JOBS_DOMAIN_ALLOWLIST_1", "chatbooks")
        r3 = client.get("/api/v1/jobs/list", params={"domain": "other"})
        r4 = client.get("/api/v1/jobs/stale", params={"domain": "other"})
        assert r3.status_code == 403
        assert r4.status_code == 403

        # Allowed domain -> 200 (empty lists are fine)
        r5 = client.get("/api/v1/jobs/list", params={"domain": "chatbooks"})
        r6 = client.get("/api/v1/jobs/stale", params={"domain": "chatbooks"})
        assert r5.status_code == 200
        assert r6.status_code == 200


def test_jobs_stale_sanitizes_generic_failure(monkeypatch, tmp_path):
    _set_env(monkeypatch, tmp_path)
    monkeypatch.setenv("JOBS_DOMAIN_ALLOWLIST_1", "chatbooks")

    def boom(self):
        raise RuntimeError("jobs stale backend exploded")

    monkeypatch.setattr(JobManager, "_connect", boom)

    app, headers = _client(monkeypatch)
    with TestClient(app, headers=headers) as client:
        r = client.get("/api/v1/jobs/stale", params={"domain": "chatbooks"})
        assert r.status_code == 500
        assert r.json()["detail"] == "Stale groups failed"
