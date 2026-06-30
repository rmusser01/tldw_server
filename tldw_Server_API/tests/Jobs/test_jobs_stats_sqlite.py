import os
from typing import Dict, Tuple

from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Jobs.manager import JobManager


def _set_env(monkeypatch):


    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.delenv("SINGLE_USER_API_KEY", raising=False)
    import os as _os
    monkeypatch.setenv("JOBS_DB_PATH", _os.path.join(_os.getcwd(), "Databases", "jobs.db"))


def _map_by_key(rows):


    out: Dict[Tuple[str, str, str], Dict] = {}
    for r in rows:
        out[(r["domain"], r["queue"], r["job_type"])] = r
    return out


def test_jobs_stats_shape_and_counts_sqlite(monkeypatch, tmp_path):


     # Isolate DB in a temp CWD so Databases/jobs.db is per-test
    monkeypatch.chdir(tmp_path)
    _set_env(monkeypatch)

    # Import app after env is set
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
    reset_settings()
    from tldw_Server_API.app.main import app

    jm = JobManager()

    # Seed chatbooks/default/export: 3 queued -> acquire 1 (processing)
    j1 = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    j2 = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    j3 = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    acquired = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="w1")
    assert acquired is not None

    # Seed chatbooks/high/export: 1 queued
    jm.create_job(domain="chatbooks", queue="high", job_type="export", payload={}, owner_user_id="1")

    # Seed other/default/import: 2 queued -> acquire 1 (processing)
    k1 = jm.create_job(domain="other", queue="default", job_type="import", payload={}, owner_user_id="2")
    k2 = jm.create_job(domain="other", queue="default", job_type="import", payload={}, owner_user_id="2")
    acquired2 = jm.acquire_next_job(domain="other", queue="default", lease_seconds=30, worker_id="w2")
    assert acquired2 is not None

    # Completed job should not contribute to queued/processing counts
    jc = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    jm.complete_job(int(jc["id"]))

    headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
    with TestClient(app, headers=headers) as client:
        r = client.get("/api/v1/jobs/stats")
        assert r.status_code == 200, r.text
        data = r.json()
        assert isinstance(data, list)
        # Map by (domain, queue, job_type)
        m = _map_by_key(data)

        # chatbooks/default/export -> 2 queued (3 created - 1 acquired), 1 processing
        g = m[("chatbooks", "default", "export")]
        assert g["queued"] == 2
        assert g["processing"] == 1

        # chatbooks/high/export -> 1 queued, 0 processing
        g = m[("chatbooks", "high", "export")]
        assert g["queued"] == 1
        assert g["processing"] == 0

        # other/default/import -> 1 queued, 1 processing
        g = m[("other", "default", "import")]
        assert g["queued"] == 1
        assert g["processing"] == 1


def test_jobs_stats_filters_sqlite(monkeypatch, tmp_path):


    monkeypatch.chdir(tmp_path)
    _set_env(monkeypatch)

    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
    reset_settings()
    from tldw_Server_API.app.main import app

    jm = JobManager()

    # Seed multiple groups
    jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    jm.create_job(domain="chatbooks", queue="high", job_type="export", payload={}, owner_user_id="1")
    jm.create_job(domain="other", queue="default", job_type="import", payload={}, owner_user_id="2")

    headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
    with TestClient(app, headers=headers) as client:
        # Filter by domain
        r = client.get("/api/v1/jobs/stats", params={"domain": "chatbooks"})
        assert r.status_code == 200
        domains = {row["domain"] for row in r.json()}
        assert domains == {"chatbooks"}

        # Filter by queue
        r2 = client.get("/api/v1/jobs/stats", params={"queue": "default"})
        assert r2.status_code == 200
        queues = {row["queue"] for row in r2.json()}
        assert queues == {"default"}

        # Filter by job_type
        r3 = client.get("/api/v1/jobs/stats", params={"job_type": "export"})
        assert r3.status_code == 200
        job_types = {row["job_type"] for row in r3.json()}
        assert job_types == {"export"}

        # Combined filters -> single expected group
        r4 = client.get(
            "/api/v1/jobs/stats",
            params={"domain": "chatbooks", "queue": "high", "job_type": "export"},
        )
        assert r4.status_code == 200
        payload = r4.json()
        assert len(payload) == 1
        only = payload[0]
        assert only["domain"] == "chatbooks"
        assert only["queue"] == "high"
        assert only["job_type"] == "export"


def test_jobs_stats_sanitizes_generic_failure(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _set_env(monkeypatch)

    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
    reset_settings()
    from tldw_Server_API.app.main import app

    def boom(self, **_kwargs):
        raise RuntimeError("jobs stats backend exploded")

    monkeypatch.setattr(JobManager, "get_queue_stats", boom)

    headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
    with TestClient(app, headers=headers) as client:
        r = client.get("/api/v1/jobs/stats")
        assert r.status_code == 500
        assert r.json()["detail"] == "Stats failed"
