import os
from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Jobs.manager import JobManager


def _set_env(monkeypatch):


    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.delenv("SINGLE_USER_API_KEY", raising=False)
    import os as _os
    monkeypatch.setenv("JOBS_DB_PATH", _os.path.join(_os.getcwd(), "Databases", "jobs.db"))
    # Ensure chatbooks domain uses DESC priority ordering for these TTL tests
    monkeypatch.setenv("JOBS_ACQUIRE_PRIORITY_DESC_DOMAINS", "chatbooks")


def _backdate_sqlite_fields(job_id: int, *, created_delta_s: int = 0, runtime_delta_s: int = 0):
    jm = JobManager()
    conn = jm._connect()
    try:
        with conn:
            if created_delta_s:
                created_at = (datetime.utcnow() - timedelta(seconds=created_delta_s)).strftime("%Y-%m-%d %H:%M:%S")
                conn.execute(
                    "UPDATE jobs SET created_at = ?, updated_at = ? WHERE id = ?",
                    (created_at, created_at, int(job_id)),
                )
            if runtime_delta_s:
                started_at = (datetime.utcnow() - timedelta(seconds=runtime_delta_s)).strftime("%Y-%m-%d %H:%M:%S")
                conn.execute(
                    "UPDATE jobs SET started_at = ?, acquired_at = ? WHERE id = ?",
                    (started_at, started_at, int(job_id)),
                )
    finally:
        try:
            conn.close()
        except Exception:
            _ = None


def test_ttl_sweep_cancel(monkeypatch, tmp_path):


    monkeypatch.chdir(tmp_path)
    _set_env(monkeypatch)

    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
    reset_settings()
    from tldw_Server_API.app.main import app

    jm = JobManager()

    # Seed queued (age TTL) and processing (runtime TTL)
    j1 = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    _backdate_sqlite_fields(int(j1["id"]), created_delta_s=3_600 * 2)  # created 2h ago


    # Ensure this one is acquired: higher numeric priority wins in DESC ordering
    j2 = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1", priority=10)
    acq = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="w1")
    assert acq is not None
    _backdate_sqlite_fields(int(acq["id"]), runtime_delta_s=3_600 * 3)  # running 3h

    headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
    with TestClient(app, headers=headers) as client:
        body = {
            "age_seconds": 3600,  # 1h
            "runtime_seconds": 7200,  # 2h
            "action": "cancel",
            "domain": "chatbooks",
            "queue": "default",
            "job_type": "export",
        }
        r = client.post("/api/v1/jobs/ttl/sweep", json=body, headers={**headers, "X-Confirm": "true"})
        assert r.status_code == 200, r.text
        affected = r.json()["affected"]
        assert affected >= 2

    # Verify via stats: both queued and processing should now be 0
    with TestClient(app, headers=headers) as client2:
        rstats = client2.get("/api/v1/jobs/stats", params={"domain": "chatbooks", "queue": "default", "job_type": "export"})
        assert rstats.status_code == 200
        rows = rstats.json()
        assert len(rows) == 1
        row = rows[0]
        assert row["queued"] == 0
        assert row["processing"] == 0


def test_ttl_sweep_fail(monkeypatch, tmp_path):


    monkeypatch.chdir(tmp_path)
    _set_env(monkeypatch)

    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
    reset_settings()
    from tldw_Server_API.app.main import app

    jm = JobManager()

    # Make this one the processing target: higher numeric priority wins in DESC ordering
    j1 = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1", priority=10)
    # And a second queued job that is old enough for age TTL
    j2 = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    _backdate_sqlite_fields(int(j2["id"]), created_delta_s=3_600 * 2)

    acq = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="w2")
    assert acq is not None
    _backdate_sqlite_fields(int(acq["id"]), runtime_delta_s=3_600 * 3)

    headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
    with TestClient(app, headers=headers) as client:
        body = {
            "age_seconds": 3600,
            "runtime_seconds": 3600,
            "action": "fail",
            "domain": "chatbooks",
            "queue": "default",
            "job_type": "export",
        }
        r = client.post("/api/v1/jobs/ttl/sweep", json=body, headers={**headers, "X-Confirm": "true"})
        assert r.status_code == 200
        assert r.json()["affected"] >= 2

    # Verify via stats: both queued and processing should now be 0
    with TestClient(app, headers=headers) as client2:
        rstats = client2.get("/api/v1/jobs/stats", params={"domain": "chatbooks", "queue": "default", "job_type": "export"})
        assert rstats.status_code == 200
        rows = rstats.json()
        assert len(rows) == 1
        row = rows[0]
        assert row["queued"] == 0
        assert row["processing"] == 0


def test_ttl_sweep_sanitizes_generic_failure(monkeypatch, tmp_path):


    monkeypatch.chdir(tmp_path)
    _set_env(monkeypatch)

    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
    reset_settings()
    from tldw_Server_API.app.main import app

    def boom(self, **_kwargs):
        raise RuntimeError("jobs ttl backend exploded")

    monkeypatch.setattr(JobManager, "apply_ttl_policies", boom)

    headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
    with TestClient(app, headers=headers) as client:
        r = client.post(
            "/api/v1/jobs/ttl/sweep",
            json={
                "age_seconds": 3600,
                "runtime_seconds": 3600,
                "action": "cancel",
                "domain": "chatbooks",
                "queue": "default",
                "job_type": "export",
            },
            headers={**headers, "X-Confirm": "true"},
        )
        assert r.status_code == 500
        assert r.json()["detail"] == "TTL sweep failed"
