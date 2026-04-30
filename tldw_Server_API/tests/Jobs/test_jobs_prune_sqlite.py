import os
from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Jobs.manager import JobManager


def _set_env(monkeypatch):


    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    # Do not set SINGLE_USER_API_KEY so tests use deterministic key from settings
    monkeypatch.delenv("SINGLE_USER_API_KEY", raising=False)
    # Ensure API endpoints use the same SQLite DB as this test (per tmp CWD)
    import os as _os
    monkeypatch.setenv("JOBS_DB_PATH", _os.path.join(_os.getcwd(), "Databases", "jobs.db"))


def _backdate_sqlite(job_id: int, days: int = 2):
    jm = JobManager()
    conn = jm._connect()
    try:
        cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        with conn:
            conn.execute(
                "UPDATE jobs SET completed_at = ?, updated_at = ? WHERE id = ?",
                (cutoff, cutoff, int(job_id)),
            )
    finally:
        try:
            conn.close()
        except Exception:
            _ = None


def test_jobs_prune_dry_run_and_filters_sqlite(monkeypatch, tmp_path):


     # Isolate DB in a temp CWD so Databases/jobs.db is per-test
    monkeypatch.chdir(tmp_path)
    _set_env(monkeypatch)

    # Import app after env is set
    # Reset settings before importing app to pick up TEST_MODE
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
    reset_settings()
    from tldw_Server_API.app.main import app
    try:
        app.dependency_overrides.clear()
    except Exception:
        _ = None

    jm = JobManager()
    # Seed: 2 completed + 1 failed (old), 1 failed (recent)
    j1 = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    jm.complete_job(int(j1["id"]))
    _backdate_sqlite(int(j1["id"]))

    j2 = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    jm.complete_job(int(j2["id"]))
    _backdate_sqlite(int(j2["id"]))

    j3 = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    jm.fail_job(int(j3["id"]), error="x", retryable=False)
    _backdate_sqlite(int(j3["id"]))

    j4 = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    jm.fail_job(int(j4["id"]), error="x", retryable=False)  # recent (should not be pruned with older_than_days=1)

    # Use the deterministic single-user key from settings
    headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
    with TestClient(app, headers=headers) as client:
        body = {
            "statuses": ["completed", "failed"],
            "older_than_days": 1,
            "domain": "chatbooks",
            "queue": "default",
            "job_type": "export",
            "dry_run": True,
        }
        r = client.post("/api/v1/jobs/prune", json=body)
        assert r.status_code == 200, r.text
        assert r.json()["deleted"] == 3  # two completed + one failed (old)

        # Execute prune
        body["dry_run"] = False
        r2 = client.post("/api/v1/jobs/prune", json=body)
        assert r2.status_code == 200
        assert r2.json()["deleted"] == 3

        # Subsequent dry-run should report 0
        body["dry_run"] = True
        r3 = client.post("/api/v1/jobs/prune", json=body)
        assert r3.status_code == 200
        assert r3.json()["deleted"] == 0


def test_jobs_prune_filters_scope_sqlite(monkeypatch, tmp_path):


     # New temp CWD for isolation
    monkeypatch.chdir(tmp_path)
    _set_env(monkeypatch)

    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
    reset_settings()
    from tldw_Server_API.app.main import app
    try:
        app.dependency_overrides.clear()
    except Exception:
        _ = None

    jm = JobManager()
    # Seed one job in a different domain/queue
    jx = jm.create_job(domain="other", queue="low", job_type="export", payload={}, owner_user_id="1")
    jm.complete_job(int(jx["id"]))
    _backdate_sqlite(int(jx["id"]))

    headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
    with TestClient(app, headers=headers) as client:
        # Scoped to chatbooks/default/export - should not match the seeded job
        body = {
            "statuses": ["completed"],
            "older_than_days": 1,
            "domain": "chatbooks",
            "queue": "default",
            "job_type": "export",
            "dry_run": True,
        }
        r = client.post("/api/v1/jobs/prune", json=body)
        assert r.status_code == 200
        assert r.json()["deleted"] == 0


def test_jobs_prune_sanitizes_generic_failure(monkeypatch, tmp_path):


    monkeypatch.chdir(tmp_path)
    _set_env(monkeypatch)

    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
    reset_settings()
    from tldw_Server_API.app.main import app

    def boom(self, **_kwargs):
        raise RuntimeError("jobs prune backend exploded")

    monkeypatch.setattr(JobManager, "prune_jobs", boom)

    headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
    with TestClient(app, headers=headers) as client:
        r = client.post(
            "/api/v1/jobs/prune",
            json={
                "statuses": ["completed"],
                "older_than_days": 1,
                "domain": "chatbooks",
                "queue": "default",
                "job_type": "export",
                "dry_run": True,
            },
        )
        assert r.status_code == 500
        assert r.json()["detail"] == "Prune failed"
