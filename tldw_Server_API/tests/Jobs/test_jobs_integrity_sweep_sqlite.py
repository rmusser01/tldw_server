import asyncio
import io
import os
import pytest
from fastapi.testclient import TestClient
from loguru import logger

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.services import jobs_integrity_service


@pytest.mark.unit
def test_integrity_sweep_clears_non_processing_lease(tmp_path, monkeypatch):
    db_path = tmp_path / "jobs.db"
    monkeypatch.setenv("JOBS_DB_PATH", str(db_path))
    jm = JobManager()
    j_queued = jm.create_job(domain="test", queue="default", job_type="t", payload={}, owner_user_id="u")
    j_processing = jm.create_job(domain="test", queue="default", job_type="t", payload={}, owner_user_id="u")
    # Manually inject two invalid states:
    # 1) queued job carrying lease metadata
    # 2) processing job with expired lease
    conn = jm._connect()
    try:
        conn.execute(
            "UPDATE jobs SET status='queued', lease_id='L', worker_id='W', leased_until=DATETIME('now') WHERE id = ?",
            (int(j_queued["id"]),),
        )
        conn.execute(
            "UPDATE jobs SET status='processing', leased_until=DATETIME('now','-10 minutes') WHERE id = ?",
            (int(j_processing["id"]),),
        )
        conn.commit()
    finally:
        conn.close()

    stats = jm.integrity_sweep(fix=True)
    assert stats["non_processing_with_lease"] == 1
    assert stats["processing_expired"] == 1
    assert stats["fixed"] == 2

    queued_after = jm.get_job(int(j_queued["id"]))
    processing_after = jm.get_job(int(j_processing["id"]))
    assert queued_after and not queued_after.get("lease_id") and not queued_after.get("worker_id")
    assert processing_after and processing_after.get("status") == "queued"
    assert not processing_after.get("lease_id") and not processing_after.get("worker_id")


def _init_endpoint_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.delenv("SINGLE_USER_API_KEY", raising=False)
    monkeypatch.setenv("JOBS_DB_PATH", os.path.join(os.getcwd(), "Databases", "jobs.db"))


@pytest.mark.unit
def test_integrity_sweep_endpoint_sanitizes_generic_failure(monkeypatch, tmp_path):
    _init_endpoint_env(monkeypatch, tmp_path)
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
    reset_settings()
    from tldw_Server_API.app.main import app

    def boom(self, **_kwargs):
        raise RuntimeError("jobs integrity backend exploded")

    monkeypatch.setattr(JobManager, "integrity_sweep", boom)

    headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
    with TestClient(app, headers=headers) as client:
        r = client.post("/api/v1/jobs/integrity/sweep", json={"fix": True})
        assert r.status_code == 500
        assert r.json()["detail"] == "Integrity sweep failed"


@pytest.mark.asyncio
async def test_integrity_sweeper_loop_sanitizes_sweep_exception_log(monkeypatch):
    monkeypatch.setenv("JOBS_INTEGRITY_SWEEP_ENABLED", "true")
    monkeypatch.setenv("JOBS_INTEGRITY_SWEEP_INTERVAL_SEC", "60")
    monkeypatch.delenv("JOBS_INTEGRITY_SWEEP_FIX", raising=False)

    class FailingJobManager:
        def integrity_sweep(self, *, fix):
            raise RuntimeError("failed at /tmp/secret/jobs.db with token=abc123")

    stop_event = asyncio.Event()

    async def stop_after_first_failure(_interval):
        stop_event.set()

    monkeypatch.setattr(jobs_integrity_service, "JobManager", lambda: FailingJobManager())
    monkeypatch.setattr(jobs_integrity_service.asyncio, "sleep", stop_after_first_failure)

    stream = io.StringIO()
    sink_id = logger.add(stream, level="DEBUG", format="{message} {extra}")
    try:
        await jobs_integrity_service.run_jobs_integrity_sweeper(stop_event)
    finally:
        logger.remove(sink_id)

    logs = stream.getvalue()
    assert "Jobs integrity sweep error" in logs
    assert "/tmp/secret/jobs.db" not in logs
    assert "token=abc123" not in logs
    assert "RuntimeError" in logs
