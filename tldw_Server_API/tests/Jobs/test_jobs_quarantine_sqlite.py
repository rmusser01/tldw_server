import time

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager


@pytest.mark.unit
def test_poison_quarantine_on_retries(tmp_path, monkeypatch):
    db_path = tmp_path / "jobs.db"
    monkeypatch.setenv("JOBS_DB_PATH", str(db_path))
    monkeypatch.setenv("JOBS_QUARANTINE_THRESHOLD", "2")
    jm = JobManager()
    j = jm.create_job(domain="test", queue="default", job_type="t", payload={}, owner_user_id="u")
    acq = jm.acquire_next_job(domain="test", queue="default", lease_seconds=5, worker_id="w")
    assert acq and acq.get("id") == j["id"]
    lease_id = str(acq.get("lease_id"))
    # First retryable failure with code E1 -> requeued
    ok1 = jm.fail_job(int(j["id"]), error="boom", retryable=True, backoff_seconds=0, worker_id="w", lease_id=lease_id, error_code="E1")
    assert ok1 is True
    # Reacquire
    time.sleep(1.1)
    acq2 = jm.acquire_next_job(domain="test", queue="default", lease_seconds=5, worker_id="w")
    assert acq2 and acq2.get("id") == j["id"]
    # Second same-code failure -> hits threshold and quarantines
    ok2 = jm.fail_job(int(j["id"]), error="boom", retryable=True, worker_id="w", lease_id=str(acq2.get("lease_id")), error_code="E1")
    assert ok2 is True
    row = jm.get_job(int(j["id"]))
    assert row and row.get("status") == "quarantined"


@pytest.mark.unit
def test_different_failure_code_resets_quarantine_streak(tmp_path, monkeypatch):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs-reset.db"))
    monkeypatch.setenv("JOBS_QUARANTINE_THRESHOLD", "2")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "false")
    monkeypatch.setenv("TLDW_TEST_MODE", "true")
    manager = JobManager()
    job = manager.create_job(
        domain="test",
        queue="default",
        job_type="t",
        payload={},
        owner_user_id="u",
        max_retries=3,
    )

    def _fail_next(error_code: str) -> dict:
        acquired = manager.acquire_next_job(
            domain="test",
            queue="default",
            lease_seconds=5,
            worker_id="w",
        )
        assert acquired and acquired["id"] == job["id"]
        assert manager.fail_job(
            int(job["id"]),
            error="boom",
            retryable=True,
            backoff_seconds=0,
            worker_id="w",
            lease_id=str(acquired["lease_id"]),
            error_code=error_code,
        )
        return manager.get_job(int(job["id"])) or {}

    first = _fail_next("E1")
    reset = _fail_next("E2")
    quarantined = _fail_next("E2")

    assert (first["status"], first["failure_streak_code"], first["failure_streak_count"]) == (
        "queued",
        "E1",
        1,
    )
    assert (reset["status"], reset["failure_streak_code"], reset["failure_streak_count"]) == (
        "queued",
        "E2",
        1,
    )
    assert quarantined["status"] == "quarantined"
    assert quarantined["failure_streak_count"] == 2
