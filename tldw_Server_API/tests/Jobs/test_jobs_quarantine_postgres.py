import time

import pytest

psycopg = pytest.importorskip("psycopg")
pytestmark = pytest.mark.pg_jobs

from tldw_Server_API.app.core.Jobs.manager import JobManager


@pytest.fixture(autouse=True)
def _setup(jobs_pg_dsn):
    return


def test_poison_quarantine_on_retries_postgres(monkeypatch, jobs_pg_dsn):


    monkeypatch.setenv("JOBS_QUARANTINE_THRESHOLD", "2")
    jm = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    j = jm.create_job(domain="test", queue="default", job_type="t", payload={}, owner_user_id="u")
    acq = jm.acquire_next_job(domain="test", queue="default", lease_seconds=5, worker_id="w")
    assert acq and acq.get("id") == j["id"]
    lease_id = str(acq.get("lease_id"))
    ok1 = jm.fail_job(int(j["id"]), error="boom", retryable=True, backoff_seconds=1, worker_id="w", lease_id=lease_id, error_code="E1")
    assert ok1 is True
    time.sleep(1.1)
    acq2 = jm.acquire_next_job(domain="test", queue="default", lease_seconds=5, worker_id="w")
    assert acq2 and acq2.get("id") == j["id"]
    ok2 = jm.fail_job(int(j["id"]), error="boom", retryable=True, backoff_seconds=1, worker_id="w", lease_id=str(acq2.get("lease_id")), error_code="E1")
    assert ok2 is True
    row = jm.get_job(int(j["id"]))
    assert row and row.get("status") == "quarantined"


def test_error_code_change_restarts_streak_without_quarantining_postgres(
    monkeypatch,
    jobs_pg_dsn,
):
    monkeypatch.setenv("JOBS_QUARANTINE_THRESHOLD", "2")
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    jm = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    job = jm.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={},
        owner_user_id="owner-1",
        idempotency_key="slides:v1:" + "2" * 64,
    )
    acquired = jm.acquire_next_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        lease_seconds=30,
        worker_id="slides-worker",
    )
    assert acquired is not None
    with psycopg.connect(jobs_pg_dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            "UPDATE jobs SET failure_streak_code=%s, failure_streak_count=%s WHERE id=%s",
            ("old_error", 1, int(job["id"])),
        )

    assert jm.fail_job(
        int(job["id"]),
        error="bounded retry detail",
        retryable=True,
        backoff_seconds=1,
        worker_id="slides-worker",
        lease_id=str(acquired["lease_id"]),
        error_code="new_error",
    )

    stored = jm.get_job(int(job["id"]))
    assert stored["status"] == "queued"
    assert stored["failure_streak_code"] == "new_error"
    assert stored["failure_streak_count"] == 1
    assert stored["quarantined_at"] is None
    stats = jm.get_queue_stats(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
    )
    assert stats and stats[0]["quarantined"] == 0


@pytest.mark.parametrize(
    ("quarantine_kwargs", "expected_status"),
    [({"quarantine": False}, "queued"), ({}, "quarantined")],
    ids=["quarantine-off", "default"],
)
def test_fail_job_without_quarantine_keeps_retrying_postgres(monkeypatch, jobs_pg_dsn, quarantine_kwargs, expected_status):
    monkeypatch.setenv("JOBS_QUARANTINE_THRESHOLD", "2")
    jm = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    job = jm.create_job(domain="test", queue="default", job_type="t", payload={}, owner_user_id="u")
    for _ in range(2):
        with psycopg.connect(jobs_pg_dsn) as connection, connection.cursor() as cursor:
            cursor.execute("UPDATE jobs SET available_at = NULL WHERE id = %s", (int(job["id"]),))
        acquired = jm.acquire_next_job(domain="test", queue="default", lease_seconds=5, worker_id="w")
        assert acquired and acquired["id"] == job["id"]
        assert jm.fail_job(
            int(job["id"]),
            error="slow down",
            retryable=True,
            backoff_seconds=1,
            worker_id="w",
            lease_id=str(acquired["lease_id"]),
            error_code="rate_limit",
            **quarantine_kwargs,
        )
    stored = jm.get_job(int(job["id"])) or {}
    assert (stored["status"], stored["retry_count"], stored["failure_streak_count"]) == (expected_status, 2, 2)


@pytest.mark.parametrize(
    "configured, backoff, first_failure_quarantined",
    [
        (None, 0, False),
        ("1", 0, False),
        ("1", 10, True),
        ("2", 10, False),
    ],
)
def test_retry_quarantine_cutoff_preserves_postgres_test_policy(
    monkeypatch,
    jobs_pg_dsn,
    configured,
    backoff,
    first_failure_quarantined,
):
    """PostgreSQL retains its zero-backoff test exemption even with an explicit cutoff."""
    monkeypatch.setenv("TEST_MODE", "true")
    if configured is None:
        monkeypatch.delenv("JOBS_QUARANTINE_THRESHOLD", raising=False)
    else:
        monkeypatch.setenv("JOBS_QUARANTINE_THRESHOLD", configured)
    manager = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    assert (manager.retry_quarantine_threshold(backoff_seconds=backoff) <= 1) is first_failure_quarantined  # nosec B101 - pytest regression assertion.
