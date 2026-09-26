"""Explicit failed-job retry admission preserves Jobs identity and policy."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier
from typing import Any

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.operations.postgres import admission as pg_admission
from tldw_Server_API.app.core.Jobs.operations.sqlite import admission as sqlite_admission

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.pg_jobs)])
def jobs(request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> JobManager:
    """Use real SQLite or the official Jobs per-test PostgreSQL database."""
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES", "default,generation")
    monkeypatch.delenv("JOBS_DB_URL", raising=False)
    JobManager.set_acquire_gate(False)
    if request.param == "postgres":
        dsn = request.getfixturevalue("jobs_pg_dsn")
        return JobManager(backend="postgres", db_url=dsn)
    return JobManager(db_path=tmp_path / "jobs.db")


def failed_job(jobs: JobManager, *, key: str = "receipt-parent") -> dict[str, Any]:
    """Exhaust a real parent immediately, using the normal lease/failure gateway."""
    job = jobs.create_job(
        domain="vn_assets", queue="default", job_type="vn_asset_enqueue_batch", owner_user_id="42",
        payload={"pack_id": 1, "batch_id": 7, "user_id": 42}, idempotency_key=key, max_retries=0,
    )
    acquired = jobs.acquire_next_job(
        domain="vn_assets", queue="default", worker_id="worker", lease_seconds=60, owner_user_id="42",
    )
    assert acquired is not None
    assert jobs.fail_job(
        job["id"], error="partial fanout", retryable=True, backoff_seconds=0,
        worker_id=acquired["worker_id"], lease_id=acquired["lease_id"],
    )
    result = jobs.get_job(job["id"], owner_user_id="42")
    assert result is not None and result["status"] == "failed"
    return result


def retry(jobs: JobManager, job: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Invoke the supported exact-identity retry contract."""
    facts = {
        "job_id": job["id"], "owner_user_id": "42", "expected_uuid": job["uuid"],
        "domain": "vn_assets", "queue": "default", "job_type": "vn_asset_enqueue_batch",
        "expected_payload": {"pack_id": 1, "batch_id": 7, "user_id": 42}, "idempotency_key": job["idempotency_key"],
    }
    return jobs.retry_failed_job_admission(**{**facts, **overrides})


def snapshot(jobs: JobManager, job: dict[str, Any]) -> tuple[str, int, int]:
    """Observe state, ready counters and retry admission events independently."""
    conn = jobs._connect()
    try:
        if jobs.backend == "postgres":
            with jobs._pg_cursor(conn) as cur:
                cur.execute("SELECT ready_count FROM job_counters WHERE domain='vn_assets' AND queue='default'")
                ready = cur.fetchone()["ready_count"]
                cur.execute("SELECT COUNT(*) AS c FROM job_events WHERE event_type='job.retry_admitted'")
                events = cur.fetchone()["c"]
        else:
            ready = conn.execute(
                "SELECT ready_count FROM job_counters WHERE domain='vn_assets' AND queue='default'",
            ).fetchone()[0]
            events = conn.execute("SELECT COUNT(*) FROM job_events WHERE event_type='job.retry_admitted'").fetchone()[0]
        return (jobs.get_job(job["id"], owner_user_id="42")["status"], int(ready), int(events))
    finally:
        conn.close()


def test_explicit_retry_and_queued_replay_preserve_identity(jobs: JobManager) -> None:
    """A bounded new attempt budget is admitted once, not a new Job or key."""
    job = failed_job(jobs)
    result = retry(jobs, job)
    assert (result["id"], result["uuid"], result["idempotency_key"], result["payload"], result["max_retries"]) == (
        job["id"], job["uuid"], job["idempotency_key"], job["payload"], 0,
    )
    assert retry(jobs, job)["id"] == job["id"]
    assert snapshot(jobs, job) == ("queued", 1, 1)


@pytest.mark.parametrize("field,value", [
    ("owner_user_id", "43"), ("owner_user_id", ""), ("expected_uuid", "wrong"),
    ("domain", "other"), ("queue", "generation"), ("job_type", "wrong"),
    ("expected_payload", {"pack_id": 1, "batch_id": 8, "user_id": 42}), ("idempotency_key", "wrong"),
    ("expected_payload", {"pack_id": True, "batch_id": 7, "user_id": 42}),
    ("job_id", True), ("job_id", 0), ("job_id", -1),
])
def test_retry_rejects_identity_mismatch(jobs: JobManager, field: str, value: Any) -> None:
    """All exact identity facts must match before a failed row is changed."""
    job = failed_job(jobs)
    with pytest.raises(ValueError):
        retry(jobs, job, **{field: value})
    assert snapshot(jobs, job) == ("failed", 0, 0)


@pytest.mark.parametrize("action", ("pause", "drain"))
def test_retry_respects_queue_controls(jobs: JobManager, action: str) -> None:
    """Retry is admission, not an override of an administrator's queue state."""
    job = failed_job(jobs)
    jobs.set_queue_control("vn_assets", "default", action)
    with pytest.raises(ValueError):
        retry(jobs, job)
    assert snapshot(jobs, job) == ("failed", 0, 0)


def test_retry_respects_queued_quota(jobs: JobManager, monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry shares the serialized domain/owner queued limit with create."""
    job = failed_job(jobs)
    jobs.create_job(domain="vn_assets", queue="generation", job_type="child", payload={}, owner_user_id="42")
    monkeypatch.setenv("JOBS_QUOTA_MAX_QUEUED_VN_ASSETS_USER_42", "1")
    with pytest.raises(ValueError, match="max queued"):
        retry(jobs, job)
    assert snapshot(jobs, job) == ("failed", 0, 0)


def test_create_after_retry_observes_submission_rate(jobs: JobManager, monkeypatch: pytest.MonkeyPatch) -> None:
    """Create cannot ignore an explicit retry admission that just consumed capacity."""
    job = failed_job(jobs)
    monkeypatch.setenv("JOBS_QUOTA_SUBMITS_PER_MIN_VN_ASSETS_USER_42", "2")
    retry(jobs, job)
    with pytest.raises(ValueError, match="submits per minute"):
        jobs.create_job(domain="vn_assets", queue="generation", job_type="child", payload={}, owner_user_id="42")
    assert snapshot(jobs, job) == ("queued", 1, 1)


def test_retry_after_create_observes_submission_rate(jobs: JobManager, monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry cannot bypass the same rate quota by keeping its old Job identity."""
    job = failed_job(jobs)
    jobs.create_job(domain="vn_assets", queue="generation", job_type="child", payload={}, owner_user_id="42")
    monkeypatch.setenv("JOBS_QUOTA_SUBMITS_PER_MIN_VN_ASSETS_USER_42", "2")
    with pytest.raises(ValueError, match="submits per minute"):
        retry(jobs, job)
    assert snapshot(jobs, job) == ("failed", 0, 0)


def test_old_job_retries_alone_exhaust_submission_rate(jobs: JobManager, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same-row retries consume the rate window after the original create ages out."""
    job = failed_job(jobs)
    admission_time = jobs._clock.now_utc() + timedelta(seconds=120)
    monkeypatch.setattr(jobs._clock, "now_utc", lambda: admission_time)
    monkeypatch.setenv("JOBS_QUOTA_SUBMITS_PER_MIN_VN_ASSETS_USER_42", "2")
    for _attempt in range(2):
        assert retry(jobs, job)["created_at"] == job["created_at"]
        assert retry(jobs, job)["id"] == job["id"]
        acquired = jobs.acquire_next_job(
            domain="vn_assets", queue="default", worker_id="retry-worker", lease_seconds=60, owner_user_id="42",
        )
        assert acquired is not None
        assert jobs.fail_job(
            job["id"], error="partial fanout", retryable=True, backoff_seconds=0,
            worker_id=acquired["worker_id"], lease_id=acquired["lease_id"],
        )
    with pytest.raises(ValueError, match="submits per minute"):
        retry(jobs, job)
    assert snapshot(jobs, job) == ("failed", 0, 2)


def test_concurrent_retries_admit_once(jobs: JobManager) -> None:
    """Concurrent explicit retry callers return one queue admission and event."""
    job = failed_job(jobs)
    barrier = Barrier(2)

    def recover(_index: int) -> dict[str, Any]:
        """Align two independent Jobs connections at the facade boundary."""
        barrier.wait(timeout=10)
        return retry(jobs, job)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(recover, range(2)))
    assert {result["id"] for result in results} == {job["id"]}
    assert snapshot(jobs, job) == ("queued", 1, 1)


@pytest.mark.parametrize("quota,limit", [("MAX_QUEUED", "1"), ("SUBMITS_PER_MIN", "2")])
def test_concurrent_create_and_retry_share_quota(
    jobs: JobManager, monkeypatch: pytest.MonkeyPatch, quota: str, limit: str,
) -> None:
    """Create and retry serialize on the same owner quota, not separate budgets."""
    job = failed_job(jobs)
    monkeypatch.setenv(f"JOBS_QUOTA_{quota}_VN_ASSETS_USER_42", limit)
    barrier = Barrier(2)

    def admit(operation: str) -> tuple[str, str]:
        """Race two real connections; rejection is the intended loser outcome."""
        barrier.wait(timeout=10)
        try:
            if operation == "retry":
                retry(jobs, job)
            else:
                jobs.create_job(
                    domain="vn_assets", queue="generation", job_type="child", payload={}, owner_user_id="42",
                )
        except ValueError:
            return (operation, "rejected")
        return (operation, "admitted")

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = dict(executor.map(admit, ("create", "retry")))
    assert sorted(results.values()) == ["admitted", "rejected"]
    assert len(jobs.list_jobs(domain="vn_assets", owner_user_id="42", status="queued")) == 1
    expected = ("queued", 1, 1) if results["retry"] == "admitted" else ("failed", 0, 0)
    assert snapshot(jobs, job) == expected


def test_counter_failure_rolls_back_retry_and_event(jobs: JobManager, monkeypatch: pytest.MonkeyPatch) -> None:
    """Failed transactional bookkeeping leaves the failed row untouched."""
    job = failed_job(jobs)
    module = pg_admission if jobs.backend == "postgres" else sqlite_admission
    original = module._bump_counters

    def fail_after_bump(*args: Any, **kwargs: Any) -> None:
        """Fail after the real counter write to test the transaction, not a mock."""
        original(*args, **kwargs)
        raise RuntimeError("counter persistence interrupted")

    monkeypatch.setattr(module, "_bump_counters", fail_after_bump)
    with pytest.raises(RuntimeError, match="counter persistence interrupted"):
        retry(jobs, job)
    assert snapshot(jobs, job) == ("failed", 0, 0)


@pytest.mark.parametrize("health", ["live", "expired", "missing_expiry", "no_lease", "no_worker", "cancel_requested"])
def test_processing_lease_health_is_owner_scoped_and_read_only(
    jobs: JobManager, monkeypatch: pytest.MonkeyPatch, health: str,
) -> None:
    """Jobs supplies one clock/lease interpretation without maintenance side effects."""
    job = failed_job(jobs)
    assert jobs.has_live_processing_lease(job["id"], owner_user_id="42") is False
    retry(jobs, job)
    acquired = jobs.acquire_next_job(
        domain="vn_assets", queue="default", owner_user_id="42", worker_id="health-worker", lease_seconds=60,
    )
    assert acquired is not None
    if health != "live":
        field, value = {
            "expired": ("leased_until", "2000-01-01 00:00:00"), "missing_expiry": ("leased_until", None),
            "no_lease": ("lease_id", None), "no_worker": ("worker_id", None),
            "cancel_requested": ("cancel_requested_at", "2026-09-25 00:00:00"),
        }[health]
        conn = jobs._connect()
        try:
            with conn:
                marker = "%s" if jobs.backend == "postgres" else "?"
                with closing(conn.cursor()) as cur:
                    cur.execute(f"UPDATE jobs SET {field}={marker} WHERE id={marker}", (value, job["id"]))
        finally:
            conn.close()
    before = jobs.get_job(job["id"], owner_user_id="42")
    assert jobs.has_live_processing_lease(job["id"], owner_user_id="42") is (health == "live")
    assert jobs.has_live_processing_lease(job["id"], owner_user_id="43") is False
    assert jobs.has_live_processing_lease(999999, owner_user_id="42") is False
    assert jobs.get_job(job["id"], owner_user_id="42") == before
    if health == "live":
        deadline = datetime.fromisoformat(str(before["leased_until"]))
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        monkeypatch.setattr(jobs._clock, "now_utc", lambda: deadline)
        assert jobs.has_live_processing_lease(job["id"], owner_user_id="42") is False


@pytest.mark.parametrize("facts", [{"job_id": True, "owner_user_id": "42"}, {"job_id": 1, "owner_user_id": ""}])
def test_processing_lease_health_never_widens_invalid_identity(jobs: JobManager, facts: dict[str, Any]) -> None:
    """An invalid ID or blank owner cannot become an unscoped read."""
    with pytest.raises(ValueError):
        jobs.has_live_processing_lease(**facts)
