import pytest

from tldw_Server_API.app.core.exceptions import BadRequestError
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables


def test_max_queued_quota_sqlite(monkeypatch, tmp_path):

    db_path = tmp_path / "jobs_quota.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)

    # Global max queued per user/domain
    monkeypatch.setenv("JOBS_QUOTA_MAX_QUEUED", "1")

    # First job for user 1 in domain chatbooks succeeds
    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="1")

    # Second job for same user/domain should hit quota
    with pytest.raises(BadRequestError) as caught:
        jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="1")
    assert caught.value.code == "jobs_max_queued"
    assert caught.value.retry_after is None

    # Different user should not be blocked by user-specific quota
    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="2")

    # Different domain should not be blocked by domain scoping
    jm.create_job(domain="other", queue="default", job_type="t", payload={}, owner_user_id="1")


def test_submits_per_minute_quota_precedence_sqlite(monkeypatch, tmp_path):

    db_path = tmp_path / "jobs_quota_spm.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)

    # Global limit 1/min; domain+user override to 2/min should take precedence
    monkeypatch.setenv("JOBS_QUOTA_SUBMITS_PER_MIN", "1")
    monkeypatch.setenv("JOBS_QUOTA_SUBMITS_PER_MIN_CHATBOOKS_USER_1", "2")

    # Two submits within a minute for domain chatbooks, user 1 should be allowed
    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="1")
    jm.create_job(domain="chatbooks", queue="default", job_type="t2", payload={}, owner_user_id="1")

    # Third submit should be blocked by the 2/min override
    with pytest.raises(BadRequestError) as caught:
        jm.create_job(domain="chatbooks", queue="default", job_type="t3", payload={}, owner_user_id="1")
    assert caught.value.code == "jobs_submit_rate_limited"
    assert caught.value.retry_after == 60

    # For another domain, the global 1/min applies; second submit should fail
    jm.create_job(domain="other", queue="default", job_type="x", payload={}, owner_user_id="1")
    with pytest.raises(BadRequestError) as caught:
        jm.create_job(domain="other", queue="default", job_type="y", payload={}, owner_user_id="1")
    assert caught.value.code == "jobs_submit_rate_limited"
    assert caught.value.retry_after == 60


def test_max_inflight_quota_sqlite(monkeypatch, tmp_path):

    db_path = tmp_path / "jobs_quota_inflight.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)

    # Enforce max inflight of 1 per user/domain
    monkeypatch.setenv("JOBS_QUOTA_MAX_INFLIGHT", "1")

    # Seed two queued jobs for user 1
    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="1")
    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="1")

    # First acquire succeeds when passing owner_user_id for quota scope
    acq1 = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="w1", owner_user_id="1")
    assert acq1 is not None

    # Second acquire for same owner should be blocked by inflight quota
    acq2 = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="w2", owner_user_id="1")
    assert acq2 is None

    # Different user is not blocked by user-specific inflight quota
    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="2")
    acq_other = jm.acquire_next_job(
        domain="chatbooks", queue="default", lease_seconds=30, worker_id="w3", owner_user_id="2"
    )
    assert acq_other is not None


def test_max_inflight_ignores_expired_leases_sqlite(monkeypatch, tmp_path):

    db_path = tmp_path / "jobs_quota_inflight_expired.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)

    monkeypatch.setenv("JOBS_QUOTA_MAX_INFLIGHT", "1")

    # Two queued jobs for user 1
    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="1")
    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="1")

    acq1 = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="w1", owner_user_id="1")
    assert acq1 is not None

    # Expire the lease so the processing job should not count toward inflight
    conn = jm._connect()
    try:
        conn.execute(
            "UPDATE jobs SET leased_until = DATETIME('now', '-10 seconds') WHERE id = ?",
            (int(acq1["id"]),),
        )
        conn.commit()
    finally:
        conn.close()

    acq2 = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="w2", owner_user_id="1")
    assert acq2 is not None
