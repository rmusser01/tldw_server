from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

psycopg = pytest.importorskip("psycopg")

from tldw_Server_API.app.core.exceptions import BadRequestError
from tldw_Server_API.app.core.Jobs.manager import JobManager

pytestmark = [
    pytest.mark.pg_jobs,
]


def _run_two_concurrent_creates(*, dsn: str, domain: str) -> list[str]:
    barrier = Barrier(2)
    managers = [JobManager(backend="postgres", db_url=dsn) for _index in range(2)]

    def create(index: int) -> str:
        barrier.wait(timeout=30)
        try:
            managers[index].create_job(
                domain=domain,
                queue="default",
                job_type=f"concurrent-{index}",
                payload={},
                owner_user_id="concurrent-owner",
            )
            return "created"
        except BadRequestError as exc:
            return str(exc.code)

    with ThreadPoolExecutor(max_workers=2) as pool:
        return list(pool.map(create, range(2)))


def test_pg_max_queued_quota_is_atomic_under_concurrent_submissions(monkeypatch, jobs_pg_dsn):
    monkeypatch.setenv("JOBS_QUOTA_MAX_QUEUED", "1")

    results = _run_two_concurrent_creates(dsn=jobs_pg_dsn, domain="quota-race-queued")

    assert results.count("created") == 1
    assert results.count("jobs_max_queued") == 1


def test_pg_submit_rate_quota_is_atomic_under_concurrent_submissions(monkeypatch, jobs_pg_dsn):
    monkeypatch.setenv("JOBS_QUOTA_SUBMITS_PER_MIN", "1")

    results = _run_two_concurrent_creates(dsn=jobs_pg_dsn, domain="quota-race-rate")

    assert results.count("created") == 1
    assert results.count("jobs_submit_rate_limited") == 1


def test_pg_max_queued_quota(monkeypatch, jobs_pg_dsn):

    monkeypatch.setenv("JOBS_DB_URL", jobs_pg_dsn)
    jm = JobManager(backend="postgres", db_url=jobs_pg_dsn)

    # Global max queued per user/domain
    monkeypatch.setenv("JOBS_QUOTA_MAX_QUEUED", "1")

    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="1")
    with pytest.raises(BadRequestError) as caught:
        jm.create_job(domain="chatbooks", queue="default", job_type="t2", payload={}, owner_user_id="1")
    assert caught.value.code == "jobs_max_queued"
    assert caught.value.retry_after is None
    # Different user not blocked
    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="2")


def test_pg_submits_per_minute_quota_precedence(monkeypatch, jobs_pg_dsn):

    monkeypatch.setenv("JOBS_DB_URL", jobs_pg_dsn)
    jm = JobManager(backend="postgres", db_url=jobs_pg_dsn)

    # Global limit 1/min; domain+user override to 2/min
    monkeypatch.setenv("JOBS_QUOTA_SUBMITS_PER_MIN", "1")
    monkeypatch.setenv("JOBS_QUOTA_SUBMITS_PER_MIN_CHATBOOKS_USER_1", "2")

    jm.create_job(domain="chatbooks", queue="default", job_type="a", payload={}, owner_user_id="1")
    jm.create_job(domain="chatbooks", queue="default", job_type="b", payload={}, owner_user_id="1")
    with pytest.raises(BadRequestError) as caught:
        jm.create_job(domain="chatbooks", queue="default", job_type="c", payload={}, owner_user_id="1")
    assert caught.value.code == "jobs_submit_rate_limited"
    assert caught.value.retry_after == 60

    # Other domain -> global 1/min applies
    jm.create_job(domain="other", queue="default", job_type="x", payload={}, owner_user_id="1")
    with pytest.raises(BadRequestError) as caught:
        jm.create_job(domain="other", queue="default", job_type="y", payload={}, owner_user_id="1")
    assert caught.value.code == "jobs_submit_rate_limited"
    assert caught.value.retry_after == 60


def test_pg_max_inflight_quota(monkeypatch, jobs_pg_dsn):

    monkeypatch.setenv("JOBS_DB_URL", jobs_pg_dsn)
    jm = JobManager(backend="postgres", db_url=jobs_pg_dsn)

    monkeypatch.setenv("JOBS_QUOTA_MAX_INFLIGHT", "1")

    # Seed two queued for user 1
    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="1")
    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="1")

    acq1 = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="w", owner_user_id="1")
    assert acq1 is not None
    acq2 = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="w2", owner_user_id="1")
    assert acq2 is None

    # Different user can still acquire
    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="2")
    acq_other = jm.acquire_next_job(
        domain="chatbooks", queue="default", lease_seconds=30, worker_id="w3", owner_user_id="2"
    )
    assert acq_other is not None


def test_pg_max_inflight_ignores_expired_leases(monkeypatch, jobs_pg_dsn):

    monkeypatch.setenv("JOBS_DB_URL", jobs_pg_dsn)
    jm = JobManager(backend="postgres", db_url=jobs_pg_dsn)

    monkeypatch.setenv("JOBS_QUOTA_MAX_INFLIGHT", "1")

    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="1")
    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="1")

    acq1 = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="w", owner_user_id="1")
    assert acq1 is not None

    conn = jm._connect()
    try:
        with jm._pg_cursor(conn) as cur:
            cur.execute(
                "UPDATE jobs SET leased_until = NOW() - interval '10 seconds' WHERE id = %s",
                (int(acq1["id"]),),
            )
        conn.commit()
    finally:
        conn.close()

    acq2 = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="w2", owner_user_id="1")
    assert acq2 is not None
